"""
Corpus-wide deduplication that survives parallelism and restarts.

architecture.md §9.1 asks for "Deduplicate (exact + MinHash)". The tokenizer
downloader shipped a Python `set` of hex digests, scoped to one script
invocation — fine for an 8GB single-process run, and wrong here in three ways
at once: it does not see across sources, it does not see across a resumed
download, and at a few hundred million chunks the set alone would need tens
of gigabytes of RAM.

What this uses instead
----------------------
A **Bloom filter in shared memory**, sized once at startup and mapped into
every worker process.

* *Shared* — one filter for the whole run, so a document that appears in both
  C4 and FineWeb-Edu is caught the second time regardless of which worker
  sees it.
* *Lock-free* — the only mutation is setting bits, which is idempotent and
  order-independent. Two workers racing on the same word can, in the worst
  case, cost one missed duplicate; they can never corrupt the structure. A
  lock on the hottest path in the pipeline would cost far more than the
  occasional miss.
* *Bounded* — 8 bits per expected item is ~2% false positives, so a ~600MB
  filter covers 600M chunks. Compare against a `set`, which would need
  roughly 40x that.

The trade is that a Bloom filter has false *positives* — it can claim a chunk
is a duplicate when it is not. That direction is the safe one: the cost is
discarding a small fraction of good text from a corpus we are deliberately
over-provisioning, whereas a false negative would put a duplicate into
training. The rate is reported in the manifest rather than assumed.

Two filters, not one
--------------------
`clean.dedup_key` produces an exact hash and a normalized one. Both are
checked: exact catches byte-identical repeats, normalized catches the same
text differing only in case, whitespace or punctuation — which is what
duplication actually looks like once a page has been through two scrapers.
This covers the "exact + near-dup" pass the earlier cleanup ran as two
sequential whole-file sorts, at streaming cost instead.

Not MinHash
-----------
True MinHash/LSH would additionally catch documents that share most of their
shingles — boilerplate-heavy pages that differ in a headline. It needs an
LSH index with real coordination between workers, which is a materially
bigger piece of machinery than this. Deferred deliberately, and called out
here rather than left as a silent gap against what §9.1 promises.
"""

from __future__ import annotations

import json
import math
from multiprocessing import Array
from pathlib import Path

from .clean import DedupKey

# Bits per expected element. 8 gives ~2% false positives at 5 hash probes,
# which is the knee of the curve: 4 bits costs ~15%, 16 bits buys only ~0.05%.
BITS_PER_ITEM = 8
NUM_PROBES = 5


class SharedBloomFilter:
    """
    A bit array in shared memory, addressed by 64-bit keys.

    Created once in the master before workers are forked; `multiprocessing`
    hands each child a view of the same pages, so there is one filter and no
    copies.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.num_bits = max(8, capacity * BITS_PER_ITEM)
        self.num_bytes = (self.num_bits + 7) // 8
        # lock=False is the point: see the module docstring on why races here
        # are benign.
        self._bits = Array("B", self.num_bytes, lock=False)

    def _positions(self, key: int):
        """
        Five probe positions from one 64-bit key, by double hashing.

        `h1 + i*h2` over a prime-ish stride is the standard Kirsch-Mitzenmacher
        construction: statistically as good as five independent hashes, at the
        cost of one multiply per probe instead of five hash computations.
        """
        h1 = key & 0xFFFFFFFF
        h2 = (key >> 32) | 1  # odd, so it strides the whole space
        for i in range(NUM_PROBES):
            yield ((h1 + i * h2) & 0x7FFFFFFFFFFFFFFF) % self.num_bits

    def add_if_absent(self, key: int) -> bool:
        """
        `True` if `key` was new (and is now recorded), `False` if already seen.

        Reads and writes in one pass so a chunk costs five probes, not ten.
        """
        seen = True
        for position in self._positions(key):
            byte, bit = divmod(position, 8)
            mask = 1 << bit
            if not self._bits[byte] & mask:
                seen = False
                self._bits[byte] |= mask
        return not seen

    def estimated_fill(self) -> float:
        """
        Fraction of bits set, sampled rather than counted.

        A full count over a 600MB array takes long enough to matter at the end
        of a run; a 64k-byte sample is accurate to well under a percent, which
        is all the manifest needs.
        """
        step = max(1, self.num_bytes // 65_536)
        sampled = range(0, self.num_bytes, step)
        set_bits = sum(bin(self._bits[i]).count("1") for i in sampled)
        return set_bits / (len(sampled) * 8)

    def estimated_false_positive_rate(self) -> float:
        fill = self.estimated_fill()
        return fill**NUM_PROBES


class Deduplicator:
    """
    The two-filter front end workers actually call, plus its own counters.

    Wrapping the filters rather than exposing them keeps the "exact first,
    then normalized" order in one place, and means a worker's dedup stats come
    back in the same shape as every other stage's.
    """

    def __init__(self, exact: SharedBloomFilter, normalized: SharedBloomFilter) -> None:
        self._exact = exact
        self._normalized = normalized
        self.seen = 0
        self.dropped_exact = 0
        self.dropped_normalized = 0

    def accept(self, key: DedupKey) -> bool:
        self.seen += 1
        if not self._exact.add_if_absent(key.exact):
            self.dropped_exact += 1
            return False
        if not self._normalized.add_if_absent(key.normalized):
            self.dropped_normalized += 1
            return False
        return True

    def as_dict(self) -> dict:
        """
        This instance's own view: correct when one process does all the
        `accept()` calls, but `self.seen`/`self.dropped_*` are plain
        per-process ints, not shared state -- a multiprocessing caller
        whose workers each hold their own `Deduplicator` handle (the normal
        case, since only the underlying `SharedBloomFilter` bit array is
        actually cross-process) must sum worker-reported counts itself
        instead of calling this in the parent after workers finish, or the
        result will look like nothing was ever checked. `filter_metrics()`
        below is the piece of this that stays correct regardless.
        """
        dropped = self.dropped_exact + self.dropped_normalized
        return {
            "chunks_checked": self.seen,
            "dropped_exact_duplicate": self.dropped_exact,
            "dropped_near_duplicate": self.dropped_normalized,
            "duplicate_rate": round(dropped / self.seen, 5) if self.seen else 0.0,
            **self.filter_metrics(),
        }

    def filter_metrics(self) -> dict:
        """
        Fill and estimated false-positive rate, read directly from the
        shared Bloom filter bit array. Unlike the counters above, these
        two numbers are accurate no matter which process (or how many)
        called `accept()`, because they sample `SharedBloomFilter`'s
        actual shared memory rather than a per-instance counter.
        """
        return {
            "filter_fill": round(self._exact.estimated_fill(), 4),
            "estimated_false_positive_rate": round(
                self._exact.estimated_false_positive_rate(), 5
            ),
        }


def build_deduplicator(expected_chunks: int) -> Deduplicator:
    """
    Size both filters for `expected_chunks`.

    Callers derive the estimate from the byte budget rather than guessing:
    at ~1.2KB per cleaned chunk, a 300GB corpus is ~250M chunks, so the pair
    of filters costs ~500MB — trivial against 119GB of free RAM, and the
    reason it is worth sizing generously rather than tightly.
    """
    capacity = max(1_000, expected_chunks)
    return Deduplicator(SharedBloomFilter(capacity), SharedBloomFilter(capacity))


def estimate_chunks(total_bytes: int, mean_chunk_bytes: int = 1_200) -> int:
    return max(1_000, math.ceil(total_bytes / max(1, mean_chunk_bytes)))


def save_report(deduplicator: Deduplicator, path: str | Path) -> None:
    """Write dedup stats beside the corpus, so a later run can compare."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(deduplicator.as_dict(), indent=1) + "\n", encoding="utf-8")
