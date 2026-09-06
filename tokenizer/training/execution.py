"""
The one place in this package that knows about cores, processes and pools.

Why this module exists
----------------------
Before it, `corpus_reader.py` and `validate_tokenizer.py` were single-threaded
by construction: both streamed the whole corpus in one Python `for` loop. On
the real corpus that is ~1.44 billion eligible lines, and it pinned exactly one
core while the other 41 sat idle -- `trainer.num_threads` only ever reached
SentencePiece's own C++ trainer, which is a ~30 second step inside a ~4 hour
run.

Fixing that by sprinkling `multiprocessing` through the domain modules would
have put "how many cores do we have" into the middle of "what counts as an
eligible line", which is exactly the coupling this codebase avoids elsewhere.
So parallelism lives here, behind one small interface, and the domain modules
stay pure: they expose per-file functions and a merge, and never import
`multiprocessing` at all.

The contract
------------
`Executor.map(fn, items)` returns results **in input order**, always, for any
worker count. Callers may therefore rely on ordering for determinism, which is
what lets the corpus sampler stay byte-for-byte reproducible no matter how many
workers it runs on. `workers=1` takes a genuinely sequential path with no pool
and no pickling, so the single-process code path stays exercised by every test
rather than rotting.

`fn` must be a module-level callable (picklable) and should be pure: given the
same item it must produce the same result, because that is what makes
`workers=1` and `workers=32` produce identical output.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Leave a little of the machine for everything that is not this job -- the OS,
# and on this deployment a GPU inference process that must keep being fed.
# Only applies to `resolve_workers(0)` (auto); an explicit count is honoured.
AUTO_RESERVED_CORES = 6


def resolve_workers(workers: int) -> int:
    """
    `workers` as configured -> the number actually used.

    0 means auto-detect: every core except `AUTO_RESERVED_CORES`. A negative
    count is a configuration error rather than something to silently clamp,
    but callers validate that; here it simply floors at 1.
    """
    if workers > 0:
        return workers
    detected = (os.cpu_count() or 1) - AUTO_RESERVED_CORES
    return max(1, detected)


@dataclass(frozen=True)
class Executor:
    """
    An ordered parallel map over a fixed list of work items.

    Deliberately tiny: this codebase's parallelism is always "run this pure
    function over each corpus file and merge the results", so a general task
    framework would be more machinery than the problem needs.
    """

    workers: int = 1
    start_method: str = "fork"

    @classmethod
    def from_config(cls, workers: int) -> "Executor":
        return cls(workers=resolve_workers(workers))

    @property
    def is_parallel(self) -> bool:
        return self.workers > 1

    def map(self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """
        Apply `fn` to every item, returning results in input order.

        Falls back to the sequential path for a single worker or a single item
        -- forking a pool to process one file costs more than it saves, and it
        keeps small runs (and most tests) free of process overhead entirely.
        """
        items = list(items)
        if not items:
            return []
        if self.workers <= 1 or len(items) == 1:
            return [fn(item) for item in items]

        # chunksize=1: corpus files vary in size by orders of magnitude
        # (a 4 MB cli_tldr shard against a 2 GB fineweb_edu shard), so handing
        # out one file at a time is what keeps every worker busy to the end
        # instead of leaving one straggler holding a chunk of big files.
        ctx = mp.get_context(self.start_method)
        with ctx.Pool(processes=min(self.workers, len(items))) as pool:
            return pool.map(fn, items, chunksize=1)


def largest_remainder(total: int, weights: Sequence[int]) -> list[int]:
    """
    Split `total` across buckets proportionally to `weights`, exactly and
    deterministically.

    Used wherever a global budget has to be divided over per-file work so the
    per-file pieces still sum to exactly the global number: the training quota
    per source, the eval sample cap, and the length reservoir. Ties break on
    index so the result never depends on dict or filesystem ordering.

    Returns integers summing to exactly `min(total, sum(weights))`, with no
    bucket exceeding its own weight.
    """
    if total <= 0 or not weights:
        return [0] * len(weights)
    capacity = sum(weights)
    if capacity <= 0:
        return [0] * len(weights)
    total = min(total, capacity)

    exact = [total * w / capacity for w in weights]
    floors = [int(x) for x in exact]
    shortfall = total - sum(floors)
    if shortfall:
        # Largest fractional part first; index breaks ties.
        order = sorted(
            range(len(weights)), key=lambda i: (-(exact[i] - floors[i]), i)
        )
        for i in order[:shortfall]:
            floors[i] += 1
    return floors
