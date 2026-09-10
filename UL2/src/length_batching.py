"""
Grouping similar-length examples into the same batch, so padding stops
buying compute that teaches nothing.

The problem, as measured
-----------------------
`pad_batch` pads every row of a batch out to the longest row in it, and a
padded position costs a transformer exactly as much arithmetic as a real one
while contributing nothing to the loss. Measured over 500 real batches from
this pipeline, 18.2% of all encoder compute was padding, and essentially all
of it -- 17.85 of those 18.2 points -- came from one cause: rows inside a
batch being shorter than that batch's longest row. Rounding widths up to a
multiple of 8 accounted for 0.31%, and nothing else was material.

Splitting that by mode showed where it lived::

    mode   efficiency   share of all waste
    [R]        87.5%                 38.3%
    [X]        87.6%                 22.0%
    [S]        40.2%                 39.7%

`[S]` dominates because it is never packed (`packing.py`), so its rows are
whole documents whose lengths run from 9 to 882 tokens -- mean 185, against a
mean batch width of 460. Two thirds of every `[S]` batch was padding.

The fix
-------
Pull a *pool* of examples rather than exactly one batch, sort the pool by
length, and cut batches from adjacent entries. Each batch then pads to a local
maximum instead of a global one. (Which length to sort on is not obvious --
both padded dimensions cost real FLOPs -- and is settled by measurement in
`_example_width`.) Simulated on 8,000 real examples::

    pool           efficiency    waste   speedup
    none (today)        81.8%    18.2%     1.00x
    8 batches           96.3%     3.7%     1.18x
    16 batches          97.7%     2.3%     1.19x
    32 batches          98.6%     1.4%     1.20x
    128 batches         99.2%     0.8%     1.21x

Returns flatten past ~16-32 batches, and the pool has to be carried in every
checkpoint (below), so the default sits at the knee rather than the asymptote.

The second cut: capacity, not row count
---------------------------------------
Pooling removes padding *within* a batch but leaves every batch holding the
same number of rows, and rows differ in length by more than a hundredfold
here. Two costs survive it, both measured on this model and GPU:

  - a micro-batch costs ~0.31 s before it does any arithmetic at all -- the
    dispatch of 48 layers' worth of kernels. A 16-row `[S]` batch is 296
    tokens wide and takes 0.39 s, so four fifths of it is launch overhead with
    the GPU idle. Raising the rows to 48 at that width costs 1.6% more time
    for six times the work.
  - peak memory of a fixed-row batch is set by the widest row that turns up,
    so it is bounded by the corpus rather than by configuration. That is why
    raising the row count globally is unsafe -- 32 rows survives a sample and
    then meets a 1216x1504 window later in the run.

`BatchCapacity` bounds `rows x padded width` instead, separately for the two
padded rectangles. Narrow rows then get many to a batch and wide rows few, and
peak memory stops depending on which shapes the corpus happens to contain.
`capacity=None` keeps the fixed-`batch_size` cut exactly as it was.

Why this does not weaken shuffling
----------------------------------
Sorting is an ordering operation, and the whole value of a shuffled stream is
that ordering carries no information. Two separate things are therefore
deliberately re-randomized, and both matter:

  1. **The pool is shuffled before it is sorted.** Sorting alone is stable, so
     equal-length examples would keep their corpus order -- and length ties are
     common. Shuffling first makes the tie order independent of where an
     example came from.

  2. **The batch order is shuffled after cutting.** Emitting a sorted pool in
     order would walk the model from the shortest batch to the longest, over
     and over, one systematic sweep per pool. That is a periodic signal
     correlated with sequence length, which is exactly the kind of structure
     shuffling exists to destroy.

What stays untouched: every example the stream produces is still emitted,
exactly once, and batches are still requested one mode at a time by the
caller, so the per-mode batch ratio is whatever the caller draws. This
regroups examples; it never selects, drops, duplicates or reweights them.

Why it does not change what the model learns
--------------------------------------------
`Trainer.accumulate` weights each micro-batch's gradient by its token count
and divides by the accumulation window's total, so a window's gradient is
exactly that of one batch holding every token in it -- it depends on *which
tokens* are in the window, never on how they were grouped into micro-batches
(`test_accumulation_matches_one_big_batch` pins that invariance). Grouping is
the only thing this module changes.

Attention is masked per row (padding mask, plus the block-diagonal segment
mask for packed windows), and pretraining runs with dropout at 0.0, so an
example's logits do not depend on which rows share its batch either. Two
examples that were batched together yesterday and apart today produce the same
numbers.

Why the pool is checkpointed
----------------------------
The pool holds examples already read from the corpus but not yet emitted.
Dropping it on resume would silently lose them -- the same failure
`PretrainBatchSource._buffer` is serialized to avoid, and the same one
architecture.md §7.9 calls out as having no signal: the loss curve looks
fine, the token budget is quietly wrong. It cannot be reconstructed by
replaying the corpus instead, because the three per-mode pools were filled at
three different corpus positions and a single sequential stream cannot be
rewound to all three.

Usage
-----
Plug-and-play: it needs a callable that yields the next example for a mode,
and nothing else about the pipeline.

    batcher = LengthBucketedBatcher(
        produce_example, batch_size=16, pool_batches=16, seed=1234
    )
    examples = batcher.next_batch(mode)      # list[Example], all one mode
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .denoise import Example, example_from_dict, example_to_dict
from .errors import UL2Error

# The knee of the measured curve: 97.7% packing efficiency, against 98.6% for
# a pool four times larger. The remaining 0.9 points are not worth quadrupling
# what every checkpoint has to carry.
DEFAULT_POOL_BATCHES = 16


@dataclass(frozen=True)
class BatchCapacity:
    """
    How much padded work one micro-batch may hold.

    A batch of a fixed number of rows holds an amount of work that varies with
    the rows' length -- over this corpus, by more than a hundredfold, since a
    `[S]` row is one ~300-token document and an `[R]` row is a packed
    2048-token window. Two consequences were measured on this model and GPU,
    and this class exists to remove both:

      - a 16x296 micro-batch spends 0.31 s of its 0.39 s on kernel dispatch
        with the GPU essentially idle, because there is not enough work in it
        to cover the launch cost of 48 layers;
      - peak memory of a fixed-row batch is unbounded, since it is set by the
        widest row that turns up. That is why raising the row count is unsafe:
        a batch size that survives a sample of the corpus still meets a wider
        shape later and runs out of memory mid-run.

    Bounding `rows x padded width` instead fixes both at once: narrow rows get
    many to a batch and wide rows get few, and peak memory is bounded by
    construction rather than by whichever shape the corpus happens to hold.

    Encoder and decoder are bounded separately because they cost differently.
    Fitting measured peak memory over all 3,136 real shapes gives ~1.6 MiB per
    encoder token against ~5.6 MiB per decoder token: a decoder position
    carries cross-attention and, above all, a 33,728-wide logit row, so one
    combined budget would have to be set by the decoder and would leave
    encoder capacity unused.

    `max_rows` is an optional backstop; the token budgets already bound the row
    count, so it is only there for a caller that wants a hard ceiling as well.
    `pad_to_multiple_of` mirrors the argument of the same name in `pad_batch`,
    so the width this counts is the width that will actually be allocated
    rather than one rounding step below it.
    """

    max_encoder_tokens: int
    max_decoder_tokens: int
    max_rows: int | None = None
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_encoder_tokens", "max_decoder_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise UL2Error(f"{name} must be a positive integer, got {value!r}")
        if self.max_rows is not None and (
            not isinstance(self.max_rows, int)
            or isinstance(self.max_rows, bool)
            or self.max_rows < 1
        ):
            raise UL2Error(
                f"max_rows must be a positive integer or absent, got {self.max_rows!r}"
            )
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of < 1:
            raise UL2Error(
                f"pad_to_multiple_of must be >= 1 or absent, got {self.pad_to_multiple_of}"
            )

    def width(self, length: int) -> int:
        """The width `pad_batch` will allocate for a row of `length` tokens."""
        multiple = self.pad_to_multiple_of
        if multiple is None or multiple <= 1:
            return length
        remainder = length % multiple
        return length if remainder == 0 else length + (multiple - remainder)

    def admits(self, rows: int, encoder_length: int, target_length: int) -> bool:
        """Whether `rows` rows of this padded size stay inside every budget."""
        return (
            (self.max_rows is None or rows <= self.max_rows)
            and rows * self.width(encoder_length) <= self.max_encoder_tokens
            and rows * self.width(target_length) <= self.max_decoder_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_encoder_tokens": self.max_encoder_tokens,
            "max_decoder_tokens": self.max_decoder_tokens,
            "max_rows": self.max_rows,
            "pad_to_multiple_of": self.pad_to_multiple_of,
        }


# Key standing in for `mode=None`, which is what an unpacked run passes. A
# dict cannot be keyed by None and also JSON-serialized, and JSON object keys
# are strings regardless, so the mapping is made explicit rather than implied.
_ANY_MODE_KEY = "*"


class LengthBucketedBatcher:
    """
    Batches of similar-length examples, drawn from a shuffled pool.

    `produce(mode)` must return the next `Example` for that mode, or raise
    `StopIteration` when its source is exhausted. It is called only while a
    pool is being filled, never during emission.

    Batches are always exactly `batch_size` examples, except for a final short
    batch after `produce` raises `StopIteration` -- an infinite corpus, which
    is what pretraining uses, never reaches that.
    """

    def __init__(
        self,
        produce: Callable[[str | None], Example],
        *,
        batch_size: int,
        pool_batches: int = DEFAULT_POOL_BATCHES,
        capacity: BatchCapacity | None = None,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise UL2Error(f"batch_size must be >= 1, got {batch_size}")
        if pool_batches < 1:
            raise UL2Error(f"pool_batches must be >= 1, got {pool_batches}")

        self.produce = produce
        self.batch_size = batch_size
        self.pool_batches = pool_batches
        self.capacity = capacity
        self.rng = random.Random(seed)

        # Cut batches waiting to be emitted, per mode, in shuffled order.
        self._queues: dict[str, list[list[Example]]] = {}
        # Examples read but not yet cut into batches. Non-empty only after a
        # `StopIteration` left a partial pool behind; kept per mode so that
        # case cannot silently drop them either.
        self._pools: dict[str, list[Example]] = {}

    # -- emission -----------------------------------------------------------

    def next_batch(self, mode: str | None = None) -> list[Example]:
        """
        The next batch for `mode`, refilling that mode's pool if it is spent.

        Pools are per mode because a batch already carries exactly one mode
        (`PretrainBatchSource._batch_mode`) and because the modes sit on
        different length scales by construction -- `[R]`/`[X]` pack documents
        up to the window budget while `[S]` holds one document. Sorting them
        together would put a 2048-token packed window next to a 60-token
        document and re-create the very waste this is removing.
        """
        key = _mode_key(mode)
        queue = self._queues.get(key)
        if not queue:
            self._refill(mode)
            queue = self._queues.get(key)
        if not queue:
            raise StopIteration(f"no more examples available for mode {mode!r}")
        return queue.pop()

    def _refill(self, mode: str | None) -> None:
        """Fill one pool, sort it, cut it into batches, shuffle their order."""
        key = _mode_key(mode)
        pool = self._pools.setdefault(key, [])
        target = self.batch_size * self.pool_batches

        exhausted = False
        while len(pool) < target:
            try:
                pool.append(self.produce(mode))
            except StopIteration:
                exhausted = True
                break

        if not pool:
            return

        # Shuffle before sorting so ties break at random rather than by
        # corpus order; sort so batch-mates are near each other in length.
        self.rng.shuffle(pool)
        pool.sort(key=_example_width)

        # When `produce` was exhausted the pool is short, so the final cut may
        # hold fewer examples than a full batch. It is kept rather than
        # trimmed: those examples were already read, and dropping them is the
        # silent loss this class exists to avoid.
        batches = self._cut(pool)

        # A greedy cut ends on a partly-filled batch, and under a capacity a
        # pool yields only a handful of batches -- so emitting that remainder
        # would make a noticeable share of all batches short ones, which is
        # the padding waste this module exists to remove, reintroduced at the
        # pool boundary. It goes back into the pool to be re-sorted with the
        # next fill instead, and is checkpointed there like any other pooled
        # example.
        #
        # Only under a capacity. The fixed cut divides a pool of
        # `batch_size * pool_batches` exactly and has no remainder to carry,
        # so holding its last batch back would change the stream a run
        # without a capacity produces -- and that stream must not move.
        # Not when the source is exhausted either: there is no next fill then,
        # and holding it would be the silent loss above. Nor when it is the
        # only batch, or `next_batch` would have nothing to emit after a
        # refill.
        if self.capacity is not None and not exhausted and len(batches) > 1:
            self._pools[key] = batches.pop()
        else:
            self._pools[key] = []

        # Shuffled so the model does not see one short-to-long sweep per pool.
        # `next_batch` pops from the end, which is as arbitrary as any other
        # position once the list is shuffled.
        self.rng.shuffle(batches)
        self._queues.setdefault(key, []).extend(batches)

    def _cut(self, pool: list[Example]) -> list[list[Example]]:
        """
        Divide a sorted pool into batches.

        Without a `capacity` this is a fixed `batch_size` rows per batch, which
        is what the class did before capacities existed. With one, rows are
        added while the batch still fits every budget (`BatchCapacity.admits`),
        so a batch of short rows holds many and a batch of long rows holds few.

        The pool arrives sorted by length, so each batch is cut from rows that
        are already near each other -- the greedy pass never has to look ahead
        to find a good grouping, and a row that is admitted keeps the batch's
        padded width almost unchanged.

        A row that exceeds a budget on its own is still emitted, in a batch of
        one. Refusing it here would silently drop an example the corpus
        produced, which is the failure this module and `pack_documents` both
        take care never to commit; keeping it means the trainer sees the same
        examples whatever the capacity is set to.
        """
        if self.capacity is None:
            return [
                pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)
            ]

        batches: list[list[Example]] = []
        current: list[Example] = []
        encoder_length = 0
        target_length = 0
        for example in pool:
            widest_encoder = max(encoder_length, example.encoder_length)
            widest_target = max(target_length, example.target_length)
            if current and not self.capacity.admits(
                len(current) + 1, widest_encoder, widest_target
            ):
                batches.append(current)
                current = [example]
                encoder_length = example.encoder_length
                target_length = example.target_length
                continue
            current.append(example)
            encoder_length = widest_encoder
            target_length = widest_target
        if current:
            batches.append(current)
        return batches

    # -- introspection ------------------------------------------------------

    def pending_examples(self) -> int:
        """Examples read from the source but not yet emitted, across all modes."""
        queued = sum(len(b) for q in self._queues.values() for b in q)
        pooled = sum(len(p) for p in self._pools.values())
        return queued + pooled

    # -- Resumable ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        version, internal_state, gauss_next = self.rng.getstate()
        return {
            "batch_size": self.batch_size,
            "pool_batches": self.pool_batches,
            "capacity": None if self.capacity is None else self.capacity.to_dict(),
            "rng": [version, list(internal_state), gauss_next],
            "queues": {
                key: [[example_to_dict(e) for e in batch] for batch in queue]
                for key, queue in self._queues.items()
                if queue
            },
            "pools": {
                key: [example_to_dict(e) for e in pool]
                for key, pool in self._pools.items()
                if pool
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Whatever cut the queued batches has to still be in force, or the
        # buffered batches would be emitted at a size this run never asked
        # for. Which setting did the cutting depends on the mode: a capacity
        # governs it when set, and `batch_size` alone when it is not.
        saved_capacity = state.get("capacity")
        current_capacity = None if self.capacity is None else self.capacity.to_dict()
        if saved_capacity != current_capacity:
            raise UL2Error(
                f"checkpoint cut its buffered batches to capacity {saved_capacity!r} "
                f"but this run uses {current_capacity!r}; the buffered batches cannot "
                f"be reshaped without either dropping or duplicating examples"
            )
        saved_batch_size = int(state.get("batch_size", self.batch_size))
        if self.capacity is None and saved_batch_size != self.batch_size:
            raise UL2Error(
                f"checkpoint holds batches of {saved_batch_size} examples but this "
                f"run uses batch_size={self.batch_size}; the buffered batches cannot "
                f"be reshaped without either dropping or duplicating examples"
            )

        version, internal_state, gauss_next = state["rng"]
        self.rng.setstate((version, tuple(internal_state), gauss_next))
        # `pool_batches` is deliberately *not* restored. Unlike `batch_size` it
        # constrains nothing already built -- it governs only how large the
        # next fill is -- so the value this run was configured with should win
        # over the value the interrupted run happened to use. It is written
        # into the state anyway, as a record of what produced the queued
        # batches.
        self._queues = {
            key: [[example_from_dict(e) for e in batch] for batch in queue]
            for key, queue in state.get("queues", {}).items()
        }
        self._pools = {
            key: [example_from_dict(e) for e in pool]
            for key, pool in state.get("pools", {}).items()
        }


def _mode_key(mode: str | None) -> str:
    return _ANY_MODE_KEY if mode is None else mode


def _example_width(example: Example) -> tuple[int, int]:
    """
    Sort key: the larger of the two dimensions, then the encoder length.

    `pad_batch` pads the encoder and decoder dimensions independently, so a
    batch pays for both rectangles and the key has to serve both. Sorting on
    the encoder alone is the obvious choice and is *not* the best one, because
    `[S]` splits one document into an encoder prefix and a decoder
    continuation: the two lengths are anti-correlated, and tightening one
    dimension deliberately loosens the other.

    Chosen by measuring FLOPs rather than counting padded positions, since a
    padded decoder position is not as cheap as a padded encoder one -- a
    decoder layer runs cross-attention as well as self-attention. Over 6,000
    real examples, at this model's shape (24+24 layers, d_model 1024,
    d_ff 2816):

        sort key          FLOP efficiency
        none (arrival)          81.1%
        encoder, then target    94.7%
        encoder + target        94.6%
        target, then encoder    95.2%
        max, then encoder       95.7%   <- this one

    `[R]` and `[X]` are insensitive to the choice (~99% under every key);
    the whole spread comes from `[S]`.
    """
    return (max(example.encoder_length, example.target_length), example.encoder_length)
