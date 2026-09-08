"""
Tests for `LengthBucketedBatcher` (UL2/src/length_batching.py).

Two things have to hold at once, and they pull against each other: batches
must group similar-length examples (or the padding waste this module exists to
remove comes straight back), and the stream must stay shuffled (or sorting has
smuggled length-ordered structure into training). Most of what follows checks
one of those two without letting the other quietly regress.

The third property, and the easiest to break silently, is conservation: every
example the source produces must come out exactly once. A bucketing bug that
drops the tail of a pool would look like a small speedup and read as data loss
only in the token budget, months later.
"""

from __future__ import annotations

import random

import pytest

from src.denoise import Example
from src.errors import UL2Error
from src.length_batching import DEFAULT_POOL_BATCHES, LengthBucketedBatcher


def make_example(length: int, mode: str = "R", tag: int = 0) -> Example:
    """An example of a given encoder length. `tag` makes it individually identifiable."""
    return Example(
        mode=mode,
        encoder_input_ids=tuple(range(length)),
        decoder_target_ids=(tag, 1, 2),
        decoder_input_ids=(0, tag, 1),
        source_length=length,
        num_spans=1,
        num_corrupted_tokens=1,
    )


class SyntheticSource:
    """
    A source of examples with a wide, deliberately shuffled length spread.

    Lengths mimic the measured `[S]` distribution -- 9 to 882 tokens -- because
    that is the case bucketing exists for; a source with uniform lengths would
    pass every efficiency assertion here without the module doing anything.
    """

    def __init__(self, count: int = 4096, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.lengths = [rng.randint(9, 882) for _ in range(count)]
        self.calls = 0

    def __call__(self, mode: str | None) -> Example:
        if self.calls >= len(self.lengths):
            raise StopIteration
        example = make_example(self.lengths[self.calls], mode or "R", tag=self.calls)
        self.calls += 1
        return example


def efficiency(batches: list[list[Example]]) -> float:
    """Real encoder tokens as a fraction of the padded rectangle they occupy."""
    real = sum(e.encoder_length for batch in batches for e in batch)
    padded = sum(max(e.encoder_length for e in batch) * len(batch) for batch in batches)
    return real / padded


# -- the point of the module: less padding ----------------------------------


def test_bucketing_cuts_padding_waste_substantially():
    source = SyntheticSource(seed=1)
    batcher = LengthBucketedBatcher(source, batch_size=16, pool_batches=16, seed=1)
    bucketed = [batcher.next_batch("R") for _ in range(200)]

    # The same examples, batched in arrival order, is what we are beating.
    reference = SyntheticSource(seed=1)
    arrival = [reference(None) for _ in range(200 * 16)]
    unbucketed = [arrival[i : i + 16] for i in range(0, len(arrival), 16)]

    assert efficiency(unbucketed) < 0.65
    assert efficiency(bucketed) > 0.90


def test_a_larger_pool_never_batches_worse_than_a_smaller_one():
    """More examples to sort among cannot make the grouping worse."""
    scores = []
    for pool_batches in (1, 4, 16, 64):
        source = SyntheticSource(seed=2)
        batcher = LengthBucketedBatcher(source, batch_size=16, pool_batches=pool_batches, seed=2)
        scores.append(efficiency([batcher.next_batch("R") for _ in range(64)]))

    assert scores == sorted(scores)


def test_a_pool_of_one_batch_is_a_no_op():
    """`pool_batches=1` has nothing to sort across, so it must not change grouping."""
    source = SyntheticSource(seed=3)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=1, seed=3)
    batches = [batcher.next_batch("R") for _ in range(20)]

    reference = SyntheticSource(seed=3)
    expected = [reference(None) for _ in range(160)]
    emitted = [e for batch in batches for e in batch]

    assert sorted(e.source_length for e in emitted) == sorted(e.source_length for e in expected)


def test_examples_in_a_batch_are_close_in_length():
    source = SyntheticSource(seed=4)
    batcher = LengthBucketedBatcher(source, batch_size=16, pool_batches=32, seed=4)

    spreads = []
    for _ in range(64):
        batch = batcher.next_batch("R")
        spreads.append(max(e.encoder_length for e in batch) - min(e.encoder_length for e in batch))

    # Unbucketed, the spread within a batch of 16 draws from a 9..882 uniform
    # is ~770 on average. Sorted into pools of 512, it should be tiny.
    assert sum(spreads) / len(spreads) < 60


# -- shuffling must survive --------------------------------------------------


def test_batches_are_not_emitted_in_length_order():
    """
    A sorted pool emitted in order would sweep short-to-long once per pool.

    Checked as a rank correlation rather than "is it sorted", because a single
    unsorted pair would satisfy the weaker assertion while leaving a nearly
    monotone sweep in place.
    """
    source = SyntheticSource(seed=5)
    batcher = LengthBucketedBatcher(source, batch_size=16, pool_batches=16, seed=5)
    widths = [max(e.encoder_length for e in batcher.next_batch("R")) for _ in range(16)]

    ascending = sum(1 for a, b in zip(widths, widths[1:]) if a < b)
    # A monotone sweep gives 15 of 15; a shuffled order should sit near half.
    assert 3 <= ascending <= 12, widths


def test_equal_length_examples_do_not_keep_their_arrival_order():
    """
    Ties are common, and a stable sort would order them by corpus position.

    That would leave a real ordering signal in the stream: consecutive corpus
    records landing adjacent in every batch, for every length that repeats.
    """
    identical = [make_example(100, tag=i) for i in range(64)]

    def produce(_mode):
        if not identical:
            raise StopIteration
        return identical.pop(0)

    batcher = LengthBucketedBatcher(produce, batch_size=8, pool_batches=8, seed=6)
    emitted = [e for _ in range(8) for e in batcher.next_batch("R")]
    tags = [e.decoder_target_ids[0] for e in emitted]

    assert sorted(tags) == list(range(64))
    assert tags != list(range(64))


def test_a_different_seed_gives_a_different_order_but_the_same_examples():
    def run(seed: int) -> list[int]:
        source = SyntheticSource(seed=7)
        batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=8, seed=seed)
        return [e.decoder_target_ids[0] for _ in range(24) for e in batcher.next_batch("R")]

    first, second = run(11), run(12)
    assert first != second
    assert sorted(first) == sorted(second)


def test_the_same_seed_reproduces_the_same_order():
    def run() -> list[int]:
        source = SyntheticSource(seed=8)
        batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=4, seed=99)
        return [e.decoder_target_ids[0] for _ in range(12) for e in batcher.next_batch("R")]

    assert run() == run()


# -- conservation: nothing lost, nothing duplicated --------------------------


def test_every_example_produced_is_emitted_exactly_once():
    source = SyntheticSource(count=1024, seed=9)
    batcher = LengthBucketedBatcher(source, batch_size=16, pool_batches=8, seed=9)

    tags = []
    for _ in range(64):
        tags.extend(e.decoder_target_ids[0] for e in batcher.next_batch("R"))

    assert sorted(tags) == list(range(1024))


def test_an_exhausted_source_still_emits_its_final_partial_batch():
    """
    A short pool must not have its remainder trimmed away.

    The infinite pretraining corpus never reaches this, but a finite one --
    an eval set, a smoke test -- would otherwise silently lose up to a batch.
    """
    source = SyntheticSource(count=20, seed=10)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=8, seed=10)

    emitted = []
    while True:
        try:
            emitted.extend(batcher.next_batch("R"))
        except StopIteration:
            break

    assert sorted(e.decoder_target_ids[0] for e in emitted) == list(range(20))


def test_batches_are_full_while_the_source_has_examples():
    source = SyntheticSource(count=4096, seed=11)
    batcher = LengthBucketedBatcher(source, batch_size=16, pool_batches=8, seed=11)
    assert all(len(batcher.next_batch("R")) == 16 for _ in range(100))


# -- modes stay separate -----------------------------------------------------


def test_modes_are_pooled_separately():
    """
    Sorting across modes would put a 2048-token packed window beside a
    60-token `[S]` document and re-create the waste being removed.
    """
    def produce(mode):
        return make_example(1800 if mode in ("R", "X") else 60, mode)

    batcher = LengthBucketedBatcher(produce, batch_size=8, pool_batches=4, seed=12)
    assert {e.mode for e in batcher.next_batch("S")} == {"S"}
    assert {e.mode for e in batcher.next_batch("R")} == {"R"}
    assert all(e.encoder_length == 60 for e in batcher.next_batch("S"))


def test_the_caller_decides_the_mode_mix():
    """Bucketing regroups examples; it must not reweight the mode ratio."""
    def produce(mode):
        return make_example(random.Random(hash(mode) & 0xFF).randint(50, 500), mode)

    batcher = LengthBucketedBatcher(produce, batch_size=4, pool_batches=4, seed=13)
    requested = ["R"] * 30 + ["X"] * 20 + ["S"] * 10
    random.Random(0).shuffle(requested)

    emitted = [batcher.next_batch(m)[0].mode for m in requested]
    assert emitted == requested


def test_mode_none_is_supported_for_unpacked_runs():
    source = SyntheticSource(seed=14)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=4, seed=14)
    batch = batcher.next_batch(None)
    assert len(batch) == 8


# -- resumability ------------------------------------------------------------


def test_state_round_trip_continues_the_identical_stream():
    """
    The pool holds examples already read from the source. Losing them on
    resume is exactly the silent data loss architecture.md §7.9 forbids.
    """
    source = SyntheticSource(count=2048, seed=15)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=8, seed=15)
    for _ in range(5):
        batcher.next_batch("R")

    state = batcher.state_dict()
    # Three of the pool's eight batches are still owed. Only those are this
    # class's to restore -- where the *source* resumes is the corpus's own
    # state (`TokenizedCorpus.state_dict`), which is why the fresh source
    # below is deliberately unconsumed and only the pooled batches compared.
    expected = [e.decoder_target_ids[0] for _ in range(3) for e in batcher.next_batch("R")]

    resumed = LengthBucketedBatcher(SyntheticSource(count=2048, seed=15), batch_size=8,
                                    pool_batches=8, seed=15)
    resumed.load_state_dict(state)
    got = [e.decoder_target_ids[0] for _ in range(3) for e in resumed.next_batch("R")]

    assert got == expected
    assert len(got) == 24


def test_pending_examples_are_all_carried_in_the_state():
    source = SyntheticSource(count=512, seed=16)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=8, seed=16)
    batcher.next_batch("R")

    # A pool of 8 x 8 = 64 was filled and one batch handed out.
    assert batcher.pending_examples() == 56
    restored = LengthBucketedBatcher(source, batch_size=8, pool_batches=8, seed=16)
    restored.load_state_dict(batcher.state_dict())
    assert restored.pending_examples() == 56


def test_state_is_json_serializable():
    """`checkpoint.py` writes the data position into trainer_state.json."""
    import json

    source = SyntheticSource(count=256, seed=17)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=4, seed=17)
    batcher.next_batch("R")
    batcher.next_batch("S")

    restored = LengthBucketedBatcher(source, batch_size=8, pool_batches=4, seed=17)
    restored.load_state_dict(json.loads(json.dumps(batcher.state_dict())))
    assert restored.pending_examples() == batcher.pending_examples()


def test_resuming_into_a_different_batch_size_is_refused():
    """
    The queued batches were cut to the old size. Reshaping them means dropping
    or duplicating examples, so this fails loudly instead.
    """
    source = SyntheticSource(count=256, seed=18)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=4, seed=18)
    batcher.next_batch("R")

    other = LengthBucketedBatcher(source, batch_size=16, pool_batches=4, seed=18)
    with pytest.raises(UL2Error, match="batch_size"):
        other.load_state_dict(batcher.state_dict())


def test_pool_batches_may_change_across_a_resume():
    """
    It governs only the size of the next fill, so already-cut batches stay
    valid -- and the value this run was *configured* with must win over the
    one the interrupted run happened to use.
    """
    source = SyntheticSource(count=2048, seed=19)
    batcher = LengthBucketedBatcher(source, batch_size=8, pool_batches=4, seed=19)
    batcher.next_batch("R")

    resumed = LengthBucketedBatcher(SyntheticSource(count=2048, seed=19), batch_size=8,
                                    pool_batches=32, seed=19)
    resumed.load_state_dict(batcher.state_dict())
    assert resumed.pool_batches == 32
    assert len(resumed.next_batch("R")) == 8


# -- construction ------------------------------------------------------------


@pytest.mark.parametrize("batch_size,pool_batches", [(0, 4), (-1, 4), (4, 0), (4, -1)])
def test_invalid_sizes_are_rejected(batch_size, pool_batches):
    with pytest.raises(UL2Error):
        LengthBucketedBatcher(lambda m: make_example(10), batch_size=batch_size,
                              pool_batches=pool_batches)


def test_default_pool_is_the_measured_knee():
    """Guards against the default drifting silently; 16 is what the sim justified."""
    assert DEFAULT_POOL_BATCHES == 16


# -- the sort key serves both padded dimensions ------------------------------


def test_batching_tightens_the_decoder_dimension_too():
    """
    `pad_batch` pads encoder and decoder independently, so a key that only
    tightens the encoder leaves half the waste in place.

    Uses anti-correlated lengths on purpose -- long encoder paired with short
    decoder -- which is the shape `[S]` actually produces, since it splits one
    document into a prefix and its continuation.
    """
    examples = []
    for i in range(256):
        encoder = 50 + 4 * i
        example = make_example(encoder, "S", tag=i)
        examples.append(
            Example(
                mode="S",
                encoder_input_ids=example.encoder_input_ids,
                decoder_target_ids=tuple(range(1100 - 4 * i)),
                decoder_input_ids=tuple(range(1100 - 4 * i)),
                source_length=encoder,
                num_spans=1,
                num_corrupted_tokens=1,
            )
        )

    def produce(_mode):
        if not examples:
            raise StopIteration
        return examples.pop(0)

    batcher = LengthBucketedBatcher(produce, batch_size=8, pool_batches=32, seed=20)
    batches = []
    while True:
        try:
            batches.append(batcher.next_batch("S"))
        except StopIteration:
            break

    decoder_real = sum(e.target_length for b in batches for e in b)
    decoder_padded = sum(max(e.target_length for e in b) * len(b) for b in batches)
    assert decoder_real / decoder_padded > 0.6, (
        "the decoder dimension is barely tightened; the sort key is ignoring it"
    )
