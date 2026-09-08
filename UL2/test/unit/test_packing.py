"""
Packing several documents into one window (`src/packing.py`).

The properties tested here are the ones whose violation is *silent*. A span
that strays across a document boundary, a sentinel reused between two packed
documents, a segment id off by one — none of these crash, none show up in a
loss curve, and all of them corrupt what the model learns. So each is asserted
directly rather than inferred from an example that happens to look right.
"""

from __future__ import annotations

import random

import pytest

from src.config import UL2Config
from src.denoise import Example
from src.errors import ConfigError, SentinelBudgetExceededError, UL2Error
from src.objective import UL2Objective
from src.packing import (
    FIRST_DOCUMENT_SEGMENT_ID,
    PAD_SEGMENT_ID,
    build_packed_span_corruption,
    pack_documents,
    reconstruct_documents,
)
from src.special_tokens import SpecialTokens


@pytest.fixture
def specials() -> SpecialTokens:
    return SpecialTokens(
        sentinel_ids=list(range(1000, 1256)),
        mode_token_ids={"R": 900, "X": 901, "S": 902},
        eos_id=1,
        pad_id=0,
        decoder_start_id=0,
    )


@pytest.fixture
def objective(specials: SpecialTokens) -> UL2Objective:
    return UL2Objective(UL2Config(), specials)


def documents() -> list[list[int]]:
    """
    Three documents drawn from disjoint token ranges.

    Disjoint on purpose: it makes any leakage between documents provable from
    the token values alone, with no bookkeeping about offsets.
    """
    return [list(range(100, 400)), list(range(500, 760)), list(range(2000, 2500))]


# -- pack_documents ---------------------------------------------------------


def test_packing_stops_before_overflowing_the_window():
    docs = documents()  # costs 302, 262, 502 including each [MODE] and </s>
    assert pack_documents(docs, window_budget=700, max_documents=64) == 2
    assert pack_documents(docs, window_budget=2048, max_documents=64) == 3


def test_packing_honours_the_document_cap():
    docs = documents()
    assert pack_documents(docs, window_budget=10_000, max_documents=2) == 2


def test_a_single_oversized_document_is_handed_on_rather_than_dropped():
    """
    The length policy belongs to `UL2Objective._admit`, which counts the skip
    in telemetry (architecture.md 4.5's no-silent-loss rule). If the packer
    dropped it here instead, the record would vanish with nothing recording
    that it had.
    """
    assert pack_documents([list(range(5000))], window_budget=100, max_documents=64) == 1


def test_packing_nothing_is_not_an_error():
    assert pack_documents([], window_budget=2048, max_documents=64) == 0


# -- the core invariants ----------------------------------------------------


@pytest.mark.parametrize("mode", ["R", "X"])
def test_a_packed_window_round_trips_back_to_its_documents(specials, objective, mode):
    docs = documents()
    example = objective.corrupt_packed(docs, random.Random(0), mode=mode)
    assert reconstruct_documents(example, specials) == tuple(tuple(d) for d in docs)


@pytest.mark.parametrize("mode", ["R", "X"])
def test_no_span_ever_crosses_a_document_boundary(specials, objective, mode):
    """
    The failure this whole module exists to prevent: a span covering the tail
    of one document and the head of the next trains the model to predict text
    that is genuinely unpredictable.

    Checked over many seeds because a boundary-crossing span is a *sampling*
    accident — one seed proving nothing.
    """
    docs = documents()
    ranges = [(100, 400), (500, 760), (2000, 2500)]
    sentinels = set(specials.sentinel_ids)

    for seed in range(200):
        example = objective.corrupt_packed(docs, random.Random(seed), mode=mode)

        span_owner: set[int] | None = None
        for token in example.decoder_target_ids:
            if token in sentinels:
                span_owner = None  # a new span begins
                continue
            if token == specials.eos_id:
                continue
            owners = {i for i, (low, high) in enumerate(ranges) if low <= token < high}
            assert owners, f"token {token} belongs to no document"
            span_owner = owners if span_owner is None else span_owner & owners
            assert span_owner, f"seed {seed}: a span spans two documents"


@pytest.mark.parametrize("mode", ["R", "X"])
def test_sentinels_are_numbered_across_the_whole_window_never_reused(
    specials, objective, mode
):
    """
    Each document is corrupted separately, so the tempting implementation
    numbers each document's sentinels from zero — and then two documents in
    one window both claim `<extra_id_0>` and the target cannot be parsed.
    """
    example = objective.corrupt_packed(documents(), random.Random(1), mode=mode)
    sentinels = [t for t in example.decoder_target_ids if t in set(specials.sentinel_ids)]
    assert len(sentinels) == len(set(sentinels))
    assert len(sentinels) == example.num_spans


@pytest.mark.parametrize("mode", ["R", "X"])
def test_every_encoder_position_is_labelled_with_its_document(specials, objective, mode):
    example = objective.corrupt_packed(documents(), random.Random(2), mode=mode)
    assert len(example.segment_ids) == len(example.encoder_input_ids)
    assert sorted(set(example.segment_ids)) == [1, 2, 3]
    assert PAD_SEGMENT_ID not in example.segment_ids
    # Segments are contiguous blocks, never interleaved.
    blocks = [example.segment_ids[0]]
    for segment in example.segment_ids[1:]:
        if segment != blocks[-1]:
            blocks.append(segment)
    assert blocks == [1, 2, 3]


def test_each_document_carries_its_own_mode_token(specials, objective):
    """
    Not one shared prefix for the window. Per-document mode tokens are what
    make every document see `[MODE]` at the same relative distance, which is
    why the relative position bias needs no packing-specific change at all
    (see `packing.py`'s module docstring).
    """
    example = objective.corrupt_packed(documents(), random.Random(3), mode="R")
    mode_token = specials.mode_token("R")
    assert example.encoder_input_ids.count(mode_token) == 3
    # ...and each one starts its own document's block.
    for position, token in enumerate(example.encoder_input_ids):
        if token == mode_token:
            previous = example.segment_ids[position - 1] if position else None
            assert previous != example.segment_ids[position]


def test_every_document_is_terminated(specials, objective):
    example = objective.corrupt_packed(documents(), random.Random(4), mode="X")
    assert example.encoder_input_ids.count(specials.eos_id) == 3


# -- [S] is never packed ----------------------------------------------------


def test_prefix_denoising_refuses_more_than_one_document(objective):
    with pytest.raises(UL2Error, match="exactly one document"):
        objective.corrupt_packed(documents(), random.Random(5), mode="S")


def test_prefix_denoising_still_works_on_a_single_document(objective):
    example = objective.corrupt_packed([list(range(200))], random.Random(6), mode="S")
    assert example.mode == "S"
    assert example.num_spans == 0


# -- one document packed is still a valid ordinary example ------------------


def test_a_one_document_window_matches_the_unpacked_builder(specials, objective):
    """
    Packing must be a generalization, not a second dialect: a window holding
    one document has to produce the same encoder and target an unpacked
    example would, so the two paths cannot drift apart.
    """
    tokens = list(range(100, 400))
    packed = objective.corrupt_packed([tokens], random.Random(11), mode="R")
    plain = objective.corrupt(tokens, random.Random(11), mode="R")
    assert packed.encoder_input_ids == plain.encoder_input_ids
    assert packed.decoder_target_ids == plain.decoder_target_ids
    assert packed.decoder_input_ids == plain.decoder_input_ids


# -- budgets ----------------------------------------------------------------


def test_the_sentinel_budget_is_shared_across_the_window(specials):
    """
    A window that needs more sentinels than exist must raise, never wrap:
    a wrapped index silently reuses a marker and makes the target ambiguous
    (architecture.md 7.4).
    """
    tiny = SpecialTokens(
        sentinel_ids=[1000, 1001],
        mode_token_ids={"R": 900, "X": 901, "S": 902},
        eos_id=1,
        pad_id=0,
        decoder_start_id=0,
    )
    with pytest.raises(SentinelBudgetExceededError):
        build_packed_span_corruption(
            [list(range(300)), list(range(300)), list(range(300))],
            "R",
            tiny,
            random.Random(0),
            corruption_rate=0.15,
            mean_span_length=3.0,
        )


def test_a_packed_window_stays_inside_the_configured_budgets(objective):
    config = UL2Config()
    for mode in ("R", "X"):
        for seed in range(25):
            example = objective.corrupt_packed(documents(), random.Random(seed), mode=mode)
            assert example.encoder_length <= config.max_source_length
            assert example.target_length <= config.max_target_length


def test_too_many_documents_is_refused_rather_than_silently_packed(specials):
    config = UL2Config(max_documents_per_window=2)
    objective = UL2Objective(config, specials)
    with pytest.raises(UL2Error, match="max_documents_per_window"):
        objective.corrupt_packed(documents(), random.Random(0), mode="R")


# -- the packed capacity proof ----------------------------------------------


def test_the_packed_capacity_proof_accepts_the_real_configuration(specials):
    """256 sentinels against the shipped defaults — must hold, or no run starts."""
    UL2Config().validate_packed_capacity(specials.num_sentinels)


def test_the_packed_capacity_proof_rejects_an_unsafe_document_cap():
    """
    The proof that `validate_capacity` alone cannot make: every document needs
    at least one span, so enough documents in one window exhaust the sentinels
    even though no single document ever could.
    """
    config = UL2Config(max_documents_per_window=1000)
    with pytest.raises(ConfigError, match="sentinels"):
        config.validate_packed_capacity(256)


def test_an_unsafe_document_cap_is_caught_when_the_objective_is_built(specials):
    config = UL2Config(max_documents_per_window=1000)
    with pytest.raises(ConfigError):
        UL2Objective(config, specials)


def test_prefix_denoising_needs_no_sentinels():
    assert UL2Config().worst_case_packed_spans("S") == 0


def test_the_document_cap_must_be_a_positive_integer():
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ConfigError, match="max_documents_per_window"):
            UL2Config(max_documents_per_window=bad)


def test_the_document_cap_survives_a_config_round_trip():
    config = UL2Config(max_documents_per_window=8)
    assert UL2Config.from_dict(config.to_dict()).max_documents_per_window == 8


# -- reconstruction rejects damage ------------------------------------------


def test_reconstruction_refuses_an_example_that_was_never_packed(specials, objective):
    plain = objective.corrupt(list(range(200)), random.Random(0), mode="R")
    with pytest.raises(UL2Error, match="segment_ids"):
        reconstruct_documents(plain, specials)


def test_reconstruction_refuses_mismatched_segment_ids(specials, objective):
    example = objective.corrupt_packed(documents(), random.Random(0), mode="R")
    damaged = Example(
        mode=example.mode,
        encoder_input_ids=example.encoder_input_ids,
        decoder_target_ids=example.decoder_target_ids,
        decoder_input_ids=example.decoder_input_ids,
        source_length=example.source_length,
        num_spans=example.num_spans,
        num_corrupted_tokens=example.num_corrupted_tokens,
        segment_ids=example.segment_ids[:-1],
        num_documents=example.num_documents,
    )
    with pytest.raises(UL2Error, match="segment_ids"):
        reconstruct_documents(damaged, specials)


# -- bookkeeping ------------------------------------------------------------


def test_a_packed_example_reports_what_it_holds(objective):
    example = objective.corrupt_packed(documents(), random.Random(0), mode="R")
    assert example.num_documents == 3
    assert example.is_packed
    assert example.source_length == sum(len(d) for d in documents())


def test_an_unpacked_example_is_not_marked_packed(objective):
    example = objective.corrupt(list(range(200)), random.Random(0), mode="R")
    assert example.num_documents == 1
    assert not example.is_packed
    assert example.segment_ids == ()


def test_packing_zero_documents_is_refused(objective):
    with pytest.raises(UL2Error):
        objective.corrupt_packed([], random.Random(0), mode="R")


def test_realized_corruption_rate_is_measured_over_the_whole_window(objective):
    example = objective.corrupt_packed(documents(), random.Random(0), mode="X")
    assert 0.3 < example.realized_corruption_rate < 0.7
