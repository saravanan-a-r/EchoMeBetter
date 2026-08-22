"""
Regression lock on the frozen decisions in architecture.md.

Every value asserted here was decided deliberately and documented with its
reasoning. This file is not testing behaviour -- it is testing that the code
still agrees with the design document. A failure here means someone is about
to change the pretraining objective, and should be read as "stop and go read
section N", not as "update the expected number".

Where a number was justified by arithmetic in the document, the arithmetic is
re-derived here rather than restated, so the justification is checked and not
merely copied.
"""

from __future__ import annotations

import random

import pytest

from conftest import make_tokens
from src.config import (
    DEFAULT_MAX_SOURCE_LENGTH,
    DEFAULT_MAX_TARGET_LENGTH,
    DEFAULT_MODE_WEIGHTS,
    FROZEN_EVAL_MODE_WEIGHTS,
    UL2Config,
)
from src.objective import UL2Objective
from src.special_tokens import DEFAULT_NUM_SENTINELS, MODES


# -- section 4.4: context length, FROZEN ---------------------------------


def test_context_is_2048_encoder_and_2048_decoder():
    """
    architecture.md 4.4 (FROZEN). The decoder was raised from 1024 to match
    the encoder for two independent reasons; either alone is sufficient.
    """
    assert DEFAULT_MAX_SOURCE_LENGTH == 2048
    assert DEFAULT_MAX_TARGET_LENGTH == 2048
    assert UL2Config().max_source_length == 2048
    assert UL2Config().max_target_length == 2048


def test_a_1024_decoder_would_break_x_denoising():
    """
    architecture.md 4.4, reason 2, re-derived. X masks ~50% of a 2048-token
    input; the target is all of it plus one sentinel per span, which
    overflows a 1024-token decoder. Training on a truncated target teaches
    incomplete reconstruction -- corrupting the objective, not just serving.
    """
    config = UL2Config()
    target, _ = config.worst_case_target_length("X")

    assert target > 1024, "the reason the decoder could not stay at 1024"
    assert target <= 2048, "and the reason 2048 is sufficient"


def test_x_denoising_target_matches_the_documented_estimate():
    """architecture.md 4.4 predicted ~1,040-1,056 target tokens; verify."""
    target, at_length = UL2Config().worst_case_target_length("X")
    assert at_length > 2000
    assert 1040 <= target <= 1060


# -- section 6.2 / 6.3: sentinel budget, FROZEN --------------------------


def test_sentinel_budget_is_256():
    """architecture.md 6.2 (FROZEN): the tokenizer reserves 256 sentinels."""
    assert DEFAULT_NUM_SENTINELS == 256


def test_the_sentinel_requirement_formula_holds():
    """
    architecture.md 6.3:

        sentinels_needed = max_encoder_length x corruption_rate / mean_span_length
                         = 2048 x 0.15 / 3 = ~103

    Re-derived from the configuration rather than hardcoded, so changing the
    context length or the corruption settings forces this to be rechecked.
    """
    config = UL2Config()
    formula = config.max_source_length * config.r.corruption_rate / config.r.mean_span_length
    actual, _ = config.worst_case_spans("R")

    assert formula == pytest.approx(102.4)
    assert abs(actual - formula) <= 1


def test_t5s_original_100_sentinels_would_have_been_insufficient():
    """
    architecture.md 6.3's central claim: T5's 100 was sized for a 512-token
    context and would bind just below our encoder limit. This is why the
    budget was raised rather than the objective bent to fit it.
    """
    spans_needed, _ = UL2Config().worst_case_spans("R")
    assert spans_needed > 100

    with pytest.raises(Exception):
        UL2Config().validate_capacity(100)


def test_256_sentinels_give_the_documented_headroom():
    """architecture.md 6.3: 2.5x headroom at 2048, and enough for 4096."""
    at_2048, _ = UL2Config().worst_case_spans("R")
    assert DEFAULT_NUM_SENTINELS / at_2048 >= 2.4

    at_4096, _ = UL2Config(
        max_source_length=4096, max_target_length=4096
    ).worst_case_spans("R")
    assert at_4096 <= DEFAULT_NUM_SENTINELS


# -- section 7.2: the three modes ----------------------------------------


def test_exactly_three_denoising_modes():
    assert MODES == ("R", "X", "S")


def test_r_denoising_settings():
    """architecture.md 7.2: short spans, ~15% of tokens, mean span 3."""
    r = UL2Config().r
    assert r.corruption_rate == 0.15
    assert r.mean_span_length == 3.0


def test_x_denoising_settings():
    """architecture.md 7.2: long spans, ~50% of tokens."""
    x = UL2Config().x
    assert x.corruption_rate == 0.50
    assert x.mean_span_length == 32.0


def test_s_denoising_uses_no_sentinels(specials):
    """architecture.md 7.2: [S] is prefix -> continuation, no sentinels."""
    objective = UL2Objective(UL2Config(), specials)
    example = objective.corrupt(make_tokens(500), random.Random(0), mode="S")

    assert example.num_spans == 0
    assert not set(example.encoder_input_ids) & set(specials.sentinel_ids)


def test_every_mode_prefixes_its_own_mode_token(specials):
    """architecture.md 7.2: the model is told which problem it is solving."""
    objective = UL2Objective(UL2Config(), specials)
    seen = set()
    for mode in MODES:
        example = objective.corrupt(make_tokens(300), random.Random(0), mode=mode)
        token = example.encoder_input_ids[0]
        assert token == specials.mode_token(mode)
        seen.add(token)
    assert len(seen) == 3, "each mode must be distinguishable from position 0"


# -- section 7.3: mixture weights ----------------------------------------


def test_mixture_weights_are_40_35_25():
    """architecture.md 7.3, as sampling probabilities."""
    assert DEFAULT_MODE_WEIGHTS == {"R": 0.40, "X": 0.35, "S": 0.25}
    assert sum(DEFAULT_MODE_WEIGHTS.values()) == pytest.approx(1.0)


def test_mixture_weights_are_sampling_probabilities_not_token_shares(specials):
    """
    architecture.md 7.3's explicit warning. Sampling at 40/35/25 does not
    deliver 40/35/25 of decoder training tokens, and the pipeline must be
    able to show the difference rather than assume it away.
    """
    from src.telemetry import MixtureTelemetry

    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    objective = UL2Objective(UL2Config(), specials, telemetry)

    rng = random.Random(11)
    for _ in range(3000):
        objective.corrupt(make_tokens(600), rng)

    modes = telemetry.report()["modes"]
    assert modes["R"]["realized_example_share"] == pytest.approx(0.40, abs=0.03)
    assert abs(modes["R"]["decoder_token_share"] - 0.40) > 0.10, (
        "the probability/exposure gap must be observable, which is why "
        "architecture.md 7.3 requires both be logged"
    )


def test_all_required_telemetry_is_reported(specials):
    """architecture.md 7.3's table of required per-mode measurements."""
    from src.telemetry import MixtureTelemetry

    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    objective = UL2Objective(UL2Config(), specials, telemetry)

    rng = random.Random(3)
    for _ in range(200):
        objective.corrupt(make_tokens(400), rng)

    for entry in telemetry.report()["modes"].values():
        assert entry["configured_probability"] is not None   # mode probability
        assert entry["encoder_tokens"] > 0                   # encoder exposure
        assert entry["decoder_tokens"] > 0                   # decoder exposure
        assert entry["realized_corruption_rate"] > 0         # realized rate
        assert entry["target_length_histogram"]              # target lengths


# -- section 7.4: sentinel budget in practice ----------------------------


def test_span_overflow_fails_loudly_rather_than_wrapping(small_specials):
    """
    architecture.md 7.4: "never wrap around or silently drop a span."
    Wrapping would reuse <extra_id_0> twice in one example, making the
    target ambiguous and quietly poisoning training.
    """
    with pytest.raises(Exception, match="sentinel"):
        UL2Objective(UL2Config(), small_specials)


# -- section 7.6: held-out evaluation, FROZEN ----------------------------


def test_validation_mixture_is_fixed_and_differs_from_training():
    """
    architecture.md 7.6 (FROZEN): the validation mode mixture must not follow
    the training mixture, or an X-heavy run is scored on X-heavy validation
    data and the comparison is circular.
    """
    assert sum(FROZEN_EVAL_MODE_WEIGHTS.values()) == pytest.approx(1.0)
    assert dict(FROZEN_EVAL_MODE_WEIGHTS) != dict(DEFAULT_MODE_WEIGHTS)
    for weight in FROZEN_EVAL_MODE_WEIGHTS.values():
        assert weight == pytest.approx(1 / 3)


def test_per_mode_losses_can_be_reported_separately(specials):
    """
    architecture.md 7.6: "Report per-mode losses separately -- an aggregate
    can look healthy while one objective silently fails to learn." That
    requires the frozen set to expose its examples by mode.
    """
    from src.frozen_eval import build_frozen_eval_set

    sequences = [make_tokens(200, start=20000 + i * 500) for i in range(30)]
    frozen = build_frozen_eval_set(sequences, UL2Config(), specials, seed=1)

    for mode in MODES:
        assert len(frozen.by_mode(mode)) == 10


# -- section 4.5: over-length policy, FROZEN -----------------------------


def test_silent_truncation_is_forbidden_by_default():
    """architecture.md 4.5 (FROZEN)."""
    assert UL2Config().on_overlong_source == "error"


def test_explicit_truncation_is_flagged_on_the_example(specials):
    """Opt-in truncation is allowed; invisible truncation is not."""
    objective = UL2Objective(UL2Config(on_overlong_source="truncate"), specials)
    example = objective.corrupt(make_tokens(4000), random.Random(0), mode="R")
    assert example.truncated is True


# -- section 6.4: no hardcoded token IDs ---------------------------------


def test_nothing_assumes_contiguous_or_ordered_sentinel_ids(specials):
    """
    architecture.md 6.4: SentencePiece's ordering is an implementation
    detail. The fixture's sentinel IDs are deliberately scrambled; if any
    code assumed `sentinel(i) == base + i`, reconstruction would fail.
    """
    from src.denoise import reconstruct_source

    assert specials.sentinel_ids != tuple(sorted(specials.sentinel_ids))

    objective = UL2Objective(UL2Config(), specials)
    tokens = make_tokens(800)
    for mode in MODES:
        example = objective.corrupt(tokens, random.Random(0), mode=mode)
        assert reconstruct_source(example, specials) == tuple(tokens)
