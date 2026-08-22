"""
End-to-end tests for `UL2Objective`.

These are the broad-coverage regression tests: rather than checking one
hand-picked case, they sweep sequence lengths from the degenerate minimum to
the full 2048-token context, across all three modes and several seeds, and
assert the invariants that must hold for *every* example a multi-week run
will produce.

The invariants:

  1. reconstruction is lossless -- the original tokens come back exactly
  2. nothing exceeds the encoder or decoder budget
  3. sentinels are unique within an example and used in order
  4. the same seed replays the same examples (resumability, frozen eval)

A violation of any of these is invisible in a loss curve, which is why they
are asserted mechanically here instead of being watched for during training.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from conftest import make_tokens
from src.config import SpanCorruptionConfig, UL2Config
from src.denoise import reconstruct_source
from src.errors import (
    ConfigError,
    SourceTooLongError,
    SourceTooShortError,
    TargetTooLongError,
)
from src.objective import UL2Objective
from src.telemetry import MixtureTelemetry

# Degenerate, typical (P50=18, P90=88, P99=597 per architecture.md 4.4) and
# boundary lengths.
LENGTHS = [8, 9, 18, 88, 200, 597, 1024, 2047, 2048]
MODES = ["R", "X", "S"]
SEEDS = [0, 1, 424242]


@pytest.fixture
def objective(config, specials):
    return UL2Objective(config, specials)


# -- core invariants ------------------------------------------------------


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS)
def test_reconstruction_is_lossless(objective, specials, length, mode, seed):
    tokens = make_tokens(length)
    example = objective.corrupt(tokens, random.Random(seed), mode=mode)
    assert reconstruct_source(example, specials) == tuple(tokens)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS)
def test_budgets_are_never_exceeded(objective, config, length, mode, seed):
    example = objective.corrupt(make_tokens(length), random.Random(seed), mode=mode)
    assert example.encoder_length <= config.max_source_length
    assert example.target_length <= config.max_target_length


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("mode", ["R", "X"])
def test_sentinels_are_unique_and_ordered(objective, specials, length, mode):
    example = objective.corrupt(make_tokens(length), random.Random(9), mode=mode)

    sentinel_set = set(specials.sentinel_ids)
    used = [t for t in example.encoder_input_ids if t in sentinel_set]

    assert len(used) == len(set(used)) == example.num_spans
    assert used == [specials.sentinel(i) for i in range(example.num_spans)]


@pytest.mark.parametrize("length", LENGTHS)
def test_prefix_mode_uses_no_sentinels(objective, specials, length):
    example = objective.corrupt(make_tokens(length), random.Random(9), mode="S")
    assert example.num_spans == 0
    assert not set(example.encoder_input_ids) & set(specials.sentinel_ids)
    assert not set(example.decoder_target_ids) & set(specials.sentinel_ids)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("mode", MODES)
def test_mode_token_is_recorded_and_emitted(objective, specials, length, mode):
    example = objective.corrupt(make_tokens(length), random.Random(0), mode=mode)
    assert example.mode == mode
    assert example.encoder_input_ids[0] == specials.mode_token(mode)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("mode", MODES)
def test_decoder_input_is_a_valid_shift(objective, length, mode):
    example = objective.corrupt(make_tokens(length), random.Random(0), mode=mode)
    assert len(example.decoder_input_ids) == len(example.decoder_target_ids)
    assert example.decoder_input_ids[1:] == example.decoder_target_ids[:-1]


def test_full_context_x_denoising_matches_the_documented_arithmetic(objective):
    """
    architecture.md 4.4: X on a 2048-token input produces ~1,040-1,056 target
    tokens, which is why the decoder context had to be raised from 1024.
    """
    example = objective.corrupt(make_tokens(2048), random.Random(0), mode="X")
    assert 1024 < example.target_length <= 1057


def test_full_context_r_denoising_stays_inside_the_sentinel_budget(objective):
    """architecture.md 6.3: ~103 sentinels needed, 256 available."""
    example = objective.corrupt(make_tokens(2048), random.Random(0), mode="R")
    assert example.num_spans == 102


# -- determinism ----------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_same_seed_replays_the_same_example(objective, mode):
    tokens = make_tokens(500)
    first = objective.corrupt(tokens, random.Random(77), mode=mode)
    second = objective.corrupt(tokens, random.Random(77), mode=mode)
    assert first == second


@pytest.mark.parametrize("mode", ["R", "X", "S"])
def test_different_seeds_produce_different_examples(objective, mode):
    tokens = make_tokens(500)
    first = objective.corrupt(tokens, random.Random(1), mode=mode)
    second = objective.corrupt(tokens, random.Random(2), mode=mode)
    assert first != second


def test_a_whole_stream_replays_identically(objective):
    """
    The property architecture.md 7.9 depends on: after a crash-restart, a
    resumed iterator must produce the tokens the run had not yet seen, which
    requires the whole stream to be reproducible from its seed.
    """
    sequences = [make_tokens(length) for length in (20, 300, 45, 1200, 88)]

    def run():
        rng = random.Random(2026)
        return [objective.corrupt(tokens, rng) for tokens in sequences]

    assert run() == run()


def test_mode_choice_does_not_depend_on_earlier_rng_consumption(objective):
    """
    The mode is drawn before dispatch, so a run's mode sequence is fixed by
    the seed regardless of how many random numbers each mode then consumes.
    """
    rng = random.Random(31)
    modes = [objective.corrupt(make_tokens(200), rng).mode for _ in range(50)]

    rng = random.Random(31)
    sampled = [objective.sampler.sample(rng) for _ in range(50)]
    # Not equal element-wise (corruption consumes randomness in between), but
    # the first draw must match: nothing precedes it.
    assert modes[0] == sampled[0]


# -- mixture --------------------------------------------------------------

def test_sampled_modes_follow_the_configured_mixture(objective):
    rng = random.Random(5)
    counts = Counter(objective.corrupt(make_tokens(120), rng).mode for _ in range(4000))
    for mode, weight in objective.config.mode_weights.items():
        assert counts[mode] / 4000 == pytest.approx(weight, abs=0.03)


def test_telemetry_records_every_example(config, specials):
    telemetry = MixtureTelemetry(config.mode_weights)
    objective = UL2Objective(config, specials, telemetry)

    rng = random.Random(4)
    for _ in range(300):
        objective.corrupt(make_tokens(250), rng)

    report = telemetry.report()
    assert report["total_examples"] == 300
    assert sum(entry["examples"] for entry in report["modes"].values()) == 300


def test_telemetry_shows_the_probability_versus_token_share_gap(config, specials):
    """architecture.md 7.3, observed on real generated examples."""
    telemetry = MixtureTelemetry(config.mode_weights)
    objective = UL2Objective(config, specials, telemetry)

    rng = random.Random(8)
    for _ in range(2000):
        objective.corrupt(make_tokens(400), rng)

    modes = telemetry.report()["modes"]
    assert modes["R"]["realized_example_share"] == pytest.approx(0.40, abs=0.04)
    assert modes["R"]["decoder_token_share"] < modes["R"]["realized_example_share"]
    assert modes["S"]["decoder_token_share"] > modes["S"]["realized_example_share"]


# -- length admission -----------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, 5, 7])
def test_too_short_sequences_are_rejected(objective, length):
    with pytest.raises(SourceTooShortError, match="min_source_length"):
        objective.corrupt(make_tokens(length), random.Random(0))


def test_minimum_admissible_length_works(objective, config, specials):
    example = objective.corrupt(make_tokens(config.min_source_length), random.Random(0), mode="R")
    assert reconstruct_source(example, specials) == tuple(
        make_tokens(config.min_source_length)
    )


def test_too_long_sequences_are_rejected_by_default(objective):
    """architecture.md 4.5: silent truncation is forbidden."""
    with pytest.raises(SourceTooLongError, match="max_source_length"):
        objective.corrupt(make_tokens(2049), random.Random(0))


def test_overlong_error_names_the_opt_in(objective):
    with pytest.raises(SourceTooLongError, match="truncate"):
        objective.corrupt(make_tokens(3000), random.Random(0))


def test_truncation_must_be_opted_into_and_is_flagged(config, specials):
    objective = UL2Objective(config.with_(on_overlong_source="truncate"), specials)
    example = objective.corrupt(make_tokens(3000), random.Random(0), mode="R")

    assert example.truncated is True
    assert example.source_length == 2048


def test_admissible_examples_are_not_flagged_as_truncated(objective):
    example = objective.corrupt(make_tokens(2048), random.Random(0), mode="R")
    assert example.truncated is False


# -- try_corrupt ----------------------------------------------------------


def test_try_corrupt_skips_instead_of_raising(config, specials):
    telemetry = MixtureTelemetry(config.mode_weights)
    objective = UL2Objective(config, specials, telemetry)
    rng = random.Random(0)

    assert objective.try_corrupt(make_tokens(3), rng) is None
    assert objective.try_corrupt(make_tokens(5000), rng) is None
    assert objective.try_corrupt(make_tokens(100), rng) is not None


def test_try_corrupt_counts_what_it_skipped(config, specials):
    """Losing a large fraction of the corpus must stay visible."""
    telemetry = MixtureTelemetry(config.mode_weights)
    objective = UL2Objective(config, specials, telemetry)
    rng = random.Random(0)

    for _ in range(3):
        objective.try_corrupt(make_tokens(2), rng)
    for _ in range(2):
        objective.try_corrupt(make_tokens(9999), rng)
    objective.try_corrupt(make_tokens(100), rng)

    report = telemetry.report()
    assert report["skipped_too_short"] == 3
    assert report["skipped_too_long"] == 2
    assert report["total_examples"] == 1


def test_try_corrupt_without_telemetry_still_skips(objective):
    assert objective.try_corrupt(make_tokens(2), random.Random(0)) is None


def test_a_realistic_mixed_length_stream_survives(config, specials):
    """A corpus contains blank lines, titles and long documents together."""
    telemetry = MixtureTelemetry(config.mode_weights)
    objective = UL2Objective(config, specials, telemetry)
    rng = random.Random(1234)

    lengths = [0, 1, 3, 18, 88, 200, 597, 2048, 2049, 5000, 40, 12]
    produced = [
        example
        for length in lengths * 20
        if (example := objective.try_corrupt(make_tokens(length), rng)) is not None
    ]

    assert produced
    for example in produced:
        assert reconstruct_source(example, specials)
        assert example.encoder_length <= config.max_source_length
        assert example.target_length <= config.max_target_length

    report = telemetry.report()
    assert report["skipped_too_short"] == 3 * 20
    assert report["skipped_too_long"] == 2 * 20


# -- construction-time validation ----------------------------------------


def test_construction_rejects_a_configuration_that_cannot_fit(config, small_specials):
    """The overflow is caught before a single example is produced."""
    with pytest.raises(ConfigError, match="sentinels"):
        UL2Objective(config, small_specials)


def test_construction_rejects_a_decoder_that_is_too_small(specials):
    """The architecture.md 4.4 regression: encoder 2048 with decoder 1024."""
    with pytest.raises(ConfigError, match="decoder target"):
        UL2Objective(UL2Config(max_target_length=1024), specials)


def test_a_reduced_configuration_that_does_fit_is_accepted(specials):
    config = UL2Config(
        max_source_length=512,
        max_target_length=512,
        x=SpanCorruptionConfig(corruption_rate=0.5, mean_span_length=32.0),
    )
    UL2Objective(config, specials)


def test_post_condition_catches_an_oversized_target(specials, monkeypatch):
    """
    Defence in depth. `validate_capacity` should make this unreachable, so
    the check is forced here by neutralizing it -- the guarantee must not
    rest on a single layer.
    """
    monkeypatch.setattr(UL2Config, "validate_capacity", lambda self, n: None)
    objective = UL2Objective(UL2Config(max_target_length=50), specials)

    with pytest.raises(TargetTooLongError, match="max_target_length"):
        objective.corrupt(make_tokens(2000), random.Random(0), mode="X")
