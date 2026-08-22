"""
Unit tests for mode selection.

Two distinct mechanisms live here, for two different jobs:

  - `ModeSampler` -- random draws for *training*, where variety is the point.
  - `deterministic_schedule` -- exact apportionment for *frozen evaluation*,
    where architecture.md 7.6 requires the mixture to be identical in every
    run so per-mode losses are computed over identical example counts.

Confusing the two would make validation losses jitter between runs for
reasons unrelated to the model.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from src.config import DEFAULT_MODE_WEIGHTS
from src.errors import ConfigError
from src.mixture import ModeSampler, deterministic_schedule


# -- ModeSampler ----------------------------------------------------------


def test_sampler_reports_its_modes_in_canonical_order():
    """
    Canonical order, not insertion order: the cumulative boundaries -- and
    so the mode drawn for a given RNG state -- must not depend on how the
    weights dict happened to be built.
    """
    sampler = ModeSampler({"S": 0.25, "R": 0.40, "X": 0.35})
    assert sampler.modes == ("R", "X", "S")


def test_dict_ordering_does_not_change_the_draw_sequence():
    forward = ModeSampler({"R": 0.4, "X": 0.35, "S": 0.25})
    reversed_ = ModeSampler({"S": 0.25, "X": 0.35, "R": 0.4})
    assert forward.sample_many(200, random.Random(7)) == reversed_.sample_many(
        200, random.Random(7)
    )


def test_sampled_frequencies_track_the_configured_weights():
    sampler = ModeSampler(DEFAULT_MODE_WEIGHTS)
    counts = Counter(sampler.sample_many(20000, random.Random(3)))

    for mode, weight in DEFAULT_MODE_WEIGHTS.items():
        assert counts[mode] / 20000 == pytest.approx(weight, abs=0.02)


def test_sampling_is_deterministic_given_a_seed():
    sampler = ModeSampler(DEFAULT_MODE_WEIGHTS)
    assert sampler.sample_many(500, random.Random(11)) == sampler.sample_many(
        500, random.Random(11)
    )


def test_sampling_varies_across_seeds():
    sampler = ModeSampler(DEFAULT_MODE_WEIGHTS)
    assert sampler.sample_many(500, random.Random(1)) != sampler.sample_many(
        500, random.Random(2)
    )


def test_each_draw_consumes_exactly_one_random_number():
    """
    A frozen eval set must replay identically. That holds only if the number
    of RNG calls per draw is fixed, so a mode change cannot shift the stream.
    """
    sampler = ModeSampler(DEFAULT_MODE_WEIGHTS)

    rng = random.Random(5)
    sampler.sample_many(100, rng)
    after_sampling = rng.random()

    rng = random.Random(5)
    for _ in range(100):
        rng.random()
    assert rng.random() == after_sampling


def test_zero_weight_mode_is_never_drawn():
    sampler = ModeSampler({"R": 0.5, "X": 0.0, "S": 0.5})
    assert "X" not in set(sampler.sample_many(5000, random.Random(0)))
    assert sampler.modes == ("R", "S")


def test_single_mode_sampler_always_returns_it():
    sampler = ModeSampler({"X": 1.0})
    assert set(sampler.sample_many(100, random.Random(0))) == {"X"}


def test_unnormalized_weights_are_normalized_by_the_sampler():
    """
    `UL2Config` insists weights sum to 1; the sampler itself is tolerant so
    it can be used directly with raw ratios in a diagnostic script.
    """
    sampler = ModeSampler({"R": 2.0, "X": 2.0})
    assert sampler.probability("R") == pytest.approx(0.5)

    counts = Counter(sampler.sample_many(4000, random.Random(0)))
    assert counts["R"] / 4000 == pytest.approx(0.5, abs=0.03)


def test_probability_reports_the_normalized_value():
    sampler = ModeSampler(DEFAULT_MODE_WEIGHTS)
    assert sampler.probability("R") == pytest.approx(0.40)
    assert sampler.probability("X") == pytest.approx(0.35)
    assert sampler.probability("S") == pytest.approx(0.25)


def test_weights_property_is_a_copy():
    sampler = ModeSampler(DEFAULT_MODE_WEIGHTS)
    sampler.weights["R"] = 99.0
    assert sampler.probability("R") == pytest.approx(0.40)


@pytest.mark.parametrize(
    "weights,match",
    [
        ({}, "empty"),
        ({"Q": 1.0}, "unknown modes"),
        ({"R": -1.0, "X": 2.0}, "negative"),
        ({"R": 0.0, "X": 0.0, "S": 0.0}, "every mode weight is zero"),
    ],
)
def test_invalid_weights_rejected(weights, match):
    with pytest.raises(ConfigError, match=match):
        ModeSampler(weights)


def test_a_draw_at_the_very_top_of_the_range_still_returns_a_mode():
    """Floating-point safety: 0.999... must not fall off the end of the table."""

    class AlmostOne(random.Random):
        def random(self) -> float:  # type: ignore[override]
            return 0.9999999999999999

    assert ModeSampler(DEFAULT_MODE_WEIGHTS).sample(AlmostOne()) == "S"


def test_a_draw_at_zero_returns_the_first_mode():
    class Zero(random.Random):
        def random(self) -> float:  # type: ignore[override]
            return 0.0

    assert ModeSampler(DEFAULT_MODE_WEIGHTS).sample(Zero()) == "R"


# -- deterministic_schedule ----------------------------------------------


def test_schedule_length_always_matches_the_request():
    for count in (0, 1, 2, 7, 100, 999, 1000):
        assert len(deterministic_schedule({"R": 1 / 3, "X": 1 / 3, "S": 1 / 3}, count)) == count


def test_equal_thirds_divide_exactly_when_they_can():
    schedule = deterministic_schedule({"R": 1 / 3, "X": 1 / 3, "S": 1 / 3}, 300)
    assert Counter(schedule) == {"R": 100, "X": 100, "S": 100}


def test_remainders_are_apportioned_not_dropped():
    """
    100 examples across equal thirds cannot divide evenly; the leftover must
    still be allocated, never silently discarded.
    """
    counts = Counter(deterministic_schedule({"R": 1 / 3, "X": 1 / 3, "S": 1 / 3}, 100))
    assert sum(counts.values()) == 100
    assert set(counts.values()) <= {33, 34}


def test_schedule_matches_uneven_weights():
    counts = Counter(deterministic_schedule(DEFAULT_MODE_WEIGHTS, 1000))
    assert counts == {"R": 400, "X": 350, "S": 250}


def test_schedule_is_a_pure_function():
    a = deterministic_schedule(DEFAULT_MODE_WEIGHTS, 137)
    b = deterministic_schedule(DEFAULT_MODE_WEIGHTS, 137)
    assert a == b


def test_schedule_ties_break_by_canonical_mode_order():
    """Deterministic even when fractional parts are exactly equal."""
    assert list(deterministic_schedule({"R": 1 / 3, "X": 1 / 3, "S": 1 / 3}, 1)) == ["R"]
    assert Counter(deterministic_schedule({"R": 1 / 3, "X": 1 / 3, "S": 1 / 3}, 2)) == {
        "R": 1,
        "X": 1,
    }


def test_schedule_skips_zero_weight_modes():
    counts = Counter(deterministic_schedule({"R": 0.5, "X": 0.0, "S": 0.5}, 100))
    assert counts == {"R": 50, "S": 50}


def test_schedule_rejects_a_negative_count():
    with pytest.raises(ConfigError, match="non-negative"):
        deterministic_schedule(DEFAULT_MODE_WEIGHTS, -1)


def test_schedule_rejects_all_zero_weights():
    with pytest.raises(ConfigError, match="every mode weight is zero"):
        deterministic_schedule({"R": 0.0, "X": 0.0, "S": 0.0}, 10)


def test_schedule_differs_from_sampling():
    """
    The distinction that matters: sampling 300 examples does not give exactly
    100/100/100, which is why frozen evaluation uses apportionment instead.
    """
    weights = {"R": 1 / 3, "X": 1 / 3, "S": 1 / 3}
    scheduled = Counter(deterministic_schedule(weights, 300))
    assert scheduled == {"R": 100, "X": 100, "S": 100}

    sampler = ModeSampler(weights)
    sampled = [Counter(sampler.sample_many(300, random.Random(seed))) for seed in range(5)]
    assert any(counts != scheduled for counts in sampled)
