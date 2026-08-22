"""
Unit tests for configuration and the up-front capacity proof.

`validate_capacity` exists so that an impossible setup fails in milliseconds
instead of hours into a run. These tests check both directions: that the real
configuration passes, and that each specific way of breaking it is actually
caught rather than merely hoped about.
"""

from __future__ import annotations

import pytest

from src.config import (
    DEFAULT_MODE_WEIGHTS,
    FROZEN_EVAL_MODE_WEIGHTS,
    PrefixConfig,
    SpanCorruptionConfig,
    UL2Config,
)
from src.errors import ConfigError


# -- component validation -------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_corruption_rate_must_be_a_proper_fraction(rate):
    with pytest.raises(ConfigError, match="corruption_rate"):
        SpanCorruptionConfig(corruption_rate=rate, mean_span_length=3.0)


@pytest.mark.parametrize("mean", [0.0, 0.5, -1.0])
def test_mean_span_length_must_be_at_least_one(mean):
    with pytest.raises(ConfigError, match="mean_span_length"):
        SpanCorruptionConfig(corruption_rate=0.15, mean_span_length=mean)


@pytest.mark.parametrize("low,high", [(0.0, 0.5), (0.5, 1.0), (-0.1, 0.5)])
def test_prefix_fractions_must_be_proper_fractions(low, high):
    with pytest.raises(ConfigError, match="fraction"):
        PrefixConfig(min_prefix_fraction=low, max_prefix_fraction=high)


def test_prefix_fractions_must_be_ordered():
    with pytest.raises(ConfigError, match="exceeds"):
        PrefixConfig(min_prefix_fraction=0.8, max_prefix_fraction=0.2)


# -- top-level validation -------------------------------------------------


def test_defaults_are_valid():
    UL2Config()  # must not raise


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"max_source_length": 1}, "max_source_length"),
        ({"max_target_length": 0}, "max_target_length"),
        ({"min_source_length": 1}, "min_source_length"),
        ({"min_source_length": 4096}, "exceeds"),
        ({"on_overlong_source": "ignore"}, "on_overlong_source"),
    ],
)
def test_invalid_lengths_and_policies_rejected(changes, match):
    with pytest.raises(ConfigError, match=match):
        UL2Config(**changes)


@pytest.mark.parametrize(
    "weights,match",
    [
        ({}, "empty"),
        ({"R": 0.5, "Q": 0.5}, "unknown modes"),
        ({"R": -0.5, "X": 1.5}, "negative"),
        ({"R": 0.5, "X": 0.2, "S": 0.2}, "sum to 1"),
        ({"R": 0.0, "X": 0.0, "S": 0.0}, "sum to zero"),
    ],
)
def test_invalid_mode_weights_rejected(weights, match):
    with pytest.raises(ConfigError, match=match):
        UL2Config(mode_weights=weights)


def test_weights_are_not_silently_renormalized():
    """
    A typo in a run manifest must surface, not be quietly fixed. Implicit
    renormalization would make 40/35/20 look like a valid run.
    """
    with pytest.raises(ConfigError, match="sum to 1"):
        UL2Config(mode_weights={"R": 0.40, "X": 0.35, "S": 0.20})


def test_a_two_mode_mixture_is_allowed_if_it_sums_to_one():
    config = UL2Config(mode_weights={"R": 0.5, "S": 0.5})
    assert config.active_modes == ("R", "S")


def test_zero_weight_mode_is_inactive():
    config = UL2Config(mode_weights={"R": 0.5, "X": 0.0, "S": 0.5})
    assert config.active_modes == ("R", "S")


def test_span_config_lookup():
    config = UL2Config()
    assert config.span_config("R") is config.r
    assert config.span_config("X") is config.x
    with pytest.raises(ConfigError, match="not a span-corruption mode"):
        config.span_config("S")


# -- capacity proof -------------------------------------------------------


def test_default_configuration_fits_the_real_sentinel_budget():
    UL2Config().validate_capacity(256)  # must not raise


def test_worst_case_spans_matches_the_documented_arithmetic():
    """architecture.md 6.3: R-denoising at 2048 needs ~103 sentinels."""
    spans, at_length = UL2Config().worst_case_spans("R")
    assert spans == 102
    assert at_length <= 2048


def test_worst_case_spans_for_prefix_mode_is_zero():
    assert UL2Config().worst_case_spans("S") == (0, 0)


def test_worst_case_target_lengths_fit_the_decoder():
    config = UL2Config()
    for mode in ("R", "X", "S"):
        target, _ = config.worst_case_target_length(mode)
        assert target <= config.max_target_length


def test_worst_case_encoder_lengths_fit_the_encoder():
    config = UL2Config()
    for mode in ("R", "X", "S"):
        encoder, _ = config.worst_case_encoder_length(mode)
        assert encoder <= config.max_source_length


def test_sentinel_shortfall_is_caught_before_the_run_starts():
    with pytest.raises(ConfigError, match="sentinels"):
        UL2Config().validate_capacity(100)


def test_sentinel_shortfall_error_names_the_triggering_length():
    with pytest.raises(ConfigError) as excinfo:
        UL2Config().validate_capacity(50)
    assert "token input" in str(excinfo.value)


def test_target_overflow_is_caught():
    """
    The architecture.md 4.4 failure: a 1024-token decoder cannot hold
    X-denoising's ~1057-token target, and training on a truncated target
    teaches incomplete reconstruction.
    """
    with pytest.raises(ConfigError, match="decoder target"):
        UL2Config(max_target_length=1024).validate_capacity(256)


def test_encoder_overflow_from_unit_span_length_is_caught():
    """
    With mean_span_length 1, every span is a single token, so corruption
    does not shrink the sequence at all and the mode token plus EOS push the
    encoder input past its budget. This is why the worst case is computed
    rather than assumed.
    """
    config = UL2Config(
        r=SpanCorruptionConfig(corruption_rate=0.15, mean_span_length=1.0),
        mode_weights={"R": 1.0},
    )
    with pytest.raises(ConfigError, match="encoder input"):
        config.validate_capacity(2048)


def test_capacity_check_only_covers_active_modes():
    """An X configuration that could not fit is irrelevant if X is never drawn."""
    config = UL2Config(
        x=SpanCorruptionConfig(corruption_rate=0.9, mean_span_length=1.0),
        mode_weights={"R": 0.5, "S": 0.5},
    )
    config.validate_capacity(256)  # must not raise


def test_capacity_check_rejects_a_zero_sentinel_budget():
    with pytest.raises(ConfigError, match="must be positive"):
        UL2Config().validate_capacity(0)


def test_a_larger_context_needs_more_sentinels():
    """
    architecture.md 6.3's forward-looking claim: 256 also covers a 4096
    context. Verified rather than trusted.
    """
    config = UL2Config(max_source_length=4096, max_target_length=4096)
    spans, _ = config.worst_case_spans("R")
    assert 100 < spans <= 256
    config.validate_capacity(256)


# -- serialization --------------------------------------------------------


def test_to_dict_from_dict_round_trip():
    original = UL2Config()
    assert UL2Config.from_dict(original.to_dict()) == original


def test_round_trip_preserves_non_default_settings():
    original = UL2Config(
        r=SpanCorruptionConfig(corruption_rate=0.2, mean_span_length=4.0),
        s=PrefixConfig(min_prefix_fraction=0.3, max_prefix_fraction=0.6),
        mode_weights={"R": 0.6, "X": 0.4},
        max_source_length=512,
        max_target_length=512,
        min_source_length=16,
        append_eos_to_source=False,
        on_overlong_source="truncate",
    )
    assert UL2Config.from_dict(original.to_dict()) == original


def test_from_dict_rejects_unknown_keys():
    """
    A stale or misspelled key in a run manifest must fail rather than be
    ignored -- silently dropping `corruption_rate` would change the objective.
    """
    payload = UL2Config().to_dict()
    payload["curruption_rate"] = 0.3
    with pytest.raises(ConfigError, match="unknown configuration keys"):
        UL2Config.from_dict(payload)


def test_from_dict_fills_missing_keys_with_defaults():
    config = UL2Config.from_dict({"max_source_length": 512, "max_target_length": 512})
    assert config.r == UL2Config().r
    assert config.max_source_length == 512


def test_from_dict_revalidates():
    payload = UL2Config().to_dict()
    payload["mode_weights"] = {"R": 0.9, "X": 0.9, "S": 0.9}
    with pytest.raises(ConfigError, match="sum to 1"):
        UL2Config.from_dict(payload)


def test_with_creates_a_validated_copy():
    config = UL2Config()
    narrowed = config.with_(max_source_length=512)

    assert narrowed.max_source_length == 512
    assert config.max_source_length == 2048, "the original must not be mutated"

    with pytest.raises(ConfigError):
        config.with_(max_target_length=0)


def test_config_is_immutable():
    config = UL2Config()
    with pytest.raises(Exception):
        config.max_source_length = 4096  # type: ignore[misc]


# -- the frozen evaluation mixture ---------------------------------------


def test_frozen_eval_mixture_is_equal_thirds_and_sums_to_one():
    """
    architecture.md 7.6: the validation mixture must be fixed and must not
    follow the training mixture, or the comparison is circular.
    """
    assert sum(FROZEN_EVAL_MODE_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(FROZEN_EVAL_MODE_WEIGHTS) == {"R", "X", "S"}
    for weight in FROZEN_EVAL_MODE_WEIGHTS.values():
        assert weight == pytest.approx(1 / 3)


def test_frozen_eval_mixture_differs_from_the_training_mixture():
    assert dict(FROZEN_EVAL_MODE_WEIGHTS) != dict(DEFAULT_MODE_WEIGHTS)


def test_frozen_eval_mixture_is_a_valid_configuration():
    UL2Config(mode_weights=dict(FROZEN_EVAL_MODE_WEIGHTS))
