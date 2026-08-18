"""
Configuration for the UL2 mixture-of-denoisers, with up-front capacity proof.

The defaults in this file are the frozen decisions from architecture.md
sections 4.4, 6.3, 7.2 and 7.3. They are not suggestions to be tuned casually
-- architecture.md 7.3 is explicit that the precise mixture ratios matter far
less than having complementary denoisers at all, and that effort belongs in
data quality instead.

Why this module computes worst cases
------------------------------------
A sentinel overflow or a decoder-target overflow is not a crash you want to
discover on a long input, hours into a multi-week run, after the loss curve
has already been polluted. Both failures are fully predictable from the
configuration alone, so `validate_capacity()` proves the settings are safe
for *every* admissible input length before the first example is produced.

It does that by scanning every length rather than assuming the worst case
occurs at the maximum. The scan is a few thousand cheap integer operations,
run once at construction -- there is no reason to trade that for an
assumption about monotonicity that a future edit could quietly break.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .errors import ConfigError
from .spans import MIN_CORRUPTIBLE_LENGTH, plan_span_counts
from .special_tokens import MODES

# architecture.md 4.4: encoder 2048 / decoder 2048, equal. FROZEN.
DEFAULT_MAX_SOURCE_LENGTH = 2048
DEFAULT_MAX_TARGET_LENGTH = 2048

# architecture.md 7.3: mode *sampling probabilities*, not token shares.
DEFAULT_MODE_WEIGHTS: Mapping[str, float] = {"R": 0.40, "X": 0.35, "S": 0.25}

_OVERLONG_POLICIES = ("error", "truncate")


@dataclass(frozen=True)
class SpanCorruptionConfig:
    """Settings for a sentinel-based denoiser ([R] or [X])."""

    corruption_rate: float
    mean_span_length: float

    def __post_init__(self) -> None:
        if not 0.0 < self.corruption_rate < 1.0:
            raise ConfigError(
                f"corruption_rate must be strictly between 0 and 1, got {self.corruption_rate}; "
                f"0 corrupts nothing and 1 leaves the encoder no context"
            )
        if self.mean_span_length < 1.0:
            raise ConfigError(
                f"mean_span_length must be >= 1, got {self.mean_span_length}"
            )


@dataclass(frozen=True)
class PrefixConfig:
    """Settings for [S] sequential denoising (prefix -> continuation)."""

    min_prefix_fraction: float
    max_prefix_fraction: float

    def __post_init__(self) -> None:
        for name, value in (
            ("min_prefix_fraction", self.min_prefix_fraction),
            ("max_prefix_fraction", self.max_prefix_fraction),
        ):
            if not 0.0 < value < 1.0:
                raise ConfigError(f"{name} must be strictly between 0 and 1, got {value}")
        if self.min_prefix_fraction > self.max_prefix_fraction:
            raise ConfigError(
                f"min_prefix_fraction ({self.min_prefix_fraction}) exceeds "
                f"max_prefix_fraction ({self.max_prefix_fraction})"
            )


@dataclass(frozen=True)
class UL2Config:
    """
    The complete objective configuration.

    Frozen, and copied with `replace()` rather than mutated, so a config
    recorded in a run manifest cannot drift from the one actually used.
    """

    # architecture.md 7.2: [R] short spans, ~15% of tokens, mean span 3.
    r: SpanCorruptionConfig = field(
        default_factory=lambda: SpanCorruptionConfig(corruption_rate=0.15, mean_span_length=3.0)
    )
    # architecture.md 7.2: [X] long spans, ~50% of tokens. Mean span 32 is
    # UL2's long-span setting; at a 2048 source this yields ~1,057 target
    # tokens, matching the arithmetic in architecture.md 4.4 that fixed the
    # decoder context at 2048.
    x: SpanCorruptionConfig = field(
        default_factory=lambda: SpanCorruptionConfig(corruption_rate=0.50, mean_span_length=32.0)
    )
    # architecture.md 7.2: [S] gives a prefix and generates the continuation.
    # Centred on half the sequence, varied so the model never learns a fixed
    # split point.
    s: PrefixConfig = field(
        default_factory=lambda: PrefixConfig(min_prefix_fraction=0.25, max_prefix_fraction=0.75)
    )

    mode_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MODE_WEIGHTS)
    )

    max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH
    max_target_length: int = DEFAULT_MAX_TARGET_LENGTH

    # Below this, corruption is degenerate: a 4-token line masked at 15%
    # teaches nothing and produces a 1-token span that is mostly noise.
    min_source_length: int = 8

    # T5 terminates both encoder input and decoder target with EOS. The
    # worked examples in architecture.md 7.2 omit the encoder EOS for
    # readability; it is standard and kept here.
    append_eos_to_source: bool = True
    append_eos_to_target: bool = True

    # architecture.md 4.5 forbids *silent* truncation. "error" is the default;
    # "truncate" is an explicit opt-in and every truncation is counted in
    # telemetry so it can never happen unnoticed.
    on_overlong_source: str = "error"

    def __post_init__(self) -> None:
        if self.max_source_length < MIN_CORRUPTIBLE_LENGTH:
            raise ConfigError(
                f"max_source_length must be >= {MIN_CORRUPTIBLE_LENGTH}, "
                f"got {self.max_source_length}"
            )
        if self.max_target_length < 1:
            raise ConfigError(f"max_target_length must be >= 1, got {self.max_target_length}")
        if self.min_source_length < MIN_CORRUPTIBLE_LENGTH:
            raise ConfigError(
                f"min_source_length must be >= {MIN_CORRUPTIBLE_LENGTH}, "
                f"got {self.min_source_length}"
            )
        if self.min_source_length > self.max_source_length:
            raise ConfigError(
                f"min_source_length ({self.min_source_length}) exceeds "
                f"max_source_length ({self.max_source_length})"
            )
        if self.on_overlong_source not in _OVERLONG_POLICIES:
            raise ConfigError(
                f"on_overlong_source must be one of {_OVERLONG_POLICIES}, "
                f"got {self.on_overlong_source!r}"
            )

        _validate_mode_weights(self.mode_weights)

    # -- mode lookup ------------------------------------------------------

    def span_config(self, mode: str) -> SpanCorruptionConfig:
        """The span settings for a sentinel-based mode."""
        if mode == "R":
            return self.r
        if mode == "X":
            return self.x
        raise ConfigError(f"{mode!r} is not a span-corruption mode; expected 'R' or 'X'")

    @property
    def active_modes(self) -> tuple[str, ...]:
        """Modes with non-zero sampling probability, in canonical order."""
        return tuple(m for m in MODES if self.mode_weights.get(m, 0.0) > 0.0)

    # -- capacity proof ---------------------------------------------------

    def worst_case_spans(self, mode: str) -> tuple[int, int]:
        """
        Most spans any admissible input can require in `mode`.

        Returns `(num_spans, at_length)` so an error message can name the
        input length that triggers it. [S] uses no sentinels, so it reports 0.
        """
        if mode == "S":
            return 0, 0

        span_config = self.span_config(mode)
        worst = (0, 0)
        for length in range(self.min_source_length, self.max_source_length + 1):
            _, num_spans = plan_span_counts(
                length, span_config.corruption_rate, span_config.mean_span_length
            )
            if num_spans > worst[0]:
                worst = (num_spans, length)
        return worst

    def worst_case_target_length(self, mode: str) -> tuple[int, int]:
        """Longest decoder target any admissible input can produce in `mode`."""
        eos = 1 if self.append_eos_to_target else 0
        worst = (0, 0)

        if mode == "S":
            for length in range(self.min_source_length, self.max_source_length + 1):
                # Shortest admissible prefix leaves the longest continuation.
                prefix = int(round(length * self.s.min_prefix_fraction))
                prefix = max(1, min(prefix, length - 1))
                target = (length - prefix) + eos
                if target > worst[0]:
                    worst = (target, length)
            return worst

        span_config = self.span_config(mode)
        for length in range(self.min_source_length, self.max_source_length + 1):
            num_corrupted, num_spans = plan_span_counts(
                length, span_config.corruption_rate, span_config.mean_span_length
            )
            # target = one sentinel per span + every corrupted token + EOS
            target = num_spans + num_corrupted + eos
            if target > worst[0]:
                worst = (target, length)
        return worst

    def worst_case_encoder_length(self, mode: str) -> tuple[int, int]:
        """
        Longest encoder input any admissible input can produce in `mode`.

        Span corruption usually *shrinks* the sequence (a span of length k
        collapses to one sentinel), but with `mean_span_length` near 1 it
        does not shrink at all -- so this is computed, not assumed.
        """
        prefix_tokens = 1 + (1 if self.append_eos_to_source else 0)  # mode token + EOS
        worst = (0, 0)

        if mode == "S":
            for length in range(self.min_source_length, self.max_source_length + 1):
                prefix = int(round(length * self.s.max_prefix_fraction))
                prefix = max(1, min(prefix, length - 1))
                encoder = prefix + prefix_tokens
                if encoder > worst[0]:
                    worst = (encoder, length)
            return worst

        span_config = self.span_config(mode)
        for length in range(self.min_source_length, self.max_source_length + 1):
            num_corrupted, num_spans = plan_span_counts(
                length, span_config.corruption_rate, span_config.mean_span_length
            )
            encoder = (length - num_corrupted + num_spans) + prefix_tokens
            if encoder > worst[0]:
                worst = (encoder, length)
        return worst

    def validate_capacity(self, num_sentinels: int) -> None:
        """
        Prove this configuration can never overflow, for any admissible input.

        Called once at `UL2Objective` construction. Raises `ConfigError` with
        the offending input length rather than letting the run start and fail
        later on a long line.
        """
        if num_sentinels < 1:
            raise ConfigError(f"num_sentinels must be positive, got {num_sentinels}")

        for mode in self.active_modes:
            spans_needed, at_length = self.worst_case_spans(mode)
            if spans_needed > num_sentinels:
                raise ConfigError(
                    f"[{mode}] needs up to {spans_needed} sentinels (at a "
                    f"{at_length}-token input) but only {num_sentinels} exist. "
                    f"Fix the sentinel budget or the corruption settings -- never "
                    f"let the pipeline wrap or drop spans (architecture.md 6.3, 7.4)."
                )

            target_length, at_length = self.worst_case_target_length(mode)
            if target_length > self.max_target_length:
                raise ConfigError(
                    f"[{mode}] can produce a {target_length}-token decoder target "
                    f"(at a {at_length}-token input), exceeding max_target_length="
                    f"{self.max_target_length}. Truncating it would train the model "
                    f"to produce incomplete reconstructions (architecture.md 4.4)."
                )

            encoder_length, at_length = self.worst_case_encoder_length(mode)
            if encoder_length > self.max_source_length:
                raise ConfigError(
                    f"[{mode}] can produce a {encoder_length}-token encoder input "
                    f"(at a {at_length}-token input), exceeding max_source_length="
                    f"{self.max_source_length}. Lower max_source_length to leave room "
                    f"for the mode token and EOS."
                )

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form for run manifests and frozen-eval provenance."""
        return {
            "r": {
                "corruption_rate": self.r.corruption_rate,
                "mean_span_length": self.r.mean_span_length,
            },
            "x": {
                "corruption_rate": self.x.corruption_rate,
                "mean_span_length": self.x.mean_span_length,
            },
            "s": {
                "min_prefix_fraction": self.s.min_prefix_fraction,
                "max_prefix_fraction": self.s.max_prefix_fraction,
            },
            "mode_weights": dict(self.mode_weights),
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "min_source_length": self.min_source_length,
            "append_eos_to_source": self.append_eos_to_source,
            "append_eos_to_target": self.append_eos_to_target,
            "on_overlong_source": self.on_overlong_source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UL2Config":
        """Inverse of `to_dict`. Unknown keys are rejected, not ignored."""
        known = set(cls().to_dict())
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")

        defaults = cls()
        return cls(
            r=SpanCorruptionConfig(**data["r"]) if "r" in data else defaults.r,
            x=SpanCorruptionConfig(**data["x"]) if "x" in data else defaults.x,
            s=PrefixConfig(**data["s"]) if "s" in data else defaults.s,
            mode_weights=dict(data.get("mode_weights", defaults.mode_weights)),
            max_source_length=data.get("max_source_length", defaults.max_source_length),
            max_target_length=data.get("max_target_length", defaults.max_target_length),
            min_source_length=data.get("min_source_length", defaults.min_source_length),
            append_eos_to_source=data.get("append_eos_to_source", defaults.append_eos_to_source),
            append_eos_to_target=data.get("append_eos_to_target", defaults.append_eos_to_target),
            on_overlong_source=data.get("on_overlong_source", defaults.on_overlong_source),
        )

    def with_(self, **changes: Any) -> "UL2Config":
        """Copy with overrides; re-runs all validation."""
        return replace(self, **changes)


def _validate_mode_weights(weights: Mapping[str, float]) -> None:
    if not weights:
        raise ConfigError("mode_weights is empty; there is nothing to sample")

    unknown = set(weights) - set(MODES)
    if unknown:
        raise ConfigError(f"unknown modes in mode_weights: {sorted(unknown)}; expected {MODES}")

    for mode, weight in weights.items():
        if weight < 0:
            raise ConfigError(f"mode weight for {mode!r} is negative: {weight}")

    total = sum(weights.values())
    if total <= 0:
        raise ConfigError("mode weights sum to zero; no mode could ever be sampled")
    if abs(total - 1.0) > 1e-9:
        raise ConfigError(
            f"mode weights must sum to 1.0, got {total}. They are sampling "
            f"probabilities (architecture.md 7.3) -- an implicit renormalization "
            f"would hide a typo in a run manifest."
        )


# The fixed validation mixture required by architecture.md 7.6: it must NOT
# follow the training mixture, or an X-heavy run gets scored on an X-heavy
# validation set and the comparison is circular.
FROZEN_EVAL_MODE_WEIGHTS: Mapping[str, float] = {
    "R": 1.0 / 3.0,
    "X": 1.0 / 3.0,
    "S": 1.0 - 2.0 * (1.0 / 3.0),
}
