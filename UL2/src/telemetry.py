"""
The per-mode measurements architecture.md 7.3 marks as a requirement.

Why this is not optional
------------------------
Mode sampling probability and mode *training* contribution are different
numbers. [R] produces a short target, [X] a long one, [S] a long one -- so a
40/35/25 split of examples delivers a materially different split of decoder
tokens. If a run only records the configured probabilities, nobody can answer
"how much generation practice did [S] actually provide?", and a per-mode loss
comparison has no denominator.

So this module records what actually happened:

  - realized mode share (examples), against the configured probability
  - encoder-token exposure per mode  -- how much source text each mode showed
  - decoder-token exposure per mode  -- how much generation practice each gave
  - realized corruption rate per mode -- configured rate is pre-rounding
  - target-length distribution per mode, bucketed
  - skipped/truncated counts -- data loss must never be invisible

Everything is a plain counter. No sampling, no windowing: over a multi-week
run these are exact totals, and `report()` is cheap enough to log at every
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .denoise import Example
from .special_tokens import MODES

# Aligned with the input-length evaluation buckets in architecture.md 7.7, so
# target-length telemetry and length-bucketed quality can be read together.
TARGET_LENGTH_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024, 1536, 2048)


@dataclass
class ModeCounters:
    """Totals for a single denoising mode."""

    examples: int = 0
    source_tokens: int = 0
    encoder_tokens: int = 0
    decoder_tokens: int = 0
    corrupted_tokens: int = 0
    spans: int = 0
    truncated: int = 0
    target_length_histogram: dict[str, int] = field(default_factory=dict)

    def record(self, example: Example) -> None:
        self.examples += 1
        self.source_tokens += example.source_length
        self.encoder_tokens += example.encoder_length
        self.decoder_tokens += example.target_length
        self.corrupted_tokens += example.num_corrupted_tokens
        self.spans += example.num_spans
        if example.truncated:
            self.truncated += 1
        bucket = _bucket_label(example.target_length)
        self.target_length_histogram[bucket] = self.target_length_histogram.get(bucket, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "source_tokens": self.source_tokens,
            "encoder_tokens": self.encoder_tokens,
            "decoder_tokens": self.decoder_tokens,
            "corrupted_tokens": self.corrupted_tokens,
            "spans": self.spans,
            "truncated": self.truncated,
            "mean_spans_per_example": _safe_divide(self.spans, self.examples),
            "mean_target_length": _safe_divide(self.decoder_tokens, self.examples),
            "realized_corruption_rate": _safe_divide(self.corrupted_tokens, self.source_tokens),
            "mean_span_length": _safe_divide(self.corrupted_tokens, self.spans),
            "target_length_histogram": dict(sorted(self.target_length_histogram.items())),
        }


class MixtureTelemetry:
    """Accumulates per-mode counters across a run."""

    def __init__(self, configured_weights: Mapping[str, float] | None = None) -> None:
        self._configured = dict(configured_weights or {})
        self._modes: dict[str, ModeCounters] = {}
        self.skipped_too_short = 0
        self.skipped_too_long = 0

    def record(self, example: Example) -> None:
        self._modes.setdefault(example.mode, ModeCounters()).record(example)

    def record_skipped_too_short(self) -> None:
        self.skipped_too_short += 1

    def record_skipped_too_long(self) -> None:
        self.skipped_too_long += 1

    @property
    def total_examples(self) -> int:
        return sum(counters.examples for counters in self._modes.values())

    def counters(self, mode: str) -> ModeCounters:
        return self._modes.setdefault(mode, ModeCounters())

    def report(self) -> dict[str, Any]:
        """
        Full telemetry snapshot.

        The three `*_share` values per mode are the point of this module: the
        gap between `configured_probability` and `decoder_token_share` is
        exactly what architecture.md 7.3 warns must never be assumed away.
        """
        total_examples = self.total_examples
        total_encoder = sum(c.encoder_tokens for c in self._modes.values())
        total_decoder = sum(c.decoder_tokens for c in self._modes.values())

        modes: dict[str, Any] = {}
        for mode in MODES:
            if mode not in self._modes:
                continue
            counters = self._modes[mode]
            entry = counters.as_dict()
            entry["configured_probability"] = self._configured.get(mode)
            entry["realized_example_share"] = _safe_divide(counters.examples, total_examples)
            entry["encoder_token_share"] = _safe_divide(counters.encoder_tokens, total_encoder)
            entry["decoder_token_share"] = _safe_divide(counters.decoder_tokens, total_decoder)
            modes[mode] = entry

        return {
            "total_examples": total_examples,
            "total_encoder_tokens": total_encoder,
            "total_decoder_tokens": total_decoder,
            "skipped_too_short": self.skipped_too_short,
            "skipped_too_long": self.skipped_too_long,
            "modes": modes,
        }

    def format_table(self) -> str:
        """One-screen summary for a training log."""
        report = self.report()
        header = (
            f"{'mode':<5}{'examples':>10}{'p(cfg)':>9}{'p(real)':>9}"
            f"{'enc share':>11}{'dec share':>11}{'corrupt':>9}{'span len':>10}"
        )
        lines = [header, "-" * len(header)]
        for mode, entry in report["modes"].items():
            configured = entry["configured_probability"]
            lines.append(
                f"{mode:<5}{entry['examples']:>10}"
                f"{'-' if configured is None else format(configured, '.3f'):>9}"
                f"{entry['realized_example_share']:>9.3f}"
                f"{entry['encoder_token_share']:>11.3f}"
                f"{entry['decoder_token_share']:>11.3f}"
                f"{entry['realized_corruption_rate']:>9.3f}"
                f"{entry['mean_span_length']:>10.2f}"
            )
        lines.append(
            f"skipped: {report['skipped_too_short']} too short, "
            f"{report['skipped_too_long']} too long"
        )
        return "\n".join(lines)


def _bucket_label(length: int) -> str:
    lower = 0
    for upper in TARGET_LENGTH_BUCKETS:
        if length <= upper:
            return f"{lower + 1:04d}-{upper:04d}"
        lower = upper
    return f"{TARGET_LENGTH_BUCKETS[-1] + 1:04d}+"


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
