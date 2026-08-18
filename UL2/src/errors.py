"""
Exception hierarchy for the UL2 objective.

Every failure mode in this module raises a *specific* type, never a bare
`ValueError` or an `assert`. Two reasons:

1. A multi-week pretraining run must be able to distinguish "this one line
   of corpus is unusable, skip it" (SourceTooShortError) from "the
   configuration is impossible, stop the run" (ConfigError). A caller
   cannot make that distinction against a generic exception.
2. `assert` statements vanish under `python -O`. The sentinel-budget check
   in architecture.md 7.4 is required to "fail loudly at data-generation
   time" -- it must not be something an optimization flag can silently
   disable.
"""

from __future__ import annotations


class UL2Error(Exception):
    """Base class for every error raised by this module."""


class ConfigError(UL2Error):
    """
    The configuration itself is invalid or impossible.

    Fatal and non-recoverable: raised at construction time, before a single
    example is produced, so an impossible setup cannot be discovered three
    hours into a run.
    """


class SpecialTokenError(UL2Error):
    """The special-token table is missing, malformed, or internally inconsistent."""


class SourceTooShortError(UL2Error):
    """
    The input token sequence is below `min_source_length`.

    Recoverable at the caller: skip this line and continue. Corrupting a
    4-token line teaches nothing and produces degenerate spans.
    """


class SourceTooLongError(UL2Error):
    """
    The input token sequence exceeds `max_source_length`.

    Raised only when `on_overlong_source="error"` (the default). architecture.md
    4.5 forbids *silent* truncation; truncation is available but must be
    opted into explicitly, and is counted in telemetry when it happens.
    """


class SentinelBudgetExceededError(UL2Error):
    """
    An example needs more spans than there are sentinel tokens.

    architecture.md 7.4: "Hard assert that no single example requires more
    than 256 spans. Fail loudly at data-generation time; never wrap around
    or silently drop a span." Wrapping would reuse `<extra_id_0>` twice in
    one example, making the target ambiguous and quietly poisoning training.
    """


class TargetTooLongError(UL2Error):
    """
    The decoder target exceeds `max_target_length`.

    This is the failure architecture.md 4.4 exists to prevent: a target that
    overflows the decoder gets truncated, training the model to produce
    *incomplete* reconstructions -- the opposite of the objective's purpose.
    Better to stop than to train on it.
    """


class FrozenEvalError(UL2Error):
    """A frozen evaluation set failed to build, load, or verify its digest."""
