"""
Exception hierarchy for the training loop.

Typed, never bare `ValueError` or `assert`, for the same reason `model/` and
`UL2/` are: a caller must be able to tell "this configuration is impossible,
stop before allocating anything" from "this checkpoint is unreadable, fall
back to an earlier one", and `assert` disappears under `python -O` — a
resumption check an optimization flag can disable is not a check.
"""

from __future__ import annotations


class TrainingError(Exception):
    """Base class for every error raised by this package."""


class TrainingConfigError(TrainingError):
    """
    The training configuration is invalid or internally inconsistent.

    Fatal before the first step. A run started with a bad schedule or a
    mismatched accumulation setting produces numbers that look like training
    and are not, and the cost is measured in GPU-weeks.
    """


class CheckpointError(TrainingError):
    """
    A checkpoint cannot be written, found, or read back.

    Raised loudly on resumption problems in particular. Silently restarting a
    multi-week run from step 0 — or from the right step but the wrong data
    position — is the failure architecture.md §7.9 exists to prevent: the
    model re-reads tokens it has already seen and the token budget is
    corrupted invisibly.
    """


class ScheduleError(TrainingError):
    """The learning-rate schedule is unknown or its parameters are unusable."""
