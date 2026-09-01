"""
Exception hierarchy for the pretraining corpus pipeline.

Typed, never a bare `ValueError`, for the same reason every sibling package
(`model/`, `UL2/`, `training/`) does this: a caller must be able to tell "this
corpus path is unusable, stop before a single step runs" from "this line is
malformed, skip it and keep counting" — and `assert` disappears under
`python -O`, which is not acceptable for a check that exists to catch a
mis-pointed corpus before it silently trains on nothing.
"""

from __future__ import annotations


class CorpusError(Exception):
    """Base class for every error raised by this package."""


class CorpusConfigError(CorpusError):
    """
    The corpus source itself is unusable: no files found, or an empty
    file list was passed where at least one file is required.

    Fatal and raised before a single example is produced, mirroring
    `UL2Config.validate_capacity`'s "prove it before you start" discipline.
    """
