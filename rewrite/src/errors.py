"""
Exception hierarchy for the synthetic rewrite task.

Typed rather than bare `ValueError`, for the same reason every sibling package
(`UL2/`, `corpus/`, `training/`) does it: a caller must be able to tell "this
configuration cannot start a run" from "this document is unusable, skip it and
keep counting", and `assert` disappears under `python -O`.
"""

from __future__ import annotations


class RewriteError(Exception):
    """Base class for every error raised by this package."""


class RewriteConfigError(RewriteError):
    """A setting that cannot produce examples: refused before the run starts."""


class RewriteTokenError(RewriteError):
    """A token this task needs is missing from, or malformed in, the token map."""
