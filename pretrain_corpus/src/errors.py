"""
Exception hierarchy for the pretraining corpus downloader.

Typed, never a bare `ValueError`, for the same reason every sibling package
(`corpus/`, `model/`, `UL2/`, `training/`) does this: a multi-day download
must be able to tell "this source is misconfigured, stop before burning a
week of bandwidth" from "this one record is malformed, skip it and keep
counting". `assert` is not used for either, because it disappears under
`python -O`.
"""

from __future__ import annotations


class PretrainCorpusError(Exception):
    """Base class for every error raised by this package."""


class SourceConfigError(PretrainCorpusError):
    """
    A source is unusable as configured: unknown source id, a blend whose
    shares do not sum to 1.0, a dataset that needs credentials we do not
    have, a missing external tool.

    Fatal and raised before any bytes are written, so a typo in `blend.yml`
    costs a second rather than a day.
    """


class PIIRegistryError(PretrainCorpusError):
    """
    The PII registry could not be read, reserved from, or written back.

    Fatal by design: the registry is the *only* thing standing between us
    and re-issuing a synthetic identity that was already used elsewhere in
    the corpus. Continuing without it would silently break the uniqueness
    guarantee this package exists to provide, so a broken registry stops
    the run instead.
    """


class ExtractionError(PretrainCorpusError):
    """
    A downloaded archive could not be unpacked or parsed.

    Raised per-archive and caught by the runner, which records the failure
    in the manifest and continues with other archives — one bad Stack
    Exchange site must not abort a 98GB download.
    """
