"""Errors this package raises. One base class, so a caller can catch the lot."""

from __future__ import annotations


class OwnershipError(Exception):
    """Base for every error raised by `ownership/`."""


class OwnershipConfigError(OwnershipError):
    """`ownership_config.yml` is malformed, or names a technique that does not exist."""


__all__ = ["OwnershipConfigError", "OwnershipError"]
