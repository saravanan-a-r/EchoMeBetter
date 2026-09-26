"""
Exception hierarchy for the rephrase-adapter SFT module.

Typed, never bare `ValueError` or `assert`, for the reason every other module
in this repo gives: a caller has to be able to tell "this configuration is
impossible, stop" from "this record is malformed, skip it", and `assert`
disappears under `python -O`.
"""

from __future__ import annotations


class SFTError(Exception):
    """Base class for every error raised by this package."""


class SFTConfigError(SFTError):
    """The adapter/training configuration is invalid or internally inconsistent."""


class AdapterError(SFTError):
    """An adapter cannot be attached, saved or loaded as asked."""


class BaseModelMismatchError(AdapterError):
    """
    An adapter is being loaded onto a base model it was not trained on.

    Fatal on purpose: a LoRA delta learned against one set of base weights,
    applied to another, produces fluent-looking text from a model nobody
    trained -- the worst kind of silent failure for a product checkpoint.
    """


class DatasetError(SFTError):
    """The SFT dataset cannot be read or yields no usable examples."""


class ResourceError(SFTError):
    """
    The requested device plan would disturb something else on this machine
    (the pretraining run, above all) or cannot be satisfied.
    """


class ResumeError(SFTError):
    """A run cannot be resumed from the given directory."""
