"""
Exception hierarchy for the model.

Typed, never bare `ValueError` or `assert`, for the same reason the UL2
module is: a caller must be able to tell "this configuration is impossible,
stop" from "this batch is malformed, skip it", and `assert` disappears under
`python -O` — a parameter-count check that an optimization flag can silently
disable is not a check.
"""

from __future__ import annotations


class ModelError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(ModelError):
    """
    The model configuration is invalid, internally inconsistent, or absent.

    Fatal at construction time: a model built from a bad configuration would
    train for weeks before anyone noticed it was the wrong shape.
    """


class ParameterCountError(ModelError):
    """
    The built model's parameter count disagrees with the configuration's
    declared `expected_parameters`.

    This means a dimension changed without the expectation being updated.
    Raised loudly because the failure is otherwise invisible: the model
    trains perfectly well at the wrong size, and the discrepancy surfaces
    only when a checkpoint will not load or a deployment target is missed.
    """


class ShapeError(ModelError):
    """A batch's tensors do not have the shapes the model requires."""


class ConversionError(ModelError):
    """
    A checkpoint cannot be translated between this package's parameter names
    and HuggingFace's.

    Raised rather than passing an unrecognized name through untouched. A
    parameter silently exported under the wrong name is dropped by
    `from_pretrained` with a warning, leaving that component at its random
    initialization — a checkpoint that loads, runs, generates plausible text,
    and is not the model that was trained.
    """
