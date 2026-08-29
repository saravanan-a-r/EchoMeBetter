"""
Learning-rate schedules (architecture.md §13).

A schedule here is a **pure function of the step number** returning a
multiplier on the peak learning rate, not a stateful object wrapped around an
optimizer. Two reasons, both about resumption:

  - A run that stops at step 40,000 and restarts must produce exactly the
    learning rate step 40,001 would have had. With a pure function that is
    automatic; with a counter living inside a scheduler object it depends on
    the counter having been checkpointed and restored correctly, which is a
    thing that can be got wrong and is invisible when it is.
  - It can be tested by evaluating it, at any step, in any order, without
    building a model or an optimizer.

Agreement with HuggingFace
--------------------------
The names are `transformers.SchedulerType` values and the arithmetic is
`transformers.optimization`'s, line for line. This is not cosmetic: someone
who continues training a released checkpoint with `Trainer` and
`lr_scheduler_type="inverse_sqrt"` gets the same curve this run used, so the
learning rate does not silently jump at the hand-over. `test_schedule.py`
checks every step of every schedule against `transformers` itself rather than
against a transcription of it.

Which to use
------------
§13 specifies **inverse_sqrt for pretraining** and **cosine for SFT**. The
difference matters: inverse-sqrt never needs to know how long the run is, so a
50–100B-token pretraining run can be extended without the schedule being wrong
in hindsight. Cosine must know `max_steps` up front, and a cosine schedule
whose horizon is later moved decays to a floor at the old horizon and stays
there.
"""

from __future__ import annotations

import math

from .errors import ScheduleError

# `transformers.SchedulerType` values, restricted to the ones implemented.
# Naming them identically is what makes a HuggingFace-side continuation of a
# run produce the same curve rather than a similar one.
CONSTANT = "constant"
CONSTANT_WITH_WARMUP = "constant_with_warmup"
LINEAR = "linear"
COSINE = "cosine"
INVERSE_SQRT = "inverse_sqrt"

SCHEDULES = (CONSTANT, CONSTANT_WITH_WARMUP, LINEAR, COSINE, INVERSE_SQRT)

# Schedules that decay towards zero at `max_steps` and are therefore
# meaningless without one. Recorded here so the configuration can refuse a
# cosine run with no horizon instead of quietly treating `max_steps` as 1.
NEEDS_TOTAL_STEPS = frozenset({LINEAR, COSINE})


def warmup_multiplier(step: int, warmup_steps: int) -> float:
    """
    The linear ramp every warmed-up schedule shares.

    Note it reaches 1.0 *at* `warmup_steps`, not one step before: at
    `step == warmup_steps` the caller has already left the warmup branch.
    Step 0 therefore has a learning rate of exactly zero, which is deliberate
    — the first gradient of a randomly initialized model is the least
    trustworthy one in the run.
    """
    return float(step) / float(max(1, warmup_steps))


def learning_rate_multiplier(
    step: int,
    *,
    schedule: str,
    warmup_steps: int,
    max_steps: int | None = None,
    timescale: int | None = None,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> float:
    """
    The factor to apply to the peak learning rate at `step`.

    `step` counts **optimizer steps**, not micro-batches: a run with
    `gradient_accumulation_steps=32` advances this once per 32 forward passes,
    which is what makes a schedule length comparable between runs that used
    different accumulation to reach the same effective batch size.

    `timescale` applies to `inverse_sqrt` only and defaults to
    `warmup_steps` (HuggingFace's own default, falling back to 10,000 when
    there is no warmup). `num_cycles` and `min_lr_ratio` apply to `cosine`.
    """
    if step < 0:
        raise ScheduleError(f"step must be non-negative, got {step}")
    if warmup_steps < 0:
        raise ScheduleError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if schedule not in SCHEDULES:
        raise ScheduleError(
            f"unknown schedule {schedule!r}; implemented: {list(SCHEDULES)}. "
            f"The names are transformers.SchedulerType values."
        )
    if schedule in NEEDS_TOTAL_STEPS and not max_steps:
        raise ScheduleError(
            f"the {schedule!r} schedule decays to zero at max_steps and cannot be "
            f"evaluated without one. Use {INVERSE_SQRT!r} for a run whose length is "
            f"not fixed in advance (architecture.md §13 specifies it for pretraining)."
        )

    if schedule == CONSTANT:
        return 1.0

    if step < warmup_steps:
        return warmup_multiplier(step, warmup_steps)

    if schedule == CONSTANT_WITH_WARMUP:
        return 1.0

    if schedule == LINEAR:
        assert max_steps is not None
        return max(
            0.0,
            float(max_steps - step) / float(max(1, max_steps - warmup_steps)),
        )

    if schedule == COSINE:
        assert max_steps is not None
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, factor * (1.0 - min_lr_ratio) + min_lr_ratio)

    # INVERSE_SQRT. `timescale` sets where the 1/sqrt decay is anchored; the
    # shift makes the curve continuous at the end of warmup rather than
    # dropping discontinuously the moment the ramp finishes.
    scale = timescale if timescale is not None else (warmup_steps or 10_000)
    if scale <= 0:
        raise ScheduleError(f"timescale must be positive, got {scale}")
    return 1.0 / math.sqrt((step + scale - warmup_steps) / scale)


def learning_rate_at(step: int, peak_learning_rate: float, **kwargs: object) -> float:
    """The learning rate itself, for logging and for setting on an optimizer."""
    return peak_learning_rate * learning_rate_multiplier(step, **kwargs)  # type: ignore[arg-type]


def resolve_warmup_steps(
    warmup_steps: int, warmup_ratio: float, max_steps: int | None
) -> int:
    """
    Reconcile HuggingFace's two ways of specifying warmup.

    `TrainingArguments` accepts both and lets `warmup_steps` win when it is
    non-zero; the same precedence is used here so a configuration copied from
    a HuggingFace run behaves the same way. A ratio with no `max_steps` to
    take a fraction of is an error rather than a silent zero — "no warmup"
    from a typo is a diverged first hour.
    """
    if warmup_steps:
        return warmup_steps
    if not warmup_ratio:
        return 0
    if not max_steps:
        raise ScheduleError(
            "warmup_ratio needs max_steps to be a fraction of; set warmup_steps "
            "directly for a run of unbounded length"
        )
    return int(max_steps * warmup_ratio)
