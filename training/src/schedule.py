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

`warmup_stable_decay` (architecture_improvements.md item 2) is the third
option and the one `training_config.yml` now uses for pretraining: warm up,
hold the peak flat, then decay to (near) zero over the final stretch. It gives
up inverse-sqrt's extend-at-any-time property in exchange for the late-run
loss drop a decay phase reliably recovers — the mechanism behind MiniCPM's,
OLMo 2's and DeepSeek's reported annealing gains, and Llama 3's "annealing"
phase. Note the trade honestly: the horizon is now load-bearing, because
moving it after the fact means the decay happened at the wrong place.

The stable phase is **constant**, not inverse-sqrt. That is what
`warmup_stable_decay` means in `transformers` (and in the WSD literature the
name comes from), and matching it is the whole reason for using HuggingFace's
name — a run continued under `Trainer` with `lr_scheduler_type=
"warmup_stable_decay"` follows the same curve rather than a similar one.
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
WARMUP_STABLE_DECAY = "warmup_stable_decay"

SCHEDULES = (
    CONSTANT,
    CONSTANT_WITH_WARMUP,
    LINEAR,
    COSINE,
    INVERSE_SQRT,
    WARMUP_STABLE_DECAY,
)

# Schedules that decay towards zero at `max_steps` and are therefore
# meaningless without one. Recorded here so the configuration can refuse a
# cosine run with no horizon instead of quietly treating `max_steps` as 1.
NEEDS_TOTAL_STEPS = frozenset({LINEAR, COSINE, WARMUP_STABLE_DECAY})

# `warmup_stable_decay`'s decay shapes, and `transformers`' own names for
# them (`get_wsd_schedule(decay_type=...)`). The stable phase is flat in every
# case; only how it comes down differs.
DECAY_TYPES = ("linear", "cosine", "1-sqrt")


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
    decay_steps: int | None = None,
    decay_type: str = "cosine",
) -> float:
    """
    The factor to apply to the peak learning rate at `step`.

    `step` counts **optimizer steps**, not micro-batches: a run with
    `gradient_accumulation_steps=32` advances this once per 32 forward passes,
    which is what makes a schedule length comparable between runs that used
    different accumulation to reach the same effective batch size.

    `timescale` applies to `inverse_sqrt` only and defaults to
    `warmup_steps` (HuggingFace's own default, falling back to 10,000 when
    there is no warmup). `num_cycles` and `min_lr_ratio` apply to `cosine` and
    to `warmup_stable_decay`; `decay_steps` and `decay_type` to
    `warmup_stable_decay` alone.
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

    if schedule == WARMUP_STABLE_DECAY:
        assert max_steps is not None
        return _warmup_stable_decay_multiplier(
            step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            decay_steps=decay_steps,
            decay_type=decay_type,
            min_lr_ratio=min_lr_ratio,
            num_cycles=num_cycles,
        )

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


def _warmup_stable_decay_multiplier(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    decay_steps: int | None,
    decay_type: str,
    min_lr_ratio: float,
    num_cycles: float,
) -> float:
    """
    Warmup-Stable-Decay, `transformers.optimization._get_wsd_scheduler_lambda`.

    Transcribed branch for branch from that function rather than re-derived,
    for the reason the module docstring gives: the name is HuggingFace's, so
    the curve has to be too. The one argument not exposed is `warmup_type`,
    which stays at HuggingFace's own default (`"linear"`) — every other
    schedule in this module ramps linearly, and a second warmup shape would be
    a knob with nothing asking for it.

    The stable phase is derived, not configured: `max_steps - warmup -
    decay`. HuggingFace lets a caller pass `num_stable_steps` instead and
    silently holds the minimum rate for any steps the three phases fail to
    cover; deriving it means those three numbers cannot disagree in the first
    place.
    """
    if decay_steps is None or decay_steps < 1:
        raise ScheduleError(
            f"the {WARMUP_STABLE_DECAY!r} schedule needs a positive decay_steps — it "
            f"is the length of the phase the schedule exists for. Got "
            f"{decay_steps!r}."
        )
    if decay_type not in DECAY_TYPES:
        raise ScheduleError(
            f"unknown decay_type {decay_type!r}; implemented: {list(DECAY_TYPES)}. "
            f"The names are transformers' get_wsd_schedule(decay_type=...) values."
        )
    stable_steps = max_steps - warmup_steps - decay_steps
    if stable_steps < 0:
        raise ScheduleError(
            f"warmup_steps ({warmup_steps}) + decay_steps ({decay_steps}) exceeds "
            f"max_steps ({max_steps}), so there is no stable phase between them "
            f"and the run would decay out of its own warmup"
        )

    if step < warmup_steps:
        progress = float(step) / float(max(1, warmup_steps))
        return max(0.0, progress * (1.0 - min_lr_ratio) + min_lr_ratio)

    if step < warmup_steps + stable_steps:
        return 1.0

    if step < warmup_steps + stable_steps + decay_steps:
        progress = float(step - warmup_steps - stable_steps) / float(max(1, decay_steps))
        if decay_type == "linear":
            factor = 1.0 - progress
        elif decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        else:  # "1-sqrt"
            factor = 1.0 - math.sqrt(progress)
        return max(0.0, factor * (1.0 - min_lr_ratio) + min_lr_ratio)

    return min_lr_ratio


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


def resolve_decay_steps(
    decay_steps: int, decay_ratio: float, max_steps: int | None
) -> int:
    """
    The `warmup_stable_decay` cooldown length, from steps or from a ratio.

    Same two-ways-to-say-it shape as `resolve_warmup_steps`, and the same
    precedence — an explicit count wins over a ratio — so the two halves of the
    schedule are configured the same way round. HuggingFace has no
    `decay_ratio` of its own (it takes `num_decay_steps` inside
    `lr_scheduler_kwargs`), but "the last 10% of the run" is how the phase is
    reasoned about, and computing that by hand every time `max_steps` moves is
    exactly how a cooldown ends up the wrong length.

    Returns 0 when neither is set, which is only valid for a schedule that has
    no decay phase; `warmup_stable_decay` refuses it upstream.
    """
    if decay_steps:
        return decay_steps
    if not decay_ratio:
        return 0
    if not max_steps:
        raise ScheduleError(
            "lr_decay_ratio needs max_steps to be a fraction of; set "
            "lr_decay_steps directly if the horizon is not fixed"
        )
    return int(max_steps * decay_ratio)
