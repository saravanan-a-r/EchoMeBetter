"""
Unit tests for the learning-rate schedules.

The important tests here compare against `transformers` **itself**, not
against a transcription of its formulas into this file. A transcription would
keep passing after HuggingFace changed something, which defeats the entire
purpose: the schedules are named after `SchedulerType` values precisely so
that someone continuing a released checkpoint under `Trainer` gets the same
curve, and "the same curve" is a claim about the other library's current
behaviour, not about a formula remembered from it.
"""

from __future__ import annotations

import math

import pytest

from src.errors import ScheduleError
from src.schedule import (
    CONSTANT,
    CONSTANT_WITH_WARMUP,
    COSINE,
    DECAY_TYPES,
    INVERSE_SQRT,
    LINEAR,
    SCHEDULES,
    WARMUP_STABLE_DECAY,
    learning_rate_at,
    learning_rate_multiplier,
    resolve_decay_steps,
    resolve_warmup_steps,
)

try:
    import torch
    from transformers import optimization as hf_optimization

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - exercised only where it is absent
    HAVE_TRANSFORMERS = False

needs_transformers = pytest.mark.skipif(
    not HAVE_TRANSFORMERS, reason="transformers is not installed"
)

WARMUP = 100
TOTAL = 1000
# `warmup_stable_decay`'s cooldown: 20% of the run, so the stable phase is
# steps 100-799 and the decay 800-999. Both boundaries appear in STEPS below.
DECAY = 200
# Deliberately includes 0, both sides of the warmup boundary, both sides of the
# stable/decay boundary, and a step past the end of the run — where linear must
# clamp at zero rather than go negative.
STEPS = [0, 1, 50, 99, 100, 101, 250, 500, 799, 800, 801, 999, 1000, 1500]


def extra_arguments(schedule: str, decay_type: str = "cosine") -> dict:
    """
    The arguments a schedule needs beyond warmup and horizon.

    Only `warmup_stable_decay` has any, and it has no usable default for them:
    a decay phase of unstated length is not a schedule.
    """
    if schedule == WARMUP_STABLE_DECAY:
        return {"decay_steps": DECAY, "decay_type": decay_type}
    return {}


def huggingface_multipliers(schedule: str, decay_type: str = "cosine") -> list[float]:
    """The multipliers `transformers` produces, read off a real scheduler."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    builder = {
        CONSTANT: lambda: hf_optimization.get_constant_schedule(optimizer),
        CONSTANT_WITH_WARMUP: lambda: hf_optimization.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP
        ),
        LINEAR: lambda: hf_optimization.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP, num_training_steps=TOTAL
        ),
        COSINE: lambda: hf_optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP, num_training_steps=TOTAL
        ),
        INVERSE_SQRT: lambda: hf_optimization.get_inverse_sqrt_schedule(
            optimizer, num_warmup_steps=WARMUP
        ),
        WARMUP_STABLE_DECAY: lambda: hf_optimization.get_wsd_schedule(
            optimizer,
            num_warmup_steps=WARMUP,
            num_decay_steps=DECAY,
            num_training_steps=TOTAL,
            decay_type=decay_type,
        ),
    }[schedule]
    scheduler = builder()
    return [scheduler.lr_lambdas[0](step) for step in STEPS]


@needs_transformers
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_schedule_agrees_with_transformers(schedule):
    """
    The load-bearing test of this module.

    A run pretrained here and continued under HuggingFace's `Trainer` with the
    same `lr_scheduler_type` must not see the learning rate jump at the
    hand-over. Checked against the live library so it stays true.
    """
    theirs = huggingface_multipliers(schedule)
    ours = [
        learning_rate_multiplier(
            step,
            schedule=schedule,
            warmup_steps=WARMUP,
            max_steps=TOTAL,
            **extra_arguments(schedule),
        )
        for step in STEPS
    ]
    for step, mine, hers in zip(STEPS, ours, theirs):
        assert mine == pytest.approx(hers, abs=1e-12), f"{schedule} at step {step}"


# -- shape properties, which hold whether or not transformers is installed --


WARMED = [CONSTANT_WITH_WARMUP, LINEAR, COSINE, INVERSE_SQRT, WARMUP_STABLE_DECAY]


@pytest.mark.parametrize("schedule", WARMED)
def test_warmed_schedules_start_at_zero(schedule):
    """
    Step 0 has a learning rate of exactly zero, deliberately. The first
    gradient of a randomly initialized model is the least trustworthy one in
    the run, and applying it at full rate is a well-known way to spend the
    first thousand steps recovering.
    """
    assert (
        learning_rate_multiplier(
            0,
            schedule=schedule,
            warmup_steps=WARMUP,
            max_steps=TOTAL,
            **extra_arguments(schedule),
        )
        == 0.0
    )


@pytest.mark.parametrize("schedule", WARMED)
def test_warmup_reaches_the_peak_exactly_once(schedule):
    """
    At `warmup_steps` the multiplier is 1.0 and not a fraction more or less.
    A ramp that overshoots is a spike at the least stable moment of the run.
    """
    peak = learning_rate_multiplier(
        WARMUP,
        schedule=schedule,
        warmup_steps=WARMUP,
        max_steps=TOTAL,
        **extra_arguments(schedule),
    )
    assert peak == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("schedule", WARMED)
def test_the_ramp_is_monotonic(schedule):
    values = [
        learning_rate_multiplier(
            step,
            schedule=schedule,
            warmup_steps=WARMUP,
            max_steps=TOTAL,
            **extra_arguments(schedule),
        )
        for step in range(WARMUP + 1)
    ]
    assert values == sorted(values)


def test_constant_ignores_warmup_entirely():
    """
    `constant` is HuggingFace's un-warmed schedule; `constant_with_warmup` is
    the warmed one. Conflating them would give a run a warmup it did not ask
    for, or take away one it did.
    """
    assert learning_rate_multiplier(0, schedule=CONSTANT, warmup_steps=WARMUP) == 1.0
    assert learning_rate_multiplier(5, schedule=CONSTANT, warmup_steps=WARMUP) == 1.0


def test_linear_reaches_zero_and_stays_there():
    assert learning_rate_multiplier(
        TOTAL, schedule=LINEAR, warmup_steps=WARMUP, max_steps=TOTAL
    ) == 0.0
    assert learning_rate_multiplier(
        TOTAL * 2, schedule=LINEAR, warmup_steps=WARMUP, max_steps=TOTAL
    ) == 0.0


def test_cosine_reaches_zero_at_the_horizon():
    assert learning_rate_multiplier(
        TOTAL, schedule=COSINE, warmup_steps=WARMUP, max_steps=TOTAL
    ) == pytest.approx(0.0, abs=1e-12)


def test_cosine_can_hold_a_floor():
    """
    `min_lr_ratio` keeps a fine-tune from finishing at a learning rate of
    exactly zero, where the last few thousand steps do nothing at all.
    """
    end = learning_rate_multiplier(
        TOTAL, schedule=COSINE, warmup_steps=WARMUP, max_steps=TOTAL, min_lr_ratio=0.1
    )
    assert end == pytest.approx(0.1, abs=1e-12)


# -- warmup_stable_decay (architecture_improvements.md item 2) -------------


def wsd(step, **overrides):
    arguments = {
        "schedule": WARMUP_STABLE_DECAY,
        "warmup_steps": WARMUP,
        "max_steps": TOTAL,
        "decay_steps": DECAY,
        "decay_type": "linear",
    }
    arguments.update(overrides)
    return learning_rate_multiplier(step, **arguments)


def test_the_stable_phase_is_flat_at_the_peak():
    """
    The whole middle of the run sits at exactly the peak — that is what
    separates WSD from inverse_sqrt, which is already decaying by step 101.

    This is the property the schedule is chosen for and the one most likely to
    be broken by a well-meaning edit that reuses the inverse-sqrt branch.
    """
    stable = [wsd(step) for step in (WARMUP, WARMUP + 1, 300, 500, 700, 799)]
    assert stable == [1.0] * len(stable)


def test_the_decay_phase_starts_exactly_where_the_stable_phase_ends():
    """
    Step 800 is the first decay step and still sits at the peak (progress 0);
    801 is the first step actually below it. An off-by-one here moves the
    cooldown boundary — and `cooldown_start_step`, which the data mixture
    switches on, is derived from the same arithmetic.
    """
    assert wsd(TOTAL - DECAY) == pytest.approx(1.0, abs=1e-12)
    assert wsd(TOTAL - DECAY + 1) < 1.0


def test_the_decay_reaches_zero_at_the_horizon_and_stays_there():
    """
    A cooldown that does not finish at (near) zero has not annealed; a
    cooldown that goes negative would reverse the last few steps of the run.
    """
    assert wsd(TOTAL) == 0.0
    assert wsd(TOTAL * 2) == 0.0
    assert all(wsd(step) >= 0.0 for step in range(TOTAL, TOTAL + 50))


def test_the_decay_is_monotonic():
    values = [wsd(step) for step in range(TOTAL - DECAY, TOTAL + 1)]
    assert values == sorted(values, reverse=True)


def test_a_floor_holds_after_the_horizon():
    """`min_lr_ratio` is honoured by the tail, not only by the decay curve."""
    assert wsd(TOTAL, min_lr_ratio=0.1) == pytest.approx(0.1, abs=1e-12)
    assert wsd(TOTAL * 2, min_lr_ratio=0.1) == pytest.approx(0.1, abs=1e-12)


@needs_transformers
@pytest.mark.parametrize("decay_type", DECAY_TYPES)
def test_every_decay_shape_agrees_with_transformers(decay_type):
    """
    Each shape separately, not just the default — the project uses `linear`
    while `transformers`' own default is `cosine`, so the one this run
    actually depends on is not the one the parametrized agreement test above
    happens to exercise.
    """
    theirs = huggingface_multipliers(WARMUP_STABLE_DECAY, decay_type=decay_type)
    ours = [wsd(step, decay_type=decay_type) for step in STEPS]
    for step, mine, hers in zip(STEPS, ours, theirs):
        assert mine == pytest.approx(hers, abs=1e-12), f"{decay_type} at step {step}"


def test_a_decay_phase_of_unstated_length_is_refused():
    """
    Silently treating a missing `decay_steps` as zero would produce a run that
    holds the peak learning rate to the last step and then stops — the exact
    behaviour this schedule was chosen to replace.
    """
    with pytest.raises(ScheduleError, match="decay_steps"):
        learning_rate_multiplier(
            0, schedule=WARMUP_STABLE_DECAY, warmup_steps=WARMUP, max_steps=TOTAL
        )


def test_a_decay_longer_than_the_run_is_refused():
    with pytest.raises(ScheduleError, match="exceeds"):
        wsd(0, decay_steps=TOTAL)


def test_an_unknown_decay_shape_is_refused():
    with pytest.raises(ScheduleError, match="decay_type"):
        wsd(0, decay_type="exponential")


def test_decay_steps_beat_a_ratio():
    assert resolve_decay_steps(250, 0.1, TOTAL) == 250


def test_a_decay_ratio_is_taken_of_max_steps():
    assert resolve_decay_steps(0, 0.1, TOTAL) == 100


def test_no_decay_phase_at_all_is_allowed():
    """The five schedules without a decay phase resolve to zero, not an error."""
    assert resolve_decay_steps(0, 0.0, TOTAL) == 0


def test_a_decay_ratio_with_no_horizon_is_refused():
    with pytest.raises(ScheduleError, match="max_steps"):
        resolve_decay_steps(0, 0.1, None)


def test_inverse_sqrt_never_reaches_zero():
    """
    The property that makes it the right choice for pretraining (§13): the run
    can be extended at any point without the schedule having been wrong in
    hindsight. A cosine schedule whose horizon is later moved has already
    decayed to its floor at the old horizon.
    """
    late = learning_rate_multiplier(
        10_000_000, schedule=INVERSE_SQRT, warmup_steps=WARMUP
    )
    assert late > 0.0
    assert late < 0.01


def test_inverse_sqrt_is_continuous_at_the_end_of_warmup():
    """
    The `shift` in the formula exists for this. Without it the multiplier
    drops discontinuously the moment the ramp finishes — a visible loss bump
    at exactly the step the warmup ends, which is easy to misread as a data
    problem.
    """
    before = learning_rate_multiplier(
        WARMUP - 1, schedule=INVERSE_SQRT, warmup_steps=WARMUP
    )
    at = learning_rate_multiplier(WARMUP, schedule=INVERSE_SQRT, warmup_steps=WARMUP)
    assert at == pytest.approx(1.0, abs=1e-12)
    assert abs(at - before) < 0.02


def test_inverse_sqrt_decays_as_the_inverse_square_root():
    """Checked against the closed form, not against itself."""
    timescale = 500
    for step in (600, 2000, 12_000):
        expected = 1.0 / math.sqrt((step + timescale - WARMUP) / timescale)
        assert learning_rate_multiplier(
            step, schedule=INVERSE_SQRT, warmup_steps=WARMUP, timescale=timescale
        ) == pytest.approx(expected, abs=1e-12)


def test_inverse_sqrt_without_warmup_uses_huggingfaces_fallback_timescale():
    """`timescale = num_warmup_steps or 10_000`, exactly as transformers does."""
    expected = 1.0 / math.sqrt((5000 + 10_000) / 10_000)
    assert learning_rate_multiplier(
        5000, schedule=INVERSE_SQRT, warmup_steps=0
    ) == pytest.approx(expected, abs=1e-12)


# -- refusals --------------------------------------------------------------


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ScheduleError, match="unknown schedule"):
        learning_rate_multiplier(0, schedule="triangular", warmup_steps=10)


@pytest.mark.parametrize("schedule", [LINEAR, COSINE])
def test_a_decaying_schedule_without_a_horizon_is_refused(schedule):
    """
    Defaulting `max_steps` to 1 would make the multiplier zero from step 1
    onwards: a run that trains for weeks and learns nothing, with no error
    and a perfectly flat loss curve.
    """
    with pytest.raises(ScheduleError, match="max_steps"):
        learning_rate_multiplier(10, schedule=schedule, warmup_steps=5)


def test_a_negative_step_is_refused():
    with pytest.raises(ScheduleError, match="non-negative"):
        learning_rate_multiplier(-1, schedule=CONSTANT, warmup_steps=0)


# -- the peak rate and warmup resolution -----------------------------------


def test_learning_rate_at_scales_the_multiplier():
    peak = 3e-4
    assert learning_rate_at(
        WARMUP, peak, schedule=INVERSE_SQRT, warmup_steps=WARMUP
    ) == pytest.approx(peak, abs=1e-15)
    assert learning_rate_at(0, peak, schedule=INVERSE_SQRT, warmup_steps=WARMUP) == 0.0


def test_explicit_warmup_steps_beat_a_ratio():
    """HuggingFace's precedence, so a copied configuration behaves the same."""
    assert resolve_warmup_steps(warmup_steps=250, warmup_ratio=0.5, max_steps=1000) == 250


def test_a_ratio_is_taken_of_max_steps():
    assert resolve_warmup_steps(warmup_steps=0, warmup_ratio=0.01, max_steps=200_000) == 2000


def test_no_warmup_at_all_is_allowed():
    assert resolve_warmup_steps(warmup_steps=0, warmup_ratio=0.0, max_steps=1000) == 0


def test_a_ratio_with_no_horizon_is_refused():
    """
    Silently resolving to zero would turn a typo into "no warmup", which is a
    diverged first hour rather than a visible error.
    """
    with pytest.raises(ScheduleError, match="max_steps"):
        resolve_warmup_steps(warmup_steps=0, warmup_ratio=0.01, max_steps=None)
