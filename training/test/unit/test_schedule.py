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
    INVERSE_SQRT,
    LINEAR,
    SCHEDULES,
    learning_rate_at,
    learning_rate_multiplier,
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
# Deliberately includes 0, both sides of the warmup boundary, and a step past
# the end of the run — where linear must clamp at zero rather than go negative.
STEPS = [0, 1, 50, 99, 100, 101, 250, 500, 999, 1000, 1500]


def huggingface_multipliers(schedule: str) -> list[float]:
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
            step, schedule=schedule, warmup_steps=WARMUP, max_steps=TOTAL
        )
        for step in STEPS
    ]
    for step, mine, hers in zip(STEPS, ours, theirs):
        assert mine == pytest.approx(hers, abs=1e-12), f"{schedule} at step {step}"


# -- shape properties, which hold whether or not transformers is installed --


@pytest.mark.parametrize("schedule", [CONSTANT_WITH_WARMUP, LINEAR, COSINE, INVERSE_SQRT])
def test_warmed_schedules_start_at_zero(schedule):
    """
    Step 0 has a learning rate of exactly zero, deliberately. The first
    gradient of a randomly initialized model is the least trustworthy one in
    the run, and applying it at full rate is a well-known way to spend the
    first thousand steps recovering.
    """
    assert (
        learning_rate_multiplier(
            0, schedule=schedule, warmup_steps=WARMUP, max_steps=TOTAL
        )
        == 0.0
    )


@pytest.mark.parametrize("schedule", [CONSTANT_WITH_WARMUP, LINEAR, COSINE, INVERSE_SQRT])
def test_warmup_reaches_the_peak_exactly_once(schedule):
    """
    At `warmup_steps` the multiplier is 1.0 and not a fraction more or less.
    A ramp that overshoots is a spike at the least stable moment of the run.
    """
    peak = learning_rate_multiplier(
        WARMUP, schedule=schedule, warmup_steps=WARMUP, max_steps=TOTAL
    )
    assert peak == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("schedule", [CONSTANT_WITH_WARMUP, LINEAR, COSINE, INVERSE_SQRT])
def test_the_ramp_is_monotonic(schedule):
    values = [
        learning_rate_multiplier(
            step, schedule=schedule, warmup_steps=WARMUP, max_steps=TOTAL
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
