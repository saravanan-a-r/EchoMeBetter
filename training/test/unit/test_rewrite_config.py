"""
The synthetic rewrite task's two configuration fields (item 4).

Same shape as the cooldown-blend tests in `test_config.py`: every way the
pair can describe a task the run would never perform is refused, and the
real `training_config.yml` is checked for the shape of the judgement rather
than its exact numbers.
"""

from __future__ import annotations

import pytest

from conftest import TINY_TRAINING
from src.config import OURS_ALONE, TrainingConfig, load_training_config
from src.errors import TrainingConfigError

BLEND = {"fineweb_edu": 2.0, "c4": 1.0}


def config(**overrides) -> TrainingConfig:
    return TrainingConfig(**dict(TINY_TRAINING, **overrides))


def wsd(**overrides) -> TrainingConfig:
    return config(
        lr_scheduler_type="warmup_stable_decay",
        lr_decay_ratio=0.5,
        lr_decay_type="linear",
        **overrides,
    )


def test_the_task_is_off_by_default():
    assert config().rewrite_share == 0.0
    assert config().rewrite_blend is None


def test_a_share_with_a_blend_on_a_wsd_run_is_accepted():
    settings = wsd(rewrite_share=0.2, rewrite_blend=BLEND)
    assert settings.rewrite_share == 0.2
    assert settings.cooldown_start_step == 4


def test_a_share_without_a_blend_is_refused():
    with pytest.raises(TrainingConfigError, match="nothing to draw"):
        wsd(rewrite_share=0.2)


def test_a_blend_without_a_share_is_refused():
    with pytest.raises(TrainingConfigError, match="never be read"):
        wsd(rewrite_blend=BLEND)


@pytest.mark.parametrize("share", [-0.1, 1.0, 1.5, "half", True])
def test_a_share_outside_the_unit_interval_is_refused(share):
    with pytest.raises(TrainingConfigError, match="rewrite_share"):
        wsd(rewrite_share=share, rewrite_blend=BLEND)


def test_the_task_needs_a_decay_phase_to_run_in():
    with pytest.raises(TrainingConfigError, match="no decay phase"):
        config(lr_scheduler_type="cosine", rewrite_share=0.2, rewrite_blend=BLEND)


@pytest.mark.parametrize("blend", [{}, {"c4": 0.0}, {"c4": -1.0}, {"c4": "half"}, {3: 1.0}])
def test_a_malformed_blend_is_refused(blend):
    with pytest.raises(TrainingConfigError, match="rewrite_blend"):
        wsd(rewrite_share=0.2, rewrite_blend=blend)


def test_both_fields_are_classified_as_ours():
    assert "rewrite_share" in OURS_ALONE and "rewrite_blend" in OURS_ALONE
    exported = wsd(rewrite_share=0.2, rewrite_blend=BLEND).to_huggingface_training_arguments()
    assert "rewrite_share" not in exported and "rewrite_blend" not in exported


# -- the real configuration -------------------------------------------------


def test_pretraining_runs_the_task_on_a_fifth_of_the_cooldown():
    settings = load_training_config(stage="pretrain")
    assert settings.rewrite_share == pytest.approx(0.20)
    assert settings.rewrite_blend


def test_the_rewrite_blend_holds_prose_sources_only():
    """
    The judgement worth pinning: CLI text and source code have no grammar to
    break and restore, and a target that "corrects" them teaches editing
    code. Which prose sources are in, and at what weight, may change.
    """
    blend = load_training_config(stage="pretrain").rewrite_blend
    assert not any(source.startswith("cli_") for source in blend)
    assert "permissive_code" not in blend
    assert {"fineweb_edu", "c4", "cosmopedia"} <= set(blend)


def test_the_rewrite_blend_follows_the_corpus_proportions():
    """Weighted by natural representation, so the largest source leads."""
    blend = load_training_config(stage="pretrain").rewrite_blend
    assert blend["fineweb_edu"] > blend["c4"] > blend["cosmopedia"]


@pytest.mark.parametrize("stage", ["sft", "rehearsal"])
def test_other_stages_do_not_run_the_task(stage):
    settings = load_training_config(stage=stage)
    assert settings.rewrite_share == 0.0
    assert settings.rewrite_blend is None
