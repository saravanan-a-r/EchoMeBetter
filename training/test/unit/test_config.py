"""
Unit tests for `TrainingConfig` and the YAML it is loaded from.

Two things are being locked down. First, that a mistyped or impossible setting
fails *before* the run starts rather than after a week of it — every refusal
below stands in for a specific way a run can be wrong while looking fine.
Second, that the real `training_config.yml` says what architecture.md §13 says
it should, so the reference configuration and the file that implements it
cannot drift.
"""

from __future__ import annotations

import pytest
import yaml

from conftest import TINY_TRAINING, TRAINING_CONFIG_PATH
from src.config import (
    OURS_ALONE,
    TrainingConfig,
    available_stages,
    build_training_config,
    load_training_config,
)
from src.errors import TrainingConfigError


def config(**overrides) -> TrainingConfig:
    return TrainingConfig(**dict(TINY_TRAINING, **overrides))


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
    ],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_counts_must_be_positive(field, bad):
    with pytest.raises(TrainingConfigError, match=field):
        config(**{field: bad})


@pytest.mark.parametrize("field", ["max_steps", "logging_steps"])
def test_a_boolean_is_not_a_count(field):
    """
    `True` is an `int` in Python and passes a naive `> 0` check. A YAML typo
    that writes `logging_steps: yes` would otherwise become 1.
    """
    with pytest.raises(TrainingConfigError):
        config(**{field: True})


def test_the_learning_rate_must_be_positive():
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        config(learning_rate=0.0)


def test_an_unknown_schedule_is_refused():
    with pytest.raises(TrainingConfigError, match="lr_scheduler_type"):
        config(lr_scheduler_type="triangular")


@pytest.mark.parametrize(
    "field", ["weight_decay", "warmup_ratio", "label_smoothing_factor", "min_lr_ratio"]
)
@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_ratios_must_be_in_the_unit_interval(field, bad):
    with pytest.raises(TrainingConfigError, match=field):
        config(**{field: bad})


def test_a_run_that_is_entirely_warmup_is_refused():
    """
    `warmup_steps >= max_steps` means the learning rate ramps for the whole
    run and never reaches its peak. Nothing errors; the model simply trains at
    a fraction of the intended rate and the result is read as "this
    configuration underperforms".
    """
    with pytest.raises(TrainingConfigError, match="entirely warmup"):
        config(max_steps=100, warmup_steps=100)


def test_clipping_cannot_be_disabled_by_setting_it_to_zero():
    with pytest.raises(TrainingConfigError, match="max_grad_norm"):
        config(max_grad_norm=0.0)


def test_fp16_is_refused_with_a_reason():
    """
    Not an oversight. fp16 needs a loss scaler and stalls silently on overflow
    in long runs; bf16 has the same memory saving with float32's exponent
    range. The field exists only so a TrainingArguments export is complete.
    """
    with pytest.raises(TrainingConfigError, match="bf16"):
        config(fp16=True)


def test_load_best_model_at_end_needs_the_best_checkpoint_kept():
    """
    Otherwise retention deletes the checkpoint the run is about to load, and
    the failure happens at the very end of a multi-week run.
    """
    with pytest.raises(TrainingConfigError, match="keep_best_checkpoint"):
        config(load_best_model_at_end=True, keep_best_checkpoint=False)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("save_total_limit", 0),
        ("eval_max_batches", 0),
        ("lr_timescale", 0),
        ("num_cycles", 0.0),
        ("dropout_rate", 1.0),
        ("adam_beta1", 1.0),
        ("adam_epsilon", 0.0),
    ],
)
def test_optional_settings_are_still_validated(field, bad):
    with pytest.raises(TrainingConfigError, match=field):
        config(**{field: bad})


def test_absent_optional_settings_are_accepted():
    assert config(save_total_limit=None, eval_max_batches=None, dropout_rate=None)


# -- derived ---------------------------------------------------------------


def test_warmup_resolves_from_a_ratio_when_steps_are_not_set():
    assert config(warmup_steps=0, warmup_ratio=0.25, max_steps=400).resolved_warmup_steps == 100


def test_the_learning_rate_curve_comes_from_the_configuration():
    settings = config(
        learning_rate=1e-3,
        warmup_steps=10,
        max_steps=2000,
        lr_scheduler_type="inverse_sqrt",
    )
    assert settings.learning_rate_at(0) == 0.0
    assert settings.learning_rate_at(10) == pytest.approx(1e-3)
    assert settings.learning_rate_at(1000) < 1e-3


# -- serialization ---------------------------------------------------------


def test_a_configuration_round_trips_through_a_dict():
    original = config()
    assert TrainingConfig.from_dict(original.to_dict()) == original


def test_an_unknown_key_is_rejected_not_ignored():
    """
    A silently-ignored key means running with a setting the file says is in
    force and that nothing actually reads.
    """
    with pytest.raises(TrainingConfigError, match="unknown"):
        TrainingConfig.from_dict(dict(TINY_TRAINING, learnign_rate=1e-4))


def test_a_missing_required_key_is_rejected():
    incomplete = dict(TINY_TRAINING)
    del incomplete["learning_rate"]
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        TrainingConfig.from_dict(incomplete)


def test_with_reruns_validation():
    with pytest.raises(TrainingConfigError):
        config().with_(max_grad_norm=-1.0)


# -- the YAML document -----------------------------------------------------


def document(stages, defaults=None, active="a"):
    return {
        "active_stage": active,
        "defaults": dict(TINY_TRAINING, **(defaults or {})),
        "stages": stages,
    }


def test_a_stage_overrides_the_defaults():
    built = build_training_config(document({"a": {"learning_rate": 7e-5}}))
    assert built.learning_rate == 7e-5
    assert built.stage == "a"


def test_a_stage_key_outside_the_allowed_set_is_refused():
    """
    Stage blocks are restricted for the same reason `model_config.yml`'s
    profile blocks are: a key that belongs in `defaults` but was written in a
    stage would be applied to one stage and silently missing from the others.
    """
    with pytest.raises(TrainingConfigError, match="unknown keys"):
        build_training_config(document({"a": {"weight_decay": 0.2}}))


def test_an_unknown_stage_is_refused():
    with pytest.raises(TrainingConfigError, match="unknown stage"):
        build_training_config(document({"a": {}}), stage="b")


def test_a_document_with_no_active_stage_and_no_request_is_refused():
    doc = document({"a": {}})
    del doc["active_stage"]
    with pytest.raises(TrainingConfigError, match="active_stage"):
        build_training_config(doc)


@pytest.mark.parametrize("block", ["defaults", "stages"])
def test_a_missing_block_is_refused(block):
    doc = document({"a": {}})
    del doc[block]
    with pytest.raises(TrainingConfigError, match=block):
        build_training_config(doc)


def test_a_missing_file_is_refused():
    with pytest.raises(TrainingConfigError, match="not found"):
        load_training_config("/nonexistent/training_config.yml")


# -- the real file ---------------------------------------------------------


def test_the_real_file_declares_the_three_stages():
    assert available_stages() == ["pretrain", "rehearsal", "sft"]


@pytest.mark.parametrize("stage", ["pretrain", "sft", "rehearsal"])
def test_every_real_stage_loads(stage):
    assert load_training_config(stage=stage).stage == stage


def test_pretraining_uses_inverse_sqrt():
    """
    architecture.md §13. Not interchangeable with cosine: inverse_sqrt never
    needs to know how long the run is, and a 50-100B token run's length is not
    settled in advance (§7.5).
    """
    assert load_training_config(stage="pretrain").lr_scheduler_type == "inverse_sqrt"


def test_sft_uses_cosine_and_a_lower_rate():
    """architecture.md §13: warmup -> cosine, at a lower peak than pretraining."""
    sft = load_training_config(stage="sft")
    pretrain = load_training_config(stage="pretrain")
    assert sft.lr_scheduler_type == "cosine"
    assert sft.learning_rate < pretrain.learning_rate


def test_dropout_is_off_for_pretraining_and_on_for_sft():
    """architecture.md §13's table: 0.0 pretraining, 0.1 SFT."""
    assert load_training_config(stage="pretrain").dropout_rate == 0.0
    assert load_training_config(stage="sft").dropout_rate == 0.1


def test_label_smoothing_is_sft_only():
    """architecture.md §13: no smoothing during pretraining, 0.1 for SFT."""
    assert load_training_config(stage="pretrain").label_smoothing_factor == 0.0
    assert load_training_config(stage="sft").label_smoothing_factor == 0.1


def test_the_pretraining_effective_batch_reaches_a_million_tokens():
    """
    architecture.md §13: ">=1M tokens/step via gradient accumulation". Worked
    against the model's real context length rather than a number repeated
    here, so raising the context without revisiting the batch size fails.
    """
    from src.interop import rephrase_model

    pretrain = load_training_config(stage="pretrain")
    model = rephrase_model.load_model_config(profile="large")
    tokens = (
        pretrain.per_device_train_batch_size
        * pretrain.gradient_accumulation_steps
        * model.max_encoder_length
    )
    assert tokens >= 1_000_000


def test_the_reference_optimizer_settings_are_section_13s():
    """β 0.9/0.95 and wd 0.1 — and 0.95 is not PyTorch's default."""
    pretrain = load_training_config(stage="pretrain")
    assert (pretrain.adam_beta1, pretrain.adam_beta2) == (0.9, 0.95)
    assert pretrain.weight_decay == 0.1


def test_pretraining_warmup_is_one_to_two_percent():
    """architecture.md §13: "warmup ~1-2% of steps"."""
    pretrain = load_training_config(stage="pretrain")
    fraction = pretrain.resolved_warmup_steps / pretrain.max_steps
    assert 0.01 <= fraction <= 0.02


def test_gradient_checkpointing_and_bf16_are_on_for_pretraining():
    """architecture.md §13's memory and precision rows."""
    pretrain = load_training_config(stage="pretrain")
    assert pretrain.gradient_checkpointing is True
    assert pretrain.bf16 is True
    assert pretrain.fp16 is False


def test_resumption_defaults_on():
    """§7.9: a 50-100B token run *will* be interrupted."""
    assert load_training_config(stage="pretrain").resume_from_checkpoint is True


def test_the_rehearsal_stage_builds_the_base_profile():
    """§7.8: the full-pipeline rehearsal runs on Base (223M), not Large."""
    assert load_training_config(stage="rehearsal").model_profile == "base"


def test_every_field_the_yaml_sets_is_a_real_field():
    """
    Guards against a setting that looks configured and is not read by
    anything — the failure mode where someone changes a value in the file and
    nothing happens.
    """
    known = set(TrainingConfig.__dataclass_fields__)
    parsed = yaml.safe_load(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(parsed["defaults"]) - known == set()
    for name, stage in parsed["stages"].items():
        assert set(stage) - known == set(), name


def test_our_own_fields_are_declared_as_ours():
    """
    `OURS_ALONE` is frozen so that adding a field forces a decision: either it
    matches a `transformers.TrainingArguments` name, or it is consciously
    this project's. Silence is what lets a near-miss name like
    `warmup_fraction` ship next to HuggingFace's `warmup_ratio`.
    """
    for name in OURS_ALONE:
        assert name in TrainingConfig.__dataclass_fields__, name
