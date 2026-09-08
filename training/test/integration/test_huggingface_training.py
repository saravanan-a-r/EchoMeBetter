"""
HuggingFace compatibility of the *training* configuration.

`model/` made the network and its weights loadable by `transformers`. This
closes the remaining gap for someone fine-tuning a released checkpoint: they
should be able to start from the settings the model was actually pretrained
under, rather than guessing them from a README.

As with the model, the tests check against the live library rather than
against a transcription of it. `test_the_export_builds_real_training_arguments`
constructs an actual `TrainingArguments` and reads the values back, so a field
`transformers` renames or drops fails here rather than in someone else's
fine-tune.
"""

from __future__ import annotations

import pytest

from conftest import TINY_TRAINING
from src.config import OURS_ALONE, TrainingConfig, load_training_config
from src.schedule import SCHEDULES

try:
    from transformers import SchedulerType, TrainingArguments

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - exercised only where it is absent
    HAVE_TRANSFORMERS = False

needs_transformers = pytest.mark.skipif(
    not HAVE_TRANSFORMERS, reason="transformers is not installed"
)

pytestmark = pytest.mark.huggingface

# Fields that exist in `transformers.TrainingArguments` under the same name
# and with the same meaning. Frozen alongside `OURS_ALONE`, so that adding a
# field to `TrainingConfig` forces a decision rather than passing silently:
# either it matches a TrainingArguments name, or it is consciously ours.
SHARED_WITH_TRAINING_ARGUMENTS = (
    "output_dir",
    "seed",
    "data_seed",
    "max_steps",
    "num_train_epochs",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_steps",
    "warmup_ratio",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "bf16",
    "fp16",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "label_smoothing_factor",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "metric_for_best_model",
    "greater_is_better",
    "load_best_model_at_end",
    "resume_from_checkpoint",
)


def config(**overrides) -> TrainingConfig:
    return TrainingConfig(**dict(TINY_TRAINING, **overrides))


# -- the field vocabulary --------------------------------------------------


def test_every_field_is_classified():
    """
    Forces a decision when a field is added. Silence is what lets a near-miss
    name like `warmup_fraction` ship next to HuggingFace's `warmup_ratio`, and
    then everyone who copies a configuration between the two gets it wrong
    once.
    """
    declared = set(SHARED_WITH_TRAINING_ARGUMENTS) | set(OURS_ALONE)
    assert set(TrainingConfig.__dataclass_fields__) == declared


@needs_transformers
def test_every_shared_field_exists_in_training_arguments():
    """
    Checked against `transformers` itself, so it stays true as the library
    changes rather than as of whenever this list was written.
    """
    import dataclasses

    theirs = {field.name for field in dataclasses.fields(TrainingArguments)}
    missing = [name for name in SHARED_WITH_TRAINING_ARGUMENTS if name not in theirs]
    assert missing == []


@needs_transformers
def test_none_of_our_own_fields_collide_with_training_arguments():
    """
    A field of ours sharing a HuggingFace name but not its meaning is worse
    than an unfamiliar name: it reads as understood and behaves differently.
    """
    import dataclasses

    theirs = {field.name for field in dataclasses.fields(TrainingArguments)}
    assert [name for name in OURS_ALONE if name in theirs] == []


@needs_transformers
def test_our_schedule_names_are_huggingfaces():
    """
    Not "similar to" — the same strings. `lr_scheduler_type: inverse_sqrt`
    must mean the same schedule on both sides, which is what makes a
    hand-over produce the same curve rather than a similar one.
    """
    theirs = {item.value for item in SchedulerType}
    assert set(SCHEDULES) <= theirs


# -- the export ------------------------------------------------------------


@needs_transformers
def test_the_export_builds_real_training_arguments(tmp_path):
    """
    The load-bearing test: a real `TrainingArguments` is constructed from the
    export and the values are read back. Asserting on the dict alone would
    keep passing after `transformers` renamed a field.
    """
    settings = config(output_dir=str(tmp_path))
    arguments = TrainingArguments(**settings.to_huggingface_training_arguments())

    assert arguments.learning_rate == settings.learning_rate
    assert arguments.max_steps == settings.max_steps
    assert arguments.weight_decay == settings.weight_decay
    assert (arguments.adam_beta1, arguments.adam_beta2) == (
        settings.adam_beta1,
        settings.adam_beta2,
    )
    assert arguments.max_grad_norm == settings.max_grad_norm
    assert arguments.gradient_accumulation_steps == settings.gradient_accumulation_steps
    assert arguments.per_device_train_batch_size == settings.per_device_train_batch_size
    assert arguments.label_smoothing_factor == settings.label_smoothing_factor


@needs_transformers
@pytest.mark.parametrize("stage", ["pretrain", "sft", "rehearsal"])
def test_every_real_stage_exports_cleanly(tmp_path, stage):
    """
    The configurations that will actually be published, not a synthetic one.
    """
    settings = load_training_config(stage=stage).with_(output_dir=str(tmp_path))
    arguments = TrainingArguments(**settings.to_huggingface_training_arguments())
    assert arguments.lr_scheduler_type.value == settings.lr_scheduler_type


@needs_transformers
def test_the_exported_schedule_survives_the_round_trip(tmp_path):
    """
    `TrainingArguments` coerces the string into a `SchedulerType`; it must
    come back as the same schedule and not as the enum's default.
    """
    for schedule in SCHEDULES:
        settings = config(
            lr_scheduler_type=schedule, output_dir=str(tmp_path), max_steps=100
        )
        arguments = TrainingArguments(**settings.to_huggingface_training_arguments())
        assert arguments.lr_scheduler_type.value == schedule


@needs_transformers
def test_the_exported_warmup_is_already_resolved(tmp_path):
    """
    `warmup_steps` and `warmup_ratio` are two specifications of one thing.
    Exporting both would leave HuggingFace to apply its own precedence to a
    pair we have already reconciled — and if the two ever disagreed, the
    continuation would warm up for a different length than pretraining did.
    """
    settings = config(
        warmup_steps=0, warmup_ratio=0.05, max_steps=1000, output_dir=str(tmp_path)
    )
    exported = settings.to_huggingface_training_arguments()
    assert exported["warmup_steps"] == 50
    assert "warmup_ratio" not in exported
    assert TrainingArguments(**exported).warmup_steps == 50


def test_the_export_omits_our_own_fields():
    """
    Omitted rather than renamed. `TrainingArguments` would reject them, and
    inventing a near-equivalent name would be worse than leaving the user to
    set the thing themselves.
    """
    exported = config().to_huggingface_training_arguments()
    assert [name for name in OURS_ALONE if name in exported] == []


def test_the_export_omits_fp16():
    """
    Not supported here (see `config.py`), and exporting `fp16: False` next to
    `bf16: True` is fine — but exporting it as a *choice* would suggest it is
    one. It is simply absent.
    """
    assert "fp16" not in config().to_huggingface_training_arguments()


def test_the_export_declares_step_based_strategies():
    """
    HuggingFace defaults to epoch-based evaluation and saving. This run has no
    epochs — pretraining is a token budget over a stream (§7.5) — so a
    continuation that inherited `eval_strategy="epoch"` would evaluate once,
    at the end, and the §7.7 diagnostics would never run.
    """
    exported = config().to_huggingface_training_arguments()
    assert exported["eval_strategy"] == "steps"
    assert exported["save_strategy"] == "steps"
    assert exported["logging_strategy"] == "steps"


@needs_transformers
def test_the_pretraining_settings_a_finetuner_would_read_are_all_present(tmp_path):
    """
    The scenario this whole file exists for: someone downloads the released
    checkpoint and wants to know what it was trained under. Every number that
    changes the outcome must be in the export.
    """
    settings = load_training_config(stage="pretrain").with_(output_dir=str(tmp_path))
    exported = settings.to_huggingface_training_arguments()
    for key in (
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "max_grad_norm",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "bf16",
        "label_smoothing_factor",
    ):
        assert key in exported, key
