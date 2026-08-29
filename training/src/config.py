"""
Training configuration, loaded from `training_config.yml`.

Same discipline as `model_config.yml`: every knob lives in one YAML file, no
hyperparameter is written in Python, and a run manifest can record exactly one
object that determines how the run behaved. The split between the two files is
by *lifetime*, not by taste — `model_config.yml` holds what is welded into a
checkpoint's weight shapes and can never change after pretraining begins;
this holds what a later run may legitimately choose differently.

HuggingFace naming
------------------
Every field that exists in `transformers.TrainingArguments` uses **that
field's exact name and meaning**, for the same reason the model config uses
`T5Config`'s:

    output_dir  seed  data_seed  max_steps  num_train_epochs
    learning_rate  lr_scheduler_type  warmup_steps  warmup_ratio
    weight_decay  adam_beta1  adam_beta2  adam_epsilon  max_grad_norm
    per_device_train_batch_size  gradient_accumulation_steps
    gradient_checkpointing  bf16  fp16  label_smoothing_factor
    logging_steps  eval_steps  save_steps  save_total_limit
    metric_for_best_model  greater_is_better  load_best_model_at_end
    resume_from_checkpoint

`to_huggingface_training_arguments()` turns this into a dict that
`TrainingArguments(**d)` accepts, so someone continuing a released checkpoint
with `Trainer` can start from the settings the model was actually pretrained
under rather than guessing them. `test_huggingface_arguments.py` builds a real
`TrainingArguments` from it and checks the values survive.

Fields with no `TrainingArguments` counterpart are marked in the YAML and
listed in `OURS_ALONE` below, so a reader can tell at a glance which half of
the file is a shared vocabulary and which half is this project's.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import TrainingConfigError
from .schedule import SCHEDULES, learning_rate_at, resolve_warmup_steps

# training/src/config.py -> training/src -> training -> project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "training_config.yml"

# Fields a stage may set. Anything else in a stage block is a typo, and a
# silently-ignored typo means a run that is not the one intended — the same
# rule, for the same reason, as `model_config.yml`'s profiles.
_STAGE_FIELDS = frozenset(
    {
        "max_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "warmup_ratio",
        "dropout_rate",
        "label_smoothing_factor",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "model_profile",
        "output_dir",
    }
)

# Fields this project has that `transformers.TrainingArguments` does not.
# Frozen so a test forces a decision when a field is added: either it matches
# a TrainingArguments name, or it is consciously ours.
OURS_ALONE = (
    "stage",
    "model_profile",
    "dropout_rate",
    "lr_timescale",
    "min_lr_ratio",
    "num_cycles",
    "eval_max_batches",
    "keep_best_checkpoint",
    "log_per_mode_loss",
)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Everything that determines how a run behaves, and nothing about the model.

    Frozen: a mid-run change to the learning rate or the accumulation count
    would make the recorded configuration a lie about what actually happened,
    and a resumed run would silently differ from the one it claims to
    continue.
    """

    # -- where and how reproducibly ---------------------------------------
    output_dir: str
    seed: int
    data_seed: int

    # -- length ------------------------------------------------------------
    # `max_steps` counts optimizer steps, not micro-batches — HuggingFace's
    # meaning. `num_train_epochs` is carried for TrainingArguments
    # compatibility and is unused here: pretraining is defined by a token
    # budget over a streamed corpus (§7.5), which has no epochs.
    max_steps: int
    num_train_epochs: float

    # -- optimization ------------------------------------------------------
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int
    warmup_ratio: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float

    # -- batching ----------------------------------------------------------
    per_device_train_batch_size: int
    gradient_accumulation_steps: int

    # -- memory and precision ---------------------------------------------
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool

    # -- loss --------------------------------------------------------------
    # Must agree with `ModelConfig.label_smoothing`, because the model owns
    # the loss computation. `Trainer` refuses to start if they disagree
    # rather than picking one — see `loop.py`.
    label_smoothing_factor: float

    # -- cadence -----------------------------------------------------------
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int | None
    metric_for_best_model: str
    greater_is_better: bool
    load_best_model_at_end: bool
    resume_from_checkpoint: bool

    # -- ours; no TrainingArguments counterpart ---------------------------
    stage: str = "pretrain"
    model_profile: str | None = None
    dropout_rate: float | None = None
    lr_timescale: int | None = None
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5
    eval_max_batches: int | None = None
    keep_best_checkpoint: bool = True
    log_per_mode_loss: bool = True

    def __post_init__(self) -> None:
        positive = {
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise TrainingConfigError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        if self.learning_rate <= 0.0:
            raise TrainingConfigError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.lr_scheduler_type not in SCHEDULES:
            raise TrainingConfigError(
                f"lr_scheduler_type must be one of {list(SCHEDULES)}, got "
                f"{self.lr_scheduler_type!r}. The names are "
                f"transformers.SchedulerType values."
            )

        for name, value in (
            ("weight_decay", self.weight_decay),
            ("warmup_ratio", self.warmup_ratio),
            ("label_smoothing_factor", self.label_smoothing_factor),
            ("min_lr_ratio", self.min_lr_ratio),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise TrainingConfigError(f"{name} must be in [0.0, 1.0), got {value!r}")

        if self.warmup_steps < 0:
            raise TrainingConfigError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.warmup_steps and self.warmup_steps >= self.max_steps:
            raise TrainingConfigError(
                f"warmup_steps ({self.warmup_steps}) must be less than max_steps "
                f"({self.max_steps}), or the run is entirely warmup and the "
                f"learning rate never reaches its peak"
            )

        for name, value in (("adam_beta1", self.adam_beta1), ("adam_beta2", self.adam_beta2)):
            if not 0.0 <= float(value) < 1.0:
                raise TrainingConfigError(f"{name} must be in [0.0, 1.0), got {value!r}")
        if self.adam_epsilon <= 0.0:
            raise TrainingConfigError(
                f"adam_epsilon must be positive, got {self.adam_epsilon}"
            )
        if self.max_grad_norm <= 0.0:
            raise TrainingConfigError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}. "
                f"Clipping is not optional here: a single bad batch in a "
                f"multi-week run can otherwise destroy weeks of progress in one "
                f"step."
            )

        if self.fp16:
            raise TrainingConfigError(
                "fp16 is not supported. bf16 has the same memory saving with a "
                "float32 exponent range, so it needs no loss scaler and cannot "
                "silently stall on overflow — which is the failure fp16 produces "
                "in long runs. The field exists only so a TrainingArguments "
                "export is complete."
            )

        if self.dropout_rate is not None and not 0.0 <= float(self.dropout_rate) < 1.0:
            raise TrainingConfigError(
                f"dropout_rate must be in [0.0, 1.0) or absent, got {self.dropout_rate}"
            )

        if self.save_total_limit is not None and self.save_total_limit < 1:
            raise TrainingConfigError(
                f"save_total_limit must be positive or absent, got "
                f"{self.save_total_limit}"
            )

        if self.eval_max_batches is not None and self.eval_max_batches < 1:
            raise TrainingConfigError(
                f"eval_max_batches must be positive or absent, got "
                f"{self.eval_max_batches}"
            )

        if self.lr_timescale is not None and self.lr_timescale < 1:
            raise TrainingConfigError(
                f"lr_timescale must be positive or absent, got {self.lr_timescale}"
            )

        if self.num_cycles <= 0.0:
            raise TrainingConfigError(
                f"num_cycles must be positive, got {self.num_cycles}"
            )

        if self.load_best_model_at_end and not self.keep_best_checkpoint:
            raise TrainingConfigError(
                "load_best_model_at_end needs keep_best_checkpoint: the best "
                "checkpoint cannot be loaded at the end if retention was allowed "
                "to delete it"
            )

    # -- derived ----------------------------------------------------------

    @property
    def resolved_warmup_steps(self) -> int:
        """`warmup_steps` if set, else `warmup_ratio` of `max_steps`."""
        return resolve_warmup_steps(self.warmup_steps, self.warmup_ratio, self.max_steps)

    @property
    def schedule_arguments(self) -> dict[str, Any]:
        """Keyword arguments for `schedule.learning_rate_multiplier`."""
        return {
            "schedule": self.lr_scheduler_type,
            "warmup_steps": self.resolved_warmup_steps,
            "max_steps": self.max_steps,
            "timescale": self.lr_timescale,
            "num_cycles": self.num_cycles,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def learning_rate_at(self, step: int) -> float:
        """The learning rate this run uses at `step`."""
        return learning_rate_at(step, self.learning_rate, **self.schedule_arguments)

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, for run manifests and checkpoint metadata."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        """Inverse of `to_dict`. Unknown keys are rejected, never ignored."""
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TrainingConfigError(
                f"unknown training configuration keys: {sorted(unknown)}"
            )
        optional = {field.name for field in fields(cls) if field.default is not MISSING}
        missing = {name for name in known if name not in data} - optional
        if missing:
            raise TrainingConfigError(
                f"missing training configuration keys: {sorted(missing)}"
            )
        return cls(**data)

    def with_(self, **changes: Any) -> "TrainingConfig":
        """Copy with overrides; re-runs all validation."""
        return replace(self, **changes)

    # -- HuggingFace interoperability -------------------------------------

    def to_huggingface_training_arguments(self) -> dict[str, Any]:
        """
        Arguments for `transformers.TrainingArguments`, for continuing this
        run under HuggingFace's `Trainer`.

            from transformers import Seq2SeqTrainingArguments
            args = Seq2SeqTrainingArguments(
                **config.to_huggingface_training_arguments()
            )

        Every key is a real `TrainingArguments` field with the same meaning,
        so the continuation starts from the settings the checkpoint was
        actually trained under. The fields in `OURS_ALONE` are omitted rather
        than renamed — `TrainingArguments` would reject them, and inventing a
        near-equivalent would be worse than leaving the user to set it.
        """
        return {
            "output_dir": self.output_dir,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": self.resolved_warmup_steps,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.bf16,
            "label_smoothing_factor": self.label_smoothing_factor,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "load_best_model_at_end": self.load_best_model_at_end,
            # `warmup_steps` above already resolved the ratio, so passing the
            # ratio too would be a second, conflicting specification.
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "logging_strategy": "steps",
        }


def load_training_config(
    path: str | Path | None = None,
    stage: str | None = None,
) -> TrainingConfig:
    """
    Load a stage from `training_config.yml`.

    `path` defaults to the project-root file; `stage` defaults to the file's
    own `active_stage`, so a pipeline that passes neither gets exactly what
    the YAML says it should.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise TrainingConfigError(f"training configuration not found: {path}")

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TrainingConfigError(
            f"training configuration {path} is not valid YAML: {exc}"
        ) from exc

    if not isinstance(document, dict):
        raise TrainingConfigError(
            f"training configuration {path} must be a YAML mapping"
        )

    return build_training_config(document, stage=stage, source=str(path))


def build_training_config(
    document: Mapping[str, Any],
    stage: str | None = None,
    source: str = "<mapping>",
) -> TrainingConfig:
    """
    Build a `TrainingConfig` from an already-parsed document.

    Separate from `load_training_config` so the merge logic can be tested
    without touching the filesystem — the same split `model/src/config.py`
    uses, for the same reason.
    """
    defaults = document.get("defaults")
    if not isinstance(defaults, dict):
        raise TrainingConfigError(f"{source}: missing or malformed 'defaults' block")

    stages = document.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise TrainingConfigError(f"{source}: missing or empty 'stages' block")

    name = stage if stage is not None else document.get("active_stage")
    if name is None:
        raise TrainingConfigError(
            f"{source}: no stage requested and no 'active_stage' declared"
        )
    if name not in stages:
        raise TrainingConfigError(
            f"{source}: unknown stage {name!r}; available: {sorted(stages)}"
        )

    selected = stages[name]
    if not isinstance(selected, dict):
        raise TrainingConfigError(f"{source}: stage {name!r} must be a mapping")

    unknown = set(selected) - _STAGE_FIELDS
    if unknown:
        raise TrainingConfigError(
            f"{source}: stage {name!r} has unknown keys {sorted(unknown)}. A "
            f"silently-ignored key means a run that is not the one intended; put "
            f"shared settings in 'defaults'."
        )

    merged: dict[str, Any] = {**defaults, **selected, "stage": name}

    try:
        return TrainingConfig.from_dict(merged)
    except TrainingConfigError as exc:
        raise TrainingConfigError(f"{source}: stage {name!r}: {exc}") from exc


def available_stages(path: str | Path | None = None) -> list[str]:
    """Stage names declared in the configuration file."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise TrainingConfigError(f"training configuration not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return sorted((document or {}).get("stages", {}))


__all__ = [
    "TrainingConfig",
    "load_training_config",
    "build_training_config",
    "available_stages",
    "DEFAULT_CONFIG_PATH",
    "OURS_ALONE",
]
