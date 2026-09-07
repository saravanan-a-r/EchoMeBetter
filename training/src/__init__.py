"""
The training loop: turning a stream of UL2 batches into a trained checkpoint.

Scope
-----
This package is the *trainer*. It owns the optimizer, the learning-rate
schedule, gradient accumulation, clipping, mixed precision, checkpointing and
resumption. It does not define the network (`model/`), does not build training
examples (`UL2/`), and does not read a corpus.

Configuration
-------------
Every hyperparameter comes from `training_config.yml` at the project root, in
the same way every architectural dimension comes from `model_config.yml`. The
split is by lifetime: `model_config.yml` holds what is welded into a
checkpoint's weight shapes, this holds what a later run may choose
differently.

HuggingFace
-----------
Field names are `transformers.TrainingArguments`'s wherever a counterpart
exists, schedule names are `SchedulerType`'s and the schedule arithmetic is
`transformers.optimization`'s — so a run continued under HuggingFace's
`Trainer` follows the same learning-rate curve rather than a similar one.
Every checkpoint this package writes is a directory
`T5ForConditionalGeneration.from_pretrained` can load directly, with no
export step.

Usage
-----
    from src import Trainer, load_training_config      # training/
    from src import build_model, load_model_config     # model/

    config = load_training_config()                    # reads active_stage
    model = build_model(load_model_config())           # verifies its own size

    trainer = Trainer(model, config)
    if config.resume_from_checkpoint:
        trainer.resume(data=corpus)
    trainer.train(batches, evaluate=lambda: trainer.evaluate(validation), data=corpus)
"""

from __future__ import annotations

from .checkpoint import (
    Resumable,
    TrainerState,
    checkpoint_directory,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    step_of,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    OURS_ALONE,
    TrainingConfig,
    available_stages,
    build_training_config,
    load_training_config,
)
from .errors import (
    CheckpointError,
    ScheduleError,
    TrainingConfigError,
    TrainingError,
)
from .loop import REQUIRED_KEYS, EvalTier, StepReport, Trainer
from .metrics import (
    MODES,
    LossTracker,
    MeanLoss,
    format_summary,
    per_example_losses,
    per_token_losses,
)
from .optimizer import (
    build_optimizer,
    current_learning_rate,
    decayed_and_exempt,
    parameter_groups,
    scale_gradients,
    set_learning_rate,
)
from .schedule import (
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

__all__ = [
    # configuration
    "TrainingConfig",
    "load_training_config",
    "build_training_config",
    "available_stages",
    "DEFAULT_CONFIG_PATH",
    "OURS_ALONE",
    # the loop
    "Trainer",
    "StepReport",
    "EvalTier",
    "REQUIRED_KEYS",
    # schedules
    "learning_rate_multiplier",
    "learning_rate_at",
    "resolve_warmup_steps",
    "SCHEDULES",
    "CONSTANT",
    "CONSTANT_WITH_WARMUP",
    "LINEAR",
    "COSINE",
    "INVERSE_SQRT",
    # optimizer
    "build_optimizer",
    "parameter_groups",
    "decayed_and_exempt",
    "set_learning_rate",
    "current_learning_rate",
    "scale_gradients",
    # metrics
    "LossTracker",
    "MeanLoss",
    "MODES",
    "per_token_losses",
    "per_example_losses",
    "format_summary",
    # checkpoints
    "TrainerState",
    "Resumable",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    "latest_checkpoint",
    "checkpoint_directory",
    "prune_checkpoints",
    "step_of",
    # errors
    "TrainingError",
    "TrainingConfigError",
    "CheckpointError",
    "ScheduleError",
]
