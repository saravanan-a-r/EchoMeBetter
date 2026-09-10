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
    torch_compile  torch_compile_backend  torch_compile_mode
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
from .schedule import (
    DECAY_TYPES,
    SCHEDULES,
    WARMUP_STABLE_DECAY,
    learning_rate_at,
    resolve_decay_steps,
    resolve_warmup_steps,
)

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
        # The `warmup_stable_decay` cooldown (item 2). Per-stage for the same
        # reason `lr_scheduler_type` is: pretraining anneals over its last 10%,
        # SFT uses cosine over its whole (much shorter) horizon, and the
        # rehearsal is too short to anneal meaningfully at all.
        "lr_decay_steps",
        "lr_decay_ratio",
        "lr_decay_type",
        "cooldown_blend",
        # The synthetic rewrite task (item 4) runs inside the same cooldown and
        # is per-stage for the same reason `cooldown_blend` is.
        "rewrite_share",
        "rewrite_blend",
        "dropout_rate",
        "label_smoothing_factor",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        # Capacity-based micro-batching, per-stage for the same reason the two
        # above are: the budgets bound memory, and how much a batch of a given
        # size costs depends on the model profile the stage runs. Values
        # measured for Large would be wrong for a rehearsal on a smaller one.
        "tokens_per_step",
        "max_batch_encoder_tokens",
        "max_batch_decoder_tokens",
        "max_batch_rows",
        "model_profile",
        "output_dir",
        # torch_compile is genuinely a per-stage choice, not a global one: the
        # pretrain stage runs for weeks on a fixed GPU and shape, exactly where
        # `torch.compile`'s startup recompilation cost pays for itself, while a
        # short, small-shape rehearsal (§7.8) — sometimes run on a different
        # card entirely — is exactly where forcing it on would be riskiest for
        # the least benefit.
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        # A short stage (the §7.8 rehearsal is 2,000 steps) would never reach a
        # 20,000-step master evaluation, so the cadences have to be settable
        # per stage or the rehearsal cannot rehearse the thing it exists to
        # rehearse.
        "quick_eval_steps",
        "master_eval_steps",
        # Same argument as the two cadences above: what a per-mode sample is
        # worth depends on how long the stage runs. A 2,000-step rehearsal
        # wants it often enough to see the curve at all; a 200,000-step
        # pretraining run does not, and pays for every extra sample.
        "per_mode_loss_steps",
    }
)

# `torch.compile`'s own accepted `mode=` values (it validates these itself,
# but only at the first forward call — see the `torch_compile_mode` check
# below for why that is too late to be useful).
TORCH_COMPILE_MODES = frozenset(
    {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
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
    "lr_decay_steps",
    "lr_decay_ratio",
    "lr_decay_type",
    "cooldown_blend",
    "rewrite_share",
    "rewrite_blend",
    "eval_max_batches",
    "keep_best_checkpoint",
    "log_per_mode_loss",
    "per_mode_loss_steps",
    "quick_eval_steps",
    "master_eval_steps",
    "tokens_per_step",
    "max_batch_encoder_tokens",
    "max_batch_decoder_tokens",
    "max_batch_rows",
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

    # Report [R]/[X]/[S] losses separately (§7.6). The split is a *diagnostic*,
    # not a training signal, and it is not free: it is measured by a second
    # forward pass over the whole accumulation window, which costs ~22% of a
    # step. `per_mode_loss_steps` is how often that pass runs.
    #
    # `None` means "every logging step", which is what the cadence was always
    # documented to be. An explicit value samples more rarely than the log
    # line and must be a multiple of `logging_steps`, so every sample lands on
    # a log line and the split always describes the step it is printed next
    # to. Overhead falls as 1/n: at `logging_steps` it is already under half a
    # percent of the run, and there is nothing left to save below that.
    log_per_mode_loss: bool = True
    per_mode_loss_steps: int | None = None

    # Capacity-based micro-batching (`UL2/src/length_batching.BatchCapacity`).
    #
    # `per_device_train_batch_size` puts a fixed number of *rows* in every
    # micro-batch. Rows here differ in length by more than a hundredfold, so
    # that fixes the row count and lets the work vary: narrow micro-batches
    # spend most of their time on kernel dispatch with the GPU idle, and peak
    # memory is set by the widest shape the corpus happens to hold rather than
    # by anything configured. The two token budgets below bound
    # `rows x padded width` instead — separately, because a decoder position
    # costs ~3.6x an encoder one in memory (cross-attention, and a 33,728-wide
    # logit row).
    #
    # `tokens_per_step` then closes the accumulation window on loss-bearing
    # tokens rather than on a count of micro-batches. That is what keeps the
    # effective batch size — the hyperparameter the learning rate is tuned
    # against — fixed while the grouping underneath it changes. Setting the
    # budgets without it would let the effective batch size drift with the
    # lengths the corpus served, which is a silent change to training.
    #
    # All absent: micro-batching behaves exactly as it did before, at
    # `per_device_train_batch_size` rows and `gradient_accumulation_steps`
    # micro-batches per step.
    tokens_per_step: int | None = None
    max_batch_encoder_tokens: int | None = None
    max_batch_decoder_tokens: int | None = None
    max_batch_rows: int | None = None

    # Cadences for the two-tier held-out evaluation (`EvalTier` in loop.py).
    # `quick_eval_steps` is the frequent, cheap trend signal; `master_eval_steps`
    # is the rare, precise one that alone decides which checkpoint is best. Only
    # the cadences live here: how many records each set holds is a property of
    # the eval files themselves, and duplicating it in configuration would
    # invite the two to disagree.
    quick_eval_steps: int = 1000
    master_eval_steps: int = 20000

    # -- compilation (item 3: SDPA + torch.compile) ------------------------
    # Shared with TrainingArguments (same names, same meaning, same defaults:
    # off unless asked for), but defaulted here — unlike gradient_checkpointing
    # and bf16 above — so every existing stage block and test fixture that
    # predates this feature keeps loading unchanged. `loop.py`'s `Trainer`
    # compiles only the forward path used for training/eval (see its own
    # comment on `_forward_model`); checkpointing always addresses the
    # uncompiled module, so a compiled run's checkpoint is byte-identical in
    # shape to an uncompiled one's.
    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = None

    # -- the WSD cooldown (item 2) ----------------------------------------
    # No `TrainingArguments` counterpart by name: HuggingFace hides all three
    # inside `lr_scheduler_kwargs`, the same way it hides `lr_timescale`
    # above. `to_huggingface_training_arguments` puts them back there, so a
    # continuation under `Trainer` gets the same three phases rather than a
    # schedule that warms up and then never comes down.
    #
    # `lr_decay_steps` wins over `lr_decay_ratio` when both are set, matching
    # the `warmup_steps` / `warmup_ratio` precedence a few fields up. Both
    # default to "no decay phase", which is the only correct default for the
    # five schedules that do not have one.
    lr_decay_steps: int = 0
    lr_decay_ratio: float = 0.0
    lr_decay_type: str = "cosine"  # transformers' get_wsd_schedule default

    # The other half of item 2, and the half that moves downstream eval most:
    # during the decay phase alone, sample the corpus from an upweighted
    # high-value mixture instead of reading it as it lies on disk. Maps a
    # corpus source (the directory name under the corpus root — `cli_tldr`,
    # `permissive_code`, ...) to a relative weight; weights are normalized
    # over the sources actually present, so a partial corpus still produces a
    # correctly proportioned mix rather than silently dropping the phase.
    #
    # `None` means "do not switch the mixture", which leaves a WSD run as a
    # pure learning-rate change. The two halves are deliberately separable:
    # the LR decay is a settled, low-risk gain, while the mixture switch is a
    # judgement about which data is worth repeating.
    cooldown_blend: Mapping[str, float] | None = None

    # -- the synthetic rewrite task (item 4) -------------------------------
    # Cooldown-only, like `cooldown_blend`, and mixed in one level above it:
    # during the decay phase, this share of the micro-batches is drawn from
    # the rewrite task (`rewrite/`) instead of the UL2 denoisers -- a share of
    # *examples*, in the same sense `mode_weights` are sampling probabilities
    # rather than token shares. The remaining batches keep following
    # `cooldown_blend`, whose relative weights therefore need no change.
    #
    # `rewrite_blend` names the sources the task corrupts, with relative
    # weights, exactly as `cooldown_blend` does. `rewrite_share: 0.0` turns
    # the task off; a blend with nothing to draw it is refused.
    rewrite_share: float = 0.0
    rewrite_blend: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        positive = {
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "quick_eval_steps": self.quick_eval_steps,
            "master_eval_steps": self.master_eval_steps,
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

        if self.torch_compile_mode is not None and self.torch_compile_mode not in TORCH_COMPILE_MODES:
            raise TrainingConfigError(
                f"torch_compile_mode must be one of {sorted(TORCH_COMPILE_MODES)} or "
                f"absent, got {self.torch_compile_mode!r}. `torch.compile` itself only "
                f"raises this at the first forward call, not when it is constructed — "
                f"catching it here means a typo fails before a single batch runs "
                f"rather than after the model, optimizer and data pipeline have all "
                f"already spun up."
            )

        self._check_per_mode_loss_cadence()
        self._check_batch_capacity()
        self._check_decay_phase()
        self._check_cooldown_blend()
        self._check_rewrite_task()

        if self.load_best_model_at_end and not self.keep_best_checkpoint:
            raise TrainingConfigError(
                "load_best_model_at_end needs keep_best_checkpoint: the best "
                "checkpoint cannot be loaded at the end if retention was allowed "
                "to delete it"
            )

    # -- capacity-based micro-batching, validated -------------------------

    def _check_batch_capacity(self) -> None:
        """
        Refuse a half-configured capacity.

        The three settings are one mechanism, and any subset of them is a
        misconfiguration with no useful meaning:

          - budgets without `tokens_per_step` would leave the window closing
            on a count of micro-batches whose sizes now vary, so the effective
            batch size would drift with the corpus. That is a silent change to
            training, which is the class of bug this file exists to refuse.
          - `tokens_per_step` without budgets would close windows on tokens
            while every micro-batch still holds a fixed row count. Harmless,
            but it means one of the two was forgotten, and guessing which is
            not this module's job.
        """
        budgets = {
            "max_batch_encoder_tokens": self.max_batch_encoder_tokens,
            "max_batch_decoder_tokens": self.max_batch_decoder_tokens,
        }
        named = {name: value for name, value in budgets.items() if value is not None}
        for name, value in ({**named, "max_batch_rows": self.max_batch_rows,
                             "tokens_per_step": self.tokens_per_step}).items():
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise TrainingConfigError(
                    f"{name} must be a positive integer or absent, got {value!r}"
                )
        if named and len(named) != len(budgets):
            missing = sorted(set(budgets) - set(named))
            raise TrainingConfigError(
                f"capacity-based micro-batching needs both token budgets; "
                f"{missing} {'is' if len(missing) == 1 else 'are'} absent. The encoder "
                f"and decoder rectangles are padded independently, so one budget "
                f"cannot bound the other."
            )
        if named and self.tokens_per_step is None:
            raise TrainingConfigError(
                "max_batch_encoder_tokens/max_batch_decoder_tokens need "
                "tokens_per_step: with variable-size micro-batches, a window of "
                "gradient_accumulation_steps of them holds a token count that "
                "drifts with the corpus, silently changing the effective batch size"
            )
        if self.tokens_per_step is not None and not named:
            raise TrainingConfigError(
                "tokens_per_step needs max_batch_encoder_tokens and "
                "max_batch_decoder_tokens: without them every micro-batch still "
                "holds a fixed number of rows, and closing the window on tokens "
                "instead of on a count changes nothing"
            )
        if self.max_batch_rows is not None and not named:
            raise TrainingConfigError(
                "max_batch_rows only bounds a capacity-based cut; it needs "
                "max_batch_encoder_tokens and max_batch_decoder_tokens"
            )

    # -- the per-mode diagnostic, validated -------------------------------

    def _check_per_mode_loss_cadence(self) -> None:
        """
        Refuse a per-mode cadence that would not line up with the log.

        The split is printed as part of a log line, so sampling it on a
        cadence that is not a multiple of `logging_steps` puts the two out of
        phase: the numbers still appear, but next to a *different* step's
        aggregate loss, and a reader comparing `loss` against `loss_X` on one
        line would be comparing two different moments in the run.
        """
        value = self.per_mode_loss_steps
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TrainingConfigError(
                f"per_mode_loss_steps must be a positive integer or absent, "
                f"got {value!r}"
            )
        if value % self.logging_steps != 0:
            raise TrainingConfigError(
                f"per_mode_loss_steps ({value}) must be a multiple of logging_steps "
                f"({self.logging_steps}): the per-mode split is printed on a log "
                f"line, so a cadence that does not land on one would report it "
                f"beside a different step's aggregate loss"
            )

    @property
    def resolved_per_mode_loss_steps(self) -> int:
        """
        How often the per-mode forward pass runs, in optimizer steps.

        `per_mode_loss_steps` if set, else every logging step — the cadence
        `log_per_mode_loss` was always documented to have.
        """
        return self.per_mode_loss_steps or self.logging_steps

    # -- the decay phase, validated ---------------------------------------

    def _check_decay_phase(self) -> None:
        """
        Refuse a decay phase that does not fit the run it belongs to.

        Every check here catches something that otherwise surfaces only as a
        loss curve nobody can explain: a cooldown longer than the run, one
        configured on a schedule with no cooldown to configure, or a WSD run
        with no cooldown at all — which is a constant-LR run wearing the name
        of a schedule chosen specifically for its ending.
        """
        if self.lr_decay_steps < 0:
            raise TrainingConfigError(
                f"lr_decay_steps must be non-negative, got {self.lr_decay_steps}"
            )
        if not 0.0 <= float(self.lr_decay_ratio) < 1.0:
            raise TrainingConfigError(
                f"lr_decay_ratio must be in [0.0, 1.0), got {self.lr_decay_ratio!r}"
            )
        if self.lr_decay_type not in DECAY_TYPES:
            raise TrainingConfigError(
                f"lr_decay_type must be one of {list(DECAY_TYPES)}, got "
                f"{self.lr_decay_type!r}. The names are transformers'"
                f" get_wsd_schedule(decay_type=...) values."
            )

        decay = self.resolved_decay_steps

        if self.lr_scheduler_type != WARMUP_STABLE_DECAY:
            if decay:
                raise TrainingConfigError(
                    f"lr_decay_steps/lr_decay_ratio configure the "
                    f"{WARMUP_STABLE_DECAY!r} cooldown, but lr_scheduler_type is "
                    f"{self.lr_scheduler_type!r}, which has no cooldown phase to "
                    f"give them to. They would be silently ignored — and a run "
                    f"whose recorded configuration describes an annealing phase "
                    f"that never happened is not reproducible from its manifest."
                )
            return

        if decay < 1:
            raise TrainingConfigError(
                f"lr_scheduler_type is {WARMUP_STABLE_DECAY!r} but no decay phase is "
                f"configured; set lr_decay_steps or lr_decay_ratio. Without one the "
                f"schedule is constant_with_warmup under a name that promises an "
                f"ending it will not deliver."
            )
        if self.resolved_warmup_steps + decay > self.max_steps:
            raise TrainingConfigError(
                f"warmup ({self.resolved_warmup_steps}) + decay ({decay}) exceeds "
                f"max_steps ({self.max_steps}): there is no stable phase between "
                f"them and the run would decay out of its own warmup"
            )

    def _check_cooldown_blend(self) -> None:
        """
        Refuse a cooldown mixture that cannot be applied as written.

        The mixture is switched at the moment the decay phase starts, so
        without a decay phase there is no moment to switch it at. Weights are
        relative and normalized at use, but a non-positive one is always a
        mistake: zero says "include this source and never draw from it", which
        is what leaving it out already means.
        """
        if self.cooldown_blend is None:
            return
        _check_blend("cooldown_blend", self.cooldown_blend)
        if self.lr_scheduler_type != WARMUP_STABLE_DECAY:
            raise TrainingConfigError(
                f"cooldown_blend switches the data mixture when the decay phase "
                f"begins, but lr_scheduler_type is {self.lr_scheduler_type!r}, which "
                f"has no decay phase — there is no step at which the switch would "
                f"happen, so the upweighted mixture would never be used"
            )

    def _check_rewrite_task(self) -> None:
        """
        Refuse a rewrite task that could not run as written.

        The share and the blend go together: a share with no sources to
        corrupt has nothing to draw, and a blend with a zero share would be
        read by nobody -- either one alone is a configuration that describes
        a task the run never performs. And like `cooldown_blend`, the task
        lives inside the decay phase, so a schedule without one has no step
        at which it would start.
        """
        share = self.rewrite_share
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise TrainingConfigError(f"rewrite_share must be a number, got {share!r}")
        if not 0.0 <= float(share) < 1.0:
            raise TrainingConfigError(
                f"rewrite_share must be in [0.0, 1.0), got {share!r}; it is the share "
                f"of cooldown micro-batches the task takes, and 1.0 would leave the "
                f"UL2 denoisers nothing"
            )
        if self.rewrite_blend is not None:
            _check_blend("rewrite_blend", self.rewrite_blend)
        if share > 0.0 and self.rewrite_blend is None:
            raise TrainingConfigError(
                f"rewrite_share is {share} but no rewrite_blend names the sources to "
                f"corrupt; the task would have nothing to draw from"
            )
        if share == 0.0 and self.rewrite_blend is not None:
            raise TrainingConfigError(
                "rewrite_blend is set but rewrite_share is 0, so it would never be "
                "read; set the share or drop the blend"
            )
        if share > 0.0 and self.lr_scheduler_type != WARMUP_STABLE_DECAY:
            raise TrainingConfigError(
                f"rewrite_share runs the rewrite task during the decay phase, but "
                f"lr_scheduler_type is {self.lr_scheduler_type!r}, which has no decay "
                f"phase — there is no step at which the task would start"
            )

    # -- derived ----------------------------------------------------------

    @property
    def resolved_decay_steps(self) -> int:
        """`lr_decay_steps` if set, else `lr_decay_ratio` of `max_steps`."""
        return resolve_decay_steps(
            self.lr_decay_steps, self.lr_decay_ratio, self.max_steps
        )

    @property
    def cooldown_start_step(self) -> int | None:
        """
        The step the decay phase — and the `cooldown_blend` switch — begins at.

        `None` for a schedule with no decay phase, so a caller cannot mistake
        "no cooldown" for "a cooldown starting at step 0".
        """
        if self.lr_scheduler_type != WARMUP_STABLE_DECAY:
            return None
        return self.max_steps - self.resolved_decay_steps

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
            "decay_steps": self.resolved_decay_steps or None,
            "decay_type": self.lr_decay_type,
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
        arguments: dict[str, Any] = {
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
            "torch_compile": self.torch_compile,
            "torch_compile_backend": self.torch_compile_backend,
            "torch_compile_mode": self.torch_compile_mode,
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

        # `warmup_stable_decay` is the one schedule whose shape is not fully
        # determined by the fields above: `transformers.get_wsd_schedule`
        # *requires* num_decay_steps and has no default for it, so an export
        # that omitted these would not merely drift from this run's curve — it
        # would fail to build a scheduler at all. Emitted only for that
        # schedule, because `lr_scheduler_kwargs` is passed straight to the
        # scheduler factory and the other five would reject keys they have no
        # parameter for.
        if self.lr_scheduler_type == WARMUP_STABLE_DECAY:
            arguments["lr_scheduler_kwargs"] = {
                "num_decay_steps": self.resolved_decay_steps,
                "decay_type": self.lr_decay_type,
                "min_lr_ratio": self.min_lr_ratio,
                "num_cycles": self.num_cycles,
            }

        return arguments


def _check_blend(name: str, blend: Any) -> None:
    """
    A source-to-weight mapping that can be applied as written.

    Weights are relative and normalized at use, but a non-positive one is
    always a mistake: zero says "include this source and never draw from it",
    which is what leaving it out already means.
    """
    if not isinstance(blend, Mapping) or not blend:
        raise TrainingConfigError(
            f"{name} must be a non-empty mapping of corpus source to weight, or "
            f"absent; got {blend!r}"
        )
    for source, weight in blend.items():
        if not isinstance(source, str) or not source:
            raise TrainingConfigError(f"{name} keys are corpus source names; got {source!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TrainingConfigError(f"{name}[{source!r}] must be a number, got {weight!r}")
        if float(weight) <= 0.0:
            raise TrainingConfigError(
                f"{name}[{source!r}] must be positive, got {weight!r}. Drop the "
                f"source instead of weighting it zero."
            )


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
    "TORCH_COMPILE_MODES",
]
