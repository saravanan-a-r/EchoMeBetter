"""
The training loop (architecture.md §7, §13).

Scope
-----
This drives a model through a stream of already-built batches. It does not
build batches — `UL2/` does that — and it does not read a corpus. It consumes
exactly what `UL2/src/batching.pad_batch` produces, wrapped in tensors, which
is the same contract `model/` was written against, so nothing translates
between the three modules.

The four things worth reading before changing anything
------------------------------------------------------

**1. Accumulation is weighted by tokens, not by micro-batch count.**

The usual pattern is `(loss / accumulation_steps).backward()`. It is exactly
right only when every micro-batch contains the same number of target tokens,
because the model's loss is a *mean over tokens*, and averaging means of
different denominators is not the mean of the whole.

UL2 batches do not have equal token counts — [R] targets are short, [S]
targets are long, and the mixture is sampled per example — so the usual
pattern systematically over-weights the short-target micro-batches. Instead,
each micro-batch contributes `loss * its_token_count`, and all gradients are
divided by the run's total token count once at the end of the accumulation
window. The result is *identical* to running the whole window as one batch —
`test_accumulation_matches_one_big_batch` proves it with deliberately unequal
micro-batches, where the conventional version fails.

**2. Gradients are clipped after normalization, before the step.**

Order matters: clipping a sum of un-normalized gradients would clip against a
threshold that depends on the accumulation count, so changing accumulation
(a memory decision) would silently change the clipping (a stability decision).

**3. A non-finite gradient skips the step, it does not abort the run.**

One bad batch in a multi-week run should not end the run — but it must not be
*applied* either, because AdamW's moments would carry the NaN forward into
every subsequent step and the run would never recover. The step is dropped,
the gradients cleared, and the count logged, so a rising skip rate is visible
rather than being an unexplained plateau.

**4. `step` counts optimizer steps.**

Not micro-batches. The learning-rate schedule, the eval cadence and the save
cadence are all in these units, which is what makes them comparable between
runs that used different accumulation to reach the same effective batch size.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from .checkpoint import (
    Resumable,
    TrainerState,
    checkpoint_directory,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from .config import TrainingConfig
from .errors import TrainingConfigError
from .metrics import LossTracker, format_summary, per_example_losses
from .optimizer import build_optimizer, scale_gradients, set_learning_rate
from .schedule import learning_rate_at

# The keys every batch must carry. Exactly `pad_batch`'s output plus nothing:
# a missing key is caught here, once, rather than as a confusing shape error
# several frames into the model.
REQUIRED_KEYS = (
    "encoder_input_ids",
    "decoder_input_ids",
    "encoder_attention_mask",
    "decoder_attention_mask",
    "labels",
)

# Keys a batch *may* carry, moved to the device when present and ignored when
# absent. `encoder_segment_ids` appears only when the batch holds packed
# windows (`UL2/src/packing.py`); a batch without it takes exactly the path it
# took before packing existed, which is what keeps packing an addition rather
# than a change to every existing caller.
OPTIONAL_KEYS = ("encoder_segment_ids",)

Batch = Mapping[str, Any]


@dataclass(frozen=True)
class EvalTier:
    """
    One held-out evaluation set, with its own cadence and its own authority.

    A run asks two different questions of held-out data, and they do not want
    the same answer shape. "Is the loss still falling?" wants an answer often,
    and tolerates noise because it is read as a trend across many points.
    "Which checkpoint is the best one?" is asked rarely and answered once, and
    a noisy answer to it is *worse than no answer*: eval loss on a small
    sample has a standard error that shrinks only as 1/sqrt(n), so a thin
    validation set routinely ranks a worse checkpoint above a better one by an
    amount smaller than its own measurement error, and the run then keeps the
    wrong weights.

    Separating the two is what lets each be sized for its own job — a small,
    cheap set read every `save_steps`, and a large, precise one read rarely
    and given sole authority over `best_step`. `drives_best_checkpoint` is the
    authority, and exactly one tier may hold it: two tiers both writing
    `best_metric` would compare numbers computed over different data, where
    "better" means nothing.
    """

    name: str
    evaluate: Callable[[], Mapping[str, float]]
    every_steps: int
    drives_best_checkpoint: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise TrainingConfigError("an eval tier needs a name; it labels its log entries")
        if (
            not isinstance(self.every_steps, int)
            or isinstance(self.every_steps, bool)
            or self.every_steps < 1
        ):
            raise TrainingConfigError(
                f"eval tier {self.name!r}: every_steps must be a positive integer, "
                f"got {self.every_steps!r}"
            )


@dataclass
class StepReport:
    """What one optimizer step did, for logging and for tests to assert on."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens: int
    skipped: bool = False

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.loss) and math.isfinite(self.grad_norm)


class Trainer:
    """
    Drives a `RephraseSeq2Seq` through a stream of batches.

    Constructed with everything it needs and nothing it can discover, so a
    test can supply a list of five batches and a real run can supply a
    resumable corpus iterator, through the same interface.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
        on_log: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device) if device is not None else _default_device()
        self.optimizer = optimizer if optimizer is not None else build_optimizer(model, config)
        self.on_log = on_log
        self.state = TrainerState()
        self.skipped_steps = 0

        self._check_loss_settings_agree()

        self.model.to(self.device)
        if config.gradient_checkpointing:
            if not hasattr(model, "enable_gradient_checkpointing"):
                raise TrainingConfigError(
                    "gradient_checkpointing is on but the model does not support it"
                )
            model.enable_gradient_checkpointing()

        # `self.model` stays the uncompiled module for everything that is not
        # a forward pass: checkpointing (`rephrase_model.save_pretrained` /
        # `load_huggingface_state_dict` build their state dicts from
        # `named_parameters()`), `.train()`/`.eval()`, gradient clipping, and
        # the `getattr(self.model, "config", ...)` lookups below. That matters
        # because `torch.compile` returns a wrapper whose own `state_dict()`
        # prefixes every key with `_orig_mod.` — feeding that wrapper to the
        # checkpoint code would silently save/load the wrong key names (a
        # `strict=False` load absorbs the mismatch without erroring, so the
        # model would keep its random init and nothing would say why). Only
        # `_forward_model`, used solely inside `accumulate`/`evaluate`/
        # `_track` to run the model, is ever compiled; `.train()`/`.eval()`
        # called on `self.model` still governs it correctly, since compiling
        # wraps the same module instance rather than copying it.
        self._forward_model = self.model
        if config.torch_compile:
            compile_kwargs: dict[str, Any] = {}
            if config.torch_compile_backend is not None:
                compile_kwargs["backend"] = config.torch_compile_backend
            if config.torch_compile_mode is not None:
                compile_kwargs["mode"] = config.torch_compile_mode
            self._forward_model = torch.compile(self.model, **compile_kwargs)

        torch.manual_seed(config.seed)

    # -- consistency ------------------------------------------------------

    def _check_loss_settings_agree(self) -> None:
        """
        Refuse a run whose two label-smoothing settings disagree.

        The model owns the loss, so `ModelConfig.label_smoothing` is what
        actually applies; `TrainingConfig.label_smoothing_factor` exists
        because that is HuggingFace's name for the same number and a
        `TrainingArguments` export needs it. Two places holding one value is a
        compromise, and the compromise is only safe if a disagreement is
        impossible — a run silently trained with smoothing 0.0 while its
        recorded configuration says 0.1 is a result nobody can reproduce.
        """
        model_config = getattr(self.model, "config", None)
        declared = getattr(model_config, "label_smoothing", None)
        if declared is None:
            return
        if float(declared) != float(self.config.label_smoothing_factor):
            raise TrainingConfigError(
                f"label smoothing disagrees between the two configurations: "
                f"model_config.yml says label_smoothing={declared}, "
                f"training_config.yml says "
                f"label_smoothing_factor={self.config.label_smoothing_factor}. "
                f"The model computes the loss, so its value is the one that would "
                f"apply and the run's recorded settings would be wrong. Change "
                f"whichever is stale."
            )

    # -- one step ---------------------------------------------------------

    def accumulate(self, micro_batches: Sequence[Batch]) -> tuple[float, float]:
        """
        Run one accumulation window and leave normalized gradients in place.

        Returns `(mean loss, total tokens)`. Split out from `train_step` so
        the gradients it produces can be inspected before clipping and before
        `zero_grad` erases them — `test_accumulation_matches_one_big_batch`
        compares them tensor-for-tensor against a single large batch, and
        there is no way to do that through a method that steps and clears.

        On return the gradients are exactly those of one batch containing
        every token in the window. See the module docstring for why that is
        not the same as dividing each micro-batch's loss by the window size.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_tokens = 0.0
        total_loss = 0.0
        for micro_batch in micro_batches:
            batch = self._to_device(micro_batch)
            with self._autocast():
                output = self._forward_model(
                    encoder_input_ids=batch["encoder_input_ids"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    encoder_attention_mask=batch["encoder_attention_mask"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                    encoder_segment_ids=batch.get("encoder_segment_ids"),
                )
            tokens = float(
                (batch["labels"] != self._label_pad_token_id()).sum().item()
            )
            if tokens == 0:
                # Every position masked out. Backward would produce NaN from
                # a 0/0 mean, so the micro-batch contributes nothing instead.
                continue

            # Scaled by tokens here and divided by the total below, which is
            # what makes the window equal to one large batch exactly.
            (output.loss.float() * tokens).backward()
            total_loss += float(output.loss.detach()) * tokens
            total_tokens += tokens

        if total_tokens == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return float("nan"), 0.0

        scale_gradients(self.model.parameters(), 1.0 / total_tokens)
        return total_loss / total_tokens, total_tokens

    def train_step(self, micro_batches: Sequence[Batch]) -> StepReport:
        """
        One optimizer step from `gradient_accumulation_steps` micro-batches.

        Returns before the step number is advanced by `train`, so a caller
        driving steps by hand sees the step that was just performed.
        """
        loss, total_tokens = self.accumulate(micro_batches)

        step = self.state.global_step
        learning_rate = learning_rate_at(
            step, self.config.learning_rate, **self.config.schedule_arguments
        )

        if total_tokens == 0:
            self.skipped_steps += 1
            return StepReport(step, float("nan"), learning_rate, 0.0, 0, skipped=True)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        )

        if not math.isfinite(grad_norm) or not math.isfinite(loss):
            # Dropped, not applied: a NaN reaching AdamW's moments poisons
            # every subsequent step and the run never recovers.
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            return StepReport(
                step, loss, learning_rate, grad_norm, int(total_tokens), skipped=True
            )

        set_learning_rate(self.optimizer, learning_rate)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return StepReport(step, loss, learning_rate, grad_norm, int(total_tokens))

    # -- the run ----------------------------------------------------------

    def train(
        self,
        batches: Iterable[Batch],
        *,
        evaluate: Callable[[], Mapping[str, float]] | None = None,
        eval_tiers: Sequence[EvalTier] | None = None,
        data: Resumable | None = None,
        max_steps: int | None = None,
    ) -> TrainerState:
        """
        Run until `max_steps` or until the batch stream is exhausted.

        `evaluate` is a callable rather than a dataset so this module never
        has to know what evaluation means — checkpoint diagnostics (§7.7) are
        more than a loss, and the pieces that compute them do not exist yet.
        Whatever it returns is recorded in `log_history` and its
        `metric_for_best_model` entry drives best-checkpoint retention.

        `eval_tiers` is the general form: several held-out sets, each on its
        own cadence, exactly one of them holding authority over `best_step`
        (see `EvalTier`). `evaluate=` is the one-tier shorthand and is exactly
        equivalent to a single tier named "eval", running every `eval_steps`,
        that drives best-checkpoint selection — so every existing caller keeps
        its current behaviour unchanged. Passing both is refused rather than
        merged: which one ranked the checkpoints would be a coin flip.

        `data` is the resumable data pipeline. Beyond being checkpointed, it
        is told which step is about to run if it offers a `set_step` method
        (see `_announce_step`) — which is how the WSD cooldown's corpus
        mixture switches at the right step without this module knowing what a
        corpus mixture is.

        Exhausting the stream early is not an error: a rehearsal run over a
        fixed slice (§7.8) is expected to end that way. It is reported in the
        final log entry so it cannot be mistaken for having reached
        `max_steps`.
        """
        tiers = self._resolve_eval_tiers(evaluate, eval_tiers)
        limit = max_steps if max_steps is not None else self.config.max_steps
        source = _micro_batches(batches, self.config.gradient_accumulation_steps)
        window = LossTracker()
        started = time.monotonic()
        exhausted = False

        while self.state.global_step < limit:
            self._announce_step(data, self.state.global_step)
            try:
                micro_batches = next(source)
            except StopIteration:
                exhausted = True
                break

            report = self.train_step(micro_batches)
            self.state.global_step += 1
            self.state.tokens_seen += report.tokens

            if self.config.log_per_mode_loss and not report.skipped:
                self._track(window, micro_batches)
            elif not report.skipped:
                window.overall.add(report.loss * report.tokens, report.tokens)

            step = self.state.global_step
            if step % self.config.logging_steps == 0:
                self._log(
                    {
                        "step": step,
                        **window.summary(),
                        "learning_rate": report.learning_rate,
                        "grad_norm": report.grad_norm,
                        "tokens_seen": self.state.tokens_seen,
                        "skipped_steps": self.skipped_steps,
                        "seconds": round(time.monotonic() - started, 2),
                    }
                )
                window.reset()

            for tier in tiers:
                if step % tier.every_steps == 0:
                    self._evaluate_and_record(tier, step)

            if step % self.config.save_steps == 0:
                self.save(step, data=data)

        self._log(
            {
                "step": self.state.global_step,
                "finished": True,
                "reason": "exhausted" if exhausted else "max_steps",
                "tokens_seen": self.state.tokens_seen,
                "skipped_steps": self.skipped_steps,
            }
        )
        return self.state

    # -- evaluation -------------------------------------------------------

    def evaluate(
        self,
        batches: Iterable[Batch],
        *,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """
        Token-weighted loss over `batches`, reported per UL2 mode (§7.6).

        `torch.no_grad` and `model.eval()`, so dropout is off — a validation
        loss measured with dropout on is both higher than the real one and
        noisier, which is exactly wrong for comparing checkpoints.

        The batches must be the *frozen* validation set §7.6 requires: fixed
        examples with fixed corruption. Re-randomizing corruption at each
        evaluation adds sampling noise on the same order as the differences
        between adjacent checkpoints, which is what the whole diagnostic is
        trying to see.
        """
        limit = max_batches if max_batches is not None else self.config.eval_max_batches
        was_training = self.model.training
        self.model.eval()
        tracker = LossTracker()

        try:
            with torch.no_grad():
                for index, batch in enumerate(batches):
                    if limit is not None and index >= limit:
                        break
                    prepared = self._to_device(batch)
                    with self._autocast():
                        output = self._forward_model(
                            encoder_input_ids=prepared["encoder_input_ids"],
                            decoder_input_ids=prepared["decoder_input_ids"],
                            encoder_attention_mask=prepared["encoder_attention_mask"],
                            decoder_attention_mask=prepared["decoder_attention_mask"],
                            encoder_segment_ids=prepared.get("encoder_segment_ids"),
                        )
                    totals, counts = per_example_losses(
                        output.logits,
                        prepared["labels"],
                        ignore_index=self._label_pad_token_id(),
                        label_smoothing=self.config.label_smoothing_factor,
                    )
                    tracker.update(totals, counts, batch.get("modes"))
        finally:
            self.model.train(was_training)

        return tracker.summary()

    def _resolve_eval_tiers(
        self,
        evaluate: Callable[[], Mapping[str, float]] | None,
        eval_tiers: Sequence[EvalTier] | None,
    ) -> tuple[EvalTier, ...]:
        """
        Normalize the two ways of asking for evaluation into one list.

        The checks here all guard the same thing: that `best_step` has exactly
        one, well-defined meaning. A run whose "best checkpoint" was chosen by
        whichever tier happened to run last, or which names a step no
        checkpoint was written at, has a best checkpoint on paper and not on
        disk — and both failures surface at the very end of a multi-week run,
        which is the worst possible time to discover them.
        """
        if evaluate is not None and eval_tiers:
            raise TrainingConfigError(
                "pass either evaluate= or eval_tiers=, not both: they would each "
                "claim authority over best-checkpoint selection and which one won "
                "would depend on argument order"
            )

        if not eval_tiers:
            if evaluate is None:
                return ()
            return (
                EvalTier(
                    "eval",
                    evaluate,
                    self.config.eval_steps,
                    drives_best_checkpoint=True,
                ),
            )

        tiers = tuple(eval_tiers)
        names = [tier.name for tier in tiers]
        if len(set(names)) != len(names):
            raise TrainingConfigError(
                f"eval tier names must be unique — they label the log entries that "
                f"separate one tier's curve from another's; got {names}"
            )

        drivers = [tier for tier in tiers if tier.drives_best_checkpoint]
        if len(drivers) > 1:
            raise TrainingConfigError(
                f"only one eval tier may drive best-checkpoint selection, got "
                f"{[tier.name for tier in drivers]}. Two tiers writing best_metric "
                f"would compare losses measured over different held-out sets, where "
                f"'better' does not mean anything."
            )

        for tier in drivers:
            if tier.every_steps % self.config.save_steps != 0:
                raise TrainingConfigError(
                    f"eval tier {tier.name!r} drives best-checkpoint selection but "
                    f"its cadence ({tier.every_steps}) is not a multiple of "
                    f"save_steps ({self.config.save_steps}), so the step it names "
                    f"as best is a step no checkpoint was written at — retention "
                    f"would protect nothing and load_best_model_at_end would have "
                    f"nothing to load"
                )

        return tiers

    def _evaluate_and_record(self, tier: EvalTier, step: int) -> None:
        metrics = dict(tier.evaluate())
        self._log({"step": step, "eval": True, "eval_tier": tier.name, **metrics})

        if not tier.drives_best_checkpoint:
            return

        key = self.config.metric_for_best_model
        if key not in metrics:
            return
        value = float(metrics[key])
        if not math.isfinite(value):
            return
        if self.state.best_metric is None or _is_better(
            value, self.state.best_metric, self.config.greater_is_better
        ):
            self.state.best_metric = value
            self.state.best_step = step

    # -- checkpoints ------------------------------------------------------

    def save(self, step: int, *, data: Resumable | None = None) -> Path:
        """
        Write a checkpoint and apply the retention policy.

        Retention runs *after* the write, and protects `best_step`, so the
        best checkpoint in the run survives a `save_total_limit` that would
        otherwise age it out (§13: "keep best-by-eval, not last").
        """
        directory = save_checkpoint(
            checkpoint_directory(self.config.output_dir, step),
            model=self.model,
            optimizer=self.optimizer,
            state=self.state,
            training_config=self.config,
            data=data,
        )
        prune_checkpoints(
            self.config.output_dir,
            self.config.save_total_limit,
            protect_step=self.state.best_step if self.config.keep_best_checkpoint else None,
        )
        return directory

    def resume(
        self,
        directory: str | Path | None = None,
        *,
        data: Resumable | None = None,
    ) -> TrainerState:
        """
        Restore the run from a checkpoint, defaulting to the latest.

        Resuming a run whose output directory holds no checkpoint is an
        error, not a fresh start. A crash-loop that keeps "resuming" from
        step 0 burns the same budget forever and its loss curve looks
        plausible every time.
        """
        target = Path(directory) if directory is not None else latest_checkpoint(
            self.config.output_dir
        )
        if target is None:
            from .errors import CheckpointError

            raise CheckpointError(
                f"no checkpoint to resume from in {self.config.output_dir}. Start "
                f"the run with resume_from_checkpoint disabled if that is intended "
                f"— silently starting from step 0 would repeat the whole run."
            )
        self.state = load_checkpoint(
            target, model=self.model, optimizer=self.optimizer, data=data
        )
        self.model.to(self.device)
        return self.state

    # -- plumbing ---------------------------------------------------------

    def _track(self, tracker: LossTracker, micro_batches: Sequence[Batch]) -> None:
        """
        Per-mode training loss, recomputed from the batches just stepped on.

        A second forward pass, under `no_grad`, because the training pass
        already discarded its logits — keeping them alive to avoid this would
        hold a `(batch, length, 33728)` tensor through the backward pass,
        which is the single largest activation in the model. Paying one
        inference pass per logging window is cheaper than that, and the window
        is `logging_steps` wide.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for micro_batch in micro_batches:
                    if not micro_batch.get("modes"):
                        continue
                    prepared = self._to_device(micro_batch)
                    with self._autocast():
                        output = self._forward_model(
                            encoder_input_ids=prepared["encoder_input_ids"],
                            decoder_input_ids=prepared["decoder_input_ids"],
                            encoder_attention_mask=prepared["encoder_attention_mask"],
                            decoder_attention_mask=prepared["decoder_attention_mask"],
                            encoder_segment_ids=prepared.get("encoder_segment_ids"),
                        )
                    totals, counts = per_example_losses(
                        output.logits,
                        prepared["labels"],
                        ignore_index=self._label_pad_token_id(),
                        label_smoothing=self.config.label_smoothing_factor,
                    )
                    tracker.update(totals, counts, micro_batch["modes"])
        finally:
            self.model.train(was_training)

    def _announce_step(self, data: Resumable | None, step: int) -> None:
        """
        Tell the data pipeline which step is about to be run, if it cares.

        One optional method, called once per optimizer step, *before* the
        window's micro-batches are pulled — so a pipeline that changes what it
        emits at a step boundary changes it for the whole window rather than
        halfway through one. This exists for the `warmup_stable_decay`
        cooldown (architecture_improvements.md item 2), where the corpus
        mixture switches at the step the decay phase begins.

        Duck-typed rather than required: `data` is already optional, most
        pipelines have nothing to do with the step number, and every existing
        caller — including the plain lists the tests use — keeps working
        untouched. The trainer deliberately does not interpret the call at
        all; *what* changes at a step boundary is a data decision, and this
        module has no business holding an opinion about corpus mixtures.

        Note the asymmetry with `learning_rate_at`: the schedule reads the
        step, while the pipeline is told it. The pipeline is stateful (it has
        already read documents, it holds a pool), so it needs a single,
        ordered notification rather than the freedom to ask at any moment.
        """
        if data is None:
            return
        announce = getattr(data, "set_step", None)
        if callable(announce):
            announce(step)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.bf16 and self.device.type in ("cuda", "cpu"),
        )

    def _to_device(self, batch: Batch) -> dict[str, torch.Tensor]:
        missing = [key for key in REQUIRED_KEYS if key not in batch]
        if missing:
            raise TrainingConfigError(
                f"batch is missing {missing}; expected the keys "
                f"UL2/src/batching.pad_batch produces"
            )
        # `non_blocking=True` is only safe (and only faster) on CUDA with
        # pinned host memory. On MPS it has been observed to return before
        # the copy completes: a later `.to()` call in the same comprehension
        # can start writing into the destination buffer of an earlier one
        # before it is done being read, silently replacing its values —
        # not a crash, a wrong tensor. Non-blocking is switched off for
        # every other backend so a batch is either transferred correctly or
        # not transferred; it is never observed half-done.
        non_blocking = self.device.type == "cuda"
        keys = [*REQUIRED_KEYS, *(key for key in OPTIONAL_KEYS if key in batch)]
        return {
            key: _as_tensor(batch[key]).to(self.device, non_blocking=non_blocking)
            for key in keys
        }

    def _label_pad_token_id(self) -> int:
        model_config = getattr(self.model, "config", None)
        return int(getattr(model_config, "label_pad_token_id", -100))

    def _log(self, record: Mapping[str, Any]) -> None:
        entry = dict(record)
        self.state.log_history.append(entry)
        if self.on_log is not None:
            self.on_log(entry)


# -- free helpers ----------------------------------------------------------


def _micro_batches(
    batches: Iterable[Batch], accumulation_steps: int
) -> Iterator[list[Batch]]:
    """
    Group a batch stream into accumulation windows.

    A trailing partial window is **discarded**, not stepped on. Stepping a
    short window is not wrong numerically — the token weighting handles it —
    but it makes the last step of every run a different size from the rest,
    which is exactly the sort of thing that makes two runs incomparable for no
    benefit. The stream is effectively infinite in a real run, so this only
    ever bites the final step of a fixed slice.
    """
    window: list[Batch] = []
    for batch in batches:
        window.append(batch)
        if len(window) == accumulation_steps:
            yield window
            window = []


def _as_tensor(value: Any) -> torch.Tensor:
    """
    Accept a tensor or the plain nested lists `pad_batch` returns.

    `UL2/` deliberately has no torch dependency (its module docstring says
    so), so the conversion has to happen somewhere; here is the boundary.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)


def _is_better(value: float, best: float, greater_is_better: bool) -> bool:
    return value > best if greater_is_better else value < best


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "Trainer",
    "StepReport",
    "EvalTier",
    "REQUIRED_KEYS",
    "format_summary",
]
