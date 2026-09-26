"""
The SFT loop.

One optimizer step consumes one step batch (`effective_batch_size` examples,
see `batching.py`), split into micro-batches for accumulation. The loss is
the **sum** of target-token NLL divided by the step's total target tokens, so
an accumulated step is exactly the loss of one big batch -- a micro-batch of
short targets does not count as much as one of long targets (tested).

Cadence
-------
  every `logging.every_steps`        train/*, perf/*, adapter/* scalars
  every `evaluation.every_steps`     eval/*, generation/*, quality/*, samples;
                                     best adapter saved when the selected
                                     metric improves; early stopping
  every `checkpointing.every_steps`  a resumable checkpoint (adapter +
                                     optimizer + scheduler + data position +
                                     RNG), last `keep_last` kept
  step 0                             a baseline evaluation of the untouched
                                     base (the adapter starts as a no-op), and
                                     the reference ceiling (reference/*)
  end                                best adapter re-loaded, final evaluation
                                     on the full eval split, `final/` written

Run directory
-------------
    runs/<run>/checkpoints/step-NNNNNN/   adapter + trainer_state.pt
    runs/<run>/best/                      adapter at the best evaluation
    runs/<run>/final/                     best adapter + final metrics
    runs/<run>/summary.json               data report, sizes, final metrics
    runs/<run>/config.json                the resolved configuration
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from . import evaluation
from .adapter_io import TENSORS_FILE, save_adapter
from .base_model import BaseModelInfo
from .batching import StepBatchPlan, StepBatchStream, collate, example_length, split_micro_batches
from .codec import TextCodec
from .config import SFTConfig
from .data import DatasetReport, SFTExample
from .errors import ResumeError, SFTConfigError, SFTError
from .lora import AttachedAdapter
from .resources import DevicePlan, memory_stats
from .run_logging import MetricsWriter

STATE_FILE = "trainer_state.pt"
MAX_CONSECUTIVE_NONFINITE = 5


@dataclass(frozen=True)
class RunPaths:
    name: str
    run_dir: Path
    log_dir: Path

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def best(self) -> Path:
        return self.run_dir / "best"

    @property
    def final(self) -> Path:
        return self.run_dir / "final"


def lr_multiplier(step: int, *, schedule: str, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """
    Factor on the peak learning rate for the update made at `step` (0-based):
    linear warmup to 1, then cosine/linear decay to `min_lr_ratio` at
    `total_steps`, or constant.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    span = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / span, 1.0)
    if schedule == "cosine":
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return 1.0 - (1.0 - min_lr_ratio) * progress


class SFTTrainer:
    def __init__(
        self,
        *,
        config: SFTConfig,
        model: torch.nn.Module,
        adapter: AttachedAdapter,
        base_info: BaseModelInfo,
        codec: TextCodec,
        train_examples: Sequence[SFTExample],
        eval_examples: Sequence[SFTExample],
        data_report: DatasetReport,
        plan: DevicePlan,
        paths: RunPaths,
        logger,
        writer: MetricsWriter,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.model = model
        self.adapter = adapter
        self.base_info = base_info
        self.codec = codec
        self.train_examples = list(train_examples)
        self.eval_examples = list(eval_examples)
        self.data_report = data_report
        self.plan = plan
        self.paths = paths
        self.logger = logger
        self.writer = writer
        self.clock = clock
        self.autocast = plan.autocast()

        t, e = config.training, config.evaluation
        if e.select_metric.startswith(("generation/", "quality/")) and e.generation_examples == 0:
            raise SFTConfigError(
                f"select_metric {e.select_metric!r} needs generation, but generation_examples is 0"
            )
        self.batch_plan = StepBatchPlan(
            [example_length(x) for x in self.train_examples],
            t.effective_batch_size,
            t.length_grouping_window,
            t.seed,
        )
        self.stream = StepBatchStream(self.batch_plan)
        self.total_steps = t.max_steps or math.ceil(t.epochs * self.batch_plan.steps_per_epoch)

        self.parameters = adapter.parameters()
        self.optimizer = torch.optim.AdamW(
            adapter.param_groups(t.learning_rate, t.weight_decay),
            lr=t.learning_rate,
            betas=(t.adam_beta1, t.adam_beta2),
            eps=t.adam_epsilon,
        )
        schedule = dict(
            schedule=t.lr_schedule,
            warmup_steps=t.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=t.min_lr_ratio,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: lr_multiplier(step, **schedule)
        )

        self.step = 0
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.bad_evaluations = 0
        self.skipped_steps = 0
        self._consecutive_nonfinite = 0
        self._window = _Window()
        self._reference_logged = False

    # -- public -------------------------------------------------------------

    def resume(self, checkpoint: Path) -> None:
        """Restore everything from a checkpoint written by `_save_checkpoint`."""
        from safetensors.torch import load_file

        state_path = checkpoint / STATE_FILE
        if not state_path.is_file():
            raise ResumeError(f"{checkpoint} has no {STATE_FILE}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)

        expected = self.config.resume_signature()
        differences = _diff(state["resume_signature"], expected)
        if differences:
            raise ResumeError(
                "the configuration differs from the one this run was started with, in: "
                + ", ".join(differences)
                + ". Resume with the original configuration, or start a new run."
            )
        if state["base_fingerprint"] != self.base_info.fingerprint:
            raise ResumeError("the base checkpoint's weights differ from the ones this run was trained on")
        if state["data_fingerprint"] != self.data_report.fingerprint:
            raise ResumeError("the dataset split differs from the one this run was trained on")

        self.adapter.load_state_dict(load_file(str(checkpoint / TENSORS_FILE)))
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.stream.load_state_dict(state["stream"])
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if self.plan.device.type == "cuda" and state["rng"].get("cuda") is not None:
            try:
                torch.cuda.set_rng_state(state["rng"]["cuda"], self.plan.device)
            except RuntimeError:
                self.logger.warning("could not restore the CUDA RNG state (different device); continuing")
        self.step = int(state["step"])
        self.best_metric = state["best_metric"]
        self.best_step = state["best_step"]
        self.bad_evaluations = int(state["bad_evaluations"])
        self.skipped_steps = int(state["skipped_steps"])
        self._reference_logged = True
        self.logger.info(
            "resumed from %s at step %d (epoch %.3f); best %s at step %s",
            checkpoint, self.step, self.stream.fractional_epoch, self.best_metric, self.best_step,
        )

    def train(self) -> dict:
        """Run to completion (or early stop / interrupt). Returns the summary."""
        cfg = self.config
        self.logger.info(
            "training %d steps (%d per epoch, effective batch %d = %d x %d accumulation), "
            "%d train / %d eval examples",
            self.total_steps, self.batch_plan.steps_per_epoch, cfg.training.effective_batch_size,
            cfg.training.batch_size, cfg.training.gradient_accumulation_steps,
            len(self.train_examples), len(self.eval_examples),
        )
        status = "completed"
        self.model.train()
        try:
            if self.step == 0:
                self._evaluate(label="baseline (untouched base, adapter is a no-op)")
            while self.step < self.total_steps:
                self._train_one_step()
                if self.step % cfg.logging.every_steps == 0:
                    self._log_training()
                if self.step % cfg.evaluation.every_steps == 0:
                    if self._evaluate_and_select():
                        status = "early_stopped"
                        break
                if self.step % cfg.checkpointing.every_steps == 0:
                    self._save_checkpoint()
        except KeyboardInterrupt:
            status = "interrupted"
            self.logger.warning("interrupted at step %d; saving a resumable checkpoint", self.step)
            self._save_checkpoint()
            self.writer.flush()
            return self._write_summary(status, final_metrics=None)

        if self.step % cfg.evaluation.every_steps != 0 and status == "completed":
            self._evaluate_and_select()
        if self.step % cfg.checkpointing.every_steps != 0:
            self._save_checkpoint()
        final_metrics = self._final_evaluation()
        return self._write_summary(status, final_metrics)

    # -- one step -----------------------------------------------------------

    def _train_one_step(self) -> None:
        t = self.config.training
        started = self.clock()
        indices = self.stream.next()
        micro_batches = [
            collate([self.train_examples[i] for i in micro], self.codec.frame)
            for micro in split_micro_batches(indices, t.batch_size)
        ]
        step_tokens = sum(b.target_tokens for b in micro_batches)

        nll_sum, correct = 0.0, 0
        for batch in micro_batches:
            batch = batch.to(self.plan.device)
            with self.autocast():
                output = self.model(
                    encoder_input_ids=batch.encoder_input_ids,
                    decoder_input_ids=batch.decoder_input_ids,
                    encoder_attention_mask=batch.encoder_attention_mask,
                    decoder_attention_mask=batch.decoder_attention_mask,
                )
            logits = output.logits.float().reshape(-1, output.logits.size(-1))
            labels = batch.labels.reshape(-1)
            loss_sum = F.cross_entropy(
                logits, labels, ignore_index=-100, reduction="sum", label_smoothing=t.label_smoothing
            )
            (loss_sum / step_tokens).backward()
            with torch.no_grad():
                if t.label_smoothing > 0:
                    nll_sum += float(F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum"))
                else:
                    nll_sum += float(loss_sum)
                mask = labels != -100
                correct += int((logits.argmax(dim=-1)[mask] == labels[mask]).sum())

        log_now = (self.step + 1) % self.config.logging.every_steps == 0
        group_norms = self.adapter.gradient_norms() if log_now else {}
        max_norm = t.max_grad_norm if t.max_grad_norm is not None else float("inf")
        total_norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, max_norm))

        if not math.isfinite(total_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            self._consecutive_nonfinite += 1
            self.logger.warning(
                "non-finite gradient norm at step %d; update skipped (%d consecutive)",
                self.step, self._consecutive_nonfinite,
            )
            if self._consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
                raise SFTError(
                    f"{MAX_CONSECUTIVE_NONFINITE} consecutive non-finite gradients; stopping "
                    f"rather than training on garbage. Lower the learning rate."
                )
            return
        self._consecutive_nonfinite = 0

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1

        self._window.add(
            nll_sum=nll_sum,
            tokens=step_tokens,
            correct=correct,
            examples=len(indices),
            real_tokens=sum(b.real_tokens for b in micro_batches),
            padded_tokens=sum(b.padded_tokens for b in micro_batches),
            grad_norm=total_norm,
            clipped=t.max_grad_norm is not None and total_norm > t.max_grad_norm,
            seconds=self.clock() - started,
            group_norms=group_norms,
        )

    def _log_training(self) -> None:
        w = self._window
        if not w.steps:
            return
        seconds_per_step = w.seconds / w.steps
        remaining = (self.total_steps - self.step) * seconds_per_step
        metrics = {
            "train/loss": w.nll_sum / max(w.tokens, 1),
            "train/token_accuracy": w.correct / max(w.tokens, 1),
            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
            "train/grad_norm": w.grad_norm / w.steps,
            "train/grad_norm_max": w.grad_norm_max,
            "train/clipped_fraction": w.clipped / w.steps,
            "train/skipped_steps": float(self.skipped_steps),
            "progress/epoch": self.stream.fractional_epoch,
            "progress/examples_seen": float(self.step * self.config.training.effective_batch_size),
            "perf/seconds_per_step": seconds_per_step,
            "perf/examples_per_second": w.examples / max(w.seconds, 1e-9),
            "perf/tokens_per_second": w.real_tokens / max(w.seconds, 1e-9),
            "perf/padding_efficiency": w.real_tokens / max(w.padded_tokens, 1),
            "perf/eta_hours": remaining / 3600,
        }
        metrics.update(memory_stats(self.plan.device))
        metrics.update({f"adapter/relative_delta/{k}": v for k, v in self.adapter.relative_delta_norms().items()})
        metrics.update({f"adapter/grad_norm/{k}": v for k, v in w.last_group_norms.items()})
        token_names = {v: k for k, v in self.codec.token_map.items()}
        metrics.update(
            {
                f"adapter/token_delta_norm/{token_names.get(token_id, token_id)}": norm
                for token_id, norm in self.adapter.token_delta_norms().items()
            }
        )
        self.writer.scalars(self.step, metrics)
        self.logger.info(
            "step %d/%d | loss %.4f | acc %.3f | lr %.2e | gnorm %.3f | %.1fs/step | %.0f tok/s | ETA %.1fh",
            self.step, self.total_steps, metrics["train/loss"], metrics["train/token_accuracy"],
            metrics["train/learning_rate"], metrics["train/grad_norm"], seconds_per_step,
            metrics["perf/tokens_per_second"], metrics["perf/eta_hours"],
        )
        self._window = _Window()

    # -- evaluation ---------------------------------------------------------

    def _evaluate(self, *, label: str = "", final: bool = False) -> dict[str, float]:
        e = self.config.evaluation
        started = self.clock()
        loss_count = (e.final_loss_examples or len(self.eval_examples)) if final else e.loss_examples
        metrics = evaluation.teacher_forced_metrics(
            self.model,
            self.eval_examples[:loss_count],
            self.codec,
            batch_size=self.config.training.batch_size * 2,
            device=self.plan.device,
            autocast=self.autocast,
            calibration_bins=e.calibration_bins,
        )
        if e.generation_examples:
            records = evaluation.generate(
                self.model,
                self.eval_examples[: e.generation_examples],
                self.codec,
                e.decoding,
                batch_size=e.generation_batch_size,
                device=self.plan.device,
                autocast=self.autocast,
            )
            metrics.update(evaluation.generation_metrics(records, e.acceptance))
            prefix = "final" if final else "generation"
            for tag, values in evaluation.distributions(records).items():
                self.writer.histogram(self.step, tag.replace("generation", prefix, 1), values)
            if e.text_samples:
                self.writer.text(self.step, f"{prefix}/samples", evaluation.samples_markdown(records, e.text_samples))
            if not self._reference_logged:
                reference = evaluation.generation_metrics(
                    records, e.acceptance, prefix="reference", use_reference_as_output=True
                )
                self.writer.scalars(self.step, reference)
                self.logger.info("reference ceiling: %s", _format(reference))
                self._reference_logged = True
        metrics["perf/eval_seconds"] = self.clock() - started

        logged = {f"final/{k.split('/', 1)[1]}" if final else k: v for k, v in metrics.items()}
        self.writer.scalars(self.step, logged)
        self.writer.flush()
        self.logger.info("eval @ step %d%s: %s", self.step, f" [{label}]" if label else "", _format(metrics))
        return metrics

    def _evaluate_and_select(self) -> bool:
        """Evaluate; save the best adapter on improvement. Returns True to stop early."""
        e = self.config.evaluation
        metrics = self._evaluate()
        value = metrics.get(e.select_metric)
        if value is None or not math.isfinite(value):
            self.logger.warning("select_metric %s missing or non-finite; not selecting", e.select_metric)
            return False
        improved = self.best_metric is None or (
            value > self.best_metric if e.greater_is_better else value < self.best_metric
        )
        if improved:
            self.best_metric, self.best_step, self.bad_evaluations = value, self.step, 0
            self._save_adapter(self.paths.best, {"step": self.step, "metrics": metrics})
            self.logger.info("new best %s = %.5f at step %d", e.select_metric, value, self.step)
        else:
            self.bad_evaluations += 1
            self.logger.info(
                "no improvement on %s (%.5f vs best %.5f @ %d): %d/%s",
                e.select_metric, value, self.best_metric, self.best_step, self.bad_evaluations,
                e.early_stopping_patience,
            )
        self.writer.scalars(self.step, {"eval/best_metric": self.best_metric})
        patience = e.early_stopping_patience
        if patience is not None and self.bad_evaluations >= patience:
            self.logger.info("early stopping: %d evaluations without improvement", self.bad_evaluations)
            return True
        return False

    def _final_evaluation(self) -> dict[str, float]:
        from safetensors.torch import load_file

        if (self.paths.best / TENSORS_FILE).is_file():
            self.adapter.load_state_dict(load_file(str(self.paths.best / TENSORS_FILE)))
            self.logger.info("final evaluation uses the best adapter (step %s)", self.best_step)
        else:
            self.logger.warning("no best adapter was saved; final evaluation uses the last weights")
        metrics = self._evaluate(label="final, full eval split", final=True)
        self._save_adapter(
            self.paths.final,
            {"step": self.best_step or self.step, "final_metrics": metrics},
        )
        return metrics

    # -- persistence --------------------------------------------------------

    def _save_adapter(self, directory: Path, provenance: dict) -> None:
        temporary = directory.with_name(directory.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        save_adapter(
            temporary,
            self.adapter,
            base=asdict(self.base_info),
            token_map=self.codec.token_map,
            style_token=self.config.data.style_token,
            extra={"run": self.paths.name, "data": self._data_provenance(), **provenance},
        )
        if directory.exists():
            shutil.rmtree(directory)
        temporary.rename(directory)

    def _data_provenance(self) -> dict:
        """Where the adapter's data came from: `sft.py evaluate` defaults to it."""
        data = self.config.data
        return {
            "style": data.style,
            "train_data": str(data.train_data),
            "eval_data": str(data.eval_data) if data.eval_data is not None else None,
            "fingerprint": self.data_report.fingerprint,
        }

    def _save_checkpoint(self) -> Path:
        directory = self.paths.checkpoints / f"step-{self.step:06d}"
        temporary = directory.with_name(directory.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        save_adapter(
            temporary,
            self.adapter,
            base=asdict(self.base_info),
            token_map=self.codec.token_map,
            style_token=self.config.data.style_token,
            extra={"run": self.paths.name, "data": self._data_provenance(), "step": self.step},
        )
        torch.save(
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "stream": self.stream.state_dict(),
                "rng": {
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(self.plan.device) if self.plan.device.type == "cuda" else None,
                },
                "best_metric": self.best_metric,
                "best_step": self.best_step,
                "bad_evaluations": self.bad_evaluations,
                "skipped_steps": self.skipped_steps,
                "resume_signature": self.config.resume_signature(),
                "base_fingerprint": self.base_info.fingerprint,
                "data_fingerprint": self.data_report.fingerprint,
            },
            temporary / STATE_FILE,
        )
        if directory.exists():
            shutil.rmtree(directory)
        temporary.rename(directory)
        self._prune_checkpoints()
        self.logger.info("checkpoint saved: %s", directory)
        return directory

    def _prune_checkpoints(self) -> None:
        keep = self.config.checkpointing.keep_last
        existing = sorted(p for p in self.paths.checkpoints.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
        for old in existing[:-keep]:
            shutil.rmtree(old)

    def _write_summary(self, status: str, final_metrics: dict | None) -> dict:
        summary = {
            "run": self.paths.name,
            "status": status,
            "steps": self.step,
            "total_steps": self.total_steps,
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "select_metric": self.config.evaluation.select_metric,
            "skipped_steps": self.skipped_steps,
            "adapter_parameters": self.adapter.parameter_count(),
            "base": {k: v for k, v in asdict(self.base_info).items() if k != "config"},
            "data": self.data_report.to_dict(),
            "device_plan": self.plan.describe(),
            "final_metrics": final_metrics,
        }
        path = self.paths.run_dir / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.logger.info("summary written to %s (status: %s)", path, status)
        return summary


def latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        p for p in (run_dir / "checkpoints").glob("step-*")
        if p.is_dir() and not p.name.endswith(".tmp") and (p / STATE_FILE).is_file()
    )
    if not candidates:
        raise ResumeError(f"no resumable checkpoint under {run_dir / 'checkpoints'}")
    return candidates[-1]


class _Window:
    """Accumulates training statistics between two log lines."""

    def __init__(self) -> None:
        self.steps = 0
        self.nll_sum = 0.0
        self.tokens = 0
        self.correct = 0
        self.examples = 0
        self.real_tokens = 0
        self.padded_tokens = 0
        self.grad_norm = 0.0
        self.grad_norm_max = 0.0
        self.clipped = 0
        self.seconds = 0.0
        self.last_group_norms: dict[str, float] = {}

    def add(self, *, nll_sum, tokens, correct, examples, real_tokens, padded_tokens, grad_norm, clipped, seconds, group_norms):
        self.steps += 1
        self.nll_sum += nll_sum
        self.tokens += tokens
        self.correct += correct
        self.examples += examples
        self.real_tokens += real_tokens
        self.padded_tokens += padded_tokens
        self.grad_norm += grad_norm
        self.grad_norm_max = max(self.grad_norm_max, grad_norm)
        self.clipped += int(clipped)
        self.seconds += seconds
        if group_norms:
            self.last_group_norms = group_norms


def _diff(saved, current, prefix: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        out = []
        for key in sorted(set(saved) | set(current)):
            out += _diff(saved.get(key), current.get(key), f"{prefix}{key}.")
        return out
    return [] if saved == current else [prefix.rstrip(".")]


def _format(metrics: dict[str, float]) -> str:
    keys = (
        "eval/loss", "eval/token_accuracy", "eval/eos_accuracy", "eval/ece",
        "generation/sari", "generation/chrf_input", "generation/length_ratio",
        "generation/span_preservation", "generation/loop_rate", "generation/terminated_rate",
        "quality/acceptable_rate",
        "reference/sari", "reference/chrf_input", "reference/length_ratio",
        "reference/span_preservation", "reference/acceptable_rate",
    )
    return " | ".join(f"{k.split('/', 1)[1]} {metrics[k]:.4f}" for k in keys if k in metrics)
