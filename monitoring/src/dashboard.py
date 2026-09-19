"""
`TrainingMonitor`: the `on_log` sink that turns trainer records into a dashboard.

What lands where
----------------
TensorBoard groups tags by their first path component, so each prefix below
is one section of the Scalars tab. `catalogue` gives every scalar a title and
an explanation, lays out the Custom Scalars tab, and writes the metric guide
to the Text tab.

    loss/        training loss, perplexity, per-mode loss, spike ratio
    eval/        every eval tier's loss and per-mode loss; the current best
    optim/       learning rate, pre-clip gradient norm, skipped steps
    tokens/      running token totals
    throughput/  seconds per step, tokens per second, ETA
    memory/      the trainer's peak allocated / reserved, and their gap
    gpu/         NVML utilization, SM clock, device memory       (GpuSampler)
    update_ratio/ weight_rms/ grad_norm/ group_lr/  per optimizer group
                                                           (OptimizerStats)
    position/    encoder shuffle cosine, position-bias tables  (ModelProbes)
    ownership/   checkpoint hash writes, fingerprint injections, signature
                 match count and margin -- whether each provenance technique
                 is running, not the run's training signal (see `ownership/`)

plus weight and gradient histograms per optimizer group (Histograms tab),
greedy samples per UL2 mode and the guide (Text tab).

Cost
----
Everything from the record is arithmetic on numbers the trainer already
computed, handed to `SummaryWriter`, which queues events for a background
thread to write — tens of microseconds per scalar. The components that touch
the GPU document their own costs and run on their own cadence.

`samples_every` puts the `samples/<mode>` generations (see `ModelProbes.
sample_texts`) on a tighter cadence than the rest of the probes, so the Text
tab tracks recent behaviour instead of only refreshing at the quick-eval
cadence; skipped on any step the full probe already ran, so the same step's
text is never generated twice.
"""

from __future__ import annotations

import math
import statistics
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from torch.utils.tensorboard import SummaryWriter

from . import catalogue
from .gpu import GpuSampler
from .optimizer_stats import OptimizerStats
from .probes import ModelProbes

# Record keys that describe the record rather than measure anything.
_EVAL_BOOKKEEPING = {"step", "time", "eval", "eval_tier", "best_metric", "best_step"}

# A record carrying any of these came from `ownership/` (the checkpoint hash
# chain, the trigger fingerprint, or the embedding signature) rather than
# from a training step, and is routed to `_write_ownership` instead of
# `_on_step`. Matched on the payload rather than a marker key, so `ownership/`
# stays free of any knowledge of who reads its events -- the same convention
# `pretrain.py`'s `ProgressLog` uses for the console line.
_OWNERSHIP_FIELDS = (
    "checkpoint_hashed",
    "fingerprint_injected",
    "ownership_error",
    "signature_error",
    "signature_matched",
)


class TrainingMonitor:
    """
    Write every trainer record to TensorBoard, plus what the components measure.

    Called with each record, exactly as `Trainer.on_log` is. `start_step` is
    the step the run resumes from: events a crashed session wrote past it are
    hidden (`purge_step`), so a resumed run's curves continue instead of
    doubling back over the steps it is about to repeat.

    `start_tokens_seen` / `start_input_tokens_seen` seed the throughput
    tracker at that same resume point (see `_Throughput`), so the very first
    step of a resumed session still gets a `throughput/seconds_per_step` and
    `throughput/eta_days` point instead of a one-step gap.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        max_steps: int,
        start_step: int = 0,
        start_tokens_seen: float = 0.0,
        start_input_tokens_seen: float = 0.0,
        optimizer_stats: OptimizerStats | None = None,
        gpu: GpuSampler | None = None,
        probes: ModelProbes | None = None,
        probe_every: int | None = None,
        samples_every: int | None = None,
        spike_threshold: float = 1.2,
        spike_window: int = 100,
        write: Callable[[str], Any] = print,
        now: Callable[[], datetime] = datetime.now,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if probes is not None and (probe_every is None or probe_every < 1):
            raise ValueError("probes need a positive probe_every")
        if samples_every is not None and (probes is None or samples_every < 1):
            raise ValueError("samples_every needs probes, and must be positive")
        self.log_dir = Path(log_dir)
        self._stats = optimizer_stats
        self._gpu = gpu
        self._probes = probes
        self._probe_every = probe_every
        self._samples_every = samples_every
        self._write = write
        self._now = now
        self._disabled: set[str] = set()
        self._throughput = _Throughput(
            max_steps,
            start_step=start_step,
            start_time=clock(),
            start_tokens=start_tokens_seen + start_input_tokens_seen,
        )
        self._spikes = _LossSpikes(spike_window, spike_threshold)
        self._probed_once = False
        # Tags already written this session. A tag's first event carries its
        # title and explanation; TensorBoard ignores them on every later one.
        self._introduced: set[str] = set()

        self._writer = SummaryWriter(
            str(self.log_dir), purge_step=start_step + 1, max_queue=1000, flush_secs=30
        )
        self._writer.add_custom_scalars(catalogue.layout())
        self._writer.add_text("guide", catalogue.guide(), start_step)
        self._writer.add_text(
            "run/sessions",
            f"session started {self._stamp()} at step {start_step:,} of {max_steps:,}",
            start_step,
        )

    # -- the sink -------------------------------------------------------------

    def __call__(self, record: Mapping[str, Any]) -> None:
        if record.get("finished"):
            self._guard("dashboard", self._writer.flush)
        elif record.get("eval"):
            self._guard("dashboard", lambda: self._write_scalars(eval_scalars(record), record["step"]))
        elif any(field in record for field in _OWNERSHIP_FIELDS):
            self._guard("dashboard", lambda: self._write_ownership(record))
        else:
            self._on_step(record)

    def close(self) -> None:
        for component in (self._stats, self._gpu):
            if component is not None:
                component.close()
        self._writer.close()

    # -- one ownership record -----------------------------------------------

    def _write_ownership(self, record: Mapping[str, Any]) -> None:
        """
        One point per provenance event (`ownership/`), routed here rather
        than through `_on_step`.

        These records carry no `tokens_seen` or `input_tokens_seen`, and
        `_on_step` -> `_write_training` feeds every record's `step` and token
        totals to `_Throughput.update` unconditionally. An ownership event
        landing on the same step as a training record would overwrite the
        throughput tracker's last-seen token count with zero, corrupting the
        very next `throughput/tokens_per_second` point. Keeping these out of
        `_on_step` entirely avoids that rather than working around it.

        A record with no `step` at all (`checkpoint_hash`'s "hasher is
        closed", which fires from `close()` rather than from a step) has
        nothing to plot against and is skipped, exactly as `eval_scalars`
        already requires `record["step"]` to exist for an eval record.
        """
        step = record.get("step")
        if not _is_number(step):
            return
        self._write_scalars(ownership_scalars(record), int(step))

    # -- one training record ----------------------------------------------------

    def _on_step(self, record: Mapping[str, Any]) -> None:
        step = int(record["step"])
        self._guard("dashboard", lambda: self._write_training(record, step))

        if self._gpu is not None:
            gpu = self._guard("gpu", self._gpu.drain)
            if gpu:
                self._guard("dashboard", lambda: self._write_scalars(gpu, step))

        if self._stats is not None:
            sample = self._guard("optimizer stats", self._stats.take)
            if sample is not None:
                self._guard("dashboard", lambda: self._write_stats(sample, step))
            self._guard("optimizer stats", lambda: self._stats.schedule(step + 1))

        if self._probes is not None and self._probe_due(step):
            self._probed_once = True
            result = self._guard("probes", self._probes.run)
            if result is not None:
                self._guard("dashboard", lambda: self._write_probes(result, step))
        elif self._probes is not None and self._samples_due(step):
            # The full probe above already wrote fresh samples this step;
            # skip the lighter cadence so the same step's text is not
            # generated twice.
            texts = self._guard("sample probes", self._probes.sample_texts)
            if texts is not None:
                self._guard("dashboard", lambda: self._write_texts(texts, step))

    def _write_training(self, record: Mapping[str, Any], step: int) -> None:
        scalars = training_scalars(record)
        scalars.update(self._throughput.update(record))
        ratio, started = self._spikes.update(record.get("loss"))
        if ratio is not None:
            scalars["loss/spike_ratio"] = ratio
        if started:
            self._emit(
                f"{self._stamp()} | WARNING loss spike at step {step:,}: loss "
                f"{float(record['loss']):.4f} is {ratio:.2f}x the median of the "
                f"steps before it"
            )
        self._write_scalars(scalars, step)

    def _write_stats(self, sample, step: int) -> None:
        self._write_scalars(sample.scalars, step)
        for tag, histogram in sample.histograms.items():
            self._writer.add_histogram_raw(tag, global_step=step, **histogram.as_kwargs())

    def _write_probes(self, result, step: int) -> None:
        self._write_scalars(result.scalars, step)
        self._write_texts(result.texts, step)

    def _write_texts(self, texts: Mapping[str, str], step: int) -> None:
        for tag, text in texts.items():
            self._writer.add_text(tag, text, step)

    def _write_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        for tag, value in scalars.items():
            first = None if tag in self._introduced else catalogue.described_scalar(tag, value)
            if first is None:
                self._writer.add_scalar(tag, value, step, new_style=True)
            else:
                self._writer.file_writer.add_summary(first, step)
            self._introduced.add(tag)

    def _probe_due(self, step: int) -> bool:
        # The first record of a session is probed too, so a probe that cannot
        # run fails in the first minute rather than at the first cadence step.
        return not self._probed_once or step % self._probe_every == 0

    def _samples_due(self, step: int) -> bool:
        return self._samples_every is not None and step % self._samples_every == 0

    # -- plumbing ---------------------------------------------------------------

    def _guard(self, name: str, action: Callable[[], Any]) -> Any:
        """
        Run one component's work; on its first error, disable it and say so.

        Disabled rather than retried: a component that failed once will
        almost always fail the same way on every step, and a warning per step
        for weeks buries the run's real output.
        """
        if name in self._disabled:
            return None
        try:
            return action()
        except Exception as exc:  # never let the dashboard end the run
            self._disabled.add(name)
            self._emit(
                f"{self._stamp()} | WARNING dashboard {name} disabled after "
                f"{type(exc).__name__}: {exc} -- training continues without it\n"
                + "".join(traceback.format_exception(exc)).rstrip()
            )
            return None

    def _emit(self, text: str) -> None:
        try:
            self._write(text)
        except OSError:
            pass

    def _stamp(self) -> str:
        return self._now().strftime("%Y-%m-%d %H:%M:%S")


# -- records -> scalars -----------------------------------------------------------


def training_scalars(record: Mapping[str, Any]) -> dict[str, float]:
    """The scalars one training record carries, under their dashboard tags."""
    scalars: dict[str, float] = {}
    _put(scalars, "loss/train", record.get("loss"))
    _put(scalars, "loss/perplexity", record.get("perplexity"))
    for key, value in record.items():
        if key.startswith("loss_"):
            _put(scalars, f"loss/train_{key.removeprefix('loss_')}", value)

    _put(scalars, "optim/learning_rate", record.get("learning_rate"))
    _put(scalars, "optim/grad_norm", record.get("grad_norm"))
    _put(scalars, "optim/skipped_steps", record.get("skipped_steps"))

    graded, inputs = record.get("tokens_seen"), record.get("input_tokens_seen")
    _put(scalars, "tokens/graded", graded)
    _put(scalars, "tokens/input", inputs)
    if _is_number(graded) and _is_number(inputs):
        scalars["tokens/total"] = float(graded) + float(inputs)

    allocated = record.get("gpu_peak_allocated_gib")
    reserved = record.get("gpu_peak_reserved_gib")
    _put(scalars, "memory/peak_allocated_gib", allocated)
    _put(scalars, "memory/peak_reserved_gib", reserved)
    if _is_number(allocated) and _is_number(reserved):
        # Allocator fragmentation: a gap that grows over days is the early
        # warning of an out-of-memory with room still nominally free.
        scalars["memory/fragmentation_gib"] = float(reserved) - float(allocated)
    return scalars


def ownership_scalars(record: Mapping[str, Any]) -> dict[str, float]:
    """
    One point per provenance event, under `ownership/`.

    The ask this answers is only "is it running" -- so every number here is
    one the technique already computed and logged; nothing is derived. A flat
    or missing line where one is expected is itself the signal, exactly the
    way a stalled `throughput/tokens_per_second` says the run has stalled.

    `checkpoint_hashed` carries no number of its own -- a checkpoint is
    either hashed or it isn't -- so the point is a constant `1.0`: a mark at
    every step a checkpoint was written and confirmed, on the plain
    `save_steps` cadence the console log already shows.
    """
    scalars: dict[str, float] = {}
    if "checkpoint_hashed" in record:
        scalars["ownership/checkpoint_hash_written"] = 1.0
    if "fingerprint_injected" in record:
        _put(scalars, "ownership/fingerprint_injections", record.get("fingerprint_injected"))
    if "signature_matched" in record:
        _put(scalars, "ownership/signature_matched_bits", record.get("signature_matched"))
        _put(scalars, "ownership/signature_min_margin", record.get("signature_min_margin"))
        _put(scalars, "ownership/signature_corrections", record.get("signature_corrections"))
    return scalars


def eval_scalars(record: Mapping[str, Any]) -> dict[str, float]:
    """One eval tier's record under `eval/<tier>/`, and the current best checkpoint."""
    tier = record.get("eval_tier", "eval")
    scalars: dict[str, float] = {}
    for key, value in record.items():
        if key not in _EVAL_BOOKKEEPING:
            _put(scalars, f"eval/{tier}/{key}", value)
    _put(scalars, "eval/best_metric", record.get("best_metric"))
    _put(scalars, "eval/best_step", record.get("best_step"))
    return scalars


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _put(scalars: dict[str, float], tag: str, value: Any) -> None:
    if _is_number(value):
        scalars[tag] = float(value)


class _Throughput:
    """
    Speed and time remaining from the records' wall-clock `time`.

    Measured between records rather than inside the step, like the console's
    progress line, so it includes evaluation, probes and checkpoint writes —
    what a time-remaining figure has to include to be worth reading. The ETA
    averages over this session only: a resume's startup is not the run's pace.

    Seeded at construction with the step/time/tokens the session starts from
    (`start_step` may be > 0 after a resume) — exactly what `pretrain.py`'s
    own console progress line does in `ProgressLog.start`. Without a seed,
    the first record of every session has no prior point to diff against, so
    `throughput/seconds_per_step` and `throughput/eta_days` would have no
    value at that step at all: a real gap in the TensorBoard chart even
    though the console line for that same step shows both, because the
    console seeds itself the same way. That gap reopens on every resume, not
    just at step 0, since this object is recreated each session.
    """

    def __init__(
        self,
        max_steps: int,
        *,
        start_step: int = 0,
        start_time: float | None = None,
        start_tokens: float = 0.0,
    ) -> None:
        self.max_steps = max_steps
        moment = start_time if start_time is not None else time.time()
        self._origin: tuple[int, float] | None = (start_step, moment)
        self._last: tuple[int, float, float] | None = (start_step, moment, start_tokens)

    def update(self, record: Mapping[str, Any]) -> dict[str, float]:
        moment = record.get("time")
        if not _is_number(moment):
            return {}
        step = int(record["step"])
        tokens = float(record.get("tokens_seen", 0)) + float(record.get("input_tokens_seen", 0))
        scalars: dict[str, float] = {}

        if self._last is not None:
            last_step, last_time, last_tokens = self._last
            steps, seconds = step - last_step, moment - last_time
            if steps > 0 and seconds > 0:
                scalars["throughput/seconds_per_step"] = seconds / steps
                scalars["throughput/tokens_per_second"] = (tokens - last_tokens) / seconds

        if self._origin is None:
            self._origin = (step, moment)
        else:
            origin_step, origin_time = self._origin
            done = step - origin_step
            if done > 0:
                remaining = (self.max_steps - step) * (moment - origin_time) / done
                scalars["throughput/eta_days"] = remaining / 86_400

        self._last = (step, moment, tokens)
        return scalars


class _LossSpikes:
    """
    Training loss against the median of the steps before it.

    The median rather than the mean, so one earlier spike does not raise the
    baseline and hide the next. A warning fires on the rising edge only — the
    step a spike starts — so a slow recovery is one line in the log, not one
    per step. Silent until it has a minimum of history: warmup's first steps
    fall too fast for a baseline to mean anything.
    """

    MIN_HISTORY = 20

    def __init__(self, window: int, threshold: float) -> None:
        self.threshold = threshold
        self._recent: deque[float] = deque(maxlen=window)
        self._spiking = False

    def update(self, loss: Any) -> tuple[float | None, bool]:
        if not _is_number(loss) or not math.isfinite(loss):
            return None, False
        ratio = None
        started = False
        if len(self._recent) >= min(self.MIN_HISTORY, self._recent.maxlen or 0):
            ratio = float(loss) / statistics.median(self._recent)
            spiking = ratio >= self.threshold
            started = spiking and not self._spiking
            self._spiking = spiking
        self._recent.append(float(loss))
        return ratio, started
