"""
Per-optimizer-group health, measured through PyTorch's optimizer step hooks.

Why per group
-------------
The optimizer's parameter groups are the axis this project's worst training
bug ran along: under a flat learning rate the query projections grew fastest
while the position-bias tables barely moved, and the encoder came out
position-blind — invisible in the loss for thousands of steps. So each group
gets its own curve of:

  - `update_ratio/<group>`  ||w_after - w_before|| / ||w_before|| for one step,
    measured exactly. The single most diagnostic optimizer number: ~1e-3 is
    healthy for matrices; a group far above that is being thrashed, one far
    below is frozen.
  - `weight_rms/<group>`    the weights' root mean square, whose drift over the
    run is what exposed the bug above.
  - `grad_norm/<group>`     the gradient norm the optimizer actually stepped
    on, i.e. after clipping. Clipping scales every group by the same factor,
    so the groups' relative sizes are exactly the pre-clip ones; the pre-clip
    total is `optim/grad_norm`.
  - `group_lr/<group>`      the learning rate in force, proving each group's
    multiplier reached the schedule.

and, less often, a histogram of each group's weights and gradients.

How
---
`register_step_pre_hook` / `register_step_post_hook` are PyTorch's public
optimizer hooks: they run immediately before and after `optimizer.step()`,
after the trainer has clipped, with nothing about the trainer changed. On a
sampled step the pre-hook copies the weights and measures norms and
histograms; the post-hook subtracts. Nothing is written back: the copies are
the only tensors touched in place.

Cost (Large profile, A100X-40C vGPU, measured)
---------------------------------------------
A sampled step copies every weight once — about 3 GB, allocated after the
backward pass has freed its activations, so it stays under the step's peak —
and runs a handful of fused norm kernels: ~14 ms before the step and ~13 ms
after, on a step of ~36 s, and only every `every_steps`. Histograms bucket
every element on the GPU, ~370 ms, every `histogram_every_steps`.

Timing contract
---------------
`schedule(step)` arms the next optimizer step when `step` is on a cadence;
`take()` returns what that step measured. The monitor calls `take()` then
`schedule(step + 1)` on every training record, so with `logging_steps: 1` a
sample is written at exactly the step it measured. A step the trainer skips
(non-finite gradients) never calls `optimizer.step()`, so the sample simply
lands on the next real step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils import get_total_norm


@dataclass(frozen=True)
class Histogram:
    """One histogram in the form `SummaryWriter.add_histogram_raw` takes."""

    min: float
    max: float
    num: float
    sum: float
    sum_squares: float
    bucket_limits: list[float]
    bucket_counts: list[float]

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "min": self.min,
            "max": self.max,
            "num": self.num,
            "sum": self.sum,
            "sum_squares": self.sum_squares,
            "bucket_limits": self.bucket_limits,
            "bucket_counts": self.bucket_counts,
        }


@dataclass(frozen=True)
class StatsSample:
    scalars: dict[str, float]
    histograms: dict[str, Histogram]


class OptimizerStats:
    """Measure each parameter group on sampled optimizer steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        every_steps: int = 10,
        histogram_every_steps: int = 1000,
    ) -> None:
        if every_steps < 1 or histogram_every_steps < 1:
            raise ValueError("every_steps and histogram_every_steps must be positive")
        self.optimizer = optimizer
        self.every_steps = every_steps
        self.histogram_every_steps = histogram_every_steps
        self._armed = False
        # The first sample of a session carries histograms, so the tab has a
        # baseline before the first histogram cadence step.
        self._histograms_due = True
        self._in_flight: list[_GroupMeasurement] | None = None
        self._histograms: dict[str, Histogram] = {}
        self._sample: StatsSample | None = None
        self._error: Exception | None = None
        self._edges: dict[torch.device, torch.Tensor] = {}
        self._handles = [
            optimizer.register_step_pre_hook(self._before_step),
            optimizer.register_step_post_hook(self._after_step),
        ]

    def schedule(self, next_step: int) -> None:
        """Arm the next optimizer step if `next_step` falls on a cadence."""
        self._raise_if_failed()
        on_histogram_cadence = next_step % self.histogram_every_steps == 0
        if next_step % self.every_steps == 0 or on_histogram_cadence:
            self._armed = True
        if on_histogram_cadence:
            self._histograms_due = True

    def take(self) -> StatsSample | None:
        """The last sampled step's measurement, once; `None` if none is ready."""
        self._raise_if_failed()
        sample, self._sample = self._sample, None
        return sample

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._in_flight = None

    # -- the hooks --------------------------------------------------------------
    #
    # They run inside `optimizer.step()`, where an exception would end the run,
    # so they never raise: an error is kept, the hooks are removed, and the
    # next `take()`/`schedule()` hands it to the monitor to report.

    def _before_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if not self._armed:
            return
        try:
            self._measure_before()
        except Exception as exc:
            self._fail(exc)

    def _after_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._in_flight is None:
            return
        try:
            self._measure_after()
        except Exception as exc:
            self._fail(exc)

    @torch.no_grad()
    def _measure_before(self) -> None:
        histograms_due = self._histograms_due
        measurements: list[_GroupMeasurement] = []
        histograms: dict[str, Histogram] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("name", f"group_{index}"))
            weights = [parameter.detach() for parameter in group["params"]]
            grads = [
                parameter.grad.detach()
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            measurements.append(
                _GroupMeasurement(
                    name=name,
                    weights=weights,
                    before=[weight.clone() for weight in weights],
                    weight_norm=get_total_norm(weights),
                    grad_norm=get_total_norm(grads) if grads else None,
                    numel=sum(weight.numel() for weight in weights),
                    lr=float(group["lr"]),
                )
            )
            if histograms_due:
                histograms[f"weights/{name}"] = self._histogram(weights)
                if grads:
                    histograms[f"gradients/{name}"] = self._histogram(grads)
        self._in_flight = measurements
        self._histograms = histograms

    @torch.no_grad()
    def _measure_after(self) -> None:
        measurements, self._in_flight = self._in_flight, None
        norms: list[torch.Tensor] = []
        for group in measurements:
            for before, weight in zip(group.before, group.weights):
                before.sub_(weight)
            norms.append(get_total_norm(group.before))
            group.before.clear()
            norms.append(group.weight_norm)
            norms.append(group.grad_norm if group.grad_norm is not None else torch.zeros_like(group.weight_norm))
        # One device sync for every group's numbers.
        values = torch.stack([norm.float() for norm in norms]).tolist()

        scalars: dict[str, float] = {}
        for index, group in enumerate(measurements):
            delta, weight_norm, grad_norm = values[3 * index : 3 * index + 3]
            if weight_norm > 0:
                scalars[f"update_ratio/{group.name}"] = delta / weight_norm
            scalars[f"weight_rms/{group.name}"] = weight_norm / math.sqrt(max(group.numel, 1))
            if group.grad_norm is not None:
                scalars[f"grad_norm/{group.name}"] = grad_norm
            scalars[f"group_lr/{group.name}"] = group.lr

        self._sample = StatsSample(scalars=scalars, histograms=self._histograms)
        self._histograms = {}
        self._armed = False
        self._histograms_due = False

    # -- histograms ---------------------------------------------------------------

    def _histogram(self, tensors: list[torch.Tensor]) -> Histogram:
        """
        Every element of `tensors`, bucketed on the GPU.

        Copying weights to the host for `add_histogram` would move gigabytes
        per call. The buckets are TensorBoard's own exponentially spaced
        defaults, so a group whose tensors differ in scale by 30x (embeddings
        against projections) still resolves both.
        """
        device = tensors[0].device
        edges = self._edges.get(device)
        if edges is None:
            edges = torch.tensor(_tensorboard_bucket_edges(), dtype=torch.float32, device=device)
            self._edges[device] = edges

        counts = torch.zeros(edges.numel() + 1, dtype=torch.long, device=device)
        total = torch.zeros((), dtype=torch.float64, device=device)
        squares = torch.zeros((), dtype=torch.float64, device=device)
        lows, highs = [], []
        for tensor in tensors:
            flat = tensor.reshape(-1).float()
            counts += torch.bincount(torch.bucketize(flat, edges), minlength=edges.numel() + 1)
            total += flat.sum(dtype=torch.float64)
            squares += flat.square().sum(dtype=torch.float64)
            low, high = torch.aminmax(flat)
            lows.append(low)
            highs.append(high)

        low, high, total_sum, total_squares = torch.stack(
            [torch.stack(lows).min().double(), torch.stack(highs).max().double(), total, squares]
        ).tolist()
        # The final bucket holds only what no edge bounds (non-finite values);
        # it has no upper limit to report, so it is left out.
        bucket_counts = counts[:-1].tolist()
        nonzero = [i for i, count in enumerate(bucket_counts) if count]
        if not nonzero:
            return Histogram(low, high, 0.0, total_sum, total_squares, [high], [0.0])
        first, last = nonzero[0], nonzero[-1]
        limits = edges[first : last + 1].tolist()
        kept = [float(count) for count in bucket_counts[first : last + 1]]
        return Histogram(low, high, float(sum(kept)), total_sum, total_squares, limits, kept)

    # -- plumbing -------------------------------------------------------------------

    def _fail(self, exc: Exception) -> None:
        self._error = exc
        self._armed = False
        self.close()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


@dataclass
class _GroupMeasurement:
    name: str
    weights: list[torch.Tensor]
    before: list[torch.Tensor]
    weight_norm: torch.Tensor
    grad_norm: torch.Tensor | None
    numel: int
    lr: float


def _tensorboard_bucket_edges() -> list[float]:
    """TensorBoard's default histogram edges: +-1e-12 to 1e20, each 1.1x the last."""
    positive = []
    value = 1e-12
    while value < 1e20:
        positive.append(value)
        value *= 1.1
    return [-edge for edge in reversed(positive)] + [0.0] + positive
