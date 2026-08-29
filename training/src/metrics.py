"""
Loss accounting, per UL2 mode (architecture.md §7.6, §7.7).

Why per-mode, and why it is not optional
----------------------------------------
§7.6 is explicit: "Report per-mode losses separately — [R] loss, [X] loss,
[S] loss — not just an aggregate. An aggregate can look healthy while one
objective silently fails to learn."

That is the whole reason this module exists. [S]-denoising (prefix
continuation) has a much higher intrinsic loss than [R]-denoising (short
spans), so a mixture loss is dominated by whichever mode is both heavily
weighted and hard. A run where [S] plateaus at its initial value and [R] keeps
improving produces a total-loss curve that descends smoothly and a model that
cannot do half of what it was trained for.

Token weighting, not batch averaging
------------------------------------
Losses are accumulated as `(sum of per-token losses, number of tokens)` and
divided at the end. Averaging per-batch means instead would weight a batch of
8-token targets equally with a batch of 900-token targets, which is not the
quantity anyone means by "[X] loss" — and UL2's three modes produce targets of
systematically different lengths, so the distortion is not random noise but a
consistent bias against the long-target modes.

Relationship to the model's own loss
------------------------------------
`per_token_losses` recomputes what `RephraseSeq2Seq.forward` already computed,
at token granularity instead of as a scalar. `test_metrics.py` requires their
aggregates to agree to floating-point tolerance — so this is a second
derivation that can disagree, not a wrapper that cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

# The three UL2 denoisers (architecture.md §7.2). Fixed order so a log line
# has stable columns across steps and runs.
MODES = ("R", "X", "S")


def per_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy at every position, `(batch, length)`, zero where ignored.

    Computed in float32 regardless of the model's dtype, matching
    `RephraseSeq2Seq.forward`. Under bf16 the difference is not cosmetic: a
    bf16 log-softmax over a 32k vocabulary loses enough precision that the
    logged loss moves in visible steps, which makes small checkpoint-to-
    checkpoint differences — the thing §7.7 is looking for — unreadable.
    """
    flat = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    return flat.view(labels.shape)


def per_example_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-row `(summed loss, token count)` — what a mode-aware tracker needs.

    Kept as a sum and a count rather than a mean so the caller can combine
    rows across batches without re-weighting, which is the only way a
    token-weighted average over a whole epoch stays exact.
    """
    losses = per_token_losses(
        logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing
    )
    counted = (labels != ignore_index).to(losses.dtype)
    return (losses * counted).sum(dim=-1), counted.sum(dim=-1)


@dataclass
class MeanLoss:
    """A running token-weighted mean."""

    total: float = 0.0
    tokens: float = 0.0

    def add(self, total: float, tokens: float) -> None:
        self.total += float(total)
        self.tokens += float(tokens)

    @property
    def value(self) -> float:
        """
        The mean, or NaN when nothing has been counted.

        NaN rather than 0.0 on purpose: a mode that produced no tokens has an
        *unknown* loss, and reporting 0.0 would make an absent objective look
        like a perfectly-learned one — precisely the confusion §7.6 warns
        about.
        """
        return self.total / self.tokens if self.tokens else float("nan")

    @property
    def perplexity(self) -> float:
        value = self.value
        return float("nan") if value != value else math.exp(value)

    def __bool__(self) -> bool:
        return self.tokens > 0


@dataclass
class LossTracker:
    """
    Accumulates overall and per-mode loss over a window of steps.

    Reset explicitly at each logging or evaluation boundary rather than
    automatically, so a caller decides what a window means and a forgotten
    reset shows up as an implausibly smooth curve rather than as silently
    wrong numbers.
    """

    overall: MeanLoss = field(default_factory=MeanLoss)
    by_mode: dict[str, MeanLoss] = field(default_factory=dict)

    def update(
        self,
        totals: torch.Tensor,
        counts: torch.Tensor,
        modes: Sequence[str] | None = None,
    ) -> None:
        """
        Add one batch, given per-row loss sums and token counts.

        `modes` is the `modes` list `UL2/src/batching.pad_batch` already puts
        in every batch — this consumes it unchanged, which is why nothing
        translates between the two modules.
        """
        totals = totals.detach().to(torch.float64).cpu()
        counts = counts.detach().to(torch.float64).cpu()
        if totals.shape != counts.shape:
            raise ValueError(
                f"loss totals {tuple(totals.shape)} and token counts "
                f"{tuple(counts.shape)} describe different numbers of rows"
            )
        if modes is not None and len(modes) != totals.shape[0]:
            raise ValueError(
                f"{len(modes)} modes for {totals.shape[0]} rows; the batch and its "
                f"mode labels disagree, so per-mode losses would be attributed to "
                f"the wrong denoiser"
            )

        self.overall.add(float(totals.sum()), float(counts.sum()))

        if modes is None:
            return
        for index, mode in enumerate(modes):
            self.by_mode.setdefault(mode, MeanLoss()).add(
                float(totals[index]), float(counts[index])
            )

    def reset(self) -> None:
        self.overall = MeanLoss()
        self.by_mode = {}

    def summary(self) -> dict[str, float]:
        """
        Flat, JSON-safe numbers for a log line or `trainer_state.json`.

        Keys are `loss`, `perplexity`, and `loss_R` / `loss_X` / `loss_S` for
        whichever modes were seen. Modes absent from the window are omitted
        rather than reported as NaN, so a log line stays readable when a
        window happened to contain only one denoiser.
        """
        summary = {"loss": self.overall.value, "perplexity": self.overall.perplexity}
        for mode in _ordered(self.by_mode):
            summary[f"loss_{mode}"] = self.by_mode[mode].value
        return summary


def _ordered(modes: Iterable[str]) -> list[str]:
    """`MODES` order first, then anything unexpected, alphabetically."""
    seen = set(modes)
    known = [mode for mode in MODES if mode in seen]
    return known + sorted(seen - set(MODES))


def format_summary(summary: Mapping[str, float]) -> str:
    """One compact, aligned line per log event."""
    parts = []
    for key, value in summary.items():
        parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return " ".join(parts)
