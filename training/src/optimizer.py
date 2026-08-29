"""
Building the optimizer (architecture.md §13: AdamW, β 0.9/0.95, wd 0.1).

The only real decision here is **which parameters weight decay applies to**,
and it is worth spelling out because getting it wrong is invisible.

Decay is applied to matrices and withheld from normalization gains. An
RMSNorm weight is a per-channel gain initialized to exactly 1.0; decaying it
pulls it towards 0, which scales that layer's entire output down. Nothing
crashes and no shape changes — the model simply trains to a worse place, and
the cause is not visible in any curve. The same argument applies to biases,
which this model does not have at all (T5 convention), so the rule below
reduces to "norms are exempt, everything else decays".

The rule is stated as `parameter.ndim < 2` rather than by matching names.
Name matching is what `transformers` does (`get_decay_parameter_names` looks
for `LayerNorm` and `.bias`), and it breaks silently the moment a module is
renamed — `self_attn_norm` does not contain the string `LayerNorm`.
Dimensionality is a property of the tensor, so it cannot drift away from the
module tree. `test_optimizer.py` checks the two rules agree on this model.

Tied embeddings
---------------
`named_parameters()` deduplicates by tensor identity, so the shared embedding
appears once and lands in exactly one group. That matters more than it looks:
a tied tensor placed in two parameter groups is stepped twice per optimizer
step, at two different effective learning rates, and AdamW keeps two
disagreeing moment estimates for it. `test_every_parameter_is_optimized_once`
is the lock.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from .config import TrainingConfig
from .errors import TrainingConfigError


def decayed_and_exempt(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """
    Split trainable parameters into (weight-decayed, exempt), with names.

    Names are carried so a failure is reportable — "the exempt set is wrong"
    is not actionable, "`encoder.final_norm.weight` is being decayed" is.
    """
    decayed: list[tuple[str, nn.Parameter]] = []
    exempt: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decayed if parameter.ndim >= 2 else exempt).append((name, parameter))
    return decayed, exempt


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """
    AdamW parameter groups, in the form `torch.optim.AdamW` expects.

    An empty group is dropped rather than passed through: PyTorch rejects one,
    and a model with no normalization layers is a legitimate thing to want to
    optimize.
    """
    decayed, exempt = decayed_and_exempt(model)
    if not decayed and not exempt:
        raise TrainingConfigError(
            "the model has no trainable parameters; there is nothing to optimize"
        )

    groups: list[dict[str, Any]] = []
    if decayed:
        groups.append(
            {
                "params": [parameter for _, parameter in decayed],
                "weight_decay": weight_decay,
                "name": "decayed",
            }
        )
    if exempt:
        groups.append(
            {
                "params": [parameter for _, parameter in exempt],
                "weight_decay": 0.0,
                "name": "no_decay",
            }
        )
    return groups


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """
    AdamW, configured from the run's settings.

    `β2 = 0.95` rather than PyTorch's 0.999 (architecture.md §13). The shorter
    second-moment window adapts faster to the loss-scale changes a long
    pretraining run goes through; 0.999 is tuned for shorter fine-tuning runs
    and is slow to recover after a loss spike.

    `foreach=True` batches the element-wise update across parameters, which is
    a meaningful speedup at 750M parameters and changes nothing numerically
    beyond float reassociation.
    """
    return torch.optim.AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        foreach=True,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Apply one learning rate to every group.

    Every group, deliberately: the groups differ in weight decay only, and a
    schedule that reached one of them and not the other would be a bug whose
    only symptom is that norms learn on a flat learning rate for the whole
    run.
    """
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """The learning rate in force, for logging."""
    return float(optimizer.param_groups[0]["lr"])


def scale_gradients(parameters: Iterable[nn.Parameter], factor: float) -> None:
    """
    Multiply every accumulated gradient in place.

    Used to turn a sum of per-micro-batch gradients into the mean the run
    actually wants — see `loop.py`, where the factor is `1 / total_tokens`
    rather than `1 / accumulation_steps`.
    """
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)
