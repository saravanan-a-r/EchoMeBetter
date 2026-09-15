"""
Building the optimizer (architecture.md §13: AdamW, β 0.9/0.95, wd 0.1).

The two real decisions here are **which parameters weight decay applies to**
and **which tensors need a learning rate of their own** (see the last section),
and both are worth spelling out because getting either wrong is invisible.

Decay is applied to matrices and withheld from normalization gains. An
RMSNorm weight is a per-channel gain initialized to exactly 1.0; decaying it
pulls it towards 0, which scales that layer's entire output down. Nothing
crashes and no shape changes — the model simply trains to a worse place, and
the cause is not visible in any curve. The same argument applies to biases,
which this model does not have at all (T5 convention), so the rule below
reduces to "norms are exempt, everything else decays" — apart from the
position-bias tables and query projections, exempt for the reason given at
the end.

The rule is stated as `parameter.ndim < 2` rather than by matching names.
Name matching is what `transformers` does (`get_decay_parameter_names` looks
for `LayerNorm` and `.bias`), and it breaks silently the moment a module is
renamed — `self_attn_norm` does not contain the string `LayerNorm`.
Dimensionality is a property of the tensor, so it cannot drift away from the
module tree. `test_optimizer.py` checks the two rules agree on this model,
outside those two exempt families.

Tied embeddings
---------------
`named_parameters()` deduplicates by tensor identity, so the shared embedding
appears once and lands in exactly one group. That matters more than it looks:
a tied tensor placed in two parameter groups is stepped twice per optimizer
step, at two different effective learning rates, and AdamW keeps two
disagreeing moment estimates for it. `test_every_parameter_is_optimized_once`
is the lock.

Position-bias tables and query projections
------------------------------------------
AdamW moves every element by about `lr` per step whatever the tensor's size.
T5's parametrization was designed for Adafactor, whose steps are relative to
each tensor's scale, and two tensor families sit far from the ~0.03 scale of
every other matrix:

  - the query projections are initialized 8x smaller ((d_model*d_kv)^-0.5,
    standing in for the missing 1/sqrt(d_kv)), so a uniform lr grows them
    fastest, and the content attention logits with them;
  - the relative-position-bias tables must grow to several units to shape
    attention at all, and an absolute step of `lr` gets them there only
    after most of the run.

Under plain AdamW the first pretraining run's encoder was position-blind at
step 5,000 for exactly this reason: q had grown 4.2x while the bias tables
stayed near 1, so the bias moved encoder attention by a KL of ~0.03 against
~2 in a healthy T5. Each family therefore gets its own group and its own lr
multiplier, and neither is decayed: decay drags the bias tables back towards
zero while they must grow, and on a query projection held at a small lr it
would shrink the tensor instead of holding it in place.

The families are found by module class name, not `isinstance`: `pretrain.py`
and this package load the model package under different module names, so the
classes are distinct objects and `isinstance` would silently match nothing.
`test_optimizer.py` checks both groups against parameter names, so a class
rename fails a test instead of quietly undoing the fix.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from .config import TrainingConfig
from .errors import TrainingConfigError


POSITION_BIAS_MODULE = "RelativePositionBias"
ATTENTION_MODULE = "MultiHeadAttention"


def position_bias_and_query_ids(model: nn.Module) -> tuple[set[int], set[int]]:
    """Tensor ids of every position-bias table and every attention query projection."""
    position_bias: set[int] = set()
    query: set[int] = set()
    for module in model.modules():
        kind = type(module).__name__
        if kind == POSITION_BIAS_MODULE:
            position_bias.update(id(parameter) for parameter in module.parameters())
        elif kind == ATTENTION_MODULE:
            query.add(id(module.q_proj.weight))
    return position_bias, query


def decayed_and_exempt(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """
    Split trainable parameters into (weight-decayed, exempt), with names.

    Names are carried so a failure is reportable — "the exempt set is wrong"
    is not actionable, "`encoder.final_norm.weight` is being decayed" is.
    Position-bias tables and query projections are exempt (module docstring).
    """
    special = set().union(*position_bias_and_query_ids(model))
    decayed: list[tuple[str, nn.Parameter]] = []
    exempt: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decays = parameter.ndim >= 2 and id(parameter) not in special
        (decayed if decays else exempt).append((name, parameter))
    return decayed, exempt


def parameter_groups(
    model: nn.Module,
    weight_decay: float,
    position_bias_lr_multiplier: float = 1.0,
    query_lr_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """
    AdamW parameter groups, in the form `torch.optim.AdamW` expects.

    Each group carries an `lr_multiplier` that `set_learning_rate` applies to
    the scheduled rate. An empty group is dropped rather than passed through:
    PyTorch rejects one, and a model with no normalization layers is a
    legitimate thing to want to optimize.
    """
    position_bias, query = position_bias_and_query_ids(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise TrainingConfigError(
            "the model has no trainable parameters; there is nothing to optimize"
        )

    buckets: dict[str, list[nn.Parameter]] = {
        "decayed": [], "no_decay": [], "position_bias": [], "query": []
    }
    for parameter in trainable:
        if id(parameter) in position_bias:
            buckets["position_bias"].append(parameter)
        elif id(parameter) in query:
            buckets["query"].append(parameter)
        elif parameter.ndim >= 2:
            buckets["decayed"].append(parameter)
        else:
            buckets["no_decay"].append(parameter)

    settings = {
        "decayed": (weight_decay, 1.0),
        "no_decay": (0.0, 1.0),
        "position_bias": (0.0, position_bias_lr_multiplier),
        "query": (0.0, query_lr_multiplier),
    }
    return [
        {
            "params": params,
            "weight_decay": settings[name][0],
            "lr_multiplier": settings[name][1],
            "name": name,
        }
        for name, params in buckets.items()
        if params
    ]


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
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            config.weight_decay,
            config.position_bias_lr_multiplier,
            config.query_lr_multiplier,
        ),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        foreach=True,
    )
    set_learning_rate(optimizer, config.learning_rate)
    return optimizer


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Apply the scheduled learning rate to every group, times its multiplier.

    Every group, deliberately: a schedule that reached one group and not
    another would leave it on a flat learning rate for the whole run. The
    multiplier must be applied here, on every step — a group lr set once at
    construction would be overwritten by the first scheduled step.
    """
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * group.get("lr_multiplier", 1.0)


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """The scheduled learning rate in force (before any group multiplier), for logging."""
    group = optimizer.param_groups[0]
    return float(group["lr"]) / group.get("lr_multiplier", 1.0)


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
