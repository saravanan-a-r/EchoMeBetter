"""
The adapter: LoRA pairs on frozen linears, plus learned frame-token embeddings.

LoRA (Hu et al. 2021)
---------------------
Each targeted projection `W` (frozen) becomes

    y = W x + scaling * B (A dropout(x))        A: (r, in)   B: (out, r)

`B` starts at zero, so an adapter that has just been attached is an exact
no-op -- step 0 of every run evaluates the untouched base model, which is
what makes the first evaluation a real baseline. `A` uses PEFT's
Kaiming-uniform initialization. `scaling` is `alpha / rank`, or
`alpha / sqrt(rank)` with rsLoRA (Kalajdzievski 2023), which keeps the size
of the update stable when only the rank is changed.

Frame-token embeddings
----------------------
The product frame `<style:X> <text_to_rewrite> ... </text_to_rewrite>`
uses tokens the base never trained on before the pretraining cooldown: their
rows are still random vectors shrunk by weight decay. LoRA cannot give a
token a meaning of its own -- it only changes what the network does with the
vector it is handed -- so the adapter also learns an additive delta for those
few rows (1,024 parameters each).

The delta applies to the **input lookup only**. The embedding is tied to the
output projection (`RephraseSeq2Seq.project` reads `shared.weight`), and
leaving `shared` itself untouched keeps every output logit on the frozen base
rows: the adapter never changes how likely the model is to *emit* a frame
token, only what a frame token in the input means.

Nothing here modifies a base tensor. The base weights stay bit-identical
through training (tested), so one base checkpoint serves every adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TARGET_GROUPS, AdapterConfig
from .errors import AdapterError

TOKEN_DELTA_KEY = "token_embedding_delta"


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float, dropout: float) -> None:
        super().__init__()
        if base.bias is not None:
            raise AdapterError("the base model's projections are bias-free; got one with a bias")
        self.base = base
        self.rank = rank
        self.scaling = scaling
        weight = base.weight
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=weight.dtype, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=weight.dtype, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """The frozen base weight, for code that inspects `module.weight`."""
        return self.base.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        low_rank = F.linear(F.linear(self.dropout(hidden_states), self.lora_A), self.lora_B)
        return self.base(hidden_states) + low_rank * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """`scaling * B @ A`, the dense update this pair is equivalent to."""
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def delta_norm(self) -> float:
        """
        Frobenius norm of `delta_weight()` without materializing it:
        ||BA||_F^2 = trace((B^T B)(A A^T)), two rank x rank products.
        """
        a = self.lora_A.float()
        b = self.lora_B.float()
        squared = torch.trace((b.t() @ b) @ (a @ a.t())).clamp_min(0.0)
        return float(squared.sqrt()) * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.4g}"


class TokenEmbeddingDelta(nn.Module):
    """
    An input embedding lookup whose rows for `token_ids` carry a learned delta.

    Stands in for the shared `nn.Embedding` at the encoder's and decoder's
    input only; `weight` still reports the frozen base matrix.
    """

    def __init__(self, base: nn.Embedding, token_ids: Sequence[int]) -> None:
        super().__init__()
        ids = [int(i) for i in token_ids]
        if not ids:
            raise AdapterError("TokenEmbeddingDelta needs at least one token id")
        if len(set(ids)) != len(ids):
            raise AdapterError(f"duplicate token ids: {ids}")
        vocab = base.num_embeddings
        if min(ids) < 0 or max(ids) >= vocab:
            raise AdapterError(f"token ids {ids} fall outside the vocabulary [0, {vocab})")
        self.base = base
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        # Slot 0 of `_padded` is a zero row; token i's delta lives at slot i+1.
        slots = torch.zeros(vocab, dtype=torch.long)
        slots[torch.tensor(ids)] = torch.arange(1, len(ids) + 1)
        self.register_buffer("slots", slots.to(base.weight.device), persistent=False)
        self.delta = nn.Parameter(
            torch.zeros(len(ids), base.embedding_dim, dtype=base.weight.dtype, device=base.weight.device)
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def num_embeddings(self) -> int:
        return self.base.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([self.delta.new_zeros(1, self.delta.shape[1]), self.delta], dim=0)
        return self.base(input_ids) + F.embedding(self.slots[input_ids], padded)


@dataclass
class AttachedAdapter:
    """Handle to an adapter attached to a model: its modules and its accounting."""

    config: AdapterConfig
    lora_modules: dict[str, LoRALinear]
    module_groups: dict[str, str]
    token_ids: tuple[int, ...] = ()
    token_delta: TokenEmbeddingDelta | None = None
    base_weight_norms: dict[str, float] = field(default_factory=dict)

    # -- parameters ------------------------------------------------------

    def named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for name, module in self.lora_modules.items():
            yield f"{name}.lora_A", module.lora_A
            yield f"{name}.lora_B", module.lora_B
        if self.token_delta is not None:
            yield TOKEN_DELTA_KEY, self.token_delta.delta

    def parameters(self) -> list[nn.Parameter]:
        return [p for _, p in self.named_parameters()]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, learning_rate: float, weight_decay: float) -> list[dict]:
        """
        Optimizer groups: one per distinct learning-rate multiplier, so the
        `lr_multipliers` in the config reach the optimizer and the scheduler's
        per-group `initial_lr` keeps each group's ratio through the schedule.
        The token delta never takes weight decay: decaying it pulls the frame
        tokens back towards their untrained random rows.
        """
        by_multiplier: dict[float, list[nn.Parameter]] = {}
        for name, module in self.lora_modules.items():
            projection = name.rsplit(".", 1)[-1]
            multiplier = float(self.config.lr_multipliers.get(projection, 1.0))
            by_multiplier.setdefault(multiplier, []).extend([module.lora_A, module.lora_B])
        groups = [
            {
                "name": f"lora_x{multiplier:g}",
                "params": params,
                "lr": learning_rate * multiplier,
                "lr_multiplier": multiplier,
                "weight_decay": weight_decay,
            }
            for multiplier, params in sorted(by_multiplier.items())
        ]
        if self.token_delta is not None:
            groups.append(
                {
                    "name": "token_delta",
                    "params": [self.token_delta.delta],
                    "lr": learning_rate,
                    "lr_multiplier": 1.0,
                    "weight_decay": 0.0,
                }
            )
        return groups

    # -- state -----------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().to("cpu", torch.float32).contiguous() for name, p in self.named_parameters()}

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Strict: every adapter tensor must be present, correctly shaped, and nothing extra."""
        own = dict(self.named_parameters())
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise AdapterError(
                f"adapter state does not match the attached adapter: "
                f"missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"unexpected {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        for name, parameter in own.items():
            tensor = state[name]
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise AdapterError(
                    f"{name}: saved shape {tuple(tensor.shape)} != attached {tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(parameter.dtype))

    # -- diagnostics -----------------------------------------------------

    def relative_delta_norms(self) -> dict[str, float]:
        """
        ||scaling * B A||_F / ||W||_F, averaged per projection and per group.

        How far the adapter has moved each kind of layer away from the base,
        in units of that layer's own size -- the number that shows whether
        the query projections (8x smaller weights in T5) are being moved
        disproportionately.
        """
        per_projection: dict[str, list[float]] = {}
        per_group: dict[str, list[float]] = {}
        for name, module in self.lora_modules.items():
            ratio = module.delta_norm() / max(self.base_weight_norms[name], 1e-12)
            per_projection.setdefault(name.rsplit(".", 1)[-1], []).append(ratio)
            per_group.setdefault(self.module_groups[name], []).append(ratio)
        result = {f"projection/{k}": sum(v) / len(v) for k, v in per_projection.items()}
        result.update({f"group/{k}": sum(v) / len(v) for k, v in per_group.items()})
        return result

    def gradient_norms(self) -> dict[str, float]:
        """Pre-clipping gradient norm per target group (and the token delta)."""
        squared: dict[str, float] = {}
        for name, module in self.lora_modules.items():
            group = self.module_groups[name]
            for parameter in (module.lora_A, module.lora_B):
                if parameter.grad is not None:
                    squared[group] = squared.get(group, 0.0) + float(parameter.grad.float().pow(2).sum())
        if self.token_delta is not None and self.token_delta.delta.grad is not None:
            squared["token_delta"] = float(self.token_delta.delta.grad.float().pow(2).sum())
        return {group: math.sqrt(value) for group, value in squared.items()}

    def token_delta_norms(self) -> dict[int, float]:
        if self.token_delta is None:
            return {}
        norms = self.token_delta.delta.detach().float().norm(dim=1).tolist()
        return dict(zip(self.token_ids, norms))


def attach_adapter(
    model: nn.Module,
    config: AdapterConfig,
    token_ids: Sequence[int] = (),
) -> AttachedAdapter:
    """
    Freeze every base parameter, then add the configured LoRA pairs and token
    delta in place. Returns the handle the trainer and the saver use.
    """
    if getattr(model, "_sft_adapter_attached", False):
        raise AdapterError("an adapter is already attached to this model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lora_modules: dict[str, LoRALinear] = {}
    module_groups: dict[str, str] = {}
    base_weight_norms: dict[str, float] = {}
    for group, projections in config.targets.items():
        if not projections:
            continue
        stack_name, sublayer_name, _ = TARGET_GROUPS[group]
        stack = getattr(model, stack_name)
        for index, layer in enumerate(stack.layers):
            sublayer = getattr(layer, sublayer_name)
            for projection in projections:
                base = getattr(sublayer, projection)
                if not isinstance(base, nn.Linear):
                    raise AdapterError(
                        f"{stack_name}.layers.{index}.{sublayer_name}.{projection} is "
                        f"{type(base).__name__}, not nn.Linear"
                    )
                wrapped = LoRALinear(base, config.rank, config.scaling, config.dropout)
                setattr(sublayer, projection, wrapped)
                name = f"{stack_name}.layers.{index}.{sublayer_name}.{projection}"
                lora_modules[name] = wrapped
                module_groups[name] = group
                base_weight_norms[name] = float(base.weight.detach().float().norm())

    token_delta = None
    if token_ids:
        shared = model.shared
        token_delta = TokenEmbeddingDelta(shared, token_ids)
        # The encoder's and decoder's input lookups; `model.shared` (and so
        # the tied output projection) stays the frozen base embedding.
        model.encoder.embedding = token_delta
        model.decoder.embedding = token_delta

    model._sft_adapter_attached = True
    return AttachedAdapter(
        config=config,
        lora_modules=lora_modules,
        module_groups=module_groups,
        token_ids=tuple(int(i) for i in token_ids),
        token_delta=token_delta,
        base_weight_norms=base_weight_norms,
    )


def expected_adapter_parameters(model_config, adapter: AdapterConfig) -> int:
    """
    Closed-form adapter size from the two configs -- independent of the module
    tree, so agreement with the attached count is real evidence, the way
    `ModelConfig.parameter_count` is for the base.
    """
    d, d_ff, inner = model_config.d_model, model_config.d_ff, model_config.inner_dim
    shapes = {
        "q_proj": (d, inner), "k_proj": (d, inner), "v_proj": (d, inner), "o_proj": (inner, d),
        "gate_proj": (d, d_ff), "up_proj": (d, d_ff), "down_proj": (d_ff, d),
    }
    layers = {"encoder": model_config.num_layers, "decoder": model_config.num_decoder_layers}
    total = 0
    for group, projections in adapter.targets.items():
        stack_name = TARGET_GROUPS[group][0]
        for projection in projections:
            fan_in, fan_out = shapes[projection]
            total += layers[stack_name] * adapter.rank * (fan_in + fan_out)
    return total + len(adapter.trainable_tokens) * d
