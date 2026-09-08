"""
Multi-head attention, covering all three uses in the model.

One class serves encoder self-attention, decoder self-attention and decoder
cross-attention, because they differ only in where keys/values come from and
what mask is applied — not in their arithmetic. Three near-identical classes
would be three places for a mask bug to hide.

The T5 scaling convention
-------------------------
There is **no division by sqrt(d_kv)** here. That is not an omission: T5
folds the scaling into its query initializer, which uses
`(d_model * d_kv)^-0.5` instead of the usual `d_model^-0.5`. The two
choices are coupled, and `model_config.yml` says so at the
`scale_attention_scores` setting. Enabling the scaling without also changing
the initializer produces attention logits that are too small and a model
that trains poorly.

Masks
-----
Masks arriving here are **additive** float tensors broadcastable to
`(batch, heads, query, key)`: `0.0` where attention is allowed and a large
negative value where it is forbidden. `torch.finfo(dtype).min` is used rather
than `-inf` so that a row which is entirely masked produces a uniform
distribution instead of NaN — a fully-padded row is possible in a real batch
and must not poison the whole loss.

The attention kernel
---------------------
The core `matmul -> mask -> softmax -> dropout -> matmul` arithmetic runs
through `torch.nn.functional.scaled_dot_product_attention` (SDPA) rather than
those five lines written out by hand. SDPA dispatches to a fused CUDA kernel
(flash attention or memory-efficient attention, whichever supports the given
mask/dtype) when one is available, and falls back to mathematically the same
computation otherwise — so this is a kernel-selection change, not an
arithmetic one. It is given:

  - `attn_mask` = `position_bias + attention_mask` (whichever of the two are
    present, summed — the same value the manual version added to the scores
    before its softmax, just computed once instead of via two `+=`);
  - `scale=self.scale`, so SDPA's own default `1/sqrt(head_dim)` scaling is
    replaced with T5's convention (see above): `1.0` normally, or
    `d_kv^-0.5` when `scale_attention_scores` is on;
  - `dropout_p=self.dropout.p if self.training else 0.0`, which is SDPA's own
    dropout applied to the post-softmax weights — the same place
    `self.dropout` used to be called, and off in eval for the same reason
    `nn.Dropout` is a no-op in eval.

Fused kernels accumulate the softmax in float32 internally regardless of the
input dtype, which is what the old manual `.float()` upcast was doing by
hand — so this is not a numerical-stability regression under bf16 training,
it is the same safeguard moved into the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class LayerCache:
    """
    Cached keys and values for one decoder layer during generation.

    Self-attention keys/values grow by one position per step. Cross-attention
    keys/values are computed once from the encoder output and then reused
    unchanged for the whole sequence — recomputing them every step is the
    single most common waste in a naive decoder loop.
    """

    self_key: torch.Tensor | None = None
    self_value: torch.Tensor | None = None
    cross_key: torch.Tensor | None = None
    cross_value: torch.Tensor | None = None

    @property
    def self_length(self) -> int:
        return 0 if self.self_key is None else self.self_key.shape[2]

    def reorder(self, index: torch.Tensor) -> "LayerCache":
        """
        Select a subset of batch rows, in the given order.

        Beam search needs this: after each step the surviving beams are a
        permutation of the previous ones, and their caches must follow them.
        """

        def take(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else tensor.index_select(0, index)

        return LayerCache(
            self_key=take(self.self_key),
            self_value=take(self.self_value),
            cross_key=take(self.cross_key),
            cross_value=take(self.cross_value),
        )


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional cross-attention and KV caching.

    No biases on any projection (T5 convention) — see `layers.py` for why
    that matters to the parameter budget.
    """

    def __init__(self, config: ModelConfig, is_cross_attention: bool = False) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_kv
        self.inner_dim = config.inner_dim
        self.is_cross_attention = is_cross_attention
        self.scale = config.d_kv**-0.5 if config.scale_attention_scores else 1.0

        self.q_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, length, inner) -> (batch, heads, length, head_dim)"""
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, heads, length, head_dim) -> (batch, length, inner)"""
        batch, _, length, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, length, self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns `(output, key, value)`; the key/value are `None` unless
        `use_cache` is set.

        For cross-attention, passing `past_key`/`past_value` skips the
        key/value projection entirely — the encoder output does not change
        between decoding steps, so neither do its keys and values.
        """
        query = self._split_heads(self.q_proj(hidden_states))

        if self.is_cross_attention and past_key is not None and past_value is not None:
            key, value = past_key, past_value
        else:
            source = key_value_states if self.is_cross_attention else hidden_states
            if source is None:
                raise ValueError("cross-attention requires key_value_states")
            key = self._split_heads(self.k_proj(source))
            value = self._split_heads(self.v_proj(source))

            # Self-attention during generation: extend the cached prefix.
            if not self.is_cross_attention and past_key is not None:
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)

        attn_mask = _combine_additive_masks(position_bias, attention_mask)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        output = self.o_proj(self._merge_heads(attended))

        if use_cache:
            return output, key, value
        return output, None, None


def _combine_additive_masks(
    position_bias: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    Sum the two additive masks SDPA takes as a single `attn_mask` argument.

    Order does not matter (addition commutes), and this is exactly what the
    manual implementation did as two separate `scores +=` lines before its
    softmax.
    """
    if position_bias is None:
        return attention_mask
    if attention_mask is None:
        return position_bias
    return position_bias + attention_mask


def build_padding_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Turn a `(batch, length)` 1/0 mask into an additive `(batch, 1, 1, length)`
    mask that blocks padded key positions.
    """
    mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


def build_segment_mask(
    segment_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Additive `(batch, 1, length, length)` mask confining attention to one
    document.

    Several documents are packed into one encoder window during pretraining
    (`UL2/src/packing.py`). Without this, document 3's representation is
    contaminated by document 1 — they are unrelated texts that happen to share
    a window, and letting them attend to each other teaches a relationship
    that does not exist.

    `segment_ids` is `(batch, length)`: 0 for padding, and 1, 2, 3, ... for
    each packed document. A query may attend to a key only when both carry the
    same id, which makes the mask block-diagonal, one block per document.

    Padding is left to `build_padding_mask`: two padded positions do share id
    0 and are *allowed* by this mask, and are then blocked by that one. The
    two masks are summed, so a position is blocked when either forbids it —
    which keeps each mask responsible for exactly one thing.
    """
    if segment_ids.dim() != 2:
        raise ValueError(
            f"segment_ids must be (batch, length), got {tuple(segment_ids.shape)}"
        )
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    return torch.where(
        same_segment[:, None, :, :],
        torch.zeros((), dtype=dtype, device=segment_ids.device),
        torch.full((), torch.finfo(dtype).min, dtype=dtype, device=segment_ids.device),
    )


def build_causal_mask(
    query_length: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Additive `(1, 1, query_length, key_length)` mask forbidding attention to
    future positions.

    When `key_length > query_length` the queries sit at the *end* of the key
    range — the incremental-decoding case, where the cached prefix is always
    visible and only the current step's future is hidden.
    """
    offset = key_length - query_length
    query_positions = torch.arange(query_length, device=device)[:, None] + offset
    key_positions = torch.arange(key_length, device=device)[None, :]
    allowed = key_positions <= query_positions
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
    )[None, None, :, :]
