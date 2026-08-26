"""
Encoder and decoder layers.

Pre-norm residual structure (architecture.md §4.1):

    x = x + dropout(sublayer(norm(x)))

Normalizing the *input* to each sub-layer rather than its output leaves the
residual path unnormalized all the way from the embedding to the final norm.
That clean path is what keeps gradients well-behaved through 24 layers; a
post-norm stack of this depth typically needs a warmup schedule and careful
tuning to train at all.

Parameter counts per layer (architecture.md §4.3), which the norms below are
counted into:

    encoder: self-attn (4 x d x d) + GeGLU (3 x d x d_ff) + 2 RMSNorm
    decoder: self-attn + cross-attn (4 x d x d each) + GeGLU + 3 RMSNorm
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import LayerCache, MultiHeadAttention
from .config import ModelConfig
from .layers import GeGLUFeedForward, RMSNorm


class EncoderLayer(nn.Module):
    """Bidirectional self-attention followed by a GeGLU feed-forward."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.ffn = GeGLUFeedForward(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _, _ = self.self_attn(
            self.self_attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attended)
        hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states)))
        return hidden_states


class DecoderLayer(nn.Module):
    """
    Causal self-attention, then cross-attention to the encoder, then GeGLU.

    The ordering is not arbitrary: self-attention first lets the decoder
    resolve what it has generated so far, and cross-attention then queries
    the source with that resolved state rather than with a raw embedding.

    Cross-attention receives **no position bias** — see `position_bias.py`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.cross_attn = MultiHeadAttention(config, is_cross_attention=True)
        self.ffn_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.ffn = GeGLUFeedForward(config)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        cache: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        attended, self_key, self_value = self.self_attn(
            self.self_attn_norm(hidden_states),
            attention_mask=self_attention_mask,
            position_bias=position_bias,
            past_key=None if cache is None else cache.self_key,
            past_value=None if cache is None else cache.self_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attended)

        crossed, cross_key, cross_value = self.cross_attn(
            self.cross_attn_norm(hidden_states),
            key_value_states=encoder_hidden_states,
            attention_mask=cross_attention_mask,
            position_bias=None,  # content-based only; see position_bias.py
            past_key=None if cache is None else cache.cross_key,
            past_value=None if cache is None else cache.cross_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(crossed)

        hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states)))

        if use_cache:
            return hidden_states, LayerCache(self_key, self_value, cross_key, cross_value)
        return hidden_states, None
