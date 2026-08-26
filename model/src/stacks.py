"""
The encoder and decoder stacks.

Each stack owns exactly one relative-position-bias table, computes the bias
once per forward pass, and hands the same tensor to every layer. That is T5's
design and it is what makes the position bias cost `num_buckets x num_heads`
per stack rather than per layer (architecture.md §4.3).

The token embedding is **not** owned here. It is created once by the
top-level model and passed in, because encoder input, decoder input and the
output projection all share it (architecture.md §4.1, tied embeddings). A
stack that created its own embedding would silently untie them and add 33.5M
parameters.

Gradient checkpointing
----------------------
Both stacks support it. It trades compute for memory by discarding
intermediate activations and recomputing them during the backward pass —
roughly a 30% slowdown for a large reduction in activation memory
(architecture.md §4.3 budgets ~201 MB per 2048-token sequence with it on).
Required to fit useful batch sizes on a single 40GB card, and essential on
the smaller card the pipeline is rehearsed on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import LayerCache, build_causal_mask, build_padding_mask
from .blocks import DecoderLayer, EncoderLayer
from .config import ModelConfig
from .layers import RMSNorm
from .position_bias import RelativePositionBias


class Encoder(nn.Module):
    """Bidirectional encoder stack."""

    def __init__(self, config: ModelConfig, embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.position_bias = RelativePositionBias(
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance,
            num_heads=config.num_heads,
            bidirectional=True,
        )
        self.layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.dropout(self.embedding(input_ids))

        additive_mask = (
            None
            if attention_mask is None
            else build_padding_mask(attention_mask, hidden_states.dtype)
        )
        length = input_ids.shape[1]
        bias = self.position_bias(length, length, input_ids.device).to(hidden_states.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer, hidden_states, additive_mask, bias, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, additive_mask, bias)

        return self.dropout(self.final_norm(hidden_states))


class Decoder(nn.Module):
    """Causal decoder stack with cross-attention to the encoder."""

    def __init__(self, config: ModelConfig, embedding: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.position_bias = RelativePositionBias(
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance,
            num_heads=config.num_heads,
            bidirectional=False,
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        cache: list[LayerCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[LayerCache] | None]:
        hidden_states = self.dropout(self.embedding(input_ids))
        dtype, device = hidden_states.dtype, input_ids.device

        query_length = input_ids.shape[1]
        past_length = cache[0].self_length if cache else 0
        key_length = past_length + query_length

        # Causal mask, plus the padding mask if one was supplied. Combining
        # them additively is safe: both are 0 or a large negative, so the sum
        # is masked if either is.
        self_mask = build_causal_mask(query_length, key_length, dtype, device)
        if decoder_attention_mask is not None:
            self_mask = self_mask + build_padding_mask(decoder_attention_mask, dtype)

        cross_mask = (
            None
            if encoder_attention_mask is None
            else build_padding_mask(encoder_attention_mask, dtype)
        )

        bias = self.position_bias(query_length, key_length, device).to(dtype)

        present: list[LayerCache] = []
        for index, layer in enumerate(self.layers):
            layer_cache = cache[index] if cache else None

            if self.gradient_checkpointing and self.training and not use_cache:
                hidden_states, _ = checkpoint(
                    layer,
                    hidden_states,
                    encoder_hidden_states,
                    self_mask,
                    cross_mask,
                    bias,
                    None,
                    False,
                    use_reentrant=False,
                )
            else:
                hidden_states, updated = layer(
                    hidden_states,
                    encoder_hidden_states,
                    self_attention_mask=self_mask,
                    cross_attention_mask=cross_mask,
                    position_bias=bias,
                    cache=layer_cache,
                    use_cache=use_cache,
                )
                if use_cache and updated is not None:
                    present.append(updated)

        hidden_states = self.dropout(self.final_norm(hidden_states))
        return hidden_states, (present if use_cache else None)
