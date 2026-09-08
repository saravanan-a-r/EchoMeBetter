"""
The complete encoder-decoder model (architecture.md §4).

Consumes the batch format `UL2/src/batching.py` produces, unchanged:
`encoder_input_ids`, `encoder_attention_mask`, `decoder_input_ids`,
`decoder_attention_mask`, `labels`. The two modules were designed against the
same contract so nothing has to translate between them.

Tied embeddings
---------------
One `nn.Embedding` serves the encoder input, the decoder input, and — via
`F.linear` against the same matrix — the output projection. That is where
architecture.md §4.1's "saves 33.5M params at zero quality cost" comes from.

It also forces the output scaling below. An embedding matrix is initialized
for use as a lookup table, not as a classifier head; used directly as one it
produces logits roughly `sqrt(d_model)` too large, and training diverges
early. T5's fix — scaling the decoder output by `d_model^-0.5` before the
projection — is applied here whenever the embeddings are tied.

Initialization
--------------
T5's scheme, where every tensor's initial standard deviation is derived from
its own fan-in. It is not interchangeable with the more common
`normal(0, 0.02)`: the query projection deliberately uses
`(d_model * d_kv)^-0.5`, which is what compensates for the missing
`1/sqrt(d_kv)` in the attention scores (see `attention.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LayerCache, MultiHeadAttention
from .config import ModelConfig
from .errors import ParameterCountError, ShapeError
from .layers import GeGLUFeedForward, RMSNorm
from .position_bias import RelativePositionBias
from .stacks import Decoder, Encoder


@dataclass
class Seq2SeqOutput:
    """What a forward pass returns."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None
    cache: list[LayerCache] | None = None


class RephraseSeq2Seq(nn.Module):
    """Modernized T5 (T5.1.1 / UL2 style) encoder-decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Created here, shared by both stacks and the output projection.
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = Encoder(config, self.shared)
        self.decoder = Decoder(config, self.shared)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.apply(self._init_weights)

    # -- initialization ---------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """
        T5's fan-in-derived initialization.

        Applied post-order (children before parents), which matters:
        `RelativePositionBias` contains an `nn.Embedding` that the generic
        embedding branch would give `std=1.0`. The parent branch runs
        afterwards and overwrites it with the correct, much smaller value.
        """
        factor = self.config.initializer_factor
        d_model = self.config.d_model

        if isinstance(module, RelativePositionBias):
            module.embedding.weight.data.normal_(0.0, factor * d_model**-0.5)

        elif isinstance(module, nn.Embedding):
            # Large by conventional standards, and correct for T5: RMSNorm
            # immediately rescales it, and the output scaling above handles
            # the tied-classifier side.
            module.weight.data.normal_(0.0, factor * 1.0)

        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(factor * 1.0)

        elif isinstance(module, GeGLUFeedForward):
            module.gate_proj.weight.data.normal_(0.0, factor * d_model**-0.5)
            module.up_proj.weight.data.normal_(0.0, factor * d_model**-0.5)
            module.down_proj.weight.data.normal_(0.0, factor * self.config.d_ff**-0.5)

        elif isinstance(module, MultiHeadAttention):
            d_kv = self.config.d_kv
            inner_dim = self.config.inner_dim
            # The query scaling that replaces 1/sqrt(d_kv) in the scores.
            module.q_proj.weight.data.normal_(0.0, factor * (d_model * d_kv) ** -0.5)
            module.k_proj.weight.data.normal_(0.0, factor * d_model**-0.5)
            module.v_proj.weight.data.normal_(0.0, factor * d_model**-0.5)
            module.o_proj.weight.data.normal_(0.0, factor * inner_dim**-0.5)

        elif isinstance(module, nn.Linear) and module is self.lm_head:
            module.weight.data.normal_(0.0, factor * d_model**-0.5)

    # -- parameter accounting ---------------------------------------------

    def parameter_count(self) -> int:
        """
        Total trainable parameters as actually built.

        `parameters()` deduplicates by tensor identity, so a tied embedding
        is counted once — which is the whole point of tying it.
        """
        return sum(p.numel() for p in self.parameters())

    def verify_parameter_count(self) -> int:
        """
        Check the built model against both independent expectations.

        Two checks, and they are not redundant:

          1. against `config.parameter_count()` — a closed-form formula
             derived separately from the module tree, so agreement is real
             evidence rather than a tautology;
          2. against `expected_parameters` in `model_config.yml` — the
             regression lock that catches a dimension being changed without
             anyone recomputing what the change costs.

        Raises `ParameterCountError` on any disagreement.
        """
        actual = self.parameter_count()

        formula = self.config.parameter_count()
        if actual != formula:
            raise ParameterCountError(
                f"built model has {actual:,} parameters but the configuration formula "
                f"predicts {formula:,} (difference {actual - formula:+,}). The module "
                f"tree and config.parameter_count() have diverged."
            )

        declared = self.config.expected_parameters
        if declared is not None and actual != declared:
            raise ParameterCountError(
                f"profile {self.config.profile!r} has {actual:,} parameters but "
                f"model_config.yml declares expected_parameters={declared:,} "
                f"(difference {actual - declared:+,}). If a dimension was changed "
                f"deliberately, recompute and update expected_parameters."
            )
        return actual

    # -- memory -----------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Trade ~30% compute for a large cut in activation memory."""
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    @property
    def gradient_checkpointing(self) -> bool:
        return self.encoder.gradient_checkpointing and self.decoder.gradient_checkpointing

    # -- forward ----------------------------------------------------------

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the encoder alone. Generation calls this once, then reuses it.

        `encoder_segment_ids` marks which packed document each position
        belongs to, confining self-attention to one document
        (`UL2/src/packing.py`). It is a *pretraining* concern: at inference
        the encoder input is one real document, so the argument is omitted and
        attention is unrestricted, exactly as before packing existed.
        """
        self._check_ids("encoder_input_ids", encoder_input_ids, self.config.max_encoder_length)
        if encoder_segment_ids is not None and encoder_segment_ids.shape != encoder_input_ids.shape:
            raise ShapeError(
                f"encoder_segment_ids {tuple(encoder_segment_ids.shape)} must match "
                f"encoder_input_ids {tuple(encoder_input_ids.shape)}"
            )
        return self.encoder(encoder_input_ids, encoder_attention_mask, encoder_segment_ids)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decoder hidden states -> vocabulary logits."""
        if self.config.tie_word_embeddings:
            if self.config.scale_output_for_tied_embeddings:
                hidden_states = hidden_states * (self.config.d_model**-0.5)
            return F.linear(hidden_states, self.shared.weight)
        assert self.lm_head is not None
        return self.lm_head(hidden_states)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        cache: list[LayerCache] | None = None,
        use_cache: bool = False,
        encoder_segment_ids: torch.Tensor | None = None,
    ) -> Seq2SeqOutput:
        """
        One training or evaluation step.

        `labels` are the `decoder_target_ids` from the UL2 objective, padded
        with `label_pad_token_id` (-100). Those positions are excluded from
        the loss, which is what stops the model learning to predict padding.

        `encoder_segment_ids` blocks attention across the documents packed
        into one pretraining window; see `encode`. Only *encoder*
        self-attention is confined. Decoder self-attention and cross-attention
        stay unrestricted deliberately: the decoder target is one ordered
        fill-in-the-blanks sequence whose sentinels already say which document
        each span came from, and the encoder states it reads are themselves
        already document-isolated by the time cross-attention sees them.
        """
        self._check_ids("decoder_input_ids", decoder_input_ids, self.config.max_decoder_length)

        if encoder_hidden_states is None:
            encoder_hidden_states = self.encode(
                encoder_input_ids, encoder_attention_mask, encoder_segment_ids
            )

        if labels is not None and labels.shape != decoder_input_ids.shape:
            raise ShapeError(
                f"labels {tuple(labels.shape)} must match decoder_input_ids "
                f"{tuple(decoder_input_ids.shape)}"
            )

        hidden_states, present = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        logits = self.project(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=self.config.label_pad_token_id,
                label_smoothing=self.config.label_smoothing,
            )

        return Seq2SeqOutput(
            logits=logits,
            loss=loss,
            encoder_hidden_states=encoder_hidden_states,
            cache=present,
        )

    # -- validation -------------------------------------------------------

    def _check_ids(self, name: str, ids: torch.Tensor, limit: int) -> None:
        if ids.dim() != 2:
            raise ShapeError(f"{name} must be (batch, length), got {tuple(ids.shape)}")
        if ids.shape[1] == 0:
            raise ShapeError(f"{name} is empty")
        if ids.shape[1] > limit:
            # architecture.md §4.5: never truncate silently. A rephrase of a
            # truncated source looks valid and is not.
            raise ShapeError(
                f"{name} has {ids.shape[1]} tokens, exceeding the configured limit "
                f"of {limit}. Chunk upstream — this must never be truncated silently."
            )
        if ids.numel() and (int(ids.max()) >= self.config.vocab_size or int(ids.min()) < 0):
            raise ShapeError(
                f"{name} contains token IDs outside [0, {self.config.vocab_size})"
            )


def build_model(config: ModelConfig, verify: bool = True) -> RephraseSeq2Seq:
    """
    Construct a model and, by default, verify its size before it is used.

    Verifying at construction means a shape mistake costs a second, not a
    week of training a model that turns out to be the wrong size.
    """
    model = RephraseSeq2Seq(config)
    if verify:
        model.verify_parameter_count()
    return model
