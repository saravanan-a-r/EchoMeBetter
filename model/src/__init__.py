"""
The rephrase model: a modernized T5 (T5.1.1 / UL2 style) encoder-decoder.

Scope
-----
This package is the *model*. It defines the network, its configuration, and
enough decoding to inspect a checkpoint. It does not train, does not read a
corpus, and does not build training examples — the training loop and the
UL2 objective are separate modules.

Configuration
-------------
Every architectural dimension comes from `model_config.yml` at the project
root. Nothing here hardcodes a layer count or a width. To change the model,
edit that file.

Interface with the UL2 objective
--------------------------------
`RephraseSeq2Seq.forward` consumes the batch `UL2/src/batching.py` produces
without translation:

    from src import build_model, load_model_config      # model/
    from src import UL2Objective, pad_batch             # UL2/

    model = build_model(load_model_config())            # verifies its own size
    batch = pad_batch(examples, specials)

    output = model(
        encoder_input_ids=torch.tensor(batch["encoder_input_ids"]),
        decoder_input_ids=torch.tensor(batch["decoder_input_ids"]),
        encoder_attention_mask=torch.tensor(batch["encoder_attention_mask"]),
        decoder_attention_mask=torch.tensor(batch["decoder_attention_mask"]),
        labels=torch.tensor(batch["labels"]),
    )
    output.loss.backward()
"""

from __future__ import annotations

from .attention import (
    LayerCache,
    MultiHeadAttention,
    attention_compute_dtype,
    build_causal_mask,
    build_padding_mask,
)
from .blocks import DecoderLayer, EncoderLayer
from .config import (
    DEFAULT_CONFIG_PATH,
    HUGGINGFACE_ARCHITECTURE,
    HUGGINGFACE_MODEL_TYPE,
    ModelConfig,
    available_profiles,
    build_model_config,
    load_model_config,
)
from .errors import (
    ConfigError,
    ConversionError,
    ModelError,
    ParameterCountError,
    ShapeError,
)
from .generation import beam_generate, greedy_generate, sample_generate
from .huggingface import (
    from_huggingface_name,
    from_huggingface_state_dict,
    load_huggingface_state_dict,
    load_pretrained,
    model_config_from_huggingface,
    save_pretrained,
    to_huggingface_name,
    to_huggingface_state_dict,
)
from .layers import GeGLUFeedForward, RMSNorm
from .position_bias import RelativePositionBias, relative_position_bucket
from .seq2seq import RephraseSeq2Seq, Seq2SeqOutput, build_model
from .stacks import Decoder, Encoder

__all__ = [
    # configuration
    "ModelConfig",
    "load_model_config",
    "build_model_config",
    "available_profiles",
    "DEFAULT_CONFIG_PATH",
    "HUGGINGFACE_MODEL_TYPE",
    "HUGGINGFACE_ARCHITECTURE",
    # model
    "RephraseSeq2Seq",
    "Seq2SeqOutput",
    "build_model",
    # stacks and blocks
    "Encoder",
    "Decoder",
    "EncoderLayer",
    "DecoderLayer",
    # leaf modules
    "RMSNorm",
    "GeGLUFeedForward",
    "MultiHeadAttention",
    "RelativePositionBias",
    "relative_position_bucket",
    "LayerCache",
    "attention_compute_dtype",
    "build_padding_mask",
    "build_causal_mask",
    # decoding
    "greedy_generate",
    "sample_generate",
    "beam_generate",
    # HuggingFace interoperability
    "save_pretrained",
    "load_pretrained",
    "to_huggingface_state_dict",
    "from_huggingface_state_dict",
    "load_huggingface_state_dict",
    "to_huggingface_name",
    "from_huggingface_name",
    "model_config_from_huggingface",
    # errors
    "ModelError",
    "ConfigError",
    "ParameterCountError",
    "ShapeError",
    "ConversionError",
]
