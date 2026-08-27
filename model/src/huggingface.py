"""
Releasing this model as a HuggingFace checkpoint, and reading one back.

`config.py` made the *configuration* HuggingFace-compatible: every field that
exists in `transformers.T5Config` uses that field's name, and
`ModelConfig.to_huggingface_config()` emits a real `config.json`. That closed
half the gap. This module closes the other half — the **weights**.

Why a converter rather than a rename
------------------------------------
HuggingFace's T5 names parameters after its module tree, which nests every
sub-layer inside a numbered `layer` list:

    encoder.block.3.layer.0.SelfAttention.q.weight
    decoder.block.3.layer.1.EncDecAttention.o.weight
    decoder.block.3.layer.2.DenseReluDense.wi_0.weight

The index `0`/`1`/`2` *is* the meaning: in the encoder `layer.1` is the feed
forward, in the decoder `layer.1` is cross-attention and `layer.2` is the feed
forward. Reading a decoder key means knowing that off by heart. This package
names the same tensors after what they are — `self_attn`, `cross_attn`, `ffn`,
`gate_proj`, `up_proj`, `down_proj` — and that readability is worth keeping.

So the two vocabularies are kept, and the translation between them is written
down once, here, where it can be tested. The alternative — renaming every
module to HuggingFace's scheme — would make the model's own source harder to
read forever, in exchange for deleting a file that a test proves correct.

What "compatible" is taken to mean
----------------------------------
Not "the names line up". Names lining up is necessary and proves nothing: a
converter can map `gate_proj -> wi_1` instead of `wi_0`, produce a state dict
that loads with `strict=True` and no warnings, and give a model that is
quietly wrong. `wi_0` is the branch HuggingFace passes through the activation
and `wi_1` the one it does not, and swapping them is invisible to every
shape check.

The test that matters therefore does not compare names at all: it exports a
model, builds a real `T5ForConditionalGeneration` from the exported files, and
requires the two to produce **the same logits** for the same input. That
covers the naming, the gate ordering, the position-bias bucketing, the tied
output scaling, and the activation, all at once.

The three places the trees genuinely differ
-------------------------------------------
Everything else is a regular substitution; these are not:

1. **The relative position bias lives somewhere else.** Both models keep one
   bias table per stack shared by every layer, but HuggingFace stores it
   *inside the first block's self-attention* rather than on the stack:

       encoder.position_bias.embedding.weight
       -> encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight

   Same shape, `(num_buckets, num_heads)`, and same bucketing function.

2. **Tied embeddings appear three extra times.** A tied HuggingFace T5 lists
   `encoder.embed_tokens.weight`, `decoder.embed_tokens.weight` and
   `lm_head.weight` alongside `shared.weight` — all four views of one tensor.
   They are omitted on export, exactly as HuggingFace's own `save_pretrained`
   omits them, because `safetensors` refuses to store the same storage twice
   and `from_pretrained` re-ties them from `shared.weight` anyway. Our own
   state dict has the same duplication (`encoder.embedding.weight`,
   `decoder.embedding.weight`) and it is dropped for the same reason.

3. **Norms are named by position, not by role.** `self_attn_norm`,
   `cross_attn_norm` and `ffn_norm` all become `layer_norm`, distinguished
   only by which `layer.N` they sit in.

Untied embeddings (`tie_word_embeddings: false`) are handled too, and there
`lm_head.weight` is a real, separately-stored parameter on both sides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ModelConfig
from .errors import ConfigError, ConversionError
from .seq2seq import RephraseSeq2Seq, build_model

# -- the vocabulary difference, written down once -------------------------

# Which `layer.N` a sub-layer occupies inside a HuggingFace block. The
# encoder has two sub-layers and the decoder three, and the numbering is
# positional — which is exactly why it is spelled out here rather than
# inferred at each call site.
_SUBLAYER_INDEX = {
    "encoder": {"self_attn": 0, "ffn": 1},
    "decoder": {"self_attn": 0, "cross_attn": 1, "ffn": 2},
}

# HuggingFace's class name for each attention, used as the attribute name.
_ATTENTION_MODULE = {"self_attn": "SelfAttention", "cross_attn": "EncDecAttention"}

# Projection names. The feed-forward pair is the dangerous one: `wi_0` is the
# branch HuggingFace applies the activation to and `wi_1` the branch it does
# not, so `gate_proj -> wi_0` is load-bearing and shape-checking cannot catch
# it being wrong. See `T5DenseGatedActDense.forward`.
_ATTENTION_PROJECTIONS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"}
_FEED_FORWARD_PROJECTIONS = {"gate_proj": "wi_0", "up_proj": "wi_1", "down_proj": "wo"}

# Tied views of the token embedding. Present in both state dicts, stored in
# neither file: `shared.weight` is the one real copy and both loaders re-tie.
_OURS_TIED_ALIASES = ("encoder.embedding.weight", "decoder.embedding.weight")
_HUGGINGFACE_TIED_ALIASES = (
    "encoder.embed_tokens.weight",
    "decoder.embed_tokens.weight",
    "lm_head.weight",
)

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


def to_huggingface_name(name: str) -> str:
    """
    Translate one of this package's parameter names into HuggingFace's.

    Raises `ConversionError` for anything unrecognized rather than passing it
    through. That refusal is the point: a parameter added to the model later
    and not accounted for here would otherwise be exported under its own name,
    land in the file as an unexpected key, and be dropped by `from_pretrained`
    with a warning nobody reads — a released checkpoint missing a tensor.
    """
    parts = name.split(".")

    if name == "shared.weight" or name == "lm_head.weight":
        return name

    stack = parts[0]
    if stack not in ("encoder", "decoder") or len(parts) < 2:
        raise ConversionError(f"no HuggingFace name for parameter {name!r}")

    rest = parts[1:]

    # encoder.final_norm.weight -> encoder.final_layer_norm.weight
    if rest == ["final_norm", "weight"]:
        return f"{stack}.final_layer_norm.weight"

    # encoder.position_bias.embedding.weight -> block 0's self-attention
    if rest == ["position_bias", "embedding", "weight"]:
        return (
            f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        )

    # encoder.layers.{i}.{sublayer}[.{projection}].weight
    if len(rest) >= 3 and rest[0] == "layers":
        index, tail = rest[1], rest[2:]
        if not index.isdigit():
            raise ConversionError(f"no HuggingFace name for parameter {name!r}")
        return f"{stack}.block.{index}." + _sublayer_to_huggingface(stack, tail, name)

    raise ConversionError(f"no HuggingFace name for parameter {name!r}")


def _sublayer_to_huggingface(stack: str, tail: list[str], name: str) -> str:
    """The part of a layer's parameter name below `encoder.block.{i}.`."""
    indices = _SUBLAYER_INDEX[stack]

    # A norm: named for its role here, for its position there.
    if len(tail) == 2 and tail[1] == "weight" and tail[0].endswith("_norm"):
        role = tail[0][: -len("_norm")]
        if role in indices:
            return f"layer.{indices[role]}.layer_norm.weight"

    if len(tail) == 3 and tail[2] == "weight":
        role, projection = tail[0], tail[1]
        if role in _ATTENTION_MODULE and projection in _ATTENTION_PROJECTIONS:
            module = _ATTENTION_MODULE[role]
            return (
                f"layer.{indices[role]}.{module}."
                f"{_ATTENTION_PROJECTIONS[projection]}.weight"
            )
        if role == "ffn" and projection in _FEED_FORWARD_PROJECTIONS:
            return (
                f"layer.{indices['ffn']}.DenseReluDense."
                f"{_FEED_FORWARD_PROJECTIONS[projection]}.weight"
            )

    raise ConversionError(f"no HuggingFace name for parameter {name!r}")


def from_huggingface_name(name: str) -> str:
    """
    Translate a HuggingFace T5 parameter name into this package's.

    The exact inverse of `to_huggingface_name` — `test_huggingface_weights.py`
    checks that on every name the real model has, in both directions, rather
    than trusting the two functions to have been written consistently.
    """
    if name in ("shared.weight", "lm_head.weight"):
        return name

    parts = name.split(".")
    stack = parts[0]
    if stack not in ("encoder", "decoder"):
        raise ConversionError(f"no local name for HuggingFace parameter {name!r}")

    if parts[1:] == ["final_layer_norm", "weight"]:
        return f"{stack}.final_norm.weight"

    # encoder.block.{i}.layer.{n}.{module}.{projection}.weight
    if len(parts) >= 6 and parts[1] == "block" and parts[3] == "layer":
        index, sublayer, tail = parts[2], parts[4], parts[5:]
        if index.isdigit() and sublayer.isdigit():
            role = _role_for(stack, int(sublayer), name)

            if tail == ["layer_norm", "weight"]:
                return f"{stack}.layers.{index}.{role}_norm.weight"

            if tail == ["SelfAttention", "relative_attention_bias", "weight"]:
                # Only block 0 carries it; the table belongs to the stack.
                if index != "0":
                    raise ConversionError(
                        f"{name!r} puts a relative position bias on block {index}. "
                        f"T5 keeps exactly one table per stack, on block 0, shared "
                        f"by every layer — a second table is a different model."
                    )
                return f"{stack}.position_bias.embedding.weight"

            if len(tail) == 3 and tail[2] == "weight":
                module, projection = tail[0], tail[1]
                if module in ("SelfAttention", "EncDecAttention"):
                    inverse = {v: k for k, v in _ATTENTION_PROJECTIONS.items()}
                    if projection in inverse:
                        return f"{stack}.layers.{index}.{role}.{inverse[projection]}.weight"
                if module == "DenseReluDense":
                    inverse = {v: k for k, v in _FEED_FORWARD_PROJECTIONS.items()}
                    if projection in inverse:
                        return f"{stack}.layers.{index}.ffn.{inverse[projection]}.weight"

    raise ConversionError(f"no local name for HuggingFace parameter {name!r}")


def _role_for(stack: str, sublayer: int, name: str) -> str:
    """Invert the positional `layer.N` numbering back to a role."""
    for role, index in _SUBLAYER_INDEX[stack].items():
        if index == sublayer:
            return role
    raise ConversionError(
        f"{name!r} refers to sub-layer {sublayer} of a {stack} block, which has "
        f"only {len(_SUBLAYER_INDEX[stack])}"
    )


# -- state dicts ----------------------------------------------------------


def to_huggingface_state_dict(model: RephraseSeq2Seq) -> dict[str, torch.Tensor]:
    """
    This model's weights under HuggingFace's names, ready to serialize.

    Built from `named_parameters()` rather than `state_dict()` so the tied
    aliases never appear: `named_parameters()` deduplicates by tensor
    identity, which is precisely the set of tensors a file should contain.
    Tensors are detached and cloned, so writing a checkpoint cannot be
    perturbed by training continuing on the model afterwards.
    """
    converted: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        converted[to_huggingface_name(name)] = parameter.detach().cpu().clone()
    return converted


def from_huggingface_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    A HuggingFace T5 state dict under this package's names.

    The tied aliases HuggingFace's in-memory `state_dict()` carries
    (`encoder.embed_tokens.weight` and friends) are dropped rather than
    translated: they are views of `shared.weight`, and `load_state_dict` would
    reject them as unexpected keys.
    """
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name in _HUGGINGFACE_TIED_ALIASES and "shared.weight" in state_dict:
            continue
        local = from_huggingface_name(name)
        if local in converted:
            raise ConversionError(
                f"two HuggingFace parameters map to {local!r}; the state dict does "
                f"not describe a single T5"
            )
        converted[local] = tensor
    return converted


def load_huggingface_state_dict(
    model: RephraseSeq2Seq,
    state_dict: Mapping[str, torch.Tensor],
) -> RephraseSeq2Seq:
    """
    Load HuggingFace-named weights into this model, in place.

    `strict=True` on purpose. A missing tensor leaves a component at its
    random initialization, which produces a model that runs, generates
    plausible-looking text and is not the checkpoint anyone meant to load.
    """
    local = from_huggingface_state_dict(state_dict)
    # The tied aliases in *our* tree are views of `shared.weight`; supplying
    # `shared.weight` alone is enough, so they are not expected either.
    missing, unexpected = model.load_state_dict(local, strict=False)
    real_missing = [name for name in missing if name not in _OURS_TIED_ALIASES]
    if real_missing or unexpected:
        raise ConversionError(
            f"state dict does not match the model: missing {sorted(real_missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    return model


# -- files ----------------------------------------------------------------


def save_pretrained(model: RephraseSeq2Seq, directory: str | Path) -> Path:
    """
    Write a directory `T5ForConditionalGeneration.from_pretrained` can load.

    Produces `config.json` and `model.safetensors` — the same two files, with
    the same contents, that HuggingFace's own `save_pretrained` would write
    for this network. Nothing here is HuggingFace-specific beyond the naming:
    `safetensors` is the format, and it is preferred to `torch.save` because
    loading it does not execute pickled code from a downloaded file.

    The configuration is written through `to_huggingface_config()`, so its
    refusals apply here too — a model whose behaviour `T5Config` cannot
    express is not written at all, rather than written misleadingly.
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    document = model.config.to_huggingface_config()
    (directory / CONFIG_FILE).write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )

    # `contiguous()` because safetensors stores raw buffers and a transposed
    # or sliced view has no single one.
    tensors = {
        name: tensor.contiguous()
        for name, tensor in to_huggingface_state_dict(model).items()
    }
    save_file(tensors, str(directory / WEIGHTS_FILE))
    return directory


def load_pretrained(directory: str | Path) -> RephraseSeq2Seq:
    """
    Rebuild this package's model from a directory written by `save_pretrained`.

    The point of this direction is that it makes the export *checkable*: a
    round trip that returns a model producing identical logits proves the
    files carry everything, which reading the writer's source cannot.

    Note `expected_parameters` is absent from a `config.json` — it is this
    project's regression lock, not an architectural fact — so the reloaded
    model is verified against the configuration formula only.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    document = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))

    model = build_model(model_config_from_huggingface(document), verify=True)
    return load_huggingface_state_dict(model, load_file(str(directory / WEIGHTS_FILE)))


# -- configuration, the other direction -----------------------------------

# Fields a `ModelConfig` needs that a `config.json` does not carry, with the
# value the export guarantees. `to_huggingface_config()` refuses to write a
# file unless the first two hold, so reading them back as constants is not a
# guess — it is the same contract read from the other end.
_NOT_IN_CONFIG_JSON = {
    "scale_attention_scores": False,   # export refuses if True
    "scale_output_for_tied_embeddings": True,  # export refuses if False while tied
    # Training-time settings that belong to the trainer, not the network.
    # A fine-tuner sets these from their own training configuration.
    "label_smoothing": 0.0,
    "label_pad_token_id": -100,
}

def model_config_from_huggingface(document: Mapping[str, Any]) -> ModelConfig:
    """
    Read a `config.json` back into a `ModelConfig`.

    Unknown keys are **ignored**, not rejected — the opposite of
    `ModelConfig.from_dict`, and deliberately so. A `config.json` in the wild
    carries whatever the writing version of `transformers` felt like adding
    (`transformers_version`, `dtype`, `task_specific_params`, …), and refusing
    to read a checkpoint because of a bookkeeping key would make this useless
    for the one job it has. The fields that determine the network are all
    required, so nothing load-bearing can be missing quietly.

    `max_encoder_length` / `max_decoder_length` fall back to the largest
    context in `architecture.md` §4.4 when a foreign checkpoint omits them,
    since HuggingFace's T5 declares no limit at all.
    """
    required = (
        "vocab_size",
        "d_model",
        "d_kv",
        "d_ff",
        "num_layers",
        "num_heads",
        "relative_attention_num_buckets",
        "relative_attention_max_distance",
        "layer_norm_epsilon",
        "initializer_factor",
        "feed_forward_proj",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise ConversionError(
            f"config.json is missing fields that determine the network: {missing}"
        )

    fields: dict[str, Any] = {key: document[key] for key in required}
    fields["num_decoder_layers"] = document.get(
        "num_decoder_layers", document["num_layers"]
    )
    fields["dropout_rate"] = document.get("dropout_rate", 0.0)
    fields["tie_word_embeddings"] = document.get("tie_word_embeddings", True)
    fields["pad_token_id"] = document.get("pad_token_id", 0)
    fields["eos_token_id"] = document.get("eos_token_id", 1)
    fields["decoder_start_token_id"] = document.get(
        "decoder_start_token_id", fields["pad_token_id"]
    )
    fields["max_encoder_length"] = document.get("max_encoder_length", 2048)
    fields["max_decoder_length"] = document.get("max_decoder_length", 2048)
    fields["profile"] = document.get("profile", "huggingface")
    fields.update(_NOT_IN_CONFIG_JSON)

    try:
        return ModelConfig.from_dict(fields)
    except ConfigError as exc:
        raise ConversionError(f"config.json does not describe a usable model: {exc}") from exc
