"""
Weight-level HuggingFace compatibility (`src/huggingface.py`).

`test_huggingface_compat.py` covers the *configuration*. This file covers the
**weights**, and the distinction matters: a compatible config.json with an
incompatible state dict produces a checkpoint that fails to load, and — worse
— a state dict with correct *names* but a swapped pair of tensors produces one
that loads cleanly and is quietly wrong.

So the load-bearing test here is not a name comparison. It is
`test_huggingface_reproduces_our_logits`: export the model, build a real
`T5ForConditionalGeneration` from the exported files, and require the same
logits for the same input. That single assertion covers the naming, the
`wi_0`/`wi_1` gate ordering, the relative-position bucketing, the tied-output
scaling and the activation simultaneously — every one of which is invisible to
a shape check.

The name-level tests are still here, because when the numerical test fails
they are what tells you *which* tensor went where.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conftest import TINY, make_batch
from src.config import ModelConfig
from src.errors import ConversionError
from src.huggingface import (
    CONFIG_FILE,
    WEIGHTS_FILE,
    from_huggingface_name,
    from_huggingface_state_dict,
    load_huggingface_state_dict,
    load_pretrained,
    model_config_from_huggingface,
    save_pretrained,
    to_huggingface_name,
    to_huggingface_state_dict,
)
from src.seq2seq import build_model

try:  # transformers is an optional dependency of the test suite only.
    from transformers import T5Config, T5ForConditionalGeneration

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - exercised only where it is absent
    HAVE_TRANSFORMERS = False

needs_transformers = pytest.mark.skipif(
    not HAVE_TRANSFORMERS, reason="transformers is not installed"
)

pytestmark = pytest.mark.huggingface


# A deliberately *asymmetric* shape: 3 encoder layers and 2 decoder layers, so
# a test cannot pass by accident when the two stacks are confused, and enough
# blocks that an index bug shows up as more than an off-by-one on block 0.
DEEPER = dict(TINY, num_layers=3, num_decoder_layers=2)


@pytest.fixture
def deeper_config() -> ModelConfig:
    return ModelConfig(**DEEPER)


@pytest.fixture
def deeper_model(deeper_config: ModelConfig):
    torch.manual_seed(0)
    return build_model(deeper_config, verify=False).eval()


def huggingface_twin(config: ModelConfig):
    """An empty `T5ForConditionalGeneration` of the same shape."""
    document = dict(config.to_huggingface_config())
    document.pop("architectures", None)
    document.pop("max_encoder_length", None)
    document.pop("max_decoder_length", None)
    return T5ForConditionalGeneration(T5Config(**document))


# -- the name mapping ------------------------------------------------------


def test_every_parameter_has_a_huggingface_name(deeper_model):
    """
    Totality. `to_huggingface_name` raises on anything it does not recognize,
    so this failing means a parameter was added to the model and the converter
    was not told about it — which is exactly when a released checkpoint would
    silently lose a tensor.
    """
    for name, _ in deeper_model.named_parameters():
        assert to_huggingface_name(name)  # raises ConversionError otherwise


def test_the_mapping_is_injective(deeper_model):
    """
    Two parameters landing on one name would overwrite each other, leaving the
    file short a tensor and `from_pretrained` filling the gap with random
    values rather than failing.
    """
    names = [to_huggingface_name(name) for name, _ in deeper_model.named_parameters()]
    assert len(names) == len(set(names))


def test_the_two_directions_are_exact_inverses(deeper_model):
    """
    Written as two separate functions, so consistency is a property to check
    rather than something to trust.
    """
    for name, _ in deeper_model.named_parameters():
        assert from_huggingface_name(to_huggingface_name(name)) == name


@needs_transformers
def test_our_names_are_exactly_a_real_t5s_names(deeper_config, deeper_model):
    """
    The set comparison, against `transformers` itself rather than against a
    list copied into this file — a list would keep passing after HuggingFace
    renamed something.
    """
    theirs = {name for name, _ in huggingface_twin(deeper_config).named_parameters()}
    ours = {to_huggingface_name(name) for name, _ in deeper_model.named_parameters()}
    assert ours == theirs


@needs_transformers
def test_every_translated_tensor_has_the_shape_huggingface_expects(
    deeper_config, deeper_model
):
    """
    Catches a transposed projection, which name equality alone would not.
    """
    theirs = dict(huggingface_twin(deeper_config).named_parameters())
    for name, parameter in deeper_model.named_parameters():
        assert parameter.shape == theirs[to_huggingface_name(name)].shape, name


def test_the_position_bias_moves_into_block_zero():
    """
    One of the three real structural differences: this package keeps the bias
    table on the stack, HuggingFace keeps it inside the first block's
    self-attention. Both mean "one table per stack".
    """
    assert to_huggingface_name("encoder.position_bias.embedding.weight") == (
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    )
    assert to_huggingface_name("decoder.position_bias.embedding.weight") == (
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    )


def test_the_gate_branch_maps_to_wi_zero():
    """
    The mapping most worth pinning explicitly. `wi_0` is the branch
    HuggingFace passes through the activation and `wi_1` the one it does not
    (`T5DenseGatedActDense.forward`). The two have identical shapes, so
    swapping them loads without complaint and changes what the network
    computes — the kind of defect that surfaces as "fine-tuning mysteriously
    starts from a worse loss than the released eval reported".
    """
    stem = "encoder.layers.0.ffn"
    assert to_huggingface_name(f"{stem}.gate_proj.weight").endswith("wi_0.weight")
    assert to_huggingface_name(f"{stem}.up_proj.weight").endswith("wi_1.weight")
    assert to_huggingface_name(f"{stem}.down_proj.weight").endswith("wo.weight")


def test_decoder_sublayer_numbering_is_not_the_encoders():
    """
    In an encoder block the feed-forward is `layer.1`; in a decoder block
    `layer.1` is cross-attention and the feed-forward is `layer.2`. Reusing
    the encoder's numbering would put every decoder FFN weight on top of the
    cross-attention norm.
    """
    assert ".layer.1.DenseReluDense." in to_huggingface_name(
        "encoder.layers.0.ffn.gate_proj.weight"
    )
    assert ".layer.2.DenseReluDense." in to_huggingface_name(
        "decoder.layers.0.ffn.gate_proj.weight"
    )
    assert ".layer.1.EncDecAttention." in to_huggingface_name(
        "decoder.layers.0.cross_attn.q_proj.weight"
    )


def test_norms_keep_their_role_through_a_round_trip():
    """
    All three decoder norms are called `layer_norm` on the HuggingFace side and
    are told apart only by position, so the inverse has real work to do.
    """
    for role in ("self_attn", "cross_attn", "ffn"):
        name = f"decoder.layers.1.{role}_norm.weight"
        assert from_huggingface_name(to_huggingface_name(name)) == name


@pytest.mark.parametrize(
    "name",
    [
        "encoder.layers.0.self_attn.qkv_proj.weight",  # a projection we do not have
        "encoder.layers.0.rotary.inv_freq",            # a component we do not have
        "encoder.layers.x.ffn.gate_proj.weight",       # not an index
        "adapter.0.weight",                            # not part of the model
        "encoder.mystery.weight",
    ],
)
def test_an_unknown_parameter_is_refused_not_passed_through(name):
    """
    Passing an unrecognized name through unchanged is the failure mode worth
    preventing: it exports cleanly, `from_pretrained` discards it as an
    unexpected key with a warning, and the component stays at its random
    initialization in a checkpoint everyone believes is trained.
    """
    with pytest.raises(ConversionError):
        to_huggingface_name(name)


def test_a_second_position_bias_table_is_refused():
    """
    T5 has exactly one table per stack, on block 0, shared by every layer. A
    checkpoint with a table on block 3 is a different architecture, not a
    naming variant, and importing it by ignoring the block index would throw
    away weights.
    """
    with pytest.raises(ConversionError, match="one table per stack"):
        from_huggingface_name(
            "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight"
        )


def test_a_sublayer_the_encoder_does_not_have_is_refused():
    with pytest.raises(ConversionError, match="sub-layer"):
        from_huggingface_name("encoder.block.0.layer.2.DenseReluDense.wo.weight")


# -- state dicts -----------------------------------------------------------


def test_the_exported_state_dict_omits_the_tied_aliases(deeper_model):
    """
    `shared.weight`, `encoder.embedding.weight` and `decoder.embedding.weight`
    are one tensor viewed three ways. `safetensors` refuses to store the same
    storage twice, and HuggingFace re-ties on load, so only `shared.weight`
    is written — matching what HuggingFace's own `save_pretrained` produces.
    """
    exported = to_huggingface_state_dict(deeper_model)
    assert "shared.weight" in exported
    assert "encoder.embed_tokens.weight" not in exported
    assert "decoder.embed_tokens.weight" not in exported
    assert "lm_head.weight" not in exported  # tied here
    assert len(exported) == len(list(deeper_model.named_parameters()))


def test_the_export_does_not_alias_the_live_weights(deeper_model):
    """
    The exported tensors are clones. Sharing storage with the model would mean
    a checkpoint being written while training continues records whatever the
    weights happened to become mid-write.
    """
    exported = to_huggingface_state_dict(deeper_model)
    with torch.no_grad():
        deeper_model.shared.weight.add_(1.0)
    assert not torch.equal(exported["shared.weight"], deeper_model.shared.weight)


def test_an_untied_model_exports_its_own_lm_head():
    torch.manual_seed(0)
    config = ModelConfig(**dict(DEEPER, tie_word_embeddings=False))
    exported = to_huggingface_state_dict(build_model(config, verify=False))
    assert "lm_head.weight" in exported
    assert "shared.weight" in exported


def test_a_state_dict_missing_a_tensor_is_refused(deeper_model):
    exported = to_huggingface_state_dict(deeper_model)
    del exported["encoder.block.1.layer.1.DenseReluDense.wo.weight"]
    with pytest.raises(ConversionError, match="missing"):
        load_huggingface_state_dict(deeper_model, exported)


def test_a_state_dict_with_an_extra_tensor_is_refused(deeper_model):
    exported = to_huggingface_state_dict(deeper_model)
    exported["encoder.block.0.layer.0.SelfAttention.q.bias"] = torch.zeros(4)
    with pytest.raises(ConversionError):
        load_huggingface_state_dict(deeper_model, exported)


def test_a_local_round_trip_is_exact(deeper_model):
    exported = to_huggingface_state_dict(deeper_model)
    restored = from_huggingface_state_dict(exported)
    original = dict(deeper_model.named_parameters())
    assert set(restored) == set(original)
    for name, tensor in restored.items():
        assert torch.equal(tensor, original[name])


# -- the tests that actually prove compatibility ---------------------------


@needs_transformers
@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_huggingface_reproduces_our_logits(tmp_path, tie_word_embeddings):
    """
    The load-bearing test of this module.

    Save a checkpoint, hand the directory to `from_pretrained`, and require
    HuggingFace to compute the same logits we do. Two independently written
    implementations agreeing tensor-for-tensor is the only evidence that
    "compatible" means anything — it covers the gate ordering, the position
    bucketing, the output scaling and the activation at once, none of which a
    name or shape comparison can see.

    Both tying modes, because tying changes which tensors are written *and*
    whether the `d_model**-0.5` output scaling applies.
    """
    torch.manual_seed(0)
    config = ModelConfig(**dict(DEEPER, tie_word_embeddings=tie_word_embeddings))
    ours = build_model(config, verify=False).eval()

    save_pretrained(ours, tmp_path)
    theirs = T5ForConditionalGeneration.from_pretrained(tmp_path).eval()

    batch = make_batch(config, batch_size=3, source_length=13, target_length=9)
    with torch.no_grad():
        mine = ours(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
        ).logits
        hers = theirs(
            input_ids=batch["encoder_input_ids"],
            attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
        ).logits

    assert mine.shape == hers.shape
    # Float32 reassociation only: the two implementations perform the same
    # arithmetic in a different order. A real mismatch — a swapped gate, a
    # missing scale — is orders of magnitude larger than this.
    assert torch.allclose(mine, hers, atol=1e-4, rtol=1e-4)


@needs_transformers
def test_a_checkpoint_trained_in_huggingface_loads_back_here(deeper_config):
    """
    The import direction, which is the point of releasing at all: someone
    fine-tunes the checkpoint with `transformers` and their weights must come
    back into this package. Uses HuggingFace's *in-memory* state dict, which
    still carries the tied aliases a file does not.
    """
    torch.manual_seed(1)
    theirs = huggingface_twin(deeper_config).eval()
    ours = build_model(deeper_config, verify=False)
    load_huggingface_state_dict(ours, theirs.state_dict())
    ours.eval()

    batch = make_batch(deeper_config, batch_size=2, source_length=10, target_length=6)
    with torch.no_grad():
        mine = ours(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
        ).logits
        hers = theirs(
            input_ids=batch["encoder_input_ids"],
            attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
        ).logits
    assert torch.allclose(mine, hers, atol=1e-4, rtol=1e-4)


def test_saving_and_loading_here_returns_the_same_model(tmp_path, deeper_model):
    """
    A round trip through the files, not through the functions — which is what
    proves `config.json` and `model.safetensors` between them carry everything
    needed to rebuild the model.
    """
    save_pretrained(deeper_model, tmp_path)
    restored = load_pretrained(tmp_path).eval()

    batch = make_batch(deeper_model.config, batch_size=2, source_length=9, target_length=5)
    with torch.no_grad():
        before = deeper_model(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
        ).logits
        after = restored(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
        ).logits
    assert torch.equal(before, after)


def test_save_pretrained_writes_the_two_files_huggingface_looks_for(
    tmp_path, deeper_model
):
    save_pretrained(deeper_model, tmp_path)
    assert (tmp_path / CONFIG_FILE).is_file()
    assert (tmp_path / WEIGHTS_FILE).is_file()

    document = json.loads((tmp_path / CONFIG_FILE).read_text())
    assert document["model_type"] == "t5"
    assert document["architectures"] == ["T5ForConditionalGeneration"]


def test_save_pretrained_refuses_a_model_huggingface_cannot_reproduce(tmp_path):
    """
    `to_huggingface_config`'s refusals have to reach the file writer, or the
    weights get written next to a config.json that describes something else.
    """
    from src.errors import ConfigError

    torch.manual_seed(0)
    config = ModelConfig(**dict(DEEPER, scale_attention_scores=True))
    model = build_model(config, verify=False)
    with pytest.raises(ConfigError, match="scale_attention_scores"):
        save_pretrained(model, tmp_path)
    assert not (tmp_path / WEIGHTS_FILE).exists()


@needs_transformers
def test_the_written_file_holds_exactly_the_tensors_huggingface_writes(
    tmp_path, deeper_config, deeper_model
):
    """
    Compared against a directory HuggingFace wrote itself, so "the same two
    files" is checked rather than asserted.
    """
    from safetensors.torch import load_file

    save_pretrained(deeper_model, tmp_path / "ours")
    huggingface_twin(deeper_config).save_pretrained(tmp_path / "theirs")

    ours = load_file(str(tmp_path / "ours" / WEIGHTS_FILE))
    theirs = load_file(str(tmp_path / "theirs" / WEIGHTS_FILE))
    assert set(ours) == set(theirs)


# -- reading a configuration back ------------------------------------------


def test_config_json_round_trips_through_model_config(deeper_config):
    restored = model_config_from_huggingface(deeper_config.to_huggingface_config())
    for field in (
        "d_model",
        "d_kv",
        "d_ff",
        "num_layers",
        "num_decoder_layers",
        "num_heads",
        "vocab_size",
        "relative_attention_num_buckets",
        "relative_attention_max_distance",
        "layer_norm_epsilon",
        "feed_forward_proj",
        "tie_word_embeddings",
        "dropout_rate",
        "max_encoder_length",
        "max_decoder_length",
        "pad_token_id",
        "eos_token_id",
        "decoder_start_token_id",
    ):
        assert getattr(restored, field) == getattr(deeper_config, field), field


def test_reading_a_config_ignores_bookkeeping_keys(deeper_config):
    """
    Deliberately the opposite of `ModelConfig.from_dict`, which rejects
    unknown keys. Every config.json in the wild carries whatever the writing
    version of `transformers` added — `transformers_version`, `dtype`,
    `task_specific_params` — and refusing to read a checkpoint over one of
    those would defeat the only purpose this function has.
    """
    document = dict(
        deeper_config.to_huggingface_config(),
        transformers_version="5.13.0",
        dtype="bfloat16",
        task_specific_params={"summarization": {"max_length": 200}},
        some_future_key=17,
    )
    assert model_config_from_huggingface(document).d_model == deeper_config.d_model


def test_reading_a_config_missing_a_shape_field_is_refused(deeper_config):
    """
    Bookkeeping keys may be missing; the fields that determine the network may
    not. Defaulting `d_model` would build a model of the wrong size and load
    nothing into it.
    """
    document = dict(deeper_config.to_huggingface_config())
    del document["d_model"]
    with pytest.raises(ConversionError, match="d_model"):
        model_config_from_huggingface(document)


def test_a_foreign_t5_config_is_readable():
    """
    A stock `google/t5-v1_1-base` config.json, transcribed. Nothing in this
    project produced it, so it exercises the defaults for the fields
    HuggingFace's T5 simply does not have (the context limits).
    """
    document = {
        "architectures": ["T5ForConditionalGeneration"],
        "model_type": "t5",
        "d_model": 768,
        "d_kv": 64,
        "d_ff": 2048,
        "num_layers": 12,
        "num_decoder_layers": 12,
        "num_heads": 12,
        "vocab_size": 32128,
        "relative_attention_num_buckets": 32,
        "relative_attention_max_distance": 128,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu",
        "tie_word_embeddings": False,
        "is_encoder_decoder": True,
        "use_cache": True,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "transformers_version": "4.30.0",
    }
    config = model_config_from_huggingface(document)
    assert config.d_model == 768
    assert config.tie_word_embeddings is False
    assert config.max_encoder_length == 2048  # ours; T5 declares no limit
    assert config.dense_act_fn == "gelu_new"


def test_the_exported_directory_is_a_plain_path(tmp_path, deeper_model):
    """`save_pretrained` returns where it wrote, so callers can chain."""
    written = save_pretrained(deeper_model, tmp_path / "nested" / "run")
    assert written == Path(tmp_path / "nested" / "run")
    assert written.is_dir()
