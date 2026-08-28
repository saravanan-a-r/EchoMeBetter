"""
HuggingFace interoperability.

Why this file exists
--------------------
The promise being tested is: *someone who downloads a released checkpoint can
fine-tune it with `transformers` without a translation layer.* That promise
lives in field names and value conventions, which are exactly the sort of
thing a well-meaning refactor renames. So it is locked down here.

Two tiers of test:

  - **Vocabulary tests** (no `transformers` needed) assert that our field
    names *are* T5Config's field names. These are pure regression locks: they
    fail the moment someone renames `num_layers` back to something friendlier.

  - **Round-trip tests** (skipped without `transformers`) build a real
    `T5ForConditionalGeneration` from our exported config and check it is the
    same network. Name agreement is necessary but not sufficient — the
    parameter-count comparison is what proves the two descriptions actually
    denote the same model.

The strongest assertion in this file is
`test_huggingface_builds_a_model_of_exactly_our_size`: HuggingFace,
independently, from our config.json alone, must arrive at 750,906,368
parameters. Nothing else establishes that our config really is a T5 config.
"""

from __future__ import annotations

import inspect
import json

import pytest

from conftest import MODEL_CONFIG_PATH, TINY
from src.config import (
    HUGGINGFACE_ARCHITECTURE,
    HUGGINGFACE_MODEL_TYPE,
    ModelConfig,
    load_model_config,
    parse_feed_forward_proj,
)
from src.errors import ConfigError
from src.seq2seq import build_model

try:  # transformers is a convenience for interop tests, not a dependency
    from transformers import T5Config, T5ForConditionalGeneration
    from transformers.activations import ACT2FN

    HAVE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - depends on the environment
    T5Config = T5ForConditionalGeneration = ACT2FN = None
    HAVE_TRANSFORMERS = False

needs_transformers = pytest.mark.skipif(
    not HAVE_TRANSFORMERS, reason="transformers is not installed"
)


# Fields that must carry T5Config's exact name. This list is the contract; if
# `transformers` ever drops one, `test_every_shared_field_exists_in_t5config`
# fails and the divergence is a decision rather than a surprise.
SHARED_WITH_T5CONFIG = (
    "vocab_size",
    "d_model",
    "d_kv",
    "d_ff",
    "num_layers",
    "num_decoder_layers",
    "num_heads",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "dropout_rate",
    "layer_norm_epsilon",
    "initializer_factor",
    "feed_forward_proj",
    "pad_token_id",
    "eos_token_id",
)

# Our fields with no T5Config counterpart. Every one is deliberate; the test
# below fails if a new field appears without being classified, which is the
# point — an unclassified field is one nobody decided about.
OURS_ALONE = (
    "max_encoder_length",
    "max_decoder_length",
    "scale_output_for_tied_embeddings",
    "scale_attention_scores",
    "label_smoothing",
    "label_pad_token_id",
    "expected_parameters",
    "profile",
    # `tie_word_embeddings` and `decoder_start_token_id` are real HuggingFace
    # fields, but they live on `PretrainedConfig` rather than in T5Config's
    # own signature, so they are handled separately below.
    "tie_word_embeddings",
    "decoder_start_token_id",
)


# -- vocabulary: our names are HuggingFace's names -------------------------


def test_every_field_is_classified():
    """
    No field may be neither-shared-nor-ours.

    Adding a configuration field is exactly when someone should decide whether
    HuggingFace has a name for it. This test forces that decision instead of
    letting a new field drift into the file unclassified.
    """
    declared = set(SHARED_WITH_T5CONFIG) | set(OURS_ALONE)
    actual = set(ModelConfig.__dataclass_fields__)
    assert actual == declared, (
        f"unclassified: {sorted(actual - declared)}; "
        f"stale in this test: {sorted(declared - actual)}"
    )


@needs_transformers
@pytest.mark.huggingface
def test_every_shared_field_exists_in_t5config():
    """Each name we claim to share must really be a T5Config parameter."""
    signature = set(inspect.signature(T5Config.__init__).parameters)
    missing = [name for name in SHARED_WITH_T5CONFIG if name not in signature]
    assert not missing, f"not T5Config parameters: {missing}"


@needs_transformers
@pytest.mark.huggingface
def test_none_of_our_own_fields_collide_with_t5config():
    """
    A field of ours must not silently occupy a T5Config name and mean
    something else — that is worse than having a different name.
    """
    signature = set(inspect.signature(T5Config.__init__).parameters)
    collisions = [name for name in OURS_ALONE if name in signature]
    assert not collisions, f"our fields shadow T5Config parameters: {collisions}"


@needs_transformers
@pytest.mark.huggingface
def test_our_aliases_match_huggingfaces_attribute_map():
    """
    `T5Config.attribute_map` is HuggingFace's own alias table. We mirror it,
    so `config.hidden_size` means the same thing in both codebases.
    """
    config = ModelConfig(**TINY)
    for alias, canonical in T5Config.attribute_map.items():
        assert hasattr(config, alias), f"missing HuggingFace alias {alias!r}"
        assert getattr(config, alias) == getattr(config, canonical)


def test_aliases_are_read_only_and_never_serialized(large_config):
    """
    The aliases are a convenience, not a second source of truth. `to_dict()`
    must emit canonical names only, or a round-trip could set the wrong one.
    """
    payload = large_config.to_dict()
    for alias in ("hidden_size", "num_attention_heads", "num_hidden_layers", "head_dim"):
        assert alias not in payload
    with pytest.raises(Exception):
        large_config.head_dim = 128


def test_head_dim_alias_still_reads(large_config):
    """`head_dim` is in HuggingFace's attribute_map, so it keeps working."""
    assert large_config.head_dim == large_config.d_kv == 64


# -- the activation string -------------------------------------------------


def test_feed_forward_proj_uses_huggingfaces_gating_form(large_config):
    assert large_config.feed_forward_proj == "gated-gelu"
    assert large_config.is_gated_act is True
    # Not "gelu" — transformers special-cases "gated-gelu" to mean the tanh
    # approximation. See `parse_feed_forward_proj`.
    assert large_config.dense_act_fn == "gelu_new"


@pytest.mark.parametrize(
    "value,gated,activation",
    [
        ("gated-gelu_new", True, "gelu_new"),
        # The special case: NOT "gelu".
        ("gated-gelu", True, "gelu_new"),
        # `ModelConfig` rejects these, but the parser must still classify them
        # correctly: a parser that reported everything as gated would be
        # invisible if it could only be reached through a valid config.
        ("relu", False, "relu"),
        ("gelu", False, "gelu"),
        ("gelu_new", False, "gelu_new"),
    ],
)
def test_the_parser_classifies_gating_the_way_huggingface_does(value, gated, activation):
    assert parse_feed_forward_proj(value) == (gated, activation)


@needs_transformers
@pytest.mark.huggingface
@pytest.mark.parametrize("value", ["gated-gelu_new", "gated-gelu", "relu", "gelu"])
def test_the_parser_agrees_with_transformers_on_every_form(value):
    """
    Checked against `transformers` itself, including the ungated forms this
    model does not use — so the agreement is about the parsing rule, not about
    the two values that happen to be configured.
    """
    hf = T5Config(feed_forward_proj=value)
    assert parse_feed_forward_proj(value) == (hf.is_gated_act, hf.dense_act_fn)


@needs_transformers
@pytest.mark.huggingface
def test_transformers_derives_the_same_activation(large_config):
    """
    We derive `dense_act_fn`/`is_gated_act` with the same split HuggingFace
    uses. This checks the two derivations actually agree rather than trusting
    that they do.
    """
    hf = T5Config(**large_config.to_huggingface_config())
    assert hf.dense_act_fn == large_config.dense_act_fn
    assert hf.is_gated_act == large_config.is_gated_act


@needs_transformers
@pytest.mark.huggingface
def test_dense_act_fn_is_a_real_activation_key(large_config):
    """
    An activation name transformers cannot look up would raise only when the
    FFN first runs — i.e. after someone has already started fine-tuning.
    """
    assert large_config.dense_act_fn in ACT2FN
    assert type(ACT2FN[large_config.dense_act_fn]).__name__ == "NewGELUActivation"


# -- the exported document -------------------------------------------------


def test_export_declares_the_architecture(large_config):
    document = large_config.to_huggingface_config()
    assert document["model_type"] == HUGGINGFACE_MODEL_TYPE == "t5"
    assert document["architectures"] == [HUGGINGFACE_ARCHITECTURE]
    assert document["is_encoder_decoder"] is True


def test_export_carries_the_shared_fields_unchanged(large_config):
    document = large_config.to_huggingface_config()
    for name in SHARED_WITH_T5CONFIG:
        assert document[name] == getattr(large_config, name), name


def test_export_carries_the_context_limits(large_config):
    """
    T5Config has no context field, but the limit is a real property of the
    released weights — this model was never trained past 2048 either side.
    Dropping it would let a fine-tuner assume unlimited context.
    """
    document = large_config.to_huggingface_config()
    assert document["max_encoder_length"] == 2048
    assert document["max_decoder_length"] == 2048


def test_export_omits_our_training_only_fields(large_config):
    """These belong to the trainer and the collator, not to the network."""
    document = large_config.to_huggingface_config()
    for name in ("label_smoothing", "label_pad_token_id", "expected_parameters", "profile"):
        assert name not in document


def test_export_is_json_serializable(large_config):
    """It is written to config.json; a non-serializable value fails there."""
    restored = json.loads(json.dumps(large_config.to_huggingface_config()))
    assert restored["d_model"] == 1024


# -- refusing to export a lie ----------------------------------------------


def test_export_refuses_when_attention_scaling_diverges(large_config):
    """
    T5 never scales attention scores and T5Config cannot express that it
    might. Exporting anyway would produce a config.json that looks right and
    describes a different model.
    """
    with pytest.raises(ConfigError, match="scale_attention_scores"):
        large_config.with_(scale_attention_scores=True).to_huggingface_config()


def test_export_refuses_when_tied_output_scaling_is_disabled(large_config):
    """HuggingFace always scales when embeddings are tied; it has no switch."""
    with pytest.raises(ConfigError, match="scale_output_for_tied_embeddings"):
        large_config.with_(
            scale_output_for_tied_embeddings=False
        ).to_huggingface_config()


def test_untied_embeddings_export_fine(large_config):
    """
    With embeddings untied, neither we nor HuggingFace scale the output, so
    `scale_output_for_tied_embeddings` is irrelevant and must not block.
    """
    document = large_config.with_(
        tie_word_embeddings=False, scale_output_for_tied_embeddings=False
    ).to_huggingface_config()
    assert document["tie_word_embeddings"] is False


# -- round trip: does HuggingFace build the same network? ------------------


@needs_transformers
@pytest.mark.huggingface
def test_t5config_accepts_the_export(large_config):
    hf = T5Config(**large_config.to_huggingface_config())
    assert hf.model_type == "t5"
    assert hf.d_model == 1024
    assert hf.num_layers == 24
    assert hf.num_decoder_layers == 24


@needs_transformers
@pytest.mark.huggingface
def test_export_survives_a_config_json_round_trip(large_config, tmp_path):
    """The realistic path: write config.json, then `from_pretrained` it."""
    (tmp_path / "config.json").write_text(
        json.dumps(large_config.to_huggingface_config(), indent=2), encoding="utf-8"
    )
    hf = T5Config.from_pretrained(tmp_path)

    assert hf.d_model == large_config.d_model
    assert hf.d_kv == large_config.d_kv
    assert hf.num_layers == large_config.num_layers
    assert hf.feed_forward_proj == large_config.feed_forward_proj
    assert hf.tie_word_embeddings == large_config.tie_word_embeddings
    # Unrecognized keys survive PretrainedConfig, so the limit travels too.
    assert hf.max_encoder_length == large_config.max_encoder_length


@needs_transformers
@pytest.mark.huggingface
def test_huggingface_builds_a_model_of_our_size_tiny():
    """
    The load-bearing test. Two independent implementations, given the same
    configuration, must produce the same number of parameters. Name agreement
    alone would still permit a config that describes a different network.
    """
    config = ModelConfig(**TINY)
    hf_model = T5ForConditionalGeneration(T5Config(**config.to_huggingface_config()))
    assert hf_model.num_parameters() == build_model(config).parameter_count()


@needs_transformers
@pytest.mark.huggingface
@pytest.mark.slow
@pytest.mark.parametrize(
    "profile,expected", [("base", 223_395_072), ("large", 750_906_368)]
)
def test_huggingface_builds_a_model_of_exactly_our_size(profile, expected):
    """
    architecture.md §4.3's frozen totals, reached by HuggingFace from our
    exported config.json alone.
    """
    config = load_model_config(MODEL_CONFIG_PATH, profile=profile)
    hf_model = T5ForConditionalGeneration(T5Config(**config.to_huggingface_config()))
    assert config.parameter_count() == expected
    assert hf_model.num_parameters() == expected


@needs_transformers
@pytest.mark.huggingface
def test_huggingface_ties_the_embedding_the_way_we_do():
    """
    Tying is what makes the counts above match. If HuggingFace untied it, the
    parameter comparison would silently be comparing different architectures.
    """
    config = ModelConfig(**TINY)
    hf_model = T5ForConditionalGeneration(T5Config(**config.to_huggingface_config()))
    assert hf_model.lm_head.weight.data_ptr() == hf_model.shared.weight.data_ptr()


@needs_transformers
@pytest.mark.huggingface
def test_huggingface_uses_no_biases():
    """
    A single bias vector per projection would shift every count in §4.3. This
    checks the T5.1.1 no-bias convention holds on their side too.
    """
    config = ModelConfig(**TINY)
    hf_model = T5ForConditionalGeneration(T5Config(**config.to_huggingface_config()))
    named = [name for name, _ in hf_model.named_parameters() if name.endswith(".bias")]
    assert named == []
