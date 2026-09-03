"""
Unit tests for configuration loading and validation.

`model_config.yml` is the single place the model's shape is defined, so these
tests cover two things: that the file is read and merged correctly, and that
a malformed or impossible configuration fails at load time rather than
producing a model of the wrong shape that trains perfectly well and is
useless.
"""

from __future__ import annotations

import pytest
import yaml

from conftest import MODEL_CONFIG_PATH, TINY
from src.config import (
    ModelConfig,
    available_profiles,
    build_model_config,
    load_model_config,
)
from src.errors import ConfigError


def _document(**overrides):
    document = {
        "active_profile": "small",
        "defaults": {
            key: value
            for key, value in TINY.items()
            if key not in {"d_model", "d_ff", "num_layers", "num_decoder_layers",
                           "num_heads", "profile", "expected_parameters"}
        },
        "profiles": {
            "small": {
                "d_model": 64,
                "d_ff": 176,
                "num_layers": 2,
                "num_decoder_layers": 2,
                "num_heads": 4,
            }
        },
    }
    document.update(overrides)
    return document


# -- loading the real file ------------------------------------------------


def test_the_real_config_file_exists():
    assert MODEL_CONFIG_PATH.is_file(), "model_config.yml is the single source of truth"


def test_both_profiles_load():
    for profile in ("base", "large"):
        config = load_model_config(MODEL_CONFIG_PATH, profile=profile)
        assert config.profile == profile


def test_active_profile_is_used_when_none_is_requested():
    document = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    config = load_model_config(MODEL_CONFIG_PATH)
    assert config.profile == document["active_profile"]


def test_available_profiles_lists_all():
    assert available_profiles(MODEL_CONFIG_PATH) == ["base", "large", "medium"]


def test_profiles_inherit_shared_defaults():
    """Only the shape fields differ; everything else comes from `defaults`."""
    base = load_model_config(MODEL_CONFIG_PATH, profile="base")
    large = load_model_config(MODEL_CONFIG_PATH, profile="large")

    assert base.vocab_size == large.vocab_size
    assert base.max_encoder_length == large.max_encoder_length
    assert base.relative_attention_num_buckets == large.relative_attention_num_buckets
    assert base.d_model != large.d_model


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "absent.yml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "model_config.yml"
    path.write_text("defaults: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_model_config(path)


def test_non_mapping_document_raises(tmp_path):
    path = tmp_path / "model_config.yml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_model_config(path)


def test_round_trip_through_a_written_file(tmp_path):
    path = tmp_path / "model_config.yml"
    path.write_text(yaml.safe_dump(_document()), encoding="utf-8")
    assert load_model_config(path).d_model == 64


# -- merge logic ----------------------------------------------------------


def test_profile_overrides_a_default():
    document = _document()
    document["defaults"]["dropout_rate"] = 0.0
    document["profiles"]["small"]["d_model"] = 128
    assert build_model_config(document).d_model == 128


def test_unknown_profile_raises():
    with pytest.raises(ConfigError, match="unknown profile"):
        build_model_config(_document(), profile="enormous")


def test_missing_defaults_block_raises():
    document = _document()
    del document["defaults"]
    with pytest.raises(ConfigError, match="defaults"):
        build_model_config(document)


def test_missing_profiles_block_raises():
    document = _document()
    del document["profiles"]
    with pytest.raises(ConfigError, match="profiles"):
        build_model_config(document)


def test_empty_profiles_block_raises():
    with pytest.raises(ConfigError, match="profiles"):
        build_model_config(_document(profiles={}))


def test_no_profile_and_no_active_profile_raises():
    document = _document()
    del document["active_profile"]
    with pytest.raises(ConfigError, match="no 'active_profile'"):
        build_model_config(document)


def test_unknown_key_in_a_profile_raises():
    """
    A silently-ignored typo means training a model that is not the one
    intended, so profile keys are restricted to the shape fields.
    """
    document = _document()
    document["profiles"]["small"]["d_modle"] = 128
    with pytest.raises(ConfigError, match="unknown keys"):
        build_model_config(document)


def test_unknown_key_in_defaults_raises():
    document = _document()
    document["defaults"]["mystery_setting"] = 1
    with pytest.raises(ConfigError, match="unknown model configuration keys"):
        build_model_config(document)


def test_missing_required_setting_raises():
    document = _document()
    del document["defaults"]["vocab_size"]
    with pytest.raises(ConfigError, match="missing model configuration keys"):
        build_model_config(document)


def test_error_messages_name_the_profile():
    document = _document()
    document["defaults"]["vocab_size"] = -1
    with pytest.raises(ConfigError, match="'small'"):
        build_model_config(document)


# -- field validation -----------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "d_model", "d_ff", "num_layers", "num_decoder_layers",
        "num_heads", "d_kv", "vocab_size", "max_encoder_length",
        "max_decoder_length", "relative_attention_max_distance",
    ],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_dimensions_must_be_positive(field, bad):
    with pytest.raises(ConfigError, match=field):
        ModelConfig(**{**TINY, field: bad})


def test_dimensions_must_be_integers():
    with pytest.raises(ConfigError, match="d_model"):
        ModelConfig(**{**TINY, "d_model": 64.0})


def test_booleans_are_not_accepted_as_dimensions():
    """`True` is an int subclass in Python; it is still not a layer count."""
    with pytest.raises(ConfigError, match="num_heads"):
        ModelConfig(**{**TINY, "num_heads": True})


def test_bucket_count_must_be_even():
    """The bidirectional encoder bucketing splits the range in half."""
    with pytest.raises(ConfigError, match="must be even"):
        ModelConfig(**{**TINY, "relative_attention_num_buckets": 9})


def test_bucket_count_must_be_at_least_four():
    with pytest.raises(ConfigError, match="relative_attention_num_buckets"):
        ModelConfig(**{**TINY, "relative_attention_num_buckets": 2})


def test_max_distance_must_exceed_the_exact_region():
    """Otherwise every distant position collapses into one bucket."""
    with pytest.raises(ConfigError, match="max_distance"):
        ModelConfig(
            **{**TINY, "relative_attention_num_buckets": 32,
               "relative_attention_max_distance": 4}
        )


def test_unknown_activation_rejected():
    with pytest.raises(ConfigError, match="feed_forward_proj"):
        ModelConfig(**{**TINY, "feed_forward_proj": "swiglu"})


def test_ungated_activations_are_rejected():
    """
    HuggingFace accepts plain "relu"/"gelu" for T5's original FFN. This model
    is GeGLU-only (architecture.md §4.1) and its §4.3 parameter budget assumes
    three matrices, so an ungated value must fail rather than be misread as
    gated.
    """
    with pytest.raises(ConfigError, match="feed_forward_proj"):
        ModelConfig(**{**TINY, "feed_forward_proj": "relu"})
    with pytest.raises(ConfigError, match="feed_forward_proj"):
        ModelConfig(**{**TINY, "feed_forward_proj": "gelu"})


@pytest.mark.parametrize("field", ["dropout_rate", "label_smoothing"])
@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_probabilities_must_be_in_range(field, bad):
    with pytest.raises(ConfigError, match=field):
        ModelConfig(**{**TINY, field: bad})


@pytest.mark.parametrize("field", ["layer_norm_epsilon", "initializer_factor"])
def test_positive_floats_required(field):
    with pytest.raises(ConfigError, match=field):
        ModelConfig(**{**TINY, field: 0.0})


@pytest.mark.parametrize("field", ["pad_token_id", "decoder_start_token_id", "eos_token_id"])
def test_token_ids_must_be_inside_the_vocabulary(field):
    with pytest.raises(ConfigError, match=field):
        ModelConfig(**{**TINY, field: TINY["vocab_size"]})


@pytest.mark.parametrize("field", ["pad_token_id", "decoder_start_token_id", "eos_token_id"])
def test_token_ids_must_be_non_negative(field):
    with pytest.raises(ConfigError, match=field):
        ModelConfig(**{**TINY, field: -1})


def test_label_pad_must_be_negative():
    """It must be impossible for a padded label to be a real token."""
    with pytest.raises(ConfigError, match="label_pad_token_id"):
        ModelConfig(**{**TINY, "label_pad_token_id": 0})


def test_expected_parameters_must_be_positive_or_absent():
    ModelConfig(**{**TINY, "expected_parameters": None})
    ModelConfig(**{**TINY, "expected_parameters": 1})
    with pytest.raises(ConfigError, match="expected_parameters"):
        ModelConfig(**{**TINY, "expected_parameters": 0})


# -- derived values -------------------------------------------------------


def test_inner_dim_is_heads_times_d_kv(tiny_config):
    assert tiny_config.inner_dim == tiny_config.num_heads * tiny_config.d_kv


def test_parameter_count_is_independent_of_context_length(large_config):
    """
    architecture.md §4.4: relative position bias has no per-position
    parameters, so 2048 and 4096 produce byte-identical model files. The
    entire cost of a longer context is memory and time.
    """
    longer = large_config.with_(max_encoder_length=4096, max_decoder_length=4096)
    assert longer.parameter_count() == large_config.parameter_count()


def test_untying_embeddings_adds_a_vocabulary_matrix(large_config):
    untied = large_config.with_(tie_word_embeddings=False)
    difference = untied.parameter_count() - large_config.parameter_count()
    assert difference == large_config.vocab_size * large_config.d_model


def test_parameter_breakdown_sums_to_the_total(large_config):
    breakdown = large_config.parameter_breakdown()
    components = (
        breakdown["token_embeddings"]
        + breakdown["encoder_total"]
        + breakdown["decoder_total"]
        + breakdown["relative_position_bias"]
        + breakdown["final_norms"]
    )
    assert components == breakdown["total"] == large_config.parameter_count()


def test_decoder_layer_is_larger_than_encoder_layer(large_config):
    """The decoder carries cross-attention and a third norm."""
    breakdown = large_config.parameter_breakdown()
    difference = breakdown["decoder_per_layer"] - breakdown["encoder_per_layer"]
    attention = 4 * large_config.d_model * large_config.inner_dim
    assert difference == attention + large_config.d_model


# -- serialization --------------------------------------------------------


def test_to_dict_from_dict_round_trip(large_config):
    assert ModelConfig.from_dict(large_config.to_dict()) == large_config


def test_from_dict_rejects_unknown_keys(large_config):
    payload = large_config.to_dict()
    payload["d_modle"] = 1024
    with pytest.raises(ConfigError, match="unknown"):
        ModelConfig.from_dict(payload)


def test_from_dict_rejects_missing_keys(large_config):
    payload = large_config.to_dict()
    del payload["num_heads"]
    with pytest.raises(ConfigError, match="missing"):
        ModelConfig.from_dict(payload)


def test_with_creates_a_validated_copy(large_config):
    narrowed = large_config.with_(num_layers=6)
    assert narrowed.num_layers == 6
    assert large_config.num_layers == 24, "the original must not be mutated"
    with pytest.raises(ConfigError):
        large_config.with_(num_heads=0)


def test_config_is_immutable(large_config):
    """These values are welded into every checkpoint's weight shapes."""
    with pytest.raises(Exception):
        large_config.d_model = 2048
