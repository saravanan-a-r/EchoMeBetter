"""
Regression lock on the frozen decisions in architecture.md §4.

This file tests agreement between the code, `model_config.yml`, and the
design document — not behaviour. A failure means someone is about to change
the model's architecture, and should be read as "stop and go read section N",
not as "update the expected number".

Where the document justifies a value with arithmetic, the arithmetic is
re-derived here rather than restated, so the justification is checked rather
than copied.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MODEL_CONFIG_PATH
from src.blocks import DecoderLayer, EncoderLayer
from src.config import load_model_config
from src.layers import GeGLUFeedForward, RMSNorm
from src.seq2seq import RephraseSeq2Seq


@pytest.fixture(scope="module")
def large():
    return load_model_config(MODEL_CONFIG_PATH, profile="large")


@pytest.fixture(scope="module")
def base():
    return load_model_config(MODEL_CONFIG_PATH, profile="base")


# -- §4.2 size variants, FROZEN ------------------------------------------


def test_large_dimensions(large):
    """architecture.md §4.2, decision D2 (FROZEN)."""
    assert large.d_model == 1024
    assert large.d_ff == 2816
    assert large.num_layers == 24
    assert large.num_decoder_layers == 24
    assert large.num_heads == 16
    assert large.d_kv == 64


def test_base_dimensions(base):
    """The §7.8 rehearsal proxy — not a fallback, it has a job."""
    assert base.d_model == 768
    assert base.d_ff == 2048
    assert base.num_layers == 12
    assert base.num_decoder_layers == 12
    assert base.num_heads == 12
    assert base.d_kv == 64


def test_encoder_and_decoder_depth_are_equal(large, base):
    """
    architecture.md §4.7: a transformation task needs balanced understanding
    and generation. (Summarization can use a deep encoder and shallow decoder
    because output is much shorter than input; for rephrase, output ≈ input.)
    """
    for config in (large, base):
        assert config.num_layers == config.num_decoder_layers


def test_attention_width_equals_the_model_width(large, base):
    for config in (large, base):
        assert config.num_heads * config.d_kv == config.d_model


def test_d_kv_is_64(large, base):
    """architecture.md §4.7: the stability/quality sweet spot across scales."""
    assert large.d_kv == base.d_kv == 64


def test_d_ff_is_about_2_75_times_d_model(large, base):
    """
    architecture.md §4.7: a GeGLU FFN has 3 matrices where a ReLU FFN has 2,
    so 2.75x matches the parameter count of a conventional 4x ReLU FFN.
    """
    for config in (large, base):
        ratio = config.d_ff / config.d_model
        assert 2.6 <= ratio <= 2.9, f"{config.profile}: d_ff/d_model = {ratio:.2f}"


# -- §4.3 parameter budget -----------------------------------------------


def test_the_documented_total(large):
    assert large.parameter_count() == 750_906_368
    assert large.expected_parameters == 750_906_368


def test_the_documented_base_total(base):
    assert base.parameter_count() == 223_395_072
    assert base.expected_parameters == 223_395_072


def test_bf16_serving_size_is_about_1_5_gb(large):
    """architecture.md §4.3: ~1,502 MB in bf16, the default serving format."""
    assert round(large.parameter_count() * 2 / 1_000_000) == 1502


# -- §4.1 backbone choices, FROZEN ---------------------------------------


def test_embeddings_are_tied(large):
    """architecture.md §4.1: saves 33.5M params at zero quality cost."""
    assert large.tie_word_embeddings is True


def test_tying_saves_the_documented_amount(large):
    untied = large.with_(tie_word_embeddings=False, expected_parameters=None)
    saved = untied.parameter_count() - large.parameter_count()
    assert saved == 33_554_432


def test_the_model_uses_rmsnorm_not_layernorm(tiny_config):
    model = RephraseSeq2Seq(tiny_config)
    assert any(isinstance(m, RMSNorm) for m in model.modules())
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())


def test_the_feed_forward_is_geglu(tiny_config):
    model = RephraseSeq2Seq(tiny_config)
    assert any(isinstance(m, GeGLUFeedForward) for m in model.modules())


def test_no_projection_anywhere_carries_a_bias(tiny_config):
    """
    The parameter budget in §4.3 assumes this everywhere. One stray
    `bias=True` changes the model's size without changing anything visible.
    """
    model = RephraseSeq2Seq(tiny_config)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, name


def test_layers_are_pre_norm(tiny_config):
    """
    architecture.md §4.1. Zeroing every sub-layer output must leave a layer
    as the exact identity, which is only true if the residual path bypasses
    normalization entirely.
    """
    layer = EncoderLayer(tiny_config).eval()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.ffn.down_proj.weight.zero_()
    x = torch.randn(2, 5, tiny_config.d_model)
    assert torch.allclose(layer(x), x, atol=1e-6)


def test_dropout_defaults_to_zero_for_pretraining(large):
    """
    architecture.md §4.1: pretraining data is effectively unlimited, so
    dropout would only slow learning. Stage 2 (SFT) raises it to 0.1.
    """
    assert large.dropout_rate == 0.0


# -- §4.4 context length, FROZEN -----------------------------------------


def test_encoder_and_decoder_contexts_are_both_2048(large):
    """
    architecture.md §4.4 (FROZEN). The decoder was raised from 1024 for two
    independent reasons: <style:elaborate> needs to exceed its input, and
    X-denoising targets (~1,057 tokens) overflow a 1024 decoder — which
    would corrupt the pretraining objective itself.
    """
    assert large.max_encoder_length == 2048
    assert large.max_decoder_length == 2048


def test_a_1024_decoder_could_not_hold_an_x_denoising_target(large):
    """Re-derives §4.4's arithmetic: 50% of 2048, plus one sentinel per span."""
    corrupted = int(large.max_encoder_length * 0.5)
    sentinels = corrupted // 32
    target = corrupted + sentinels + 1

    assert target > 1024, "the reason 1024 was insufficient"
    assert target <= large.max_decoder_length, "and the reason 2048 suffices"


def test_context_length_costs_no_parameters(large):
    """
    architecture.md §4.4's key claim: relative position bias has no
    per-position parameters, so 2048 and 4096 produce byte-identical model
    files. The entire cost of a longer context is memory and time.
    """
    doubled = large.with_(
        max_encoder_length=4096, max_decoder_length=4096, expected_parameters=None
    )
    assert doubled.parameter_count() == large.parameter_count()


# -- §4.5 over-length policy, FROZEN -------------------------------------


def test_over_length_input_is_rejected_never_truncated(tiny_model, tiny_config):
    """
    architecture.md §4.5 (FROZEN): "Silent truncation is forbidden." A
    rephrase of a truncated input looks valid and is not.
    """
    from src.errors import ShapeError

    too_long = torch.zeros(1, tiny_config.max_encoder_length + 1, dtype=torch.long)
    with pytest.raises(ShapeError):
        tiny_model.encode(too_long)


# -- §4.6 relative position bias -----------------------------------------


def test_position_bias_uses_the_t5_defaults(large):
    """
    architecture.md §4.6: keep the proven defaults rather than scaling them
    with context on speculation. Evaluation is bucketed by input length
    (§7.7) so length-dependent degradation would be visible if it appeared.
    """
    assert large.relative_attention_num_buckets == 32
    assert large.relative_attention_max_distance == 128


def test_one_position_bias_table_per_stack(tiny_config):
    """
    architecture.md §4.3 counts `(32 x 16) x 2 stacks`, not per layer. One
    table per layer would multiply that by 24.
    """
    model = RephraseSeq2Seq(tiny_config)
    tables = [
        name
        for name, _ in model.named_parameters()
        if "position_bias" in name
    ]
    assert sorted(tables) == [
        "decoder.position_bias.embedding.weight",
        "encoder.position_bias.embedding.weight",
    ]


def test_position_bias_parameter_count(large):
    assert large.parameter_breakdown()["relative_position_bias"] == 1024
    assert 2 * large.relative_attention_num_buckets * large.num_heads == 1024


def test_the_encoder_is_bidirectional_and_the_decoder_is_not(tiny_config):
    model = RephraseSeq2Seq(tiny_config)
    assert model.encoder.position_bias.bidirectional is True
    assert model.decoder.position_bias.bidirectional is False


def test_cross_attention_carries_no_position_bias(tiny_config, monkeypatch):
    """
    Content-based only — there is no meaningful distance between a decoder
    position and an encoder position. Easy to "fix" by mistake.
    """
    layer = DecoderLayer(tiny_config).eval()
    seen = {}
    original = layer.cross_attn.forward

    def record(*args, **kwargs):
        seen["bias"] = kwargs.get("position_bias", "absent")
        return original(*args, **kwargs)

    monkeypatch.setattr(layer.cross_attn, "forward", record)
    layer(
        torch.randn(1, 3, tiny_config.d_model),
        torch.randn(1, 5, tiny_config.d_model),
        position_bias=torch.randn(1, tiny_config.num_heads, 3, 3),
    )
    assert seen["bias"] is None


# -- §6.1 tokenizer agreement, FROZEN ------------------------------------


def test_vocab_size_matches_the_tokenizer(large, base):
    """
    architecture.md §6.1 (FROZEN): 32,768. Must equal the trained
    tokenizer's vocabulary or every embedding row is misaligned.
    """
    assert large.vocab_size == base.vocab_size == 32768


def test_there_is_no_bos_token(large):
    """
    architecture.md §6.1: bos_id = -1, so T5's convention of starting the
    decoder with <pad> applies.
    """
    assert large.decoder_start_token_id == large.pad_token_id


def test_label_padding_matches_the_ul2_objective(large):
    """Both modules configure -100 independently; they must agree."""
    assert large.label_pad_token_id == -100


# -- config file hygiene --------------------------------------------------


def test_the_active_profile_is_a_real_profile():
    from src.config import available_profiles

    assert load_model_config(MODEL_CONFIG_PATH).profile in available_profiles(
        MODEL_CONFIG_PATH
    )


def test_profiles_differ_only_in_shape(large, base):
    """
    Everything except the six shape fields must come from `defaults`, so the
    two profiles cannot drift apart on a behavioural setting.
    """
    shape_fields = {
        "profile", "d_model", "d_ff", "num_layers",
        "num_decoder_layers", "num_heads", "expected_parameters",
    }
    large_settings = {k: v for k, v in large.to_dict().items() if k not in shape_fields}
    base_settings = {k: v for k, v in base.to_dict().items() if k not in shape_fields}
    assert large_settings == base_settings
