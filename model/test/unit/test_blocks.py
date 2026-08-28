"""
Unit tests for the encoder and decoder layers.

Two things are locked here that nothing else can catch:

  1. **Per-layer parameter counts**, checked against architecture.md §4.3's
     arithmetic. These are what the 750,906,368 total is built from; a stray
     bias or an extra norm changes the model's size silently.

  2. **Cross-attention receives no position bias.** T5 cross-attention is
     purely content-based — there is no meaningful distance between a decoder
     position and an encoder position. This is easy to "fix" by mistake, so
     it is asserted directly.
"""

from __future__ import annotations

import torch

from src.blocks import DecoderLayer, EncoderLayer


def _random_init(module):
    torch.manual_seed(0)
    for parameter in module.parameters():
        parameter.data.normal_(0, 0.05)
    return module.eval()


# -- parameter budget -----------------------------------------------------


def test_encoder_layer_parameter_count(tiny_config):
    """architecture.md §4.3: self-attention + GeGLU + 2 RMSNorm."""
    expected = (
        4 * tiny_config.d_model * tiny_config.inner_dim
        + 3 * tiny_config.d_model * tiny_config.d_ff
        + 2 * tiny_config.d_model
    )
    assert sum(p.numel() for p in EncoderLayer(tiny_config).parameters()) == expected


def test_decoder_layer_parameter_count(tiny_config):
    """architecture.md §4.3: self + cross attention, GeGLU, 3 RMSNorm."""
    expected = (
        2 * (4 * tiny_config.d_model * tiny_config.inner_dim)
        + 3 * tiny_config.d_model * tiny_config.d_ff
        + 3 * tiny_config.d_model
    )
    assert sum(p.numel() for p in DecoderLayer(tiny_config).parameters()) == expected


def test_encoder_layer_has_exactly_two_norms(tiny_config):
    names = [n for n, _ in EncoderLayer(tiny_config).named_parameters() if n.endswith("norm.weight")]
    assert sorted(names) == ["ffn_norm.weight", "self_attn_norm.weight"]


def test_decoder_layer_has_exactly_three_norms(tiny_config):
    names = [n for n, _ in DecoderLayer(tiny_config).named_parameters() if n.endswith("norm.weight")]
    assert sorted(names) == [
        "cross_attn_norm.weight",
        "ffn_norm.weight",
        "self_attn_norm.weight",
    ]


# -- encoder layer --------------------------------------------------------


def test_encoder_layer_preserves_shape(tiny_config):
    layer = _random_init(EncoderLayer(tiny_config))
    x = torch.randn(2, 7, tiny_config.d_model)
    assert layer(x).shape == x.shape


def test_encoder_layer_is_residual(tiny_config):
    """
    With both sub-layer outputs zeroed, a pre-norm layer must be the exact
    identity — proof the residual path is unnormalized end to end.
    """
    layer = EncoderLayer(tiny_config).eval()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.ffn.down_proj.weight.zero_()

    x = torch.randn(2, 5, tiny_config.d_model)
    assert torch.allclose(layer(x), x, atol=1e-6)


def test_encoder_layer_uses_its_position_bias(tiny_config):
    layer = _random_init(EncoderLayer(tiny_config))
    x = torch.randn(2, 6, tiny_config.d_model)
    bias = torch.randn(1, tiny_config.num_heads, 6, 6)
    assert not torch.allclose(layer(x), layer(x, position_bias=bias), atol=1e-6)


def test_encoder_layer_gradients_reach_everything(tiny_config):
    layer = EncoderLayer(tiny_config)
    layer(torch.randn(2, 5, tiny_config.d_model)).sum().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name


# -- decoder layer --------------------------------------------------------


def test_decoder_layer_preserves_shape(tiny_config):
    layer = _random_init(DecoderLayer(tiny_config))
    x = torch.randn(2, 4, tiny_config.d_model)
    memory = torch.randn(2, 9, tiny_config.d_model)
    output, cache = layer(x, memory)
    assert output.shape == x.shape
    assert cache is None, "no cache unless requested"


def test_decoder_layer_is_residual(tiny_config):
    layer = DecoderLayer(tiny_config).eval()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.cross_attn.o_proj.weight.zero_()
        layer.ffn.down_proj.weight.zero_()

    x = torch.randn(2, 4, tiny_config.d_model)
    output, _ = layer(x, torch.randn(2, 6, tiny_config.d_model))
    assert torch.allclose(output, x, atol=1e-6)


def test_decoder_layer_attends_to_the_encoder(tiny_config):
    """Changing the encoder output must change the decoder output."""
    layer = _random_init(DecoderLayer(tiny_config))
    x = torch.randn(2, 4, tiny_config.d_model)
    first, _ = layer(x, torch.randn(2, 6, tiny_config.d_model))
    second, _ = layer(x, torch.randn(2, 6, tiny_config.d_model))
    assert not torch.allclose(first, second)


def test_cross_attention_has_no_position_bias(tiny_config, monkeypatch):
    """
    Regression lock. T5 cross-attention is content-based only; there is no
    distance between a decoder position and an encoder position to encode.
    Intercepting the call is the only way to prove nothing was passed.
    """
    layer = _random_init(DecoderLayer(tiny_config))
    seen = {}

    original = layer.cross_attn.forward

    def record(*args, **kwargs):
        seen["position_bias"] = kwargs.get("position_bias", "absent")
        return original(*args, **kwargs)

    monkeypatch.setattr(layer.cross_attn, "forward", record)
    layer(
        torch.randn(2, 4, tiny_config.d_model),
        torch.randn(2, 6, tiny_config.d_model),
        position_bias=torch.randn(1, tiny_config.num_heads, 4, 4),
    )
    assert seen["position_bias"] is None


def test_decoder_layer_self_attention_uses_its_position_bias(tiny_config):
    """The counterpart: self-attention *does* receive the bias."""
    layer = _random_init(DecoderLayer(tiny_config))
    x = torch.randn(2, 4, tiny_config.d_model)
    memory = torch.randn(2, 6, tiny_config.d_model)
    bias = torch.randn(1, tiny_config.num_heads, 4, 4)

    plain, _ = layer(x, memory)
    biased, _ = layer(x, memory, position_bias=bias)
    assert not torch.allclose(plain, biased, atol=1e-6)


def test_decoder_layer_returns_a_populated_cache(tiny_config):
    layer = _random_init(DecoderLayer(tiny_config))
    _, cache = layer(
        torch.randn(2, 3, tiny_config.d_model),
        torch.randn(2, 6, tiny_config.d_model),
        use_cache=True,
    )
    assert cache is not None
    assert cache.self_key.shape == (2, tiny_config.num_heads, 3, tiny_config.d_kv)
    assert cache.cross_key.shape == (2, tiny_config.num_heads, 6, tiny_config.d_kv)


def test_decoder_layer_gradients_reach_everything(tiny_config):
    layer = DecoderLayer(tiny_config)
    output, _ = layer(
        torch.randn(2, 4, tiny_config.d_model), torch.randn(2, 6, tiny_config.d_model)
    )
    output.sum().backward()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
