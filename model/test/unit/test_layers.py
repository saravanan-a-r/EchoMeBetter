"""
Unit tests for RMSNorm and the GeGLU feed-forward.

Two things are being locked down: the arithmetic (checked against an
independent implementation rather than against itself), and the **absence of
bias parameters**. Every bias added here would silently change the model's
size, so the parameter budget in architecture.md §4.3 depends on these
modules staying bias-free.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.layers import GeGLUFeedForward, RMSNorm


# -- RMSNorm --------------------------------------------------------------


def test_rmsnorm_matches_an_independent_implementation():
    torch.manual_seed(0)
    norm = RMSNorm(8, eps=1e-6)
    norm.weight.data.normal_()
    x = torch.randn(3, 5, 8)

    expected = norm.weight * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_does_not_subtract_the_mean():
    """
    The defining difference from LayerNorm. Adding a constant to every
    channel must change the output — with mean subtraction it would not.
    """
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    assert not torch.allclose(norm(x), norm(x + 5.0), atol=1e-4)


def test_rmsnorm_normalizes_scale():
    """Scaling the input by a constant leaves the output unchanged."""
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    assert torch.allclose(norm(x), norm(x * 7.0), atol=1e-5)


def test_rmsnorm_has_a_weight_and_no_bias():
    """architecture.md §4.3 counts exactly `d_model` parameters per norm."""
    norm = RMSNorm(32)
    names = [name for name, _ in norm.named_parameters()]
    assert names == ["weight"]
    assert sum(p.numel() for p in norm.parameters()) == 32


def test_rmsnorm_is_identity_at_initialization():
    """Weight starts at 1.0, so the norm only rescales."""
    norm = RMSNorm(8)
    x = torch.ones(1, 1, 8) * 3.0
    assert torch.allclose(norm(x), torch.ones(1, 1, 8), atol=1e-5)


def test_rmsnorm_preserves_shape_and_dtype():
    norm = RMSNorm(16)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 3, 16, dtype=dtype)
        output = norm.to(dtype)(x)
        assert output.shape == x.shape
        assert output.dtype == dtype


def test_rmsnorm_computes_in_float32_under_bfloat16():
    """
    Squaring a bf16 activation loses precision exactly where it matters —
    the sum of squares over a wide hidden dimension. The internal float32
    cast is what keeps a bf16 run stable, so it must survive refactoring.
    """
    torch.manual_seed(0)
    norm = RMSNorm(256)
    x = torch.randn(1, 1, 256) * 100.0

    reference = norm(x)
    in_bf16 = norm.to(torch.bfloat16)(x.to(torch.bfloat16)).float()

    # Close despite the storage precision, because the reduction was not done
    # in bf16. A naive bf16 reduction drifts much further than this.
    assert torch.allclose(reference, in_bf16, atol=2e-1, rtol=2e-2)


def test_rmsnorm_handles_an_all_zero_row():
    """eps must prevent a division by zero producing NaN."""
    output = RMSNorm(8)(torch.zeros(1, 1, 8))
    assert torch.isfinite(output).all()


# -- GeGLU ----------------------------------------------------------------


def test_geglu_shapes(tiny_config):
    ffn = GeGLUFeedForward(tiny_config)
    x = torch.randn(2, 5, tiny_config.d_model)
    assert ffn(x).shape == x.shape


def test_geglu_has_three_matrices_and_no_bias(tiny_config):
    """
    Three matrices where a plain FFN has two — the reason d_ff is ~2.75x
    d_model rather than 4x (architecture.md §4.7).
    """
    ffn = GeGLUFeedForward(tiny_config)
    names = sorted(name for name, _ in ffn.named_parameters())
    assert names == ["down_proj.weight", "gate_proj.weight", "up_proj.weight"]

    expected = 3 * tiny_config.d_model * tiny_config.d_ff
    assert sum(p.numel() for p in ffn.parameters()) == expected


def test_geglu_computes_the_gated_product(tiny_config):
    """Checked against the formula, not against itself."""
    torch.manual_seed(0)
    ffn = GeGLUFeedForward(tiny_config).eval()
    x = torch.randn(2, 3, tiny_config.d_model)

    gate = torch.nn.functional.gelu(ffn.gate_proj(x), approximate="tanh")
    expected = ffn.down_proj(gate * ffn.up_proj(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)


def test_geglu_gate_actually_gates(tiny_config):
    """
    Zeroing the gate projection must zero the output regardless of the up
    branch. If the two were added rather than multiplied, it would not.
    """
    ffn = GeGLUFeedForward(tiny_config).eval()
    with torch.no_grad():
        ffn.gate_proj.weight.zero_()
    x = torch.randn(2, 3, tiny_config.d_model)
    # GELU(0) == 0, so the product is zero everywhere.
    assert torch.allclose(ffn(x), torch.zeros_like(ffn(x)), atol=1e-7)


@pytest.mark.parametrize("activation", ["gated-gelu_new", "gated-gelu"])
def test_both_activations_are_usable(tiny_config, activation):
    ffn = GeGLUFeedForward(tiny_config.with_(feed_forward_proj=activation)).eval()
    output = ffn(torch.randn(2, 3, tiny_config.d_model))
    assert torch.isfinite(output).all()


def test_both_spellings_are_the_same_activation(tiny_config):
    """
    "gated-gelu" and "gated-gelu_new" must behave identically, because
    transformers special-cases the former to mean `gelu_new`. If they ever
    diverged here, the same config.json would build two different FFNs
    depending on which spelling was used.
    """
    torch.manual_seed(0)
    standard = GeGLUFeedForward(tiny_config.with_(feed_forward_proj="gated-gelu")).eval()
    torch.manual_seed(0)
    synonym = GeGLUFeedForward(
        tiny_config.with_(feed_forward_proj="gated-gelu_new")
    ).eval()

    x = torch.randn(2, 3, tiny_config.d_model) * 3.0
    assert torch.equal(standard(x), synonym(x))


def test_the_activation_is_the_tanh_approximation(tiny_config):
    """
    architecture.md §4.1 specifies T5.1.1's gated-GELU, which is the tanh
    approximation. The erf-based GELU is a different function — close enough
    to look fine, far enough that a checkpoint trained under one and
    fine-tuned under the other is a silent mismatch. This pins which one.
    """
    ffn = GeGLUFeedForward(tiny_config).eval()
    x = torch.randn(4, 5, tiny_config.d_model) * 3.0

    assert torch.equal(ffn._activate(x), torch.nn.functional.gelu(x, approximate="tanh"))
    assert not torch.allclose(ffn._activate(x), torch.nn.functional.gelu(x), atol=1e-9)


def test_an_unimplemented_activation_fails_at_construction(tiny_config):
    """
    Adding a `feed_forward_proj` value without implementing its activation
    must fail loudly at construction, not fall through to whichever branch
    happened to come last.
    """
    from src.errors import ConfigError

    class _Unsupported:
        """A config whose `dense_act_fn` has no implementation."""

        dense_act_fn = "swish_glu_9000"

        def __getattr__(self, name):
            return getattr(tiny_config, name)

    with pytest.raises(ConfigError, match="dense_act_fn"):
        GeGLUFeedForward(_Unsupported())


def test_geglu_dropout_is_active_only_in_training(tiny_config):
    ffn = GeGLUFeedForward(tiny_config.with_(dropout_rate=0.5))
    x = torch.randn(4, 8, tiny_config.d_model)

    ffn.train()
    torch.manual_seed(0)
    first = ffn(x)
    torch.manual_seed(1)
    second = ffn(x)
    assert not torch.allclose(first, second)

    ffn.eval()
    assert torch.allclose(ffn(x), ffn(x))


def test_geglu_gradients_reach_every_matrix(tiny_config):
    ffn = GeGLUFeedForward(tiny_config)
    ffn(torch.randn(2, 3, tiny_config.d_model)).sum().backward()
    for name, parameter in ffn.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_no_module_here_uses_a_bias(tiny_config):
    """Guards the parameter budget against a stray `bias=True`."""
    ffn = GeGLUFeedForward(tiny_config)
    for module in ffn.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None, "T5 convention: no biases on projections"
