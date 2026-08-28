"""
Unit tests for multi-head attention and its masks.

Mask bugs are the reason this file is long. They do not crash, they do not
produce NaN, and they do not show up as a bad loss curve — a decoder that can
see one token into the future simply learns to copy it, trains beautifully,
and generates nonsense. Every mask path is therefore tested by *behaviour*
(does changing a masked position change the output?) rather than by
inspecting the mask tensor.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.attention import (
    LayerCache,
    MultiHeadAttention,
    build_causal_mask,
    build_padding_mask,
)


def _attention(config, cross=False):
    torch.manual_seed(0)
    module = MultiHeadAttention(config, is_cross_attention=cross).eval()
    for parameter in module.parameters():
        parameter.data.normal_(0, 0.05)
    return module


# -- shapes and parameters ------------------------------------------------


def test_output_shape_matches_input(tiny_config):
    attention = _attention(tiny_config)
    x = torch.randn(2, 7, tiny_config.d_model)
    output, key, value = attention(x)
    assert output.shape == x.shape
    assert key is None and value is None, "no cache unless requested"


def test_parameter_count_is_four_projections(tiny_config):
    """architecture.md §4.3 counts `4 x d_model x inner_dim` per attention."""
    attention = MultiHeadAttention(tiny_config)
    expected = 4 * tiny_config.d_model * tiny_config.inner_dim
    assert sum(p.numel() for p in attention.parameters()) == expected


def test_no_projection_uses_a_bias(tiny_config):
    attention = MultiHeadAttention(tiny_config)
    for module in attention.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None


def test_heads_are_split_and_merged_consistently(tiny_config):
    attention = _attention(tiny_config)
    x = torch.randn(2, 5, tiny_config.d_model)
    split = attention._split_heads(x)
    assert split.shape == (2, tiny_config.num_heads, 5, tiny_config.d_kv)
    assert torch.allclose(attention._merge_heads(split), x)


# -- the T5 scaling convention -------------------------------------------


def test_scores_are_unscaled_by_default(tiny_config):
    """
    T5 folds 1/sqrt(d_kv) into the query initializer instead. The two are
    coupled; `model_config.yml` says so at `scale_attention_scores`.
    """
    assert MultiHeadAttention(tiny_config).scale == 1.0


def test_scaling_can_be_enabled(tiny_config):
    attention = MultiHeadAttention(tiny_config.with_(scale_attention_scores=True))
    assert attention.scale == pytest.approx(tiny_config.d_kv**-0.5)


def test_scaling_changes_the_output(tiny_config):
    x = torch.randn(2, 6, tiny_config.d_model)
    unscaled = _attention(tiny_config)(x)[0]
    scaled = _attention(tiny_config.with_(scale_attention_scores=True))(x)[0]
    assert not torch.allclose(unscaled, scaled, atol=1e-5)


# -- masking --------------------------------------------------------------


def test_padding_mask_blocks_padded_keys(tiny_config):
    """
    Behavioural test: change the content at padded positions and the output
    must not move at all.
    """
    attention = _attention(tiny_config)
    x = torch.randn(2, 8, tiny_config.d_model)
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[:, 5:] = 0
    additive = build_padding_mask(mask, x.dtype)

    first = attention(x, attention_mask=additive)[0]
    perturbed = x.clone()
    perturbed[:, 5:] = torch.randn_like(perturbed[:, 5:]) * 10
    second = attention(perturbed, attention_mask=additive)[0]

    assert torch.allclose(first[:, :5], second[:, :5], atol=1e-6)


def test_padding_mask_values(tiny_config):
    mask = build_padding_mask(torch.tensor([[1, 1, 0]]), torch.float32)
    assert mask.shape == (1, 1, 1, 3)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 2] == torch.finfo(torch.float32).min


def test_a_fully_masked_row_does_not_produce_nan(tiny_config):
    """
    A completely padded row is possible in a real ragged batch. Using
    finfo.min rather than -inf keeps softmax finite, so one bad row cannot
    poison the whole loss.
    """
    attention = _attention(tiny_config)
    x = torch.randn(2, 6, tiny_config.d_model)
    mask = torch.zeros(2, 6, dtype=torch.long)
    output = attention(x, attention_mask=build_padding_mask(mask, x.dtype))[0]
    assert torch.isfinite(output).all()


def test_causal_mask_shape_and_values():
    mask = build_causal_mask(4, 4, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] == torch.finfo(torch.float32).min
    assert mask[0, 0, 3, 0] == 0.0


def test_causal_mask_blocks_the_future(tiny_config):
    """The bug that trains well and generates nonsense."""
    attention = _attention(tiny_config)
    x = torch.randn(1, 6, tiny_config.d_model)
    mask = build_causal_mask(6, 6, x.dtype, x.device)

    first = attention(x, attention_mask=mask)[0]
    perturbed = x.clone()
    perturbed[:, 4:] = torch.randn_like(perturbed[:, 4:]) * 10
    second = attention(perturbed, attention_mask=mask)[0]

    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6)
    assert not torch.allclose(first[:, 4:], second[:, 4:], atol=1e-6)


def test_causal_mask_with_a_cached_prefix_sees_all_of_it():
    """
    During generation the single query sits at the end of the key range, so
    the whole cached prefix must be visible and nothing masked.
    """
    mask = build_causal_mask(1, 5, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 1, 1, 5)
    assert (mask == 0.0).all()


def test_causal_mask_for_a_partial_chunk():
    """Two new queries over a 3-token prefix: each sees the prefix plus itself."""
    mask = build_causal_mask(2, 5, torch.float32, torch.device("cpu"))[0, 0]
    blocked = torch.finfo(torch.float32).min
    assert (mask[0, :4] == 0.0).all() and mask[0, 4] == blocked
    assert (mask[1, :5] == 0.0).all()


# -- cross-attention ------------------------------------------------------


def test_cross_attention_reads_the_other_sequence(tiny_config):
    attention = _attention(tiny_config, cross=True)
    queries = torch.randn(2, 4, tiny_config.d_model)
    memory = torch.randn(2, 9, tiny_config.d_model)
    assert attention(queries, key_value_states=memory)[0].shape == queries.shape


def test_cross_attention_requires_a_memory(tiny_config):
    attention = _attention(tiny_config, cross=True)
    with pytest.raises(ValueError, match="key_value_states"):
        attention(torch.randn(1, 3, tiny_config.d_model))


def test_cross_attention_output_depends_on_the_memory(tiny_config):
    attention = _attention(tiny_config, cross=True)
    queries = torch.randn(2, 4, tiny_config.d_model)
    first = attention(queries, key_value_states=torch.randn(2, 9, tiny_config.d_model))[0]
    second = attention(queries, key_value_states=torch.randn(2, 9, tiny_config.d_model))[0]
    assert not torch.allclose(first, second)


# -- caching --------------------------------------------------------------


def test_self_attention_cache_extends_the_prefix(tiny_config):
    attention = _attention(tiny_config)
    x = torch.randn(2, 1, tiny_config.d_model)

    _, key, value = attention(x, use_cache=True)
    assert key.shape == (2, tiny_config.num_heads, 1, tiny_config.d_kv)

    _, key2, value2 = attention(x, past_key=key, past_value=value, use_cache=True)
    assert key2.shape[2] == 2
    assert torch.allclose(key2[:, :, :1], key)


def test_incremental_self_attention_matches_a_full_pass(tiny_config):
    """
    The equivalence generation depends on. Any mismatch here means the model
    behaves differently at inference than it was trained to.
    """
    attention = _attention(tiny_config)
    x = torch.randn(2, 6, tiny_config.d_model)

    full = attention(x, attention_mask=build_causal_mask(6, 6, x.dtype, x.device))[0]

    key = value = None
    steps = []
    for i in range(6):
        output, key, value = attention(
            x[:, i : i + 1], past_key=key, past_value=value, use_cache=True
        )
        steps.append(output)

    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_cross_attention_cache_skips_recomputation(tiny_config):
    """
    The encoder output does not change between steps, so neither do its keys
    and values. Passing a cache must bypass the projections entirely.
    """
    attention = _attention(tiny_config, cross=True)
    queries = torch.randn(2, 1, tiny_config.d_model)
    memory = torch.randn(2, 7, tiny_config.d_model)

    first, key, value = attention(queries, key_value_states=memory, use_cache=True)

    # Passing garbage as the memory must be irrelevant once cached.
    cached, _, _ = attention(
        queries,
        key_value_states=torch.randn(2, 7, tiny_config.d_model) * 100,
        past_key=key,
        past_value=value,
        use_cache=True,
    )
    assert torch.allclose(first, cached, atol=1e-6)


# -- LayerCache -----------------------------------------------------------


def test_layer_cache_reports_its_length():
    assert LayerCache().self_length == 0
    assert LayerCache(self_key=torch.zeros(2, 4, 5, 8)).self_length == 5


def test_layer_cache_reorder_selects_rows():
    """Beam search permutes beams every step; their caches must follow."""
    cache = LayerCache(
        self_key=torch.arange(24).float().view(3, 1, 2, 4),
        self_value=torch.arange(24).float().view(3, 1, 2, 4),
        cross_key=torch.arange(24).float().view(3, 1, 2, 4),
        cross_value=torch.arange(24).float().view(3, 1, 2, 4),
    )
    reordered = cache.reorder(torch.tensor([2, 0, 0]))

    assert torch.equal(reordered.self_key[0], cache.self_key[2])
    assert torch.equal(reordered.self_key[1], cache.self_key[0])
    assert torch.equal(reordered.cross_value[2], cache.cross_value[0])


def test_layer_cache_reorder_tolerates_empty_slots():
    reordered = LayerCache(self_key=torch.zeros(2, 1, 1, 4)).reorder(torch.tensor([1, 0]))
    assert reordered.cross_key is None
