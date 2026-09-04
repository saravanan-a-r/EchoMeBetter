"""
The parameter budget, checked three independent ways.

architecture.md §4.3 works out 750,971,904 parameters by hand. This file
verifies that number against:

  1. `ModelConfig.parameter_count()` — a closed-form formula
  2. `model_config.yml`'s declared `expected_parameters`
  3. the *actually built* PyTorch module tree

Three derivations agreeing is evidence; one asserting itself is not. Building
the full Large model costs ~3 GB of RAM, so that check is marked `slow`;
the formula checks run always.
"""

from __future__ import annotations

import pytest
import torch

from src.seq2seq import RephraseSeq2Seq

# architecture.md §4.3, computed by hand in the document.
LARGE_TOTAL = 750_971_904
LARGE_COMPONENTS = {
    "token_embeddings": 33_619_968,
    "encoder_per_layer": 12_847_104,
    "encoder_total": 308_330_496,
    "decoder_per_layer": 17_042_432,
    "decoder_total": 409_018_368,
    "relative_position_bias": 1_024,
    "final_norms": 2_048,
}

# architecture.md §4.2's table.
BASE_TOTAL = 223_444_224


# -- the documented totals ------------------------------------------------


def test_large_matches_the_documented_total(large_config):
    assert large_config.parameter_count() == LARGE_TOTAL


def test_large_matches_the_declared_expectation(large_config):
    assert large_config.expected_parameters == LARGE_TOTAL


def test_base_matches_the_documented_total(base_config):
    assert base_config.parameter_count() == BASE_TOTAL


def test_base_matches_the_declared_expectation(base_config):
    assert base_config.expected_parameters == BASE_TOTAL


@pytest.mark.parametrize("component,expected", sorted(LARGE_COMPONENTS.items()))
def test_each_component_matches_the_documented_table(large_config, component, expected):
    """Every row of architecture.md §4.3, individually."""
    assert large_config.parameter_breakdown()[component] == expected


def test_the_components_sum_to_the_total(large_config):
    breakdown = large_config.parameter_breakdown()
    assert sum(LARGE_COMPONENTS[key] for key in
               ("token_embeddings", "encoder_total", "decoder_total",
                "relative_position_bias", "final_norms")) == LARGE_TOTAL
    assert breakdown["total"] == LARGE_TOTAL


# -- serving footprint (architecture.md §4.3) ----------------------------


@pytest.mark.parametrize(
    "bytes_per_param,expected_mb",
    [(4, 3004), (2, 1502), (1, 751)],
)
def test_serving_footprint_matches_the_documented_table(
    large_config, bytes_per_param, expected_mb
):
    """fp32 ~3,004 MB / bf16 ~1,502 MB / int8 ~751 MB."""
    megabytes = large_config.parameter_count() * bytes_per_param / 1_000_000
    assert round(megabytes) == expected_mb


# -- the built model ------------------------------------------------------


def test_a_scaled_down_model_agrees_with_its_formula(large_config):
    """
    Same architecture, fewer layers — cheap enough to actually build, and it
    exercises the identical module tree the full model uses.
    """
    small = large_config.with_(
        num_layers=2, num_decoder_layers=2, expected_parameters=None
    )
    model = RephraseSeq2Seq(small)
    assert model.parameter_count() == small.parameter_count()


def test_layer_counts_scale_the_total_linearly(large_config):
    """
    Confirms the per-layer figures are what they claim: adding a layer must
    add exactly one layer's worth of parameters.
    """
    breakdown = large_config.parameter_breakdown()

    one_more_encoder = large_config.with_(num_layers=25, expected_parameters=None)
    assert (
        one_more_encoder.parameter_count() - large_config.parameter_count()
        == breakdown["encoder_per_layer"]
    )

    one_more_decoder = large_config.with_(num_decoder_layers=25, expected_parameters=None)
    assert (
        one_more_decoder.parameter_count() - large_config.parameter_count()
        == breakdown["decoder_per_layer"]
    )


@pytest.mark.slow
def test_the_full_base_model_builds_at_the_documented_size(base_config):
    """~223M parameters actually allocated, then released."""
    model = RephraseSeq2Seq(base_config)
    try:
        assert model.verify_parameter_count() == BASE_TOTAL
    finally:
        del model


@pytest.mark.slow
def test_the_full_large_model_builds_at_the_documented_size(large_config):
    """
    The real thing: ~751M parameters, ~3 GB in fp32. This is the test that
    proves architecture.md §4.3 describes the model that actually gets built.
    """
    model = RephraseSeq2Seq(large_config)
    try:
        assert model.verify_parameter_count() == LARGE_TOTAL
    finally:
        del model


@pytest.mark.slow
def test_the_large_model_runs_a_forward_pass(large_config):
    """A short sequence, to prove the full-size wiring is sound."""
    model = RephraseSeq2Seq(large_config).eval()
    try:
        with torch.no_grad():
            output = model(
                torch.randint(600, 2000, (1, 8)),
                torch.randint(600, 2000, (1, 4)),
            )
        assert output.logits.shape == (1, 4, large_config.vocab_size)
        assert torch.isfinite(output.logits).all()
    finally:
        del model
