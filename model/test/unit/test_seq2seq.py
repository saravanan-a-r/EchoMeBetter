"""
Unit tests for the assembled model.

Covers the wiring that only exists at this level: embedding tying, the tied-
output scaling, loss masking, input validation, and gradient checkpointing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import make_batch
from src.errors import ParameterCountError, ShapeError
from src.seq2seq import RephraseSeq2Seq, build_model


# -- embedding tying ------------------------------------------------------


def test_encoder_and_decoder_share_one_embedding(tiny_model):
    """
    architecture.md §4.1. Three users, one matrix — anything else silently
    adds 33.5M parameters at full scale.
    """
    assert tiny_model.encoder.embedding is tiny_model.shared
    assert tiny_model.decoder.embedding is tiny_model.shared


def test_the_output_projection_is_the_embedding(tiny_model):
    hidden = torch.randn(2, 3, tiny_model.config.d_model)
    expected = F.linear(
        hidden * (tiny_model.config.d_model**-0.5), tiny_model.shared.weight
    )
    assert torch.allclose(tiny_model.project(hidden), expected, atol=1e-6)


def test_tied_embedding_is_counted_once(tiny_config):
    """`parameters()` deduplicates by identity — the point of tying."""
    model = RephraseSeq2Seq(tiny_config)
    assert model.parameter_count() == tiny_config.parameter_count()


def test_untied_model_has_a_separate_head(tiny_config):
    untied = tiny_config.with_(tie_word_embeddings=False, expected_parameters=None)
    model = RephraseSeq2Seq(untied)

    assert model.lm_head is not None
    assert model.lm_head.weight is not model.shared.weight
    difference = model.parameter_count() - RephraseSeq2Seq(tiny_config).parameter_count()
    assert difference == untied.vocab_size * untied.d_model


def test_output_scaling_can_be_disabled(tiny_config):
    """
    Without it, logits from a matrix initialized as a lookup table are about
    sqrt(d_model) too large and training diverges early.
    """
    model = RephraseSeq2Seq(tiny_config.with_(scale_output_for_tied_embeddings=False))
    hidden = torch.randn(2, 3, tiny_config.d_model)
    assert torch.allclose(
        model.project(hidden), F.linear(hidden, model.shared.weight), atol=1e-6
    )


def test_output_scaling_changes_logit_magnitude(tiny_config):
    torch.manual_seed(0)
    scaled = RephraseSeq2Seq(tiny_config)
    torch.manual_seed(0)
    unscaled = RephraseSeq2Seq(tiny_config.with_(scale_output_for_tied_embeddings=False))

    hidden = torch.randn(2, 3, tiny_config.d_model)
    with torch.no_grad():
        ratio = float(
            unscaled.project(hidden).abs().mean() / scaled.project(hidden).abs().mean()
        )
    assert ratio == pytest.approx(tiny_config.d_model**0.5, rel=1e-3)


# -- parameter verification ----------------------------------------------


def test_verify_parameter_count_passes_for_a_consistent_config(tiny_config):
    model = RephraseSeq2Seq(tiny_config)
    assert model.verify_parameter_count() == tiny_config.parameter_count()


def test_verify_parameter_count_catches_a_stale_expectation(tiny_config):
    """
    The regression lock: change a dimension without recomputing
    `expected_parameters` and this fires, rather than the wrong-sized model
    training silently for a week.
    """
    model = RephraseSeq2Seq(tiny_config.with_(expected_parameters=999_999))
    with pytest.raises(ParameterCountError, match="expected_parameters"):
        model.verify_parameter_count()


def test_build_model_verifies_by_default(tiny_config):
    with pytest.raises(ParameterCountError):
        build_model(tiny_config.with_(expected_parameters=1))


def test_build_model_can_skip_verification(tiny_config):
    build_model(tiny_config.with_(expected_parameters=1), verify=False)


def test_absent_expectation_is_not_an_error(tiny_config):
    RephraseSeq2Seq(tiny_config.with_(expected_parameters=None)).verify_parameter_count()


# -- forward --------------------------------------------------------------


def test_forward_shapes(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    output = tiny_model(**batch)

    assert output.logits.shape == (
        batch["decoder_input_ids"].shape[0],
        batch["decoder_input_ids"].shape[1],
        tiny_config.vocab_size,
    )
    assert output.encoder_hidden_states.shape[-1] == tiny_config.d_model
    assert output.loss.ndim == 0


def test_loss_is_absent_without_labels(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    batch.pop("labels")
    assert tiny_model(**batch).loss is None


def test_loss_ignores_padded_label_positions(tiny_model, tiny_config):
    """
    Positions marked -100 must contribute nothing. If they did, the model
    would learn to predict padding — visible only as a suspiciously low loss.
    """
    batch = make_batch(tiny_config)
    labels = batch["labels"]
    assert (labels == tiny_config.label_pad_token_id).any(), "fixture must include padding"

    first = tiny_model(**batch).loss
    scrambled = labels.clone()
    scrambled[labels == tiny_config.label_pad_token_id] = tiny_config.label_pad_token_id
    changed = batch | {"labels": scrambled}
    assert torch.allclose(first, tiny_model(**changed).loss)


def test_masked_label_positions_do_not_affect_the_loss(tiny_model, tiny_config):
    """
    Stronger form: replace an ignored position with a *real* token and the
    loss must change — proving the ignore is doing work, not that the test
    changed nothing.
    """
    batch = make_batch(tiny_config)
    ignored = batch["labels"] == tiny_config.label_pad_token_id
    revealed = batch["labels"].clone()
    revealed[ignored] = 5

    assert not torch.allclose(
        tiny_model(**batch).loss, tiny_model(**(batch | {"labels": revealed})).loss
    )


def test_precomputed_encoder_states_are_reused(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    encoded = tiny_model.encode(batch["encoder_input_ids"], batch["encoder_attention_mask"])
    reused = tiny_model(**batch, encoder_hidden_states=encoded)
    assert torch.allclose(reused.logits, tiny_model(**batch).logits, atol=1e-6)


def test_label_smoothing_changes_the_loss(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    smoothed = RephraseSeq2Seq(tiny_config.with_(label_smoothing=0.1))
    smoothed.load_state_dict(tiny_model.state_dict())
    smoothed.eval()
    assert not torch.allclose(tiny_model(**batch).loss, smoothed(**batch).loss)


# -- input validation -----------------------------------------------------


def test_over_length_encoder_input_is_rejected(tiny_model, tiny_config):
    """architecture.md §4.5: never truncate silently."""
    too_long = torch.zeros(1, tiny_config.max_encoder_length + 1, dtype=torch.long)
    with pytest.raises(ShapeError, match="never be truncated silently"):
        tiny_model.encode(too_long)


def test_over_length_decoder_input_is_rejected(tiny_model, tiny_config):
    with pytest.raises(ShapeError, match="exceeding the configured limit"):
        tiny_model(
            torch.zeros(1, 8, dtype=torch.long),
            torch.zeros(1, tiny_config.max_decoder_length + 1, dtype=torch.long),
        )


def test_exactly_the_limit_is_accepted(tiny_model, tiny_config):
    at_limit = torch.zeros(1, tiny_config.max_encoder_length, dtype=torch.long)
    assert tiny_model.encode(at_limit).shape[1] == tiny_config.max_encoder_length


def test_wrong_rank_is_rejected(tiny_model):
    with pytest.raises(ShapeError, match=r"\(batch, length\)"):
        tiny_model.encode(torch.zeros(4, dtype=torch.long))


def test_empty_input_is_rejected(tiny_model):
    with pytest.raises(ShapeError, match="empty"):
        tiny_model.encode(torch.zeros(1, 0, dtype=torch.long))


def test_out_of_vocabulary_ids_are_rejected(tiny_model, tiny_config):
    with pytest.raises(ShapeError, match="outside"):
        tiny_model.encode(torch.tensor([[tiny_config.vocab_size]]))
    with pytest.raises(ShapeError, match="outside"):
        tiny_model.encode(torch.tensor([[-1]]))


def test_mismatched_labels_are_rejected(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    with pytest.raises(ShapeError, match="must match"):
        tiny_model(**(batch | {"labels": batch["labels"][:, :-1]}))


# -- gradient checkpointing ----------------------------------------------


def test_gradient_checkpointing_toggles_both_stacks(tiny_model):
    assert not tiny_model.gradient_checkpointing
    tiny_model.enable_gradient_checkpointing()
    assert tiny_model.encoder.gradient_checkpointing
    assert tiny_model.decoder.gradient_checkpointing
    assert tiny_model.gradient_checkpointing

    tiny_model.disable_gradient_checkpointing()
    assert not tiny_model.gradient_checkpointing


def test_gradient_checkpointing_does_not_change_the_loss(tiny_config):
    """
    It trades compute for memory by recomputing activations. The arithmetic
    must be identical — a difference here means recomputation diverged from
    the forward pass.
    """
    torch.manual_seed(0)
    model = build_model(tiny_config)
    model.train()
    batch = make_batch(tiny_config)

    plain = model(**batch).loss
    model.enable_gradient_checkpointing()
    checkpointed = model(**batch).loss

    assert torch.allclose(plain, checkpointed, atol=1e-6)


def test_gradient_checkpointing_produces_the_same_gradients(tiny_config):
    torch.manual_seed(0)
    model = build_model(tiny_config)
    model.train()
    batch = make_batch(tiny_config)

    model(**batch).loss.backward()
    plain = {name: p.grad.clone() for name, p in model.named_parameters()}

    model.zero_grad()
    model.enable_gradient_checkpointing()
    model(**batch).loss.backward()

    for name, parameter in model.named_parameters():
        assert torch.allclose(plain[name], parameter.grad, atol=1e-5), name
