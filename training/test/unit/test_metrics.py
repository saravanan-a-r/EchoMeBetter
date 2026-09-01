"""
Unit tests for loss accounting.

The test that carries the most weight is
`test_per_token_losses_aggregate_to_the_models_own_loss`: `metrics.py`
recomputes at token granularity what the model already computed as a scalar,
and if those two derivations can disagree then every per-mode number this
module reports is unanchored. It is written as an agreement between two
independent computations rather than as a check of one against a constant.

The second theme is that a mode with no tokens reports NaN, not zero. §7.6
exists because "an aggregate can look healthy while one objective silently
fails to learn" — reporting 0.0 for an absent objective would make that
failure look like a perfect score.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import make_batch
from src.metrics import (
    MODES,
    LossTracker,
    MeanLoss,
    format_summary,
    per_example_losses,
    per_token_losses,
)


@pytest.fixture
def scored(tiny_model):
    """Logits and labels from a real forward pass, plus the model's own loss."""
    batch = make_batch(tiny_model.config, batch_size=3, target_length=9)
    tiny_model.eval()
    with torch.no_grad():
        output = tiny_model(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
    return output, batch


# -- agreement with the model ----------------------------------------------


def test_per_token_losses_aggregate_to_the_models_own_loss(scored):
    """
    The anchor for everything else in this module. `RephraseSeq2Seq.forward`
    reduces with `mean` over non-ignored tokens; summing the per-token losses
    and dividing by their count must reproduce that exactly.
    """
    output, batch = scored
    losses = per_token_losses(output.logits, batch["labels"])
    counted = batch["labels"] != -100
    recomputed = losses[counted].sum() / counted.sum()
    assert recomputed.item() == pytest.approx(output.loss.item(), rel=1e-6)


def test_per_example_losses_aggregate_to_the_same_number(scored):
    output, batch = scored
    totals, counts = per_example_losses(output.logits, batch["labels"])
    assert (totals.sum() / counts.sum()).item() == pytest.approx(
        output.loss.item(), rel=1e-6
    )


def test_ignored_positions_contribute_nothing(scored):
    """
    -100 is `ignore_index`, and a padded position must add exactly zero — not
    a small number. Using `<pad>` instead would train the model to predict
    padding, which reads as a mysteriously low loss.
    """
    output, batch = scored
    losses = per_token_losses(output.logits, batch["labels"])
    assert torch.equal(
        losses[batch["labels"] == -100],
        torch.zeros_like(losses[batch["labels"] == -100]),
    )


def test_token_counts_exclude_the_ignored(scored):
    output, batch = scored
    _, counts = per_example_losses(output.logits, batch["labels"])
    expected = (batch["labels"] != -100).sum(dim=-1).float()
    assert torch.equal(counts, expected)


def test_label_smoothing_is_carried_through(scored):
    """
    Checked against the definition rather than against a direction. Smoothing
    is not reliably *higher* on an untrained model — with near-uniform logits
    the smoothed target is close to what the model already predicts — so
    "smoothed > plain" would be a test that passes or fails on the seed.

    What must hold is that the argument reaches the loss at all: an SFT run
    (§13 uses 0.1) whose logged numbers came from the unsmoothed objective
    would be reporting a different quantity from the one it optimizes.
    """
    import torch.nn.functional as F

    output, batch = scored
    smoothed = per_token_losses(output.logits, batch["labels"], label_smoothing=0.1)
    expected = F.cross_entropy(
        output.logits.reshape(-1, output.logits.size(-1)).float(),
        batch["labels"].reshape(-1),
        ignore_index=-100,
        label_smoothing=0.1,
        reduction="none",
    ).view(batch["labels"].shape)

    assert torch.allclose(smoothed, expected)
    assert not torch.allclose(smoothed, per_token_losses(output.logits, batch["labels"]))


def test_a_custom_ignore_index_is_honoured(scored):
    output, batch = scored
    labels = batch["labels"].masked_fill(batch["labels"] == -100, -1)
    losses = per_token_losses(output.logits, labels, ignore_index=-1)
    assert torch.isfinite(losses).all()
    assert float(losses[labels == -1].sum()) == 0.0


def test_the_shape_is_the_labels_shape(scored):
    output, batch = scored
    assert per_token_losses(output.logits, batch["labels"]).shape == batch["labels"].shape


# -- the running mean ------------------------------------------------------


def test_a_mean_loss_is_token_weighted():
    """
    Not the mean of the means. UL2's three modes produce targets of
    systematically different lengths, so averaging per-batch means would bias
    the result against the long-target modes consistently rather than
    randomly.
    """
    mean = MeanLoss()
    mean.add(total=10.0, tokens=2.0)   # a mean of 5 over 2 tokens
    mean.add(total=10.0, tokens=8.0)   # a mean of 1.25 over 8 tokens
    assert mean.value == 2.0           # not (5 + 1.25) / 2 == 3.125


def test_an_empty_mean_is_not_zero():
    """
    Zero would be a perfect loss. NaN says "unknown", which is the truth and
    is what stops an objective that produced no tokens from looking like the
    best-learned one in the run.
    """
    assert math.isnan(MeanLoss().value)
    assert math.isnan(MeanLoss().perplexity)
    assert not MeanLoss()


def test_perplexity_is_the_exponential_of_the_loss():
    mean = MeanLoss()
    mean.add(total=4.0, tokens=2.0)
    assert mean.perplexity == pytest.approx(math.exp(2.0))


# -- the tracker -----------------------------------------------------------


def test_losses_are_attributed_to_the_right_mode():
    tracker = LossTracker()
    tracker.update(
        torch.tensor([10.0, 4.0, 6.0]),
        torch.tensor([5.0, 2.0, 2.0]),
        modes=["R", "X", "S"],
    )
    assert tracker.by_mode["R"].value == 2.0
    assert tracker.by_mode["X"].value == 2.0
    assert tracker.by_mode["S"].value == 3.0
    assert tracker.overall.value == pytest.approx(20.0 / 9.0)


def test_modes_accumulate_across_batches():
    tracker = LossTracker()
    tracker.update(torch.tensor([10.0]), torch.tensor([5.0]), modes=["R"])
    tracker.update(torch.tensor([2.0]), torch.tensor([5.0]), modes=["R"])
    assert tracker.by_mode["R"].value == pytest.approx(1.2)


def test_a_batch_without_modes_still_counts_towards_the_overall_loss():
    tracker = LossTracker()
    tracker.update(torch.tensor([6.0]), torch.tensor([3.0]))
    assert tracker.overall.value == 2.0
    assert tracker.by_mode == {}


def test_mislabelled_batches_are_refused_not_misattributed():
    """
    A modes list of the wrong length would silently attribute rows to the
    wrong denoiser — and per-mode loss is the one diagnostic §7.6 says must be
    trustworthy.
    """
    tracker = LossTracker()
    with pytest.raises(ValueError, match="mode labels disagree"):
        tracker.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]), modes=["R"])


def test_mismatched_totals_and_counts_are_refused():
    tracker = LossTracker()
    with pytest.raises(ValueError, match="different numbers of rows"):
        tracker.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


def test_reset_clears_both_the_overall_and_the_per_mode_totals():
    tracker = LossTracker()
    tracker.update(torch.tensor([1.0]), torch.tensor([1.0]), modes=["R"])
    tracker.reset()
    assert tracker.by_mode == {}
    assert not tracker.overall


def test_the_summary_reports_every_mode_it_saw():
    tracker = LossTracker()
    tracker.update(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        modes=["S", "R", "X"],
    )
    summary = tracker.summary()
    assert set(summary) == {"loss", "perplexity", "loss_R", "loss_X", "loss_S"}


def test_the_summary_keeps_a_stable_column_order():
    """
    R, X, S in that order regardless of the order they arrived, so a log file
    can be read as columns rather than as key-value soup.
    """
    tracker = LossTracker()
    tracker.update(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        modes=["S", "X", "R"],
    )
    keys = [key for key in tracker.summary() if key.startswith("loss_")]
    assert keys == [f"loss_{mode}" for mode in MODES]


def test_an_unseen_mode_is_omitted_rather_than_reported_as_nan():
    """A window containing only [R] should log a readable line, not two NaNs."""
    tracker = LossTracker()
    tracker.update(torch.tensor([1.0]), torch.tensor([1.0]), modes=["R"])
    assert "loss_X" not in tracker.summary()


def test_an_unexpected_mode_is_reported_rather_than_dropped():
    """
    A fourth denoiser appearing means the objective changed. Silently
    discarding it would hide that; it is sorted after the known three so the
    known columns stay in place.
    """
    tracker = LossTracker()
    tracker.update(
        torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]), modes=["R", "Z"]
    )
    assert list(tracker.summary())[-1] == "loss_Z"


def test_the_tracker_survives_gradient_carrying_tensors():
    """`update` detaches; holding a graph alive across a window would leak."""
    tracker = LossTracker()
    totals = torch.tensor([2.0], requires_grad=True)
    tracker.update(totals, torch.tensor([1.0]), modes=["R"])
    assert tracker.overall.value == 2.0


def test_format_summary_is_one_readable_line():
    line = format_summary({"loss": 3.14159, "loss_R": 2.0})
    assert line == "loss=3.1416 loss_R=2.0000"
