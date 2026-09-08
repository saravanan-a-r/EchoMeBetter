"""
Packed batches through the trainer.

`encoder_segment_ids` is an *optional* batch key: batches that carry it must
have it forwarded to the model, and batches that do not must behave exactly as
they did before packing existed. Both halves matter. Forwarding it is what
makes document isolation real during training; leaving unpacked batches
untouched is what keeps this an addition rather than a change to every
existing caller, including every fine-tuning run that never packs anything.

The failure mode being pinned is quiet in both directions: a batch whose
segment ids are silently dropped trains with documents reading each other, and
a required-key check that grew to include segment ids would reject every
legitimate unpacked batch.
"""

from __future__ import annotations

import pytest
import torch

from conftest import make_batch, make_batches
from src.errors import TrainingConfigError
from src.loop import OPTIONAL_KEYS, REQUIRED_KEYS, Trainer


def packed(batch: dict, boundary: int = 6) -> dict:
    """
    The same batch, relabelled as two packed documents split at `boundary`.

    Padding keeps segment 0, matching what `pad_batch` produces, so the row
    with padding exercises the case where the two masks must combine.
    """
    mask = batch["encoder_attention_mask"]
    positions = torch.arange(mask.shape[1])
    segments = torch.where(positions < boundary, 1, 2).expand_as(mask).clone()
    segments = segments.masked_fill(mask == 0, 0)
    return {**batch, "encoder_segment_ids": segments}


def test_segment_ids_are_not_required(training_config, tiny_model, model_config, cpu):
    """An ordinary unpacked batch must still be accepted."""
    assert "encoder_segment_ids" not in REQUIRED_KEYS
    assert "encoder_segment_ids" in OPTIONAL_KEYS

    trainer = Trainer(tiny_model, training_config, device=cpu)
    report = trainer.train_step([make_batch(model_config)])
    assert report.loss == pytest.approx(report.loss)


def test_a_packed_batch_trains(training_config, tiny_model, model_config, cpu):
    trainer = Trainer(tiny_model, training_config, device=cpu)
    report = trainer.train_step([packed(make_batch(model_config))])
    assert torch.isfinite(torch.tensor(report.loss))
    assert not report.skipped


def test_segment_ids_actually_reach_the_model(
    training_config, tiny_model, model_config, cpu, monkeypatch
):
    """
    The one thing that can go wrong without any visible symptom: the key is
    carried in the batch, moved to the device, and then never passed on. The
    run would train, the loss would fall, and every packed document would be
    reading its neighbours.
    """
    seen = {}
    original = tiny_model.forward

    def spy(*args, **kwargs):
        seen["segment_ids"] = kwargs.get("encoder_segment_ids")
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "forward", spy)

    trainer = Trainer(tiny_model, training_config, device=cpu)
    trainer.train_step([packed(make_batch(model_config))])

    assert seen["segment_ids"] is not None
    assert seen["segment_ids"].shape == make_batch(model_config)["encoder_input_ids"].shape


def test_an_unpacked_batch_passes_no_segment_ids(
    training_config, tiny_model, model_config, cpu, monkeypatch
):
    seen = {}
    original = tiny_model.forward

    def spy(*args, **kwargs):
        seen["segment_ids"] = kwargs.get("encoder_segment_ids")
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "forward", spy)

    trainer = Trainer(tiny_model, training_config, device=cpu)
    trainer.train_step([make_batch(model_config)])

    assert seen["segment_ids"] is None


def test_packing_changes_the_loss(training_config, tiny_model, model_config, cpu):
    """
    Proof the mask is doing something, not being accepted and ignored. The
    same tokens scored with and without document isolation must differ.
    """
    batch = make_batch(model_config)

    plain = Trainer(tiny_model, training_config, device=cpu)
    plain_loss = plain.evaluate([batch])["loss"]

    isolated = Trainer(tiny_model, training_config, device=cpu)
    packed_loss = isolated.evaluate([packed(batch)])["loss"]

    assert plain_loss != packed_loss


def test_evaluation_accepts_packed_batches(training_config, tiny_model, model_config, cpu):
    trainer = Trainer(tiny_model, training_config, device=cpu)
    metrics = trainer.evaluate([packed(b) for b in make_batches(model_config, 3)])
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_per_mode_logging_still_works_on_packed_batches(
    training_config, tiny_model, model_config, cpu
):
    """
    Per-mode loss runs a second forward pass of its own (`_track`). It has to
    forward the segment ids too, or the number it reports describes a
    different computation from the one that produced the gradient.
    """
    config = training_config.with_(log_per_mode_loss=True, logging_steps=1, max_steps=2)
    trainer = Trainer(tiny_model, config, device=cpu)
    trainer.train([packed(b) for b in make_batches(model_config, 8)])
    assert trainer.state.log_history


def test_a_batch_still_missing_a_required_key_is_refused(
    training_config, tiny_model, model_config, cpu
):
    trainer = Trainer(tiny_model, training_config, device=cpu)
    incomplete = make_batch(model_config)
    del incomplete["labels"]
    with pytest.raises(TrainingConfigError, match="missing"):
        trainer.train_step([incomplete])
