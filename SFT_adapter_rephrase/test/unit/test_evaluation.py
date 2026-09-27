"""
Evaluation: the teacher-forced loss must be the model's own loss (padding
excluded), and the acceptance bar must reject what it claims to reject.
"""

from __future__ import annotations

import pytest
import torch

from conftest import build_tiny_model, make_config
from src.batching import collate
from src.data import SFTExample
from src.evaluation import GenerationRecord, generation_metrics, teacher_forced_metrics


def _examples():
    return [
        SFTExample("a", "x", "y", (3, 4, 20, 21, 5, 1), (30, 31, 1)),
        SFTExample("b", "x", "y", (3, 4, 22, 5, 1), (32, 1)),
        SFTExample("c", "x", "y", (3, 4, 23, 24, 25, 26, 5, 1), (33, 34, 35, 36, 1)),
    ]


def test_teacher_forced_loss_equals_the_models_own_token_mean_loss(codec):
    model = build_tiny_model()
    examples = _examples()
    batch = collate(examples, codec.frame)
    with torch.no_grad():
        expected = model(
            encoder_input_ids=batch.encoder_input_ids,
            decoder_input_ids=batch.decoder_input_ids,
            encoder_attention_mask=batch.encoder_attention_mask,
            decoder_attention_mask=batch.decoder_attention_mask,
            labels=batch.labels,
        ).loss
    # Batch size 2 forces two differently padded batches.
    metrics = teacher_forced_metrics(model, examples, codec, batch_size=2, device=torch.device("cpu"))
    assert metrics["eval/loss"] == pytest.approx(float(expected), rel=1e-5)
    assert 0.0 <= metrics["eval/ece"] <= 1.0


def test_acceptance_rejects_each_failure_it_names(tmp_path):
    acceptance = make_config(tmp_path).evaluation.acceptance
    source = "Please call @bob at 10:30 about invoice 4521 before the meeting today."
    good = "Call @bob at 10:30 about invoice 4521 before today's meeting."
    records = [
        GenerationRecord(source, good, good, True, 10),                                  # acceptable
        GenerationRecord(source, good, "Call @bob at 10:30 about the invoice.", True, 8),  # lost 4521
        GenerationRecord(source, good, good, False, 99),                                 # never stopped
        GenerationRecord(source, good, "", True, 0),                                     # empty
        GenerationRecord(source, good, source + " " + source, True, 30),                 # longer than input
    ]
    metrics = generation_metrics(records, acceptance)
    assert metrics["quality/acceptable_rate"] == pytest.approx(1 / 5)
    assert metrics["generation/terminated_rate"] == pytest.approx(4 / 5)
    assert metrics["generation/empty_rate"] == pytest.approx(1 / 5)

    reference = generation_metrics(records, acceptance, prefix="reference", use_reference_as_output=True)
    assert reference["reference/acceptable_rate"] == 1.0
    assert "quality/acceptable_rate" not in reference
