"""
Batch planning and padding: coverage, determinism, resumability, grouping,
and the teacher-forcing shift.
"""

from __future__ import annotations

import random

import torch

from src.batching import StepBatchPlan, StepBatchStream, collate, split_micro_batches
from src.codec import Frame
from src.data import SFTExample

FRAME = Frame(style_id=3, open_id=4, close_id=5, eos_id=1, pad_id=0, decoder_start_id=0)


def _lengths(count: int = 101, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(5, 300) for _ in range(count)]


def test_each_epoch_covers_every_example_exactly_once():
    plan = StepBatchPlan(_lengths(), step_batch_size=8, window=4, seed=1)
    for epoch in range(3):
        batches = plan.epoch(epoch)
        flat = [i for batch in batches for i in batch]
        assert sorted(flat) == list(range(101))
        assert len(batches) == plan.steps_per_epoch == 13
        assert sum(len(b) < 8 for b in batches) == 1


def test_plan_is_deterministic_and_reshuffled_every_epoch():
    first = StepBatchPlan(_lengths(), 8, 4, seed=1)
    again = StepBatchPlan(_lengths(), 8, 4, seed=1)
    assert first.epoch(0) == again.epoch(0)
    assert first.epoch(0) != first.epoch(1)
    assert first.epoch(0) != StepBatchPlan(_lengths(), 8, 4, seed=2).epoch(0)


def test_length_grouping_cuts_padding():
    """The reason grouping exists: padded tokens dominate CPU cost."""
    lengths = _lengths(2000)

    def padding_waste(batches):
        return sum(max(lengths[i] for i in b) * len(b) - sum(lengths[i] for i in b) for b in batches)

    grouped = StepBatchPlan(lengths, 16, 32, seed=0).epoch(0)
    ungrouped = StepBatchPlan(lengths, 16, 1, seed=0).epoch(0)
    assert padding_waste(grouped) < 0.25 * padding_waste(ungrouped)


def test_stream_resumes_at_the_same_batch_across_an_epoch_boundary():
    plan = StepBatchPlan(_lengths(40), 8, 2, seed=3)
    straight = StepBatchStream(plan)
    expected = [straight.next() for _ in range(12)]

    interrupted = StepBatchStream(plan)
    for _ in range(4):
        interrupted.next()
    resumed = StepBatchStream(plan)
    resumed.load_state_dict(interrupted.state_dict())
    assert [resumed.next() for _ in range(8)] == expected[4:]
    assert resumed.epoch == 2 and resumed.index == 2


def test_micro_batches_partition_the_step_batch():
    assert split_micro_batches([5, 6, 7, 8, 9], 2) == [[5, 6], [7, 8], [9]]


def test_collate_shifts_the_target_and_masks_padding_out_of_the_loss():
    short = SFTExample("a", "x", "y", (3, 4, 20, 5, 1), (30, 1))
    long = SFTExample("b", "x", "y", (3, 4, 20, 21, 22, 5, 1), (31, 32, 33, 1))
    batch = collate([short, long], FRAME)
    assert batch.encoder_input_ids.tolist() == [[3, 4, 20, 5, 1, 0, 0], [3, 4, 20, 21, 22, 5, 1]]
    assert batch.encoder_attention_mask.tolist() == [[1, 1, 1, 1, 1, 0, 0], [1] * 7]
    assert batch.decoder_input_ids.tolist() == [[0, 30, 0, 0], [0, 31, 32, 33]]
    assert batch.labels.tolist() == [[30, 1, -100, -100], [31, 32, 33, 1]]
    assert batch.target_tokens == 6
    assert torch.equal(batch.decoder_attention_mask, (batch.labels != -100).long())
