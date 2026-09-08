"""
Block-diagonal attention for packed pretraining windows
(`src/attention.py::build_segment_mask`).

Several unrelated documents share one encoder window during pretraining
(`UL2/src/packing.py`). This mask is the only thing stopping them reading each
other. If it were wrong — inverted, off by one, or silently a no-op — nothing
would crash and no loss curve would look unusual; the model would simply learn
relationships between texts that have nothing to do with each other. So the
allowed/forbidden pairs are asserted position by position.
"""

from __future__ import annotations

import pytest
import torch

from src.attention import build_padding_mask, build_segment_mask


def allowed(mask: torch.Tensor) -> torch.Tensor:
    """`True` where attention is permitted (an additive mask of 0.0)."""
    return mask[:, 0] == 0.0


def test_a_token_attends_inside_its_own_document_and_nowhere_else():
    #                     doc 1     doc 2   padding
    segment_ids = torch.tensor([[1, 1, 1, 2, 2, 0, 0]])
    permitted = allowed(build_segment_mask(segment_ids, torch.float32))[0]

    assert permitted[0, 1] and permitted[1, 0]  # within document 1
    assert permitted[3, 4] and permitted[4, 3]  # within document 2
    assert not permitted[0, 3]  # document 1 cannot read document 2
    assert not permitted[3, 0]  # ...nor the other way round


def test_the_block_structure_is_exactly_the_segment_layout():
    segment_ids = torch.tensor([[1, 1, 2, 2, 2, 3]])
    permitted = allowed(build_segment_mask(segment_ids, torch.float32))[0]
    expected = segment_ids[0][:, None] == segment_ids[0][None, :]
    assert torch.equal(permitted, expected)


def test_each_row_of_a_batch_gets_its_own_layout():
    """
    Packing is per example: two rows of one batch hold different documents at
    different offsets. A mask computed once and broadcast would apply the
    first row's boundaries to the second.
    """
    segment_ids = torch.tensor([[1, 1, 2, 2], [1, 2, 2, 2]])
    permitted = allowed(build_segment_mask(segment_ids, torch.float32))

    assert not permitted[0][0, 2]  # row 0: position 2 belongs to document 2
    assert not permitted[1][0, 1]  # row 1: the boundary sits one earlier
    assert permitted[1][1, 2]


def test_padding_is_left_to_the_padding_mask():
    """
    Two padded positions share segment 0 and this mask allows them; the
    padding mask forbids them. Each mask owns exactly one rule, and they are
    summed — so a position is blocked when either says so.
    """
    segment_ids = torch.tensor([[1, 1, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])

    segment = build_segment_mask(segment_ids, torch.float32)
    padding = build_padding_mask(attention_mask, torch.float32)
    combined = (segment + padding)[:, 0][0]

    assert combined[0, 1] == 0.0  # real to real, same document: allowed
    assert combined[0, 2] < 0  # real to padding: blocked
    assert combined[2, 0] < 0  # padding to real: blocked


def test_an_unpacked_window_is_left_completely_open():
    """One document filling the window must behave exactly as before packing."""
    segment_ids = torch.tensor([[1, 1, 1, 1]])
    assert allowed(build_segment_mask(segment_ids, torch.float32))[0].all()


def test_the_mask_uses_the_dtype_minimum_not_negative_infinity():
    """
    `-inf` makes a fully-masked row's softmax NaN, which poisons the whole
    batch's loss. `attention.py` uses `finfo.min` everywhere for that reason;
    this mask must not be the one exception.
    """
    mask = build_segment_mask(torch.tensor([[1, 2]]), torch.float32)
    assert torch.isfinite(mask).all()
    assert mask.min() == torch.finfo(torch.float32).min


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_mask_is_built_in_the_requested_dtype(dtype):
    mask = build_segment_mask(torch.tensor([[1, 1, 2]]), dtype)
    assert mask.dtype == dtype


def test_the_mask_shape_broadcasts_over_heads():
    mask = build_segment_mask(torch.tensor([[1, 1, 2, 2]]), torch.float32)
    assert mask.shape == (1, 1, 4, 4)


def test_segment_ids_must_be_two_dimensional():
    with pytest.raises(ValueError, match="batch, length"):
        build_segment_mask(torch.tensor([1, 1, 2]), torch.float32)
