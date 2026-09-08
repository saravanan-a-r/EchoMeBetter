"""
Document isolation, proven on the real model rather than on the mask alone.

`test_segment_mask.py` proves the mask has the right shape. That is not the
same as proving the *model* honours it: a mask that is built correctly and
then dropped on the floor between `Seq2Seq.forward` and the attention kernel
would pass every test in that file.

The property asserted here is the one packing actually depends on:

    changing document B must not change document A's encoder output.

If it does, the two documents are reading each other and every packed
pretraining window is teaching the model a relationship that does not exist.
This is a silent failure — the run trains, the loss falls — so it is worth an
end-to-end check on real weights.
"""

from __future__ import annotations

import pytest
import torch

from src.errors import ShapeError


def test_one_documents_encoding_is_unaffected_by_the_other(tiny_model):
    """
    Two windows sharing document A but differing in document B. With segment
    ids supplied, A's encoder states must be bit-identical between them.
    """
    model = tiny_model

    document_a = [5, 6, 7, 8, 9, 10]
    first_b = [20, 21, 22, 23]
    second_b = [40, 41, 42, 43]

    segment_ids = torch.tensor([[1] * len(document_a) + [2] * len(first_b)])
    window_one = torch.tensor([document_a + first_b])
    window_two = torch.tensor([document_a + second_b])

    with torch.no_grad():
        encoded_one = model.encode(window_one, None, segment_ids)
        encoded_two = model.encode(window_two, None, segment_ids)

    a_slice = slice(0, len(document_a))
    assert torch.equal(encoded_one[:, a_slice], encoded_two[:, a_slice]), (
        "document A's encoding changed when document B changed: the packed "
        "documents are attending to each other"
    )


def test_without_segment_ids_the_documents_do_influence_each_other(tiny_model):
    """
    The control. Same two windows, no segment ids: the encodings *must*
    differ, or the test above would pass for the trivial reason that the
    model ignores its second document entirely.
    """
    model = tiny_model

    document_a = [5, 6, 7, 8, 9, 10]
    window_one = torch.tensor([document_a + [20, 21, 22, 23]])
    window_two = torch.tensor([document_a + [40, 41, 42, 43]])

    with torch.no_grad():
        encoded_one = model.encode(window_one, None)
        encoded_two = model.encode(window_two, None)

    a_slice = slice(0, len(document_a))
    assert not torch.equal(encoded_one[:, a_slice], encoded_two[:, a_slice])


def test_a_packed_document_encodes_the_same_as_it_would_alone(tiny_model):
    """
    The strongest form of the property: a document packed beside others gets
    the same encoding it would have had on its own.

    This is what makes packing a pure efficiency change rather than a change
    to what the model is trained on — and it is why the packer gives every
    document its own `[MODE]` prefix, so a packed document's geometry is
    identical to a lone one's.
    """
    model = tiny_model

    document_a = [5, 6, 7, 8, 9, 10]
    packed = torch.tensor([document_a + [20, 21, 22, 23, 24]])
    segment_ids = torch.tensor([[1] * len(document_a) + [2] * 5])
    alone = torch.tensor([document_a])

    with torch.no_grad():
        packed_states = model.encode(packed, None, segment_ids)
        alone_states = model.encode(alone, None)

    torch.testing.assert_close(
        packed_states[:, : len(document_a)], alone_states, rtol=1e-4, atol=1e-5
    )


def test_forward_accepts_segment_ids_and_still_produces_a_loss(tiny_model):
    model = tiny_model

    encoder_input_ids = torch.tensor([[5, 6, 7, 8, 20, 21]])
    segment_ids = torch.tensor([[1, 1, 1, 1, 2, 2]])
    decoder_input_ids = torch.tensor([[0, 11, 12]])
    labels = torch.tensor([[11, 12, 1]])

    output = model(
        encoder_input_ids,
        decoder_input_ids,
        labels=labels,
        encoder_segment_ids=segment_ids,
    )
    assert output.loss is not None
    assert torch.isfinite(output.loss)

    output.loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_mismatched_segment_ids_are_refused(tiny_model):
    model = tiny_model
    with pytest.raises(ShapeError, match="encoder_segment_ids"):
        model.encode(torch.tensor([[5, 6, 7]]), None, torch.tensor([[1, 1]]))
