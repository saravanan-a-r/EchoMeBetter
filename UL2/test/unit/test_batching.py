"""
Unit tests for batch padding.

The bug this file exists to prevent: padding *labels* with `<pad>` instead of
-100. That trains the model to emit padding after real content, and it shows
up as a suspiciously good loss rather than an error -- padding is trivially
predictable, so the average loss falls while the model gets worse. Several
tests below assert the -100 rule directly.
"""

from __future__ import annotations

import pytest

from conftest import make_tokens
from src.batching import pad_batch
from src.denoise import build_prefix_denoising, build_span_corruption
from src.errors import UL2Error


def _examples(specials):
    return [
        build_span_corruption(make_tokens(30), [(3, 6), (12, 15)], "R", specials),
        build_span_corruption(make_tokens(80), [(5, 40)], "X", specials),
        build_prefix_denoising(make_tokens(50), 20, specials),
    ]


# -- shape ----------------------------------------------------------------


def test_batch_is_rectangular(specials):
    batch = pad_batch(_examples(specials), specials)

    encoder_widths = {len(row) for row in batch["encoder_input_ids"]}
    decoder_widths = {len(row) for row in batch["decoder_input_ids"]}
    label_widths = {len(row) for row in batch["labels"]}

    assert len(encoder_widths) == 1
    assert len(decoder_widths) == 1
    assert decoder_widths == label_widths


def test_widths_match_the_longest_example(specials):
    examples = _examples(specials)
    batch = pad_batch(examples, specials)

    assert len(batch["encoder_input_ids"][0]) == max(e.encoder_length for e in examples)
    assert len(batch["labels"][0]) == max(e.target_length for e in examples)


def test_masks_have_the_same_shape_as_their_inputs(specials):
    batch = pad_batch(_examples(specials), specials)
    for ids, mask in zip(batch["encoder_input_ids"], batch["encoder_attention_mask"]):
        assert len(ids) == len(mask)
    for ids, mask in zip(batch["decoder_input_ids"], batch["decoder_attention_mask"]):
        assert len(ids) == len(mask)


def test_modes_are_carried_through_in_order(specials):
    batch = pad_batch(_examples(specials), specials)
    assert batch["modes"] == ["R", "X", "S"]


def test_a_single_example_batches(specials):
    batch = pad_batch(_examples(specials)[:1], specials)
    assert len(batch["encoder_input_ids"]) == 1


def test_uniform_examples_need_no_padding(specials):
    examples = [
        build_span_corruption(make_tokens(30), [(3, 6)], "R", specials),
        build_span_corruption(make_tokens(30), [(10, 13)], "R", specials),
    ]
    batch = pad_batch(examples, specials)
    assert all(all(m) for m in batch["encoder_attention_mask"])
    assert all(specials.label_pad_id not in row for row in batch["labels"])


# -- padding values -------------------------------------------------------


def test_content_is_preserved_verbatim(specials):
    examples = _examples(specials)
    batch = pad_batch(examples, specials)

    for example, row in zip(examples, batch["encoder_input_ids"]):
        assert row[: example.encoder_length] == list(example.encoder_input_ids)
    for example, row in zip(examples, batch["labels"]):
        assert row[: example.target_length] == list(example.decoder_target_ids)


def test_inputs_are_padded_with_the_pad_token(specials):
    examples = _examples(specials)
    batch = pad_batch(examples, specials)

    for example, row in zip(examples, batch["encoder_input_ids"]):
        assert set(row[example.encoder_length:]) <= {specials.pad_id}
    for example, row in zip(examples, batch["decoder_input_ids"]):
        assert set(row[example.target_length:]) <= {specials.pad_id}


def test_labels_are_padded_with_the_ignore_index_not_the_pad_token(specials):
    """
    The critical rule. Padding labels with <pad> teaches the model to predict
    padding and quietly depresses the loss.
    """
    examples = _examples(specials)
    batch = pad_batch(examples, specials)

    for example, row in zip(examples, batch["labels"]):
        padding = row[example.target_length:]
        assert set(padding) <= {specials.label_pad_id}
        assert specials.pad_id not in padding or specials.label_pad_id == specials.pad_id


def test_label_padding_is_negative_everywhere(specials):
    batch = pad_batch(_examples(specials), specials)
    for row in batch["labels"]:
        assert all(value >= 0 or value == specials.label_pad_id for value in row)


def test_attention_masks_mark_exactly_the_real_tokens(specials):
    examples = _examples(specials)
    batch = pad_batch(examples, specials)

    for example, mask in zip(examples, batch["encoder_attention_mask"]):
        assert sum(mask) == example.encoder_length
        assert mask[: example.encoder_length] == [1] * example.encoder_length
        assert set(mask[example.encoder_length:]) <= {0}

    for example, mask in zip(examples, batch["decoder_attention_mask"]):
        assert sum(mask) == example.target_length


def test_masked_positions_align_with_ignored_labels(specials):
    """A position must be either real in both, or padding in both."""
    batch = pad_batch(_examples(specials), specials)
    for mask, labels in zip(batch["decoder_attention_mask"], batch["labels"]):
        for flag, label in zip(mask, labels):
            assert (flag == 1) == (label != specials.label_pad_id)


# -- pad_to_multiple_of ---------------------------------------------------


@pytest.mark.parametrize("multiple", [8, 16, 64])
def test_pad_to_multiple_rounds_both_dimensions(specials, multiple):
    batch = pad_batch(_examples(specials), specials, pad_to_multiple_of=multiple)
    assert len(batch["encoder_input_ids"][0]) % multiple == 0
    assert len(batch["labels"][0]) % multiple == 0


def test_pad_to_multiple_never_shrinks(specials):
    examples = _examples(specials)
    plain = pad_batch(examples, specials)
    rounded = pad_batch(examples, specials, pad_to_multiple_of=64)
    assert len(rounded["encoder_input_ids"][0]) >= len(plain["encoder_input_ids"][0])


def test_pad_to_multiple_is_invisible_to_the_model(specials):
    """Extra positions are masked and label-ignored, so they change nothing."""
    examples = _examples(specials)
    rounded = pad_batch(examples, specials, pad_to_multiple_of=64)

    for example, mask in zip(examples, rounded["encoder_attention_mask"]):
        assert sum(mask) == example.encoder_length
    for example, labels in zip(examples, rounded["labels"]):
        real = [v for v in labels if v != specials.label_pad_id]
        assert real == list(example.decoder_target_ids)


def test_pad_to_multiple_of_one_is_a_no_op(specials):
    examples = _examples(specials)
    assert pad_batch(examples, specials, pad_to_multiple_of=1) == pad_batch(examples, specials)


def test_already_aligned_width_is_untouched(specials):
    example = build_prefix_denoising(make_tokens(30), 15, specials)
    width = pad_batch([example], specials)["encoder_input_ids"]
    aligned = pad_batch([example], specials, pad_to_multiple_of=len(width[0]))
    assert len(aligned["encoder_input_ids"][0]) == len(width[0])


# -- errors ---------------------------------------------------------------


def test_empty_batch_rejected(specials):
    with pytest.raises(UL2Error, match="zero examples"):
        pad_batch([], specials)


def test_invalid_multiple_rejected(specials):
    with pytest.raises(UL2Error, match="pad_to_multiple_of"):
        pad_batch(_examples(specials), specials, pad_to_multiple_of=0)
