"""
Padding a list of examples into rectangular batch arrays.

Kept in this module rather than the trainer because the label-masking rule is
easy to get wrong and expensive to get wrong quietly:

  - encoder/decoder *inputs* are padded with `<pad>`, and an attention mask
    marks the padding so it is never attended to;
  - *labels* are padded with -100, not `<pad>`.

Using `<pad>` for labels is the classic silent bug. It trains the model to
predict padding after the real content ends, which shows up as a
mysteriously low loss (padding is trivially predictable) and a model that
emits pad tokens at inference. -100 is PyTorch's `ignore_index`, so those
positions contribute nothing to the loss at all.

Output is plain Python lists -- no torch, no numpy -- so this module has zero
third-party dependencies and can be tested without a GPU stack. The trainer
wraps the result in tensors.
"""

from __future__ import annotations

from typing import Any, Sequence

from .denoise import Example
from .errors import UL2Error
from .packing import FIRST_DOCUMENT_SEGMENT_ID, PAD_SEGMENT_ID
from .special_tokens import SpecialTokens


def pad_batch(
    examples: Sequence[Example],
    specials: SpecialTokens,
    *,
    pad_to_multiple_of: int | None = None,
) -> dict[str, Any]:
    """
    Pad `examples` to rectangular arrays.

    `pad_to_multiple_of` rounds both sequence dimensions up to a multiple of
    the given value (8 or 16 keeps tensor-core kernels on their fast path).
    It changes speed only -- masked and -100 positions make it invisible to
    the model.

    Returns a dict with `encoder_input_ids`, `encoder_attention_mask`,
    `decoder_input_ids`, `decoder_attention_mask`, `labels`, and `modes`.
    """
    if not examples:
        raise UL2Error("cannot build a batch from zero examples")
    if pad_to_multiple_of is not None and pad_to_multiple_of < 1:
        raise UL2Error(f"pad_to_multiple_of must be >= 1, got {pad_to_multiple_of}")

    encoder_width = _round_up(max(e.encoder_length for e in examples), pad_to_multiple_of)
    decoder_width = _round_up(max(e.target_length for e in examples), pad_to_multiple_of)

    # Segment ids travel with the batch only when at least one example is
    # packed. A batch of unpacked examples carries no `encoder_segment_ids`
    # key at all, so the model takes exactly the path it took before packing
    # existed -- packing is additive, never a change to the unpacked case.
    any_packed = any(example.segment_ids for example in examples)

    batch: dict[str, Any] = {
        "encoder_input_ids": [],
        "encoder_attention_mask": [],
        "decoder_input_ids": [],
        "decoder_attention_mask": [],
        "labels": [],
        "modes": [e.mode for e in examples],
    }
    if any_packed:
        batch["encoder_segment_ids"] = []

    for example in examples:
        encoder_pad = encoder_width - example.encoder_length
        batch["encoder_input_ids"].append(
            list(example.encoder_input_ids) + [specials.pad_id] * encoder_pad
        )
        batch["encoder_attention_mask"].append([1] * example.encoder_length + [0] * encoder_pad)

        if any_packed:
            if example.segment_ids and len(example.segment_ids) != example.encoder_length:
                raise UL2Error(
                    f"example has {len(example.segment_ids)} segment ids for "
                    f"{example.encoder_length} encoder tokens; they must correspond "
                    f"one to one"
                )
            # An unpacked example sitting in a packed batch is one whole
            # document: every real position shares segment 1. Padding takes
            # segment 0, which no document ever uses, so a padded position can
            # never fall inside another example's document block.
            segments = (
                list(example.segment_ids)
                if example.segment_ids
                else [FIRST_DOCUMENT_SEGMENT_ID] * example.encoder_length
            )
            batch["encoder_segment_ids"].append(segments + [PAD_SEGMENT_ID] * encoder_pad)

        # decoder_input_ids and decoder_target_ids are the same length by
        # construction (shift-right preserves length), so one pad width and
        # one mask serve both.
        decoder_pad = decoder_width - example.target_length
        batch["decoder_input_ids"].append(
            list(example.decoder_input_ids) + [specials.pad_id] * decoder_pad
        )
        batch["decoder_attention_mask"].append([1] * example.target_length + [0] * decoder_pad)
        batch["labels"].append(
            list(example.decoder_target_ids) + [specials.label_pad_id] * decoder_pad
        )

    return batch


def _round_up(value: int, multiple: int | None) -> int:
    if multiple is None or multiple <= 1:
        return value
    remainder = value % multiple
    return value if remainder == 0 else value + (multiple - remainder)
