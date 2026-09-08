"""
Packing several documents into one training window.

Why this module exists
----------------------
Left to itself, a corpus of ~400-token documents fed into a 2048-token model
wastes most of every batch on padding: `pad_batch` pads to the longest example
present, so a batch of short documents is mostly `<pad>`. Packing several
documents into one window removes that waste, and is what makes the configured
token-per-step budget (`training_config.yml`) achievable rather than aspirational.

The hazard packing introduces, and how it is removed
----------------------------------------------------
Naively concatenating documents and then corrupting the concatenation is
*wrong*, and wrong in a way no loss curve reveals:

  - a sampled span can straddle two documents, so the model is trained to
    reconstruct the head of document 2 from the tail of document 1 -- text
    that is genuinely unpredictable, which is anti-signal, not weak signal;
  - self-attention can read across documents, so document 3's representation
    is contaminated by document 1.

Both are removed here by construction rather than by hoping the model learns
to ignore the boundary:

  1. **Spans are sampled per document.** `sample_spans` is called once per
     document, on that document's own length, and the resulting offsets are
     shifted into window coordinates afterwards. A span therefore cannot
     cross a boundary -- there is no code path that could produce one.
  2. **Every document carries its own `[MODE] ... </s>` wrapper**, exactly the
     shape an unpacked example has. A packed window is a concatenation of
     complete, independently-valid examples, not a new sequence format.
  3. **`segment_ids` label every position with the document it belongs to.**
     The model turns that into a block-diagonal attention mask
     (`model/src/attention.py::build_segment_mask`), so a token can only
     attend inside its own document.

Why the relative position bias needs no change
----------------------------------------------
This surprises people, so it is written down. T5's position bias is a function
of `key_index - query_index` only. A difference is translation-invariant: two
tokens 7 apart are 7 apart whether their document starts at window position 0
or at window position 1200. Because attention is *also* blocked across
documents, no bias value between two different documents is ever consulted.

The one thing that would have broken this is the mode token: if a single
`[MODE]` sat at the front of the whole window, document 1 would see it 3
tokens away and document 5 would see it 1,600 tokens away, and those are
different buckets. Giving every document its own adjacent mode token (point 2
above) makes that distance identical for every document, which is exactly why
that shape was chosen over one shared prefix.

The cost of per-document wrappers is 2 tokens (`[MODE]` and `</s>`) per
document. At this corpus's ~400-token median document that is ~0.5% of the
window -- paid deliberately, to keep every document's geometry identical to an
unpacked example and to make the position-bias argument above hold.

[S] is deliberately not packed
------------------------------
`[S]` (prefix -> continuation) has no sentinels, so a packed `[S]` window's
decoder target would be several continuations concatenated with nothing but
their order to say which prefix each belongs to. That is a different, harder
task than the one the model is being built for, and the downstream rephrase
and CLI tasks are always single-document. `[S]` therefore uses exactly one
document per window -- see `UL2Objective.corrupt_packed`.
"""

from __future__ import annotations

import random
from typing import Sequence

from .denoise import Example, _shift_right
from .errors import UL2Error
from .spans import sample_spans
from .special_tokens import SpecialTokens

# Segment id 0 is reserved for padding, so a padded position can never share a
# segment with real content. Documents are numbered from 1.
PAD_SEGMENT_ID = 0
FIRST_DOCUMENT_SEGMENT_ID = 1


def pack_documents(
    documents: Sequence[Sequence[int]],
    *,
    window_budget: int,
    max_documents: int,
) -> int:
    """
    How many of `documents` fit in one window, in order.

    Returns the count to consume. Each document costs `len(doc) + 2` window
    tokens: its own `[MODE]` prefix and its own `</s>` terminator (see the
    module docstring for why the wrapper is per-document).

    Always returns at least 1 when `documents` is non-empty, even if that one
    document overflows the budget -- rejecting it is the *caller's* job
    (`UL2Objective._admit` owns the length policy and the no-silent-truncation
    rule of architecture.md 4.5). Silently dropping it here would be exactly
    the kind of invisible data loss this project forbids.
    """
    if window_budget < 1:
        raise UL2Error(f"window_budget must be >= 1, got {window_budget}")
    if max_documents < 1:
        raise UL2Error(f"max_documents must be >= 1, got {max_documents}")
    if not documents:
        return 0

    used = 0
    count = 0
    for document in documents:
        if count >= max_documents:
            break
        cost = len(document) + 2  # [MODE] + </s>
        if count > 0 and used + cost > window_budget:
            break
        used += cost
        count += 1
    return max(1, count)


def build_packed_span_corruption(
    documents: Sequence[Sequence[int]],
    mode: str,
    specials: SpecialTokens,
    rng: random.Random,
    *,
    corruption_rate: float,
    mean_span_length: float,
    append_eos_to_target: bool = True,
    truncated: bool = False,
) -> Example:
    """
    Build one `[R]`/`[X]` example from several documents packed into a window.

    Encoder, for documents A and B:

        [MODE] A_kept <extra_id_0> A_kept </s> [MODE] B_kept <extra_id_1> B_kept </s>
        |------------- segment 1 -------------| |------------- segment 2 ------------|

    Decoder target (sentinels numbered continuously across the whole window,
    so no sentinel is ever reused inside one example):

        <extra_id_0> A_span0 <extra_id_1> B_span0 </s>

    Spans are sampled once per document, on that document's own length, so a
    span can never straddle the boundary between two documents.
    """
    if not documents:
        raise UL2Error("cannot build a packed example from zero documents")

    encoder: list[int] = []
    segment_ids: list[int] = []
    target: list[int] = []

    mode_token = specials.mode_token(mode)
    sentinel_index = 0
    corrupted_count = 0
    source_length = 0

    for offset, document in enumerate(documents):
        segment_id = FIRST_DOCUMENT_SEGMENT_ID + offset
        length = len(document)
        source_length += length

        spans = sample_spans(
            length,
            corruption_rate,
            mean_span_length,
            rng,
            # The budget is what is left after the documents already packed,
            # not the full sentinel count: the whole window shares one
            # numbering, and a sentinel reused across two documents in the
            # same example would make the target unparseable.
            max_spans=specials.num_sentinels - sentinel_index,
        )

        document_start = len(encoder)
        encoder.append(mode_token)

        cursor = 0
        for start, end in spans:
            sentinel = specials.sentinel(sentinel_index)
            sentinel_index += 1
            encoder.extend(document[cursor:start])
            encoder.append(sentinel)
            target.append(sentinel)
            target.extend(document[start:end])
            corrupted_count += end - start
            cursor = end
        encoder.extend(document[cursor:])

        # Every document is terminated, exactly as an unpacked example is.
        encoder.append(specials.eos_id)
        segment_ids.extend([segment_id] * (len(encoder) - document_start))

    if append_eos_to_target:
        target.append(specials.eos_id)

    return Example(
        mode=mode,
        encoder_input_ids=tuple(encoder),
        decoder_target_ids=tuple(target),
        decoder_input_ids=_shift_right(target, specials),
        source_length=source_length,
        num_spans=sentinel_index,
        num_corrupted_tokens=corrupted_count,
        truncated=truncated,
        segment_ids=tuple(segment_ids),
        num_documents=len(documents),
    )


def reconstruct_documents(
    example: Example, specials: SpecialTokens
) -> tuple[tuple[int, ...], ...]:
    """
    Rebuild the original documents from a packed example.

    The packed counterpart of `denoise.reconstruct_source`, and used for the
    same reason: almost every way of getting the packing wrong -- a span
    shifted by one, sentinels numbered per document instead of per window,
    a document's tokens attributed to its neighbour's segment -- breaks this
    round-trip, so one check catches a whole family of silent corruptions.

    Raises `UL2Error` if the example is internally inconsistent.
    """
    if not example.segment_ids:
        raise UL2Error("example carries no segment_ids; it was not built by the packer")
    if len(example.segment_ids) != len(example.encoder_input_ids):
        raise UL2Error(
            f"segment_ids has {len(example.segment_ids)} entries but the encoder input "
            f"has {len(example.encoder_input_ids)}"
        )

    sentinel_set = set(specials.sentinel_ids)
    mode_token = specials.mode_token(example.mode)

    # Decoder target -> {sentinel: content}, in the same shape
    # `reconstruct_source` uses.
    target_body = list(example.decoder_target_ids)
    if target_body and target_body[-1] == specials.eos_id:
        target_body.pop()

    contents: dict[int, list[int]] = {}
    current: list[int] | None = None
    for token in target_body:
        if token in sentinel_set:
            if token in contents:
                raise UL2Error(f"sentinel {token} appears twice in the decoder target")
            current = []
            contents[token] = current
            continue
        if current is None:
            raise UL2Error("decoder target begins with content before any sentinel")
        current.append(token)

    # Walk the encoder, splitting on segment id, and re-inflate each document.
    documents: list[tuple[int, ...]] = []
    rebuilt: list[int] = []
    previous_segment: int | None = None

    for token, segment in zip(example.encoder_input_ids, example.segment_ids):
        if previous_segment is not None and segment != previous_segment:
            documents.append(tuple(rebuilt))
            rebuilt = []
        previous_segment = segment

        if token == mode_token and not rebuilt:
            continue  # the document's own [MODE] prefix
        if token in sentinel_set:
            if token not in contents:
                raise UL2Error(f"sentinel {token} in the encoder input has no target content")
            rebuilt.extend(contents.pop(token))
            continue
        rebuilt.append(token)

    documents.append(tuple(rebuilt))

    # Each document ends with its own EOS terminator; strip it.
    stripped: list[tuple[int, ...]] = []
    for document in documents:
        if document and document[-1] == specials.eos_id:
            document = document[:-1]
        stripped.append(document)

    if contents:
        raise UL2Error(
            f"decoder target contains sentinels absent from the encoder input: "
            f"{sorted(contents)}"
        )

    return tuple(stripped)
