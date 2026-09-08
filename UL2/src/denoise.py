"""
Turning sampled spans into encoder/decoder sequences.

Sequence formats (architecture.md 7.2)
--------------------------------------
[R] and [X] -- sentinel-based span corruption:

    encoder: [MODE] kept... <extra_id_0> kept... <extra_id_1> kept... </s>
    decoder: <extra_id_0> span0... <extra_id_1> span1... </s>

Sentinels are numbered from 0 in left-to-right order of the spans they
replace. The target lists them in that same order, each followed by the
tokens it stands for. There is no trailing sentinel on the target -- it ends
with the last span's content, then EOS, exactly as the worked examples in
architecture.md 7.2 show.

[S] -- sequential denoising, no sentinels at all:

    encoder: [S] prefix... </s>
    decoder: continuation... </s>

The lossless-reconstruction invariant
-------------------------------------
For every mode, `reconstruct_source(example)` returns the original token
sequence exactly. This is not a convenience function: it is the property
that makes the objective *verifiable*. Almost every way of getting span
corruption wrong -- an off-by-one on a span boundary, spans emitted out of
order, a dropped final segment, a reused sentinel -- breaks reconstruction,
so a single round-trip check catches a whole family of silent corruptions
that a loss curve would never reveal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import UL2Error
from .special_tokens import SpecialTokens


@dataclass(frozen=True)
class Example:
    """
    One training example, as plain integer IDs.

    Deliberately framework-free: tuples of ints, no tensors. The trainer
    converts to whatever it needs. That keeps this module free of a torch
    dependency and makes every example trivially serializable for the frozen
    evaluation set.
    """

    mode: str
    encoder_input_ids: tuple[int, ...]
    decoder_target_ids: tuple[int, ...]
    decoder_input_ids: tuple[int, ...]
    source_length: int
    num_spans: int
    num_corrupted_tokens: int
    truncated: bool = False

    # Which document each encoder position belongs to, when several documents
    # were packed into one window (`packing.py`). Empty for an unpacked
    # example, which is what every existing caller produces -- so this stays a
    # pure addition: a batch built from unpacked examples carries no segment
    # ids and the model's attention behaves exactly as it did before.
    #
    # Segment 0 is reserved for padding; documents are numbered from 1.
    segment_ids: tuple[int, ...] = ()
    num_documents: int = 1

    @property
    def is_packed(self) -> bool:
        """Whether this example holds more than one document in one window."""
        return self.num_documents > 1

    @property
    def realized_corruption_rate(self) -> float:
        """
        Fraction of source tokens actually corrupted.

        architecture.md 7.3 requires reporting this rather than the configured
        rate: rounding to whole tokens and whole spans means the achieved
        value differs from the setting, especially on short sequences.
        """
        if self.source_length == 0:
            return 0.0
        return self.num_corrupted_tokens / self.source_length

    @property
    def encoder_length(self) -> int:
        return len(self.encoder_input_ids)

    @property
    def target_length(self) -> int:
        return len(self.decoder_target_ids)


def example_to_dict(example: Example) -> dict[str, object]:
    """
    An `Example` as JSON-safe plain data.

    Lives next to `Example` rather than in either of its two callers, because
    both the frozen eval set (`frozen_eval.py`, where the encoding feeds a
    digest) and the resumable length-bucketing pool (`length_batching.py`,
    where it feeds a checkpoint) need the same encoding. Duplicating it would
    let the two drift, and one of the two is a digest that must never move.
    """
    payload: dict[str, object] = {
        "mode": example.mode,
        "encoder_input_ids": list(example.encoder_input_ids),
        "decoder_target_ids": list(example.decoder_target_ids),
        "decoder_input_ids": list(example.decoder_input_ids),
        "source_length": example.source_length,
        "num_spans": example.num_spans,
        "num_corrupted_tokens": example.num_corrupted_tokens,
        "truncated": example.truncated,
    }
    # Emitted only when the example is packed, so an unpacked frozen set
    # serializes byte-identically to one written before packing existed and
    # its digest is unchanged.
    if example.segment_ids:
        payload["segment_ids"] = list(example.segment_ids)
        payload["num_documents"] = example.num_documents
    return payload


def example_from_dict(payload: Mapping[str, object]) -> Example:
    """Inverse of `example_to_dict`."""
    return Example(
        mode=payload["mode"],
        encoder_input_ids=tuple(payload["encoder_input_ids"]),
        decoder_target_ids=tuple(payload["decoder_target_ids"]),
        decoder_input_ids=tuple(payload["decoder_input_ids"]),
        source_length=payload["source_length"],
        num_spans=payload["num_spans"],
        num_corrupted_tokens=payload["num_corrupted_tokens"],
        truncated=payload["truncated"],
        segment_ids=tuple(payload.get("segment_ids", ())),
        num_documents=int(payload.get("num_documents", 1)),
    )


def build_span_corruption(
    tokens: Sequence[int],
    spans: Sequence[tuple[int, int]],
    mode: str,
    specials: SpecialTokens,
    *,
    append_eos_to_source: bool = True,
    append_eos_to_target: bool = True,
    truncated: bool = False,
) -> Example:
    """Build an [R]/[X] example from a token sequence and its masked spans."""
    _check_spans(spans, len(tokens))

    if len(spans) > specials.num_sentinels:
        # Defence in depth: `sample_spans` already enforces the budget, but
        # this constructor is public and must never be the path that lets a
        # sentinel get reused.
        raise UL2Error(
            f"{len(spans)} spans exceed the {specials.num_sentinels} available sentinels"
        )

    encoder: list[int] = [specials.mode_token(mode)]
    target: list[int] = []

    cursor = 0
    corrupted_count = 0
    for index, (start, end) in enumerate(spans):
        sentinel = specials.sentinel(index)
        encoder.extend(tokens[cursor:start])
        encoder.append(sentinel)
        target.append(sentinel)
        target.extend(tokens[start:end])
        corrupted_count += end - start
        cursor = end
    encoder.extend(tokens[cursor:])

    if append_eos_to_source:
        encoder.append(specials.eos_id)
    if append_eos_to_target:
        target.append(specials.eos_id)

    return Example(
        mode=mode,
        encoder_input_ids=tuple(encoder),
        decoder_target_ids=tuple(target),
        decoder_input_ids=_shift_right(target, specials),
        source_length=len(tokens),
        num_spans=len(spans),
        num_corrupted_tokens=corrupted_count,
        truncated=truncated,
    )


def build_prefix_denoising(
    tokens: Sequence[int],
    prefix_length: int,
    specials: SpecialTokens,
    *,
    mode: str = "S",
    append_eos_to_source: bool = True,
    append_eos_to_target: bool = True,
    truncated: bool = False,
) -> Example:
    """Build an [S] example: encoder holds the prefix, decoder continues it."""
    if not 1 <= prefix_length <= len(tokens) - 1:
        raise UL2Error(
            f"prefix_length {prefix_length} must leave at least one token on each "
            f"side of a {len(tokens)}-token sequence"
        )

    encoder: list[int] = [specials.mode_token(mode), *tokens[:prefix_length]]
    target: list[int] = list(tokens[prefix_length:])

    if append_eos_to_source:
        encoder.append(specials.eos_id)
    if append_eos_to_target:
        target.append(specials.eos_id)

    return Example(
        mode=mode,
        encoder_input_ids=tuple(encoder),
        decoder_target_ids=tuple(target),
        decoder_input_ids=_shift_right(target, specials),
        source_length=len(tokens),
        num_spans=0,
        # [S] withholds the continuation, so that is what "corrupted" means
        # here. Reporting 0 would make the realized corruption telemetry in
        # architecture.md 7.3 meaningless for a quarter of all examples.
        num_corrupted_tokens=len(tokens) - prefix_length,
        truncated=truncated,
    )


def build_seq2seq_example(
    encoder_input_ids: Sequence[int],
    decoder_target_ids: Sequence[int],
    mode: str,
    specials: SpecialTokens,
    *,
    source_length: int,
    num_corrupted_tokens: int = 0,
    truncated: bool = False,
) -> Example:
    """
    An `Example` from an encoder input and a decoder target that are already
    complete -- no sentinels, no prefix split, nothing masked here.

    This is how a task that is *not* a denoiser joins the batch: the
    synthetic rewrite task (`rewrite/`, architecture_improvements.md item 4)
    arrives with both sides fully formed and framed, and needs only the
    teacher-forcing shift and the bookkeeping every other example carries.
    Kept next to the two denoising builders so `Example` has exactly one
    module that knows how to construct it.

    `mode` is a label for telemetry and per-mode loss, not a UL2 mode: the
    caller chooses it, and `reconstruct_source` does not apply to it.
    `source_length` is the length of the *original* text the target
    reproduces; `num_corrupted_tokens` is whatever "corrupted" means for the
    caller's task -- for the rewrite task, the number of edits made.
    """
    if not encoder_input_ids:
        raise UL2Error("a seq2seq example needs a non-empty encoder input")
    if not decoder_target_ids:
        raise UL2Error("a seq2seq example needs a non-empty decoder target")
    if source_length < 0 or num_corrupted_tokens < 0:
        raise UL2Error("source_length and num_corrupted_tokens must be non-negative")

    target = tuple(decoder_target_ids)
    return Example(
        mode=mode,
        encoder_input_ids=tuple(encoder_input_ids),
        decoder_target_ids=target,
        decoder_input_ids=_shift_right(target, specials),
        source_length=source_length,
        num_spans=0,
        num_corrupted_tokens=num_corrupted_tokens,
        truncated=truncated,
    )


def reconstruct_source(example: Example, specials: SpecialTokens) -> tuple[int, ...]:
    """
    Rebuild the original token sequence from an example.

    Used by tests as the correctness oracle, and available for debugging a
    suspicious batch. Raises if the example is internally inconsistent.
    """
    encoder_body = list(example.encoder_input_ids[1:])  # drop the mode token
    if encoder_body and encoder_body[-1] == specials.eos_id:
        encoder_body.pop()

    target_body = list(example.decoder_target_ids)
    if target_body and target_body[-1] == specials.eos_id:
        target_body.pop()

    if example.mode == "S":
        return tuple(encoder_body + target_body)

    sentinel_set = set(specials.sentinel_ids)
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

    rebuilt: list[int] = []
    for token in encoder_body:
        if token in sentinel_set:
            if token not in contents:
                raise UL2Error(f"sentinel {token} in the encoder input has no target content")
            rebuilt.extend(contents.pop(token))
        else:
            rebuilt.append(token)

    if contents:
        raise UL2Error(
            f"decoder target contains sentinels absent from the encoder input: "
            f"{sorted(contents)}"
        )

    return tuple(rebuilt)


def _shift_right(target: Sequence[int], specials: SpecialTokens) -> tuple[int, ...]:
    """
    Teacher-forcing decoder input: start token, then the target minus its last.

    T5 has no BOS (`bos_id=-1`, architecture.md 6.1) and uses `<pad>` as the
    decoder start token, which is what `SpecialTokens.decoder_start_id`
    resolves to.
    """
    return (specials.decoder_start_id, *target[:-1])


def _check_spans(spans: Sequence[tuple[int, int]], length: int) -> None:
    """
    Reject malformed span lists.

    Overlapping or unsorted spans would produce an encoder input where the
    same source token appears both kept and masked, which reconstruction
    could not resolve.
    """
    previous_end = 0
    for start, end in spans:
        if start < previous_end:
            raise UL2Error(f"spans must be sorted and non-overlapping; got {(start, end)}")
        if start >= end:
            raise UL2Error(f"empty or inverted span {(start, end)}")
        if end > length:
            raise UL2Error(f"span {(start, end)} runs past the {length}-token sequence")
        previous_end = end
