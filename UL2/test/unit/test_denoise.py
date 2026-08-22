"""
Unit tests for sequence construction.

The centrepiece is `test_reconstructs_the_architecture_md_worked_example`,
which encodes the exact [R] example printed in architecture.md 7.2. If a
future refactor changes the sequence format -- sentinel ordering, where EOS
goes, whether a trailing sentinel is emitted -- that test fails and names the
document it diverged from.

Everything else checks the lossless-reconstruction invariant, which is the
cheap oracle for a whole family of off-by-one and ordering bugs.
"""

from __future__ import annotations

import pytest

from conftest import make_tokens
from src.denoise import (
    Example,
    build_prefix_denoising,
    build_span_corruption,
    reconstruct_source,
)
from src.errors import UL2Error


# -- the documented format ------------------------------------------------


def test_reconstructs_the_architecture_md_worked_example(specials):
    """
    architecture.md 7.2, [R]:

        Original:  The cat sat on the mat and purred loudly in the sun
        Encoder:   [R] The <extra_id_0> on the <extra_id_1> and purred
                       <extra_id_2> in the sun
        Decoder:   <extra_id_0> cat sat <extra_id_1> mat <extra_id_2> loudly </s>
    """
    words = "The cat sat on the mat and purred loudly in the sun".split()
    tokens = list(range(100, 100 + len(words)))
    spans = [(1, 3), (5, 6), (8, 9)]  # "cat sat", "mat", "loudly"

    example = build_span_corruption(tokens, spans, "R", specials)

    assert example.encoder_input_ids == (
        specials.mode_token("R"),
        tokens[0],                       # The
        specials.sentinel(0),
        tokens[3], tokens[4],            # on the
        specials.sentinel(1),
        tokens[6], tokens[7],            # and purred
        specials.sentinel(2),
        tokens[9], tokens[10], tokens[11],  # in the sun
        specials.eos_id,
    )
    assert example.decoder_target_ids == (
        specials.sentinel(0), tokens[1], tokens[2],  # cat sat
        specials.sentinel(1), tokens[5],             # mat
        specials.sentinel(2), tokens[8],             # loudly
        specials.eos_id,
    )


def test_target_has_no_trailing_sentinel(specials):
    """
    architecture.md 7.2's worked target ends with content then EOS. Some T5
    implementations append a final unused sentinel; ours does not, and a
    change would shift every target by one token.
    """
    tokens = make_tokens(20)
    example = build_span_corruption(tokens, [(2, 4), (8, 10)], "R", specials)
    assert example.decoder_target_ids[-1] == specials.eos_id
    assert example.decoder_target_ids[-2] not in set(specials.sentinel_ids)


def test_prefix_denoising_format(specials):
    """architecture.md 7.2, [S]: encoder holds the prefix, no sentinels."""
    tokens = make_tokens(10)
    example = build_prefix_denoising(tokens, 4, specials)

    assert example.encoder_input_ids == (specials.mode_token("S"), *tokens[:4], specials.eos_id)
    assert example.decoder_target_ids == (*tokens[4:], specials.eos_id)
    assert example.num_spans == 0
    assert not set(example.encoder_input_ids) & set(specials.sentinel_ids)


def test_mode_token_is_always_first(specials):
    """The model reads the mode from position 0; it must never move."""
    tokens = make_tokens(30)
    for mode, example in (
        ("R", build_span_corruption(tokens, [(3, 6)], "R", specials)),
        ("X", build_span_corruption(tokens, [(3, 20)], "X", specials)),
        ("S", build_prefix_denoising(tokens, 10, specials)),
    ):
        assert example.encoder_input_ids[0] == specials.mode_token(mode)


# -- sentinel discipline --------------------------------------------------


def test_sentinels_are_numbered_in_span_order(specials):
    tokens = make_tokens(60)
    spans = [(5, 8), (12, 14), (20, 25), (40, 42)]
    example = build_span_corruption(tokens, spans, "R", specials)

    sentinel_set = set(specials.sentinel_ids)
    in_encoder = [t for t in example.encoder_input_ids if t in sentinel_set]
    in_target = [t for t in example.decoder_target_ids if t in sentinel_set]
    expected = [specials.sentinel(i) for i in range(len(spans))]

    assert in_encoder == expected
    assert in_target == expected


def test_each_sentinel_appears_once_per_side(specials):
    tokens = make_tokens(200)
    spans = [(i, i + 2) for i in range(5, 150, 10)]
    example = build_span_corruption(tokens, spans, "R", specials)

    sentinel_set = set(specials.sentinel_ids)
    used = [t for t in example.encoder_input_ids if t in sentinel_set]
    assert len(used) == len(set(used)) == len(spans)


def test_more_spans_than_sentinels_raises(small_specials):
    """Defence in depth: this constructor is public and must not wrap either."""
    tokens = make_tokens(60)
    spans = [(i, i + 2) for i in range(1, 40, 4)]
    assert len(spans) > small_specials.num_sentinels

    with pytest.raises(UL2Error, match="exceed"):
        build_span_corruption(tokens, spans, "R", small_specials)


# -- reconstruction invariant ---------------------------------------------

RECONSTRUCTION_CASES = [
    (20, [(2, 4)]),
    (20, [(1, 2), (5, 9), (15, 20)]),          # span touching the very end
    (50, [(1, 2), (3, 4), (5, 6), (7, 8)]),    # many tiny spans
    (50, [(1, 45)]),                            # one enormous span
    (2, [(1, 2)]),                              # minimum viable sequence
]


@pytest.mark.parametrize("length,spans", RECONSTRUCTION_CASES)
def test_span_corruption_reconstructs_exactly(specials, length, spans):
    tokens = make_tokens(length)
    example = build_span_corruption(tokens, spans, "R", specials)
    assert reconstruct_source(example, specials) == tuple(tokens)


@pytest.mark.parametrize("length,prefix", [(2, 1), (10, 5), (100, 1), (100, 99)])
def test_prefix_denoising_reconstructs_exactly(specials, length, prefix):
    tokens = make_tokens(length)
    example = build_prefix_denoising(tokens, prefix, specials)
    assert reconstruct_source(example, specials) == tuple(tokens)


def test_reconstruction_works_without_eos(specials):
    tokens = make_tokens(20)
    example = build_span_corruption(
        tokens, [(3, 6)], "R", specials,
        append_eos_to_source=False, append_eos_to_target=False,
    )
    assert specials.eos_id not in example.encoder_input_ids
    assert specials.eos_id not in example.decoder_target_ids
    assert reconstruct_source(example, specials) == tuple(tokens)


def test_reconstruction_survives_tokens_that_look_like_eos(specials):
    """
    A source token equal to EOS must not be mistaken for the terminator.
    Only the *final* token is stripped, so an interior one survives.
    """
    tokens = [500, specials.eos_id, 501, 502, 503, 504]
    example = build_span_corruption(tokens, [(3, 5)], "R", specials)
    assert reconstruct_source(example, specials) == tuple(tokens)


# -- decoder input (teacher forcing) --------------------------------------


def test_decoder_input_is_the_target_shifted_right(specials):
    tokens = make_tokens(20)
    example = build_span_corruption(tokens, [(3, 6), (10, 12)], "R", specials)

    assert example.decoder_input_ids[0] == specials.decoder_start_id
    assert example.decoder_input_ids[1:] == example.decoder_target_ids[:-1]
    assert len(example.decoder_input_ids) == len(example.decoder_target_ids)


def test_decoder_input_never_contains_the_final_eos(specials):
    """Feeding EOS back in would teach the model to generate past the end."""
    tokens = make_tokens(20)
    example = build_prefix_denoising(tokens, 8, specials)
    assert example.decoder_input_ids[-1] != specials.eos_id


# -- derived metrics ------------------------------------------------------


def test_realized_corruption_rate_is_measured_not_configured(specials):
    tokens = make_tokens(100)
    example = build_span_corruption(tokens, [(1, 4), (10, 20)], "R", specials)

    assert example.num_corrupted_tokens == 13
    assert example.realized_corruption_rate == pytest.approx(0.13)


def test_prefix_denoising_counts_the_continuation_as_corrupted(specials):
    """
    [S] withholds the continuation; reporting 0 corrupted tokens would make
    the architecture.md 7.3 telemetry meaningless for a quarter of examples.
    """
    tokens = make_tokens(100)
    example = build_prefix_denoising(tokens, 40, specials)
    assert example.num_corrupted_tokens == 60
    assert example.realized_corruption_rate == pytest.approx(0.6)


def test_lengths_are_reported_consistently(specials):
    tokens = make_tokens(40)
    example = build_span_corruption(tokens, [(5, 9)], "R", specials)

    assert example.encoder_length == len(example.encoder_input_ids)
    assert example.target_length == len(example.decoder_target_ids)
    assert example.source_length == 40


def test_zero_length_source_reports_zero_rate_rather_than_dividing_by_zero():
    example = Example(
        mode="R",
        encoder_input_ids=(),
        decoder_target_ids=(),
        decoder_input_ids=(),
        source_length=0,
        num_spans=0,
        num_corrupted_tokens=0,
    )
    assert example.realized_corruption_rate == 0.0


def test_example_is_immutable(specials):
    example = build_span_corruption(make_tokens(20), [(3, 6)], "R", specials)
    with pytest.raises(Exception):
        example.mode = "X"  # type: ignore[misc]


# -- malformed input ------------------------------------------------------


@pytest.mark.parametrize(
    "spans,match",
    [
        ([(5, 3)], "empty or inverted"),
        ([(5, 5)], "empty or inverted"),
        ([(2, 6), (4, 8)], "non-overlapping"),
        ([(10, 12), (2, 4)], "non-overlapping"),
        ([(15, 25)], "runs past"),
    ],
)
def test_malformed_spans_are_rejected(specials, spans, match):
    with pytest.raises(UL2Error, match=match):
        build_span_corruption(make_tokens(20), spans, "R", specials)


@pytest.mark.parametrize("prefix", [0, -1, 10, 11])
def test_invalid_prefix_length_rejected(specials, prefix):
    with pytest.raises(UL2Error, match="at least one token"):
        build_prefix_denoising(make_tokens(10), prefix, specials)


def test_reconstruction_rejects_a_duplicated_sentinel(specials):
    broken = Example(
        mode="R",
        encoder_input_ids=(specials.mode_token("R"), 500, specials.sentinel(0), 501),
        decoder_target_ids=(specials.sentinel(0), 700, specials.sentinel(0), 701),
        decoder_input_ids=(),
        source_length=4,
        num_spans=2,
        num_corrupted_tokens=2,
    )
    with pytest.raises(UL2Error, match="twice"):
        reconstruct_source(broken, specials)


def test_reconstruction_rejects_a_target_starting_with_content(specials):
    broken = Example(
        mode="R",
        encoder_input_ids=(specials.mode_token("R"), 500, specials.sentinel(0)),
        decoder_target_ids=(700, specials.sentinel(0), 701),
        decoder_input_ids=(),
        source_length=3,
        num_spans=1,
        num_corrupted_tokens=2,
    )
    with pytest.raises(UL2Error, match="before any sentinel"):
        reconstruct_source(broken, specials)


def test_reconstruction_rejects_a_sentinel_missing_from_the_target(specials):
    broken = Example(
        mode="R",
        encoder_input_ids=(specials.mode_token("R"), 500, specials.sentinel(3)),
        decoder_target_ids=(specials.sentinel(0), 700),
        decoder_input_ids=(),
        source_length=3,
        num_spans=1,
        num_corrupted_tokens=1,
    )
    with pytest.raises(UL2Error, match="no target content"):
        reconstruct_source(broken, specials)


def test_reconstruction_rejects_an_extra_sentinel_in_the_target(specials):
    broken = Example(
        mode="R",
        encoder_input_ids=(specials.mode_token("R"), 500, specials.sentinel(0)),
        decoder_target_ids=(specials.sentinel(0), 700, specials.sentinel(1), 701),
        decoder_input_ids=(),
        source_length=3,
        num_spans=1,
        num_corrupted_tokens=2,
    )
    with pytest.raises(UL2Error, match="absent from the encoder"):
        reconstruct_source(broken, specials)
