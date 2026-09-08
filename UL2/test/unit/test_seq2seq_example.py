"""
`build_seq2seq_example` and the telemetry's handling of a mode that is not
one of UL2's three.

Both exist for the synthetic rewrite task (`rewrite/`), which joins the batch
with both sides already complete. What must not regress: the teacher-forcing
shift is the same one every denoising example gets, the batch carries the
label so per-mode loss can see it, and the telemetry report does not silently
drop a mode it did not expect.
"""

from __future__ import annotations

import pytest

from conftest import make_tokens
from src.batching import pad_batch
from src.denoise import build_seq2seq_example, build_span_corruption, example_from_dict, example_to_dict
from src.errors import UL2Error
from src.telemetry import MixtureTelemetry


def test_the_pair_is_carried_through_with_the_standard_shift(specials):
    encoder = make_tokens(10, start=2000)
    target = make_tokens(6, start=3000)
    example = build_seq2seq_example(encoder, target, "rewrite", specials, source_length=6)

    assert example.encoder_input_ids == tuple(encoder)
    assert example.decoder_target_ids == tuple(target)
    assert example.decoder_input_ids == (specials.decoder_start_id, *target[:-1])
    assert example.mode == "rewrite"
    assert example.num_spans == 0
    assert example.num_corrupted_tokens == 0
    assert not example.is_packed


def test_the_batch_carries_the_label(specials):
    example = build_seq2seq_example([5, 6, 7], [8, 9], "rewrite", specials, source_length=2)
    batch = pad_batch([example, example], specials)
    assert batch["modes"] == ["rewrite", "rewrite"]
    assert "encoder_segment_ids" not in batch


def test_the_example_survives_the_checkpoint_encoding(specials):
    example = build_seq2seq_example([5, 6, 7], [8, 9], "rewrite", specials,
                                    source_length=2, num_corrupted_tokens=1)
    assert example_from_dict(example_to_dict(example)) == example


@pytest.mark.parametrize(
    "encoder, target, source_length",
    [([], [1], 1), ([1], [], 1), ([1], [1], -1)],
)
def test_an_incomplete_pair_is_refused(specials, encoder, target, source_length):
    with pytest.raises(UL2Error):
        build_seq2seq_example(encoder, target, "rewrite", specials, source_length=source_length)


def test_telemetry_reports_an_unexpected_mode_after_ul2s_own(specials):
    telemetry = MixtureTelemetry({"R": 1.0})
    telemetry.record(build_span_corruption(make_tokens(20), [(2, 5)], "R", specials))
    telemetry.record(build_seq2seq_example([5, 6, 7], [8, 9], "rewrite", specials, source_length=2))

    report = telemetry.report()
    assert list(report["modes"]) == ["R", "rewrite"]
    assert report["modes"]["rewrite"]["examples"] == 1
    assert report["modes"]["rewrite"]["configured_probability"] is None
    assert report["total_examples"] == 2
    assert "rewrite" in telemetry.format_table()
