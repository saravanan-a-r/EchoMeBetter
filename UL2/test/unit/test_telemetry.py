"""
Unit tests for the measurement requirement in architecture.md 7.3.

The single most important assertion in this file is
`test_decoder_token_share_diverges_from_mode_probability`: it demonstrates,
with real examples, that a 40/35/25 split of *examples* is not a 40/35/25
split of *decoder training tokens*. That gap is the reason the document
insists both numbers be recorded rather than one inferred from the other,
and this test fails if the telemetry ever stops being able to show it.
"""

from __future__ import annotations

import pytest

from conftest import make_tokens
from src.config import DEFAULT_MODE_WEIGHTS
from src.denoise import build_prefix_denoising, build_span_corruption
from src.telemetry import TARGET_LENGTH_BUCKETS, MixtureTelemetry, ModeCounters


def _r_example(specials, length=100):
    spans = [(i, i + 3) for i in range(1, min(length - 4, 40), 8)]
    return build_span_corruption(make_tokens(length), spans, "R", specials)


def _x_example(specials, length=100):
    return build_span_corruption(make_tokens(length), [(1, length // 2)], "X", specials)


def _s_example(specials, length=100):
    return build_prefix_denoising(make_tokens(length), length // 4, specials)


# -- ModeCounters ---------------------------------------------------------


def test_counters_start_empty():
    counters = ModeCounters()
    assert counters.examples == 0
    assert counters.as_dict()["realized_corruption_rate"] == 0.0
    assert counters.as_dict()["mean_span_length"] == 0.0


def test_counters_accumulate(specials):
    counters = ModeCounters()
    example = _r_example(specials)
    counters.record(example)
    counters.record(example)

    assert counters.examples == 2
    assert counters.source_tokens == 2 * example.source_length
    assert counters.decoder_tokens == 2 * example.target_length
    assert counters.spans == 2 * example.num_spans


def test_counters_derive_realized_rates(specials):
    counters = ModeCounters()
    counters.record(build_span_corruption(make_tokens(100), [(1, 4), (10, 20)], "R", specials))

    summary = counters.as_dict()
    assert summary["realized_corruption_rate"] == pytest.approx(0.13)
    assert summary["mean_span_length"] == pytest.approx(6.5)
    assert summary["mean_spans_per_example"] == pytest.approx(2.0)


def test_truncated_examples_are_counted(specials):
    counters = ModeCounters()
    counters.record(build_span_corruption(make_tokens(50), [(1, 5)], "R", specials, truncated=True))
    counters.record(build_span_corruption(make_tokens(50), [(1, 5)], "R", specials))
    assert counters.truncated == 1


def test_target_length_histogram_buckets(specials):
    counters = ModeCounters()
    counters.record(_s_example(specials, length=40))    # short target
    counters.record(_s_example(specials, length=1000))  # long target

    histogram = counters.target_length_histogram
    assert sum(histogram.values()) == 2
    assert len(histogram) == 2, "the two examples must land in different buckets"


def test_histogram_bucket_labels_sort_in_numeric_order():
    """Zero-padded labels, so a sorted report reads low-to-high."""
    counters = ModeCounters()
    for length in (10, 100, 1000):
        counters.record(
            type("E", (), {
                "source_length": length, "encoder_length": length,
                "target_length": length, "num_corrupted_tokens": 1,
                "num_spans": 1, "truncated": False,
            })()
        )
    labels = list(counters.as_dict()["target_length_histogram"])
    assert labels == sorted(labels)


def test_lengths_above_every_bucket_land_in_the_overflow_bucket():
    counters = ModeCounters()
    counters.record(
        type("E", (), {
            "source_length": 9999, "encoder_length": 9999,
            "target_length": TARGET_LENGTH_BUCKETS[-1] + 1,
            "num_corrupted_tokens": 1, "num_spans": 1, "truncated": False,
        })()
    )
    assert list(counters.target_length_histogram)[0].endswith("+")


# -- MixtureTelemetry -----------------------------------------------------


def test_report_on_an_empty_run_does_not_divide_by_zero():
    report = MixtureTelemetry(DEFAULT_MODE_WEIGHTS).report()
    assert report["total_examples"] == 0
    assert report["modes"] == {}


def test_report_totals(specials):
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    telemetry.record(_r_example(specials))
    telemetry.record(_x_example(specials))
    telemetry.record(_s_example(specials))

    report = telemetry.report()
    assert report["total_examples"] == 3
    assert set(report["modes"]) == {"R", "X", "S"}
    assert report["total_decoder_tokens"] == sum(
        entry["decoder_tokens"] for entry in report["modes"].values()
    )


def test_shares_sum_to_one(specials):
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    for _ in range(4):
        telemetry.record(_r_example(specials))
    for _ in range(3):
        telemetry.record(_x_example(specials))
    for _ in range(3):
        telemetry.record(_s_example(specials))

    modes = telemetry.report()["modes"]
    for key in ("realized_example_share", "encoder_token_share", "decoder_token_share"):
        assert sum(entry[key] for entry in modes.values()) == pytest.approx(1.0)


def test_configured_probability_is_reported_alongside_the_realized_share(specials):
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    for _ in range(4):
        telemetry.record(_r_example(specials))
    for _ in range(6):
        telemetry.record(_x_example(specials))

    modes = telemetry.report()["modes"]
    assert modes["R"]["configured_probability"] == 0.40
    assert modes["R"]["realized_example_share"] == pytest.approx(0.40)
    assert modes["X"]["configured_probability"] == 0.35
    assert modes["X"]["realized_example_share"] == pytest.approx(0.60)


def test_decoder_token_share_diverges_from_mode_probability(specials):
    """
    architecture.md 7.3, the measurement that must never be assumed away.

    Modes are recorded at exactly the configured 40/35/25 example split, yet
    [R] -- whose targets are short -- delivers far less than 40% of the
    decoder training tokens.
    """
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    for _ in range(40):
        telemetry.record(_r_example(specials, length=500))
    for _ in range(35):
        telemetry.record(_x_example(specials, length=500))
    for _ in range(25):
        telemetry.record(_s_example(specials, length=500))

    modes = telemetry.report()["modes"]
    assert modes["R"]["realized_example_share"] == pytest.approx(0.40)
    assert modes["R"]["decoder_token_share"] < 0.25, (
        "[R] should deliver far fewer decoder tokens than its example share"
    )
    assert modes["S"]["decoder_token_share"] > modes["S"]["realized_example_share"]


def test_unconfigured_mode_reports_none_for_its_probability(specials):
    telemetry = MixtureTelemetry()
    telemetry.record(_r_example(specials))
    assert telemetry.report()["modes"]["R"]["configured_probability"] is None


def test_modes_are_reported_in_canonical_order(specials):
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    telemetry.record(_s_example(specials))
    telemetry.record(_x_example(specials))
    telemetry.record(_r_example(specials))
    assert list(telemetry.report()["modes"]) == ["R", "X", "S"]


def test_skipped_records_are_counted():
    """Data loss must never be invisible."""
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    telemetry.record_skipped_too_short()
    telemetry.record_skipped_too_short()
    telemetry.record_skipped_too_long()

    report = telemetry.report()
    assert report["skipped_too_short"] == 2
    assert report["skipped_too_long"] == 1
    assert report["total_examples"] == 0


def test_counters_accessor_creates_a_mode_on_demand():
    telemetry = MixtureTelemetry()
    assert telemetry.counters("R").examples == 0


def test_format_table_renders(specials):
    telemetry = MixtureTelemetry(DEFAULT_MODE_WEIGHTS)
    telemetry.record(_r_example(specials))
    telemetry.record(_s_example(specials))
    telemetry.record_skipped_too_short()

    table = telemetry.format_table()
    assert "mode" in table
    assert "R" in table and "S" in table
    assert "skipped" in table
    assert len(table.splitlines()) == 5  # header, rule, two modes, skip line


def test_format_table_handles_an_empty_run():
    assert "mode" in MixtureTelemetry().format_table()


def test_format_table_handles_unconfigured_weights(specials):
    """A missing configured probability renders as '-' rather than crashing."""
    telemetry = MixtureTelemetry()
    telemetry.record(_r_example(specials))
    assert "-" in telemetry.format_table()
