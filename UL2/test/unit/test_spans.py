"""
Unit tests for span sampling.

The properties asserted here are the ones the whole objective rests on. In
particular **non-adjacency**: two touching spans collapse into a single
sentinel once the encoder input is built, so the realized span-length
distribution would drift away from the configured mean with no error anywhere
-- a silent objective change that no loss curve would reveal. The
kept-first interleave is what prevents it, and
`test_spans_are_never_adjacent` is the regression lock on that.
"""

from __future__ import annotations

import random

import pytest

from src.errors import ConfigError, SentinelBudgetExceededError
from src.spans import (
    MIN_CORRUPTIBLE_LENGTH,
    plan_span_counts,
    random_segmentation,
    sample_prefix_split,
    sample_spans,
)

# Lengths spanning degenerate, typical and full-context cases. The real
# corpus measures P50=18 / P90=88 / P99=597 (architecture.md 4.4), so short
# sequences are the common case, not the exotic one.
LENGTHS = [2, 3, 5, 8, 13, 18, 50, 88, 200, 597, 1024, 2000, 2047, 2048]


# -- plan_span_counts -----------------------------------------------------


def test_plan_span_counts_typical_case():
    """architecture.md 6.3's worked figure: 2048 tokens at 15% / span 3."""
    corrupted, spans = plan_span_counts(2048, 0.15, 3.0)
    assert corrupted == 307
    assert spans == 102


def test_plan_span_counts_x_denoising_at_full_context():
    """architecture.md 4.4's arithmetic for why the decoder must be 2048."""
    corrupted, spans = plan_span_counts(2048, 0.50, 32.0)
    assert corrupted == 1024
    assert spans == 32
    # target = sentinels + corrupted tokens + EOS
    assert spans + corrupted + 1 == 1057


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rate,mean", [(0.15, 3.0), (0.50, 32.0), (0.05, 1.5), (0.9, 8.0)])
def test_plan_span_counts_always_leaves_room_for_the_interleave(length, rate, mean):
    """
    Both segmentations must be satisfiable: every span needs >= 1 corrupted
    token and >= 1 kept token in front of it. Violating either is what makes
    `random_segmentation` raise, so the clamps are load-bearing.
    """
    corrupted, spans = plan_span_counts(length, rate, mean)
    kept = length - corrupted

    assert 1 <= corrupted <= length - 1
    assert kept >= 1
    assert 1 <= spans <= corrupted
    assert spans <= kept


@pytest.mark.parametrize("length", [0, 1])
def test_plan_span_counts_rejects_uncorruptible_lengths(length):
    with pytest.raises(ConfigError, match="cannot corrupt"):
        plan_span_counts(length, 0.15, 3.0)


def test_plan_span_counts_is_pure():
    assert plan_span_counts(500, 0.15, 3.0) == plan_span_counts(500, 0.15, 3.0)


def test_min_corruptible_length_is_two():
    """One masked and one kept token is the minimum meaningful example."""
    assert MIN_CORRUPTIBLE_LENGTH == 2


# -- random_segmentation --------------------------------------------------


@pytest.mark.parametrize("items,segments", [(1, 1), (10, 1), (10, 3), (10, 10), (307, 102)])
def test_random_segmentation_partitions_exactly(items, segments):
    lengths = random_segmentation(items, segments, random.Random(0))
    assert len(lengths) == segments
    assert sum(lengths) == items
    assert all(length >= 1 for length in lengths), "a zero-length segment breaks non-adjacency"


def test_random_segmentation_is_deterministic_given_a_seed():
    a = random_segmentation(100, 7, random.Random(99))
    b = random_segmentation(100, 7, random.Random(99))
    assert a == b


def test_random_segmentation_varies_across_seeds():
    a = random_segmentation(100, 7, random.Random(1))
    b = random_segmentation(100, 7, random.Random(2))
    assert a != b


def test_random_segmentation_rejects_impossible_split():
    with pytest.raises(ConfigError, match="cannot split"):
        random_segmentation(3, 5, random.Random(0))


def test_random_segmentation_rejects_zero_segments():
    with pytest.raises(ConfigError, match=">= 1"):
        random_segmentation(10, 0, random.Random(0))


def test_random_segmentation_explores_many_compositions():
    """Not a fixed layout: the sampler must actually randomize."""
    seen = {tuple(random_segmentation(20, 4, random.Random(seed))) for seed in range(50)}
    assert len(seen) > 10


# -- sample_spans ---------------------------------------------------------


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rate,mean", [(0.15, 3.0), (0.50, 32.0), (0.25, 5.0)])
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_spans_are_well_formed(length, rate, mean, seed):
    spans = sample_spans(length, rate, mean, random.Random(seed))

    assert spans, "at least one span is always produced"
    for start, end in spans:
        assert 0 <= start < end <= length, f"span {(start, end)} escapes [0, {length})"


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rate,mean", [(0.15, 3.0), (0.50, 32.0), (0.25, 5.0)])
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_spans_are_sorted_and_non_overlapping(length, rate, mean, seed):
    spans = sample_spans(length, rate, mean, random.Random(seed))
    for (_, previous_end), (start, _) in zip(spans, spans[1:]):
        assert start >= previous_end


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rate,mean", [(0.15, 3.0), (0.50, 32.0), (0.25, 5.0)])
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_spans_are_never_adjacent(length, rate, mean, seed):
    """
    At least one kept token must separate consecutive spans. Without this,
    two spans merge into one sentinel and the configured mean span length
    silently stops being the realized one.
    """
    spans = sample_spans(length, rate, mean, random.Random(seed))
    for (_, previous_end), (start, _) in zip(spans, spans[1:]):
        assert start > previous_end, "adjacent spans would collapse into one sentinel"


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rate,mean", [(0.15, 3.0), (0.50, 32.0), (0.25, 5.0)])
def test_spans_never_start_at_zero(length, rate, mean):
    """
    The interleave is kept-first, so the encoder always opens with real
    content rather than a bare sentinel.
    """
    spans = sample_spans(length, rate, mean, random.Random(3))
    assert spans[0][0] >= 1


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rate,mean", [(0.15, 3.0), (0.50, 32.0), (0.25, 5.0)])
def test_span_counts_match_the_plan(length, rate, mean):
    """Sampling must realize exactly what `plan_span_counts` promised."""
    corrupted, num_spans = plan_span_counts(length, rate, mean)
    spans = sample_spans(length, rate, mean, random.Random(5))

    assert len(spans) == num_spans
    assert sum(end - start for start, end in spans) == corrupted


def test_sample_spans_is_deterministic_given_a_seed():
    a = sample_spans(500, 0.15, 3.0, random.Random(42))
    b = sample_spans(500, 0.15, 3.0, random.Random(42))
    assert a == b


def test_sample_spans_varies_across_seeds():
    a = sample_spans(500, 0.15, 3.0, random.Random(1))
    b = sample_spans(500, 0.15, 3.0, random.Random(2))
    assert a != b


def test_sample_spans_respects_the_sentinel_budget():
    """architecture.md 7.4: fail loudly, never wrap or drop."""
    with pytest.raises(SentinelBudgetExceededError, match="sentinels exist"):
        sample_spans(2048, 0.15, 3.0, random.Random(0), max_spans=50)


def test_sample_spans_within_budget_does_not_raise():
    spans = sample_spans(2048, 0.15, 3.0, random.Random(0), max_spans=256)
    assert len(spans) == 102


def test_sample_spans_budget_error_names_the_shortfall():
    """The message must be actionable at 3am, not just 'assertion failed'."""
    with pytest.raises(SentinelBudgetExceededError) as excinfo:
        sample_spans(2048, 0.15, 3.0, random.Random(0), max_spans=10)
    message = str(excinfo.value)
    assert "102" in message and "10" in message


def test_realized_mean_span_length_tracks_the_configuration():
    """Aggregate check that the mean is achieved, not merely intended."""
    total_corrupted = 0
    total_spans = 0
    for seed in range(200):
        spans = sample_spans(512, 0.15, 3.0, random.Random(seed))
        total_corrupted += sum(end - start for start, end in spans)
        total_spans += len(spans)

    assert 2.9 <= total_corrupted / total_spans <= 3.1


def test_realized_corruption_rate_tracks_the_configuration():
    total_corrupted = 0
    for seed in range(200):
        spans = sample_spans(512, 0.15, 3.0, random.Random(seed))
        total_corrupted += sum(end - start for start, end in spans)

    assert 0.14 <= total_corrupted / (200 * 512) <= 0.16


def test_minimum_length_produces_one_span_of_one_token():
    spans = sample_spans(2, 0.15, 3.0, random.Random(0))
    assert spans == [(1, 2)]


# -- sample_prefix_split --------------------------------------------------


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("seed", range(5))
def test_prefix_split_leaves_both_sides_non_empty(length, seed):
    prefix = sample_prefix_split(length, 0.25, 0.75, random.Random(seed))
    assert 1 <= prefix <= length - 1


def test_prefix_split_honours_its_fraction_range():
    prefixes = [sample_prefix_split(1000, 0.25, 0.75, random.Random(s)) for s in range(200)]
    assert min(prefixes) >= 250
    assert max(prefixes) <= 750
    assert 450 <= sum(prefixes) / len(prefixes) <= 550


def test_prefix_split_with_a_fixed_fraction_is_exact():
    assert sample_prefix_split(100, 0.5, 0.5, random.Random(0)) == 50


def test_prefix_split_is_deterministic_given_a_seed():
    a = sample_prefix_split(500, 0.25, 0.75, random.Random(11))
    b = sample_prefix_split(500, 0.25, 0.75, random.Random(11))
    assert a == b


@pytest.mark.parametrize("length", [0, 1])
def test_prefix_split_rejects_uncorruptible_lengths(length):
    with pytest.raises(ConfigError, match="cannot split"):
        sample_prefix_split(length, 0.25, 0.75, random.Random(0))


def test_prefix_split_clamps_extreme_fractions():
    """
    A 0.99 fraction on a short sequence would otherwise leave an empty
    continuation, giving the decoder nothing to learn from.
    """
    assert sample_prefix_split(3, 0.99, 0.99, random.Random(0)) == 2
    assert sample_prefix_split(3, 0.01, 0.01, random.Random(0)) == 1
