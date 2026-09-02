from __future__ import annotations

from src.clean import dedup_key
from src.dedup import SharedBloomFilter, build_deduplicator, estimate_chunks


def test_first_occurrence_is_accepted():
    filt = SharedBloomFilter(1000)
    assert filt.add_if_absent(12345)


def test_exact_repeat_is_rejected():
    filt = SharedBloomFilter(1000)
    filt.add_if_absent(12345)
    assert not filt.add_if_absent(12345)


def test_distinct_keys_are_both_accepted():
    filt = SharedBloomFilter(1000)
    assert filt.add_if_absent(1)
    assert filt.add_if_absent(2)


def test_capacity_below_one_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        SharedBloomFilter(0)


def test_deduplicator_accepts_first_and_rejects_exact_repeat():
    dedup = build_deduplicator(1000)
    key = dedup_key("hello world, this is a test document.")
    assert dedup.accept(key)
    assert not dedup.accept(key)


def test_deduplicator_rejects_near_duplicate_after_normalization():
    dedup = build_deduplicator(1000)
    assert dedup.accept(dedup_key("Hello   World! This Is A Test."))
    assert not dedup.accept(dedup_key("hello world this is a test"))


def test_deduplicator_accepts_genuinely_different_content():
    dedup = build_deduplicator(1000)
    assert dedup.accept(dedup_key("first unique document here"))
    assert dedup.accept(dedup_key("second, completely different document"))


def test_deduplicator_report_counts_are_accurate():
    dedup = build_deduplicator(1000)
    dedup.accept(dedup_key("alpha"))
    dedup.accept(dedup_key("alpha"))  # exact dup
    dedup.accept(dedup_key("beta"))
    report = dedup.as_dict()
    assert report["chunks_checked"] == 3
    assert report["dropped_exact_duplicate"] == 1


def test_estimate_chunks_scales_with_bytes():
    assert estimate_chunks(1_200_000_000, mean_chunk_bytes=1_200) == 1_000_000


def test_estimate_chunks_has_a_floor():
    assert estimate_chunks(1) >= 1_000
