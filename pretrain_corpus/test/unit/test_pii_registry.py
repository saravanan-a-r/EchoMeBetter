from __future__ import annotations

import json

import pytest

from src.errors import PIIRegistryError
from src.pii_registry import PIIRegistry, load_legacy_reserved


def test_reserve_returns_disjoint_ranges_per_worker(tmp_path):
    registry = PIIRegistry(tmp_path / "registry.json")
    reservations = registry.reserve({"email": 100, "handle": 50, "phone": 10}, workers=3)
    assert len(reservations) == 3
    email_ranges = [r.range_for("email") for r in reservations]
    starts = [r.start for r in email_ranges]
    assert starts == sorted(starts)
    for a, b in zip(email_ranges, email_ranges[1:]):
        assert a.stop == b.start  # contiguous, no gap, no overlap


def test_reservations_advance_the_high_water_mark_across_calls(tmp_path):
    path = tmp_path / "registry.json"
    registry = PIIRegistry(path)
    [first] = registry.reserve({"email": 100}, workers=1)
    [second] = registry.reserve({"email": 100}, workers=1)
    assert second.range_for("email").start == first.range_for("email").stop


def test_reservations_survive_reopening_the_registry(tmp_path):
    path = tmp_path / "registry.json"
    [first] = PIIRegistry(path).reserve({"email": 50}, workers=1)
    [second] = PIIRegistry(path).reserve({"email": 50}, workers=1)
    assert second.range_for("email").start == first.range_for("email").stop == 50


def test_unknown_pii_kind_is_rejected(tmp_path):
    registry = PIIRegistry(tmp_path / "registry.json")
    with pytest.raises(PIIRegistryError):
        registry.reserve({"carrier_pigeon": 10}, workers=1)


def test_corrupt_registry_file_raises_loudly(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("not valid json{{{")
    registry = PIIRegistry(path)
    with pytest.raises(PIIRegistryError):
        registry.reserve({"email": 10}, workers=1)


def test_snapshot_reports_current_counters(tmp_path):
    registry = PIIRegistry(tmp_path / "registry.json")
    registry.reserve({"email": 30}, workers=2)
    snapshot = registry.snapshot()
    assert snapshot["next_index"]["email"] == 60
    assert snapshot["reservations_made"] == 2


def test_load_legacy_reserved_returns_frozensets(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"emails": ["a@b.com"], "phones": ["555-1234"]}))
    legacy = load_legacy_reserved(path)
    assert legacy["email"] == frozenset({"a@b.com"})
    assert legacy["phone"] == frozenset({"555-1234"})
    assert legacy["handle"] == frozenset()


def test_load_legacy_reserved_missing_file_raises():
    with pytest.raises(PIIRegistryError):
        load_legacy_reserved("/does/not/exist.json")


def test_load_legacy_reserved_default_path_has_the_real_carried_over_values():
    legacy = load_legacy_reserved()
    assert len(legacy["email"]) > 1000
    assert len(legacy["phone"]) > 100
