from __future__ import annotations

import json

import pytest

from src.writer import BudgetWriter, estimate_tokens, gb_to_bytes, write_manifest


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_write_stops_once_budget_is_reached(tmp_path):
    writer = BudgetWriter("test", tmp_path, budget_bytes=200, shard_bytes=10_000)
    for i in range(50):
        if writer.done:
            break
        writer.write(f"line number {i} with some padding text to add bytes")
    stats = writer.close()
    assert stats["bytes_written"] >= 200
    assert stats["lines_written"] > 0


def test_output_schema_is_text_jsonl(tmp_path):
    writer = BudgetWriter("test", tmp_path, budget_bytes=10_000, shard_bytes=10_000)
    writer.write("hello world")
    writer.close()
    [record] = _read_jsonl(tmp_path / "test-00000.jsonl")
    assert record == {"text": "hello world"}


def test_rotates_into_multiple_shards_when_shard_bytes_is_small(tmp_path):
    writer = BudgetWriter("test", tmp_path, budget_bytes=1_000_000, shard_bytes=50)
    for i in range(20):
        writer.write(f"this is line number {i} padded out a bit further")
    stats = writer.close()
    assert len(stats["shards"]) >= 2
    for shard in stats["shards"]:
        assert len(shard["sha256"]) == 64


def test_shard_files_are_named_predictably(tmp_path):
    writer = BudgetWriter("mysource", tmp_path, budget_bytes=1_000_000, shard_bytes=30)
    for i in range(10):
        writer.write(f"line {i} with padding text")
    writer.close()
    names = sorted(p.name for p in tmp_path.glob("mysource-*.jsonl"))
    assert names[0] == "mysource-00000.jsonl"


def test_close_is_idempotent_safe_with_no_writes(tmp_path):
    writer = BudgetWriter("empty", tmp_path, budget_bytes=1000)
    stats = writer.close()
    assert stats["lines_written"] == 0
    assert stats["shards"] == []


def test_gb_to_bytes_conversion():
    assert gb_to_bytes(1) == 1024 * 1024 * 1024
    with pytest.raises(ValueError):
        gb_to_bytes(0)


def test_estimate_tokens_uses_bytes_per_token_constant():
    assert estimate_tokens(4_000_000, bytes_per_token=4.0) == 1_000_000


def test_write_manifest_is_atomic_and_readable(tmp_path):
    path = tmp_path / "sub" / "manifest.json"
    write_manifest(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not path.with_suffix(".json.tmp").exists()
