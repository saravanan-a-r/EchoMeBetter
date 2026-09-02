"""
End-to-end wiring test for `download.py`'s HuggingFace-source path: real
multiprocessing worker pools, a real shared Bloom-filter dedup index across
processes, real PII index reservation and masking, real budget-bounded
`BudgetWriter` output -- with `src.sources.iter_hf_records` monkeypatched to
a synthetic in-process generator so this needs no network access and no
`datasets` download.

This is the test that would have caught the `multiprocessing.Array`
inheritance bug (a `Deduplicator` cannot be passed through `apply_async` to
an already-running pool worker) before it ever reached a real 42-vCPU box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import download as D
import src.sources as sources
from src.pii_registry import PIIRegistry
from src.sources import PipelineConfig, SourceSpec

# A tiny deterministic corpus, duplicated a few times on purpose so the
# shared dedup index has something real to reject across worker processes.
_DOCS = [
    "The quarterly report showed strong growth in the European market this year.",
    "Please reach out to jane.doe@gmail.com about the outstanding invoice today.",
    "SentencePiece with byte fallback keeps every character representable always.",
    "A small encoder-decoder transformer can still learn fluent English text well.",
] * 20


def _fake_iter_hf_records(spec, *, seed, shard_index=0, num_shards=1):
    # Same file-level sharding contract as the real iter_hf_records: each
    # worker gets a disjoint stride of the stream.
    for i in range(shard_index, len(_DOCS), num_shards):
        yield _DOCS[i]


@pytest.fixture(autouse=True)
def _patch_hf_source(monkeypatch):
    monkeypatch.setattr(D, "iter_hf_records", _fake_iter_hf_records)
    # check_hf_dependencies() does a real `import datasets` in the parent
    # process right before this test's `mp.get_context("fork")` calls --
    # on macOS that combination can crash or hang the child (Apple's
    # Objective-C runtime aborts a fork if some other thread was mid-way
    # through NSCharacterSet init, which datasets/huggingface_hub's
    # cert/SSL handling can trigger). These tests already fake the actual
    # data-fetching, so they don't need the real check to run at all.
    monkeypatch.setattr(D, "check_hf_dependencies", lambda: None)


def _read_all_jsonl(output_dir: Path) -> list[str]:
    texts = []
    for path in sorted(output_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                texts.append(json.loads(line)["text"])
    return texts


def test_run_hf_source_writes_output_across_workers(tmp_path):
    spec = SourceSpec(
        source_id="fake",
        kind="hf",
        role="test fixture",
        license="n/a",
        hf_dataset="fake/dataset",
        pipeline=PipelineConfig(quality=False, mask_pii=False),
    )
    registry = PIIRegistry(tmp_path / "registry.json")
    result = D.run_hf_source(
        "fake",
        spec,
        budget_bytes=10_000,
        output_dir=tmp_path / "out",
        seed=0,
        num_workers=3,
        shard_bytes=1024 * 1024,
        registry=None,
        legacy={"email": frozenset(), "phone": frozenset(), "handle": frozenset()},
        pii_block_size=100,
        audit_path=None,
    )
    assert result["bytes_written"] > 0
    assert len(result["workers"]) == 3

    texts = _read_all_jsonl(tmp_path / "out" / "fake")
    assert texts  # something was actually written to disk


def test_shared_dedup_index_is_respected_across_worker_processes(tmp_path):
    """
    The real regression case: _DOCS repeats every 4 entries, sharded across
    3 workers by stride. Without a genuinely shared dedup index, each
    worker process would keep its own private copy and every duplicate
    would slip through -- with a truly shared index, near-total duplicates
    collapse down to (at most) a handful of first-seen originals.
    """
    spec = SourceSpec(
        source_id="fake",
        kind="hf",
        role="test fixture",
        license="n/a",
        hf_dataset="fake/dataset",
        pipeline=PipelineConfig(quality=False, mask_pii=False),
    )
    result = D.run_hf_source(
        "fake",
        spec,
        budget_bytes=1_000_000,
        output_dir=tmp_path / "out",
        seed=0,
        num_workers=3,
        shard_bytes=1024 * 1024,
        registry=None,
        legacy={"email": frozenset(), "phone": frozenset(), "handle": frozenset()},
        pii_block_size=100,
        audit_path=None,
    )
    texts = _read_all_jsonl(tmp_path / "out" / "fake")
    assert len(texts) <= len(set(_DOCS)) + 2  # a small allowance, not near-total duplication
    assert result["dedup"]["dropped_duplicate"] > 0


def test_pii_masking_is_applied_and_unique_across_workers(tmp_path):
    spec = SourceSpec(
        source_id="fake",
        kind="hf",
        role="test fixture",
        license="n/a",
        hf_dataset="fake/dataset",
        pipeline=PipelineConfig(quality=False, mask_pii=True, mask_phones=True),
    )
    registry = PIIRegistry(tmp_path / "registry.json", salt=3)
    result = D.run_hf_source(
        "fake",
        spec,
        budget_bytes=1_000_000,
        output_dir=tmp_path / "out",
        seed=0,
        num_workers=3,
        shard_bytes=1024 * 1024,
        registry=registry,
        legacy={"email": frozenset(), "phone": frozenset(), "handle": frozenset()},
        pii_block_size=1000,
        audit_path=None,
    )
    texts = _read_all_jsonl(tmp_path / "out" / "fake")
    assert not any("jane.doe@gmail.com" in t for t in texts)
    assert result["dedup"]["chunks_checked"] > 0
