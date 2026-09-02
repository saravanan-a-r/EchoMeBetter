"""
End-to-end test of `download.run()`: the top-level orchestrator that splits
a GB budget across multiple sources and runs them concurrently, each with
its own multi-worker process pool.

This is the regression test for a real bug: the outer "how many sources run
at once" layer used to be a `multiprocessing.Pool`, and each of its workers
started a *second*, inner process pool (one process pool per source, for
`--workers-per-source`). `multiprocessing.Pool` workers are daemonic, and a
daemonic process is forbidden from starting children of its own -- so the
very first run with more than one source and more than one worker per
source (i.e. any real invocation) crashed immediately with
"AssertionError: daemonic processes are not allowed to have children".
The fix makes the outer layer a `concurrent.futures.ProcessPoolExecutor`
(non-daemonic workers, and each one is single-threaded when it opens its
own inner Pool, avoiding the separate fork-from-multi-threaded-process
hazard a `ThreadPoolExecutor` would have introduced instead); this test
exercises that exact nested-pool shape (2 sources concurrently, 2 workers
each) end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import download as D

_DOCS_A = [
    "The quarterly report showed strong growth in the European market this year, "
    "driven mainly by increased demand for cloud infrastructure across the region.",
    "SentencePiece with byte fallback keeps every character representable always, "
    "which matters a great deal once real user text starts arriving at inference time.",
] * 10
_DOCS_B = [
    "A small encoder-decoder transformer can still learn fluent English text well "
    "provided the pretraining corpus is large enough and reasonably diverse in register.",
    "The committee reviewed the roadmap and agreed on next quarter's priorities, "
    "focusing mainly on reliability work instead of shipping new user-facing features.",
] * 10

_FAKE_DOCS = {
    "fineweb_edu": _DOCS_A,
    "c4": _DOCS_B,
}


def _fake_iter_hf_records(spec, *, seed, shard_index=0, num_shards=1):
    docs = _FAKE_DOCS[spec.source_id]
    for i in range(shard_index, len(docs), num_shards):
        yield docs[i]


@pytest.fixture(autouse=True)
def _patch_hf_source(monkeypatch):
    monkeypatch.setattr(D, "iter_hf_records", _fake_iter_hf_records)
    # See test_download_pipeline.py's identical fixture: check_hf_dependencies()
    # does a real `import datasets` in the parent right before this test's
    # `mp.get_context("fork")` calls, which can crash/hang on macOS. These
    # tests fake the data-fetching entirely and don't need the real check.
    monkeypatch.setattr(D, "check_hf_dependencies", lambda: None)


# Every test here calls D.run(args), which now defaults --start-method to
# "spawn" on macOS (download.py's DEFAULT_START_METHOD, chosen because fork
# is unsafe there once a library has touched Apple's Objective-C runtime).
# These tests rely on the `iter_hf_records` monkeypatch above reaching the
# worker process -- which only happens under fork, since spawn re-imports
# `download` fresh in the child and never sees a patch applied to this
# process's copy of the module. Without pinning "fork" explicitly, these
# tests would silently call the REAL iter_hf_records in the spawned worker
# and hang trying to reach the real network. This has nothing to do with
# download.py's actual (correct) platform-aware default -- it's a test-only
# requirement, so it's added here rather than by changing that default.
START_METHOD_ARGS = ["--start-method", "fork"]


def test_run_with_multiple_sources_and_multiple_workers_each_does_not_deadlock_or_crash(tmp_path):
    """
    The exact nested-pool shape that used to raise
    "daemonic processes are not allowed to have children": two sources
    running concurrently (--max-parallel-sources 2), each with its own
    2-worker process pool (--workers-per-source 2).
    """
    argv = [
        "--gb", "0.00001",  # tiny but non-zero after per-source/per-worker splitting
        "--sources", "fineweb_edu,c4",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
        "--max-parallel-sources", "2",
        "--workers-per-source", "2",
        "--shard-gb", "0.001",
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    exit_code = D.run(args)
    assert exit_code == 0

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["failed_sources"] == []
    assert set(manifest["sources"]) == {"fineweb_edu", "c4"}
    assert manifest["total_bytes_written"] > 0


def test_share_override_changes_the_blend(tmp_path):
    argv = [
        "--gb", "0.00001",
        "--sources", "fineweb_edu,c4",
        "--share", "fineweb_edu=90",
        "--share", "c4=10",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
        "--workers-per-source", "1",
        "--max-parallel-sources", "1",
        "--shard-gb", "0.001",
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    exit_code = D.run(args)
    assert exit_code == 0
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["blend"]["fineweb_edu"] == pytest.approx(0.9)
    assert manifest["blend"]["c4"] == pytest.approx(0.1)


def test_unknown_source_is_rejected(tmp_path):
    argv = [
        "--gb", "1",
        "--sources", "not_a_real_source",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    assert D.run(args) == 2


def test_malformed_share_override_is_rejected(tmp_path):
    argv = [
        "--gb", "1",
        "--share", "fineweb_edu_no_equals_sign",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    assert D.run(args) == 2


def test_share_override_with_a_non_numeric_percent_is_rejected_cleanly(tmp_path):
    """
    Regression test: `float(pct)` used to be called with no try/except, so a
    typo like `--share fineweb_edu=notanumber` crashed with a raw, unhandled
    ValueError traceback instead of the clean `error: bad --share ...`
    message the surrounding code was clearly trying to produce.
    """
    argv = [
        "--gb", "1",
        "--share", "fineweb_edu=notanumber",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    assert D.run(args) == 2


def test_main_rejects_non_positive_gb(tmp_path, capsys):
    exit_code = D.main(["--gb", "0", "--output-dir", str(tmp_path / "out")])
    assert exit_code == 2


def test_a_source_running_dry_is_reported_as_underfilled_not_silent(tmp_path):
    """
    Regression test: a source whose real dataset is smaller than its
    requested share (the SlimPajama ungated fallback caps at ~24GB
    regardless of share; Cosmopedia/permissive code are much smaller than
    FineWeb-Edu/C4/PG19) used to finish with no error and no signal
    anywhere that it delivered far less than requested -- `iter_hf_records`
    just stops yielding, `job.result()` returns normally, and the manifest
    looked identical to a fully-satisfied run. `run()` must now flag this
    in `underfilled_sources` and print a warning, not stay silent.
    """
    argv = [
        "--gb", "1",  # far more than the ~320 bytes fineweb_edu's 20 fake docs can provide
        "--sources", "fineweb_edu",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
        "--workers-per-source", "1",
        "--max-parallel-sources", "1",
        "--shard-gb", "0.001",
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    exit_code = D.run(args)

    assert exit_code == 0  # under-filling is a warning, not a hard failure
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert "fineweb_edu" in manifest["underfilled_sources"]
    assert manifest["underfilled_sources"]["fineweb_edu"]["pct_of_requested"] < 90.0


def test_one_source_failing_does_not_abort_the_others(tmp_path, monkeypatch):
    """
    download.py's whole reason for running sources independently: a broken
    source (bad credentials, a renamed HF dataset, a network blip) must be
    recorded and skipped, not take down a multi-day run downloading six
    other sources successfully.
    """
    def _flaky_iter_hf_records(spec, *, seed, shard_index=0, num_shards=1):
        if spec.source_id == "c4":
            raise RuntimeError("simulated: c4 dataset unreachable")
        yield from _fake_iter_hf_records(spec, seed=seed, shard_index=shard_index, num_shards=num_shards)

    monkeypatch.setattr(D, "iter_hf_records", _flaky_iter_hf_records)

    argv = [
        "--gb", "0.00001",
        "--sources", "fineweb_edu,c4",
        "--output-dir", str(tmp_path / "out"),
        "--registry", str(tmp_path / "registry.json"),
        "--workers-per-source", "1",
        "--max-parallel-sources", "2",
        "--shard-gb", "0.001",
    ]
    argv = argv + START_METHOD_ARGS
    args = D.parse_args(argv)
    exit_code = D.run(args)

    assert exit_code == 1  # failure reported, but the run completed
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["failed_sources"] == ["c4"]
    assert "fineweb_edu" in manifest["sources"]
    assert manifest["sources"]["fineweb_edu"]["bytes_written"] > 0
