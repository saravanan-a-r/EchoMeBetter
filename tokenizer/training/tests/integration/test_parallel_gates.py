"""
The eval scan -- the step that dominated a sweep's wall clock -- must produce
the same gate results no matter how many workers run it.

At ~6,200 lines/s on one core, the 50M-line eval used to take over two hours
per variant. It is now split per file. That is only legitimate if the merged
ScanStats is indistinguishable from the single-process one, because every gate
is a pure function of that object: if the merge were wrong, fertility,
byte-fallback rate and vocabulary utilization would all quietly shift with the
worker count.

These run against the real trained model from `conftest.py`, not a stub, so
the encode/decode path being measured is the production one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import corpus_reader  # noqa: E402
import validate_tokenizer as vt  # noqa: E402
from execution import Executor  # noqa: E402

WORKER_COUNTS = [1, 2, 4]


@pytest.fixture(scope="module")
def eval_setup(trained_run):
    """The corpus, plan and model a real eval scan would run against."""
    cfg = trained_run["cfg"]
    files = corpus_reader.discover_corpus_files(trained_run["corpus_dir"])
    stats, file_scans = corpus_reader.scan_corpus(
        files, cfg["corpus"]["max_sentence_length"], Executor(workers=1)
    )
    plan, _notes = corpus_reader.plan_split(
        stats,
        cfg["corpus"]["input_sentence_size"],
        cfg["sampling"]["seed"],
        cfg["corpus"]["max_sentence_length"],
        cfg["blend"],
        file_scans,
    )
    return {"cfg": cfg, "files": files, "plan": plan, "model_path": trained_run["model_path"]}


def _scan(eval_setup, workers: int, limit: int = 0, sample_budget: int = 50):
    return vt.scan_eval_corpus(
        eval_setup["model_path"],
        eval_setup["files"],
        eval_setup["plan"],
        limit=limit,
        length_reservoir_size=eval_setup["cfg"]["validation"]["length_reservoir_size"],
        seed=eval_setup["cfg"]["sampling"]["seed"],
        sample_budget=sample_budget,
        executor=Executor(workers=workers),
    )


def test_scan_counters_are_identical_at_every_worker_count(eval_setup):
    baseline = _scan(eval_setup, workers=1)
    assert baseline.n_records > 0, "fixture produced no eval lines to compare"

    for workers in WORKER_COUNTS[1:]:
        got = _scan(eval_setup, workers=workers)
        for field in (
            "n_records",
            "n_roundtrip_ok",
            "total_tokens",
            "total_words",
            "total_chars",
            "total_unk",
            "total_byte_tokens",
            "max_tokens_per_sentence",
        ):
            assert getattr(got, field) == getattr(baseline, field), (
                f"{field} differs at workers={workers}"
            )
        assert got.id_counts == baseline.id_counts, f"id_counts differ at workers={workers}"
        assert got.per_source == baseline.per_source, f"per_source differs at workers={workers}"
        assert got.url_fragments == baseline.url_fragments


def test_gate_results_are_identical_at_every_worker_count(eval_setup, sp):
    """What actually reaches the report: every gate's pass/fail and metrics."""
    baseline_results, _ = vt.run_all_gates(
        sp, None, eval_setup["cfg"], scan=_scan(eval_setup, workers=1)
    )
    baseline = [(r.number, r.passed, r.metrics) for r in baseline_results]

    for workers in WORKER_COUNTS[1:]:
        results, _ = vt.run_all_gates(
            sp, None, eval_setup["cfg"], scan=_scan(eval_setup, workers=workers)
        )
        assert [(r.number, r.passed, r.metrics) for r in results] == baseline, (
            f"gate results differ at workers={workers}"
        )


def test_parallel_scan_matches_the_sequential_streaming_scan(eval_setup, sp):
    """
    The parallel path against the ORIGINAL implementation: stream every eval
    line through `stream_scan` in one process, exactly as the pre-change code
    did, and require the same totals.
    """
    plan = eval_setup["plan"]
    lines = corpus_reader.iter_eval_lines(eval_setup["files"], plan)
    streamed = vt.stream_scan(
        sp,
        lines,
        vt._byte_piece_ids(sp),
        length_reservoir_size=eval_setup["cfg"]["validation"]["length_reservoir_size"],
        seed=eval_setup["cfg"]["sampling"]["seed"],
    )
    parallel = _scan(eval_setup, workers=4)

    assert parallel.n_records == streamed.n_records
    assert parallel.total_tokens == streamed.total_tokens
    assert parallel.total_chars == streamed.total_chars
    assert parallel.total_unk == streamed.total_unk
    assert parallel.total_byte_tokens == streamed.total_byte_tokens
    assert parallel.id_counts == streamed.id_counts


def test_reservoirs_stay_bounded_and_are_stable_across_worker_counts(eval_setup):
    size = eval_setup["cfg"]["validation"]["length_reservoir_size"]
    baseline = _scan(eval_setup, workers=1)
    assert len(baseline.length_reservoir) <= size
    assert len(baseline.sample) <= 50

    for workers in WORKER_COUNTS[1:]:
        got = _scan(eval_setup, workers=workers)
        assert len(got.length_reservoir) == len(baseline.length_reservoir)
        assert len(got.sample) == len(baseline.sample)
        # Deterministic: the same worker count must reproduce the same draw.
        assert _scan(eval_setup, workers=workers).sample == got.sample


def test_saved_sample_records_are_well_formed(eval_setup):
    scan = _scan(eval_setup, workers=4, sample_budget=25)
    assert scan.sample
    assert len(scan.sample) <= 25
    for record in scan.sample:
        assert set(record) == {"text", "source"}
        assert isinstance(record["text"], str) and record["text"].strip()


def test_capped_eval_scan_is_identical_at_every_worker_count(eval_setup):
    baseline = _scan(eval_setup, workers=1, limit=200)
    assert baseline.n_records == 200
    for workers in WORKER_COUNTS[1:]:
        got = _scan(eval_setup, workers=workers, limit=200)
        assert got.n_records == baseline.n_records
        assert got.total_tokens == baseline.total_tokens
        assert got.id_counts == baseline.id_counts
