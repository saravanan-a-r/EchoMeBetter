"""
The property the whole parallelisation rests on: **worker count is an
execution detail and must never change a result.**

Before this, `corpus_reader` and `validate_tokenizer` streamed the corpus in a
single Python loop, so one core did all the work while the rest of the machine
idled. Splitting that work across processes is only safe if the split is
invisible in the output -- otherwise a run's sampled corpus, its manifest, and
its gate metrics would silently depend on how busy the machine was that day.

Each test here therefore runs the same operation at several worker counts and
asserts the results are *identical*, not merely similar.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import corpus_reader  # noqa: E402
from execution import Executor  # noqa: E402

WORKER_COUNTS = [1, 2, 4]

BLEND = {"mode": "uniform", "shares": {}}


@pytest.fixture
def corpus(tmp_path: Path) -> list[Path]:
    """
    A multi-file, multi-source corpus with deliberately uneven file sizes --
    the shape that makes ordering bugs and per-file quota bugs actually show
    up, rather than a tidy grid where every file holds the same count.
    """
    rng = random.Random(20260906)
    words = ["alpha", "beta", "gamma", "delta", "kernel", "socket", "daemon", "mount"]
    sizes = {"alpha": [400, 130, 7], "beta": [250, 61], "gamma": [90]}
    for source, counts in sizes.items():
        for shard, n_lines in enumerate(counts):
            path = tmp_path / f"{source}_{shard}mb.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for i in range(n_lines):
                    text = " ".join(rng.choice(words) for _ in range(rng.randint(4, 25)))
                    fh.write(json.dumps({"text": f"{source}-{shard}-{i} {text}"}) + "\n")
    return corpus_reader.discover_corpus_files(tmp_path)


def _build(corpus, out: Path, workers: int, budget: int = 300, seed: int = 42) -> dict:
    return corpus_reader.build_corpus(
        corpus,
        corpus_out=out,
        train_budget=budget,
        seed=seed,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
        blend=BLEND,
        executor=Executor(workers=workers),
    )


# --------------------------------------------------------------------------
# pass 1: the scan
# --------------------------------------------------------------------------


def test_scan_totals_are_identical_at_every_worker_count(corpus):
    baseline = corpus_reader.count_eligible_lines(corpus, 8192, Executor(workers=1))
    for workers in WORKER_COUNTS[1:]:
        got = corpus_reader.count_eligible_lines(corpus, 8192, Executor(workers=workers))
        assert got.as_dict() == baseline.as_dict(), f"scan differs at workers={workers}"


def test_scan_counts_every_file_once(corpus):
    stats = corpus_reader.count_eligible_lines(corpus, 8192, Executor(workers=4))
    assert stats.files == len(corpus)
    assert stats.lines_eligible == 400 + 130 + 7 + 250 + 61 + 90


# --------------------------------------------------------------------------
# pass 2: the sampled corpus
# --------------------------------------------------------------------------


def test_corpus_txt_is_byte_identical_at_every_worker_count(tmp_path, corpus):
    """
    The strongest statement available: not "the same number of lines", not
    "the same sources in the same proportions", but the same bytes in the
    same order.
    """
    baseline_path = tmp_path / "baseline.txt"
    _build(corpus, baseline_path, workers=1)
    baseline = baseline_path.read_bytes()

    for workers in WORKER_COUNTS[1:]:
        other = tmp_path / f"w{workers}.txt"
        _build(corpus, other, workers=workers)
        assert other.read_bytes() == baseline, f"corpus.txt differs at workers={workers}"


def test_manifest_and_plan_are_identical_at_every_worker_count(tmp_path, corpus):
    baseline = _build(corpus, tmp_path / "a.txt", workers=1)
    for workers in WORKER_COUNTS[1:]:
        got = _build(corpus, tmp_path / f"b{workers}.txt", workers=workers)
        for key in (
            "plan",
            "scan",
            "train_lines",
            "train_lines_per_source",
            "train_bytes_per_source",
            "eval_lines_available",
            "corpus_sha256",
        ):
            assert got[key] == baseline[key], f"{key} differs at workers={workers}"


def test_sampling_still_hits_the_requested_budget_exactly(tmp_path, corpus):
    info = _build(corpus, tmp_path / "c.txt", workers=4, budget=300)
    assert info["train_lines"] == 300
    assert sum(info["train_lines_per_source"].values()) == 300


def test_per_file_quotas_sum_to_the_global_quota(tmp_path, corpus):
    """
    The invariant that replaces the old single global Algorithm S: each
    source's budget is cut into per-file pieces that must add back up.
    """
    info = _build(corpus, tmp_path / "d.txt", workers=2)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])
    for source, total in plan.train_per_source.items():
        per_file = sum(quota.get(source, 0) for quota in plan.file_train)
        assert per_file == total, f"{source}: per-file quotas sum to {per_file}, want {total}"


def test_no_file_is_asked_for_more_lines_than_it_holds(tmp_path, corpus):
    info = _build(corpus, tmp_path / "e.txt", workers=2)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])
    for capacities, quotas in zip(plan.file_capacities, plan.file_train):
        for source, quota in quotas.items():
            assert quota <= capacities.get(source, 0)


# --------------------------------------------------------------------------
# the train/eval split
# --------------------------------------------------------------------------


def test_eval_replay_reproduces_the_written_corpus_after_parallel_build(tmp_path, corpus):
    """
    corpus.txt is written by the parallel path; the split is replayed
    sequentially afterwards to stream eval. If those two ever disagreed, the
    gates would be measuring text the tokenizer was trained on.
    """
    out = tmp_path / "f.txt"
    info = _build(corpus, out, workers=4)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])

    written = out.read_text(encoding="utf-8").splitlines()
    replayed = [t for t, _label, is_train in corpus_reader.iter_split(corpus, plan) if is_train]
    assert replayed == written


def test_train_and_eval_never_overlap(tmp_path, corpus):
    out = tmp_path / "g.txt"
    info = _build(corpus, out, workers=4)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])

    trained = set(out.read_text(encoding="utf-8").splitlines())
    evaluated = {text for text, _ in corpus_reader.iter_eval_lines(corpus, plan)}
    assert trained
    assert evaluated
    assert not (trained & evaluated)


def test_eval_stream_is_identical_at_every_worker_count(tmp_path, corpus):
    baseline_info = _build(corpus, tmp_path / "h.txt", workers=1)
    baseline_plan = corpus_reader.SplitPlan.from_dict(baseline_info["plan"])
    baseline = list(corpus_reader.iter_eval_lines(corpus, baseline_plan, limit=120))

    for workers in WORKER_COUNTS[1:]:
        info = _build(corpus, tmp_path / f"i{workers}.txt", workers=workers)
        plan = corpus_reader.SplitPlan.from_dict(info["plan"])
        assert list(corpus_reader.iter_eval_lines(corpus, plan, limit=120)) == baseline


def test_capped_eval_draws_from_across_the_corpus(tmp_path, corpus):
    """
    A capped eval run must sample the whole corpus, not just exhaust the first
    files it happens to open -- otherwise `--eval-corpus N` would silently
    measure one source instead of all of them.
    """
    info = _build(corpus, tmp_path / "j.txt", workers=2)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])
    sources = {label for _t, label in corpus_reader.iter_eval_lines(corpus, plan, limit=150)}
    assert sources == {"alpha", "beta", "gamma"}


def test_eval_cap_is_respected_exactly(tmp_path, corpus):
    info = _build(corpus, tmp_path / "k.txt", workers=2)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])
    lines = list(corpus_reader.iter_eval_lines(corpus, plan, limit=100))
    assert len(lines) == 100


def test_uncapped_eval_yields_everything_not_trained_on(tmp_path, corpus):
    info = _build(corpus, tmp_path / "l.txt", workers=4)
    plan = corpus_reader.SplitPlan.from_dict(info["plan"])
    lines = list(corpus_reader.iter_eval_lines(corpus, plan))
    assert len(lines) == plan.n_eval == info["eval_lines_available"]


def test_a_different_seed_still_changes_the_sample(tmp_path, corpus):
    """Parallelism must not accidentally pin the RNG to something seed-independent."""
    a = _build(corpus, tmp_path / "m.txt", workers=4, seed=1)
    b = _build(corpus, tmp_path / "n.txt", workers=4, seed=2)
    assert (tmp_path / "m.txt").read_bytes() != (tmp_path / "n.txt").read_bytes()
    assert a["train_lines"] == b["train_lines"]  # same budget, different draw
