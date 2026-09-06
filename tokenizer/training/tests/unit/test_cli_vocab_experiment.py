"""
Unit tests for run_cli_vocab_experiment.py.

Two of these (`test_build_flattened_corpus_view_*`) exercise the actual
regression this script exists to prevent: pretrain_corpus/'s
`<source>/<source>-w00-00000.jsonl` layout silently producing bogus,
unmatched blend labels. They assert against the REAL
`corpus_reader.source_label()`, not a reimplementation of it, so a future
change to either side that breaks the naming contract fails here.

`verify_forced_sources_saturated` is tested against the real
`corpus_reader.build_corpus` (a real scan-and-sample pass, no SentencePiece
training) on a small synthetic multi-source corpus -- proving the
saturation check actually catches an under-weighted config, not just that
it returns *something*.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import yaml

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import corpus_reader  # noqa: E402
import run_cli_vocab_experiment as experiment  # noqa: E402
from tests.corpus_factory import WORDS, _make_record  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: a tiny pretrain_corpus-shaped directory tree
# --------------------------------------------------------------------------


def _write_source(
    root: Path, source_id: str, n_lines: int, *, n_shards: int = 2, seed: int = 0
) -> None:
    """`<root>/<source_id>/<source_id>-w{i:02d}-00000.jsonl`, matching
    pretrain_corpus/download.py's BudgetWriter naming exactly."""
    source_dir = root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    handles = [
        open(source_dir / f"{source_id}-w{i:02d}-00000.jsonl", "w", encoding="utf-8")
        for i in range(n_shards)
    ]
    try:
        for i in range(n_lines):
            record = {"text": _make_record(rng)}
            handles[i % n_shards].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for fh in handles:
            fh.close()


@pytest.fixture
def pretrain_corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "pretrain_output"
    _write_source(root, "fineweb_edu", 4000, seed=1)
    _write_source(root, "c4", 2000, seed=2)
    _write_source(root, "cli_helper_stack_exchange", 1500, seed=3)
    # Deliberately tiny, mirroring the real webnye counts (thousands, not
    # millions) -- this is exactly the case build_flattened_corpus_view and
    # verify_forced_sources_saturated exist to handle correctly.
    _write_source(root, "cli_tldr", 80, n_shards=1, seed=4)
    _write_source(root, "cli_nl2bash", 60, n_shards=1, seed=5)
    _write_source(root, "cli_man_pages", 90, n_shards=1, seed=6)
    _write_source(root, "cli_cheat_sheets", 20, n_shards=1, seed=7)
    return root


# --------------------------------------------------------------------------
# build_flattened_corpus_view
# --------------------------------------------------------------------------


def test_flattened_view_labels_match_source_label_regex(tmp_path, pretrain_corpus_dir):
    """
    The actual regression this function exists to prevent: every symlinked
    file must parse back to its true source id via the REAL
    corpus_reader.source_label(), not a reimplementation of its regex.
    """
    dest = tmp_path / "flattened"
    counts = experiment.build_flattened_corpus_view(pretrain_corpus_dir, dest)

    assert counts == {
        "fineweb_edu": 2, "c4": 2, "cli_helper_stack_exchange": 2,
        "cli_tldr": 1, "cli_nl2bash": 1, "cli_man_pages": 1, "cli_cheat_sheets": 1,
    }
    for link in dest.iterdir():
        expected_source = link.name.rsplit("_", 1)[0]
        assert corpus_reader.source_label(link) == expected_source, link.name


def test_flattened_view_symlinks_preserve_content(tmp_path, pretrain_corpus_dir):
    dest = tmp_path / "flattened"
    experiment.build_flattened_corpus_view(pretrain_corpus_dir, dest)

    original = next((pretrain_corpus_dir / "cli_tldr").glob("*.jsonl"))
    linked = next(dest.glob("cli_tldr_*.jsonl"))
    assert linked.is_symlink()
    assert linked.read_text(encoding="utf-8") == original.read_text(encoding="utf-8")


def test_flattened_view_rebuilds_cleanly_when_run_twice(tmp_path, pretrain_corpus_dir):
    """A re-run (e.g. after adding a source) must not fail on stale symlinks."""
    dest = tmp_path / "flattened"
    experiment.build_flattened_corpus_view(pretrain_corpus_dir, dest)
    counts = experiment.build_flattened_corpus_view(pretrain_corpus_dir, dest)
    assert counts["fineweb_edu"] == 2


def test_flattened_view_raises_on_empty_corpus_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(experiment.ExperimentError):
        experiment.build_flattened_corpus_view(empty, tmp_path / "dest")


def test_flattened_view_ignores_nested_cache_directories(tmp_path, pretrain_corpus_dir):
    """
    Regression test: sources built from a git clone (cli_tldr, cli_nl2bash,
    cli_cheat_sheets) leave that clone at `<source>/_repo/`, and it is full of
    `.json`/`.txt` files (package.json, requirements.txt, ...) that are not
    corpus shards. An `rglob("*")` here previously swept them in and
    symlinked them alongside the real shards under the same source label,
    corrupting both the training text and every stat derived from it -- on
    the real corpus this inflated cli_nl2bash's measured line count by 36%.

    Real Stack Exchange-shaped sources cache extracted XML at `<source>/_xml/`
    and raw archives at `<source>/_archives/`; those must be excluded too.
    """
    repo_dir = pretrain_corpus_dir / "cli_tldr" / "_repo" / "tldr"
    repo_dir.mkdir(parents=True)
    (repo_dir / "package.json").write_text('{"name": "tldr"}', encoding="utf-8")
    (repo_dir / "requirements.txt").write_text("pytest>=6.0\n", encoding="utf-8")

    xml_dir = pretrain_corpus_dir / "cli_helper_stack_exchange" / "_xml" / "askubuntu.com"
    xml_dir.mkdir(parents=True)
    (xml_dir / "Posts.xml").write_text("<row Body='not corpus text'/>", encoding="utf-8")

    dest = tmp_path / "flattened"
    counts = experiment.build_flattened_corpus_view(pretrain_corpus_dir, dest)

    assert counts["cli_tldr"] == 1  # only the real .jsonl shard, not the 2 _repo files
    assert counts["cli_helper_stack_exchange"] == 2  # unaffected by the .xml cache
    for link in dest.iterdir():
        assert "_repo" not in link.resolve().parts
        assert "_xml" not in link.resolve().parts


# --------------------------------------------------------------------------
# verify_forced_sources_saturated
# --------------------------------------------------------------------------


def _write_experiment_config(path: Path, shares: dict[str, float]) -> None:
    cfg = yaml.safe_load((MODULE_DIR / "config" / "smoke.yaml").read_text(encoding="utf-8"))
    cfg["blend"] = {"mode": "weighted", "shares": shares}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_verify_forced_sources_saturated_passes_with_generous_shares(tmp_path, pretrain_corpus_dir):
    flattened = tmp_path / "flattened"
    experiment.build_flattened_corpus_view(pretrain_corpus_dir, flattened)
    config_path = tmp_path / "config.yaml"
    _write_experiment_config(config_path, {
        "fineweb_edu": 0.30, "c4": 0.20, "cli_helper_stack_exchange": 0.10,
        "cli_tldr": 0.10, "cli_nl2bash": 0.10, "cli_man_pages": 0.10, "cli_cheat_sheets": 0.10,
    })

    checks = experiment.verify_forced_sources_saturated(
        flattened, config_path, seed=42, smallest_size=2000, scratch_dir=tmp_path / "scratch",
    )

    assert {c.source for c in checks} == set(experiment.FORCED_SOURCES)
    assert all(c.saturated for c in checks), checks


def test_verify_forced_sources_saturated_catches_an_under_weighted_config(tmp_path, pretrain_corpus_dir):
    """
    The whole point of this function: a config that gives the tiny CLI
    sources a negligible share must be caught, not silently accepted.
    """
    flattened = tmp_path / "flattened"
    experiment.build_flattened_corpus_view(pretrain_corpus_dir, flattened)
    config_path = tmp_path / "config.yaml"
    _write_experiment_config(config_path, {
        "fineweb_edu": 0.485, "c4": 0.485, "cli_helper_stack_exchange": 0.02,
        "cli_tldr": 0.0025, "cli_nl2bash": 0.0025, "cli_man_pages": 0.0025, "cli_cheat_sheets": 0.0025,
    })

    checks = experiment.verify_forced_sources_saturated(
        flattened, config_path, seed=42, smallest_size=2000, scratch_dir=tmp_path / "scratch",
    )

    assert any(not c.saturated for c in checks), checks


# --------------------------------------------------------------------------
# size_label
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size,expected", [
    (250_000, "250K"), (1_000_000, "1M"), (16_000_000, "16M"), (123_456, "123456"),
])
def test_size_label(size, expected):
    assert experiment.size_label(size) == expected


# --------------------------------------------------------------------------
# build_variant_jobs
# --------------------------------------------------------------------------


def test_build_variant_jobs_one_per_size(tmp_path):
    base_config = tmp_path / "base.yaml"
    _write_experiment_config(base_config, {"fineweb_edu": 1.0})

    jobs = experiment.build_variant_jobs(
        [250_000, 1_000_000], tmp_path / "corpus", base_config, tmp_path / "variants",
        seed=42, eval_size=5_000_000, num_threads=3,
    )

    assert [j.label for j in jobs] == ["250K", "1M"]
    job = jobs[0]
    assert "--input-sentence-size" in job.argv and "250000" in job.argv
    assert "--eval-corpus" in job.argv and "5000000" in job.argv
    assert "--seed" in job.argv and "42" in job.argv
    written_cfg = yaml.safe_load(job.config_path.read_text(encoding="utf-8"))
    assert written_cfg["trainer"]["num_threads"] == 3
    # blend must be untouched from the base config
    assert written_cfg["blend"]["shares"] == {"fineweb_edu": 1.0}


def test_build_variant_jobs_configs_are_independent(tmp_path):
    """Each variant's config.yaml must be its own file, not a shared mutable dict."""
    base_config = tmp_path / "base.yaml"
    _write_experiment_config(base_config, {"fineweb_edu": 1.0})

    jobs = experiment.build_variant_jobs(
        [250_000, 500_000], tmp_path / "corpus", base_config, tmp_path / "variants",
        seed=42, eval_size=1000, num_threads=2,
    )
    assert jobs[0].config_path != jobs[1].config_path
    assert jobs[0].output_dir != jobs[1].output_dir


# --------------------------------------------------------------------------
# load_variant_metrics
# --------------------------------------------------------------------------


def _fake_manifest(*, hard_pass: bool, saturated: bool) -> dict:
    forced = list(experiment.FORCED_SOURCES)
    capacities = {"fineweb_edu": 100_000, **{s: 50 for s in forced}}
    trained = dict(capacities)
    if not saturated:
        trained[forced[0]] = 40  # short of its capacity of 50
    return {
        "corpus": {"scan": {"per_source_lines": capacities}, "train_lines_per_source": trained},
        "gates": [
            {"gate": 1, "hard": True, "passed": hard_pass, "metrics": {}},
            {"gate": 4, "hard": False, "passed": True,
             "metrics": {"overall": 1.33, "per_slice": {"cli_helper_stack_exchange": 1.29}}},
            {"gate": 5, "hard": False, "passed": True, "metrics": {"rate": 0.0003}},
            {"gate": 8, "hard": False, "passed": True, "metrics": {"overall": 4.1}},
            {"gate": 10, "hard": False, "passed": True, "metrics": {"utilization": 0.95}},
            {"gate": 11, "hard": False, "passed": True, "metrics": {"rare_pieces": 1800}},
        ],
    }


def _fake_run(tmp_path: Path, *, label: str, hard_pass: bool, saturated: bool, returncode: int = 0) -> experiment.VariantRun:
    output_dir = tmp_path / label / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if returncode == 0:
        (output_dir / "manifest.json").write_text(
            json.dumps(_fake_manifest(hard_pass=hard_pass, saturated=saturated)), encoding="utf-8"
        )
    job = experiment.VariantJob(
        label=label, size=1_000_000, output_dir=output_dir,
        config_path=tmp_path / "config.yaml", argv=[],
    )
    return experiment.VariantRun(
        job=job, returncode=returncode, elapsed_seconds=12.0, log_path=tmp_path / "log", log_tail="",
    )


def test_load_variant_metrics_happy_path(tmp_path):
    run = _fake_run(tmp_path, label="1M", hard_pass=True, saturated=True)
    m = experiment.load_variant_metrics(run)
    assert m.hard_gates_passed is True
    assert m.forced_sources_saturated is True
    assert m.fertility_overall == 1.33
    assert m.fertility_cli == 1.29
    assert m.byte_fallback_rate == 0.0003
    assert m.chars_per_token == 4.1
    assert m.vocab_utilization == 0.95
    assert m.rare_pieces == 1800
    assert m.notes == ()


def test_load_variant_metrics_flags_unsaturated_forced_source(tmp_path):
    run = _fake_run(tmp_path, label="1M", hard_pass=True, saturated=False)
    m = experiment.load_variant_metrics(run)
    assert m.forced_sources_saturated is False
    assert any("NOT fully included" in n for n in m.notes)


def test_load_variant_metrics_handles_a_failed_run(tmp_path):
    run = _fake_run(tmp_path, label="1M", hard_pass=True, saturated=True, returncode=1)
    m = experiment.load_variant_metrics(run)
    assert m.hard_gates_passed is False
    assert m.forced_sources_saturated is False
    assert m.fertility_overall is None
    assert "run failed" in m.notes[0]


# --------------------------------------------------------------------------
# rank_variants
# --------------------------------------------------------------------------


def _metrics(label, *, hard=True, saturated=True, fert=1.33, fert_cli=1.33, cpt=4.0, util=0.95) -> experiment.VariantMetrics:
    return experiment.VariantMetrics(
        label=label, size=1_000_000, returncode=0, elapsed_seconds=1.0,
        hard_gates_passed=hard, forced_sources_saturated=saturated,
        fertility_overall=fert, fertility_cli=fert_cli, byte_fallback_rate=0.001,
        chars_per_token=cpt, vocab_utilization=util, rare_pieces=100, notes=(),
    )


def test_rank_variants_disqualifies_hard_gate_failures():
    good = _metrics("good", fert=1.33)
    bad = _metrics("bad", hard=False, fert=1.325)  # numerically "better" but must lose
    ranked = experiment.rank_variants([bad, good], (1.25, 1.40))
    labels_with_rank = [label for label, rank in ((m.label, r) for m, r in ranked) if rank is not None]
    assert labels_with_rank == ["good"]


def test_rank_variants_disqualifies_unsaturated_forced_sources():
    good = _metrics("good")
    unsaturated = _metrics("unsaturated", saturated=False)
    ranked = experiment.rank_variants([unsaturated, good], (1.25, 1.40))
    assert dict((m.label, r) for m, r in ranked)["unsaturated"] is None
    assert dict((m.label, r) for m, r in ranked)["good"] is not None


def test_rank_variants_prefers_fertility_closest_to_target_midpoint():
    mid = (1.25 + 1.40) / 2
    on_target = _metrics("on_target", fert=mid, fert_cli=mid, cpt=4.0, util=0.95)
    off_target = _metrics("off_target", fert=mid + 0.5, fert_cli=mid + 0.5, cpt=4.0, util=0.95)
    ranked = experiment.rank_variants([off_target, on_target], (1.25, 1.40))
    assert ranked[0][0].label == "on_target"


def test_rank_variants_returns_all_disqualified_when_no_survivors():
    ranked = experiment.rank_variants(
        [_metrics("a", hard=False), _metrics("b", saturated=False)], (1.25, 1.40),
    )
    assert all(rank is None for _m, rank in ranked)
    assert len(ranked) == 2


# --------------------------------------------------------------------------
# build_comparison_report
# --------------------------------------------------------------------------


def test_build_comparison_report_names_the_winner():
    ranked = experiment.rank_variants([_metrics("1M")], (1.25, 1.40))
    report = experiment.build_comparison_report(
        ranked, (1.25, 1.40), {"timestamp": "now", "seed": 42, "eval_size": 50_000_000},
    )
    assert "**1M**" in report
    assert "## Recommendation" in report


def test_build_comparison_report_handles_no_survivors():
    ranked = experiment.rank_variants([_metrics("1M", hard=False)], (1.25, 1.40))
    report = experiment.build_comparison_report(
        ranked, (1.25, 1.40), {"timestamp": "now", "seed": 42, "eval_size": 50_000_000},
    )
    assert "nothing to recommend" in report
