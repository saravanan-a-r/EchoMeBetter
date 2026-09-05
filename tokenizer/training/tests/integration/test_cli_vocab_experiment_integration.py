"""
End-to-end smoke test for run_cli_vocab_experiment.py: real subprocess
calls to the real train_tokenization.py, real (tiny) SentencePiece
training, real acceptance gates -- on a synthetic multi-source corpus small
enough to run in seconds.

This is the test that would catch a real regression in the pieces unit
tests can't reach: the symlink-naming trick actually being accepted by the
real corpus_reader.py pipeline end to end, the subprocess argv actually
being a valid train_tokenization.py invocation, and manifest.json's real
schema actually matching what load_variant_metrics() expects.
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

import run_cli_vocab_experiment as experiment  # noqa: E402
from tests.corpus_factory import _make_record  # noqa: E402

pytestmark = pytest.mark.slow


def _write_source(root: Path, source_id: str, n_lines: int, *, seed: int) -> None:
    source_dir = root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with open(source_dir / f"{source_id}-w00-00000.jsonl", "w", encoding="utf-8") as fh:
        for _ in range(n_lines):
            fh.write(json.dumps({"text": _make_record(rng)}, ensure_ascii=False) + "\n")


@pytest.fixture
def pretrain_corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "pretrain_output"
    # Same order of magnitude as conftest.py's own synthetic_corpus fixture
    # (15000 records), which is what smoke.yaml's 2048-piece vocab is
    # already proven to train against successfully.
    _write_source(root, "fineweb_edu", 8000, seed=1)
    _write_source(root, "c4", 4000, seed=2)
    _write_source(root, "cli_helper_stack_exchange", 3000, seed=3)
    # Deliberately tiny, mirroring the real webnye capacity ratios.
    _write_source(root, "cli_tldr", 80, seed=4)
    _write_source(root, "cli_nl2bash", 60, seed=5)
    _write_source(root, "cli_man_pages", 90, seed=6)
    _write_source(root, "cli_cheat_sheets", 20, seed=7)
    return root


@pytest.fixture
def smoke_experiment_config(tmp_path: Path) -> Path:
    """smoke.yaml's fast, tiny-vocab settings, with a blend matching the
    synthetic sources above instead of the real 12-source production mix."""
    import yaml

    cfg = yaml.safe_load((MODULE_DIR / "config" / "smoke.yaml").read_text(encoding="utf-8"))
    cfg["blend"] = {
        "mode": "weighted",
        "shares": {
            "fineweb_edu": 0.35, "c4": 0.20, "cli_helper_stack_exchange": 0.05,
            "cli_tldr": 0.10, "cli_nl2bash": 0.10, "cli_man_pages": 0.10, "cli_cheat_sheets": 0.10,
        },
    }
    path = tmp_path / "smoke_experiment.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_full_sweep_end_to_end(tmp_path, pretrain_corpus_dir, smoke_experiment_config):
    workdir = tmp_path / "workdir"
    exit_code = experiment.main([
        "--pretrain-corpus-dir", str(pretrain_corpus_dir),
        "--workdir", str(workdir),
        "--base-config", str(smoke_experiment_config),
        "--sizes", "2000,4000",
        "--eval-size", "2000",
        "--max-parallel", "2",
    ])

    assert exit_code == 0, "sweep should have produced a winning variant"

    report_path = workdir / "CLI_VOCAB_EXPERIMENT_REPORT.md"
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "## Recommendation" in report
    assert "2K" in report and "4K" in report
    assert "disqualified" not in report or "NOT fully included" not in report

    for label in ("variant_2K", "variant_4K"):
        output_dir = workdir / "variants" / label / "output"
        for name in ("manifest.json", "report.md", "spm.model", "spm.vocab"):
            assert (output_dir / name).is_file(), f"{label}: missing {name}"

        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        trained = manifest["corpus"]["train_lines_per_source"]
        capacities = manifest["corpus"]["scan"]["per_source_lines"]
        for source in experiment.FORCED_SOURCES:
            assert trained.get(source, 0) == capacities.get(source, 0), (
                f"{label}: {source} was not fully included "
                f"({trained.get(source, 0)}/{capacities.get(source, 0)})"
            )


def test_saturation_check_blocks_before_any_training(tmp_path, pretrain_corpus_dir):
    """An under-weighted config must refuse to launch the sweep at all --
    no variant subprocess should even start."""
    import yaml

    cfg = yaml.safe_load((MODULE_DIR / "config" / "smoke.yaml").read_text(encoding="utf-8"))
    cfg["blend"] = {
        "mode": "weighted",
        "shares": {
            "fineweb_edu": 0.5, "c4": 0.49,
            "cli_tldr": 0.0025, "cli_nl2bash": 0.0025,
            "cli_man_pages": 0.0025, "cli_cheat_sheets": 0.0025,
        },
    }
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    workdir = tmp_path / "workdir"
    exit_code = experiment.main([
        "--pretrain-corpus-dir", str(pretrain_corpus_dir),
        "--workdir", str(workdir),
        "--base-config", str(config_path),
        "--sizes", "2000,4000",
        "--eval-size", "2000",
    ])

    assert exit_code == 2
    assert not (workdir / "variants").exists(), "no variant should have been launched"
