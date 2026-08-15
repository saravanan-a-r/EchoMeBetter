"""
Integration tests for the freeze protocol (DESIGN.md 10).

Freezing is the point of no return: every embedding row of every model
trained afterwards is bound to these token IDs. The tests below encode the
one rule that matters -- freeze.py must refuse anything it cannot prove is
valid, and must never freeze a build whose hard gates failed.
"""

from __future__ import annotations

import json
import shutil

import pytest

import freeze


@pytest.fixture
def build_dir(tmp_path, output_dir):
    """A fresh copy of the good build, so tests can mutate it freely."""
    target = tmp_path / "build"
    shutil.copytree(output_dir, target)
    (target / "FROZEN").unlink(missing_ok=True)
    return target


def read_manifest(build_dir):
    return json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(build_dir, manifest):
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_freezes_a_valid_build(build_dir):
    assert freeze.main(["--output-dir", str(build_dir)]) == 0
    marker = json.loads((build_dir / "FROZEN").read_text(encoding="utf-8"))
    assert marker["vocab_size"] == 2048
    assert len(marker["model_sha256"]) == 64
    assert len(marker["corpus_sha256"]) == 64
    assert marker["frozen_at"]
    assert marker["hard_gates_passed"] == [1, 2, 3, 6]


def test_refuses_to_refreeze(build_dir):
    """A frozen build is immutable: re-freezing must be an explicit override."""
    assert freeze.main(["--output-dir", str(build_dir)]) == 0
    assert freeze.main(["--output-dir", str(build_dir)]) == 1


def test_force_allows_refreeze(build_dir):
    assert freeze.main(["--output-dir", str(build_dir)]) == 0
    assert freeze.main(["--output-dir", str(build_dir), "--force"]) == 0


@pytest.mark.parametrize(
    "artifact", ["spm.model", "spm.vocab", "token_map.json", "manifest.json", "report.md"]
)
def test_refuses_when_an_artifact_is_missing(build_dir, artifact):
    (build_dir / artifact).unlink()
    assert freeze.main(["--output-dir", str(build_dir)]) == 1
    assert not (build_dir / "FROZEN").exists()


def test_refuses_a_build_with_failed_hard_gates(build_dir):
    manifest = read_manifest(build_dir)
    for gate in manifest["gates"]:
        if gate["gate"] == 1:
            gate["passed"] = False
    write_manifest(build_dir, manifest)

    assert freeze.main(["--output-dir", str(build_dir)]) == 1
    assert not (build_dir / "FROZEN").exists()


def test_refuses_an_unvalidated_build(build_dir):
    """A build whose gates never ran must not be freezable."""
    manifest = read_manifest(build_dir)
    manifest["gates"] = []
    write_manifest(build_dir, manifest)

    assert freeze.main(["--output-dir", str(build_dir)]) == 1
    assert not (build_dir / "FROZEN").exists()


def test_soft_gate_warnings_do_not_block(build_dir, capsys):
    """Fertility outside the target band is information, not a blocker."""
    manifest = read_manifest(build_dir)
    for gate in manifest["gates"]:
        if gate["gate"] == 4:
            gate["passed"] = False
    write_manifest(build_dir, manifest)

    assert freeze.main(["--output-dir", str(build_dir)]) == 0
    assert "soft gate" in capsys.readouterr().out


def test_marker_records_hashes_that_match_the_artifacts(build_dir):
    from corpus_reader import sha256_file

    assert freeze.main(["--output-dir", str(build_dir)]) == 0
    marker = json.loads((build_dir / "FROZEN").read_text(encoding="utf-8"))
    assert marker["model_sha256"] == sha256_file(build_dir / "spm.model")
    assert marker["token_map_sha256"] == sha256_file(build_dir / "token_map.json")
    assert marker["corpus_sha256"] == read_manifest(build_dir)["corpus"]["corpus_sha256"]
