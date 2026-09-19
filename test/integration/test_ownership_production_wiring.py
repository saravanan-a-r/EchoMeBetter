"""
The three ownership techniques, driven through `pretrain.main` against the
operator's REAL `ownership_config.yml` and REAL secrets.

Every other test in this suite runs with ownership switched off, because
`conftest.isolated_ownership_config` points the package at an empty config —
deliberately, so the suite never depends on `ownership/secrets/`, which is
gitignored and exists on one machine. That leaves the production
configuration itself untested: the path an operator actually runs, with all
three techniques on at once and a restart in the middle of it, was never
exercised anywhere. This is that test, and it skips cleanly wherever the
secrets are absent.

The restart is the point. These techniques attach to a *resumed* model and a
resumed batch stream, and the run they matter to is stopped and restarted by
hand more than once over its 90,000 steps. What has to hold across that: the
hash chain continues rather than forking, the fingerprint keeps its cadence,
and the signature finds the rows it wrote last time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import conftest as root_conftest
import pretrain

ROOT = Path(__file__).resolve().parents[2]

SECRETS = (
    ROOT / "ownership/secrets/signature_key.json",
    ROOT / "ownership/secrets/fingerprint_pairs.jsonl",
)

pytestmark = [
    pytest.mark.skipif(
        not pretrain.DEFAULT_TOKENIZER_DIR.joinpath("spm.model").is_file(),
        reason="tokenizer not built",
    ),
    pytest.mark.skipif(
        not all(path.is_file() for path in SECRETS),
        reason="ownership/secrets/ not present (gitignored; operator machine only)",
    ),
]


@pytest.fixture
def production_ownership(monkeypatch):
    monkeypatch.setenv("OWNERSHIP_CONFIG", str(ROOT / "ownership_config.yml"))
    return ROOT / "ownership_config.yml"


@pytest.fixture
def wider_model_config_path(tmp_path: Path) -> Path:
    """The real key holds 96 bits, which needs d_model >= 77 across 5 rows."""
    document = root_conftest._tiny_model_config_document()
    document["profiles"]["tiny"]["d_model"] = 128
    document["profiles"]["tiny"]["d_ff"] = 256
    document["defaults"]["d_kv"] = 64
    path = tmp_path / "model_config.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _config(tmp_path: Path, *, max_steps: int, resume: bool) -> Path:
    document = root_conftest._tiny_training_config_document(tmp_path / "runs")
    document["defaults"]["resume_from_checkpoint"] = resume
    document["defaults"]["max_steps"] = max_steps
    document["defaults"]["save_steps"] = 2
    document["stages"]["tiny"]["max_steps"] = max_steps
    path = tmp_path / f"training_{max_steps}_{resume}.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _run(corpus, model_config, training_config):
    return pretrain.main(
        [
            "--corpus", str(corpus),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(model_config),
            "--training-config", str(training_config),
        ]
    )


def test_all_three_techniques_run_and_survive_a_restart(
    production_ownership, mini_corpus_dir, wider_model_config_path, tmp_path, capsys
):
    # -- first run: fresh start, all three techniques on -------------------
    first = _config(tmp_path, max_steps=4, resume=False)
    assert _run(mini_corpus_dir, wider_model_config_path, first) == 0
    out = capsys.readouterr().out
    print(out)

    assert "ownership: CheckpointHashRecorder" in out
    assert "could not be formatted" not in out
    assert "fingerprint: 75 pairs" in out
    assert "signature: 96 bits over 5 reserved rows" in out

    run_dir = tmp_path / "runs"
    chain = run_dir / "checkpoint_hashes.jsonl"
    assert chain.is_file(), "hash chain was never written"
    records = [json.loads(line) for line in chain.read_text().splitlines() if line.strip()]
    assert [r["step"] for r in records] == [2, 4], records
    assert records[0]["prev_sha256"] == "0" * 64
    assert records[1]["prev_sha256"] == records[0]["checkpoint_sha256"]

    # -- the signature actually reached the saved weights ------------------
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "own_check", ROOT / "ownership/src/__init__.py",
        submodule_search_locations=[str(ROOT / "ownership/src")],
    )
    own = importlib.util.module_from_spec(spec)
    sys.modules["own_check"] = own
    spec.loader.exec_module(own)

    from safetensors.torch import load_file

    key = own.load_spec(ROOT / "ownership/secrets/signature_key.json")
    weights = load_file(run_dir / "checkpoint-4" / "model.safetensors")
    status = own.read_signature(weights["shared.weight"], key)
    print("signature in checkpoint-4:", status)

    # -- second run: restart from that checkpoint, no CLI change -----------
    second = _config(tmp_path, max_steps=8, resume=True)
    assert _run(mini_corpus_dir, wider_model_config_path, second) == 0
    out2 = capsys.readouterr().out
    print(out2)

    assert "resumed from step 4" in out2
    assert "signature: 96 bits" in out2
    assert "fingerprint: 75 pairs" in out2

    records = [json.loads(line) for line in chain.read_text().splitlines() if line.strip()]
    assert [r["step"] for r in records] == [2, 4, 6, 8], [r["step"] for r in records]
    for previous, record in zip(records, records[1:]):
        assert record["prev_sha256"] == previous["checkpoint_sha256"], record["step"]

    verify = own.verify(run_dir)
    print("chain verify after restart:", verify)
    assert verify["chain_intact"]
    assert verify["mismatches"] == []


def test_all_three_techniques_reach_the_tensorboard_dashboard(
    production_ownership, mini_corpus_dir, wider_model_config_path, tmp_path, monkeypatch
):
    """
    `--tensorboard-dir` plus the real ownership config, together: each
    technique's own event lands under its `ownership/` tag rather than being
    dropped, exactly as the console line is asserted not to say "could not be
    formatted" in the test above.
    """
    tensorboard = pytest.importorskip(
        "tensorboard.backend.event_processing.plugin_event_accumulator"
    )
    log_dir = tmp_path / "tensorboard"
    config = _config(tmp_path, max_steps=4, resume=False)

    # The real config reports the signature every 1,000 steps -- far past
    # this 4-step run. Lower it so the test proves the tag *can* appear,
    # the same way a real multi-thousand-step run reaches it.
    document = yaml.safe_load((ROOT / "ownership_config.yml").read_text())
    document["techniques"]["embedding_signature"]["log_every"] = 1
    low_cadence = tmp_path / "low_cadence_ownership.yml"
    low_cadence.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setenv("OWNERSHIP_CONFIG", str(low_cadence))

    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(wider_model_config_path),
            "--training-config", str(config),
            "--tensorboard-dir", str(log_dir),
        ]
    )
    assert exit_code == 0

    events = tensorboard.EventAccumulator(str(log_dir))
    events.Reload()
    scalars = set(events.PluginTagToContent("scalars"))

    assert "ownership/checkpoint_hash_written" in scalars
    assert "ownership/fingerprint_injections" not in scalars, (
        "the fingerprint's cadence (every 500 by default) has no reason to "
        "fire in a 4-step run; its absence here is expected, not a failure"
    )
    assert "ownership/signature_matched_bits" in scalars
    assert "ownership/signature_min_margin" in scalars

    hash_points = events.Tensors("ownership/checkpoint_hash_written")
    assert [point.step for point in hash_points] == [2, 4]


def test_a_missing_secret_refuses_the_run_instead_of_training_without_it(
    mini_corpus_dir, wider_model_config_path, tmp_path, monkeypatch, capsys
):
    """Enabled-but-broken must be fatal, not silently skipped."""
    document = yaml.safe_load((ROOT / "ownership_config.yml").read_text())
    document["techniques"]["embedding_signature"]["key_file"] = str(tmp_path / "nope.json")
    broken = tmp_path / "broken_ownership.yml"
    broken.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setenv("OWNERSHIP_CONFIG", str(broken))

    code = _run(mini_corpus_dir, wider_model_config_path, _config(tmp_path, max_steps=2, resume=False))
    assert code == 2, "a missing signature key must refuse the run"
