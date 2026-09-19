"""
Tests for the checkpoint hash chain.

Each one stands for a way the record could quietly stop being evidence. That
is the bar used to decide what is here: a test earns its place by catching a
failure that would otherwise be *silent* until the moment the record is needed,
which is the moment it can no longer be fixed.

Deliberately not tested: that SHA-256 is SHA-256, that `json` round-trips, that
a thread runs. Those fail loudly, immediately, and belong to their own
libraries.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.chain import GENESIS, HASH_FILE, read_records, record_checkpoint
from src.checkpoint_hash import CheckpointHashRecorder, backfill, verify
from src.errors import OwnershipConfigError
from src.hashing import hash_directory, hash_file, root_hash
from src.registry import build_observers


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run directory with three checkpoints, shaped like the real ones."""
    for step in (1000, 2000, 3000):
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"weights-%d" % step)
        (directory / "config.json").write_text('{"model_type": "t5"}')
    return tmp_path


# -- the record has to be reproducible without this code --------------------


def test_root_hash_matches_what_sha256sum_would_produce(run_dir: Path) -> None:
    """
    A proof only its author's code can check is not much of a proof.

    `root_hash` is specified as a digest over `sha256sum`-shaped lines, so
    anyone holding a published record can verify it with coreutils. This pins
    that promise: if the line format is ever "tidied", every record already
    published becomes unverifiable by the documented method, and nothing else
    in the suite would notice.
    """
    directory = run_dir / "checkpoint-1000"
    listing = subprocess.run(
        ["sha256sum", "config.json", "model.safetensors"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    by_hand = subprocess.run(
        ["sha256sum"], input=listing, capture_output=True, text=True, check=True
    ).stdout.split()[0]

    assert root_hash(hash_directory(directory)) == by_hand


def test_file_hash_matches_sha256sum(run_dir: Path) -> None:
    """The per-file hashes are plain SHA-256 of the bytes, nothing bespoke."""
    weights = run_dir / "checkpoint-1000" / "model.safetensors"
    expected = subprocess.run(
        ["sha256sum", str(weights)], capture_output=True, text=True, check=True
    ).stdout.split()[0]
    assert hash_file(weights) == expected


# -- the chain has to stay a chain ------------------------------------------


def test_records_chain_to_their_predecessor(run_dir: Path) -> None:
    """
    `prev_sha256` is the whole reason this is evidence of *order* rather than
    a bag of unrelated hashes. A change that wrote records independently would
    still produce a plausible-looking file, and the loss would only be
    discovered by someone disputing the history.
    """
    path = run_dir / HASH_FILE
    for step in (1000, 2000, 3000):
        record_checkpoint(run_dir / f"checkpoint-{step}", path, step)

    records = read_records(path)
    assert [r["step"] for r in records] == [1000, 2000, 3000]
    assert records[0]["prev_sha256"] == GENESIS
    assert records[1]["prev_sha256"] == records[0]["checkpoint_sha256"]
    assert records[2]["prev_sha256"] == records[1]["checkpoint_sha256"]
    assert verify(run_dir)["chain_intact"]


def test_recording_a_step_twice_does_not_duplicate_it(run_dir: Path) -> None:
    """
    Two lines for one checkpoint make every `prev_sha256` after them wrong.

    This guard is what lets `backfill` be run at any time — including while a
    run is live and hashing the same directory — without being able to corrupt
    the chain it is repairing.
    """
    path = run_dir / HASH_FILE
    first = record_checkpoint(run_dir / "checkpoint-1000", path, 1000)
    again = record_checkpoint(run_dir / "checkpoint-1000", path, 1000)

    assert again == first
    assert len(read_records(path)) == 1


def test_verify_separates_tampering_from_ordinary_deletion(run_dir: Path) -> None:
    """
    Retention deletes old checkpoints by design, so "the file is gone" must
    never read as "the file was altered". Conflating them would make `verify`
    cry wolf on every healthy run and train its reader to ignore it.
    """
    path = run_dir / HASH_FILE
    for step in (1000, 2000, 3000):
        record_checkpoint(run_dir / f"checkpoint-{step}", path, step)

    (run_dir / "checkpoint-2000" / "model.safetensors").write_bytes(b"altered")
    for leftover in (run_dir / "checkpoint-1000").iterdir():
        leftover.unlink()
    (run_dir / "checkpoint-1000").rmdir()

    report = verify(run_dir)
    assert report["mismatches"] == [2000]
    assert report["absent_from_disk"] == [1000]
    assert report["verified_on_disk"] == [3000]
    assert report["chain_intact"]


def test_verify_reports_an_edited_chain(run_dir: Path) -> None:
    """An edited history must be detectable; that is the point of the chain."""
    path = run_dir / HASH_FILE
    for step in (1000, 2000, 3000):
        record_checkpoint(run_dir / f"checkpoint-{step}", path, step)

    lines = path.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["prev_sha256"] = "0" * 63 + "1"
    lines[1] = json.dumps(tampered, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    report = verify(run_dir)
    assert not report["chain_intact"]
    assert report["chain_breaks"] == [2000]


def test_a_torn_line_does_not_destroy_the_history(run_dir: Path) -> None:
    """
    One line lost to a crash must cost that line and nothing else — which is
    the only reason this file is JSON Lines rather than one JSON document.
    """
    path = run_dir / HASH_FILE
    for step in (1000, 2000):
        record_checkpoint(run_dir / f"checkpoint-{step}", path, step)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"step": 3000, "checkpo')  # killed mid-write

    assert [record["step"] for record in read_records(path)] == [1000, 2000]


# -- the run must survive whatever this module does -------------------------


def test_a_failing_checkpoint_does_not_stop_the_worker(tmp_path: Path) -> None:
    """
    The worker handles one bad item and keeps going.

    A worker that dies on its first failure would stop recording for the rest
    of a multi-week run, and the only symptom would be a hash file that quietly
    stopped growing — exactly the kind of silence this package must not have.
    """
    good = tmp_path / "checkpoint-2000"
    good.mkdir()
    (good / "model.safetensors").write_bytes(b"real")

    events: list[dict] = []
    recorder = CheckpointHashRecorder(tmp_path / HASH_FILE, on_log=events.append)
    recorder.on_checkpoint_saved(tmp_path / "checkpoint-1000", 1000)  # does not exist
    recorder.on_checkpoint_saved(good, 2000)
    recorder.close(timeout=30)

    assert [record["step"] for record in read_records(tmp_path / HASH_FILE)] == [2000]
    assert any("ownership_error" in event for event in events)
    assert any("checkpoint_hashed" in event for event in events)


def test_an_unclosed_recorder_does_not_hang_interpreter_exit(tmp_path: Path) -> None:
    """
    The worker must never keep a dead run alive.

    A non-daemon worker blocked on an empty queue hangs `python` forever, and
    CPython joins non-daemon threads *before* `atexit` runs, so the obvious
    rescue does not work. This ran as a real hang before the worker was made a
    daemon. The symptom — a finished run whose process never exits — is one an
    operator reads as "still training", so it is worth a subprocess to pin.
    """
    directory = tmp_path / "checkpoint-1000"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"x")

    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from src.checkpoint_hash import CheckpointHashRecorder
recorder = CheckpointHashRecorder({str(tmp_path / HASH_FILE)!r})
recorder.on_checkpoint_saved({str(directory)!r}, 1000)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], timeout=60, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_backfill_recovers_checkpoints_that_were_never_hashed(run_dir: Path) -> None:
    """
    The repair path for a crash between a save and its hash — and the reason
    losing an in-flight record to a killed daemon thread is acceptable.
    """
    path = run_dir / HASH_FILE
    record_checkpoint(run_dir / "checkpoint-1000", path, 1000)

    assert backfill(run_dir)["added"] == [2000, 3000]
    assert verify(run_dir)["chain_intact"]
    assert backfill(run_dir)["added"] == []


# -- switching techniques on and off ----------------------------------------


def test_an_unknown_technique_name_is_refused(tmp_path: Path) -> None:
    """
    A silently ignored typo means a run recording nothing while its operator
    believes otherwise — discovered when the record is needed and absent.
    """
    with pytest.raises(OwnershipConfigError, match="tigger_fingerprint"):
        build_observers(
            tmp_path, config={"techniques": {"tigger_fingerprint": {"enabled": True}}}
        )


def test_techniques_are_off_unless_enabled(tmp_path: Path) -> None:
    assert build_observers(tmp_path, config={}) == []
    assert (
        build_observers(
            tmp_path, config={"techniques": {"checkpoint_hash": {"enabled": False}}}
        )
        == []
    )


def test_an_enabled_technique_is_built_and_records(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint-1000"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"weights")

    observers = build_observers(
        tmp_path, config={"techniques": {"checkpoint_hash": {"enabled": True}}}
    )
    assert len(observers) == 1
    try:
        observers[0].on_checkpoint_saved(directory, 1000)
    finally:
        observers[0].close(timeout=30)

    assert [r["step"] for r in read_records(tmp_path / HASH_FILE)] == [1000]


def test_the_config_path_can_be_overridden_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    `OWNERSHIP_CONFIG` is what keeps the integration suite portable.

    Those tests drive the real `pretrain.main`, so without an override they
    read the operator's production config — and with the trigger fingerprint
    enabled there, every one of them would need a gitignored secrets file that
    exists on a single machine. The suite would pass here and fail everywhere
    else, including the moment that file is moved out of the repo as intended.
    """
    from src.registry import CONFIG_ENV_VAR, load_config

    config = tmp_path / "elsewhere.yml"
    config.write_text("techniques: {}\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config))

    assert load_config() == {"techniques": {}}
    assert build_observers(tmp_path) == []
