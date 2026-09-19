"""
Technique 1 of 4 — a dated, chained hash of every checkpoint the run writes.

What it buys
------------
The weights are going to be published openly, and nothing can stop someone
downloading them, fine-tuning them and claiming authorship. What this gives is
the other half of the argument: a public, chained record that *these exact
bytes existed, in this order, on this date*. A later claimant cannot
manufacture that, because they cannot insert entries into a history that was
already published.

The record holds fingerprints, never weights, so it is safe to push to a public
repository while the run is still going — which is exactly when it is worth
most, because the dates accumulate as the run proceeds rather than all being
claimed at the end.

Its one real limitation, stated plainly: a hash is only evidence for as long as
the matching checkpoint still exists. `prune_checkpoints` deletes old
checkpoints by design, and a record whose file is gone proves that *something*
was there, not what. Keep the milestone checkpoints you intend to rely on.

Why hashing never blocks a training step
----------------------------------------
Hashing a full checkpoint is ~7 seconds of streaming I/O against a ~38 second
step, once every 1,000 steps — under 0.02% of wall time even inline. It is
moved off the training thread anyway, because "small" and "zero" are different
promises and this run lasts weeks.

The worker is deliberately a *single* thread. Records must land in step order
for the chain to mean anything, and a pool would interleave them under exactly
the conditions — a slow disk, a delayed save — where the record matters most.
"""

from __future__ import annotations

import os
import queue
import re
import threading
from pathlib import Path
from typing import Any

from .chain import GENESIS, HASH_FILE, read_records, record_checkpoint
from .hashing import hash_directory, root_hash
from .technique import LogFn, log_event

# Checkpoint directories are named `checkpoint-<step>`, HuggingFace's
# convention and `training/src/checkpoint.py`'s. Matched here rather than
# imported from `training/` so this package stays standalone: verifying a
# published chain should not require the training code, or a GPU, or torch.
_DIRECTORY = re.compile(r"^checkpoint-(\d+)$")

# Optional operator control, e.g. CHECKPOINT_HASH_CPUS="22-27". Unset means no
# pinning. An environment variable rather than a config field on purpose: which
# cores are free is a property of the machine a run happens to be on, not of
# the run, and it must not end up baked into a checkpoint's recorded settings.
CPU_ENV_VAR = "CHECKPOINT_HASH_CPUS"


class CheckpointHashRecorder:
    """
    Hashes each checkpoint on a background thread and appends it to the chain.

    Implements `technique.CheckpointObserver`. `on_checkpoint_saved` queues and
    returns; the training step never waits on I/O.
    """

    _SENTINEL = object()

    def __init__(self, hash_path: str | Path, *, on_log: LogFn | None = None) -> None:
        self.hash_path = Path(hash_path)
        self.on_log = on_log
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = threading.Event()
        # A daemon, and the choice is not obvious — the reasoning is worth
        # keeping, because the intuitive alternative deadlocks.
        #
        # A non-daemon worker blocks on an empty queue forever, and CPython
        # joins non-daemon threads *before* running `atexit` handlers, so an
        # `atexit` that sends the stop sentinel never gets the chance: any run
        # that fails before `close()` hangs at exit instead of exiting. That is
        # far worse than the failure a non-daemon thread protects against — an
        # operator staring at a process that will never return.
        #
        # So: daemon, plus `close()` on the normal path (pretrain.py drains it
        # in a `finally`). The cost is that a record in flight when the process
        # dies is lost — and `backfill` recreates it from the checkpoint, which
        # is still on disk. `append_record` makes the write itself atomic, so
        # what is lost is a whole record, never half of one.
        self._thread = threading.Thread(
            target=self._run, name="checkpoint-hasher", daemon=True
        )
        self._thread.start()

    # -- CheckpointObserver ------------------------------------------------

    def on_checkpoint_saved(self, directory: Path, step: int) -> None:
        """Queue a checkpoint for hashing. Returns immediately."""
        if self._closed.is_set():
            log_event(
                self.on_log,
                {"ownership_error": "hasher is closed", "checkpoint": str(directory)},
            )
            return
        self._queue.put((Path(directory), int(step)))

    def close(self, timeout: float | None = 300.0) -> None:
        """Drain the queue and stop the worker. Safe to call more than once."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=timeout)

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        _pin_to_configured_cpus()
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                directory, step = item
                record = record_checkpoint(directory, self.hash_path, step)
                log_event(
                    self.on_log,
                    {
                        "checkpoint_hashed": record["checkpoint"],
                        "step": record["step"],
                        "checkpoint_sha256": record["checkpoint_sha256"],
                    },
                )
            except Exception as exc:  # a failed hash must not stop the worker
                log_event(
                    self.on_log,
                    {"ownership_error": f"{type(exc).__name__}: {exc}", "step": item},
                )
            finally:
                self._queue.task_done()


def _pin_to_configured_cpus() -> None:
    """
    Honour `CHECKPOINT_HASH_CPUS` if the operator set it. Best effort.

    Accepts the `taskset`/`numactl` spelling, e.g. "22-27" or "22,23,24".
    Anything unparseable is ignored: a malformed affinity string must not be
    the reason a checkpoint goes unhashed.
    """
    spec = os.environ.get(CPU_ENV_VAR, "").strip()
    if not spec:
        return
    try:
        cpus: set[int] = set()
        for part in filter(None, (p.strip() for p in spec.split(","))):
            if "-" in part:
                low, high = part.split("-", 1)
                cpus.update(range(int(low), int(high) + 1))
            else:
                cpus.add(int(part))
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
    except (ValueError, OSError):
        return


# -- repair and audit ------------------------------------------------------


def checkpoints_on_disk(root: str | Path) -> dict[int, Path]:
    """Every `checkpoint-<step>` directory under `root`, keyed by step."""
    root = Path(root)
    if not root.is_dir():
        return {}
    found = {}
    for entry in root.iterdir():
        match = _DIRECTORY.match(entry.name)
        if match and entry.is_dir():
            found[int(match.group(1))] = entry
    return dict(sorted(found.items()))


def backfill(root: str | Path, hash_path: str | Path | None = None) -> dict[str, Any]:
    """
    Hash every checkpoint on disk that has no record yet.

    Recovers from a crash between a save and its hash. It cannot recover a
    checkpoint that retention already deleted — those bytes are gone — so the
    report names what is on disk and what was recorded, rather than papering
    over the difference.
    """
    root = Path(root)
    hash_path = Path(hash_path) if hash_path is not None else root / HASH_FILE

    recorded = {record.get("step") for record in read_records(hash_path)}
    on_disk = checkpoints_on_disk(root)

    added = [
        record_checkpoint(on_disk[step], hash_path, step)["step"]
        for step in sorted(set(on_disk) - recorded)
    ]
    return {
        "hash_file": str(hash_path),
        "added": added,
        "already_recorded": sorted(s for s in recorded if s is not None),
        "on_disk": sorted(on_disk),
    }


def verify(root: str | Path, hash_path: str | Path | None = None) -> dict[str, Any]:
    """
    Re-check the chain, and re-hash whatever is still on disk.

    Three separate questions, reported separately because they fail for
    different reasons and have different remedies:

      chain_breaks   a `prev_sha256` that does not match the line above it.
                     The file was edited, reordered, or lost a line.
      mismatches     a checkpoint on disk whose bytes no longer produce its
                     recorded hash. The file changed after it was recorded.
      absent         recorded, but no longer on disk. Expected and fine —
                     retention deletes old checkpoints by design.
    """
    root = Path(root)
    hash_path = Path(hash_path) if hash_path is not None else root / HASH_FILE
    records = read_records(hash_path)

    chain_breaks = []
    expected = GENESIS
    for record in records:
        if record.get("prev_sha256") != expected:
            chain_breaks.append(record.get("step"))
        expected = record.get("checkpoint_sha256", "")

    verified: list[Any] = []
    mismatches: list[Any] = []
    absent: list[Any] = []
    for record in records:
        directory = root / str(record.get("checkpoint", ""))
        if not directory.is_dir():
            absent.append(record.get("step"))
            continue
        try:
            current = root_hash(hash_directory(directory))
        except OSError:
            mismatches.append(record.get("step"))
            continue
        target = verified if current == record.get("checkpoint_sha256") else mismatches
        target.append(record.get("step"))

    return {
        "hash_file": str(hash_path),
        "records": len(records),
        "chain_intact": not chain_breaks,
        "chain_breaks": chain_breaks,
        "verified_on_disk": verified,
        "mismatches": mismatches,
        "absent_from_disk": absent,
    }


__all__ = [
    "CPU_ENV_VAR",
    "CheckpointHashRecorder",
    "backfill",
    "checkpoints_on_disk",
    "verify",
]
