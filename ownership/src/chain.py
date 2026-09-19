"""
The hash chain: one JSON line per checkpoint, appended in step order.

What a line looks like
----------------------
    {"step": 8000,
     "checkpoint": "checkpoint-8000",
     "algorithm": "sha256",
     "checkpoint_sha256": "<root over the files below>",
     "prev_sha256": "<the previous line's checkpoint_sha256>",
     "files": {"config.json": "...", "model.safetensors": "...", ...},
     "bytes": 9126805504,
     "recorded_utc": "2026-09-19T14:32:07Z"}

A few hundred bytes of text that reveals nothing about the weights, which is
what makes the file safe to publish while the run is still going.

Why the lines are chained
-------------------------
`checkpoint_sha256` alone proves "this file existed". `prev_sha256` proves
something the individual hashes cannot: *the order, and that nothing was
inserted or removed later*. Forging one line means rewriting every line after
it — and once those later lines are public, they cannot be rewritten. The chain
is what turns a list of hashes into a history.

Why JSON Lines rather than one JSON document
--------------------------------------------
Appending must be a single `write` of a self-contained line. Rewriting a whole
document on every checkpoint risks losing the entire history to one interrupted
write, which is the opposite of what a tamper-evidence file is for. A line torn
by a crash costs that line and nothing else — `read_records` skips it and the
break is then visible in `verify`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .hashing import ALGORITHM, directory_bytes, hash_directory, root_hash

HASH_FILE = "checkpoint_hashes.jsonl"
RECORD_VERSION = 1

# The first line has no predecessor. Zeros are the conventional genesis value
# and are unmistakably not a real digest.
GENESIS = "0" * 64


# -- writing ---------------------------------------------------------------


def build_record(
    directory: str | Path, step: int, previous: str | None = None
) -> dict[str, Any]:
    """Hash a checkpoint directory and shape it into a chain record."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")

    files = hash_directory(directory)
    if not files:
        raise ValueError(f"checkpoint directory is empty: {directory}")

    return {
        "version": RECORD_VERSION,
        "step": int(step),
        "checkpoint": directory.name,
        "algorithm": ALGORITHM,
        "checkpoint_sha256": root_hash(files),
        "prev_sha256": GENESIS if previous is None else previous,
        "files": files,
        "bytes": directory_bytes(directory),
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def append_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """
    Append one complete line, atomically, and flush it to the platter.

    Two deliberate choices, for two different failure modes:

    `O_APPEND` + a single `os.write` of the whole line, rather than a buffered
    text write. POSIX makes an append-mode write atomic with respect to its
    offset, so a line cannot interleave with another writer and — since the
    writer is killable at interpreter exit — cannot be left half-written.
    Either the whole record lands or none of it does, and a missing record is
    repairable by `backfill` while half a record is not.

    `fsync`, which matters here and nowhere else in this package: the value of
    a record is that it existed *before* whatever is later disputed, and a line
    still sitting in the page cache when the machine loses power was never
    written at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record_checkpoint(
    directory: str | Path, path: str | Path, step: int
) -> dict[str, Any]:
    """
    Hash a checkpoint and append it to the chain. Returns the record written.

    A step already present is returned unchanged rather than written twice.
    Two lines for one checkpoint would make every `prev_sha256` after them
    wrong, so this is the guard that lets `backfill` run at any time, including
    while a run is live, without being able to corrupt the chain.
    """
    path = Path(path)
    records = read_records(path)

    for record in records:
        if record.get("step") == int(step):
            return record

    previous = records[-1]["checkpoint_sha256"] if records else None
    record = build_record(directory, step, previous)
    append_record(path, record)
    return record


# -- reading ---------------------------------------------------------------


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Every record in the file, in written order.

    An unreadable line is skipped, not fatal: one line torn by a power loss
    must not make the surviving history unreadable. `verify` reports the
    resulting gap.
    """
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def last_record(path: str | Path) -> dict[str, Any] | None:
    """The most recent record, or `None` for a fresh chain."""
    records = read_records(path)
    return records[-1] if records else None


__all__ = [
    "GENESIS",
    "HASH_FILE",
    "RECORD_VERSION",
    "append_record",
    "build_record",
    "last_record",
    "read_records",
    "record_checkpoint",
]
