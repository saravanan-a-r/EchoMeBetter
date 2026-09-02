"""
The persistent state that makes synthetic-PII uniqueness survive a restart.

`pii_allocator.py` guarantees that *distinct indices* produce distinct fake
identities. This module is what guarantees the indices themselves are never
handed out twice — across workers within a run, across successive runs, and
across the batches this project already cleaned months ago.

Three pieces of state
---------------------
1. **`pii_registry.json`** — the authoritative file. Holds a high-water mark
   per PII kind plus the salt that pins the allocator's permutation. It is
   tiny and O(1): because uniqueness comes from the index, the registry never
   needs to store the values themselves. That is the whole reason this scales
   to a hundred-million-replacement corpus, where the earlier design's flat
   "array of every email ever issued" would have become a multi-gigabyte JSON
   document rewritten on every run.

2. **`data/legacy_reserved.json`** — the 3,946 emails and 1,258 phone numbers
   the earlier StackExchange cleanup already issued. Vendored into this repo
   rather than read from that project's path, so the guarantee travels to the
   training box instead of quietly lapsing there. Checked at render time; a
   hit advances to the next index.

3. **`pii_audit.jsonl`** — append-only, optional, for spot-checking. Records
   the *generated* value and its index, and **never the original**. Writing
   the real email next to its replacement would reconstruct, in a side file,
   precisely the PII the pipeline just removed.

Reservation protocol
--------------------
The master reserves one disjoint range per worker *before* any work starts,
and persists the advanced high-water mark immediately — pessimistically, so
a crash mid-run burns some indices rather than replaying them. With an 809
billion-wide email space, burning a few million indices per crash is free;
re-issuing even one is not.

Workers then never touch this file. All contention is confined to a handful
of reservations at startup, which is why the hot path needs no lock at all.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import PIIRegistryError

REGISTRY_VERSION = 1
PII_KINDS = ("email", "handle", "phone")

DEFAULT_LEGACY_PATH = Path(__file__).resolve().parent.parent / "data" / "legacy_reserved.json"


@dataclass(frozen=True)
class IndexRange:
    """A half-open `[start, stop)` block of indices owned by exactly one worker."""

    kind: str
    start: int
    stop: int

    def __len__(self) -> int:
        return max(0, self.stop - self.start)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"IndexRange({self.kind}, {self.start:,}..{self.stop:,})"


@dataclass(frozen=True)
class Reservation:
    """One worker's slice of every index space, plus the shared salt."""

    salt: int
    ranges: dict[str, IndexRange]

    def range_for(self, kind: str) -> IndexRange:
        try:
            return self.ranges[kind]
        except KeyError:  # pragma: no cover - guarded by PII_KINDS
            raise PIIRegistryError(f"no reserved range for PII kind {kind!r}") from None


def load_legacy_reserved(path: str | Path = DEFAULT_LEGACY_PATH) -> dict[str, frozenset[str]]:
    """
    Values the earlier cleanup batches already used, as lookup sets.

    Missing file is fatal rather than an empty set: silently proceeding
    without the blocklist would mean the run *looks* correct while being able
    to re-issue an address that is already in the Stage-2 data — the exact
    failure this file exists to prevent.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PIIRegistryError(f"could not read legacy reserved values from {path}: {exc}") from exc
    return {
        "email": frozenset(raw.get("emails", ())),
        "phone": frozenset(raw.get("phones", ())),
        "handle": frozenset(),
    }


class PIIRegistry:
    """
    Reads, reserves from, and durably writes back the index high-water marks.

    Only the master process uses this. Reservation is done under an exclusive
    lock on the registry file so two concurrently launched runs cannot hand
    out the same block.
    """

    def __init__(self, path: str | Path, *, salt: int = 0) -> None:
        self.path = Path(path)
        self._default_salt = salt

    # -- state -----------------------------------------------------------
    def _read(self) -> dict:
        if not self.path.exists():
            return {
                "version": REGISTRY_VERSION,
                "note": (
                    "High-water marks for synthetic-PII index allocation. Uniqueness "
                    "comes from the index (see src/pii_allocator.py), so only the "
                    "counters need to persist, never the issued values."
                ),
                "salt": self._default_salt,
                "next_index": {kind: 0 for kind in PII_KINDS},
                "reservations_made": 0,
            }
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PIIRegistryError(f"corrupt PII registry at {self.path}: {exc}") from exc

        if state.get("version") != REGISTRY_VERSION:
            raise PIIRegistryError(
                f"PII registry at {self.path} is version {state.get('version')}, "
                f"this build understands version {REGISTRY_VERSION}"
            )
        for kind in PII_KINDS:
            state.setdefault("next_index", {}).setdefault(kind, 0)
        return state

    def _write_atomic(self, state: dict) -> None:
        """
        Write via a temp file in the same directory plus `os.replace`.

        A half-written registry is worse than no registry: it would either
        fail to parse on the next run (loud, recoverable) or parse with a
        rewound counter (silent, and it re-issues identities). `os.replace`
        is atomic on POSIX, so a reader sees the old file or the new one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".pii_registry.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=1)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- reservation -----------------------------------------------------
    def reserve(self, counts: dict[str, int], *, workers: int = 1) -> list[Reservation]:
        """
        Carve `workers` disjoint blocks out of each kind's index space.

        `counts[kind]` is the block size *per worker*. The high-water mark is
        advanced past every block and flushed to disk before this returns, so
        the indices are spoken for whether or not the run that asked for them
        ever finishes.
        """
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        for kind in counts:
            if kind not in PII_KINDS:
                raise PIIRegistryError(f"unknown PII kind {kind!r}; expected one of {PII_KINDS}")

        with _exclusive_lock(self.path):
            state = self._read()
            salt = int(state.get("salt", self._default_salt))
            reservations: list[Reservation] = []

            for worker in range(workers):
                ranges: dict[str, IndexRange] = {}
                for kind in PII_KINDS:
                    size = int(counts.get(kind, 0))
                    start = int(state["next_index"][kind]) + worker * size
                    ranges[kind] = IndexRange(kind, start, start + size)
                reservations.append(Reservation(salt=salt, ranges=ranges))

            for kind in PII_KINDS:
                state["next_index"][kind] = int(state["next_index"][kind]) + workers * int(
                    counts.get(kind, 0)
                )
            state["reservations_made"] = int(state.get("reservations_made", 0)) + workers
            state["salt"] = salt
            self._write_atomic(state)

        return reservations

    def snapshot(self) -> dict:
        """Current counters, for reporting into a run manifest."""
        state = self._read()
        return {
            "salt": state.get("salt", self._default_salt),
            "next_index": dict(state["next_index"]),
            "reservations_made": state.get("reservations_made", 0),
        }


class AuditLog:
    """
    Append-only record of issued values, for eyeballing realism after a run.

    Deliberately one-sided: the fake value and its index go in, the original
    never does. Opened in append mode per worker — POSIX guarantees appends
    below `PIPE_BUF` are atomic, and every line here is far below it, so
    workers can share one file without a lock.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._fh = None

    def __enter__(self) -> AuditLog:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def record(self, kind: str, index: int, value: str) -> None:
        if self._fh is None:
            return
        self._fh.write(
            json.dumps({"kind": kind, "index": index, "value": value}, ensure_ascii=False) + "\n"
        )


class _exclusive_lock:
    """
    Cross-process advisory lock on a sidecar of the registry path.

    A sidecar rather than the registry itself because `_write_atomic`
    replaces the registry inode, which would drop a lock held on it.
    """

    def __init__(self, path: Path) -> None:
        self._lock_path = Path(str(path) + ".lock")
        self._fh = None

    def __enter__(self):
        import fcntl

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._lock_path, "w")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        import fcntl

        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
