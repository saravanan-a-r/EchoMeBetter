"""
Pure hashing. No files written, no threads, no state.

Kept separate from `chain.py` (which stores records) and `checkpoint_hash.py`
(which plugs into a run) so that the part anyone might want to re-run by hand,
years later, on a machine that has none of this code, is the part with no
dependencies.

Reproducing a root hash without this package
--------------------------------------------
`root_hash` is deliberately computed over the exact lines `sha256sum` prints:

    cd checkpoint-8000 && sha256sum $(ls | LC_ALL=C sort) | sha256sum

That is the whole algorithm: hash each file, list them `sha256sum`-style in
sorted filename order, hash that listing. Anyone holding a published record can
verify it with coreutils and no Python at all, which is the point — a proof
that only its own author's code can check is not much of a proof.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

ALGORITHM = "sha256"

# Large enough that `hashlib` releases the GIL per block (it does so well below
# this size), so hashing on a worker thread does not contend with training.
_CHUNK_BYTES = 4 * 1024 * 1024


def hash_file(path: str | Path) -> str:
    """Streaming SHA-256 of one file. Never loads it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_directory(directory: str | Path) -> dict[str, str]:
    """
    Every file under `directory`, keyed by its path relative to it.

    Sorted, so the mapping — and the root hash derived from it — does not
    depend on filesystem ordering or on which machine produced it.
    """
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): hash_file(path)
        for path in sorted(p for p in directory.rglob("*") if p.is_file())
    }


def root_hash(file_hashes: Mapping[str, str]) -> str:
    """
    One digest standing for a whole directory.

    Over `sha256sum`-shaped lines — `<digest><two spaces><name>` — in sorted
    *name* order, which is the order `sha256sum` itself emits when given sorted
    arguments. Byte-for-byte that format, so the equivalent shell command in
    the module docstring really does reproduce this number.
    `test_root_hash_matches_what_sha256sum_would_produce` holds it to that.
    """
    digest = hashlib.sha256()
    for name in sorted(file_hashes):
        digest.update(f"{file_hashes[name]}  {name}\n".encode("utf-8"))
    return digest.hexdigest()


def directory_bytes(directory: str | Path) -> int:
    """Total size of a directory, recorded so a truncated checkpoint is obvious."""
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


__all__ = ["ALGORITHM", "directory_bytes", "hash_directory", "hash_file", "root_hash"]
