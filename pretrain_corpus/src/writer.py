"""
Writing cleaned chunks to disk under a byte budget, with a provenance manifest.

Successor to the tokenizer downloader's `BudgetWriter`, with the same
contract and three changes for pretraining scale.

**Budget is in GB, and it is measured on output bytes.** The argument the
operator passes is the size of the files this writes, not the size of the
source data it streamed to get there — which is the only number anyone can
plan disk around. Token counts stay out of the download path entirely:
counting them would mean running the tokenizer's `encode()` inside the hot
loop, competing for the same cores as the download itself, to produce a
number that a bytes-per-token constant gives for free.

**Rotation.** A single 100GB JSONL file is awkward to move, impossible to
inspect, and unrecoverable if the run dies mid-write. Output rotates every
`shard_bytes` into `<source>-00000.jsonl`, which is also exactly the shape
`corpus/`'s `discover_corpus_files` already globs for.

**Line-buffered append with fsync at rotation.** A multi-day download must
never lose more than the last shard to a power cut.

Output schema is unchanged and is the same one `corpus/src/reader.py`
consumes with no conversion step: one JSON object per line, `{"text": ...}`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

GB = 1024 * 1024 * 1024
DEFAULT_SHARD_BYTES = 2 * GB


@dataclass
class ShardRecord:
    """Provenance for one output file. Hashed so a corpus can be verified later."""

    path: str
    bytes_written: int
    lines: int
    sha256: str


@dataclass
class BudgetWriter:
    """
    Writes `{"text": ...}` lines until `budget_bytes` of output exists.

    `done` is the signal the source loop polls; it is checked between records
    rather than between chunks, so a record's chunks are never split across
    the budget boundary.
    """

    source_id: str
    output_dir: Path
    budget_bytes: int
    shard_bytes: int = DEFAULT_SHARD_BYTES

    bytes_written: int = 0
    lines_written: int = 0
    shards: list[ShardRecord] = field(default_factory=list)

    _fh: object | None = field(default=None, init=False, repr=False)
    _shard_index: int = field(default=0, init=False, repr=False)
    _shard_bytes_written: int = field(default=0, init=False, repr=False)
    _shard_lines: int = field(default=0, init=False, repr=False)
    _shard_hash: object = field(default=None, init=False, repr=False)
    _started: float = field(default_factory=time.time, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def done(self) -> bool:
        return self.bytes_written >= self.budget_bytes

    def _shard_path(self, index: int) -> Path:
        return self.output_dir / f"{self.source_id}-{index:05d}.jsonl"

    def _open_shard(self) -> None:
        self._fh = open(self._shard_path(self._shard_index), "w", encoding="utf-8")
        self._shard_bytes_written = 0
        self._shard_lines = 0
        self._shard_hash = hashlib.sha256()

    def _close_shard(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self.shards.append(
            ShardRecord(
                path=str(self._shard_path(self._shard_index)),
                bytes_written=self._shard_bytes_written,
                lines=self._shard_lines,
                sha256=self._shard_hash.hexdigest(),
            )
        )
        self._fh = None
        self._shard_index += 1

    def write(self, text: str) -> None:
        """Append one cleaned chunk. Rotates when the current shard is full."""
        if self._fh is None:
            self._open_shard()
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")

        self._fh.write(line)
        self._shard_hash.update(encoded)
        self.bytes_written += len(encoded)
        self._shard_bytes_written += len(encoded)
        self.lines_written += 1
        self._shard_lines += 1

        if self._shard_bytes_written >= self.shard_bytes:
            self._close_shard()

    def write_many(self, texts: list[str]) -> None:
        for text in texts:
            self.write(text)

    def close(self) -> dict:
        self._close_shard()
        elapsed = time.time() - self._started
        return {
            "source_id": self.source_id,
            "output_dir": str(self.output_dir),
            "bytes_written": self.bytes_written,
            "gb_written": round(self.bytes_written / GB, 3),
            "budget_gb": round(self.budget_bytes / GB, 3),
            "lines_written": self.lines_written,
            "shards": [vars(s) for s in self.shards],
            "elapsed_seconds": round(elapsed, 1),
            "mb_per_second": round(self.bytes_written / (1024 * 1024) / max(elapsed, 1e-6), 2),
        }

    def progress(self) -> str:
        pct = 100 * self.bytes_written / max(self.budget_bytes, 1)
        return (
            f"{self.bytes_written / GB:6.2f} / {self.budget_bytes / GB:.2f} GB "
            f"({pct:5.1f}%)  {self.lines_written:,} lines"
        )


def write_manifest(path: Path, manifest: dict) -> None:
    """
    Provenance beside the data, atomically.

    `.gitignore` keeps the `.jsonl` out of the repo and the `.manifest.json`
    in it, so what the corpus was built from travels with the code even though
    the corpus itself never can.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def gb_to_bytes(gb: float) -> int:
    if gb <= 0:
        raise ValueError(f"size must be > 0 GB, got {gb}")
    return int(gb * GB)


def estimate_tokens(total_bytes: int, bytes_per_token: float = 4.0) -> int:
    """
    The GB -> tokens conversion, applied once for reporting, never in the loop.

    ~4 bytes per token is the usual figure for English under a 32k subword
    vocabulary. It is an estimate and is labelled as one wherever it is
    printed; the authoritative count comes from `corpus/` at training time.
    """
    return int(total_bytes / max(bytes_per_token, 0.1))
