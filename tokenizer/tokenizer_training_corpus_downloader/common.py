"""
Shared streaming/download harness used by every per-dataset script in this
folder (download_fineweb_edu.py, download_c4.py, download_stackexchange.py,
download_cosmopedia.py, download_gutenberg.py).

Design goals, all in service of tokenizer/DESIGN.md:

  - Stream from HuggingFace `datasets` in streaming=True mode and stop the
    moment the requested MB budget is hit. We never download a full
    dataset just to keep a slice of it -- these corpora are TB-scale.
  - Every kept record passes through transform.clean_record() before
    being written. No script writes raw dataset text directly.
  - Output is one JSON object per line: {"text": "..."}. Same shape as
    the existing DATASET/*.jsonl files, so downstream sample.py can treat
    every source uniformly.
  - Deterministic: a fixed random seed controls any shuffling, and a
    manifest.json is written per run recording source, size, record
    count, and sha256 of the output file (tokenizer/DESIGN.md 7 --
    reproducibility).
  - Budget is measured on OUTPUT bytes (post-transform), not input bytes,
    because the tokenizer trainer's corpus-size math (DESIGN.md 4.3)
    is about the ~8GB training sample, not raw source size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

from transform import DedupFilter, clean_record

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def build_arg_parser(dataset_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Stream and download up to N MB of {dataset_name} for tokenizer training, "
        f"transformed into tokenizer-ready one-record-per-line JSONL."
    )
    p.add_argument(
        "size_mb",
        type=float,
        help="Target output size in MB (post-transform, i.e. the size of the file this script writes).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write into (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle/sample seed (default: 42, matches DESIGN.md 7).")
    p.add_argument(
        "--strip-markup",
        action="store_true",
        help="Run HTML/XML tag stripping on each record before further cleaning.",
    )
    return p


class BudgetWriter:
    """
    Writes cleaned, chunked records to a JSONL file until `budget_bytes`
    of output has been written, then signals the caller to stop pulling
    from the source iterator. Also runs the corpus-level dedup filter
    (transform.DedupFilter) so repeated boilerplate across records
    doesn't inflate vocab-fitting frequency (DESIGN.md risk R6).
    """

    def __init__(self, out_path: Path, budget_bytes: int) -> None:
        self.out_path = out_path
        self.budget_bytes = budget_bytes
        self.bytes_written = 0
        self.records_written = 0
        self.records_seen = 0
        self.records_deduped = 0
        self.records_empty_after_clean = 0
        self._dedup = DedupFilter()
        self._fh = open(out_path, "w", encoding="utf-8")

    @property
    def done(self) -> bool:
        return self.bytes_written >= self.budget_bytes

    def write_raw_text(self, raw_text: str, *, strip_markup: bool) -> None:
        """Runs the full transform pipeline on one source record and writes 0+ lines."""
        self.records_seen += 1
        chunks = clean_record(raw_text, strip_markup=strip_markup)
        if not chunks:
            self.records_empty_after_clean += 1
            return
        for chunk in chunks:
            if self.done:
                return
            if not self._dedup.accept(chunk):
                self.records_deduped += 1
                continue
            line = json.dumps({"text": chunk}, ensure_ascii=False) + "\n"
            encoded = line.encode("utf-8")
            self._fh.write(line)
            self.bytes_written += len(encoded)
            self.records_written += 1

    def mb_written_str(self) -> str:
        return f"{self.bytes_written / (1024 * 1024):.1f}"

    def close(self) -> dict:
        self._fh.close()
        sha256 = hashlib.sha256(self.out_path.read_bytes()).hexdigest()
        return {
            "output_file": str(self.out_path),
            "bytes_written": self.bytes_written,
            "mb_written": round(self.bytes_written / (1024 * 1024), 3),
            "records_written": self.records_written,
            "records_seen_from_source": self.records_seen,
            "records_dropped_empty_after_clean": self.records_empty_after_clean,
            "records_dropped_dedup": self.records_deduped,
            "sha256": sha256,
        }


def run_download(
    *,
    dataset_name: str,
    source_id: str,
    record_iterator_factory: Callable[[int], Iterator[str]],
    manifest_extra: dict | None = None,
) -> None:
    """
    Common driver: parse args, stream records from `record_iterator_factory`
    (which yields raw text strings, already given the seed), write via
    BudgetWriter, emit manifest.json.

    record_iterator_factory(seed) -> Iterator[str] lets each per-dataset
    script own its HF `load_dataset(..., streaming=True)` call and field
    extraction, while this module owns everything about budgeting,
    transforming, and manifest-writing -- so that logic is defined once.
    """
    parser = build_arg_parser(dataset_name)
    args = parser.parse_args()

    if args.size_mb <= 0:
        print(f"error: size_mb must be > 0, got {args.size_mb}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{source_id}_{int(args.size_mb)}mb.jsonl"
    budget_bytes = int(args.size_mb * 1024 * 1024)

    print(f"[{source_id}] target: {args.size_mb} MB -> {out_path}")
    writer = BudgetWriter(out_path, budget_bytes)
    t0 = time.time()

    try:
        for raw_text in record_iterator_factory(args.seed):
            if writer.done:
                break
            writer.write_raw_text(raw_text, strip_markup=args.strip_markup)
            if writer.records_seen % 5000 == 0:
                pct = 100 * writer.bytes_written / budget_bytes
                print(
                    f"[{source_id}] {writer.mb_written_str()} / {args.size_mb} MB "
                    f"({pct:.1f}%) | records written={writer.records_written} seen={writer.records_seen}",
                    end="\r",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print(f"\n[{source_id}] interrupted by user, finalizing partial output...", file=sys.stderr)

    stats = writer.close()
    elapsed = time.time() - t0

    manifest = {
        "dataset_name": dataset_name,
        "source_id": source_id,
        "requested_mb": args.size_mb,
        "seed": args.seed,
        "strip_markup": args.strip_markup,
        "elapsed_seconds": round(elapsed, 1),
        **stats,
        **(manifest_extra or {}),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[{source_id}] done: {stats['mb_written']} MB, {stats['records_written']} records "
          f"({elapsed:.1f}s). Manifest: {manifest_path}", file=sys.stderr)
    if stats["mb_written"] < args.size_mb * 0.9:
        print(
            f"[{source_id}] WARNING: wrote only {stats['mb_written']} MB of requested {args.size_mb} MB "
            f"-- source stream likely exhausted before budget was met.",
            file=sys.stderr,
        )

