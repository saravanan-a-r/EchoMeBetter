"""
Count pretraining tokens in `output/`, using the frozen SentencePiece
tokenizer, exactly the way `corpus.src.tokenized_corpus.TokenizedCorpus`
tokenizes at training time: parse each line's `"text"` field only, split it
on `"\\n"` into segments, strip, skip empty, `escape_markers()`, then
`sp.encode()` -- one call per segment. A count computed any other way (raw
characters / 4, whole-file byte counts, etc.) is not the number that will
actually reach the model, which is the number this answers.

Scope
-----
Only files inside a source subdirectory of `output/` are corpus:
`ebooks/*.jsonl` and `pretrain_output/<source>/*.jsonl`. `output/
pii_audit.jsonl` (a PII audit log, not corpus text -- it has no "text" key at
all) and every `run_manifest.json` are provenance, not training text, and are
excluded by construction: `discover_source_files` only takes `.jsonl` files
that sit at least one directory below `output/`, never a file directly in
`output/` itself.

Parallelism
-----------
CPU-bound (SentencePiece `encode`), not I/O-bound at this corpus's size (a
few hundred GB read once), so this forks one process per file, up to
`--workers`. Each worker loads its own `SentencePieceProcessor` -- the model
is 560KB, reloading it costs nothing next to the GB-scale shard it then
processes, and it sidesteps whether the C++ processor object survives a fork
cleanly (it does under `fork`, but there's no reason to rely on that when
reloading is free).

The Qwen-32B `rephrase` job runs in its own tmux pane on this box and is
GPU-bound: `top` shows it pinned to essentially one CPU core, not forty-two.
Default worker count is `cpu_count() - RESERVED_CORES`, leaving headroom for
that job's one core plus the OS, rather than claiming every core on the
machine.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from corpus.src.tokenizer_interop import escape_markers, load_processor  # noqa: E402

DEFAULT_TOKENIZER_DIR = REPO_ROOT / "tokenizer" / "training" / "output"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Cores left for the OS and the Qwen-32B `rephrase` job's CPU-side thread.
RESERVED_CORES = 2

_SP = None  # per-worker global; set once by `_init_worker`, never mutated after


def _init_worker(model_path: str) -> None:
    global _SP
    _SP = load_processor(model_path)


def discover_source_files(output_dir: Path) -> list[Path]:
    """
    Every `*.jsonl` strictly below `output_dir`'s own top level -- i.e. inside
    a source's folder (`ebooks/`, `pretrain_output/<source>/`), never a file
    sitting directly in `output_dir`. That one rule is what keeps
    `pii_audit.jsonl` out of the scan without naming it specially.
    """
    files = sorted(p for p in output_dir.rglob("*.jsonl") if p.parent != output_dir)
    if not files:
        raise SystemExit(f"no corpus .jsonl files found under {output_dir}")
    return files


def _extract_text(raw_line: str) -> str | None:
    """Same contract as `corpus.src.reader.extract_text` for jsonl lines."""
    try:
        row = json.loads(raw_line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(row, dict):
        return None
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def count_file(path_str: str) -> dict:
    """
    Token count for one file, counted exactly as `TokenizedCorpus` would at
    training time. Only the `"text"` value is ever tokenized -- no other
    JSON key on the line contributes a single token.
    """
    path = Path(path_str)
    tokens = 0
    lines_seen = 0
    lines_used = 0
    segments = 0

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            lines_seen += 1
            text = _extract_text(raw_line.rstrip("\n").rstrip("\r"))
            if text is None:
                continue
            lines_used += 1
            for segment in text.split("\n"):
                segment = segment.strip()
                if not segment:
                    continue
                segments += 1
                tokens += len(_SP.encode(escape_markers(segment), out_type=int))

    return {
        "file": str(path),
        "source": path.parent.name,
        "bytes": path.stat().st_size,
        "lines_seen": lines_seen,
        "lines_used": lines_used,
        "segments": segments,
        "tokens": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"process count; default is cpu_count() - {RESERVED_CORES}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_path = Path(args.tokenizer_dir) / "spm.model"
    if not model_path.is_file():
        raise SystemExit(f"tokenizer model not found: {model_path}")

    files = discover_source_files(output_dir)
    total_bytes = sum(p.stat().st_size for p in files)

    cpu_count = os.cpu_count() or 1
    workers = args.workers or max(1, cpu_count - RESERVED_CORES)
    workers = min(workers, len(files))

    print(f"Files: {len(files)}   Total size: {total_bytes / 1024**3:.2f} GB")
    print(
        f"CPUs available: {cpu_count}   Using {workers} worker processes "
        f"(reserving {RESERVED_CORES} for the OS / rephrase's Qwen-32B job)"
    )
    print(f"Tokenizer: {model_path}")
    print()

    started = time.time()
    ctx = mp.get_context("fork")
    results: list[dict] = []
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(str(model_path),)) as pool:
        for i, result in enumerate(pool.imap_unordered(count_file, [str(p) for p in files]), 1):
            results.append(result)
            elapsed = time.time() - started
            done_bytes = sum(r["bytes"] for r in results)
            rate_mb_s = done_bytes / max(elapsed, 1e-6) / 1024**2
            name = f"{result['source']}/{Path(result['file']).name}"
            print(f"[{i}/{len(files)}] {name}: {result['tokens']:,} tokens   ({rate_mb_s:.1f} MB/s overall)")

    elapsed = time.time() - started

    by_source: dict[str, dict] = {}
    for r in results:
        s = by_source.setdefault(r["source"], {"tokens": 0, "bytes": 0, "lines_used": 0, "files": 0})
        s["tokens"] += r["tokens"]
        s["bytes"] += r["bytes"]
        s["lines_used"] += r["lines_used"]
        s["files"] += 1

    total_tokens = sum(r["tokens"] for r in results)

    print()
    print("=== Per source ===")
    for source, s in sorted(by_source.items(), key=lambda kv: -kv[1]["tokens"]):
        print(f"  {source:>16}: {s['tokens']:>15,} tokens   {s['bytes'] / 1024**3:6.2f} GB   {s['files']:3d} files")

    print()
    print(
        f"TOTAL: {total_tokens:,} tokens across {len(files)} files "
        f"({total_bytes / 1024**3:.2f} GB) in {elapsed:.1f}s "
        f"({total_bytes / max(elapsed, 1e-6) / 1024**2:.1f} MB/s)"
    )

    report = {
        "output_dir": str(output_dir),
        "tokenizer_model": str(model_path),
        "workers": workers,
        "elapsed_seconds": round(elapsed, 1),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "by_source": by_source,
        "per_file": results,
    }
    report_path = output_dir / "token_count_report.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
