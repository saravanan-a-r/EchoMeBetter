#!/usr/bin/env python3
"""
Run all five per-source downloaders in the exact blend proportions from
tokenizer/DESIGN.md 4.3, splitting one total MB budget across them.

    FineWeb-Edu   35%
    C4 (en)       20%
    StackExchange 15%
    Cosmopedia v2 10%
    Gutenberg/PG19 20%

Each per-source script is invoked as a subprocess (not imported) so a
failure/interrupt in one source (e.g. network drop mid-stream) doesn't
take down the others, and so each script's own argparse/manifest/output
behavior (already tested individually) is reused unchanged.

Usage:
    python all_downloader_proportional.py 8000        # ~8GB total, full blend
    python all_downloader_proportional.py 500 --seed 7
    python all_downloader_proportional.py 200 --skip stackexchange
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (source_id, script filename, blend share) -- must sum to 1.0 and match
# DESIGN.md 4.3 exactly. source_id matches each script's own --output-dir
# naming (<source_id>_<mb>mb.jsonl) so results are easy to locate after.
BLEND = [
    ("fineweb_edu", "download_fineweb_edu.py", 0.35),
    ("c4", "download_c4.py", 0.20),
    ("stackexchange", "download_stackexchange.py", 0.15),
    ("cosmopedia", "download_cosmopedia.py", 0.10),
    ("gutenberg_pg19", "download_gutenberg.py", 0.20),
]

assert abs(sum(share for _, _, share in BLEND) - 1.0) < 1e-9, "BLEND shares must sum to 1.0"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoke every download_*.py script in DESIGN.md's 4.3 blend proportions."
    )
    parser.add_argument("total_mb", type=float, help="Total target output size in MB across all sources.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Forwarded to every downloader (default: each script's own output/).")
    parser.add_argument("--seed", type=int, default=42, help="Forwarded to every downloader (default: 42).")
    parser.add_argument("--strip-markup", action="store_true", help="Forwarded to every downloader.")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="SOURCE_ID",
        help="Skip a source (e.g. --skip stackexchange). May be given multiple times.",
    )
    args = parser.parse_args()

    if args.total_mb <= 0:
        print(f"error: total_mb must be > 0, got {args.total_mb}", file=sys.stderr)
        sys.exit(1)

    plan = [(sid, script, share, round(args.total_mb * share, 1)) for sid, script, share in BLEND if sid not in args.skip]
    if not plan:
        print("error: everything was skipped, nothing to run", file=sys.stderr)
        sys.exit(1)

    print(f"Total budget: {args.total_mb} MB across {len(plan)} source(s):")
    for sid, _, share, mb in plan:
        print(f"  {sid:<15} {share*100:>5.1f}%  ->  {mb} MB")
    print()

    failures: list[str] = []
    for sid, script, _share, mb in plan:
        cmd = [sys.executable, str(HERE / script), str(mb), "--seed", str(args.seed)]
        if args.output_dir:
            cmd += ["--output-dir", str(args.output_dir)]
        if args.strip_markup:
            cmd += ["--strip-markup"]

        print(f"=== [{sid}] {' '.join(cmd)} ===", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"=== [{sid}] FAILED (exit {result.returncode}) — continuing with remaining sources ===", file=sys.stderr)
            failures.append(sid)
        print()

    if failures:
        print(f"Done with failures in: {', '.join(failures)}. Check output above.", file=sys.stderr)
        sys.exit(1)
    print("All sources completed successfully.")


if __name__ == "__main__":
    main()
