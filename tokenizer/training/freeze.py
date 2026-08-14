#!/usr/bin/env python3
"""
Mark a validated tokenizer build immutable (DESIGN.md 10).

    python freeze.py --output-dir output

Refuses to run unless the artifacts exist and every hard gate passed. Once
frozen, the vocabulary is welded to every embedding row of every model
trained on it: there is no migration path, only a full retrain. The FROZEN
marker exists so that fact is visible in the filesystem, not just in a
design document.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from corpus_reader import sha256_file

MODULE_DIR = Path(__file__).resolve().parent
REQUIRED_ARTIFACTS = ("spm.model", "spm.vocab", "token_map.json", "manifest.json", "report.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a validated tokenizer build.")
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR / "output")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-freeze a directory that already carries a FROZEN marker.",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir

    marker = output_dir / "FROZEN"
    if marker.exists() and not args.force:
        print(f"error: {marker} already exists. A frozen build is immutable; produce a new "
              f"version instead of re-freezing (--force overrides).", file=sys.stderr)
        return 1

    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        print(f"error: missing artifact(s) in {output_dir}: {', '.join(missing)}. "
              f"Run train_tokenization.py first.", file=sys.stderr)
        return 1

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    gates = manifest.get("gates", [])
    if not gates:
        print("error: manifest records no validation gates. Refusing to freeze an "
              "unvalidated build.", file=sys.stderr)
        return 1

    failed = [g for g in gates if g["hard"] and not g["passed"]]
    if failed:
        print(f"error: {len(failed)} hard gate(s) failed: "
              f"{', '.join(str(g['gate']) for g in failed)}. Refusing to freeze.", file=sys.stderr)
        return 1

    warned = [g for g in gates if not g["hard"] and not g["passed"]]
    if warned:
        print(f"note: {len(warned)} soft gate(s) reported warnings: "
              f"{', '.join(str(g['gate']) for g in warned)} (not blocking).")

    frozen = {
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "vocab_size": manifest["config"]["trainer"]["vocab_size"],
        "model_sha256": sha256_file(output_dir / "spm.model"),
        "token_map_sha256": sha256_file(output_dir / "token_map.json"),
        "corpus_sha256": manifest["corpus"]["corpus_sha256"],
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "hard_gates_passed": [g["gate"] for g in gates if g["hard"]],
        "note": (
            "This tokenizer is immutable. Every embedding row of every model trained "
            "on it is bound to these token IDs. Any change requires a new version and "
            "a full retrain — never an edit."
        ),
    }
    marker.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    print(f"FROZEN: {marker}")
    print(f"  vocab_size    {frozen['vocab_size']}")
    print(f"  model sha256  {frozen['model_sha256']}")
    print(f"  corpus sha256 {frozen['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
