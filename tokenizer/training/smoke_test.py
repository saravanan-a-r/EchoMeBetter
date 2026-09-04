#!/usr/bin/env python3
"""
Pre-flight check: prove the pipeline works before committing the full corpus.

    python smoke_test.py --corpus training_corpus/output

Runs the complete pipeline (sample -> train -> gates) end to end against a
small slice of the real corpus using config/smoke.yaml, then applies a fixed
set of adversarial fidelity probes. Takes a couple of minutes instead of
hours, and answers the only question worth asking before a long run: *would
the real run have produced a usable tokenizer, or wasted the night?*

What it can and cannot tell you
-------------------------------
It CAN catch: broken corpus files, encoding problems, config errors, a
fidelity setting that silently breaks round-tripping, missing or non-atomic
reserved tokens, a corpus whose text is not what you think it is.

It CANNOT tell you the production fertility numbers -- those depend on the
full 32832-piece vocabulary fitted to the full corpus. Gate 4 output here is
directional only. Read the hard gates; treat the soft ones as a smell test.

Exit codes: 0 all good, 1 a check failed, 2 could not run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import corpus_reader
import train_tokenization
import trainer as trainer_module
import validate_tokenizer
from config_loader import load_config
from marker_escape import escape_markers, unescape_markers

MODULE_DIR = Path(__file__).resolve().parent
SMOKE_CONFIG = MODULE_DIR / "config" / "smoke.yaml"

# Adversarial inputs that must survive encode/decode byte-for-byte. These
# are the product's preservation contract written as test data: every one is
# a construct the rephrase model is required to reproduce verbatim while
# rewriting the prose around it.
FIDELITY_PROBES: list[tuple[str, str]] = [
    ("bare URL", "Check https://example.com/path?a=1&b=2#frag now."),
    ("markdown link", "See [the docs](https://docs.python.org/3/library/json.html) first."),
    ("nested punctuation", "He said \"it's a 'quoted' word\" (probably)."),
    ("code with indent", "    def f(x):\n        return x ** 2"),
    ("leading whitespace", "   three leading spaces"),
    ("trailing whitespace", "three trailing spaces   "),
    ("internal runs", "collapse    these     spaces?"),
    ("tabs", "col1\tcol2\tcol3"),
    ("numbers", "Revenue rose 1234567890 to 3.14159 (up 42%)."),
    ("version string", "Upgrade from v1.2.3-beta.4 to v10.20.30."),
    ("email", "Contact first.last+tag@sub.example.co.uk today."),
    ("unicode accents", "café naïve résumé Ångström"),
    ("unicode symbols", "→ ± ½ µs — “smart quotes” … ‰"),
    ("CJK", "日本語のテキストです"),
    ("emoji", "ship it 🚀🔥 now"),
    ("mixed script", "The Ω value is 42Ω, see §3.1"),
    ("html entities", "a &lt; b &amp;&amp; c &gt; d"),
    ("json fragment", '{"key": [1, 2, {"nested": null}]}'),
    ("regex", "const re = /^[a-z0-9_-]{3,16}$/gi;"),
    ("shell", "cat f.txt | grep -E '^\\d+' > out.txt 2>&1"),
    ("sql", "SELECT * FROM t WHERE x >= 18 AND y <> 'a';"),
    ("path", "/Users/apple/projects/file_name-v2.tar.gz"),
    ("style token in text", "<style:concise> should stay atomic"),
    ("sentinel in text", "<extra_id_0> and <extra_id_99> too"),
    ("empty-ish", " "),
    ("single char", "x"),
]


def check_probes(sp) -> tuple[list[str], list[str]]:
    """Round-trip every probe. Returns (failures, notes)."""
    failures, notes = [], []
    for name, text in FIDELITY_PROBES:
        decoded = unescape_markers(sp.decode(sp.encode(escape_markers(text))))
        if decoded != text:
            failures.append(f"{name}: expected {text!r}, got {decoded!r}")
    for name in ("<style:concise>", "<extra_id_0>", "<text_to_rewrite>", "[R]"):
        ids = sp.encode(name)
        if len(ids) != 1:
            failures.append(f"reserved token {name!r} split into {len(ids)} pieces")
    notes.append(f"{len(FIDELITY_PROBES)} fidelity probes, {len(failures)} failed")
    return failures, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--corpus", required=True, type=Path, help="Corpus file or directory.")
    parser.add_argument("--config", type=Path, default=SMOKE_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to put smoke artifacts (default: a temp dir, deleted afterwards).",
    )
    parser.add_argument("--keep", action="store_true", help="Keep artifacts from a temp-dir run.")
    args = parser.parse_args(argv)

    tmp_dir = None
    if args.output_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tokenizer_smoke_"))
        output_dir = tmp_dir
    else:
        output_dir = args.output_dir

    print("=" * 70)
    print("TOKENIZER SMOKE TEST")
    print("=" * 70)
    print(f"corpus: {args.corpus}")
    print(f"config: {args.config}")
    print(f"output: {output_dir}")
    print()

    try:
        rc = train_tokenization.main(
            [
                "--corpus", str(args.corpus),
                "--config", str(args.config),
                "--output-dir", str(output_dir),
            ]
        )
        if rc != 0:
            print(f"\nSMOKE TEST FAILED: pipeline exited {rc}", file=sys.stderr)
            return 1 if rc == 1 else 2

        cfg = load_config(args.config)
        sp = trainer_module.load_processor(output_dir / f"{cfg['output']['model_prefix']}.model")

        print("\n" + "=" * 70)
        print("FIDELITY PROBES (fixed adversarial inputs, independent of the corpus)")
        print("=" * 70)
        failures, notes = check_probes(sp)
        for note in notes:
            print(f"  {note}")
        for failure in failures:
            print(f"  FAIL  {failure}")

        eval_records = corpus_reader.load_eval_records(output_dir / "tokenizer_evals_set.jsonl")
        results, _ = validate_tokenizer.run_all_gates(
            sp, ((r["text"], r["source"]) for r in eval_records), cfg
        )
        hard_failed = validate_tokenizer.hard_failures(results)

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        fertility = next(r for r in results if r.number == 4)
        byte_rate = next(r for r in results if r.number == 5)
        print(f"  vocab size          {sp.get_piece_size()}")
        print(f"  fertility (overall) {fertility.metrics['overall']} tokens/word  "
              f"[directional only at smoke scale]")
        print(f"  byte-fallback rate  {byte_rate.metrics['rate']}")
        for source, value in (fertility.metrics.get("per_slice") or {}).items():
            print(f"    {source:<24} {value}")

        if failures or hard_failed:
            print("\n  RESULT: FAILED — do not start the full run.")
            if hard_failed:
                print(f"    hard gates failed: {[r.number for r in hard_failed]}")
            if failures:
                print(f"    fidelity probes failed: {len(failures)}")
            return 1

        print("\n  RESULT: PASSED — the pipeline is sound. Safe to run the full corpus:")
        print(f"    python train_tokenization.py --corpus {args.corpus} "
              f"--config config/default.yaml")
        return 0
    finally:
        if tmp_dir is not None and not args.keep:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif tmp_dir is not None:
            print(f"\nartifacts kept in {tmp_dir}")


if __name__ == "__main__":
    sys.exit(main())
