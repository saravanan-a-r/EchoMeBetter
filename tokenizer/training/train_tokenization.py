#!/usr/bin/env python3
"""
Entry point for tokenizer training.

    python train_tokenization.py --corpus /path/to/corpus

`--corpus` may be a single .jsonl/.txt file or a directory of them (the
`training_corpus/output/` folder produced by the downloaders). Everything
else is controlled by config/default.yaml.

Pipeline (DESIGN.md 3.1):

    discover corpus files
      -> seeded sample -> output/corpus.txt          (the training lines)
      -> SentencePieceTrainer -> output/spm.model + output/spm.vocab
      -> 11 acceptance gates, streamed over ALL REMAINING corpus
      -> output/token_map.json + manifest.json + report.md

Training and evaluation are disjoint by construction: the sampler selects
`corpus.input_sentence_size` lines to train on, and **every other eligible
line is the evaluation set**. That eval set is not written to disk (it is
~18 GB on the real corpus) -- it is replayed from the same seeded split, so
the gates measure the vocabulary against every byte it will actually meet.

Use `--eval-corpus N` to evaluate a uniform random subsample of N lines
instead, when a faster estimate is enough.

Exits non-zero if any hard gate fails, so this is safe to run in CI.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

import corpus_reader
import trainer as trainer_module
import validate_tokenizer
from config_loader import ConfigError, load_config
from corpus_reader import CorpusError
from execution import Executor
from specials import ALL_NAMED_SPECIALS

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODULE_DIR / "config" / "default.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SentencePiece Unigram + byte-fallback tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Training corpus: a .jsonl/.txt file, or a directory containing them.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML config controlling the run."
    )
    # Run-scoped overrides. Everything that shapes the *vocabulary* stays in
    # the YAML so a build is described by its config file alone.
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output.dir.")
    parser.add_argument(
        "--input-sentence-size", type=int, default=None, help="Override corpus.input_sentence_size."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override sampling.seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes for the corpus scan, sampling and eval scan. "
             "Override runtime.workers (0 = auto-detect). Results are identical "
             "at any worker count; only wall-clock time changes.",
    )
    parser.add_argument(
        "--eval-corpus",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate the gates on a uniform random sample of N lines from the remaining "
        "(non-training) corpus. Default: every remaining line. Overrides "
        "validation.eval_sentence_size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and sample the corpus, then stop before training. Use this to validate a "
        "large corpus and see its statistics without committing to a multi-hour run.",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip the acceptance gates (not recommended)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show SentencePiece's own training log (several hundred lines).",
    )
    return parser


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.input_sentence_size is not None:
        if args.input_sentence_size < 1:
            raise ConfigError("--input-sentence-size must be >= 1")
        cfg["corpus"]["input_sentence_size"] = args.input_sentence_size
    if args.seed is not None:
        cfg["sampling"]["seed"] = args.seed
    if args.eval_corpus is not None:
        if args.eval_corpus < 0:
            raise ConfigError("--eval-corpus must be >= 0 (0 = all remaining corpus)")
        cfg["validation"]["eval_sentence_size"] = args.eval_corpus
    if args.output_dir is not None:
        cfg["output"]["dir"] = str(args.output_dir)
    return cfg


def resolve_output_dir(cfg: dict) -> Path:
    """Relative output paths resolve against the module, not the cwd."""
    out = Path(cfg["output"]["dir"])
    return out if out.is_absolute() else (MODULE_DIR / out)


def write_token_map(sp, output_dir: Path) -> Path:
    """
    Resolve every reserved token name to its integer ID (DESIGN.md 5.3).

    This file is the contract with pretraining, SFT, DPO and inference:
    they look IDs up by name here. No integer token ID is ever hardcoded.
    """
    token_map = {name: sp.piece_to_id(name) for name in ALL_NAMED_SPECIALS}

    unk_id = sp.unk_id()
    unresolved = [n for n, i in token_map.items() if i == unk_id and n != "<unk>"]
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} reserved token(s) did not resolve to a real id: "
            f"{unresolved[:10]}"
        )
    collisions = {i for i in token_map.values()}
    if len(collisions) != len(token_map):
        raise RuntimeError("reserved token names collided onto the same id")

    path = output_dir / "token_map.json"
    path.write_text(json.dumps(token_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = time.time()

    try:
        cfg = apply_overrides(load_config(args.config), args)
        files = corpus_reader.discover_corpus_files(args.corpus)
    except (ConfigError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The one place a core count enters the pipeline. Everything downstream
    # takes an Executor and never asks how big the machine is.
    executor = Executor.from_config(
        args.workers if args.workers is not None else cfg["runtime"]["workers"]
    )

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_txt = output_dir / "corpus.txt"
    eval_sample_jsonl = output_dir / "tokenizer_evals_set.jsonl"

    eval_budget = cfg["validation"]["eval_sentence_size"]
    print(f"config:  {args.config}")
    print(f"corpus:  {args.corpus}  ({len(files)} file(s))")
    print(f"output:  {output_dir}")
    print(f"budget:  {cfg['corpus']['input_sentence_size']:,} train sentences, "
          f"eval on {'ALL remaining' if eval_budget == 0 else f'{eval_budget:,} sampled'} lines, "
          f"seed={cfg['sampling']['seed']}")
    print("\n[1/5] scanning and sampling corpus (two streaming passes)...", flush=True)

    try:
        corpus_info = corpus_reader.build_corpus(
            files,
            corpus_out=corpus_txt,
            train_budget=cfg["corpus"]["input_sentence_size"],
            seed=cfg["sampling"]["seed"],
            max_sentence_length=cfg["corpus"]["max_sentence_length"],
            max_drop_fraction=cfg["corpus"]["max_drop_fraction"],
            blend=cfg["blend"],
            executor=executor,
        )
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    scan = corpus_info["scan"]
    print(f"      eligible lines : {corpus_info['eligible_lines']:,}")
    print(f"      training lines : {corpus_info['train_lines']:,} -> {corpus_txt.name} "
          f"({corpus_info['corpus_bytes'] / 1e6:.1f} MB)")
    print(f"      eval lines     : {corpus_info['eval_lines_available']:,} available "
          f"(everything not trained on)")
    print(f"      dropped        : {scan['lines_oversized']:,} oversized, "
          f"{scan['records_bad_json']:,} bad JSON, {scan['records_missing_text']:,} no text "
          f"(drop fraction {scan['drop_fraction']})")
    for note in corpus_info["blend_notes"]:
        print(f"      blend note: {note}")
    print(f"\n      blend ({corpus_info['blend_mode']}) — share of sampled text bytes:")
    print(f"      {'source':<18}{'available':>14}{'sampled':>12}{'target':>9}{'achieved':>10}")
    target = corpus_info["blend_target"]
    achieved = corpus_info["blend_achieved_bytes"]
    for source, count in sorted(scan["per_source_lines"].items()):
        want = f"{target[source] * 100:.1f}%" if source in target else "-"
        got = f"{achieved[source] * 100:.1f}%" if source in achieved else "0.0%"
        sampled = corpus_info["train_lines_per_source"].get(source, 0)
        print(f"      {source:<18}{count:>13,}L{sampled:>11,}L{want:>9}{got:>10}")

    if args.dry_run:
        print("\n--dry-run: stopping before training. corpus.txt is ready.")
        return 0

    if corpus_info["train_lines"] == 0:
        print("error: no training lines were sampled", file=sys.stderr)
        return 2

    print("\n[2/5] training SentencePiece (CPU-bound; this is the long step)...", flush=True)
    try:
        training_info = trainer_module.train(cfg, corpus_txt, output_dir, verbose=args.verbose)
    except (RuntimeError, OSError) as exc:
        print(f"error: SentencePiece training failed: {exc}", file=sys.stderr)
        if "Vocabulary size too high" in str(exc):
            # By far the most common failure, and the message alone does not
            # say what to do about it.
            print(
                "\nhint: the sampled corpus does not contain enough distinct text to support "
                f"trainer.vocab_size={cfg['trainer']['vocab_size']}. Either raise "
                "corpus.input_sentence_size / point --corpus at more data, or lower "
                "trainer.vocab_size to the value SentencePiece suggests above. For the real "
                "20GB corpus this should not occur; on a small slice it is expected.",
                file=sys.stderr,
            )
        return 3
    print(f"      done in {training_info['elapsed_seconds']}s -> "
          f"{Path(training_info['model_path']).name}, {Path(training_info['vocab_path']).name}")

    model_path = Path(training_info["model_path"])
    sp = trainer_module.load_processor(model_path)

    print("\n[3/5] resolving token map...", flush=True)
    try:
        token_map_path = write_token_map(sp, output_dir)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    print(f"      {len(ALL_NAMED_SPECIALS)} reserved names -> {token_map_path.name}")

    results: list[validate_tokenizer.GateResult] = []
    evaluation: dict = {"ran": False}
    if args.skip_validation or not cfg["validation"]["run_after_training"]:
        print("\n[4/5] validation skipped.")
    else:
        plan = corpus_reader.SplitPlan.from_dict(corpus_info["plan"])
        available = plan.n_eval
        limit = cfg["validation"]["eval_sentence_size"]
        n_expected = available if limit <= 0 else min(limit, available)
        scope = (
            "all remaining corpus" if limit <= 0 else f"{n_expected:,}-line sample of remaining"
        )
        print(f"\n[4/5] running acceptance gates on {n_expected:,} eval lines "
              f"({scope})...", flush=True)
        if available == 0:
            print("warning: no eval lines available (the corpus was entirely consumed by "
                  "training); gates 1/2/4/5/6/8/9/10/11 cannot be measured", file=sys.stderr)

        eval_started = time.time()
        # The eval sample saved for export_hf.py and human inspection is drawn
        # by the scan itself (a bounded reservoir per file, merged), so the
        # eval corpus is still read exactly once no matter how many workers
        # read it.
        sample_budget = cfg["validation"]["eval_sample_save"]

        try:
            eval_scan = validate_tokenizer.scan_eval_corpus(
                model_path,
                files,
                plan,
                limit=limit,
                length_reservoir_size=cfg["validation"]["length_reservoir_size"],
                seed=cfg["sampling"]["seed"],
                sample_budget=sample_budget,
                executor=executor,
                progress=True,
            )
            results, eval_scan = validate_tokenizer.run_all_gates(
                sp, None, cfg, progress=True, scan=eval_scan
            )
        except CorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        sample = eval_scan.sample

        with open(eval_sample_jsonl, "w", encoding="utf-8") as fh:
            for record in sample:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        evaluation = {
            "ran": True,
            "scope": scope,
            "lines_available": available,
            "lines_evaluated": eval_scan.n_records,
            "eval_sentence_size": limit,
            "tokens_evaluated": eval_scan.total_tokens,
            "elapsed_seconds": round(time.time() - eval_started, 2),
            "sample_saved": len(sample),
            "sample_path": str(eval_sample_jsonl),
        }
        print(f"      evaluated {eval_scan.n_records:,} lines / "
              f"{eval_scan.total_tokens:,} tokens in {evaluation['elapsed_seconds']}s")
        print(f"      saved a {len(sample):,}-line sample -> {eval_sample_jsonl.name}\n")
        for r in results:
            if r.hard:
                status = "PASS" if r.passed else "FAIL"
            elif r.number in (7, 9, 11):
                status = "INFO"
            else:
                status = "PASS" if r.passed else "WARN"
            print(f"      [{status}] gate {r.number}: {r.name} — {r.detail}")
            for example in r.examples[:3]:
                print(f"              {example}")

    print("\n[5/5] writing artifacts...", flush=True)
    manifest = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 2),
        "corpus": {
            **corpus_info,
            "source_files": [str(p) for p in files],
        },
        "training": {
            **training_info,
            "model_sha256": corpus_reader.sha256_file(model_path),
        },
        "evaluation": evaluation,
        "config": cfg,
        "config_path": str(args.config),
        "gates": [r.as_dict() for r in results],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    # Gate 7 embeds 1000 vocabulary pieces; they belong in the human-readable
    # report, not duplicated into the machine-readable manifest.
    for gate in manifest["gates"]:
        if gate["gate"] == 7:
            gate["metrics"] = {"vocab_size": gate["metrics"]["vocab_size"]}

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if results:
        (output_dir / "report.md").write_text(
            validate_tokenizer.render_report(results, manifest), encoding="utf-8"
        )

    failures = validate_tokenizer.hard_failures(results)
    if failures:
        print(
            f"\nFAILED: {len(failures)} hard gate(s) did not pass: "
            f"{', '.join(str(r.number) for r in failures)}. See {output_dir / 'report.md'}.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK. Artifacts in {output_dir}")
    print("Run freeze.py to mark this build immutable before pretraining uses it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
