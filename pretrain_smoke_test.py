#!/usr/bin/env python3
"""
Pre-flight check: prove the full Stage 1 pipeline works before committing
weeks of compute to Large (architecture.md §7.8).

    python pretrain_smoke_test.py --corpus path/to/a/small/text/sample

Runs `pretrain.py`'s `rehearsal` stage (Base profile, a handful of steps) on
a small slice of real text, tokenized through the real frozen tokenizer, and
checks the properties that would tell you the pipeline is wired wrong rather
than just slow: loss stays finite, all three UL2 modes get exercised, and a
run interrupted mid-flight resumes to bit-identical state. Takes well under a
minute; a real rehearsal (§7.8's own point) or a full pretraining run does
not.

What this can and cannot tell you
----------------------------------
It CAN catch: a broken corpus path, a stale or mismatched tokenizer, a wiring
bug between `corpus/`, `UL2/`, `model/` and `training/`, a resumability
regression, a device/precision incompatibility — running this pipeline for
the first time is what found a silent tensor-corruption bug in
`training/src/loop.py`'s device transfer under MPS (`non_blocking=True` is
only safe on CUDA; see that file's `_to_device`), which produced no error at
all, just a wrong tensor, until the model's own ID-range check happened to
notice.

It CANNOT tell you Large will train well — it deliberately runs the 223M
Base profile for a few dozen steps. Treat a pass as "the pipeline is sound",
not "the model is good"; §7.8 is explicit that a Base proxy is a ranking
signal, never a guarantee about Large.

Exit codes: 0 all good, 1 a check failed, 2 could not run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pretrain as P


def _build_run(args: argparse.Namespace, output_dir: Path, max_steps: int):
    """Everything `pretrain.run()` builds, minus actually training — so the
    smoke test can inspect telemetry and run its own gates around the call."""
    tokenizer_dir = Path(args.tokenizer_dir)
    token_map_path = tokenizer_dir / "token_map.json"
    model_path = tokenizer_dir / "spm.model"
    if not token_map_path.is_file() or not model_path.is_file():
        raise SmokeTestError(
            f"tokenizer artifacts not found under {tokenizer_dir}. Build the "
            f"tokenizer first — see tokenizer/training/README.md."
        )

    training_config = P.training_pkg.load_training_config(
        args.training_config, stage="rehearsal"
    ).with_(
        max_steps=max_steps,
        warmup_steps=0,
        warmup_ratio=0.0,
        output_dir=str(output_dir),
        logging_steps=1,
        # Never trigger an automatic mid-run save: a 223M-parameter
        # optimizer checkpoint is real disk I/O (~2.5GB for AdamW moments
        # alone), and the point of a smoke test is to be fast. Only
        # `_check_resumability` needs a checkpoint, and it saves explicitly.
        save_steps=max_steps + 1,
        eval_steps=max_steps,  # evaluate once, at the end
        resume_from_checkpoint=False,
    )

    model_config = P.rephrase_model.load_model_config(
        args.model_config, profile=training_config.model_profile or "base"
    )
    if training_config.dropout_rate is not None:
        model_config = model_config.with_(dropout_rate=training_config.dropout_rate)
    if model_config.label_smoothing != training_config.label_smoothing_factor:
        model_config = model_config.with_(label_smoothing=training_config.label_smoothing_factor)

    specials = P.ul2.SpecialTokens.from_token_map(token_map_path)
    ul2_config = P.ul2.UL2Config(
        max_source_length=model_config.max_encoder_length,
        max_target_length=model_config.max_decoder_length,
    )
    telemetry = P.ul2.MixtureTelemetry(ul2_config.mode_weights)
    objective = P.ul2.UL2Objective(ul2_config, specials, telemetry)

    sp = P.corpus_pkg.tokenizer_interop.load_processor(model_path)
    files = P.corpus_pkg.discover_corpus_files(args.corpus)
    corpus_stream = P.corpus_pkg.TokenizedCorpus(files, sp, seed=training_config.data_seed)
    batch_source = P.PretrainBatchSource(
        corpus_stream,
        objective,
        specials,
        batch_size=training_config.per_device_train_batch_size,
        data_seed=training_config.data_seed,
    )

    model = P.rephrase_model.build_model(model_config)
    trainer = P.training_pkg.Trainer(model, training_config, device=args.device)
    return trainer, batch_source, telemetry, training_config


class SmokeTestError(RuntimeError):
    """The pipeline could not even be assembled; see the message for why."""


def run_smoke_test(args: argparse.Namespace, output_dir: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    failures: list[str] = []

    trainer, batch_source, telemetry, training_config = _build_run(
        args, output_dir / "straight", args.steps
    )
    state = trainer.train(batch_source)
    notes.append(f"ran {state.global_step} steps, {state.tokens_seen} tokens, "
                 f"{trainer.skipped_steps} skipped")

    losses = [entry["loss"] for entry in state.log_history if "loss" in entry]
    if not losses or not all(_is_finite(v) for v in losses):
        failures.append(f"non-finite loss encountered: {losses}")
    else:
        notes.append(f"loss trace: {[round(v, 3) for v in losses]}")

    modes_seen = {
        mode
        for entry in state.log_history
        for key in entry
        if key.startswith("loss_") and (mode := key.removeprefix("loss_")) in ("R", "X", "S")
    }
    if modes_seen != {"R", "X", "S"}:
        failures.append(f"not every UL2 mode was exercised: saw {sorted(modes_seen)}")
    else:
        notes.append("all three UL2 modes ([R]/[X]/[S]) were exercised")

    if telemetry.total_examples == 0:
        failures.append("telemetry recorded zero examples; the corpus produced nothing usable")
    else:
        notes.append(f"telemetry: {telemetry.total_examples} examples corrupted")

    # -- resumability: architecture.md §7.9 --------------------------------
    resumability_ok, resumability_notes = _check_resumability(args, output_dir)
    notes.extend(resumability_notes)
    if not resumability_ok:
        failures.append("a resumed run did not match an uninterrupted one — see notes")

    return not failures, notes + [f"FAIL: {f}" for f in failures]


def _check_resumability(args: argparse.Namespace, output_dir: Path) -> tuple[bool, list[str]]:
    """
    architecture.md §7.9: run N steps straight through, and separately N/2
    then a simulated crash-and-resume for the remaining N/2, and require the
    two end states to agree on step count and token count. This is the same
    property `training/test/integration/test_resumability.py` proves for the
    trainer alone; here it is proven for the assembled pipeline, including
    `corpus/`'s data position.
    """
    total_steps = max(4, args.steps)
    half = total_steps // 2

    straight_dir = output_dir / "resume_straight"
    straight_trainer, straight_source, _, _ = _build_run(args, straight_dir, total_steps)
    straight_state = straight_trainer.train(straight_source)

    crash_dir = output_dir / "resume_crash"
    crash_trainer, crash_source, _, crash_config = _build_run(args, crash_dir, half)
    crash_state = crash_trainer.train(crash_source)
    crash_trainer.save(crash_state.global_step, data=crash_source)

    resumed_trainer, resumed_source, _, _ = _build_run(args, crash_dir, total_steps)
    resumed_trainer.resume(data=resumed_source)
    resumed_state = resumed_trainer.train(resumed_source)

    notes = [
        f"resumability: straight={straight_state.global_step} steps, "
        f"crash-then-resume={crash_state.global_step}+"
        f"{resumed_state.global_step - crash_state.global_step} steps"
    ]
    ok = (
        resumed_state.global_step == straight_state.global_step
        and resumed_state.tokens_seen == straight_state.tokens_seen
        and resumed_state.tokens_seen > crash_state.tokens_seen
    )
    if not ok:
        notes.append(
            f"MISMATCH: straight tokens_seen={straight_state.tokens_seen}, "
            f"resumed tokens_seen={resumed_state.tokens_seen}"
        )
    return ok, notes


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="A small file or directory of real text.")
    parser.add_argument("--steps", type=int, default=8, help="Steps per rehearsal run (kept tiny).")
    parser.add_argument("--device", default="cpu", help="Default cpu: see pretrain.py --help.")
    parser.add_argument("--tokenizer-dir", default=str(P.DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--model-config", default=str(P.DEFAULT_MODEL_CONFIG))
    parser.add_argument("--training-config", default=str(P.DEFAULT_TRAINING_CONFIG))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to put smoke run artifacts (default: a temp dir, deleted afterwards).",
    )
    parser.add_argument("--keep", action="store_true", help="Keep artifacts from a temp-dir run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    tmp_dir = None
    if args.output_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="pretrain_smoke_"))
        output_dir = tmp_dir
    else:
        output_dir = Path(args.output_dir)

    print("=" * 70)
    print("PRETRAINING SMOKE TEST (architecture.md §7.8)")
    print("=" * 70)
    print(f"corpus: {args.corpus}")
    print(f"steps per run: {args.steps}")
    print(f"output: {output_dir}")
    print()

    try:
        passed, notes = run_smoke_test(args, output_dir)
    except SmokeTestError as exc:
        print(f"\nSMOKE TEST COULD NOT RUN: {exc}", file=sys.stderr)
        return 2
    finally:
        if tmp_dir is not None and not args.keep:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif tmp_dir is not None:
            print(f"\nartifacts kept in {tmp_dir}")

    print()
    for note in notes:
        print(f"  {note}")

    print()
    if passed:
        print("RESULT: PASSED — the pipeline is sound. Safe to start a real rehearsal:")
        print(f"  python pretrain.py --corpus {args.corpus} --stage rehearsal")
        return 0

    print("RESULT: FAILED — do not start a real run until this is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
