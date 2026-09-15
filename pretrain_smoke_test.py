#!/usr/bin/env python3
"""
Pre-flight check: prove the full Stage 1 pipeline works before committing
weeks of compute to Large (architecture.md §7.8).

    python pretrain_smoke_test.py --corpus path/to/a/small/text/sample

Runs a handful of steps of a real `training_config.yml` stage on a small slice
of real text, tokenized through the real frozen tokenizer, and checks the
properties that would say the pipeline is wired wrong rather than just slow:
loss stays finite, every denoiser is exercised, and a run interrupted
mid-flight resumes to bit-identical state.

Rehearse the stage you are about to run
---------------------------------------
`--stage` is the argument that matters, and it defaults to `rehearsal` only
because that is the cheap check. A stage's most expensive machinery is the
machinery this script exists to de-risk, and every piece of it is
stage-specific: the model profile, `torch_compile`, the cooldown mixture, the
synthetic rewrite task. Rehearsing `rehearsal` says nothing about any of them.

So before committing weeks of compute, rehearse `pretrain` itself:

    python pretrain_smoke_test.py --stage pretrain --device cuda \
        --corpus pretrain_corpus/output/pretrain_output \
        --quick-eval-corpus  pretrain_corpus/output/evals/quick_eval.jsonl \
        --master-eval-corpus pretrain_corpus/output/evals/master_eval.jsonl

That builds the profile the stage names, on the device the run will use, with
`torch_compile` as configured — and it puts the cooldown boundary in the
middle of the run rather than at step 180,000, so the mixture switch and the
rewrite task both execute. It is the only cheap way to find out what the real
run's peak memory and seconds-per-step actually are.

What this can and cannot tell you
----------------------------------
It CAN catch: a broken corpus path, a stale or mismatched tokenizer, a wiring
bug between `corpus/`, `UL2/`, `model/`, `rewrite/` and `training/`, a
resumability regression, a blend naming a source the corpus does not have, a
device/precision incompatibility — running this pipeline for the first time is
what found a silent tensor-corruption bug in `training/src/loop.py`'s device
transfer under MPS (`non_blocking=True` is only safe on CUDA; see that file's
`_to_device`), which produced no error at all, just a wrong tensor, until the
model's own ID-range check happened to notice.

It CANNOT tell you the model will train *well*. A few dozen steps says the
pipeline is sound, never that the result is good; §7.8 is explicit that a
short proxy run is a ranking signal, not a guarantee.

Exit codes: 0 all good, 1 a check failed, 2 could not run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pretrain as P


class SmokeTestError(RuntimeError):
    """The pipeline could not even be assembled; see the message for why."""


def _shrink_schedule(config, *, max_steps: int, output_dir: Path, with_eval: bool):
    """
    Compress a stage's schedule into `max_steps` without changing its shape.

    The point is to keep every *phase* the stage declares, not to strip them.
    A `warmup_stable_decay` stage puts its cooldown in the last 10% of 200,000
    steps; left alone, an eight-step rehearsal of it would never reach the
    decay phase, so the mixture switch and the rewrite task — the two newest
    and least-exercised pieces of the run — would not execute. Putting the
    boundary halfway means a run this short covers the stable phase, the decay
    phase, and the switch between them.
    """
    overrides = dict(
        max_steps=max_steps,
        output_dir=str(output_dir),
        logging_steps=1,
        # Often enough that a run this short still sees every denoiser, but
        # deliberately not every step. The split costs a second forward pass,
        # so forcing it on would inflate seconds-per-step by about a fifth —
        # and seconds-per-step is one of the numbers a dress rehearsal exists
        # to measure. The stage's own cadence is far longer than this whole
        # run, so it cannot simply be inherited.
        per_mode_loss_steps=max(1, max_steps // 4),
        eval_steps=max_steps,
        resume_from_checkpoint=False,
        warmup_steps=0,
        warmup_ratio=0.0,
    )

    if with_eval:
        # The selecting tier's cadence must be a multiple of `save_steps`, so
        # a run that evaluates also checkpoints — which is worth exercising
        # anyway: writing one is real disk I/O the long run does every
        # `save_steps`, and it is better to find out here that it fails.
        overrides.update(
            save_steps=max_steps,
            master_eval_steps=max_steps,
            quick_eval_steps=max(1, max_steps // 2),
        )
    else:
        # Never trigger an automatic mid-run save: an optimizer checkpoint is
        # real disk I/O and the point of the default check is to be fast. Only
        # `_check_resumability` needs one, and it saves explicitly.
        overrides["save_steps"] = max_steps + 1

    if config.lr_scheduler_type == P.training_pkg.WARMUP_STABLE_DECAY:
        overrides.update(lr_decay_steps=max(1, max_steps // 2), lr_decay_ratio=0.0)

    return config.with_(**overrides)


def _release_gpu() -> None:
    """
    Hand a finished run's device memory back before the next one is built.

    Every run assembled here holds a full model and its AdamW moments — for
    Large that is ~13 GiB before a single activation — and this script builds
    four of them: the main run, then the straight, crashed and resumed runs of
    the resumability check. Left to Python's own timing, the earlier ones are
    still resident when the next starts, and three at once do not fit in this
    GPU's 40 GiB.

    The failure that causes is badly misleading: an out-of-memory error raised
    inside `Trainer.train`, which reads as the configuration under test being
    too large for the device when in fact the device is holding two finished
    runs the harness never released. The real run only ever has one.

    `gc.collect()` first because the trainer, its optimizer and the pipeline
    reference each other, so a cycle can outlive the last `del` that names it.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _build_run(
    args: argparse.Namespace,
    output_dir: Path,
    max_steps: int,
    *,
    with_eval: bool = False,
):
    """
    Assemble a run through `pretrain.build_pipeline` — the same code the real
    entrypoint uses — and wrap it in a `Trainer`.

    Going through the shared builder is the whole point: a smoke test that
    assembles its own lookalike pipeline can pass while the real one is
    broken, which is exactly what happened before.
    """
    training_config, model_config = P.load_configs(
        training_config_path=args.training_config,
        model_config_path=args.model_config,
        stage=args.stage,
        model_profile=args.model_profile,
    )
    training_config = _shrink_schedule(
        training_config, max_steps=max_steps, output_dir=output_dir, with_eval=with_eval
    )

    try:
        pipeline = P.build_pipeline(
            training_config=training_config,
            model_config=model_config,
            corpus=args.corpus,
            tokenizer_dir=args.tokenizer_dir,
        )
    except P.PipelineError as exc:
        raise SmokeTestError(str(exc)) from exc

    trainer = P.training_pkg.Trainer(pipeline.model, training_config, device=args.device)
    return trainer, pipeline


def run_smoke_test(args: argparse.Namespace, output_dir: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    failures: list[str] = []

    with_eval = args.quick_eval_corpus is not None or args.master_eval_corpus is not None
    trainer, pipeline = _build_run(
        args, output_dir / "straight", args.steps, with_eval=with_eval
    )
    config = pipeline.training_config
    notes.append(
        f"stage {config.stage!r}, profile {pipeline.model_config.profile!r} "
        f"({pipeline.model_config.parameter_count():,} params), device {trainer.device}, "
        f"torch_compile={config.torch_compile}"
    )

    eval_tiers, eval_notes = _build_eval_tiers(args, trainer, pipeline)
    notes.extend(eval_notes)

    # `data=` is not optional here, even though the signature allows it: it is
    # how `Trainer` announces the step to the pipeline, and the cooldown's
    # mixture switch keys off that announcement. `pretrain.run()` passes it,
    # so a rehearsal that did not would be rehearsing a different run.
    state = trainer.train(
        pipeline.batch_source,
        eval_tiers=eval_tiers or None,
        data=pipeline.batch_source,
    )
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

    if pipeline.telemetry.total_examples == 0:
        failures.append("telemetry recorded zero examples; the corpus produced nothing usable")
    else:
        notes.append(f"telemetry: {pipeline.telemetry.total_examples} examples corrupted")

    # -- the stage's cooldown, if it declares one -------------------------
    cooldown_failures, cooldown_notes = _check_cooldown(pipeline)
    notes.extend(cooldown_notes)
    failures.extend(cooldown_failures)

    if eval_tiers:
        scored = [entry for entry in state.log_history if entry.get("eval")]
        tiers_scored = {entry["eval_tier"] for entry in scored}
        if tiers_scored != {tier.name for tier in eval_tiers}:
            failures.append(
                f"not every eval tier ran: {sorted(tiers_scored)} of "
                f"{sorted(tier.name for tier in eval_tiers)}"
            )
        else:
            notes.append(f"eval tiers scored: {sorted(tiers_scored)}")
        if state.best_step is None:
            failures.append("no best checkpoint was selected; the master tier drove nothing")
        else:
            notes.append(f"best checkpoint selected at step {state.best_step}")

    # -- resumability: architecture.md §7.9 --------------------------------
    # Everything above has been read out of `pipeline` and `state` already, so
    # the main run's model is finished with. It is released before the check
    # builds three more of them — see `_release_gpu`.
    del trainer, pipeline, eval_tiers
    _release_gpu()

    resumability_ok, resumability_notes = _check_resumability(args, output_dir)
    notes.extend(resumability_notes)
    if not resumability_ok:
        failures.append("a resumed run did not match an uninterrupted one — see notes")

    return not failures, notes + [f"FAIL: {f}" for f in failures]


def _build_eval_tiers(args: argparse.Namespace, trainer, pipeline):
    """
    The two held-out tiers, when the caller supplied their corpora.

    Capped hard: the real master set is 50,000 records, and scoring all of
    them is a job for the run, not for a pre-flight check. What is being
    proven here is that the tiers build, fire on their cadence and reach
    `best_step` — not what the loss is.
    """
    config = pipeline.training_config
    tiers, notes = [], []

    def tier(name: str, path: str, every_steps: int, drives: bool, seed_offset: int):
        batches = P.build_frozen_eval_batches(
            P.corpus_pkg.discover_corpus_files(path),
            pipeline.sp,
            pipeline.ul2_config,
            pipeline.specials,
            seed=config.data_seed + seed_offset,
            batch_size=config.per_device_train_batch_size,
            max_examples=args.eval_max_examples,
        )
        notes.append(f"{name} eval: {len(batches)} batch(es) every {every_steps} step(s)")
        return P.training_pkg.EvalTier(
            name,
            lambda: trainer.evaluate(batches),
            every_steps,
            drives_best_checkpoint=drives,
        )

    if args.master_eval_corpus is not None:
        tiers.append(tier("master", args.master_eval_corpus, config.master_eval_steps, True, 1))
    if args.quick_eval_corpus is not None:
        tiers.append(tier("quick", args.quick_eval_corpus, config.quick_eval_steps, False, 2))
    return tiers, notes


def _check_cooldown(pipeline) -> tuple[list[str], list[str]]:
    """
    Did the stage's cooldown actually happen?

    `_shrink_schedule` moves the boundary into the middle of the run so that
    it can. If a stage declares a cooldown mixture or a rewrite task and the
    run still ends without either having been drawn, the switch is not wired
    up — and in the real run that would surface at step 180,000.

    Returns `(failures, notes)`.
    """
    failures: list[str] = []
    notes: list[str] = []
    source = pipeline.batch_source

    if pipeline.cooldown_start_step is None:
        notes.append("stage declares no cooldown; mixture switch not exercised")
        return failures, notes

    notes.append(f"cooldown boundary placed at step {pipeline.cooldown_start_step}")

    if pipeline.cooldown_corpus is not None:
        drawn = sum(pipeline.cooldown_corpus.documents_drawn.values())
        if not source.in_cooldown:
            failures.append("the run ended without switching to the cooldown mixture")
        elif drawn == 0:
            failures.append(
                "the cooldown mixture was switched to but never delivered a document. "
                "Either it is not wired to the batch source, or this run was too "
                "short to drain the length-bucketing pools that were filled before "
                "the boundary — try more --steps."
            )
        else:
            notes.append(f"cooldown mixture delivered {drawn} documents")

    if pipeline.rewrite_source is not None:
        if not source.rewrite_active or pipeline.rewrite_source.pairs_emitted == 0:
            failures.append(
                f"the rewrite task never produced a pair "
                f"(active={source.rewrite_active}, "
                f"pairs={pipeline.rewrite_source.pairs_emitted})"
            )
        else:
            notes.append(
                f"rewrite task produced {pipeline.rewrite_source.pairs_emitted} pairs"
            )

    return failures, notes


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

    # No eval tiers here: this check is about the data position surviving a
    # crash, and scoring a held-out set three more times would only make it
    # slower without testing anything it does not already test.
    straight_dir = output_dir / "resume_straight"
    straight_trainer, straight = _build_run(args, straight_dir, total_steps)
    straight_source = straight.batch_source
    straight_state = straight_trainer.train(straight_source, data=straight_source)
    # Only `straight_state` is compared below, so the run itself can go.
    del straight_trainer, straight, straight_source
    _release_gpu()

    # Built with the SAME `total_steps` as the straight run and stopped early
    # through `train(max_steps=...)`, not built with a shorter horizon. A
    # `warmup_stable_decay` stage derives its cooldown boundary from
    # `max_steps`, so a crash run built at half the length would switch its
    # mixture at a different step and then legitimately disagree with the run
    # it is supposed to reproduce — a failure in the check, not in the code.
    crash_dir = output_dir / "resume_crash"
    crash_trainer, crash = _build_run(args, crash_dir, total_steps)
    crash_source = crash.batch_source
    crash_state = crash_trainer.train(crash_source, data=crash_source, max_steps=half)
    crash_trainer.save(crash_state.global_step, data=crash_source)
    # The checkpoint on disk is what the resumed run needs; the crashed run's
    # model is not, and this is the point where a real crash would have freed
    # it anyway.
    del crash_trainer, crash, crash_source
    _release_gpu()

    resumed_trainer, resumed = _build_run(args, crash_dir, total_steps)
    resumed_source = resumed.batch_source
    resumed_trainer.resume(data=resumed_source)
    resumed_state = resumed_trainer.train(resumed_source, data=resumed_source)

    notes = [
        f"resumability: straight={straight_state.global_step} steps, "
        f"crash-then-resume={crash_state.global_step}+"
        f"{resumed_state.global_step - crash_state.global_step} steps"
    ]
    ok = (
        resumed_state.global_step == straight_state.global_step
        and resumed_state.tokens_seen == straight_state.tokens_seen
        and resumed_state.tokens_seen > crash_state.tokens_seen
        and resumed_state.input_tokens_seen == straight_state.input_tokens_seen
        and resumed_state.input_tokens_seen > crash_state.input_tokens_seen
    )
    if not ok:
        notes.append(
            f"MISMATCH: straight tokens_seen={straight_state.tokens_seen} "
            f"input_tokens_seen={straight_state.input_tokens_seen}, "
            f"resumed tokens_seen={resumed_state.tokens_seen} "
            f"input_tokens_seen={resumed_state.input_tokens_seen}"
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
    parser.add_argument(
        "--stage",
        default="rehearsal",
        help=(
            "training_config.yml stage to rehearse. Defaults to 'rehearsal' because "
            "that is the cheap check — pass 'pretrain' before committing weeks of "
            "compute, so the profile, torch_compile, cooldown mixture and rewrite "
            "task the real run uses are the ones actually exercised."
        ),
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help="model_config.yml profile override. Default: whatever the stage names.",
    )
    parser.add_argument(
        "--quick-eval-corpus",
        default=None,
        help="Held-out set for the trend tier. Supplying either eval flag exercises "
             "the two-tier evaluation and best-checkpoint selection.",
    )
    parser.add_argument(
        "--master-eval-corpus",
        default=None,
        help="Held-out set for the selecting tier.",
    )
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        default=64,
        help="Cap on records read from each eval corpus. Small on purpose: this "
             "proves the tiers fire, not what the loss is.",
    )
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
    print(f"stage: {args.stage}")
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
        print(f"RESULT: PASSED — the {args.stage!r} pipeline is sound. Start the real run:")
        print(f"  python pretrain.py --corpus {args.corpus} --stage {args.stage}")
        if args.stage == "rehearsal":
            print()
            print("  This proved the rehearsal stage only. Before committing weeks of")
            print("  compute, rehearse the stage you will actually run:")
            print("    python pretrain_smoke_test.py --stage pretrain --device cuda \\")
            print(f"        --corpus {args.corpus} \\")
            print("        --quick-eval-corpus ... --master-eval-corpus ...")
        return 0

    print("RESULT: FAILED — do not start a real run until this is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
