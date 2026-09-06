#!/usr/bin/env python3
"""
CLI-helper vocabulary sweep: same corpus mix, many `input_sentence_size`
variants, run in parallel, compared in one report.

    python3 run_cli_vocab_experiment.py \\
        --pretrain-corpus-dir /path/to/pretrain_corpus/output/pretrain_output \\
        --workdir /path/to/experiment_workdir

Why this script exists
-----------------------
The tokenizer must be retrained so its vocabulary covers the CLI-helper
downstream task, not just general prose (architecture.md's later addition).
The corpus available for that retrain is now `pretrain_corpus/`'s 12 sources
(replacing the older 5-source `tokenizer_training_corpus_downloader/`), and
the open question is *how much text to sample for vocabulary fitting* --
prior research (VOCAB_RESEARCH_MATRICS.MD) found that, within a fixed
32,832-slot vocabulary, MORE sampled text made fertility/chars-per-token/
utilization worse, not better, and left untested whether that trend
continues below the smallest size tried (1M). This sweep re-tests that
question on the new, much larger and more CLI-relevant corpus, both above
and below that prior range.

This script changes nothing about the tokenizer pipeline itself
--------------------------------------------------------------
`corpus_reader.py`, `train_tokenization.py` and `validate_tokenizer.py` are
used exactly as they already exist, unmodified, via a real subprocess call
per variant -- the same command a human would type by hand for one size. The
only new work here is:

  1. presenting `pretrain_corpus/`'s output layout (`<source>/<source>-
     w00-00000.jsonl`) as the flat, correctly-named view
     `corpus_reader.source_label()` already expects (a filename convention
     from the older downloader), via symlinks -- no data copied, no bytes
     read;
  2. verifying, against the real corpus, that the four tiny CLI sources
     (config/cli_experiment.yaml's `blend.shares`) really do end up 100%
     included before committing to the full sweep;
  3. running one `train_tokenization.py` subprocess per size, in parallel;
  4. reading back the manifest.json each one already writes, and turning
     those into one ranked comparison table.

Every variant shares one blend (config/cli_experiment.yaml) and one fixed
eval sample size, so only `input_sentence_size` differs between rows of the
final report -- composition is a decision made once, not something this
sweep tests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import corpus_reader  # noqa: E402
from config_loader import ConfigError, load_config  # noqa: E402
from execution import Executor  # noqa: E402

DEFAULT_BASE_CONFIG = MODULE_DIR / "config" / "cli_experiment.yaml"
TRAIN_SCRIPT = MODULE_DIR / "train_tokenization.py"

# The structured CLI sources too small to trust to proportional sampling --
# config/cli_experiment.yaml gives them a share large enough to saturate
# them (100% of their eligible lines selected for training), and this
# script verifies that against the real corpus rather than assuming it.
#
# `cli_man_pages` is deliberately NOT here, despite looking like the same
# case (a tiny, structured CLI reference source). It was originally grouped
# with these three on that assumption -- wrong, measured against the real
# corpus: `corpus_reader.iter_eligible_lines` counts one eligible line per
# newline-separated segment of a record's text (the unit SentencePiece
# actually trains on), and a man page's troff-formatted text splits into
# ~29 such lines on average. That puts its true capacity at 492,863 eligible
# lines -- 29x its 16,910 record count, and bigger than this sweep's own
# smallest tested `input_sentence_size` (250,000). No blend share can force
# "100% included" there: it would require devoting more than the entire
# budget to this one source, verified empirically (every share up to 0.50
# tried, none sufficient at size=250,000). Forcing it was never coherent;
# it now takes an ordinary weighted share like `cli_helper_stack_exchange`.
FORCED_SOURCES: tuple[str, ...] = ("cli_tldr", "cli_nl2bash", "cli_cheat_sheets")

CORPUS_SUFFIXES = (".jsonl", ".json", ".txt")

DEFAULT_SIZES: tuple[int, ...] = (
    250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000,
    3_000_000, 5_000_000, 8_000_000, 12_000_000, 16_000_000,
)
DEFAULT_EVAL_SIZE = 50_000_000
DEFAULT_SEED = 42


class ExperimentError(Exception):
    """A configuration or corpus problem caught before any expensive run starts."""


# --------------------------------------------------------------------------
# Step 1: present pretrain_corpus/'s layout the way corpus_reader.py expects
# --------------------------------------------------------------------------


def build_flattened_corpus_view(pretrain_corpus_dir: Path, dest_dir: Path) -> dict[str, int]:
    """
    Symlink every source's shard files into one flat directory, renamed so
    `corpus_reader.source_label()` derives the right label.

    `pretrain_corpus/download.py`'s `BudgetWriter` names files
    `<source>/<source>-w00-00000.jsonl`. `corpus_reader.source_label()`
    strips only a trailing `_<N>mb` suffix -- the older `training_corpus/`
    downloader's convention -- so those filenames would each parse as their
    own bogus source (`c4-w00-00000` != `c4`) and every `blend.shares`
    lookup would fail identically. Renaming to `<source>_<i>mb.jsonl` (a
    symlink, no bytes copied) makes every shard of a source collapse back
    onto the configured blend key, without editing corpus_reader.py.

    Returns `{source_id: shard_count}` for the sources actually found.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    source_dirs = sorted(p for p in pretrain_corpus_dir.iterdir() if p.is_dir())
    for source_dir in source_dirs:
        source_id = source_dir.name
        # Direct children only -- NOT rglob. Every real shard `download.py`
        # writes lands directly in `<source>/`; every source that clones an
        # upstream git repo to build its dataset (cli_tldr, cli_nl2bash,
        # cli_cheat_sheets) leaves that clone at `<source>/_repo/`, and
        # Stack Exchange sources cache extracted XML at `<source>/_xml/` and
        # raw archives at `<source>/_archives/`. An `rglob("*")` here swept
        # all of those in too -- a repo's `package.json`, `requirements.txt`
        # and (worst) thousands of `.md`/other files under `_repo/` matched
        # `CORPUS_SUFFIXES` and got symlinked in as if they were training
        # shards, corrupting both the corpus (unrelated repo scaffolding
        # mixed into a source's training text) and every stat derived from
        # it (a source's measured "capacity" no longer meant "eligible
        # lines in its real shards"). Verified against the real corpus:
        # cli_nl2bash's line count was inflated by 8,841 lines (+36%) this
        # way before this fix.
        shard_files = sorted(
            p for p in source_dir.glob("*")
            if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES
        )
        if not shard_files:
            continue
        for i, shard in enumerate(shard_files):
            link_path = dest_dir / f"{source_id}_{i}mb{shard.suffix}"
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(shard.resolve())
        counts[source_id] = len(shard_files)

    if not counts:
        raise ExperimentError(
            f"no source subdirectories with {CORPUS_SUFFIXES} files found under "
            f"{pretrain_corpus_dir}"
        )
    return counts


# --------------------------------------------------------------------------
# Step 2: verify the forced sources will really be 100% included
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SaturationCheck:
    source: str
    capacity: int
    trained: int

    @property
    def saturated(self) -> bool:
        return self.capacity > 0 and self.trained >= self.capacity


def verify_forced_sources_saturated(
    corpus_dir: Path,
    config_path: Path,
    *,
    seed: int,
    smallest_size: int,
    scratch_dir: Path,
    executor: Executor | None = None,
) -> list[SaturationCheck]:
    """
    Run the real scan-and-sample step (no SentencePiece training) at the
    smallest configured size, and check every `FORCED_SOURCES` entry got
    100% of its eligible lines -- the cheapest real check available,
    against the actual corpus and the actual blend math
    (`corpus_reader.build_corpus`), rather than a hand-derived guess about
    whether config/cli_experiment.yaml's shares are "big enough".
    """
    cfg = load_config(config_path)
    files = corpus_reader.discover_corpus_files(corpus_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    info = corpus_reader.build_corpus(
        files,
        corpus_out=scratch_dir / "verify_corpus.txt",
        train_budget=smallest_size,
        seed=seed,
        max_sentence_length=cfg["corpus"]["max_sentence_length"],
        max_drop_fraction=cfg["corpus"]["max_drop_fraction"],
        blend=cfg["blend"],
        executor=executor or Executor(),
    )
    capacities: dict[str, int] = info["scan"]["per_source_lines"]
    trained: dict[str, int] = info["train_lines_per_source"]

    return [
        SaturationCheck(
            source=source,
            capacity=capacities.get(source, 0),
            trained=trained.get(source, 0),
        )
        for source in FORCED_SOURCES
    ]


# --------------------------------------------------------------------------
# Step 3: build one train_tokenization.py invocation per variant
# --------------------------------------------------------------------------


def size_label(size: int) -> str:
    if size % 1_000_000 == 0:
        return f"{size // 1_000_000}M"
    if size % 1_000 == 0:
        return f"{size // 1_000}K"
    return str(size)


@dataclass(frozen=True)
class VariantJob:
    label: str
    size: int
    output_dir: Path
    config_path: Path
    argv: list[str]


def build_variant_jobs(
    sizes: list[int],
    corpus_dir: Path,
    base_config_path: Path,
    output_root: Path,
    *,
    seed: int,
    eval_size: int,
    num_threads: int,
) -> list[VariantJob]:
    """
    One job per size. Building the argv here, separate from running it, is
    what makes this testable without spawning any subprocess: every field
    the actual run will use is inspectable first.

    Every variant gets its own copy of `base_config_path` so `num_threads`
    (sized for how many variants actually run concurrently) can differ from
    the checked-in file without one variant's config mutating another's.
    """
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    jobs: list[VariantJob] = []

    for size in sizes:
        label = size_label(size)
        variant_dir = output_root / f"variant_{label}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(base_cfg)
        cfg["trainer"]["num_threads"] = num_threads
        # The corpus scan, the sampling pass and the eval scan get the same
        # budget as SentencePiece itself. They used to be single-threaded no
        # matter what this said, which left `num_threads` describing about 30
        # seconds of a four-hour run; now it describes the whole run.
        cfg.setdefault("runtime", {})["workers"] = num_threads
        config_path = variant_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        output_dir = variant_dir / "output"
        argv = [
            sys.executable, str(TRAIN_SCRIPT),
            "--corpus", str(corpus_dir),
            "--config", str(config_path),
            "--output-dir", str(output_dir),
            "--input-sentence-size", str(size),
            "--seed", str(seed),
            "--eval-corpus", str(eval_size),
            "--workers", str(num_threads),
        ]
        jobs.append(VariantJob(
            label=label, size=size, output_dir=output_dir, config_path=config_path, argv=argv,
        ))
    return jobs


# --------------------------------------------------------------------------
# Step 4: run every variant in parallel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRun:
    job: VariantJob
    returncode: int
    elapsed_seconds: float
    log_path: Path
    log_tail: str = ""


def _run_one_job(job: VariantJob) -> VariantRun:
    started = time.time()
    log_path = job.output_dir.parent / "train.log"
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(job.argv, stdout=log_fh, stderr=subprocess.STDOUT)
    tail = ""
    if proc.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-40:])
    return VariantRun(
        job=job, returncode=proc.returncode,
        elapsed_seconds=round(time.time() - started, 1),
        log_path=log_path, log_tail=tail,
    )


def run_variants_in_parallel(jobs: list[VariantJob], max_parallel: int) -> list[VariantRun]:
    """
    A thread pool, not a process pool: each task's real work already runs
    in its own OS subprocess (`subprocess.run`), so the pool itself only
    ever blocks N threads on N `wait()` calls -- no pickling, no fork-safety
    question, and one variant's nonzero exit code cannot raise inside the
    pool or take down its siblings (subprocess.run never raises for that;
    the caller checks `.returncode` explicitly).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        return list(pool.map(_run_one_job, jobs))


# --------------------------------------------------------------------------
# Step 5: parse each variant's manifest.json into flat, comparable metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantMetrics:
    label: str
    size: int
    returncode: int
    elapsed_seconds: float
    hard_gates_passed: bool
    forced_sources_saturated: bool
    fertility_overall: float | None
    fertility_cli: float | None
    byte_fallback_rate: float | None
    chars_per_token: float | None
    vocab_utilization: float | None
    rare_pieces: int | None
    notes: tuple[str, ...] = field(default_factory=tuple)


# The Stack Exchange slice that is actually CLI-relevant (see
# pretrain_corpus's cli_helper_stack_exchange source) -- gate 4/8's
# per-slice breakdown already reports it under this label, so no new
# eval code is needed to isolate it.
CLI_EVAL_SLICE = "cli_helper_stack_exchange"


def load_variant_metrics(run: VariantRun) -> VariantMetrics:
    manifest_path = run.job.output_dir / "manifest.json"
    if run.returncode != 0 or not manifest_path.is_file():
        return VariantMetrics(
            label=run.job.label, size=run.job.size, returncode=run.returncode,
            elapsed_seconds=run.elapsed_seconds, hard_gates_passed=False,
            forced_sources_saturated=False, fertility_overall=None, fertility_cli=None,
            byte_fallback_rate=None, chars_per_token=None, vocab_utilization=None,
            rare_pieces=None,
            notes=(f"run failed (exit {run.returncode}); no manifest.json produced",),
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gates = {g["gate"]: g for g in manifest["gates"]}
    # Read each gate's own "hard" flag rather than hardcoding gate numbers:
    # stays correct even if validate_tokenizer.py's hard-gate set changes.
    hard_passed = all(g["passed"] for g in gates.values() if g["hard"])

    capacities: dict[str, int] = manifest["corpus"]["scan"]["per_source_lines"]
    trained: dict[str, int] = manifest["corpus"]["train_lines_per_source"]
    unsaturated = [
        source for source in FORCED_SOURCES
        if trained.get(source, 0) < capacities.get(source, 0)
    ]

    gate4 = gates.get(4, {}).get("metrics", {})
    gate5 = gates.get(5, {}).get("metrics", {})
    gate8 = gates.get(8, {}).get("metrics", {})
    gate10 = gates.get(10, {}).get("metrics", {})
    gate11 = gates.get(11, {}).get("metrics", {})

    notes = []
    if unsaturated:
        notes.append(f"NOT fully included: {', '.join(unsaturated)}")

    return VariantMetrics(
        label=run.job.label, size=run.job.size, returncode=run.returncode,
        elapsed_seconds=run.elapsed_seconds, hard_gates_passed=hard_passed,
        forced_sources_saturated=not unsaturated,
        fertility_overall=gate4.get("overall"),
        fertility_cli=gate4.get("per_slice", {}).get(CLI_EVAL_SLICE),
        byte_fallback_rate=gate5.get("rate"),
        chars_per_token=gate8.get("overall"),
        vocab_utilization=gate10.get("utilization"),
        rare_pieces=gate11.get("rare_pieces"),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Step 6: rank and report
# --------------------------------------------------------------------------


def is_eligible(m: VariantMetrics) -> bool:
    """
    Correctness before quality: a variant that failed a hard gate, or that
    did not actually include a source this whole sweep exists to include,
    can never win regardless of how good its other numbers look.
    """
    return m.hard_gates_passed and m.forced_sources_saturated and m.fertility_overall is not None


def rank_variants(
    metrics: list[VariantMetrics], fertility_target: tuple[float, float]
) -> list[tuple[VariantMetrics, float | None]]:
    """
    Rank-based, not a weighted sum of raw values: fertility (~1.3),
    chars-per-token (~4) and vocab utilization (~0.95) live on unrelated
    scales, and averaging their raw numbers would just be dominated by
    whichever happens to be numerically largest. Each metric instead
    contributes a *rank* among eligible variants, and the average rank
    decides the order -- lower is better.

    CLI-specific fertility counts once more explicitly (two entries in the
    criteria list) because that is this sweep's actual purpose: not just
    "is overall fertility fine", but "does the CLI-relevant slice fit".

    Returns every input variant paired with its average rank (`None` for
    disqualified ones), eligible variants sorted best-first, disqualified
    ones appended after in their original order.
    """
    eligible = [m for m in metrics if is_eligible(m)]
    if not eligible:
        return [(m, None) for m in metrics]

    lo, hi = fertility_target
    mid = (lo + hi) / 2

    def fertility_closeness(m: VariantMetrics) -> float:
        return abs(m.fertility_overall - mid)

    def cli_fertility_closeness(m: VariantMetrics) -> float:
        return abs(m.fertility_cli - mid) if m.fertility_cli is not None else float("inf")

    def chars_per_token(m: VariantMetrics) -> float:
        return m.chars_per_token if m.chars_per_token is not None else float("-inf")

    def vocab_utilization(m: VariantMetrics) -> float:
        return m.vocab_utilization if m.vocab_utilization is not None else float("-inf")

    # (key function, higher-is-better) -- CLI fertility closeness listed
    # twice: once at normal weight, once more so it counts double.
    criteria = [
        (fertility_closeness, False),
        (cli_fertility_closeness, False),
        (cli_fertility_closeness, False),
        (chars_per_token, True),
        (vocab_utilization, True),
    ]

    ranks: dict[str, list[int]] = {m.label: [] for m in eligible}
    for key_fn, higher_is_better in criteria:
        ordered = sorted(eligible, key=key_fn, reverse=higher_is_better)
        for position, m in enumerate(ordered):
            ranks[m.label].append(position + 1)

    avg_rank = {label: sum(rs) / len(rs) for label, rs in ranks.items()}
    scored = [(m, avg_rank[m.label]) for m in eligible]
    scored.sort(key=lambda pair: pair[1])

    eligible_labels = {m.label for m in eligible}
    disqualified = [(m, None) for m in metrics if m.label not in eligible_labels]
    return scored + disqualified


def build_comparison_report(
    ranked: list[tuple[VariantMetrics, float | None]],
    fertility_target: tuple[float, float],
    meta: dict,
) -> str:
    lo, hi = fertility_target
    lines = [
        "# CLI vocabulary experiment report",
        "",
        f"Generated: {meta.get('timestamp')}",
        f"Seed: {meta.get('seed')} (fixed across every variant)",
        f"Eval sample: {meta.get('eval_size'):,} lines (fixed across every variant)",
        f"Fertility target band: [{lo}, {hi}] (config/cli_experiment.yaml)",
        "",
        "## Results",
        "",
        "| size | hard gates | CLI sources included | fertility (overall) | "
        f"fertility ({CLI_EVAL_SLICE}) | byte-fallback rate | chars/token | "
        "vocab utilization | rare pieces | train+eval time (s) | avg rank |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, rank in ranked:
        lines.append(
            f"| {m.label} "
            f"| {'PASS' if m.hard_gates_passed else 'FAIL'} "
            f"| {'yes' if m.forced_sources_saturated else 'NO'} "
            f"| {m.fertility_overall} "
            f"| {m.fertility_cli} "
            f"| {m.byte_fallback_rate} "
            f"| {m.chars_per_token} "
            f"| {m.vocab_utilization} "
            f"| {m.rare_pieces} "
            f"| {m.elapsed_seconds} "
            f"| {round(rank, 2) if rank is not None else 'disqualified'} |"
        )

    winner = next((m for m, rank in ranked if rank is not None), None)
    lines += ["", "## Recommendation", ""]
    if winner is None:
        lines.append(
            "No variant passed all hard gates with every CLI source fully included -- "
            "nothing to recommend. See the notes below."
        )
    else:
        lines.append(
            f"**{winner.label}** (`input_sentence_size={winner.size}`) ranks best across "
            f"fertility, {CLI_EVAL_SLICE} fertility, chars/token and vocabulary utilization."
        )

    lines.append(
        "\nNote: Gate 4 (fertility) is expected to WARN on gutenberg_pg19/stackexchange-style "
        "text at every size -- `vocab_size` is fixed at 32,832 (frozen), and that gate's target "
        "band reflects a fixed slot-count ceiling, not a corpus-size problem "
        "(see VOCAB_RESEARCH_MATRICS.MD). Expected, not a regression."
    )

    for m, _rank in ranked:
        if m.notes:
            lines.append(f"\n- **{m.label}**: {'; '.join(m.notes)}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _fertility_target(config_path: Path) -> tuple[float, float]:
    cfg = load_config(config_path)
    v = cfg["validation"]
    return v["fertility_target_min"], v["fertility_target_max"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pretrain-corpus-dir", required=True, type=Path,
        help="pretrain_corpus/output/pretrain_output -- one subdirectory per source.",
    )
    parser.add_argument(
        "--workdir", required=True, type=Path,
        help="Where the flattened corpus view, per-variant outputs and the final report go.",
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument(
        "--sizes", type=str, default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated input_sentence_size variants.",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="How many variants train at once (default: all of them).",
    )
    parser.add_argument(
        "--cpu-cores", type=int, default=(os.cpu_count() or 4),
        help="Used only to size trainer.num_threads per variant (cores / max-parallel).",
    )
    parser.add_argument(
        "--skip-flatten", action="store_true",
        help="Reuse an existing flattened corpus view under <workdir>/flattened_corpus.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        print(f"error: --sizes must be a comma-separated list of integers, got {args.sizes!r}",
              file=sys.stderr)
        return 2
    if not sizes:
        print("error: --sizes must list at least one size", file=sys.stderr)
        return 2

    max_parallel = args.max_parallel or len(sizes)
    args.workdir.mkdir(parents=True, exist_ok=True)

    try:
        flattened = args.workdir / "flattened_corpus"
        if args.skip_flatten and flattened.is_dir():
            print(f"[1/4] reusing existing flattened corpus view at {flattened}")
        else:
            print(f"[1/4] building flattened corpus view from {args.pretrain_corpus_dir} ...")
            counts = build_flattened_corpus_view(args.pretrain_corpus_dir, flattened)
            for source, n in sorted(counts.items()):
                print(f"      {source:<28}{n:>6} file(s) symlinked")

        print(f"\n[2/4] verifying the forced CLI sources will be fully included "
              f"at the smallest size ({min(sizes):,})...")
        checks = verify_forced_sources_saturated(
            flattened, args.base_config, seed=args.seed, smallest_size=min(sizes),
            scratch_dir=args.workdir / "_verify_scratch",
            # This pre-flight does the same two full corpus passes a variant
            # does, so it gets the same core budget rather than crawling
            # through ~860GB on one core before the sweep even starts.
            executor=Executor.from_config(args.cpu_cores),
        )
    except (ExperimentError, ConfigError, corpus_reader.CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for check in checks:
        status = "OK" if check.saturated else "SHORT"
        print(f"      [{status}] {check.source}: {check.trained:,}/{check.capacity:,} lines trained")
    failed = [c for c in checks if not c.saturated]
    if failed:
        print(
            f"\nerror: {len(failed)} forced CLI source(s) would NOT be fully included even at "
            f"the smallest configured size ({min(sizes):,}). Raise their blend.shares in "
            f"{args.base_config} and try again. Refusing to launch the full sweep.",
            file=sys.stderr,
        )
        return 2

    num_threads = max(1, args.cpu_cores // max(1, max_parallel))
    print(f"\n[3/4] launching {len(sizes)} variant(s), up to {max_parallel} in parallel, "
          f"{num_threads} SentencePiece thread(s) each...")
    jobs = build_variant_jobs(
        sizes, flattened, args.base_config, args.workdir / "variants",
        seed=args.seed, eval_size=args.eval_size, num_threads=num_threads,
    )

    started = time.time()
    runs = run_variants_in_parallel(jobs, max_parallel)
    for run in runs:
        status = "OK" if run.returncode == 0 else f"FAILED (exit {run.returncode})"
        print(f"      [{status}] {run.job.label} in {run.elapsed_seconds}s")
        if run.returncode != 0:
            print(f"      -- see {run.log_path}", file=sys.stderr)

    print("\n[4/4] aggregating results...")
    metrics = [load_variant_metrics(run) for run in runs]
    fertility_target = _fertility_target(args.base_config)
    ranked = rank_variants(metrics, fertility_target)
    report = build_comparison_report(ranked, fertility_target, {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "eval_size": args.eval_size,
    })
    report_path = args.workdir / "CLI_VOCAB_EXPERIMENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport written to {report_path}")
    print(f"Total wall time: {round(time.time() - started, 1)}s")

    winner = next((m for m, rank in ranked if rank is not None), None)
    return 0 if winner is not None else 1


if __name__ == "__main__":
    sys.exit(main())
