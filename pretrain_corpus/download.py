#!/usr/bin/env python3
"""
Master entrypoint: download the pretraining corpus, all seven sources
(architecture.md §9.1), in one command.

    python download.py --gb 500
    python download.py --gb 500 --sources fineweb_edu,c4 --workers-per-source 8
    python download.py --gb 60 --max-parallel-sources 3 --workers-per-source 4

Everything each source needs -- what to read, how to clean it, whether it
gets PII-masked -- is already defined in `src/sources.py`'s `SourceSpec`
table and wired through `src/pipeline.py`. This script's only job is the
part that has to happen exactly once, in one place: split a total GB
budget across sources by blend share, reserve disjoint PII index ranges
before any worker starts, build one shared dedup index per source, launch
workers, and merge every worker's stats into one run manifest.

Concurrency model
------------------
Two independent knobs, because the box this runs on is shared with another
job (a training run already using the GPU) and is not ours to saturate
blindly:

  --max-parallel-sources  how many of the 7 sources run at once
  --workers-per-source    how many worker processes each running source gets

Total worker processes in flight is the product of the two. Defaults are
deliberately conservative (see DEFAULT_MAX_PARALLEL_SOURCES /
DEFAULT_WORKERS_PER_SOURCE below) -- raise them once you know how much of
the 42-vCPU box is actually free (`nproc`, `uptime`, `free -h`).

Stack Exchange is not sharded the same way as the six HuggingFace sources:
there is one stream per site, not one stream that can be arbitrarily
divided, so its workers each own a slice of the *site list* (already
ordered non-technical-first by `sources.order_sites`) instead of a shard
index into one dataset.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from src.dedup import Deduplicator, build_deduplicator, estimate_chunks
from src.errors import PretrainCorpusError
from src.pii_masker import PIIMasker
from src.pii_registry import AuditLog, PIIRegistry, Reservation, load_legacy_reserved
from src.pipeline import RecordPipeline
from src.quality import QualityFilter
from src.sources import (
    SourceSpec,
    check_git_dependencies,
    check_hf_dependencies,
    check_man_dependencies,
    check_stackexchange_dependencies,
    clone_git_repo,
    download_archive,
    extract_archive,
    iter_cheat_sheets,
    iter_hf_records,
    iter_man_pages,
    iter_nl2bash_pairs,
    iter_stackexchange_posts,
    iter_tldr_pages,
    list_cheat_sheets,
    list_man_pages,
    list_site_archives,
    list_tldr_pages,
    order_sites,
    resolve,
    CLI_CHEAT_REPO_URL,
    CLI_NL2BASH_REPO_URL,
    CLI_TLDR_REPO_URL,
)
from src.writer import BudgetWriter, estimate_tokens, gb_to_bytes, write_manifest

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_REGISTRY_PATH = ROOT / "data" / "pii_registry.json"

DEFAULT_WORKERS_PER_SOURCE = 4
DEFAULT_MAX_PARALLEL_SOURCES = 2

# `fork` is the right default on Linux (the deployment target): fast,
# well-tested, and what every worker pool here was originally built and
# regression-tested against. On macOS it is actively dangerous once a
# library has touched Apple's Objective-C runtime in the parent process
# (observed directly: `datasets`/`huggingface_hub` do this, likely via
# certificate/SSL handling) -- a subsequent fork() can abort the child
# outright ("may have been in progress in another thread when fork() was
# called... Crashing instead"), which is also *why* CPython's own
# `multiprocessing` module has defaulted to `spawn` on macOS since 3.8.
# Hardcoding "fork" everywhere had silently overridden that safe default;
# this restores it automatically on macOS while leaving Linux untouched.
DEFAULT_START_METHOD = "spawn" if sys.platform == "darwin" else "fork"

# Proposed split of a total GB budget across architecture.md §9.1's seven
# sources. No fixed percentage is specified in the design doc itself, only
# qualitative weight ("majority FineWeb-Edu; season with..."); this is a
# concrete starting point, not a frozen constant -- override per source with
# `--share source_id=pct` or replace entries wholesale with `--blend`.
# Stack Exchange is kept a deliberate minority (CC BY-SA license, §9.1);
# permissive code gets the smallest share since its only job is teaching
# verbatim code-span handling, not general fluency.
DEFAULT_BLEND: dict[str, float] = {
    "fineweb_edu": 0.4275,
    "c4": 0.1425,
    "cosmopedia": 0.076,
    "gutenberg_pg19": 0.114,
    "slimpajama": 0.114,
    "stackexchange": 0.0475,
    "permissive_code": 0.0285,
    # CLI-helper sources (see src/sources.py) -- all four are small relative
    # to a 500GB budget, so each will typically show up as an
    # "underfilled_sources" warning in the manifest (the same already-
    # handled path SlimPajama's ungated fallback and Cosmopedia hit), not an
    # error. Present so the tokenizer's blend config can point at real bytes
    # on disk for every one of them.
    "cli_tldr": 0.02,
    "cli_nl2bash": 0.01,
    "cli_cheat_sheets": 0.01,
    "cli_man_pages": 0.01,
}
assert abs(sum(DEFAULT_BLEND.values()) - 1.0) < 1e-9, "DEFAULT_BLEND shares must sum to 1.0"

# Sized generously per Stack Exchange worker: reservation.reserve() burns
# indices whether or not they're all used (pii_registry.py's own
# "pessimistic reservation" design), and re-running with a bigger
# --pii-block-size is cheap, so the default errs high rather than risking
# PIIMasker._Cursor.take() raising mid-run on a large site.
DEFAULT_PII_BLOCK_SIZE = 2_000_000


# --------------------------------------------------------------------------
# HuggingFace-backed sources
# --------------------------------------------------------------------------

# A `Deduplicator` holds a `multiprocessing.Array` (see dedup.py's
# `SharedBloomFilter`): raw shared-memory ctypes objects can ONLY be shared
# with a worker process at *process-bootstrap* time (a `Pool`'s
# `initializer`/`initargs`, or a `Process`'s `args`), never as a normal
# argument to `apply_async` sent to an already-running worker -- that path
# goes through the standard task-queue pickler, which raises
# "should only be shared between processes through inheritance" for exactly
# this reason. Each pool below is therefore built with the deduplicator
# passed via `initargs`, landing in this per-worker-process global, rather
# than threaded through every task's arguments.
_WORKER_DEDUP: Deduplicator | None = None


def _init_dedup_worker(deduplicator: Deduplicator) -> None:
    global _WORKER_DEDUP
    _WORKER_DEDUP = deduplicator


def _aggregate_dedup_stats(deduplicator: Deduplicator, worker_stats: list[dict]) -> dict:
    """
    Combine two different sources of truth into one report.

    `Deduplicator.seen` / `.dropped_exact` / `.dropped_normalized` are plain
    per-process Python ints (see `Deduplicator.accept` in dedup.py) -- only
    the `SharedBloomFilter`'s underlying `multiprocessing.Array` bit array
    is real cross-process shared memory. Calling `deduplicator.as_dict()`
    in the parent process after workers finish therefore reports the
    *parent's own* (always-zero, since the parent never called `.accept`)
    counters, not the true corpus-wide total -- the filtering itself is
    correct across processes, only this specific readout was wrong.

    The fix: sum the accurate drop/keep counts from each worker's own
    `RecordPipeline.report()` (each worker only ever touches its own
    shard, so per-worker counts are already correct and additive), and
    take fill/false-positive-rate from the deduplicator directly, since
    those genuinely do read the shared bit array and are correct however
    many processes called `accept`.
    """
    pipeline_reports = [w.get("pipeline_report", {}).get("pipeline", {}) for w in worker_stats]
    checked = sum(r.get("chunks_produced", 0) for r in pipeline_reports)
    dropped = sum(r.get("chunks_dropped_duplicate", 0) for r in pipeline_reports)
    return {
        "chunks_checked": checked,
        "dropped_duplicate": dropped,
        "duplicate_rate": round(dropped / checked, 5) if checked else 0.0,
        **deduplicator.filter_metrics(),
    }


def _run_hf_worker(
    *,
    spec: SourceSpec,
    source_id: str,
    worker_index: int,
    num_workers: int,
    seed: int,
    budget_bytes: int,
    output_dir: Path,
    shard_bytes: int,
    reservation: Reservation | None,
    legacy: dict[str, frozenset[str]],
    audit_path: Path | None,
) -> dict:
    masker = None
    if reservation is not None:
        audit = AuditLog(audit_path)
        with audit:
            masker = PIIMasker(reservation, legacy=legacy, audit=audit)
            return _drain_hf_worker(
                spec, source_id, worker_index, num_workers, seed, budget_bytes,
                output_dir, shard_bytes, masker,
            )
    return _drain_hf_worker(
        spec, source_id, worker_index, num_workers, seed, budget_bytes,
        output_dir, shard_bytes, masker,
    )


def _drain_hf_worker(
    spec: SourceSpec,
    source_id: str,
    worker_index: int,
    num_workers: int,
    seed: int,
    budget_bytes: int,
    output_dir: Path,
    shard_bytes: int,
    masker: PIIMasker | None,
) -> dict:
    pipeline = RecordPipeline(
        spec.pipeline,
        masker=masker,
        quality=QualityFilter(enabled=spec.pipeline.quality),
        deduplicator=_WORKER_DEDUP,
    )
    writer = BudgetWriter(
        source_id=f"{source_id}-w{worker_index:02d}",
        output_dir=output_dir,
        budget_bytes=budget_bytes,
        shard_bytes=shard_bytes,
    )
    records = iter_hf_records(spec, seed=seed, shard_index=worker_index, num_shards=num_workers)
    for raw_text in records:
        if writer.done:
            break
        for chunk in pipeline.process(raw_text):
            if writer.done:
                break
            writer.write(chunk)

    stats = writer.close()
    stats["pipeline_report"] = pipeline.report()
    return stats


def run_hf_source(
    source_id: str,
    spec: SourceSpec,
    *,
    budget_bytes: int,
    output_dir: Path,
    seed: int,
    num_workers: int,
    shard_bytes: int,
    registry: PIIRegistry | None,
    legacy: dict[str, frozenset[str]],
    pii_block_size: int,
    audit_path: Path | None,
    start_method: str = "fork",
) -> dict:
    check_hf_dependencies()

    output_dir = output_dir / source_id
    output_dir.mkdir(parents=True, exist_ok=True)

    deduplicator = build_deduplicator(estimate_chunks(budget_bytes))
    per_worker_budget = max(1, budget_bytes // num_workers)

    reservations: list[Reservation | None]
    if spec.pipeline.mask_pii and registry is not None:
        reservations = registry.reserve(
            {"email": pii_block_size, "handle": pii_block_size, "phone": pii_block_size},
            workers=num_workers,
        )
    else:
        reservations = [None] * num_workers

    jobs = []
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_dedup_worker, initargs=(deduplicator,)) as pool:
        for worker_index in range(num_workers):
            jobs.append(
                pool.apply_async(
                    _run_hf_worker,
                    kwds=dict(
                        spec=spec,
                        source_id=source_id,
                        worker_index=worker_index,
                        num_workers=num_workers,
                        seed=seed,
                        budget_bytes=per_worker_budget,
                        output_dir=output_dir,
                        shard_bytes=shard_bytes,
                        reservation=reservations[worker_index],
                        legacy=legacy,
                        audit_path=audit_path,
                    ),
                )
            )
        worker_stats = [job.get() for job in jobs]

    return {
        "source_id": source_id,
        "hf_dataset": spec.hf_dataset,
        "license": spec.license,
        "role": spec.role,
        "workers": worker_stats,
        "dedup": _aggregate_dedup_stats(deduplicator, worker_stats),
        "bytes_written": sum(w["bytes_written"] for w in worker_stats),
    }


# --------------------------------------------------------------------------
# Stack Exchange
# --------------------------------------------------------------------------


def _run_stackexchange_worker(
    *,
    spec: SourceSpec,
    sites: list[tuple[str, int]],
    source_id: str,
    worker_index: int,
    budget_bytes: int,
    output_dir: Path,
    shard_bytes: int,
    archive_dir: Path,
    xml_dir: Path,
    reservation: Reservation,
    legacy: dict[str, frozenset[str]],
    audit_path: Path | None,
) -> dict:
    audit = AuditLog(audit_path)
    with audit:
        masker = PIIMasker(reservation, legacy=legacy, audit=audit)
        pipeline = RecordPipeline(
            # `spec.pipeline`, not a hardcoded PipelineConfig(...): the
            # values happen to match today (see SOURCES["stackexchange"] in
            # sources.py), but hardcoding a second copy here means a future
            # tuning of that spec would silently stop applying to real
            # downloads -- exactly the config-drift the project's "no
            # hardcoded hyperparameter, edit the config" principle exists
            # to prevent.
            spec.pipeline,
            masker=masker,
            quality=QualityFilter(enabled=spec.pipeline.quality),
            deduplicator=_WORKER_DEDUP,
        )
        writer = BudgetWriter(
            source_id=f"{source_id}-w{worker_index:02d}",
            output_dir=output_dir,
            budget_bytes=budget_bytes,
            shard_bytes=shard_bytes,
        )
        site_errors: list[str] = []

        for archive_name, _size in sites:
            if writer.done:
                break
            site = Path(archive_name).name[: -len(".7z")]
            try:
                archive_path = download_archive(archive_name, archive_dir)
                xml_files = extract_archive(archive_path, xml_dir / site)
            except PretrainCorpusError as exc:
                site_errors.append(f"{site}: {exc}")
                continue

            for xml_path in xml_files:
                if writer.done:
                    break
                try:
                    for raw_text in iter_stackexchange_posts(
                        xml_path, mask_name=pipeline.mask_name
                    ):
                        if writer.done:
                            break
                        for chunk in pipeline.process(raw_text):
                            if writer.done:
                                break
                            writer.write(chunk)
                except Exception as exc:  # noqa: BLE001 - one bad file must not abort the worker
                    site_errors.append(f"{site}/{xml_path.name}: {exc}")

        stats = writer.close()
        stats["pipeline_report"] = pipeline.report()
        stats["site_errors"] = site_errors
        return stats


def run_stackexchange_source(
    spec: SourceSpec,
    *,
    budget_bytes: int,
    output_dir: Path,
    num_workers: int,
    shard_bytes: int,
    registry: PIIRegistry,
    legacy: dict[str, frozenset[str]],
    pii_block_size: int,
    audit_path: Path | None,
    include_meta: bool,
    include_non_english: bool,
    start_method: str = "fork",
) -> dict:
    check_stackexchange_dependencies()

    source_id = "stackexchange"
    output_dir = output_dir / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "_archives"
    xml_dir = output_dir / "_xml"

    archives = list_site_archives()
    ordered = order_sites(archives, include_meta=include_meta, include_non_english=include_non_english)

    # Round-robin the ordered list across workers rather than contiguous
    # blocks, so every worker gets a mix of tiers and an interrupted run
    # doesn't leave one worker stuck entirely on stackoverflow.com.
    site_slices: list[list[tuple[str, int]]] = [[] for _ in range(num_workers)]
    for i, site in enumerate(ordered):
        site_slices[i % num_workers].append(site)

    deduplicator = build_deduplicator(estimate_chunks(budget_bytes))
    per_worker_budget = max(1, budget_bytes // num_workers)
    reservations = registry.reserve(
        {"email": pii_block_size, "handle": pii_block_size, "phone": pii_block_size},
        workers=num_workers,
    )

    jobs = []
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_dedup_worker, initargs=(deduplicator,)) as pool:
        for worker_index in range(num_workers):
            jobs.append(
                pool.apply_async(
                    _run_stackexchange_worker,
                    kwds=dict(
                        spec=spec,
                        sites=site_slices[worker_index],
                        source_id=source_id,
                        worker_index=worker_index,
                        budget_bytes=per_worker_budget,
                        output_dir=output_dir,
                        shard_bytes=shard_bytes,
                        archive_dir=archive_dir,
                        xml_dir=xml_dir,
                        reservation=reservations[worker_index],
                        legacy=legacy,
                        audit_path=audit_path,
                    ),
                )
            )
        worker_stats = [job.get() for job in jobs]

    return {
        "source_id": source_id,
        "license": spec.license,
        "role": spec.role,
        "sites_considered": len(ordered),
        "workers": worker_stats,
        "dedup": _aggregate_dedup_stats(deduplicator, worker_stats),
        "bytes_written": sum(w["bytes_written"] for w in worker_stats),
    }


# --------------------------------------------------------------------------
# tldr-pages (CLI helper)
# --------------------------------------------------------------------------


def _run_git_tldr_worker(
    *,
    spec: SourceSpec,
    pages: list,
    source_id: str,
    worker_index: int,
    budget_bytes: int,
    output_dir: Path,
    shard_bytes: int,
) -> dict:
    pipeline = RecordPipeline(
        spec.pipeline,
        masker=None,
        quality=QualityFilter(enabled=spec.pipeline.quality),
        deduplicator=_WORKER_DEDUP,
    )
    writer = BudgetWriter(
        source_id=f"{source_id}-w{worker_index:02d}",
        output_dir=output_dir,
        budget_bytes=budget_bytes,
        shard_bytes=shard_bytes,
    )
    for raw_text in iter_tldr_pages(pages):
        if writer.done:
            break
        for chunk in pipeline.process(raw_text):
            if writer.done:
                break
            writer.write(chunk)

    stats = writer.close()
    stats["pipeline_report"] = pipeline.report()
    return stats


def run_git_tldr_source(
    spec: SourceSpec,
    *,
    budget_bytes: int,
    output_dir: Path,
    num_workers: int,
    shard_bytes: int,
    start_method: str = "fork",
) -> dict:
    check_git_dependencies()

    source_id = "cli_tldr"
    output_dir = output_dir / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = output_dir / "_repo"

    repo_dir = clone_git_repo(CLI_TLDR_REPO_URL, clone_dir)
    pages = list_tldr_pages(repo_dir)

    # Round-robin, same reasoning as Stack Exchange's site split: no natural
    # size skew to worry about here (pages are all tiny), it just spreads
    # the fixed list evenly.
    page_slices: list[list] = [[] for _ in range(num_workers)]
    for i, page in enumerate(pages):
        page_slices[i % num_workers].append(page)

    deduplicator = build_deduplicator(estimate_chunks(budget_bytes))
    per_worker_budget = max(1, budget_bytes // num_workers)

    jobs = []
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_dedup_worker, initargs=(deduplicator,)) as pool:
        for worker_index in range(num_workers):
            jobs.append(
                pool.apply_async(
                    _run_git_tldr_worker,
                    kwds=dict(
                        spec=spec,
                        pages=page_slices[worker_index],
                        source_id=source_id,
                        worker_index=worker_index,
                        budget_bytes=per_worker_budget,
                        output_dir=output_dir,
                        shard_bytes=shard_bytes,
                    ),
                )
            )
        worker_stats = [job.get() for job in jobs]

    return {
        "source_id": source_id,
        "license": spec.license,
        "role": spec.role,
        "pages_found": len(pages),
        "workers": worker_stats,
        "dedup": _aggregate_dedup_stats(deduplicator, worker_stats),
        "bytes_written": sum(w["bytes_written"] for w in worker_stats),
    }


# --------------------------------------------------------------------------
# NL2Bash
# --------------------------------------------------------------------------


def _run_git_nl2bash_worker(
    *, spec: SourceSpec, pairs: list, source_id: str, worker_index: int,
    budget_bytes: int, output_dir: Path, shard_bytes: int,
) -> dict:
    pipeline = RecordPipeline(
        spec.pipeline, masker=None,
        quality=QualityFilter(enabled=spec.pipeline.quality),
        deduplicator=_WORKER_DEDUP,
    )
    writer = BudgetWriter(
        source_id=f"{source_id}-w{worker_index:02d}", output_dir=output_dir,
        budget_bytes=budget_bytes, shard_bytes=shard_bytes,
    )
    for raw_text in pairs:
        if writer.done:
            break
        for chunk in pipeline.process(raw_text):
            if writer.done:
                break
            writer.write(chunk)
    stats = writer.close()
    stats["pipeline_report"] = pipeline.report()
    return stats


def run_git_nl2bash_source(
    spec: SourceSpec, *, budget_bytes: int, output_dir: Path,
    num_workers: int, shard_bytes: int, start_method: str = "fork",
) -> dict:
    check_git_dependencies()

    source_id = "cli_nl2bash"
    output_dir = output_dir / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = clone_git_repo(CLI_NL2BASH_REPO_URL, output_dir / "_repo")
    pairs = list(iter_nl2bash_pairs(repo_dir))

    slices: list[list] = [[] for _ in range(num_workers)]
    for i, pair in enumerate(pairs):
        slices[i % num_workers].append(pair)

    deduplicator = build_deduplicator(estimate_chunks(budget_bytes))
    per_worker_budget = max(1, budget_bytes // num_workers)

    jobs = []
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_dedup_worker, initargs=(deduplicator,)) as pool:
        for worker_index in range(num_workers):
            jobs.append(
                pool.apply_async(
                    _run_git_nl2bash_worker,
                    kwds=dict(
                        spec=spec, pairs=slices[worker_index], source_id=source_id,
                        worker_index=worker_index, budget_bytes=per_worker_budget,
                        output_dir=output_dir, shard_bytes=shard_bytes,
                    ),
                )
            )
        worker_stats = [job.get() for job in jobs]

    return {
        "source_id": source_id, "license": spec.license, "role": spec.role,
        "pairs_found": len(pairs), "workers": worker_stats,
        "dedup": _aggregate_dedup_stats(deduplicator, worker_stats),
        "bytes_written": sum(w["bytes_written"] for w in worker_stats),
    }


# --------------------------------------------------------------------------
# cheat.sh cheat sheets
# --------------------------------------------------------------------------


def _run_git_cheat_worker(
    *, spec: SourceSpec, paths: list, source_id: str, worker_index: int,
    budget_bytes: int, output_dir: Path, shard_bytes: int,
) -> dict:
    pipeline = RecordPipeline(
        spec.pipeline, masker=None,
        quality=QualityFilter(enabled=spec.pipeline.quality),
        deduplicator=_WORKER_DEDUP,
    )
    writer = BudgetWriter(
        source_id=f"{source_id}-w{worker_index:02d}", output_dir=output_dir,
        budget_bytes=budget_bytes, shard_bytes=shard_bytes,
    )
    for raw_text in iter_cheat_sheets(paths):
        if writer.done:
            break
        for chunk in pipeline.process(raw_text):
            if writer.done:
                break
            writer.write(chunk)
    stats = writer.close()
    stats["pipeline_report"] = pipeline.report()
    return stats


def run_git_cheat_source(
    spec: SourceSpec, *, budget_bytes: int, output_dir: Path,
    num_workers: int, shard_bytes: int, start_method: str = "fork",
) -> dict:
    check_git_dependencies()

    source_id = "cli_cheat_sheets"
    output_dir = output_dir / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = clone_git_repo(CLI_CHEAT_REPO_URL, output_dir / "_repo")
    paths = list_cheat_sheets(repo_dir)

    slices: list[list] = [[] for _ in range(num_workers)]
    for i, path in enumerate(paths):
        slices[i % num_workers].append(path)

    deduplicator = build_deduplicator(estimate_chunks(budget_bytes))
    per_worker_budget = max(1, budget_bytes // num_workers)

    jobs = []
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_dedup_worker, initargs=(deduplicator,)) as pool:
        for worker_index in range(num_workers):
            jobs.append(
                pool.apply_async(
                    _run_git_cheat_worker,
                    kwds=dict(
                        spec=spec, paths=slices[worker_index], source_id=source_id,
                        worker_index=worker_index, budget_bytes=per_worker_budget,
                        output_dir=output_dir, shard_bytes=shard_bytes,
                    ),
                )
            )
        worker_stats = [job.get() for job in jobs]

    return {
        "source_id": source_id, "license": spec.license, "role": spec.role,
        "files_found": len(paths), "workers": worker_stats,
        "dedup": _aggregate_dedup_stats(deduplicator, worker_stats),
        "bytes_written": sum(w["bytes_written"] for w in worker_stats),
    }


# --------------------------------------------------------------------------
# Local man pages
# --------------------------------------------------------------------------


def _run_man_pages_worker(
    *, spec: SourceSpec, pages: list, source_id: str, worker_index: int,
    budget_bytes: int, output_dir: Path, shard_bytes: int,
) -> dict:
    pipeline = RecordPipeline(
        spec.pipeline, masker=None,
        quality=QualityFilter(enabled=spec.pipeline.quality),
        deduplicator=_WORKER_DEDUP,
    )
    writer = BudgetWriter(
        source_id=f"{source_id}-w{worker_index:02d}", output_dir=output_dir,
        budget_bytes=budget_bytes, shard_bytes=shard_bytes,
    )
    for raw_text in iter_man_pages(pages):
        if writer.done:
            break
        for chunk in pipeline.process(raw_text):
            if writer.done:
                break
            writer.write(chunk)
    stats = writer.close()
    stats["pipeline_report"] = pipeline.report()
    return stats


def run_man_pages_source(
    spec: SourceSpec, *, budget_bytes: int, output_dir: Path,
    num_workers: int, shard_bytes: int, start_method: str = "fork",
) -> dict:
    check_man_dependencies()

    source_id = "cli_man_pages"
    output_dir = output_dir / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = list_man_pages()

    slices: list[list] = [[] for _ in range(num_workers)]
    for i, page in enumerate(pages):
        slices[i % num_workers].append(page)

    deduplicator = build_deduplicator(estimate_chunks(budget_bytes))
    per_worker_budget = max(1, budget_bytes // num_workers)

    jobs = []
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers, initializer=_init_dedup_worker, initargs=(deduplicator,)) as pool:
        for worker_index in range(num_workers):
            jobs.append(
                pool.apply_async(
                    _run_man_pages_worker,
                    kwds=dict(
                        spec=spec, pages=slices[worker_index], source_id=source_id,
                        worker_index=worker_index, budget_bytes=per_worker_budget,
                        output_dir=output_dir, shard_bytes=shard_bytes,
                    ),
                )
            )
        worker_stats = [job.get() for job in jobs]

    return {
        "source_id": source_id, "license": spec.license, "role": spec.role,
        "pages_found": len(pages), "workers": worker_stats,
        "dedup": _aggregate_dedup_stats(deduplicator, worker_stats),
        "bytes_written": sum(w["bytes_written"] for w in worker_stats),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _apply_stream_overrides(spec: SourceSpec, args: argparse.Namespace) -> SourceSpec:
    """
    Let the operator retune a source's streaming knobs without editing code.

    Both default to `None` meaning "leave the SourceSpec alone", so a run that
    passes neither flag behaves exactly as the table in `sources.py` says.
    """
    changes: dict = {}
    if args.shuffle_buffer is not None:
        changes["shuffle_buffer"] = args.shuffle_buffer
    if args.shuffle_input_shards is not None:
        changes["shuffle_input_shards"] = args.shuffle_input_shards
    return replace(spec, **changes) if changes else spec


def _available_memory_gb() -> float | None:
    """`MemAvailable` in GB, or `None` off Linux where the file is absent."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:  # pragma: no cover - platform dependent
        return None
    return None


def _check_memory_budget(args: argparse.Namespace) -> str | None:
    """
    Refuse to launch more workers than this machine's RAM can hold.

    Worth a preflight check rather than a comment, because the failure it
    guards against is silent and expensive: every worker streams into its own
    multi-GB buffers, so the total is `max_parallel_sources x
    workers_per_source x per-worker`, and overshooting does not raise
    `MemoryError` or slow down -- the kernel OOM killer simply SIGKILLs a
    process, usually hours in, with the traceback going nowhere and partial
    shards left behind. Worse, on a box with no swap it frequently kills some
    *other* service first, so the download appears fine while unrelated things
    on the machine die.

    Returns an error string, or `None` when the run fits.
    """
    total_workers = args.max_parallel_sources * args.workers_per_source
    projected = total_workers * args.mem_per_worker_gb
    available = _available_memory_gb()
    if available is None:
        return None

    headroom = 0.85  # leave room for page cache and everything else on the box
    if projected <= available * headroom:
        return None
    return (
        f"projected memory use is {projected:.0f} GB "
        f"({total_workers} workers x {args.mem_per_worker_gb:.1f} GB) but only "
        f"{available:.0f} GB is available.\n"
        f"  A worker's memory is dominated by --shuffle-input-shards "
        f"(~2.4 GB per concurrent reader); see src/sources.py.\n"
        f"  Fix by lowering --max-parallel-sources / --workers-per-source, or "
        f"adding swap.\n"
        f"  Re-measure with --mem-per-worker-gb, or bypass with "
        f"--skip-memory-check if you know better."
    )


def _run_one_source(source_id: str, args: argparse.Namespace, budget_bytes: int, registry: PIIRegistry, legacy: dict) -> dict:
    spec = _apply_stream_overrides(resolve(source_id), args)
    output_dir = Path(args.output_dir)
    audit_path = Path(args.pii_audit) if args.pii_audit else None

    if spec.kind == "stackexchange":
        return run_stackexchange_source(
            spec,
            budget_bytes=budget_bytes,
            output_dir=output_dir,
            num_workers=args.workers_per_source,
            shard_bytes=gb_to_bytes(args.shard_gb),
            registry=registry,
            legacy=legacy,
            pii_block_size=args.pii_block_size,
            audit_path=audit_path,
            include_meta=args.include_meta,
            include_non_english=args.include_non_english,
            start_method=args.start_method,
        )
    if spec.kind == "git_tldr":
        return run_git_tldr_source(
            spec,
            budget_bytes=budget_bytes,
            output_dir=output_dir,
            num_workers=args.workers_per_source,
            shard_bytes=gb_to_bytes(args.shard_gb),
            start_method=args.start_method,
        )
    if spec.kind == "git_nl2bash":
        return run_git_nl2bash_source(
            spec,
            budget_bytes=budget_bytes,
            output_dir=output_dir,
            num_workers=args.workers_per_source,
            shard_bytes=gb_to_bytes(args.shard_gb),
            start_method=args.start_method,
        )
    if spec.kind == "git_cheat":
        return run_git_cheat_source(
            spec,
            budget_bytes=budget_bytes,
            output_dir=output_dir,
            num_workers=args.workers_per_source,
            shard_bytes=gb_to_bytes(args.shard_gb),
            start_method=args.start_method,
        )
    if spec.kind == "man_pages":
        return run_man_pages_source(
            spec,
            budget_bytes=budget_bytes,
            output_dir=output_dir,
            num_workers=args.workers_per_source,
            shard_bytes=gb_to_bytes(args.shard_gb),
            start_method=args.start_method,
        )
    return run_hf_source(
        source_id,
        spec,
        budget_bytes=budget_bytes,
        output_dir=output_dir,
        seed=args.seed,
        num_workers=args.workers_per_source,
        shard_bytes=gb_to_bytes(args.shard_gb),
        registry=registry if spec.pipeline.mask_pii else None,
        legacy=legacy,
        pii_block_size=args.pii_block_size,
        start_method=args.start_method,
        audit_path=audit_path,
    )


def run(args: argparse.Namespace) -> int:
    total_bytes = gb_to_bytes(args.gb)
    blend = dict(DEFAULT_BLEND)
    for override in args.share or []:
        source_id, _, pct = override.partition("=")
        if source_id not in blend or not pct:
            print(f"error: bad --share {override!r}; expected source_id=percent", file=sys.stderr)
            return 2
        try:
            blend[source_id] = float(pct) / 100.0
        except ValueError:
            print(f"error: bad --share {override!r}; {pct!r} is not a number", file=sys.stderr)
            return 2

    selected = args.sources.split(",") if args.sources else list(blend)
    unknown = [s for s in selected if s not in blend]
    if unknown:
        print(f"error: unknown source(s) {unknown}; known: {sorted(blend)}", file=sys.stderr)
        return 2

    selected_share_total = sum(blend[s] for s in selected)
    per_source_bytes = {
        source_id: int(total_bytes * blend[source_id] / selected_share_total)
        for source_id in selected
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = PIIRegistry(Path(args.registry))
    legacy = load_legacy_reserved()

    print(f"Total budget: {args.gb} GB across {len(selected)} source(s)")
    for source_id in selected:
        gb = per_source_bytes[source_id] / (1024**3)
        print(f"  {source_id:<16} {blend[source_id] * 100:5.1f}%  ->  {gb:.2f} GB")
    print()

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    t0 = time.time()

    # A plain `multiprocessing.Pool` here would be a bug, not just a style
    # choice: `_run_one_source` itself opens a *second*, inner process Pool
    # (`run_hf_source`/`run_stackexchange_source`'s `--workers-per-source`
    # pool), and `Pool` workers are daemonic -- daemonic processes are
    # forbidden from starting children of their own ("AssertionError:
    # daemonic processes are not allowed to have children"), so a nested
    # Pool crashes on the very first source.
    #
    # `ProcessPoolExecutor` doesn't have that restriction (its own workers
    # are ordinary, non-daemonic processes), and it also sidesteps a second,
    # subtler hazard that a `ThreadPoolExecutor` here would introduce:
    # forking a multi-threaded process is unsafe in general (a lock held by
    # some other thread at the instant of `fork()` is inherited "stuck
    # held" in the child, with no thread left alive to ever release it --
    # exactly what CPython's fork-from-multi-threaded-process
    # DeprecationWarning is about). Each `ProcessPoolExecutor` worker here
    # is instead a freshly forked, single-threaded process by the time it
    # runs `_run_one_source` and opens its own inner Pool, so that hazard
    # never arises.
    with ProcessPoolExecutor(
        max_workers=args.max_parallel_sources, mp_context=mp.get_context(args.start_method)
    ) as source_pool:
        jobs = {
            source_id: source_pool.submit(
                _run_one_source, source_id, args, per_source_bytes[source_id], registry, legacy
            )
            for source_id in selected
        }
        underfilled: dict[str, dict] = {}
        for source_id, job in jobs.items():
            try:
                results[source_id] = job.result()
                written = results[source_id]["bytes_written"]
                requested = per_source_bytes[source_id]
                print(f"[{source_id}] done: {written / (1024**3):.2f} GB")
                # A source running dry before its budget is met is not an
                # error (`iter_hf_records`/Stack Exchange's site list just
                # end), so it raises nothing and `job.result()` returns
                # normally -- which means it would otherwise pass by
                # completely silently. That specifically matters here: the
                # SlimPajama fallback (DKYoon/SlimPajama-6B, used whenever
                # HF_TOKEN is unset) tops out around 24GB total regardless
                # of what share was requested, and Cosmopedia/permissive
                # code are far smaller than FineWeb-Edu/C4/PG19 -- any of
                # them can quietly under-fill a large requested share. This
                # is exactly the "fail loudly, never silently" gap the
                # tokenizer downloader's own BudgetWriter already guarded
                # against (a >=90%-of-target warning) and this one did not.
                if requested > 0 and written < requested * 0.9:
                    pct = 100 * written / requested
                    underfilled[source_id] = {
                        "requested_gb": round(requested / (1024**3), 3),
                        "written_gb": round(written / (1024**3), 3),
                        "pct_of_requested": round(pct, 1),
                    }
                    print(
                        f"[{source_id}] WARNING: only {pct:.1f}% of the requested "
                        f"{requested / (1024**3):.2f} GB was written -- the source "
                        f"likely ran out of data before the budget was met.",
                        file=sys.stderr,
                    )
            except Exception as exc:  # noqa: BLE001 - one source failing must not abort the others
                errors[source_id] = f"{exc}\n{traceback.format_exc()}"
                print(f"[{source_id}] FAILED: {exc}", file=sys.stderr)

    total_written = sum(r["bytes_written"] for r in results.values())
    manifest = {
        "requested_gb": args.gb,
        "elapsed_seconds": round(time.time() - t0, 1),
        "total_bytes_written": total_written,
        "total_gb_written": round(total_written / (1024**3), 3),
        "estimated_tokens": estimate_tokens(total_written),
        "blend": {s: blend[s] for s in selected},
        "sources": results,
        "failed_sources": list(errors),
        "underfilled_sources": underfilled,
        "pii_registry": registry.snapshot(),
    }
    write_manifest(output_dir / "run_manifest.json", manifest)

    print()
    print(f"Total: {manifest['total_gb_written']} GB (~{manifest['estimated_tokens']:,} tokens estimated)")
    if underfilled:
        print(
            f"NOTE: {len(underfilled)} source(s) delivered less than requested "
            f"(see 'underfilled_sources' in the manifest): {', '.join(underfilled)}",
            file=sys.stderr,
        )
    if errors:
        print(f"Failed sources: {', '.join(errors)}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gb", type=float, required=True, help="Total target output size in GB across all sources.")
    parser.add_argument("--sources", default=None, help="Comma-separated subset of sources (default: all seven).")
    parser.add_argument(
        "--share", action="append", metavar="SOURCE=PCT",
        help="Override one source's blend share, e.g. --share stackexchange=10 (repeatable).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH), help="PII registry state file.")
    parser.add_argument("--pii-audit", default=None, help="Optional path to append issued PII values for spot-checking.")
    parser.add_argument("--pii-block-size", type=int, default=DEFAULT_PII_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers-per-source", type=int, default=DEFAULT_WORKERS_PER_SOURCE,
        help="Worker processes per running source. Total processes = this x --max-parallel-sources.",
    )
    parser.add_argument(
        "--max-parallel-sources", type=int, default=DEFAULT_MAX_PARALLEL_SOURCES,
        help="How many sources run concurrently. Keep both this and --workers-per-source "
             "conservative on a shared box -- check `nproc`/`free -h` first.",
    )
    parser.add_argument("--shard-gb", type=float, default=2.0, help="Output file rotation size per shard.")
    parser.add_argument(
        "--shuffle-input-shards", type=int, default=None, metavar="N",
        help="Files each worker's shuffle buffer reads from at once (default: the "
             "per-source value in src/sources.py, currently 1). THE dominant memory "
             "knob: roughly 2.4 GB of RAM per reader per worker.",
    )
    parser.add_argument(
        "--shuffle-buffer", type=int, default=None, metavar="N",
        help="Override the per-source streaming shuffle buffer size. Affects mixing "
             "quality; barely affects memory (see --shuffle-input-shards).",
    )
    parser.add_argument(
        "--mem-per-worker-gb", type=float, default=4.0,
        help="Expected peak RAM per worker, used only by the preflight check. "
             "Measured at ~3 GB with --shuffle-input-shards 1; ~24 GB at 10.",
    )
    parser.add_argument(
        "--skip-memory-check", action="store_true",
        help="Launch even if the preflight check says the run will not fit in RAM.",
    )
    parser.add_argument("--include-meta", action="store_true", help="Include Stack Exchange meta.* sites.")
    parser.add_argument("--include-non-english", action="store_true", help="Include known non-English Stack Exchange sites.")
    parser.add_argument(
        "--start-method", default=DEFAULT_START_METHOD, choices=("fork", "spawn", "forkserver"),
        help=f"multiprocessing start method (default here: {DEFAULT_START_METHOD}, chosen "
             f"automatically by platform -- fork on Linux, matching the deployment target; "
             f"spawn on macOS, where fork can crash worker processes outright once a library "
             f"has touched Apple's Objective-C runtime in the parent). Override only if you "
             f"have a specific reason to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gb <= 0:
        print(f"error: --gb must be > 0, got {args.gb}", file=sys.stderr)
        return 2
    # Deliberately here and not in `run()`: this is a fact about the machine
    # the CLI was invoked on, not about the orchestration `run()` performs.
    # Keeping it out of `run()` also keeps that function's behaviour a
    # function of its arguments alone, so the orchestration tests do not
    # start passing or failing based on how much RAM the test box has.
    if not args.skip_memory_check:
        problem = _check_memory_budget(args)
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
