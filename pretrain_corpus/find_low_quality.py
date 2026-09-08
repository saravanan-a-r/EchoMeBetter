from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.clean import dedup_key  # noqa: E402
from src.quality import QualityThresholds, assess  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_LOW_QUALITY_DIR = REPO_ROOT / "output" / "low_quality"
WORK_DIRNAME = ".low_quality_work"

RESERVED_CORES = 2

# max_duplicate_line_ratio is never tightened: verified false positives from
# short repeated code/markup lines in ebooks (Dickens), cosmopedia (C#
# tutorial), slimpajama (Python Q&A) -- any source can carry code snippets.
#
# gutenberg_pg19 excluded from language tightening: a legitimate literary
# essay quoting Greek script (Pater on Winckelmann/Keats) false-flagged.
# stackexchange gets minimal treatment only (see MINIMAL_QUALITY_SOURCES):
# genealogy answers dense with URLs/dates/ToS-references false-flagged on
# alpha_ratio, digit_ratio, and boilerplate_phrase.
LANGUAGE_STRICT_SOURCES = frozenset({"fineweb_edu", "c4", "cosmopedia", "slimpajama", "ebooks"})

# quality filtering was disabled at download time for this source (code is
# not prose). Corpus-wide dedup still applies to it.
NO_PROSE_QUALITY_SOURCES = frozenset({"permissive_code"})

# dedup + ngram_spam only; assess()/boilerplate_phrase false-flagged real
# technical Q&A content (see module note above).
MINIMAL_QUALITY_SOURCES = frozenset({"stackexchange"})


def strict_thresholds_for(source: str) -> QualityThresholds:
    kwargs: dict = {}
    if source not in MINIMAL_QUALITY_SOURCES:
        kwargs["min_alpha_ratio"] = 0.70
        kwargs["max_digit_ratio"] = 0.20
    if source in LANGUAGE_STRICT_SOURCES:
        kwargs["min_stopword_ratio"] = 0.10
        kwargs["min_latin_ratio"] = 0.85
    return QualityThresholds(**kwargs)


_BOILERPLATE_PHRASES = tuple(
    p.lower()
    for p in (
        "all rights reserved", "terms of service", "privacy policy", "cookie policy",
        "javascript is disabled", "please enable javascript", "enable javascript to view",
        "add to cart", "add to wishlist", "sign up for our newsletter",
        "subscribe to our newsletter", "click here to continue", "click here for more",
        "404 not found", "page not found", "skip to content", "skip to main content",
        "this website uses cookies", "we use cookies", "accept cookies", "back to top",
        "continue reading", "share on facebook", "share on twitter", "follow us on",
        "download the app", "buy now",
    )
)
_WORD_RE = re.compile(r"[A-Za-z']+")


def boilerplate_phrase_hit(text: str) -> bool:
    lower = text.lower()
    hits = sum(1 for phrase in _BOILERPLATE_PHRASES if phrase in lower)
    if hits >= 2:
        return True
    return hits >= 1 and len(text) < 600


def ngram_spam(text: str, *, min_words: int = 40, ratio_threshold: float = 0.12) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < min_words:
        return False
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    if not grams:
        return False
    top_count = Counter(grams).most_common(1)[0][1]
    return (top_count / len(grams)) > ratio_threshold


REASON_BITS = {
    "strict_too_short": 0,
    "strict_too_long": 1,
    "strict_mean_word_length_low": 2,
    "strict_mean_word_length_high": 3,
    "strict_symbol_soup": 4,
    "strict_mostly_digits": 5,
    "strict_repeated_lines": 6,
    "strict_not_latin_script": 7,
    "strict_not_english_prose": 8,
    "boilerplate_phrase": 9,
    "ngram_spam": 10,
    "corrupt_or_empty": 11,
    "duplicate_corpus_wide": 12,
}
CORRUPT_BIT = REASON_BITS["corrupt_or_empty"]
DUP_BIT = REASON_BITS["duplicate_corpus_wide"]
BIT_TO_REASON = {v: k for k, v in REASON_BITS.items()}

KEY_STRUCT = struct.Struct("<HQQ")
KEY_DTYPE = np.dtype([("bits", "<u2"), ("exact", "<u8"), ("normalized", "<u8")])
assert KEY_DTYPE.itemsize == KEY_STRUCT.size

LOC_FILE_SHIFT = 40
LOC_LINE_MASK = (1 << LOC_FILE_SHIFT) - 1


def bits_to_reasons(bits: int) -> list[str]:
    return [BIT_TO_REASON[i] for i in range(16) if bits & (1 << i)]


def discover_source_files(output_dir: Path) -> list[Path]:
    files = sorted(
        p
        for p in output_dir.rglob("*.jsonl")
        if p.parent != output_dir and "low_quality" not in p.relative_to(output_dir).parts
    )
    if not files:
        raise SystemExit(f"no corpus .jsonl files found under {output_dir}")
    return files


def _extract_text(raw_line: str) -> str | None:
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


# --- Phase 1: per-file quality bits + dedup keys -> binary side-file ----

_WORK_DIR: Path | None = None


def _init_phase1(work_dir: str) -> None:
    global _WORK_DIR
    _WORK_DIR = Path(work_dir)


def phase1_file(args: tuple[int, str]) -> dict:
    file_idx, path_str = args
    path = Path(path_str)
    source = path.parent.name
    keys_path = _WORK_DIR / f"{file_idx:04d}.keys.bin"

    lines_seen = 0
    with open(path, encoding="utf-8") as fh, open(keys_path, "wb") as out:
        for raw_line in fh:
            lines_seen += 1
            text = _extract_text(raw_line.rstrip("\n").rstrip("\r"))

            if text is None:
                out.write(KEY_STRUCT.pack(1 << CORRUPT_BIT, 0, 0))
                continue

            bits = 0
            if source not in NO_PROSE_QUALITY_SOURCES | MINIMAL_QUALITY_SOURCES:
                reason = assess(text, strict_thresholds_for(source))
                if reason:
                    bits |= 1 << REASON_BITS[f"strict_{reason}"]
                if boilerplate_phrase_hit(text):
                    bits |= 1 << REASON_BITS["boilerplate_phrase"]
            if source not in NO_PROSE_QUALITY_SOURCES:
                if ngram_spam(text):
                    bits |= 1 << REASON_BITS["ngram_spam"]

            key = dedup_key(text)
            out.write(KEY_STRUCT.pack(bits, key.exact, key.normalized))

    return {"file_idx": file_idx, "file": str(path), "source": source, "lines_seen": lines_seen, "keys_path": str(keys_path)}


# --- Phase 2: exact corpus-wide dedup via sort-and-scan -----------------


def phase2_global_dedup(file_infos: list[dict]) -> dict[str, np.ndarray]:
    per_file_bits: dict[str, np.ndarray] = {}
    exact_hash_parts, exact_loc_parts = [], []
    norm_hash_parts, norm_loc_parts = [], []

    for info in file_infos:
        arr = np.fromfile(info["keys_path"], dtype=KEY_DTYPE)
        assert len(arr) == info["lines_seen"], f"{info['file']}: keys file line count mismatch"
        per_file_bits[info["file"]] = arr["bits"].copy()

        valid = (arr["bits"] & (1 << CORRUPT_BIT)) == 0
        if not valid.any():
            continue
        line_idx = np.nonzero(valid)[0].astype(np.uint64)
        loc = (np.uint64(info["file_idx"]) << np.uint64(LOC_FILE_SHIFT)) | line_idx

        exact_hash_parts.append(arr["exact"][valid])
        exact_loc_parts.append(loc)
        norm_hash_parts.append(arr["normalized"][valid])
        norm_loc_parts.append(loc)

    def duplicate_locs(hash_parts: list[np.ndarray], loc_parts: list[np.ndarray]) -> np.ndarray:
        if not hash_parts:
            return np.empty(0, dtype=np.uint64)
        hashes = np.concatenate(hash_parts)
        locs = np.concatenate(loc_parts)
        order = np.argsort(hashes, kind="stable")
        sorted_hashes = hashes[order]
        sorted_locs = locs[order]
        is_first = np.empty(len(sorted_hashes), dtype=bool)
        is_first[0] = True
        is_first[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
        return sorted_locs[~is_first]

    dup_locs = np.unique(
        np.concatenate(
            [duplicate_locs(exact_hash_parts, exact_loc_parts), duplicate_locs(norm_hash_parts, norm_loc_parts)]
        )
    )

    file_idx_of_dup = (dup_locs >> np.uint64(LOC_FILE_SHIFT)).astype(np.int64)
    line_idx_of_dup = (dup_locs & np.uint64(LOC_LINE_MASK)).astype(np.int64)

    by_idx = {info["file_idx"]: info["file"] for info in file_infos}
    for idx, file_path in by_idx.items():
        mask = file_idx_of_dup == idx
        if not mask.any():
            continue
        lines = line_idx_of_dup[mask]
        per_file_bits[file_path][lines] |= np.uint16(1 << DUP_BIT)

    return per_file_bits


# --- Phase 3: re-read raw lines, route by final bits ---------------------

_FINAL_BITS: dict[str, np.ndarray] | None = None
_P3_OUTPUT_DIR: Path | None = None
_P3_LOW_QUALITY_DIR: Path | None = None
_P3_APPLY: bool = False
_P3_SAMPLES: int = 8


def _init_phase3(final_bits: dict[str, np.ndarray], output_dir: str, low_quality_dir: str, apply: bool, samples: int) -> None:
    global _FINAL_BITS, _P3_OUTPUT_DIR, _P3_LOW_QUALITY_DIR, _P3_APPLY, _P3_SAMPLES
    _FINAL_BITS = final_bits
    _P3_OUTPUT_DIR = Path(output_dir)
    _P3_LOW_QUALITY_DIR = Path(low_quality_dir)
    _P3_APPLY = apply
    _P3_SAMPLES = samples


def phase3_file(path_str: str) -> dict:
    path = Path(path_str)
    source = path.parent.name
    bits_arr = _FINAL_BITS[path_str]
    rel = path.relative_to(_P3_OUTPUT_DIR)
    move_path = _P3_LOW_QUALITY_DIR / rel
    tmp_kept_path = path.with_suffix(path.suffix + ".kept.tmp")

    kept_fh = move_fh = None
    if _P3_APPLY:
        tmp_kept_path.parent.mkdir(parents=True, exist_ok=True)
        move_path.parent.mkdir(parents=True, exist_ok=True)
        kept_fh = open(tmp_kept_path, "w", encoding="utf-8")
        move_fh = open(move_path, "w", encoding="utf-8")

    lines_seen = kept_count = moved_count = 0
    reason_counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}

    with open(path, encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh):
            lines_seen += 1
            bits = int(bits_arr[i])
            if bits == 0:
                kept_count += 1
                if kept_fh is not None:
                    kept_fh.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            moved_count += 1
            reasons = bits_to_reasons(bits)
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
                bucket = samples.setdefault(r, [])
                if len(bucket) < _P3_SAMPLES:
                    text = _extract_text(raw_line.rstrip("\n").rstrip("\r"))
                    bucket.append((text or "")[:300])
            if move_fh is not None:
                move_fh.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")

    if kept_fh is not None:
        kept_fh.close()
    if move_fh is not None:
        move_fh.close()

    replaced = False
    error = None
    if _P3_APPLY:
        if kept_count + moved_count == lines_seen == len(bits_arr):
            os.replace(tmp_kept_path, path)
            replaced = True
        else:
            error = (
                f"line-count mismatch: kept={kept_count} moved={moved_count} "
                f"seen={lines_seen} bits_len={len(bits_arr)}; original left untouched"
            )
            if move_path.exists():
                move_path.unlink()
            if tmp_kept_path.exists():
                tmp_kept_path.unlink()

    return {
        "file": str(path),
        "source": source,
        "lines_seen": lines_seen,
        "kept": kept_count,
        "moved": moved_count,
        "reason_counts": reason_counts,
        "samples": samples,
        "replaced": replaced,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--low-quality-dir", default=str(DEFAULT_LOW_QUALITY_DIR))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--samples-per-reason", type=int, default=8)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    low_quality_dir = Path(args.low_quality_dir)
    files = discover_source_files(output_dir)
    total_bytes = sum(p.stat().st_size for p in files)

    cpu_count = os.cpu_count() or 1
    workers = args.workers or max(1, cpu_count - RESERVED_CORES)
    workers = min(workers, len(files))

    mode = "APPLY (files will be rewritten)" if args.apply else "DRY RUN (no files touched)"
    print(f"Mode: {mode}")
    print(f"Files: {len(files)}   Total size: {total_bytes / 1024**3:.2f} GB")
    print(f"CPUs available: {cpu_count}   Using {workers} workers (reserving {RESERVED_CORES})")

    work_dir = output_dir / WORK_DIRNAME
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    ctx = mp.get_context("fork")
    started = time.time()

    print("\n=== Phase 1: quality bits + dedup keys ===")
    file_infos: list[dict] = []
    with ctx.Pool(processes=workers, initializer=_init_phase1, initargs=(str(work_dir),)) as pool:
        tasks = list(enumerate(str(p) for p in files))
        for i, info in enumerate(pool.imap_unordered(phase1_file, tasks), 1):
            file_infos.append(info)
            print(f"  [{i}/{len(files)}] {info['source']}/{Path(info['file']).name}: {info['lines_seen']:,} lines")
    print(f"Phase 1 done in {time.time() - started:.0f}s")

    print("\n=== Phase 2: corpus-wide exact dedup ===")
    t2 = time.time()
    final_bits = phase2_global_dedup(file_infos)
    print(f"Phase 2 done in {time.time() - t2:.0f}s")

    print("\n=== Phase 3: route lines, {} ===".format("write + replace" if args.apply else "report only"))
    t3 = time.time()
    results: list[dict] = []
    with ctx.Pool(
        processes=workers,
        initializer=_init_phase3,
        initargs=(final_bits, str(output_dir), str(low_quality_dir), args.apply, args.samples_per_reason),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(phase3_file, [str(p) for p in files]), 1):
            results.append(result)
            pct = 100 * result["moved"] / max(result["lines_seen"], 1)
            status = "" if result["error"] is None else f"  ERROR: {result['error']}"
            print(
                f"  [{i}/{len(files)}] {result['source']}/{Path(result['file']).name}: "
                f"{result['lines_seen']:,} lines, {result['moved']:,} moved ({pct:.2f}%){status}"
            )
    print(f"Phase 3 done in {time.time() - t3:.0f}s")

    elapsed = time.time() - started

    by_source: dict[str, dict] = {}
    global_reason_counts: dict[str, int] = {}
    global_samples: dict[str, list[dict]] = {}
    errors = []
    for r in results:
        s = by_source.setdefault(
            r["source"], {"lines_seen": 0, "kept": 0, "moved": 0, "files": 0, "reason_counts": {}}
        )
        s["lines_seen"] += r["lines_seen"]
        s["kept"] += r["kept"]
        s["moved"] += r["moved"]
        s["files"] += 1
        for reason, count in r["reason_counts"].items():
            s["reason_counts"][reason] = s["reason_counts"].get(reason, 0) + count
            global_reason_counts[reason] = global_reason_counts.get(reason, 0) + count
        for reason, texts in r["samples"].items():
            bucket = global_samples.setdefault(reason, [])
            for t in texts:
                if len(bucket) < args.samples_per_reason:
                    bucket.append({"source": r["source"], "file": Path(r["file"]).name, "text": t})
        if r["error"]:
            errors.append({"file": r["file"], "error": r["error"]})

    total_lines = sum(r["lines_seen"] for r in results)
    total_moved = sum(r["moved"] for r in results)
    total_kept = sum(r["kept"] for r in results)

    print("\n=== Per source ===")
    for source, s in sorted(by_source.items(), key=lambda kv: -kv[1]["moved"]):
        pct = 100 * s["moved"] / max(s["lines_seen"], 1)
        print(f"  {source:>16}: {s['lines_seen']:>12,} lines   {s['moved']:>10,} moved ({pct:5.2f}%)")

    print("\n=== Global reasons ===")
    for reason, count in sorted(global_reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:>28}: {count:>10,}")

    print(
        f"\nTOTAL: {total_lines:,} lines seen, {total_kept:,} kept, {total_moved:,} moved "
        f"({100 * total_moved / max(total_lines, 1):.2f}%) in {elapsed:.1f}s"
    )
    if errors:
        print(f"\n{len(errors)} file(s) had a line-count mismatch and were LEFT UNTOUCHED:")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")

    report = {
        "mode": "apply" if args.apply else "dry_run",
        "output_dir": str(output_dir),
        "low_quality_dir": str(low_quality_dir),
        "workers": workers,
        "elapsed_seconds": round(elapsed, 1),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "total_kept": total_kept,
        "total_moved": total_moved,
        "by_source": by_source,
        "global_reason_counts": global_reason_counts,
        "global_samples": global_samples,
        "errors": errors,
        "per_file": [{k: v for k, v in r.items() if k != "samples"} for r in results],
    }
    report_name = "low_quality_apply_report.json" if args.apply else "low_quality_audit_report.json"
    report_path = output_dir / report_name
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
