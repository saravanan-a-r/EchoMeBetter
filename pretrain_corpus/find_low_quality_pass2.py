"""
Second low-quality pass over the pretraining corpus, along four new axes.

`find_low_quality.py` (pass 1) asked one kind of question: what are this
document's *character-class statistics*, and is its hash already in the corpus?
Alpha ratio, digit ratio, Latin ratio, stopword ratio, symbol soup, repeated
lines, plus exact and punctuation-folded duplicate hashes. That pass ran, and
moved 1,629,100 chunks (0.70%).

This pass deliberately asks nothing that pass 1 asked. Its four axes are:

  A. LAYOUT -- is this laid out as prose at all?
     A navigation menu, a Wikipedia category list, a book index, a tag cloud
     and a "Previous article / Next article" tail all have perfectly ordinary
     alpha and stopword ratios. They are not prose, and a model trained to
     *rewrite* prose should not be denoising them.
     -> nonprose_layout, nonprose_tail, no_sentence_punct, link_farm

  B. SURFACE DAMAGE -- is the text presentationally broken?
     ALL-CAPS legal boilerplate, "!!!!!!" decoration, and U+FFFD corruption
     ("Europea<?><?>s" for "Europe's"). The last of these matters more here
     than in a general LM: README's fidelity requirement means the model must
     reproduce characters exactly, and a corpus that spells an apostrophe
     three different ways teaches it that all three are correct.
     -> all_caps, spam_punctuation, replacement_chars

  C. REDUNDANCY BEYOND EXACT HASHES -- near-duplicates.
     `src/dedup.py` says, in its own words: "True MinHash/LSH would
     additionally catch documents that share most of their shingles ...
     Deferred deliberately, and called out here rather than left as a silent
     gap against what §9.1 promises." This pass closes that gap: 64-permutation
     MinHash, banded 8x8, corpus-wide.
     -> near_duplicate, low_lexical_diversity

  D. TOKENIZER FITNESS -- can our own trained tokenizer represent this?
     Every other check in either pass is a generic web-corpus heuristic. This
     one is specific to this project: run tokenizer/training/output/spm.model
     over the chunk and measure how much of it falls back to raw bytes, and
     how many tokens it costs per word. Hebrew with niqqud, Tamil, Arabic and
     CJK come out as byte soup; book indexes and ASCII tables come out at 4+
     tokens per word. Both are text this model can only learn to garble.
     -> tokenizer_byte_fallback, tokenizer_fertility

Rules that were built, measured, and then DELIBERATELY DROPPED, so that the
next person does not rebuild them:

  * markup_residue (>=3 HTML tags). Fires on documents *about* HTML and on
    fenced code blocks in Cosmopedia tutorials and SlimPajama's Stack Overflow
    content. No threshold separated leftover markup from discussed markup.
  * pg_censor_placeholder (Project Gutenberg's "<DW64>" slur placeholders).
    Real corpus damage, but it sits inside 4.7% of PG-19's otherwise excellent
    literary prose. This is a cleaning fix for clean.py, not a corpus cut.
  * control characters (zero-width space, bidi marks). 0.5% of C4 and
    SlimPajama, and the hits were ordinary good prose carrying one invisible
    character from a website editor.
  * a mojibake regex ("Ã©", "â€™"). Zero hits in every source: ftfy is
    installed and `clean.repair_mojibake` already did this job at ingest.

Per-source exemptions follow pass 1's reasoning, re-verified on fresh samples:
`permissive_code` is source code and no prose rule applies to it;
`stackexchange` carries fenced code, shell transcripts and link-dense answers
as its actual content; `gutenberg_pg19` and `ebooks` are exempt from
nonprose_layout specifically, because indented verse is short unterminated
lines and poetry is exactly what this corpus wants to keep.

Usage:
    python find_low_quality_pass2.py                # dry run, writes a report
    python find_low_quality_pass2.py --apply        # rewrite + move
"""

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
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_LOW_QUALITY_DIR = REPO_ROOT / "output" / "low_quality"
DEFAULT_SPM = REPO_ROOT.parent / "tokenizer" / "training" / "output" / "spm.model"
WORK_DIRNAME = ".low_quality_pass2_work"

RESERVED_CORES = 2

# --- source classes ---------------------------------------------------------

# Source code. Prose layout, prose surface and prose redundancy rules are all
# meaningless here; only byte-fallback and near-duplicate detection apply.
CODE_SOURCES = frozenset({"permissive_code"})

# Technical Q&A. Fenced code, shell transcripts and link-heavy answers are the
# content, not noise -- pass 1 recorded the same conclusion after its own
# false positives on this source.
QA_SOURCES = frozenset({"stackexchange"})

# Literary prose. Exempt from nonprose_layout only: indented verse is a run of
# short lines with no terminal punctuation, and reads to that rule exactly like
# a navigation menu.
VERSE_SOURCES = frozenset({"gutenberg_pg19", "ebooks"})


# --- rule primitives --------------------------------------------------------

WORD = re.compile(r"[A-Za-z']+")
TOKEN = re.compile(r"\S+")
SHINGLE_TOKEN = re.compile(r"[a-z0-9']+")
URL = re.compile(r"https?://|www\.")
SENT_END = re.compile(r"[.!?…]")
TERMINAL = ".!?…\"')]»”’:;"

# Runs of - = _ . * ~ # + | / \ < > brackets and quotes are horizontal rules,
# table borders, TOC dot leaders and scene breaks: ordinary typography, and
# 6.7% of Cosmopedia when they were not excluded.
#
# The non-ASCII dashes and leaders are easy to miss and matter for the same
# reason: an earlier cut of this rule excluded only "-", and then flagged H.G.
# Wells and Dickens, whose typesetting uses long em-dash rules
# ("————————————"). That is precisely the literary register this corpus most
# wants to keep, so they are excluded too.
#
# What survives the exclusions is decoration used as emphasis: "!!!!!!",
# "?????", "▬▬▬▬▬▬", "★★★★★".
SPAM_PUNCT = re.compile(
    "[!?]{5,}|([^\\w\\s\\-=_.*~#+|/\\\\<>()\\[\\]{}:;,'\"—–―‒−·•∙◦‥…¯])\\1{14,}"
)
# Fenced code blocks. Inside one, short unterminated lines are code, not a
# navigation menu -- Cosmopedia's synthetic tutorials and the Stack Overflow
# content inside SlimPajama are full of them.
CODE_FENCE = re.compile(r"```|~~~")
# LaTeX guard for low_lexical_diversity: arXiv papers in SlimPajama repeat
# \alpha, \beta and $ enough to look like SEO word-salad, and are not.
LATEX = re.compile(r"\\[A-Za-z]{2,}|\$")


def layout_short_fraction(text: str) -> float:
    """
    Share of the document's *characters* that live in short, unterminated lines.

    Character-weighted rather than line-weighted, and that is the whole reason
    the rule is usable: a Wikipedia article with a twelve-line table of
    contents and four lines carrying 2,500 characters of prose is prose. A
    line-weighted ratio calls it a menu and throws it away.
    """
    lines = [l for l in (l.strip() for l in text.split("\n")) if l]
    if len(lines) < 5:
        return 0.0
    total = sum(len(l) for l in lines)
    if total == 0:
        return 0.0
    short = sum(len(l) for l in lines if len(TOKEN.findall(l)) <= 6 and l[-1] not in TERMINAL)
    return short / total


def structural_reasons(text: str, source: str) -> list[str]:
    """Axis A, B and the non-MinHash half of C. Pure function of one string."""
    out: list[str] = []
    is_code = source in CODE_SOURCES
    if is_code:
        return out
    is_qa = source in QA_SOURCES

    words = WORD.findall(text)
    n_words = len(words)
    n_tokens = len(TOKEN.findall(text))

    if not is_qa:
        if (
            source not in VERSE_SOURCES
            and n_words >= 30
            and not CODE_FENCE.search(text)
            and layout_short_fraction(text) >= 0.50
        ):
            out.append("nonprose_layout")

        # A document whose last 200 characters contain no sentence terminator
        # and which ends on a lowercase letter has stopped being prose before
        # it stopped: the tail is a tag cloud, a link list, a "Previous article
        # / Next article" pair or a bibliography.
        s = text.rstrip()
        if n_words >= 200 and s and s[-1].isalpha() and s[-1].islower() and not SENT_END.search(s[-200:]):
            out.append("nonprose_tail")

        if n_tokens >= 100 and not SENT_END.search(text):
            out.append("no_sentence_punct")

        urls = len(URL.findall(text))
        if urls >= 8 and 100 * urls / max(n_words, 1) >= 5:
            out.append("link_farm")

    if n_words >= 120:
        lower = [w.lower() for w in words]
        if len(set(lower)) / len(lower) <= 0.22 and len(LATEX.findall(text)) < 10:
            out.append("low_lexical_diversity")

    if n_words >= 40 and sum(1 for w in words if len(w) >= 2 and w.isupper()) / n_words >= 0.50:
        out.append("all_caps")

    if text.count("\ufffd") >= 3:
        out.append("replacement_chars")

    if SPAM_PUNCT.search(text):
        out.append("spam_punctuation")

    return out


# --- MinHash / LSH ----------------------------------------------------------
#
# 64 permutations, banded into 8 bands of 8 rows. The S-curve threshold is
# (1/8)^(1/8) ~= 0.77 Jaccard. Detection probability is 0.99 at J=0.90, 0.77 at
# J=0.80, 0.38 at J=0.70 and 0.03 at J=0.50 -- deliberately biased toward
# precision, because this corpus is over-provisioned and a false near-duplicate
# throws away good text.
#
# Only the 8 band hashes are stored, never the 64-value signature: 8 uint64 per
# chunk is ~15GB across the corpus, where the full signatures would be ~118GB.

MINHASH_PERMS = 64
LSH_BANDS = 8
LSH_ROWS = MINHASH_PERMS // LSH_BANDS
SHINGLE_N = 5
MIN_SHINGLES = 20  # below this a sketch is too noisy to trust

_RNG = np.random.default_rng(0x5EED)
_PERM_A = (_RNG.integers(1, 2**63, size=MINHASH_PERMS, dtype=np.uint64) | np.uint64(1))
_PERM_B = _RNG.integers(0, 2**63, size=MINHASH_PERMS, dtype=np.uint64)
_BAND_MIX = _RNG.integers(1, 2**63, size=(LSH_BANDS, LSH_ROWS), dtype=np.uint64) | np.uint64(1)

# Five odd 64-bit constants, one per position in the shingle, so that a 5-gram
# hash depends on word order.
_GRAM_MUL = np.array(
    [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5],
    dtype=np.uint64,
)

_NO_BANDS = np.zeros(LSH_BANDS, dtype=np.uint64)


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def band_hashes(text: str) -> np.ndarray:
    """
    8 band hashes for one chunk, or 8 zeros if it is too short to sketch.

    Word hashing uses Python's built-in `hash()` over interned strings, which
    is fast and -- with PYTHONHASHSEED pinned in `main()` and workers created
    by fork -- identical in every process of one run. It is not stable across
    runs, and does not need to be: signatures are computed and compared inside
    a single invocation.
    """
    words = SHINGLE_TOKEN.findall(text.lower())
    if len(words) < SHINGLE_N + MIN_SHINGLES - 1:
        return _NO_BANDS
    h = np.fromiter((hash(w) & 0xFFFFFFFFFFFFFFFF for w in words), dtype=np.uint64, count=len(words))
    grams = (
        h[:-4] * _GRAM_MUL[0]
        ^ h[1:-3] * _GRAM_MUL[1]
        ^ h[2:-2] * _GRAM_MUL[2]
        ^ h[3:-1] * _GRAM_MUL[3]
        ^ h[4:] * _GRAM_MUL[4]
    )
    grams = np.unique(grams)
    if len(grams) < MIN_SHINGLES:
        return _NO_BANDS
    sig = np.min(_PERM_A[:, None] * grams[None, :] + _PERM_B[:, None], axis=1)
    bands = _splitmix64((sig.reshape(LSH_BANDS, LSH_ROWS) * _BAND_MIX).sum(axis=1))
    # 0 is the "no sketch" sentinel; nudge the astronomically unlikely real 0.
    return np.where(bands == 0, np.uint64(1), bands)


# --- bit layout -------------------------------------------------------------

REASON_BITS = {
    "corrupt_or_empty": 0,
    "nonprose_layout": 1,
    "nonprose_tail": 2,
    "no_sentence_punct": 3,
    "link_farm": 4,
    "low_lexical_diversity": 5,
    "all_caps": 6,
    "replacement_chars": 7,
    "spam_punctuation": 8,
    "tokenizer_byte_fallback": 9,
    "tokenizer_fertility": 10,
    "near_duplicate": 11,
}
CORRUPT_BIT = REASON_BITS["corrupt_or_empty"]
NEAR_DUP_BIT = REASON_BITS["near_duplicate"]
BIT_TO_REASON = {v: k for k, v in REASON_BITS.items()}

BITS_DTYPE = np.dtype("<u4")
BAND_DTYPE = np.dtype("<u8")

LOC_FILE_SHIFT = 40
LOC_LINE_MASK = (1 << LOC_FILE_SHIFT) - 1


def bits_to_reasons(bits: int) -> list[str]:
    return [BIT_TO_REASON[i] for i in range(32) if bits & (1 << i) and i in BIT_TO_REASON]


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


# --- Phase 1: per-file rule bits + MinHash bands ----------------------------

_WORK_DIR: Path | None = None
_RESUME = False
_SP = None
_BF_LO = _BF_HI = 0
_BF_WHITESPACE: frozenset[int] = frozenset()

# Byte-fallback thresholds. Every source's p99 is 0.0 once whitespace bytes are
# excluded, so 0.02 sits far out in the tail; source files legitimately carry
# some non-Latin string literals, so code gets 0.05.
BYTE_FALLBACK_THRESHOLD = 0.02
BYTE_FALLBACK_THRESHOLD_CODE = 0.05
# Prose sits at 1.3-1.5 tokens per word (p99 <= 2.8). 4.0 is book indexes and
# ASCII tables. Code and Q&A are exempt: their medians are 4.2 and 1.5 with a
# p90 of 7.7 and 3.1, so no single prose threshold means anything there.
FERTILITY_THRESHOLD = 4.0

_BATCH = 512


def _init_phase1(work_dir: str, spm_path: str, resume: bool) -> None:
    global _WORK_DIR, _RESUME, _SP, _BF_LO, _BF_HI, _BF_WHITESPACE
    import sentencepiece as spm

    _WORK_DIR = Path(work_dir)
    _RESUME = resume
    _SP = spm.SentencePieceProcessor(model_file=spm_path)
    byte_ids = [i for i in range(_SP.get_piece_size()) if re.fullmatch(r"<0x[0-9A-F]{2}>", _SP.id_to_piece(i))]
    _BF_LO, _BF_HI = min(byte_ids), max(byte_ids)
    # \n and \t dominate byte fallback in every source (79% of PG-19 chunks
    # contain one) and say nothing about quality. Only non-whitespace fallback
    # means "the tokenizer has no piece for this script".
    _BF_WHITESPACE = frozenset(
        i for i in byte_ids if _SP.id_to_piece(i) in ("<0x0A>", "<0x09>", "<0x0D>", "<0x20>")
    )


def _side_file_lines(work_dir: Path, file_idx: int) -> int | None:
    """
    Line count implied by an existing set of side files, or None if they are
    absent or inconsistent.

    Phase 1 is the expensive phase (80 minutes on this corpus) and a worker
    killed mid-file leaves a short or missing band file behind. The nine files
    for one input must agree on their implied line count before any of them is
    trusted; otherwise the input is recomputed from scratch.
    """
    bits_path = work_dir / f"{file_idx:04d}.bits.bin"
    if not bits_path.exists():
        return None
    size = bits_path.stat().st_size
    if size % BITS_DTYPE.itemsize:
        return None
    n = size // BITS_DTYPE.itemsize
    for b in range(LSH_BANDS):
        band_path = work_dir / f"{file_idx:04d}.band{b}.bin"
        if not band_path.exists() or band_path.stat().st_size != n * BAND_DTYPE.itemsize:
            return None
    return n


def phase1_file(args: tuple[int, str]) -> dict:
    file_idx, path_str = args
    path = Path(path_str)
    source = path.parent.name
    bits_path = _WORK_DIR / f"{file_idx:04d}.bits.bin"
    band_paths = [_WORK_DIR / f"{file_idx:04d}.band{b}.bin" for b in range(LSH_BANDS)]

    if _RESUME:
        cached = _side_file_lines(_WORK_DIR, file_idx)
        if cached is not None:
            return {
                "file_idx": file_idx,
                "file": str(path),
                "source": source,
                "lines_seen": cached,
                "bits_path": str(bits_path),
                "band_paths": [str(p) for p in band_paths],
                "resumed": True,
            }

    is_code = source in CODE_SOURCES
    is_qa = source in QA_SOURCES
    bf_threshold = BYTE_FALLBACK_THRESHOLD_CODE if is_code else BYTE_FALLBACK_THRESHOLD
    check_fertility = not is_code and not is_qa

    # Accumulated one numpy array per batch, not one per line: a 1M-line file
    # would otherwise hold a million small arrays in a worker at once.
    bits_chunks: list[np.ndarray] = []
    band_chunks: list[np.ndarray] = []
    lines_seen = 0

    def flush(texts: list[str | None]) -> None:
        """Score one batch: sentencepiece is ~75 MB/s/core when batched."""
        encodable = [t for t in texts if t is not None]
        encoded = _SP.encode(encodable) if encodable else []
        it = iter(encoded)
        bits_batch = np.zeros(len(texts), dtype=BITS_DTYPE)
        band_batch = np.zeros((len(texts), LSH_BANDS), dtype=BAND_DTYPE)
        for i, text in enumerate(texts):
            if text is None:
                bits_batch[i] = 1 << CORRUPT_BIT
                continue
            ids = next(it)
            bits = 0
            for reason in structural_reasons(text, source):
                bits |= 1 << REASON_BITS[reason]
            if ids:
                fallback = sum(1 for x in ids if _BF_LO <= x <= _BF_HI and x not in _BF_WHITESPACE)
                if fallback / len(ids) >= bf_threshold:
                    bits |= 1 << REASON_BITS["tokenizer_byte_fallback"]
                if check_fertility:
                    n_words = len(WORD.findall(text))
                    if n_words >= 30 and len(ids) / n_words >= FERTILITY_THRESHOLD:
                        bits |= 1 << REASON_BITS["tokenizer_fertility"]
            bits_batch[i] = bits
            band_batch[i] = band_hashes(text)
        bits_chunks.append(bits_batch)
        band_chunks.append(band_batch)

    batch: list[str | None] = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            lines_seen += 1
            batch.append(_extract_text(raw_line.rstrip("\n").rstrip("\r")))
            if len(batch) >= _BATCH:
                flush(batch)
                batch = []
    if batch:
        flush(batch)

    bits = np.concatenate(bits_chunks) if bits_chunks else np.zeros(0, BITS_DTYPE)
    bands = np.concatenate(band_chunks) if band_chunks else np.zeros((0, LSH_BANDS), BAND_DTYPE)
    del bits_chunks, band_chunks
    bits.tofile(bits_path)
    for b, band_path in enumerate(band_paths):
        np.ascontiguousarray(bands[:, b]).tofile(band_path)

    return {
        "file_idx": file_idx,
        "file": str(path),
        "source": source,
        "lines_seen": lines_seen,
        "bits_path": str(bits_path),
        "band_paths": [str(p) for p in band_paths],
        "resumed": False,
    }


# --- Phase 2: corpus-wide near-duplicate marking ----------------------------


def phase2_near_duplicates(file_infos: list[dict], work_dir: Path) -> dict[str, np.ndarray]:
    """
    One stable sort per band. Within a band, chunks sharing a band hash are
    near-duplicates; the first occurrence in corpus order is kept and every
    later one is marked.

    Bands are processed one at a time and the locator array is built once and
    reused, so peak memory is about 5 x 8 bytes x (number of sketched chunks)
    rather than 8 bands' worth at once.
    """
    per_file_bits: dict[str, np.ndarray] = {}
    for info in file_infos:
        arr = np.fromfile(info["bits_path"], dtype=BITS_DTYPE)
        assert len(arr) == info["lines_seen"], f"{info['file']}: bits length mismatch"
        per_file_bits[info["file"]] = arr

    ordered = sorted(file_infos, key=lambda i: i["file_idx"])
    dup_mask_by_file = {info["file"]: np.zeros(info["lines_seen"], dtype=bool) for info in ordered}

    total_marked = 0
    for b in range(LSH_BANDS):
        hash_parts, loc_parts = [], []
        for info in ordered:
            col = np.fromfile(info["band_paths"][b], dtype=BAND_DTYPE)
            sketched = np.nonzero(col)[0]
            if len(sketched) == 0:
                continue
            hash_parts.append(col[sketched])
            loc_parts.append(
                (np.uint64(info["file_idx"]) << np.uint64(LOC_FILE_SHIFT)) | sketched.astype(np.uint64)
            )
        if not hash_parts:
            continue
        hashes = np.concatenate(hash_parts)
        locs = np.concatenate(loc_parts)
        del hash_parts, loc_parts

        order = np.argsort(hashes, kind="stable")
        sorted_hashes = hashes[order]
        sorted_locs = locs[order]
        del hashes, locs, order

        is_first = np.empty(len(sorted_hashes), dtype=bool)
        is_first[0] = True
        np.not_equal(sorted_hashes[1:], sorted_hashes[:-1], out=is_first[1:])
        dup_locs = sorted_locs[~is_first]
        del sorted_hashes, sorted_locs, is_first

        file_of = (dup_locs >> np.uint64(LOC_FILE_SHIFT)).astype(np.int64)
        line_of = (dup_locs & np.uint64(LOC_LINE_MASK)).astype(np.int64)
        for info in ordered:
            sel = file_of == info["file_idx"]
            if sel.any():
                dup_mask_by_file[info["file"]][line_of[sel]] = True
        total_marked += len(dup_locs)
        print(f"  band {b}: {len(dup_locs):,} band-level collisions")

    unique_marked = 0
    for info in ordered:
        mask = dup_mask_by_file[info["file"]]
        unique_marked += int(mask.sum())
        per_file_bits[info["file"]][mask] |= np.uint32(1 << NEAR_DUP_BIT)
    print(f"  {total_marked:,} band collisions -> {unique_marked:,} distinct chunks marked near-duplicate")
    return per_file_bits


# --- Phase 3: route lines ---------------------------------------------------

_FINAL_BITS: dict[str, np.ndarray] | None = None
_P3_OUTPUT_DIR: Path | None = None
_P3_LOW_QUALITY_DIR: Path | None = None
_P3_APPLY = False
_P3_SAMPLES = 8


def _init_phase3(final_bits, output_dir: str, low_quality_dir: str, apply: bool, samples: int) -> None:
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
    move_size_before = 0
    if _P3_APPLY:
        tmp_kept_path.parent.mkdir(parents=True, exist_ok=True)
        move_path.parent.mkdir(parents=True, exist_ok=True)
        kept_fh = open(tmp_kept_path, "w", encoding="utf-8")
        # Pass 1 already wrote a file at this path for most sources; append so
        # that low_quality/ ends up holding both passes' rejects, not just this
        # one's. The pre-append size is remembered so that a failed file can be
        # rolled back to exactly pass 1's content -- deleting the file would
        # destroy pass 1's rejects along with this pass's.
        move_size_before = move_path.stat().st_size if move_path.exists() else 0
        move_fh = open(move_path, "a", encoding="utf-8")

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
            for reason in bits_to_reasons(bits):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                bucket = samples.setdefault(reason, [])
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
            if tmp_kept_path.exists():
                tmp_kept_path.unlink()
            with open(move_path, "r+b") as fh:
                fh.truncate(move_size_before)

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
    parser.add_argument("--spm", default=str(DEFAULT_SPM))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--samples-per-reason", type=int, default=8)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse phase-1 side files left in the work directory by an interrupted run "
             "(each input's nine side files must agree on their implied line count)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    low_quality_dir = Path(args.low_quality_dir)
    if not Path(args.spm).exists():
        raise SystemExit(f"tokenizer model not found: {args.spm}")

    files = discover_source_files(output_dir)
    total_bytes = sum(p.stat().st_size for p in files)

    cpu_count = os.cpu_count() or 1
    workers = args.workers or max(1, cpu_count - RESERVED_CORES)
    workers = min(workers, len(files))

    mode = "APPLY (files will be rewritten)" if args.apply else "DRY RUN (no files touched)"
    print(f"Mode: {mode}")
    print(f"Files: {len(files)}   Total size: {total_bytes / 1024**3:.2f} GB")
    print(f"CPUs available: {cpu_count}   Using {workers} workers (reserving {RESERVED_CORES})")
    print(f"Tokenizer: {args.spm}")

    work_dir = output_dir / WORK_DIRNAME
    if work_dir.exists() and not args.resume:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("fork")
    started = time.time()

    print("\n=== Phase 1: rule bits + MinHash bands ===")
    file_infos: list[dict] = []
    with ctx.Pool(
        processes=workers, initializer=_init_phase1, initargs=(str(work_dir), args.spm, args.resume)
    ) as pool:
        tasks = list(enumerate(str(p) for p in files))
        for i, info in enumerate(pool.imap_unordered(phase1_file, tasks), 1):
            file_infos.append(info)
            tag = " (resumed)" if info.get("resumed") else ""
            print(
                f"  [{i}/{len(files)}] {info['source']}/{Path(info['file']).name}: "
                f"{info['lines_seen']:,} lines{tag}",
                flush=True,
            )
    n_resumed = sum(1 for i in file_infos if i.get("resumed"))
    if n_resumed:
        print(f"  reused phase-1 output for {n_resumed}/{len(files)} files")
    print(f"Phase 1 done in {time.time() - started:.0f}s")

    print(f"\n=== Phase 2: corpus-wide near-duplicates ({MINHASH_PERMS} perms, {LSH_BANDS}x{LSH_ROWS} bands) ===")
    t2 = time.time()
    final_bits = phase2_near_duplicates(file_infos, work_dir)
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
                f"{result['lines_seen']:,} lines, {result['moved']:,} moved ({pct:.2f}%){status}",
                flush=True,
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
        "pass": 2,
        "mode": "apply" if args.apply else "dry_run",
        "output_dir": str(output_dir),
        "low_quality_dir": str(low_quality_dir),
        "tokenizer": args.spm,
        "minhash": {
            "permutations": MINHASH_PERMS,
            "bands": LSH_BANDS,
            "rows_per_band": LSH_ROWS,
            "shingle_n": SHINGLE_N,
            "min_shingles": MIN_SHINGLES,
        },
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
    name = args.report_name or (
        "low_quality_pass2_apply_report.json" if args.apply else "low_quality_pass2_audit_report.json"
    )
    report_path = output_dir / name
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    # Word hashing uses Python's str hash; pin the seed so a run is
    # reproducible and every forked worker agrees.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
