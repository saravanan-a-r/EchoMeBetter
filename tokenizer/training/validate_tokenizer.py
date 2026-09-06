"""
The acceptance gates (DESIGN.md 9).

Hard gates (1, 2, 3, 6) fail the run with a non-zero exit. Soft gates
(4, 5, 8, 10) are measurements that inform the vocab-size decision
(DESIGN.md 5.1) and warn rather than fail. Gates 7, 9 and 11 are report-only
diagnostics for manual review.

Gate 1 -- byte-exact round-trip -- is the keystone. It is the empirical proof
that the product's "preserve URLs, code and numbers verbatim" contract holds
at the foundation, *below* the level any downstream validator can observe.

Streaming, by design
--------------------
`stream_scan` consumes an **iterator** of (text, source) pairs and keeps only
bounded state, so the gates can be measured against the entire evaluation
corpus (~100M lines / 18 GB on the real download) without holding it in
memory. Every accumulator is bounded by the vocabulary or by a fixed cap,
never by the corpus:

  * per-piece frequency  -> one counter per vocabulary id (32,832), exact
  * sequence lengths     -> fixed-size reservoir sample, plus an exact max
  * URL/link fragments   -> capped unique set (MAX_URL_FRAGMENTS)
  * failure examples     -> capped at MAX_EXAMPLES

Encoding is batched: crossing the Python/C++ boundary once per line rather
than once per batch dominates runtime at this scale.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import sentencepiece as spm

from execution import largest_remainder
from marker_escape import escape_markers, unescape_markers
from specials import BUILTIN_SPECIALS, USER_DEFINED_SYMBOLS

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
MD_LINK_RE = re.compile(r"\[[^\]\n]{0,200}\]\([^)\s\n]{1,500}\)")

# Cap on how many failing examples to keep, so a catastrophic failure
# produces a readable report instead of a gigabyte of text.
MAX_EXAMPLES = 20

# Unique URL/markdown fragments retained for Gate 6. Bounded so a 100M-line
# eval corpus cannot grow this set without limit; well above the ~861 unique
# fragments a 20k-line sample yields, so the gate stays meaningful.
MAX_URL_FRAGMENTS = 200_000

# Texts per encode/decode call. Amortizes the Python->C++ transition.
ENCODE_BATCH = 5_000

# Gate 10 (vocabulary utilization) is only meaningful once the eval corpus is
# large enough that a well-used piece would be expected to appear. Below this
# many tokens per vocabulary slot the metric measures the sample size, not
# the vocabulary, and is reported without a warning.
MIN_TOKENS_PER_VOCAB_SLOT = 100

PROGRESS_EVERY = 2_000_000


@dataclass
class GateResult:
    number: int
    name: str
    hard: bool
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gate": self.number,
            "name": self.name,
            "hard": self.hard,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": self.metrics,
            "examples": self.examples,
        }


@dataclass
class ScanStats:
    """Bounded accumulators for one streaming pass over the eval corpus."""

    n_records: int = 0
    n_roundtrip_ok: int = 0
    total_tokens: int = 0
    total_words: int = 0
    total_chars: int = 0
    total_unk: int = 0
    total_byte_tokens: int = 0
    max_tokens_per_sentence: int = 0
    roundtrip_failures: list[str] = field(default_factory=list)
    unk_failures: list[str] = field(default_factory=list)
    id_counts: list[int] = field(default_factory=list)  # index = piece id, exact
    length_reservoir: list[int] = field(default_factory=list)
    url_fragments: set[str] = field(default_factory=set)
    url_fragments_capped: bool = False
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    # Reservoir of evaluated records, saved beside the model for export_hf.py
    # and human inspection. Accumulated here rather than by the caller so the
    # eval corpus is still read exactly once when the scan is split over
    # workers -- a caller-side reservoir would need its own pass.
    sample: list[dict] = field(default_factory=list)

    @classmethod
    def merged(
        cls,
        parts: list["ScanStats"],
        *,
        seed: int,
        length_reservoir_size: int,
        sample_budget: int,
    ) -> "ScanStats":
        """
        Combine per-file scans into the single corpus-wide result the gates read.

        Sums, maxima and set unions are order-independent, so they merge
        trivially. The two reservoirs are the only interesting part: each part
        holds a uniform sample of *its own* file, so the union is drawn by
        giving each part a share proportional to how many records it actually
        saw (`largest_remainder`), then taking that many of its reservoir
        entries. That yields a uniform sample of the whole eval stream, and a
        seeded shuffle keeps which entries survive reproducible.
        """
        parts = [p for p in parts if p is not None]
        if not parts:
            return cls()

        total = cls(id_counts=[0] * len(parts[0].id_counts))
        for part in parts:
            total.n_records += part.n_records
            total.n_roundtrip_ok += part.n_roundtrip_ok
            total.total_tokens += part.total_tokens
            total.total_words += part.total_words
            total.total_chars += part.total_chars
            total.total_unk += part.total_unk
            total.total_byte_tokens += part.total_byte_tokens
            total.max_tokens_per_sentence = max(
                total.max_tokens_per_sentence, part.max_tokens_per_sentence
            )
            for i, count in enumerate(part.id_counts):
                total.id_counts[i] += count
            for source, values in part.per_source.items():
                slot = total.per_source.setdefault(
                    source,
                    {"records": 0, "tokens": 0, "words": 0, "chars": 0, "unk": 0, "byte_tokens": 0},
                )
                for key, value in values.items():
                    slot[key] = slot.get(key, 0) + value
            if not total.url_fragments_capped:
                total.url_fragments.update(part.url_fragments)
                if len(total.url_fragments) >= MAX_URL_FRAGMENTS:
                    total.url_fragments_capped = True
            total.url_fragments_capped = total.url_fragments_capped or part.url_fragments_capped
            # Examples are illustrative, not exhaustive; the cap is what keeps
            # a catastrophic failure from producing a gigabyte of report.
            for src, dst in (
                (part.roundtrip_failures, total.roundtrip_failures),
                (part.unk_failures, total.unk_failures),
            ):
                dst.extend(src[: max(0, MAX_EXAMPLES - len(dst))])

        counts = [p.n_records for p in parts]
        rng = random.Random(seed)
        total.length_reservoir = _merge_reservoirs(
            [p.length_reservoir for p in parts], counts, length_reservoir_size, rng
        )
        total.sample = _merge_reservoirs(
            [p.sample for p in parts], counts, sample_budget, rng
        )
        return total


def _merge_reservoirs(
    reservoirs: list[list], counts: list[int], capacity: int, rng: random.Random
) -> list:
    """
    Draw `capacity` items uniformly from the union of per-file reservoirs.

    Each file's reservoir is already uniform over that file, so a share
    proportional to the file's record count -- capped at what its reservoir
    actually holds -- gives a uniform sample of the whole.
    """
    if capacity <= 0:
        return []
    quotas = largest_remainder(min(capacity, sum(counts)), counts)
    merged: list = []
    for reservoir, quota in zip(reservoirs, quotas):
        take = min(quota, len(reservoir))
        if take <= 0:
            continue
        pool = list(reservoir)
        rng.shuffle(pool)
        merged.extend(pool[:take])
    return merged


def _byte_piece_ids(sp: spm.SentencePieceProcessor) -> set[int]:
    """IDs of the 256 `<0xHH>` byte-fallback pieces, if present."""
    ids = set()
    for value in range(256):
        piece_id = sp.piece_to_id(f"<0x{value:02X}>")
        if piece_id > 0 and piece_id != sp.unk_id():
            ids.add(piece_id)
    return ids


def stream_scan(
    sp: spm.SentencePieceProcessor,
    lines: Iterable[tuple[str, str]],
    byte_ids: set[int],
    *,
    length_reservoir_size: int,
    seed: int,
    progress: bool = False,
    sample_budget: int = 0,
) -> ScanStats:
    """
    One streaming pass feeding gates 1, 2, 4, 5, 6, 8, 9, 10 and 11.

    Done once rather than per-gate: encoding is the expensive part, and every
    gate that needs it needs it over the same lines.
    """
    unk_id = sp.unk_id()
    stats = ScanStats(id_counts=[0] * sp.get_piece_size())
    rng = random.Random(seed)
    sample_rng = random.Random(seed + 2)
    started = time.time()

    def flush(batch: list[tuple[str, str]]) -> None:
        if not batch:
            return
        texts = [text for text, _ in batch]
        # Escape SentencePiece's own marker character out of the raw text
        # before encoding (Marker_Encoder_Design_EDGE_CASE.md), and reverse
        # it after decoding, so gate 1 compares against the true original.
        escaped_texts = [escape_markers(text) for text in texts]
        id_lists = sp.encode(escaped_texts)
        decoded_list = [unescape_markers(d) for d in sp.decode(id_lists)]

        for (text, source), ids, decoded in zip(batch, id_lists, decoded_list):
            if decoded == text:
                stats.n_roundtrip_ok += 1
            elif len(stats.roundtrip_failures) < MAX_EXAMPLES:
                stats.roundtrip_failures.append(f"expected={text!r}\n     got={decoded!r}")

            n_tokens = len(ids)
            n_unk = 0
            n_byte = 0
            for piece_id in ids:
                stats.id_counts[piece_id] += 1
                if piece_id == unk_id:
                    n_unk += 1
                elif piece_id in byte_ids:
                    n_byte += 1
            if n_unk and len(stats.unk_failures) < MAX_EXAMPLES:
                stats.unk_failures.append(repr(text))

            n_words = len(text.split())
            n_chars = len(text)

            # Algorithm R (reservoir): each sentence's token count has equal
            # probability of surviving to the end of an unbounded stream, in
            # one pass, without knowing the stream length in advance.
            i = stats.n_records
            if i < length_reservoir_size:
                stats.length_reservoir.append(n_tokens)
            else:
                j = rng.randint(0, i)
                if j < length_reservoir_size:
                    stats.length_reservoir[j] = n_tokens

            if sample_budget > 0:
                if len(stats.sample) < sample_budget:
                    stats.sample.append({"text": text, "source": source})
                else:
                    j = sample_rng.randint(0, i)
                    if j < sample_budget:
                        stats.sample[j] = {"text": text, "source": source}

            if not stats.url_fragments_capped:
                for fragment in MD_LINK_RE.findall(text) + URL_RE.findall(text):
                    stats.url_fragments.add(fragment)
                if len(stats.url_fragments) >= MAX_URL_FRAGMENTS:
                    stats.url_fragments_capped = True

            stats.n_records += 1
            stats.total_tokens += n_tokens
            stats.total_words += n_words
            stats.total_chars += n_chars
            stats.total_unk += n_unk
            stats.total_byte_tokens += n_byte
            if n_tokens > stats.max_tokens_per_sentence:
                stats.max_tokens_per_sentence = n_tokens

            slice_stats = stats.per_source.setdefault(
                source,
                {"records": 0, "tokens": 0, "words": 0, "chars": 0, "unk": 0, "byte_tokens": 0},
            )
            slice_stats["records"] += 1
            slice_stats["tokens"] += n_tokens
            slice_stats["words"] += n_words
            slice_stats["chars"] += n_chars
            slice_stats["unk"] += n_unk
            slice_stats["byte_tokens"] += n_byte

        if progress and stats.n_records % PROGRESS_EVERY < len(batch):
            elapsed = time.time() - started
            rate = stats.n_records / elapsed if elapsed else 0.0
            print(
                f"      ...{stats.n_records:,} lines evaluated "
                f"({rate:,.0f} lines/s, {elapsed:.0f}s elapsed)",
                flush=True,
            )

    batch: list[tuple[str, str]] = []
    for text, source in lines:
        batch.append((text, source))
        if len(batch) >= ENCODE_BATCH:
            flush(batch)
            batch = []
    flush(batch)
    return stats


# One SentencePiece processor per worker process, keyed by model path. Loading
# it costs ~0.2s and a pool worker handles many files, so caching it turns a
# per-file cost into a per-worker one. A plain module global is the right scope
# here: each pool worker is its own process.
_WORKER_SP: dict[str, tuple[Any, set[int]]] = {}


def _worker_processor(model_path: str):
    cached = _WORKER_SP.get(model_path)
    if cached is None:
        sp = spm.SentencePieceProcessor(model_file=model_path)
        cached = (sp, _byte_piece_ids(sp))
        _WORKER_SP[model_path] = cached
    return cached


def _scan_eval_file(task: dict) -> ScanStats:
    """
    Encode and measure one file's share of the eval set.

    The unit of parallel work for the gates. Pure with respect to its inputs
    -- the same file, plan and quota always produce the same counters -- which
    is what lets the merged result be independent of worker count.
    """
    import corpus_reader

    sp, byte_ids = _worker_processor(task["model_path"])
    plan = corpus_reader.SplitPlan.from_dict(task["plan"])
    stats_sink = corpus_reader.CorpusStats(files=1)
    lines = corpus_reader.iter_eval_file(
        Path(task["path"]), task["file_idx"], plan, task["quota"], stats_sink
    )
    return stream_scan(
        sp,
        lines,
        byte_ids,
        length_reservoir_size=task["length_reservoir_size"],
        # Per-file seed: reservoirs must differ between files, or every file
        # would keep the same positions and the merge would not be uniform.
        seed=task["seed"] + task["file_idx"],
        sample_budget=task["sample_budget"],
    )


def scan_eval_corpus(
    model_path: Path,
    files: list[Path],
    plan,
    *,
    limit: int,
    length_reservoir_size: int,
    seed: int,
    sample_budget: int,
    executor,
    progress: bool = False,
) -> ScanStats:
    """
    Run the eval scan over the whole corpus, one file at a time, in parallel.

    This is the step that dominated a sweep's wall clock: encoding ~50M eval
    lines through SentencePiece at ~6k lines/s on a single core took over two
    hours per variant while 41 cores idled. It is embarrassingly parallel --
    each file's share of the eval set is independent -- so all it needed was
    a per-file unit of work and a merge that respects the reservoirs.
    """
    import corpus_reader

    quotas = corpus_reader.eval_file_quotas(plan, limit)
    plan_dict = plan.as_dict()
    tasks = [
        {
            "model_path": str(model_path),
            "path": str(path),
            "file_idx": idx,
            "plan": plan_dict,
            "quota": quotas[idx] if idx < len(quotas) else 0,
            "length_reservoir_size": length_reservoir_size,
            "seed": seed,
            "sample_budget": sample_budget,
        }
        for idx, path in enumerate(files)
    ]
    if progress:
        print(
            f"      scanning {len(tasks)} file(s) across {executor.workers} worker(s)...",
            flush=True,
        )
    parts = executor.map(_scan_eval_file, tasks)
    return ScanStats.merged(
        parts,
        seed=seed,
        length_reservoir_size=length_reservoir_size,
        sample_budget=sample_budget,
    )


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Nearest-rank percentile over an already-sorted list. pct in [0, 100]."""
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


# --- gates ----------------------------------------------------------------


def gate_1_roundtrip(scan: ScanStats) -> GateResult:
    n = scan.n_records
    ok = scan.n_roundtrip_ok
    failed = n - ok
    return GateResult(
        number=1,
        name="Byte-exact round-trip",
        hard=True,
        passed=failed == 0 and n > 0,
        detail=(
            f"{ok:,}/{n:,} records satisfy decode(encode(x)) == x"
            if n
            else "no eval records to test"
        ),
        metrics={"records": n, "passed": ok, "failed": failed},
        examples=scan.roundtrip_failures,
    )


def gate_2_no_unk(scan: ScanStats) -> GateResult:
    total_unk = scan.total_unk
    per_slice = {
        source: round(s["unk"] / s["tokens"], 8) if s["tokens"] else None
        for source, s in sorted(scan.per_source.items())
    }
    rate = scan.total_unk / scan.total_tokens if scan.total_tokens else 0.0
    return GateResult(
        number=2,
        name="No <unk> emitted",
        hard=True,
        passed=total_unk == 0,
        detail=f"{total_unk:,} <unk> token(s) across {scan.n_records:,} records (rate {rate:.8f})",
        metrics={"unk_tokens": total_unk, "rate": rate, "per_slice": per_slice},
        examples=scan.unk_failures,
    )


def gate_3_specials_atomic(sp: spm.SentencePieceProcessor, cfg: dict) -> GateResult:
    """
    Every reserved name must survive as a single, indivisible token.

    The 373 user-defined symbols are checked by encoding them: each must
    produce exactly one id. pad/eos/unk are *control* symbols -- SentencePiece
    deliberately never matches them in input text (that is what stops user
    input from forging a control token), so encoding their literal spelling
    correctly yields several pieces. They are therefore checked the only way
    that is meaningful: that the name resolves to the configured id and back.
    """
    trainer = cfg["trainer"]
    failures: list[str] = []

    for name in USER_DEFINED_SYMBOLS:
        ids = sp.encode(name)
        if len(ids) != 1:
            failures.append(f"{name!r} -> {len(ids)} tokens {ids}")
        elif ids[0] == sp.unk_id():
            failures.append(f"{name!r} -> <unk>")

    expected_ids = {
        "<pad>": trainer["pad_id"],
        "</s>": trainer["eos_id"],
        "<unk>": trainer["unk_id"],
    }
    for name in BUILTIN_SPECIALS:
        want = expected_ids[name]
        got = sp.piece_to_id(name)
        if got != want:
            failures.append(f"{name!r} resolves to id {got}, expected {want}")
        elif sp.id_to_piece(got) != name:
            failures.append(f"id {got} maps back to {sp.id_to_piece(got)!r}, expected {name!r}")

    n_checked = len(USER_DEFINED_SYMBOLS) + len(BUILTIN_SPECIALS)
    return GateResult(
        number=3,
        name="Reserved tokens are atomic and resolvable",
        hard=True,
        passed=not failures,
        detail=f"{n_checked - len(failures)}/{n_checked} reserved names OK",
        metrics={"checked": n_checked, "failed": len(failures)},
        examples=failures[:MAX_EXAMPLES],
    )


def gate_4_fertility(scan: ScanStats, cfg: dict) -> GateResult:
    """Tokens per whitespace-word, measured separately per corpus slice."""
    v = cfg["validation"]
    lo, hi = v["fertility_target_min"], v["fertility_target_max"]
    warn_above = v["fertility_code_warn_above"]

    per_slice = {
        source: round(s["tokens"] / s["words"], 4) if s["words"] else None
        for source, s in sorted(scan.per_source.items())
    }
    overall = round(scan.total_tokens / scan.total_words, 4) if scan.total_words else None

    warnings = []
    for source, value in per_slice.items():
        if value is None:
            continue
        if value > warn_above:
            warnings.append(f"{source} fertility {value} exceeds {warn_above}")
        elif not lo <= value <= hi:
            warnings.append(f"{source} fertility {value} outside target [{lo}, {hi}]")

    return GateResult(
        number=4,
        name="Fertility per slice",
        hard=False,
        passed=not warnings,
        detail=f"overall {overall} tokens/word; target [{lo}, {hi}]",
        metrics={"overall": overall, "per_slice": per_slice, "target": [lo, hi]},
        examples=warnings[:MAX_EXAMPLES],
    )


def gate_5_byte_fallback_rate(scan: ScanStats, cfg: dict) -> GateResult:
    threshold = cfg["validation"]["byte_fallback_rate_warn_above"]
    total = scan.total_tokens
    rate = scan.total_byte_tokens / total if total else 0.0
    per_slice = {
        source: round(s["byte_tokens"] / s["tokens"], 6) if s["tokens"] else None
        for source, s in sorted(scan.per_source.items())
    }
    return GateResult(
        number=5,
        name="Byte-fallback rate",
        hard=False,
        passed=rate <= threshold,
        detail=f"{rate:.6f} of emitted tokens are byte pieces (warn above {threshold})",
        metrics={"rate": round(rate, 8), "threshold": threshold, "per_slice": per_slice},
    )


def gate_6_url_survival(sp: spm.SentencePieceProcessor, scan: ScanStats) -> GateResult:
    """
    URLs and markdown links extracted from the real corpus must round-trip
    exactly, standalone. This is the product's preservation contract tested
    on precisely the constructs it promises to preserve.
    """
    unique = sorted(scan.url_fragments)
    failures = [
        f for f in unique
        if unescape_markers(sp.decode(sp.encode(escape_markers(f)))) != f
    ]
    capped_note = " (fragment collection hit its cap)" if scan.url_fragments_capped else ""

    return GateResult(
        number=6,
        name="URL and markdown-link survival",
        hard=True,
        # No fragments found is not a pass by omission -- it is reported so
        # the absence of evidence is visible rather than mistaken for a pass.
        passed=not failures,
        detail=(
            f"{len(unique) - len(failures):,}/{len(unique):,} unique URL/link fragments "
            f"round-trip{capped_note}"
            if unique
            else "no URL or markdown-link fragments found in eval corpus (gate vacuous)"
        ),
        metrics={
            "fragments": len(unique),
            "failed": len(failures),
            "capped": scan.url_fragments_capped,
        },
        examples=[repr(f) for f in failures[:MAX_EXAMPLES]],
    )


def gate_7_vocab_sanity(sp: spm.SentencePieceProcessor, seed: int) -> GateResult:
    """Report-only: the samples a human reviews for boilerplate domination."""
    vocab_size = sp.get_piece_size()
    top = [sp.id_to_piece(i) for i in range(min(500, vocab_size))]
    rng = random.Random(seed)
    pool = range(min(500, vocab_size), vocab_size)
    sample_ids = rng.sample(list(pool), min(500, len(pool))) if len(pool) else []
    sampled = [sp.id_to_piece(i) for i in sorted(sample_ids)]
    return GateResult(
        number=7,
        name="Vocabulary sanity (manual review)",
        hard=False,
        passed=True,
        detail=f"vocab_size={vocab_size:,}; see report for top-500 and random-500 pieces",
        metrics={"vocab_size": vocab_size, "top_500": top, "random_500": sampled},
    )


def gate_8_chars_per_token(scan: ScanStats, cfg: dict) -> GateResult:
    """
    Characters per token: how much text each token carries.

    The complement to fertility -- fertility counts whitespace-words, this
    counts Unicode codepoints, so it stays meaningful on text where
    "words" is a poor unit (code, URLs, run-on strings).
    """
    warn_below = cfg["validation"]["chars_per_token_warn_below"]
    overall = round(scan.total_chars / scan.total_tokens, 4) if scan.total_tokens else None
    per_slice = {
        source: round(s["chars"] / s["tokens"], 4) if s["tokens"] else None
        for source, s in sorted(scan.per_source.items())
    }
    warnings = [
        f"{source} chars/token {value} below {warn_below}"
        for source, value in per_slice.items()
        if value is not None and value < warn_below
    ]
    return GateResult(
        number=8,
        name="Characters per token",
        hard=False,
        passed=not warnings,
        detail=f"overall {overall} chars/token (warn below {warn_below})",
        metrics={"overall": overall, "per_slice": per_slice, "warn_below": warn_below},
        examples=warnings[:MAX_EXAMPLES],
    )


def gate_9_sequence_length(scan: ScanStats) -> GateResult:
    """
    Report-only: how long encoded sequences actually are.

    The mean alone hides the tail that decides context-window budgeting, so
    the percentiles are what matter here. P50-P99 come from the reservoir
    sample; the maximum is tracked exactly, because a reservoir can easily
    miss the single longest sentence in a large corpus.
    """
    sorted_lengths = sorted(scan.length_reservoir)
    mean = round(scan.total_tokens / scan.n_records, 4) if scan.n_records else None
    metrics = {
        "mean": mean,
        "p50": _percentile(sorted_lengths, 50),
        "p90": _percentile(sorted_lengths, 90),
        "p95": _percentile(sorted_lengths, 95),
        "p99": _percentile(sorted_lengths, 99),
        "max": scan.max_tokens_per_sentence,
        "reservoir_size": len(sorted_lengths),
        "estimated": len(sorted_lengths) < scan.n_records,
    }
    return GateResult(
        number=9,
        name="Sequence length distribution",
        hard=False,
        passed=True,
        detail=(
            f"mean {mean}, P50 {metrics['p50']}, P90 {metrics['p90']}, "
            f"P99 {metrics['p99']}, max {metrics['max']} tokens/sentence"
        ),
        metrics=metrics,
    )


def gate_10_vocab_utilization(scan: ScanStats, cfg: dict) -> GateResult:
    """
    How much of the vocabulary the eval corpus actually exercises.

    Only warns when the corpus is big enough for the answer to mean
    something: on a small eval slice a piece can be absent simply because it
    never came up, which measures the sample, not the vocabulary. Below
    MIN_TOKENS_PER_VOCAB_SLOT tokens per slot this is reported without a
    warning rather than raising a false alarm.
    """
    warn_below = cfg["validation"]["vocab_utilization_warn_below"]
    vocab_size = len(scan.id_counts)
    used = sum(1 for c in scan.id_counts if c > 0)
    dead = vocab_size - used
    utilization = round(used / vocab_size, 4) if vocab_size else None

    meaningful = scan.total_tokens >= MIN_TOKENS_PER_VOCAB_SLOT * vocab_size
    warnings = []
    if not meaningful:
        detail = (
            f"{utilization:.2%} of {vocab_size:,} pieces used -- INFORMATIONAL: the eval "
            f"corpus ({scan.total_tokens:,} tokens) is too small to judge utilization; "
            f"{MIN_TOKENS_PER_VOCAB_SLOT * vocab_size:,}+ tokens needed"
        )
    else:
        detail = f"{utilization:.2%} of {vocab_size:,} pieces used (warn below {warn_below:.2%})"
        if utilization is not None and utilization < warn_below:
            warnings.append(
                f"vocabulary utilization {utilization:.2%} below {warn_below:.2%}: "
                f"{dead:,} pieces never appeared in {scan.total_tokens:,} tokens"
            )

    return GateResult(
        number=10,
        name="Vocabulary utilization",
        hard=False,
        passed=not warnings,
        metrics={
            "vocab_size": vocab_size,
            "used": used,
            "dead": dead,
            "utilization": utilization,
            "warn_below": warn_below,
            "meaningful": meaningful,
            "tokens_scanned": scan.total_tokens,
        },
        detail=detail,
        examples=warnings,
    )


def gate_11_token_frequency(scan: ScanStats, cfg: dict) -> GateResult:
    """
    Report-only: how token usage is distributed across the vocabulary.

    Counts are exact regardless of corpus size -- there are only vocab_size
    possible ids, so this is bounded by the vocabulary, not the corpus.
    A vocabulary where a tiny fraction of pieces carries almost all traffic
    is one whose remaining slots bought little.
    """
    rare_max = cfg["validation"]["rare_token_max_count"]
    vocab_size = len(scan.id_counts)
    tokens = scan.total_tokens
    rare = sum(1 for c in scan.id_counts if 0 < c <= rare_max)

    sorted_counts = sorted(scan.id_counts, reverse=True)
    concentration = {}
    for top_pct in (1, 5, 10, 25, 50):
        n_top = max(1, round(vocab_size * top_pct / 100))
        concentration[f"top_{top_pct}pct"] = (
            round(sum(sorted_counts[:n_top]) / tokens, 4) if tokens else 0.0
        )

    return GateResult(
        number=11,
        name="Token frequency distribution",
        hard=False,
        passed=True,
        detail=(
            f"top 1% of vocab carries {concentration['top_1pct']:.1%} of tokens; "
            f"{rare:,} piece(s) used <= {rare_max} times"
        ),
        metrics={
            "rare_pieces": rare,
            "rare_threshold": rare_max,
            "concentration": concentration,
        },
    )


def run_all_gates(
    sp: spm.SentencePieceProcessor,
    lines: Iterable[tuple[str, str]] | None,
    cfg: dict,
    *,
    progress: bool = False,
    scan: ScanStats | None = None,
) -> tuple[list[GateResult], ScanStats]:
    """
    Run every gate against a stream of (text, source) pairs.

    Returns the gate results and the raw scan, so callers can report corpus
    counts without a second pass.

    `scan` lets a caller supply an already-computed scan (see
    `scan_eval_corpus`, which produces one in parallel) instead of a stream.
    The gates themselves are pure functions of the scan, so they are identical
    either way -- only how the counters were gathered differs.
    """
    v = cfg["validation"]
    byte_ids = _byte_piece_ids(sp)
    if scan is None:
        scan = stream_scan(
            sp,
            lines or [],
            byte_ids,
            length_reservoir_size=v["length_reservoir_size"],
            seed=cfg["sampling"]["seed"],
            progress=progress,
            sample_budget=v.get("eval_sample_save", 0),
        )
    results = [
        gate_1_roundtrip(scan),
        gate_2_no_unk(scan),
        gate_3_specials_atomic(sp, cfg),
        gate_4_fertility(scan, cfg),
        gate_5_byte_fallback_rate(scan, cfg),
        gate_6_url_survival(sp, scan),
        gate_7_vocab_sanity(sp, cfg["sampling"]["seed"]),
        gate_8_chars_per_token(scan, cfg),
        gate_9_sequence_length(scan),
        gate_10_vocab_utilization(scan, cfg),
        gate_11_token_frequency(scan, cfg),
    ]
    return results, scan


def hard_failures(results: Iterable[GateResult]) -> list[GateResult]:
    return [r for r in results if r.hard and not r.passed]


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value}"


def render_report(results: list[GateResult], manifest: dict) -> str:
    """Human-readable report.md for the artifact directory."""
    by_number = {r.number: r for r in results}
    corpus = manifest.get("corpus", {})
    evaluation = manifest.get("evaluation", {})
    cfg = manifest.get("config", {})

    lines = [
        "# Tokenizer evaluation report",
        "",
        f"- Built: {manifest.get('timestamp', 'unknown')}",
        f"- Model: `{manifest.get('training', {}).get('model_path', 'unknown')}`",
        f"- Vocabulary size: {cfg.get('trainer', {}).get('vocab_size', '?'):,}"
        if isinstance(cfg.get("trainer", {}).get("vocab_size"), int)
        else "- Vocabulary size: ?",
        f"- Trained on: {corpus.get('train_lines', '?'):,} lines"
        if isinstance(corpus.get("train_lines"), int)
        else "- Trained on: ? lines",
        f"- Evaluated on: {evaluation.get('lines_evaluated', '?'):,} lines "
        f"({evaluation.get('scope', 'unknown')})"
        if isinstance(evaluation.get("lines_evaluated"), int)
        else "- Evaluated on: ? lines",
        "",
        "The evaluation set is disjoint from the training set by construction:",
        "every line the sampler did not select for training is eval data.",
        "",
        "## Summary",
        "",
        "| # | Gate | Type | Result | Detail |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r.hard:
            kind, status = "hard", ("PASS" if r.passed else "**FAIL**")
        elif r.number in (7, 9, 11):
            kind, status = "report", "—"
        else:
            kind, status = "soft", ("PASS" if r.passed else "WARN")
        lines.append(f"| {r.number} | {r.name} | {kind} | {status} | {r.detail} |")

    # --- correctness ------------------------------------------------------
    g1, g2, g6 = by_number[1], by_number[2], by_number[6]
    lines += [
        "",
        "## Correctness (hard gates)",
        "",
        f"- Byte-exact round-trip: **{'PASS' if g1.passed else 'FAIL'}** — "
        f"{g1.metrics.get('failed', 0):,} failure(s) in "
        f"{g1.metrics.get('records', 0):,} records",
        f"- `<unk>` tokens emitted: **{g2.metrics.get('unk_tokens', 0):,}** "
        f"(rate {g2.metrics.get('rate', 0):.8f})",
        f"- Reserved tokens atomic: **{by_number[3].detail}**",
        f"- URL / markdown-link survival: {g6.detail}",
    ]

    # --- efficiency -------------------------------------------------------
    g4, g5, g8, g9 = by_number[4], by_number[5], by_number[8], by_number[9]
    lines += [
        "",
        "## Efficiency",
        "",
        f"- Fertility (tokens/word): **{_fmt(g4.metrics.get('overall'))}** "
        f"(target {g4.metrics.get('target')})",
        f"- Characters/token: **{_fmt(g8.metrics.get('overall'))}**",
        f"- Byte-fallback rate: **{g5.metrics.get('rate', 0):.8f}** "
        f"(warn above {g5.metrics.get('threshold')})",
        "",
        "### Sequence length (tokens per sentence)",
        "",
        "| Mean | P50 | P90 | P95 | P99 | Max |",
        "|---|---|---|---|---|---|",
        f"| {_fmt(g9.metrics.get('mean'))} | {g9.metrics.get('p50')} | "
        f"{g9.metrics.get('p90')} | {g9.metrics.get('p95')} | "
        f"{g9.metrics.get('p99')} | {g9.metrics.get('max')} |",
    ]
    if g9.metrics.get("estimated"):
        lines.append(
            f"\nP50–P99 estimated from a {g9.metrics.get('reservoir_size', 0):,}-sentence "
            f"reservoir sample; mean and max are exact."
        )

    # --- per source -------------------------------------------------------
    sources = sorted(
        set(g4.metrics.get("per_slice", {}))
        | set(g5.metrics.get("per_slice", {}))
        | set(g8.metrics.get("per_slice", {}))
    )
    if sources:
        lines += [
            "",
            "## Per source",
            "",
            "| Source | Fertility | Chars/token | Byte-fallback | UNK rate |",
            "|---|---|---|---|---|",
        ]
        for source in sources:
            lines.append(
                f"| {source} "
                f"| {_fmt(g4.metrics.get('per_slice', {}).get(source))} "
                f"| {_fmt(g8.metrics.get('per_slice', {}).get(source))} "
                f"| {_fmt(g5.metrics.get('per_slice', {}).get(source))} "
                f"| {_fmt(g2.metrics.get('per_slice', {}).get(source))} |"
            )

    # --- vocabulary -------------------------------------------------------
    g10, g11 = by_number[10], by_number[11]
    lines += [
        "",
        "## Vocabulary usage",
        "",
        f"- Pieces used: **{g10.metrics.get('used', 0):,}** of "
        f"{g10.metrics.get('vocab_size', 0):,} "
        f"({(g10.metrics.get('utilization') or 0):.2%})",
        f"- Never used: {g10.metrics.get('dead', 0):,}",
        f"- Used ≤ {g11.metrics.get('rare_threshold')} times: "
        f"{g11.metrics.get('rare_pieces', 0):,}",
    ]
    if not g10.metrics.get("meaningful", True):
        lines.append(
            "\n> Utilization is **informational** here: the eval corpus is too small for "
            "an absent piece to mean the piece is unused. Run against more data for a "
            "meaningful number."
        )
    concentration = g11.metrics.get("concentration", {})
    if concentration:
        lines += [
            "",
            "### Token frequency concentration",
            "",
            "| Top % of vocabulary | Share of all tokens |",
            "|---|---|",
        ]
        for top_pct in (1, 5, 10, 25, 50):
            value = concentration.get(f"top_{top_pct}pct", 0.0)
            lines.append(f"| {top_pct}% | {value:.2%} |")

    # --- failures ---------------------------------------------------------
    for r in results:
        if r.number in (7, 9, 11):
            continue
        if not r.passed and r.examples:
            lines += ["", f"### Gate {r.number} findings ({r.name})", "", "```"]
            lines += r.examples
            lines.append("```")

    sanity = by_number.get(7)
    if sanity:
        lines += ["", "## Gate 7 — vocabulary sample for manual review", ""]
        lines += ["First 500 pieces (includes the reserved block):", "", "```"]
        lines.append(" ".join(repr(p) for p in sanity.metrics["top_500"]))
        lines += ["```", "", "Random 500 learned pieces:", "", "```"]
        lines.append(" ".join(repr(p) for p in sanity.metrics["random_500"]))
        lines.append("```")

    return "\n".join(lines) + "\n"
