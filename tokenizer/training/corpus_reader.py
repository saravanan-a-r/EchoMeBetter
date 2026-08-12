"""
Turn a downloaded corpus (any size) into the fixed, seeded, RAM-bounded
`corpus.txt` that SentencePiece trains on, plus a disjoint held-out set for
the validation gates.

Why this module exists at all
----------------------------
SentencePieceTrainer's own `input_sentence_size` + `shuffle_input_sentence`
would do the subsampling for us, but its internal shuffle uses an unseeded
RNG, so the build would not be reproducible (DESIGN.md 7, risk R7). We
therefore do the sampling ourselves, seeded, and hand SentencePiece a fixed
file to consume in fixed order.

How the sampling works
----------------------
Two streaming passes, **O(1) RAM regardless of corpus size** -- this is what
makes a 20 GB (or 200 GB) corpus safe on a laptop:

  pass 1: count eligible lines N, gather per-source stats  -> SplitPlan
  pass 2: Algorithm S (selection sampling) picks exactly K of the N lines
          uniformly at random, writing straight to disk

Reservoir sampling would need the K sampled lines resident in memory (~5 GB
at K=20M); selection sampling needs only two counters, at the cost of one
extra sequential read.

Train / eval split, and why eval is never written to disk
---------------------------------------------------------
Every eligible line is either training data or evaluation data: the K lines
Algorithm S selects are trained on, and **everything else is the evaluation
set**. On the real corpus that is ~100M lines / ~18 GB, so materializing it
would cost as much disk as the download itself for no benefit.

Instead the split is *replayed*. `iter_split()` is the single function that
decides train-vs-eval, and it is deterministic given (files, plan, seed):
`build_corpus()` runs it to write corpus.txt, and the validation stage runs
it again afterwards to stream the eval side. Because both paths call the
same function with the same plan, they cannot drift -- there is no second
implementation of the split to keep in sync, which is exactly the bug this
structure exists to prevent.

Fidelity invariants (DESIGN.md NF1)
-----------------------------------
This module never rewrites text. It does not normalize Unicode, strip
characters, collapse whitespace or truncate. Its only transformations are
(a) selecting which records to keep, and (b) splitting records on newlines,
because SentencePiece is line-oriented and would do exactly that split
itself. Records too long for SentencePiece's `max_sentence_length` are
**dropped and counted**, never truncated -- silent truncation is risk R1,
the single easiest way to train on a corrupted subset with no error.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

CORPUS_SUFFIXES = (".jsonl", ".json", ".txt")

# Trailing "_1234mb" produced by training_corpus/common.py's output naming,
# stripped so slices are labelled by source rather than by download size.
_SIZE_SUFFIX_RE = re.compile(r"_\d+(?:\.\d+)?mb$", re.IGNORECASE)


@dataclass
class CorpusStats:
    """Counters for the loud-failure guard and the run manifest."""

    files: int = 0
    records_seen: int = 0
    records_bad_json: int = 0
    records_missing_text: int = 0
    lines_eligible: int = 0
    lines_empty: int = 0
    lines_oversized: int = 0
    bytes_eligible: int = 0
    per_source_lines: dict[str, int] = field(default_factory=dict)
    per_source_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def lines_dropped(self) -> int:
        return self.lines_empty + self.lines_oversized

    @property
    def records_dropped(self) -> int:
        return self.records_bad_json + self.records_missing_text

    def drop_fraction(self) -> float:
        """
        Fraction of encountered content lost to malformed or oversized data.

        Whitespace-only lines are excluded from both numerator and
        denominator: discarding them is structural, not data loss, and a
        .txt corpus using blank lines as paragraph separators would
        otherwise trip the guard for no real reason. The guard exists to
        catch bad JSON, missing fields, and R1-style oversized-record loss.
        """
        lost = self.lines_oversized + self.records_dropped
        total = self.lines_eligible + lost
        return 0.0 if total == 0 else lost / total

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "records_seen": self.records_seen,
            "records_bad_json": self.records_bad_json,
            "records_missing_text": self.records_missing_text,
            "lines_eligible": self.lines_eligible,
            "lines_empty": self.lines_empty,
            "lines_oversized": self.lines_oversized,
            "bytes_eligible": self.bytes_eligible,
            "drop_fraction": round(self.drop_fraction(), 6),
            "per_source_lines": dict(sorted(self.per_source_lines.items())),
            "per_source_bytes": dict(sorted(self.per_source_bytes.items())),
        }


class CorpusError(RuntimeError):
    """Raised when the corpus is unusable or degraded beyond the allowed threshold."""


def source_label(path: Path) -> str:
    """`output/fineweb_edu_2800mb.jsonl` -> `fineweb_edu` (slice label for Gate 4)."""
    return _SIZE_SUFFIX_RE.sub("", path.stem) or path.stem


def discover_corpus_files(corpus_path: str | Path) -> list[Path]:
    """
    Resolve --corpus (a file or a directory) to a sorted list of corpus files.

    Sorted so that pass 1 and pass 2 traverse in identical order -- the
    sampling is only reproducible if the traversal order is.
    """
    corpus_path = Path(corpus_path)
    if corpus_path.is_file():
        files = [corpus_path]
    elif corpus_path.is_dir():
        files = sorted(
            p for p in corpus_path.rglob("*") if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES
        )
    else:
        raise CorpusError(f"corpus path does not exist: {corpus_path}")

    if not files:
        raise CorpusError(
            f"no corpus files found under {corpus_path} "
            f"(looking for {', '.join(CORPUS_SUFFIXES)})"
        )
    return files


def _extract_text(raw_line: str, is_jsonl: bool, stats: CorpusStats) -> str | None:
    """One raw file line -> the text it carries, or None if unusable."""
    if is_jsonl:
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            stats.records_bad_json += 1
            return None
        if not isinstance(row, dict):
            stats.records_missing_text += 1
            return None
        text = row.get("text")
        if not isinstance(text, str) or not text:
            stats.records_missing_text += 1
            return None
        return text
    return raw_line


def iter_eligible_lines(
    files: list[Path], max_sentence_length: int, stats: CorpusStats
) -> Iterator[tuple[str, str]]:
    """
    Stream `(text_line, source_label)` for every line eligible for training.

    A record containing newlines yields one entry per newline-separated
    segment: SentencePiece reads line-by-line, so this is the split it would
    perform anyway, made explicit and countable here (DESIGN.md 6.1f).

    `max_sentence_length` is compared in **UTF-8 bytes**, matching
    SentencePiece's own semantics -- comparing characters would let a
    multi-byte line slip past this check and be silently dropped by the
    trainer instead (risk R1).
    """
    stats.files = len(files)
    for path in files:
        label = source_label(path)
        is_jsonl = path.suffix.lower() in (".jsonl", ".json")
        # errors="strict" (the default) is deliberate: errors="replace" would
        # substitute U+FFFD for undecodable bytes, i.e. silently corrupt the
        # training text in exactly the way NF1 forbids. Fail loudly instead.
        with open(path, encoding="utf-8") as fh:
            try:
                for raw_line in fh:
                    raw_line = raw_line.rstrip("\n").rstrip("\r")
                    stats.records_seen += 1
                    if not raw_line.strip():
                        stats.lines_empty += 1
                        continue
                    text = _extract_text(raw_line, is_jsonl, stats)
                    if text is None:
                        continue
                    for segment in text.split("\n"):
                        segment = segment.rstrip("\r")
                        if not segment.strip():
                            stats.lines_empty += 1
                            continue
                        n_bytes = len(segment.encode("utf-8"))
                        if n_bytes > max_sentence_length:
                            stats.lines_oversized += 1
                            continue
                        stats.lines_eligible += 1
                        stats.bytes_eligible += n_bytes
                        stats.per_source_lines[label] = stats.per_source_lines.get(label, 0) + 1
                        stats.per_source_bytes[label] = (
                            stats.per_source_bytes.get(label, 0) + n_bytes
                        )
                        yield segment, label
            except UnicodeDecodeError as exc:
                raise CorpusError(
                    f"{path} is not valid UTF-8 ({exc}). Refusing to decode with "
                    f"replacement characters, which would silently corrupt training "
                    f"text (DESIGN.md NF1). Re-generate this file from its downloader."
                ) from exc


def count_eligible_lines(files: list[Path], max_sentence_length: int) -> CorpusStats:
    """Pass 1: how many lines are eligible, and how many were dropped and why."""
    stats = CorpusStats()
    for _ in iter_eligible_lines(files, max_sentence_length, stats):
        pass
    return stats


def _train_budget(total: int, requested: int) -> int:
    """
    How many lines to actually train on, given how much corpus exists.

    Normally the requested budget: everything not selected becomes the
    evaluation set. When the corpus is *smaller* than the budget, a floor of
    10% is held back so the gates still have unseen text to measure against
    -- a run that trained on literally every line could not be validated at
    all, and silently reporting gates measured on training data would be
    worse than reporting nothing.
    """
    if total <= 0:
        return 0
    if total <= requested:
        return total - max(1, total // 10)
    return requested


def _allocate(budget: int, weights: dict[str, float], capacities: dict[str, int]) -> dict[str, int]:
    """
    Split `budget` across sources proportionally to `weights`, never exceeding
    `capacities`, redistributing any shortfall to sources that still have room.

    Integer, deterministic (largest-remainder with name tie-breaks), and
    exact: the result sums to min(budget, total capacity).
    """
    alloc = {key: 0 for key in capacities}
    remaining = min(budget, sum(capacities.values()))
    active = {k for k in capacities if weights.get(k, 0.0) > 0 and capacities[k] > 0}

    while remaining > 0 and active:
        weight_sum = sum(weights[k] for k in active)
        if weight_sum <= 0:
            break
        raw = {k: remaining * weights[k] / weight_sum for k in active}
        share = {k: int(raw[k]) for k in active}
        leftover = remaining - sum(share.values())
        # Largest fractional remainder first; name breaks ties so the result
        # does not depend on dict ordering.
        for key in sorted(active, key=lambda k: (-(raw[k] - share[k]), k))[:leftover]:
            share[key] += 1

        progressed = False
        for key in list(active):
            take = min(share[key], capacities[key] - alloc[key])
            if take > 0:
                alloc[key] += take
                remaining -= take
                progressed = True
        active = {k for k in active if capacities[k] > alloc[k]}
        if not progressed:
            break
    return alloc


def compute_blend_weights(
    blend: dict, per_source_lines: dict[str, int], per_source_bytes: dict[str, int]
) -> tuple[dict[str, float], list[str]]:
    """
    Turn the configured blend into per-source *line* weights.

    Why this exists: the blend in DESIGN.md 4.3 is a share of **text**, but
    the sampler draws **lines**, and average line length varies enormously
    between sources -- in the real corpus PG-19 averages ~69 bytes per line
    (books are newline-wrapped) against C4's ~447. Sampling lines uniformly
    would therefore hand the tokenizer ~55% Gutenberg and ~8% C4 instead of
    the specified 20%/20%, i.e. fit the vocabulary to the wrong corpus mix
    (risk R4) while every byte-level download share looked correct.

    Weighting each source by share/avg_bytes_per_line makes the *bytes*
    drawn land on the specified proportions.

    Returns (weights, notes) where notes describe any renormalization.
    """
    notes: list[str] = []
    if blend["mode"] == "uniform":
        return {src: float(n) for src, n in per_source_lines.items()}, notes

    shares: dict[str, float] = blend["shares"]
    present = {src for src in per_source_lines if per_source_lines[src] > 0}
    known = present & set(shares)
    unknown = sorted(present - set(shares))
    absent = sorted(set(shares) - present)

    if not known:
        raise CorpusError(
            f"blend.mode is 'weighted' but none of the configured sources "
            f"{sorted(shares)} are present in the corpus (found: {sorted(present)}). "
            f"Source names come from the corpus filenames, e.g. "
            f"'fineweb_edu_7000mb.jsonl' -> 'fineweb_edu'."
        )
    if unknown:
        notes.append(f"excluded (no blend share configured): {', '.join(unknown)}")
    if absent:
        notes.append(f"configured but absent from corpus: {', '.join(absent)}")

    total_share = sum(shares[src] for src in known)
    if len(known) != len(shares):
        notes.append(f"renormalized {len(known)} present share(s) over {total_share:.3f}")

    weights: dict[str, float] = {}
    for src in per_source_lines:
        if src not in known:
            weights[src] = 0.0
            continue
        avg_bytes_per_line = per_source_bytes[src] / per_source_lines[src]
        weights[src] = (shares[src] / total_share) / avg_bytes_per_line
    return weights, notes


@dataclass
class SplitPlan:
    """
    Everything needed to reproduce the train/eval split without re-deciding it.

    Serializable, so it round-trips through manifest.json: the validation
    stage reconstructs the plan after training and replays the identical
    split to stream the eval side (see the module docstring).
    """

    capacities: dict[str, int]  # per-source eligible lines, from pass 1
    train_per_source: dict[str, int]  # how many of those to train on
    seed: int
    max_sentence_length: int

    @property
    def n_total(self) -> int:
        return sum(self.capacities.values())

    @property
    def n_train(self) -> int:
        return sum(self.train_per_source.values())

    @property
    def n_eval(self) -> int:
        return self.n_total - self.n_train

    def eval_capacities(self) -> dict[str, int]:
        return {
            src: n - self.train_per_source.get(src, 0) for src, n in self.capacities.items()
        }

    def as_dict(self) -> dict:
        return {
            "capacities": dict(sorted(self.capacities.items())),
            "train_per_source": dict(sorted(self.train_per_source.items())),
            "seed": self.seed,
            "max_sentence_length": self.max_sentence_length,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SplitPlan":
        return cls(
            capacities=dict(payload["capacities"]),
            train_per_source=dict(payload["train_per_source"]),
            seed=payload["seed"],
            max_sentence_length=payload["max_sentence_length"],
        )


def plan_split(
    stats: CorpusStats, train_budget: int, seed: int, max_sentence_length: int, blend: dict
) -> tuple[SplitPlan, list[str]]:
    """Pass-1 statistics + the configured blend -> a reproducible SplitPlan."""
    n_train = _train_budget(stats.lines_eligible, train_budget)
    capacities = dict(stats.per_source_lines)
    weights, notes = compute_blend_weights(blend, stats.per_source_lines, stats.per_source_bytes)
    train_per_source = _allocate(n_train, weights, capacities)
    return (
        SplitPlan(
            capacities=capacities,
            train_per_source=train_per_source,
            seed=seed,
            max_sentence_length=max_sentence_length,
        ),
        notes,
    )


def iter_split(
    files: list[Path], plan: SplitPlan, stats: CorpusStats | None = None
) -> Iterator[tuple[str, str, bool]]:
    """
    Yield `(text, source_label, is_train)` for every eligible line, in order.

    **The single source of truth for the train/eval split.** Deterministic
    given (files, plan): the same call produces the same decisions every
    time, which is what lets the eval stream be replayed after training
    instead of being written to disk.

    Algorithm S per source: keep a line for training with probability
    (still needed) / (still available). Yields exactly `plan.n_train` train
    lines, uniformly at random within each source, buffering nothing.
    """
    remaining_pool = dict(plan.capacities)
    remaining_select = dict(plan.train_per_source)
    rng = random.Random(plan.seed)

    for segment, label in iter_eligible_lines(
        files, plan.max_sentence_length, stats if stats is not None else CorpusStats()
    ):
        pool = remaining_pool.get(label, 0)
        if pool <= 0:
            # Beyond what pass 1 counted for this source: the corpus grew
            # mid-run. Never training data (the plan has no budget for it);
            # surfaced as a hard error by build_corpus's reconciliation.
            yield segment, label, False
            continue
        remaining_pool[label] = pool - 1
        need = remaining_select.get(label, 0)
        # Short-circuit matters: rng is consumed only when this source still
        # needs lines, so the RNG sequence depends solely on the plan.
        if need > 0 and rng.random() * pool < need:
            remaining_select[label] = need - 1
            yield segment, label, True
        else:
            yield segment, label, False


def iter_eval_lines(
    files: list[Path], plan: SplitPlan, limit: int = 0, stats: CorpusStats | None = None
) -> Iterator[tuple[str, str]]:
    """
    Stream the evaluation set: every eligible line NOT selected for training.

    `limit` (0 = no limit) caps the stream at that many lines, drawn with
    Algorithm S over the eval pool so a capped run is an unbiased subsample
    of exactly what the full run would have measured -- `--eval-corpus 20000`
    is therefore a faster estimate of the same numbers, not a different
    measurement of a different slice.
    """
    remaining = plan.n_eval
    need = remaining if limit <= 0 else min(limit, remaining)
    # A stream independent of the train/eval decision itself, so capping the
    # eval size cannot perturb which lines were trained on.
    rng = random.Random(plan.seed + 1)

    for segment, label, is_train in iter_split(files, plan, stats):
        if is_train:
            continue
        if need <= 0:
            break
        if rng.random() * remaining < need:
            need -= 1
            yield segment, label
        remaining -= 1


def build_corpus(
    files: list[Path],
    *,
    corpus_out: Path,
    train_budget: int,
    seed: int,
    max_sentence_length: int,
    max_drop_fraction: float,
    blend: dict | None = None,
) -> dict:
    """
    Produce `corpus.txt`: the plain text, one sentence per line, that
    SentencePiece trains on.

    The evaluation set is everything else, and is not written here -- it is
    replayed on demand by `iter_eval_lines(files, plan)` using the returned
    plan. Train and eval are disjoint by construction, so the gates are
    always measured on text the tokenizer was genuinely not fitted to.

    Deterministic: identical inputs + seed produce byte-identical outputs.
    """
    count_stats = count_eligible_lines(files, max_sentence_length)

    if count_stats.lines_eligible == 0:
        raise CorpusError(
            f"no eligible lines found in {count_stats.files} file(s). "
            f"Dropped: {count_stats.records_bad_json} bad JSON, "
            f"{count_stats.records_missing_text} missing/empty 'text' field, "
            f"{count_stats.lines_oversized} over {max_sentence_length} bytes, "
            f"{count_stats.lines_empty} empty."
        )

    drop_fraction = count_stats.drop_fraction()
    if drop_fraction > max_drop_fraction:
        raise CorpusError(
            f"corpus drop fraction {drop_fraction:.4f} exceeds "
            f"corpus.max_drop_fraction={max_drop_fraction}. Refusing to train on a "
            f"degraded sample (DESIGN.md risk R1). Breakdown: "
            f"{count_stats.records_bad_json} bad JSON, "
            f"{count_stats.records_missing_text} missing 'text', "
            f"{count_stats.lines_oversized} over {max_sentence_length} bytes, "
            f"{count_stats.lines_empty} empty, "
            f"{count_stats.lines_eligible} kept."
        )

    blend = blend or {"mode": "uniform", "shares": {}}
    total = count_stats.lines_eligible

    plan, blend_notes = plan_split(count_stats, train_budget, seed, max_sentence_length, blend)
    # A weighted blend can exclude sources entirely, so the achievable total
    # may be below the requested budget.
    n_train = plan.n_train

    written_train = 0
    train_written: dict[str, int] = {}
    train_bytes: dict[str, int] = {}

    pass2_stats = CorpusStats()
    corpus_out.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_out, "w", encoding="utf-8") as corpus_fh:
        for segment, label, is_train in iter_split(files, plan, pass2_stats):
            if not is_train:
                continue
            corpus_fh.write(segment + "\n")
            train_written[label] = train_written.get(label, 0) + 1
            train_bytes[label] = train_bytes.get(label, 0) + len(segment.encode("utf-8"))
            written_train += 1

    # Pass 1 and pass 2 must agree; if they do not, the corpus changed on
    # disk mid-run and the sample is not the one the manifest describes.
    # This also guards the eval replay: a plan that no longer matches the
    # files on disk would silently stream a different eval set.
    if written_train != n_train or pass2_stats.lines_eligible != total:
        raise CorpusError(
            f"sampling produced {written_train} train lines from "
            f"{pass2_stats.lines_eligible} eligible, expected {n_train} from {total}. "
            f"This usually means the corpus changed on disk during the run; do not "
            f"modify it while training."
        )

    sampled_bytes = sum(train_bytes.values())
    achieved_blend = {
        src: round(count / sampled_bytes, 4) for src, count in sorted(train_bytes.items())
    } if sampled_bytes else {}

    # A weighted blend can only be honoured if there is more corpus than
    # budget to draw from. When a source is exhausted -- most commonly
    # because the whole corpus fits inside the sample budget -- the achieved
    # shares drift from the target. That must be visible, not silent
    # (Principle 4): it is the difference between "fitted to the specified
    # mix" and "fitted to whatever happened to be on disk".
    if blend["mode"] == "weighted" and achieved_blend:
        present_shares = {s: blend["shares"][s] for s in achieved_blend if s in blend["shares"]}
        share_total = sum(present_shares.values())
        if share_total > 0:
            worst = max(
                abs(achieved_blend[s] - present_shares[s] / share_total) for s in present_shares
            )
            if worst > 0.02:
                reason = (
                    "the whole corpus fits inside the sample budget, so nearly every line "
                    "was taken"
                    if n_train >= total - max(1, total // 10)
                    else "at least one source was exhausted before reaching its share"
                )
                blend_notes.append(
                    f"BLEND NOT APPLIED (max deviation {worst:.1%}): {reason}. "
                    f"The vocabulary is fitted to the corpus as-is, not to the configured "
                    f"shares. Add more corpus or lower corpus.input_sentence_size to honour "
                    f"the blend."
                )

    return {
        "scan": count_stats.as_dict(),
        "eligible_lines": total,
        "train_lines": written_train,
        "eval_lines_available": plan.n_eval,
        "blend_mode": blend["mode"],
        "blend_target": dict(sorted(blend.get("shares", {}).items())),
        "blend_achieved_bytes": achieved_blend,
        "blend_notes": blend_notes,
        "train_lines_per_source": dict(sorted(train_written.items())),
        "train_bytes_per_source": dict(sorted(train_bytes.items())),
        "eval_lines_per_source": dict(sorted(plan.eval_capacities().items())),
        "sampled_fraction": round(written_train / total, 6) if total else 0.0,
        "seed": seed,
        "plan": plan.as_dict(),
        "corpus_path": str(corpus_out),
        "corpus_bytes": corpus_out.stat().st_size,
        "corpus_sha256": sha256_file(corpus_out),
    }


def load_eval_records(path: Path) -> list[dict]:
    """Read a saved eval sample back as a list of {"text":..., "source":...} records."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                records.append(json.loads(line))
    return records


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
