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

  pass 1: count eligible lines N, gather per-source and per-FILE stats
          -> SplitPlan (including a per-file quota table)
  pass 2: Algorithm S (selection sampling) picks exactly K of the N lines
          uniformly at random, writing straight to disk

Reservoir sampling would need the K sampled lines resident in memory (~5 GB
at K=20M); selection sampling needs only two counters, at the cost of one
extra sequential read.

Both passes run **per file**, which is what lets them run in parallel
--------------------------------------------------------------------
Originally both passes were one long Python loop over every file in sequence,
so a 1.44-billion-line corpus pinned exactly one core for hours. The work is
naturally independent per file, but the *sampling* was not: one shared RNG
consumed in traversal order meant file 7's decisions depended on how many
random numbers files 0-6 had drawn first.

That dependency is removed by deciding each source's budget per file up front
(`allocate_file_quotas`, proportional and exact via `largest_remainder`) and
seeding each file's RNG from `(seed, file_index)`. Each file is then an
independent, reproducible unit of work that `execution.Executor` can place on
any core, in any order.

The sample this produces is stratified by file rather than a single global
simple random sample. Every line still has essentially the same selection
probability, the per-source totals are exactly as before, and the draw is
guaranteed to span every file holding a source rather than merely being very
likely to. Worker count never changes the result -- see
`tests/unit/test_parallel_equivalence.py`, which asserts corpus.txt is
byte-identical at 1, 2 and 4 workers.

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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from execution import Executor, largest_remainder

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

    def as_counters(self) -> dict:
        """
        The constructor's own kwargs, for shipping stats back from a worker.

        Distinct from `as_dict()`, which is the manifest's view and includes
        a derived field that is not a constructor argument.
        """
        return {
            "files": self.files,
            "records_seen": self.records_seen,
            "records_bad_json": self.records_bad_json,
            "records_missing_text": self.records_missing_text,
            "lines_eligible": self.lines_eligible,
            "lines_empty": self.lines_empty,
            "lines_oversized": self.lines_oversized,
            "bytes_eligible": self.bytes_eligible,
            "per_source_lines": dict(self.per_source_lines),
            "per_source_bytes": dict(self.per_source_bytes),
        }

    def absorb(self, other: "CorpusStats") -> None:
        """Fold another stats object into this one, in place."""
        self.files += other.files
        self.records_seen += other.records_seen
        self.records_bad_json += other.records_bad_json
        self.records_missing_text += other.records_missing_text
        self.lines_eligible += other.lines_eligible
        self.lines_empty += other.lines_empty
        self.lines_oversized += other.lines_oversized
        self.bytes_eligible += other.bytes_eligible
        for src, n in other.per_source_lines.items():
            self.per_source_lines[src] = self.per_source_lines.get(src, 0) + n
        for src, n in other.per_source_bytes.items():
            self.per_source_bytes[src] = self.per_source_bytes.get(src, 0) + n

    @classmethod
    def merged(cls, parts: Iterable["CorpusStats"]) -> "CorpusStats":
        """
        Combine per-file stats into one corpus-wide total.

        Every field is a sum, a set union or a max, so the result does not
        depend on the order the parts arrive in -- which is what allows the
        scan to be split across workers without changing its answer.
        """
        total = cls()
        for part in parts:
            total.absorb(part)
        return total


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


def iter_file_lines(
    path: Path, max_sentence_length: int, stats: CorpusStats
) -> Iterator[tuple[str, str]]:
    """
    Stream `(text_line, source_label)` for every eligible line of ONE file,
    folding that file's counters into `stats`.

    This is the indivisible unit of corpus reading. Counting, sampling and
    eval replay are all composed from per-file calls to it, which is what lets
    each of them be split across worker processes without any caller needing
    to know that it was: a file is never shared between two workers, so there
    is no shared state to coordinate.

    A record containing newlines yields one entry per newline-separated
    segment: SentencePiece reads line-by-line, so this is the split it would
    perform anyway, made explicit and countable here (DESIGN.md 6.1f).

    `max_sentence_length` is compared in **UTF-8 bytes**, matching
    SentencePiece's own semantics -- comparing characters would let a
    multi-byte line slip past this check and be silently dropped by the
    trainer instead (risk R1).
    """
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


def iter_eligible_lines(
    files: list[Path], max_sentence_length: int, stats: CorpusStats
) -> Iterator[tuple[str, str]]:
    """Sequential stream over every file, for callers that want one iterator."""
    stats.files = len(files)
    for path in files:
        yield from iter_file_lines(path, max_sentence_length, stats)


def scan_file(task: tuple[str, int]) -> CorpusStats:
    """
    Pass 1 for a single file: its own counters, nothing else.

    Module-level, and taking one picklable argument, because this is what gets
    handed to worker processes. It is pure -- same file in, same counters out
    -- which is precisely why a 32-worker scan returns the same totals as a
    1-worker scan.
    """
    path_str, max_sentence_length = task
    stats = CorpusStats(files=1)
    for _segment, _label in iter_file_lines(Path(path_str), max_sentence_length, stats):
        pass
    return stats


def scan_corpus(
    files: list[Path], max_sentence_length: int, executor: Executor | None = None
) -> tuple[CorpusStats, list[CorpusStats]]:
    """
    Pass 1 over the whole corpus: merged totals, plus each file's own counters.

    The per-file breakdown is not a debugging extra. It is what allows the
    sampling budget to be divided into independent per-file quotas below,
    which is the entire reason pass 2 can run in parallel too.
    """
    executor = executor or Executor()
    per_file = executor.map(scan_file, [(str(p), max_sentence_length) for p in files])
    return CorpusStats.merged(per_file), per_file


def count_eligible_lines(
    files: list[Path], max_sentence_length: int, executor: Executor | None = None
) -> CorpusStats:
    """Pass 1: how many lines are eligible, and how many were dropped and why."""
    merged, _per_file = scan_corpus(files, max_sentence_length, executor)
    return merged


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
    # Per-file breakdown of the two dicts above, in corpus file order. This is
    # what turns one global sampling decision into N independent per-file ones
    # that can run in any order, in any process, and still add up to exactly
    # the same totals. Carried in the plan (and therefore in manifest.json) so
    # the eval replay reconstructs the identical split without re-scanning.
    file_capacities: list[dict[str, int]] = field(default_factory=list)
    file_train: list[dict[str, int]] = field(default_factory=list)

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
            "file_capacities": [dict(sorted(d.items())) for d in self.file_capacities],
            "file_train": [dict(sorted(d.items())) for d in self.file_train],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SplitPlan":
        return cls(
            capacities=dict(payload["capacities"]),
            train_per_source=dict(payload["train_per_source"]),
            seed=payload["seed"],
            max_sentence_length=payload["max_sentence_length"],
            file_capacities=[dict(d) for d in payload.get("file_capacities", [])],
            file_train=[dict(d) for d in payload.get("file_train", [])],
        )

    def file_eval_capacities(self) -> list[int]:
        """Eval lines available in each file: its eligible lines minus its train quota."""
        return [
            sum(cap.values()) - sum(train.values())
            for cap, train in zip(self.file_capacities, self.file_train)
        ]


def allocate_file_quotas(
    train_per_source: dict[str, int], file_capacities: list[dict[str, int]]
) -> list[dict[str, int]]:
    """
    Split each source's training quota across the files that hold that source.

    Proportional to how many of the source's eligible lines each file holds,
    with `largest_remainder` making the pieces sum to exactly the global quota.
    The result is a stratified sample rather than the simple random sample the
    single-pass version drew: every line still has essentially the same chance
    of selection (quota_f / lines_f is the same ratio in every file), but the
    per-file counts are fixed in advance instead of emerging from one shared
    RNG stream. That is what removes the sequential dependency between files.

    A welcome side effect: the sample is now guaranteed to span every file
    holding a source, where the global version merely made it overwhelmingly
    likely.
    """
    quotas: list[dict[str, int]] = [{} for _ in file_capacities]
    for source, budget in train_per_source.items():
        if budget <= 0:
            continue
        weights = [caps.get(source, 0) for caps in file_capacities]
        for idx, share in enumerate(largest_remainder(budget, weights)):
            if share:
                quotas[idx][source] = share
    return quotas


def plan_split(
    stats: CorpusStats,
    train_budget: int,
    seed: int,
    max_sentence_length: int,
    blend: dict,
    file_scans: list[CorpusStats] | None = None,
) -> tuple[SplitPlan, list[str]]:
    """Pass-1 statistics + the configured blend -> a reproducible SplitPlan."""
    n_train = _train_budget(stats.lines_eligible, train_budget)
    capacities = dict(stats.per_source_lines)
    weights, notes = compute_blend_weights(blend, stats.per_source_lines, stats.per_source_bytes)
    train_per_source = _allocate(n_train, weights, capacities)

    file_capacities = [dict(scan.per_source_lines) for scan in (file_scans or [])]
    file_train = allocate_file_quotas(train_per_source, file_capacities)
    return (
        SplitPlan(
            capacities=capacities,
            train_per_source=train_per_source,
            seed=seed,
            max_sentence_length=max_sentence_length,
            file_capacities=file_capacities,
            file_train=file_train,
        ),
        notes,
    )


def iter_file_split(
    path: Path, file_idx: int, plan: SplitPlan, stats: CorpusStats
) -> Iterator[tuple[str, str, bool]]:
    """
    Yield `(text, source_label, is_train)` for one file.

    **The single source of truth for the train/eval split.** Every caller --
    the parallel corpus writer, the sequential replay, and the eval stream --
    goes through this one function, so there is no second implementation of
    the split to drift out of sync.

    Algorithm S per (file, source): keep a line with probability
    (still needed) / (still available). Yields exactly this file's quota,
    uniformly at random within the file, buffering nothing.

    The RNG is seeded from the plan seed *and the file's index*, so each file
    is independently reproducible. That is the change that makes the pass
    parallelisable: file 7's decisions no longer depend on how many random
    numbers files 0-6 happened to consume first.
    """
    quota = dict(plan.file_train[file_idx]) if file_idx < len(plan.file_train) else {}
    pool = dict(plan.file_capacities[file_idx]) if file_idx < len(plan.file_capacities) else {}
    rng = random.Random(f"{plan.seed}:{file_idx}")

    for segment, label in iter_file_lines(path, plan.max_sentence_length, stats):
        available = pool.get(label, 0)
        if available <= 0:
            # Beyond what pass 1 counted for this source in this file: the
            # corpus changed mid-run. Never training data (no budget for it);
            # surfaced as a hard error by build_corpus's reconciliation.
            yield segment, label, False
            continue
        pool[label] = available - 1
        need = quota.get(label, 0)
        # Short-circuit matters: rng is consumed only when this source still
        # needs lines, so the sequence depends solely on the plan.
        if need > 0 and rng.random() * available < need:
            quota[label] = need - 1
            yield segment, label, True
        else:
            yield segment, label, False


def iter_split(
    files: list[Path], plan: SplitPlan, stats: CorpusStats | None = None
) -> Iterator[tuple[str, str, bool]]:
    """Sequential replay of the whole split, file by file, in corpus order."""
    if stats is None:
        stats = CorpusStats()
    stats.files = len(files)
    for file_idx, path in enumerate(files):
        yield from iter_file_split(path, file_idx, plan, stats)


def eval_file_quotas(plan: SplitPlan, limit: int = 0) -> list[int]:
    """
    How many eval lines to draw from each file, for a capped eval run.

    Proportional to each file's eval pool, so a capped sample is spread over
    the whole corpus rather than exhausting the first files it reads.
    """
    capacities = plan.file_eval_capacities()
    total = sum(capacities)
    need = total if limit <= 0 else min(limit, total)
    return largest_remainder(need, capacities)


def iter_eval_file(
    path: Path, file_idx: int, plan: SplitPlan, quota: int, stats: CorpusStats
) -> Iterator[tuple[str, str]]:
    """
    Stream one file's share of the evaluation set: eligible lines NOT selected
    for training, subsampled to `quota` by Algorithm S.

    Seeded independently of the train/eval decision itself (`seed + 1`), so
    capping the eval size cannot perturb which lines were trained on.
    """
    capacities = plan.file_eval_capacities()
    remaining = capacities[file_idx] if file_idx < len(capacities) else 0
    need = quota
    rng = random.Random(f"{plan.seed + 1}:{file_idx}")

    for segment, label, is_train in iter_file_split(path, file_idx, plan, stats):
        if is_train:
            continue
        if need <= 0:
            break
        if rng.random() * remaining < need:
            need -= 1
            yield segment, label
        remaining -= 1


def iter_eval_lines(
    files: list[Path], plan: SplitPlan, limit: int = 0, stats: CorpusStats | None = None
) -> Iterator[tuple[str, str]]:
    """
    Stream the evaluation set: every eligible line NOT selected for training.

    `limit` (0 = no limit) caps the stream at that many lines, drawn with
    Algorithm S over each file's eval pool so a capped run is an unbiased
    subsample of exactly what the full run would have measured --
    `--eval-corpus 20000` is a faster estimate of the same numbers, not a
    measurement of a different slice.
    """
    if stats is None:
        stats = CorpusStats()
    stats.files = len(files)
    quotas = eval_file_quotas(plan, limit)
    for file_idx, path in enumerate(files):
        quota = quotas[file_idx] if file_idx < len(quotas) else 0
        yield from iter_eval_file(path, file_idx, plan, quota, stats)


def _select_file(task: tuple[str, int, dict, str]) -> dict:
    """
    Pass 2 for one file: write its selected training lines to its own shard.

    Each worker owns one input file and one output shard, so there is no
    shared handle and no lock. The parent concatenates the shards in file
    order afterwards, which is what keeps corpus.txt byte-identical
    regardless of how many workers produced it or what order they finished in.
    """
    path_str, file_idx, plan_dict, shard_str = task
    plan = SplitPlan.from_dict(plan_dict)
    stats = CorpusStats(files=1)
    lines: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    written = 0

    with open(shard_str, "w", encoding="utf-8") as shard_fh:
        for segment, label, is_train in iter_file_split(Path(path_str), file_idx, plan, stats):
            if not is_train:
                continue
            shard_fh.write(segment + "\n")
            lines[label] = lines.get(label, 0) + 1
            byte_counts[label] = byte_counts.get(label, 0) + len(segment.encode("utf-8"))
            written += 1

    return {
        "file_idx": file_idx,
        "shard": shard_str,
        "written": written,
        "lines": lines,
        "bytes": byte_counts,
        "stats": stats.as_counters(),
    }


def _write_corpus(
    files: list[Path], plan: SplitPlan, corpus_out: Path, executor: Executor
) -> list[dict]:
    """Run pass 2 over every file and stitch the shards into `corpus_out`."""
    shard_dir = corpus_out.parent / f".{corpus_out.name}.shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan_dict = plan.as_dict()
        tasks = [
            (str(path), idx, plan_dict, str(shard_dir / f"{idx:06d}.txt"))
            for idx, path in enumerate(files)
        ]
        results = executor.map(_select_file, tasks)
        results.sort(key=lambda r: r["file_idx"])

        with open(corpus_out, "wb") as out_fh:
            for result in results:
                with open(result["shard"], "rb") as shard_fh:
                    shutil.copyfileobj(shard_fh, out_fh, 4 * 1024 * 1024)
        return results
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)


def build_corpus(
    files: list[Path],
    *,
    corpus_out: Path,
    train_budget: int,
    seed: int,
    max_sentence_length: int,
    max_drop_fraction: float,
    blend: dict | None = None,
    executor: Executor | None = None,
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
    executor = executor or Executor()
    count_stats, file_scans = scan_corpus(files, max_sentence_length, executor)

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

    plan, blend_notes = plan_split(
        count_stats, train_budget, seed, max_sentence_length, blend, file_scans
    )
    # A weighted blend can exclude sources entirely, so the achievable total
    # may be below the requested budget.
    n_train = plan.n_train

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    results = _write_corpus(files, plan, corpus_out, executor)

    written_train = sum(r["written"] for r in results)
    train_written: dict[str, int] = {}
    train_bytes: dict[str, int] = {}
    for result in results:
        for label, count in result["lines"].items():
            train_written[label] = train_written.get(label, 0) + count
        for label, count in result["bytes"].items():
            train_bytes[label] = train_bytes.get(label, 0) + count
    pass2_stats = CorpusStats.merged(CorpusStats(**r["stats"]) for r in results)

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
