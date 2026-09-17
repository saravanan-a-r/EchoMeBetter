#!/usr/bin/env python3
"""
Stage 1 entrypoint — wires `corpus/`, `UL2/`, `model/`, `training/` and
`rewrite/` together into an actual pretraining run (architecture.md §7).

This is the piece none of the packages owns: `corpus/` reads text, `UL2/`
corrupts it, `rewrite/` builds the cooldown's rewrite pairs from it, `model/`
scores it, and `training/` drives the loop — each package is deliberately
ignorant of the others, so something outside all of them has to do the
wiring. This script is that something, and nothing more: no new corruption
logic, no new training logic, no new model logic.

Usage
-----
    python pretrain.py --corpus path/to/corpus --stage rehearsal
    python pretrain.py --corpus path/to/corpus --eval-corpus path/to/heldout

    # two-tier held-out evaluation (training_config.yml's *_eval_steps):
    python pretrain.py --corpus pretrain_corpus/output/pretrain_output \
        --quick-eval-corpus  pretrain_corpus/output/evals/quick_eval.jsonl \
        --master-eval-corpus pretrain_corpus/output/evals/master_eval.jsonl

The two eval sets exist because one cannot do both jobs: eval loss on a small
sample has a standard error shrinking as 1/sqrt(n), so a set thin enough to
score every thousand steps is too noisy to rank adjacent checkpoints, and a
set precise enough to rank them is too slow to score that often. The quick set
is the trend line; the master set alone decides which checkpoint is kept.

`--stage` selects a block from `training_config.yml` (defaults to its
`active_stage`); the stage's own `model_profile`, if set, selects the
`model_config.yml` profile unless `--model-profile` overrides it. This is the
same two-file split `training/README.md` documents.

See `pretrain_smoke_test.py` for the "does this actually work" check
architecture.md §7.8 asks be run before committing weeks of compute to Large.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import random
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_SRC = PROJECT_ROOT / "corpus" / "src"
UL2_SRC = PROJECT_ROOT / "UL2" / "src"
TRAINING_SRC = PROJECT_ROOT / "training" / "src"
MODEL_SRC = PROJECT_ROOT / "model" / "src"
REWRITE_SRC = PROJECT_ROOT / "rewrite" / "src"
MONITORING_SRC = PROJECT_ROOT / "monitoring" / "src"

DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "training" / "output"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "model_config.yml"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "training_config.yml"


# -- sibling loading, same trick training/src/interop.py uses ---------------


def _load(name: str, source: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, source / "__init__.py", submodule_search_locations=[str(source)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


corpus_pkg = _load("echomebetter_corpus", CORPUS_SRC)
ul2 = _load("echomebetter_ul2", UL2_SRC)
training_pkg = _load("echomebetter_training", TRAINING_SRC)
rephrase_model = _load("echomebetter_model", MODEL_SRC)
rewrite = _load("echomebetter_rewrite", REWRITE_SRC)


# Both sequence dimensions are padded up to a multiple of this, which keeps
# the tensor-core kernels on their fast path. Named rather than repeated
# because `BatchCapacity` has to count against the same rounding the batches
# are actually padded to.
PAD_TO_MULTIPLE_OF = 8


# -- gluing corpus + UL2 into what the trainer needs -------------------------


class PretrainBatchSource:
    """
    Turns a `TokenizedCorpus` into a stream of padded UL2 batches, and is
    itself `Resumable` (`training/src/checkpoint.py`): resuming restores the
    corpus position, the corruption RNG *and* the buffered documents not yet
    packed into a window, so a resumed run's sequence of (source, mode, spans)
    picks up exactly where it left off rather than reseeding — cheap to do
    correctly, and avoids repeating the same corruption pattern in the tokens
    right after every resume.

    Packing
    -------
    Documents from this corpus average ~400 tokens against a 2048-token
    window, so one document per example would leave most of every batch as
    padding. `[R]` and `[X]` therefore pack several documents into one window
    (`UL2/src/packing.py`), which is safe because spans are sampled per
    document and attention is blocked across documents.

    `[S]` is never packed — see `UL2Objective.corrupt_packed`. The mode is
    drawn *before* deciding how many documents to consume, because how many to
    consume depends on it.

    Length bucketing
    ----------------
    Even one mode at a time, rows within a batch differ in length and
    `pad_batch` pads them all to the longest — 18.2% of encoder compute,
    measured. `LengthBucketedBatcher` (`UL2/src/length_batching.py`) groups
    similar-length examples into the same batch and re-randomizes the order
    afterwards; `bucket_batches=False` turns it off and restores the previous
    one-batch-at-a-time behaviour exactly.

    The cooldown
    ------------
    With `cooldown_corpus` and `cooldown_start_step` set, documents come from
    the ordinary corpus until that step and from an upweighted
    `BlendedCorpus` afterwards — the data half of the WSD schedule
    (architecture_improvements.md item 2). The step arrives through
    `set_step`, which `Trainer.train` calls once per optimizer step; nothing
    here counts batches to guess where the run is, because a guess that drifts
    by one accumulation window would move the boundary silently.

    Both streams are checkpointed, not just the active one. The cooldown
    stream has usually read nothing when a mid-run checkpoint is written, and
    the main stream is exactly where the cooldown left it — saving only the
    active one would restart the other from its first file on the next resume.

    The rewrite task
    ----------------
    With `rewrite_source` and `rewrite_share` set, a share of the micro-batches
    drawn inside the cooldown come from the synthetic rewrite task (`rewrite/`,
    architecture_improvements.md item 4) instead of the UL2 denoisers. The
    decision is made per batch, in `_batch_mode`, where the mode is already
    drawn: a rewrite batch is one more kind of batch, on its own length scale,
    exactly as `[S]` is. The task's examples never pass through the objective
    -- they arrive as complete pairs (`RewriteExampleSource`) and are only
    given the teacher-forcing shift here (`build_seq2seq_example`). Before the
    cooldown, and in a run with no rewrite task, the stream of batches is
    identical to one produced before the task existed: the extra draw only
    happens once the task is active.
    """

    def __init__(
        self,
        corpus: Any,
        objective: Any,
        specials: Any,
        *,
        batch_size: int,
        data_seed: int,
        pad_to_multiple_of: int | None = PAD_TO_MULTIPLE_OF,
        pack: bool = True,
        bucket_batches: bool = True,
        pool_batches: int = ul2.DEFAULT_POOL_BATCHES,
        capacity: "ul2.BatchCapacity | None" = None,
        cooldown_corpus: Any | None = None,
        cooldown_start_step: int | None = None,
        rewrite_source: Any | None = None,
        rewrite_share: float = 0.0,
    ) -> None:
        needs_step = cooldown_corpus is not None or rewrite_source is not None
        if needs_step != (cooldown_start_step is not None):
            raise ValueError(
                "cooldown_corpus / rewrite_source and cooldown_start_step go "
                "together: a cooldown stream or rewrite task with no step to start "
                "at would never be read, and a step with nothing to switch to "
                "would do nothing at it"
            )
        if (rewrite_source is None) != (rewrite_share <= 0.0):
            raise ValueError(
                "rewrite_source and a positive rewrite_share go together: a task "
                "with no share would never be drawn, and a share with no task has "
                "nothing to draw"
            )
        if rewrite_source is not None and not rewrite_share < 1.0:
            raise ValueError(
                f"rewrite_share must be below 1.0, got {rewrite_share}; the UL2 "
                f"denoisers need some of the cooldown too"
            )
        self.main_corpus = corpus
        self.cooldown_corpus = cooldown_corpus
        self.cooldown_start_step = cooldown_start_step
        self.rewrite_source = rewrite_source
        self.rewrite_share = float(rewrite_share)
        self._rewrite_active = False
        self.corpus = corpus
        self.objective = objective
        self.specials = specials
        self.batch_size = batch_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pack = pack
        self.rng = random.Random(data_seed)
        # Documents pulled from the corpus but not yet packed into a window.
        # Part of the saved state: dropping it on resume would silently lose
        # up to a window's worth of records at every interruption, and a
        # multi-week run is interrupted many times.
        self._buffer: list[list[int]] = []
        # Its own seed, offset from `data_seed`, so the bucketing shuffles are
        # an independent stream from the corruption draws rather than
        # interleaved with them — one less way for a change here to move the
        # corruption a run would otherwise have produced.
        if capacity is not None and not bucket_batches:
            raise ValueError(
                "capacity needs bucket_batches: a capacity is applied when a pool "
                "is cut into batches, and there is no pool without bucketing"
            )
        self.capacity = capacity
        self._batcher = (
            ul2.LengthBucketedBatcher(
                self._produce_example,
                batch_size=batch_size,
                pool_batches=pool_batches,
                capacity=capacity,
                seed=data_seed + 7919,
            )
            if bucket_batches
            else None
        )

    def __iter__(self) -> "PretrainBatchSource":
        return self

    def __next__(self) -> Mapping[str, Any]:
        mode = self._batch_mode()
        if self._batcher is not None:
            examples = self._batcher.next_batch(mode)
        else:
            examples = [self._produce_example(mode) for _ in range(self.batch_size)]
        return ul2.pad_batch(examples, self.specials, pad_to_multiple_of=self.pad_to_multiple_of)

    # -- the cooldown phase --------------------------------------------------

    def set_step(self, step: int) -> None:
        """
        Point the stream at the phase `step` belongs to.

        Called by `Trainer.train` before each optimizer step. Deriving the
        phase from the step every time, rather than latching a flag the first
        time the boundary is crossed, is what makes a resume land in the right
        phase: `trainer.resume()` restores `global_step` and the next call
        arrives with it, so a run interrupted inside the cooldown comes back
        inside the cooldown without any of that having to be checkpointed.
        """
        if self.cooldown_start_step is None:
            return
        in_decay = step >= self.cooldown_start_step
        if self.cooldown_corpus is not None:
            self.corpus = self.cooldown_corpus if in_decay else self.main_corpus
        self._rewrite_active = self.rewrite_source is not None and in_decay

    @property
    def in_cooldown(self) -> bool:
        """Whether the stream is currently drawing from the cooldown mixture."""
        return self.cooldown_corpus is not None and self.corpus is self.cooldown_corpus

    @property
    def rewrite_active(self) -> bool:
        """Whether rewrite batches can currently be drawn."""
        return self._rewrite_active

    def _produce_example(self, mode: str | None):
        """
        The next example for `mode`, skipping ones the objective rejects.

        `_next_example` returns `None` when the corrupted result does not fit
        the configured budgets. Retrying rather than propagating keeps "how
        many examples came back" equal to "how many were asked for", which is
        what both the direct path and the bucketing pool rely on.
        """
        while True:
            example = self._next_example(mode)
            if example is not None:
                return example

    def _batch_mode(self) -> str | None:
        """
        One denoising mode for the whole micro-batch, when packing.

        Why per batch rather than per example
        -------------------------------------
        The three modes produce very differently sized windows: `[R]`/`[X]`
        pack documents up to `max_source_length`, while `[S]` is never packed
        and holds a single ~400-token document. `pad_batch` pads every row to
        the longest row present, so a batch mixing them pads each `[S]` row out
        to a packed row's width — and padding costs exactly as much compute as
        real content while teaching nothing. Drawing one mode per batch keeps
        every row in it on the same length scale, which is what removes that
        waste.

        Why this does not change what the model learns
        ---------------------------------------------
        `Trainer.accumulate` weights each micro-batch's gradient by its token
        count and divides by the window total, so an accumulation window's
        gradient is exactly that of one batch containing every token in it —
        it depends on *which tokens* are in the window, never on how they were
        grouped into micro-batches (`test_accumulation_matches_one_big_batch`
        pins that invariance).

        Grouping is all this changes. Each of the 32 micro-batches in a step
        still draws independently from `mode_weights`, so the per-example mode
        distribution is unchanged and unbiased; only its per-step spread grows,
        and Adam's first moment averages that back down well inside the
        tolerance `UL2/src/config.py` describes for these ratios.

        Returns `None` for an unpacked run, leaving the objective to sample per
        example exactly as it did before packing existed — unpacked examples
        are all single documents, so they share a length scale already and have
        nothing to gain here.

        The rewrite task, when active, is decided first and at this level for
        both packed and unpacked runs: its batches are whole framed documents
        on their own length scale, and drawing them per batch is what keeps
        the per-batch share exactly `rewrite_share` rather than a mixture of
        rewrite rows inside denoising batches.
        """
        if self._rewrite_active and self.rng.random() < self.rewrite_share:
            return rewrite.REWRITE_MODE
        if not self.pack:
            return None
        return self.objective.sampler.sample(self.rng)

    def _next_rewrite_example(self):
        """
        One framed rewrite pair, as an `Example`.

        Recorded in the objective's telemetry under its own mode so the final
        table reads the task's realized share next to [R]/[X]/[S].
        `num_corrupted_tokens` carries the edit count: one edit is roughly one
        token, and it is the closest thing this task has to a corruption rate.
        """
        pair = self.rewrite_source.next_pair()
        example = ul2.build_seq2seq_example(
            pair.encoder_input_ids,
            pair.decoder_target_ids,
            rewrite.REWRITE_MODE,
            self.specials,
            source_length=pair.source_length,
            num_corrupted_tokens=pair.edits,
        )
        telemetry = getattr(self.objective, "telemetry", None)
        if telemetry is not None:
            telemetry.record(example)
        return example

    def _next_example(self, mode: str | None):
        if mode == rewrite.REWRITE_MODE:
            return self._next_rewrite_example()

        if mode is None:
            token_ids = next(self.corpus)  # StopIteration only if the corpus is empty
            return self.objective.try_corrupt(token_ids, self.rng)

        config = self.objective.config

        # [S] takes exactly one document; [R]/[X] take as many as the window
        # holds — so the count follows from the mode, which is why the mode is
        # settled before any document is consumed.
        if mode == "S":
            documents = [self._take_document()]
        else:
            self._fill_buffer(config.max_source_length, config.max_documents_per_window)
            count = ul2.pack_documents(
                self._buffer,
                window_budget=config.max_source_length,
                max_documents=config.max_documents_per_window,
            )
            documents = [self._buffer.pop(0) for _ in range(count)]

        return self.objective.try_corrupt_packed(documents, self.rng, mode=mode)

    def _next_admissible_document(self) -> list[int]:
        """
        The next document the objective will actually accept.

        Length admission has to happen **here**, before a document joins a
        window, not inside `corrupt_packed`. A window is corrupted as a unit:
        one inadmissible document among sixty-three good ones would make the
        objective reject the whole window, discarding every document in it.
        That is silent data loss of exactly the kind this project forbids —
        and the more documents packing puts in a window, the more each such
        rejection would cost.

        Skips are counted in the objective's telemetry, so a corpus that is
        mostly unusable stays visible rather than quietly shrinking the run.
        """
        config = self.objective.config
        telemetry = getattr(self.objective, "telemetry", None)

        while True:
            document = next(self.corpus)
            if len(document) < config.min_source_length:
                if telemetry is not None:
                    telemetry.record_skipped_too_short()
                continue
            if (
                len(document) > config.max_source_length
                and config.on_overlong_source == "error"
            ):
                if telemetry is not None:
                    telemetry.record_skipped_too_long()
                continue
            return document

    def _take_document(self) -> list[int]:
        if self._buffer:
            return self._buffer.pop(0)
        return self._next_admissible_document()

    def _fill_buffer(self, window_budget: int, max_documents: int) -> None:
        """
        Read until the buffer can fill one window, or holds the document cap.

        Filling past the budget rather than exactly to it is what lets
        `pack_documents` see one document too many and stop at the right
        place, instead of packing a window that merely happens to end where
        the buffer did.
        """
        total = sum(len(document) + 2 for document in self._buffer)
        while total <= window_budget and len(self._buffer) < max_documents:
            document = self._next_admissible_document()
            self._buffer.append(document)
            total += len(document) + 2

    # -- Resumable ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        version, internal_state, gauss_next = self.rng.getstate()
        state: dict[str, Any] = {
            "corpus": self.main_corpus.state_dict(),
            "rng": [version, list(internal_state), gauss_next],
            "buffer": [list(document) for document in self._buffer],
        }
        # Both phases' positions, always — see the class docstring. Keyed
        # separately from "corpus" so a checkpoint written before the cooldown
        # existed still names exactly what it holds.
        if self.cooldown_corpus is not None:
            state["cooldown_corpus"] = self.cooldown_corpus.state_dict()
        # The rewrite task's own stream and corruption RNG, for the same
        # reason as the cooldown stream: usually unread at a mid-run
        # checkpoint, and exactly where the cooldown left it after that.
        if self.rewrite_source is not None:
            state["rewrite_source"] = self.rewrite_source.state_dict()
        # Examples already built and waiting in a length pool. Saved for the
        # same reason as `buffer`: they have been read from the corpus, so
        # dropping them on resume is silent data loss (architecture.md §7.9).
        if self._batcher is not None:
            state["batcher"] = self._batcher.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.main_corpus.load_state_dict(state["corpus"])
        # `.get` so a checkpoint written before the cooldown existed still
        # loads: it simply resumes with an unread cooldown stream, which is
        # what it had. A checkpoint that *does* carry one against a run
        # configured without a cooldown is the reverse case, and is left
        # alone rather than guessed at — the stream it describes does not
        # exist here to restore into.
        if self.cooldown_corpus is not None and "cooldown_corpus" in state:
            self.cooldown_corpus.load_state_dict(state["cooldown_corpus"])
        # Same tolerance for the rewrite task: a checkpoint written before it
        # existed resumes with an unread task, which is what it had.
        if self.rewrite_source is not None and "rewrite_source" in state:
            self.rewrite_source.load_state_dict(state["rewrite_source"])
        version, internal_state, gauss_next = state["rng"]
        self.rng.setstate((version, tuple(internal_state), gauss_next))
        # `.get` so a checkpoint written before packing existed still loads;
        # it simply resumes with an empty buffer, which is what it had.
        self._buffer = [list(document) for document in state.get("buffer", [])]
        # Likewise for checkpoints written before bucketing existed: an absent
        # key means the run had nothing pooled, so an empty batcher is the
        # faithful restoration rather than a fallback.
        if self._batcher is not None and "batcher" in state:
            self._batcher.load_state_dict(state["batcher"])


def build_frozen_eval_batches(
    eval_files: Sequence[Path],
    sp: Any,
    ul2_config: Any,
    specials: Any,
    *,
    seed: int,
    batch_size: int,
    max_examples: int | None = 512,
    pack: bool = True,
) -> list[Mapping[str, Any]]:
    """
    A fixed list of padded batches from held-out text — architecture.md §7.6:
    frozen corruption, fixed equal-thirds mode mixture, built once and reused
    byte-identically for the life of the run.

    `max_examples=None` means "every record in the files", which is what the
    two-tier eval sets want: they were sized deliberately when they were
    extracted from the corpus, so capping them again here would silently
    discard the precision they were built to provide.

    Reads whole **records** (`iter_records`) and packs them exactly as
    training does, because the loss this produces is compared against training
    loss and, for the master tier, decides which checkpoint is kept. Built
    from single lines instead, it would score the model on short fragments it
    is not trained on — a number that still looks like a loss and is not one.
    """
    escape_markers = corpus_pkg.tokenizer_interop.escape_markers
    records = corpus_pkg.iter_records(list(eval_files))
    sequences: list[list[int]] = []
    for record in records:
        sequences.append(sp.encode(escape_markers(record), out_type=int))
        if max_examples is not None and len(sequences) >= max_examples:
            break

    frozen = ul2.build_frozen_eval_set(
        sequences, ul2_config, specials, seed=seed, max_examples=max_examples, pack=pack
    )
    examples = list(frozen.examples)
    return [
        ul2.pad_batch(examples[i : i + batch_size], specials, pad_to_multiple_of=8)
        for i in range(0, len(examples), batch_size)
    ]


# -- assembling a run ---------------------------------------------------------


class PipelineError(RuntimeError):
    """
    A run that cannot be assembled as configured.

    Carries an operator-facing message, because every case is something a
    person has to go and fix: a tokenizer that was never built, a blend naming
    a corpus source that is not there. `run` prints it and exits 2; the smoke
    test reports it as "could not run" rather than as a failed check.
    """


@dataclass(frozen=True)
class Pipeline:
    """
    Everything a pretraining run needs, built and cross-checked.

    This exists so `pretrain.py` and `pretrain_smoke_test.py` assemble the run
    through *one* piece of code. They used to assemble it twice, and the two
    drifted: the smoke test built the rehearsal stage on CPU with no cooldown,
    no rewrite task and no evaluation, so passing it said nothing about the
    stage that was actually about to run for weeks. A pre-flight check that
    exercises a different pipeline from the flight is not a pre-flight check.
    """

    training_config: Any
    model_config: Any
    specials: Any
    ul2_config: Any
    telemetry: Any
    objective: Any
    sp: Any
    corpus_files: list[Path]
    batch_source: "PretrainBatchSource"
    cooldown_corpus: Any | None
    rewrite_source: Any | None
    cooldown_start_step: int | None
    model: Any


def build_pipeline(
    *,
    training_config: Any,
    model_config: Any,
    corpus: str | Path,
    tokenizer_dir: str | Path,
) -> Pipeline:
    """
    Build the tokenizer, corpus, objective, data stream and model for a run.

    Takes configurations rather than loading them, so a caller that needs to
    adjust one first — the smoke test scales the cooldown down to a handful of
    steps — adjusts it before anything is built from it, not after.

    The two blended streams are built **eagerly**, here, rather than on first
    use. A blend naming a source the corpus does not have, or a token map
    without the rewrite frame tokens, has to fail before the run starts: the
    cooldown does not begin until step 180,000, and that is the single worst
    moment in the run to discover a typo.

    Raises `PipelineError` for anything an operator must fix.
    """
    tokenizer_dir = Path(tokenizer_dir)
    token_map_path = tokenizer_dir / "token_map.json"
    model_path = tokenizer_dir / "spm.model"
    if not token_map_path.is_file() or not model_path.is_file():
        raise PipelineError(
            f"tokenizer artifacts not found under {tokenizer_dir} "
            f"(need spm.model and token_map.json). Build it first — see "
            f"tokenizer/training/README.md."
        )

    # The stage owns dropout and label smoothing; the model owns the loss. The
    # trainer refuses to start if the two disagree, so they are reconciled
    # here, once, where every caller gets it.
    if training_config.dropout_rate is not None:
        model_config = model_config.with_(dropout_rate=training_config.dropout_rate)
    if model_config.label_smoothing != training_config.label_smoothing_factor:
        model_config = model_config.with_(label_smoothing=training_config.label_smoothing_factor)

    specials = ul2.SpecialTokens.from_token_map(token_map_path)
    ul2_config = ul2.UL2Config(
        max_source_length=model_config.max_encoder_length,
        max_target_length=model_config.max_decoder_length,
    )
    telemetry = ul2.MixtureTelemetry(ul2_config.mode_weights)
    objective = ul2.UL2Objective(ul2_config, specials, telemetry)

    sp = corpus_pkg.tokenizer_interop.load_processor(model_path)

    files = corpus_pkg.discover_corpus_files(corpus)
    corpus_stream = corpus_pkg.TokenizedCorpus(files, sp, seed=training_config.data_seed)

    cooldown_corpus = None
    rewrite_source = None
    grouped = None
    if training_config.cooldown_blend or training_config.rewrite_share > 0.0:
        grouped = corpus_pkg.group_files_by_source(files, corpus)
    if training_config.cooldown_blend:
        try:
            cooldown_corpus = corpus_pkg.build_blended_corpus(
                grouped,
                sp,
                training_config.cooldown_blend,
                # Offset so the cooldown's own draws are an independent stream
                # from the main corpus's file shuffle rather than a repeat of
                # it, the same reasoning as the batcher's seed below.
                seed=training_config.data_seed + 104_729,
            )
        except corpus_pkg.CorpusConfigError as exc:
            raise PipelineError(f"cooldown_blend: {exc}") from exc

    if training_config.rewrite_share > 0.0:
        try:
            rewrite_records = corpus_pkg.build_blended_corpus(
                grouped,
                sp,
                training_config.rewrite_blend,
                seed=training_config.data_seed + 15_485_863,
            )
            rewrite_source = rewrite.RewriteExampleSource(
                rewrite_records,
                rewrite.Corruptor(),
                functools.partial(corpus_pkg.tokenizer_interop.encode_text, sp),
                rewrite.RewriteTokens.from_token_map(token_map_path),
                max_source_length=ul2_config.max_source_length,
                max_target_length=ul2_config.max_target_length,
                min_source_length=ul2_config.min_source_length,
                seed=training_config.data_seed + 32_452_843,
            )
        except (corpus_pkg.CorpusConfigError, rewrite.RewriteError) as exc:
            raise PipelineError(f"rewrite task: {exc}") from exc

    # The step both halves of the cooldown key off. `None` when nothing
    # switches at it, so a pure learning-rate WSD run announces nothing.
    cooldown_start_step = (
        training_config.cooldown_start_step
        if cooldown_corpus is not None or rewrite_source is not None
        else None
    )

    # Capacity-based micro-batching, when the configuration asks for it.
    # `TrainingConfig` has already refused a half-configured set, so one
    # budget being present means both are. The padding multiple must be the
    # one `PretrainBatchSource` pads to, or the budgets would be counted
    # against a narrower rectangle than the one actually allocated.
    capacity = (
        ul2.BatchCapacity(
            max_encoder_tokens=training_config.max_batch_encoder_tokens,
            max_decoder_tokens=training_config.max_batch_decoder_tokens,
            max_rows=training_config.max_batch_rows,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        )
        if training_config.max_batch_encoder_tokens is not None
        else None
    )

    batch_source = PretrainBatchSource(
        corpus_stream,
        objective,
        specials,
        batch_size=training_config.per_device_train_batch_size,
        data_seed=training_config.data_seed,
        capacity=capacity,
        cooldown_corpus=cooldown_corpus,
        cooldown_start_step=cooldown_start_step,
        rewrite_source=rewrite_source,
        rewrite_share=training_config.rewrite_share,
    )

    # `seed` is documented to drive initialization, but `Trainer` only seeds
    # once the model already exists, so the weights must be seeded here.
    torch.manual_seed(training_config.seed)
    model = rephrase_model.build_model(model_config)

    return Pipeline(
        training_config=training_config,
        model_config=model_config,
        specials=specials,
        ul2_config=ul2_config,
        telemetry=telemetry,
        objective=objective,
        sp=sp,
        corpus_files=files,
        batch_source=batch_source,
        cooldown_corpus=cooldown_corpus,
        rewrite_source=rewrite_source,
        cooldown_start_step=cooldown_start_step,
        model=model,
    )


def load_configs(
    *,
    training_config_path: str | Path,
    model_config_path: str | Path,
    stage: str | None = None,
    model_profile: str | None = None,
    max_steps: int | None = None,
) -> tuple[Any, Any]:
    """
    The two configurations a run is built from, with any overrides applied.

    Takes the overrides as arguments rather than an `argparse.Namespace`, so a
    caller with a different set of flags — the smoke test has `--steps`, not
    `--max-steps` — can use it without having to fake the other one's.

    Note the profile precedence, which is the part that is easy to get wrong:
    an explicit `--model-profile` wins, then the stage's own `model_profile`,
    and `None` falls through to `model_config.yml`'s `active_profile`. That
    last step is what makes `--stage pretrain` build Large rather than
    silently rehearsing something smaller.
    """
    training_config = training_pkg.load_training_config(training_config_path, stage=stage)
    if max_steps is not None:
        training_config = training_config.with_(max_steps=max_steps)
    profile = model_profile or training_config.model_profile
    model_config = rephrase_model.load_model_config(model_config_path, profile=profile)
    return training_config, model_config


# -- what a run reports while it runs -----------------------------------------


def _human_duration(seconds: float) -> str:
    """`38d 01h`, `10h54m` or `3m07s` -- the two units that matter at that scale."""
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def _human_count(value: float) -> str:
    for suffix, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.2f}{suffix}"
    return f"{value:.0f}"


class ProgressLog:
    """
    One readable line per `Trainer` record, so a run shows what it is doing.

    `Trainer` already builds a record every `logging_steps` -- loss, per-mode
    loss, learning rate, gradient norm, tokens seen -- and hands it to
    `on_log`. Nothing was listening, so a multi-week run printed its banner and
    then nothing at all until it finished: from the outside, a healthy run and
    a hung one looked identical.

    Speed and time remaining are computed here, from the wall clock between
    consecutive records, rather than taken from the trainer: they describe the
    run as an operator experiences it, including data building, evaluation and
    checkpoint writes, which is what a time-remaining figure has to include to
    be worth reading.

    GPU memory is the window's peak, *allocated* and *reserved*. The gap
    between them is allocator fragmentation, which is what has run this GPU
    out of memory before with room still nominally free -- a gap that grows
    over days is the early warning.

    It never raises. It runs inside the training loop, so a formatting bug in a
    log line must not be what ends a multi-week run: a record it cannot format
    is written out raw, with the reason, and training carries on.
    """

    def __init__(
        self,
        *,
        max_steps: int,
        warmup_steps: int,
        cooldown_start_step: int | None,
        save_steps: int,
        output_dir: str | Path,
        device_type: str,
        clock: Any = time.monotonic,
        now: Any = datetime.now,
        write: Any = print,
    ) -> None:
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.cooldown_start_step = cooldown_start_step
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.device_type = device_type
        self._clock = clock
        self._now = now
        self._write = write
        self._start_step: int | None = None
        self._start_time: float | None = None
        self._last: tuple[int, float, float] | None = None

    def start(self, step: int, tokens_seen: float, input_tokens_seen: float = 0.0) -> None:
        """Mark where this process begins, which is not step 0 after a resume."""
        moment = self._clock()
        self._start_step, self._start_time = step, moment
        self._last = (step, moment, float(tokens_seen) + float(input_tokens_seen))
        how = "resumed" if step else "fresh start"
        self._emit(
            f"{self._stamp()} | training from step {step:,} of {self.max_steps:,} ({how})"
        )

    def __call__(self, record: Mapping[str, Any]) -> None:
        try:
            text = self.format(record)
        except Exception as exc:  # see the class docstring: never end the run
            text = (
                f"{self._stamp()} | log record could not be formatted "
                f"({type(exc).__name__}: {exc}): {dict(record)!r}"
            )
        self._emit(text)

    def format(self, record: Mapping[str, Any]) -> str:
        if record.get("eval"):
            return self._format_eval(record)
        if record.get("finished"):
            return self._format_finished(record)
        return self._format_step(record)

    # -- the three kinds of record -------------------------------------------

    def _format_step(self, record: Mapping[str, Any]) -> str:
        step = int(record["step"])
        moment = self._clock()
        trained_on = self._trained_on(record)

        parts = [
            self._stamp(),
            f"step {step:>6,}/{self.max_steps:,} ({100 * step / self.max_steps:5.2f}%) "
            f"{self._phase(step)}",
            self._losses(record),
            f"lr {float(record['learning_rate']):.2e}  gnorm {float(record['grad_norm']):.3f}",
        ]

        if self._last is not None:
            last_step, last_time, last_tokens = self._last
            steps, seconds = step - last_step, moment - last_time
            if steps > 0 and seconds > 0:
                parts.append(
                    f"{seconds / steps:6.2f} s/step  "
                    f"{(trained_on - last_tokens) / seconds:,.0f} tok/s"
                )
        parts.append(self._token_totals(record))

        memory = self._memory(record)
        if memory:
            parts.append(memory)
        if record.get("skipped_steps"):
            parts.append(f"skipped {int(record['skipped_steps'])}")
        if self._start_time is not None and self._start_step is not None:
            done = step - self._start_step
            elapsed = moment - self._start_time
            timing = f"elapsed {_human_duration(elapsed)}"
            if done > 0:
                remaining = (self.max_steps - step) * elapsed / done
                timing += f"  eta {_human_duration(remaining)}"
            parts.append(timing)

        self._last = (step, moment, trained_on)
        line = " | ".join(parts)

        # Written after the trainer logs this step, so the line says what is
        # about to happen rather than what has: a failed write shows up next
        # in the log as a traceback, directly under this line.
        if self.save_steps and step % self.save_steps == 0:
            line += (
                f"\n{self._stamp()} | checkpoint: saving "
                f"{self.output_dir / f'checkpoint-{step}'}"
            )
        return line

    def _format_eval(self, record: Mapping[str, Any]) -> str:
        tier = record.get("eval_tier", "eval")
        return (
            f"{self._stamp()} | eval [{tier}] at step {int(record['step']):,} | "
            f"{self._losses(record)}"
        )

    def _format_finished(self, record: Mapping[str, Any]) -> str:
        elapsed = (
            f" | ran {_human_duration(self._clock() - self._start_time)}"
            if self._start_time is not None
            else ""
        )
        return (
            f"{self._stamp()} | finished at step {int(record['step']):,} "
            f"({record.get('reason', 'unknown')}) | "
            f"{self._token_totals(record)} | "
            f"skipped {int(record.get('skipped_steps', 0))}{elapsed}"
        )

    # -- pieces --------------------------------------------------------------

    @staticmethod
    def _trained_on(record: Mapping[str, Any]) -> float:
        return float(record.get("tokens_seen", 0)) + float(record.get("input_tokens_seen", 0))

    def _token_totals(self, record: Mapping[str, Any]) -> str:
        """Running totals since step 0, all unpadded: input + graded = trained on."""
        return (
            f"tokens: input {_human_count(float(record.get('input_tokens_seen', 0)))}"
            f"  graded {_human_count(float(record.get('tokens_seen', 0)))}"
            f"  trained on {_human_count(self._trained_on(record))}"
        )

    def _losses(self, record: Mapping[str, Any]) -> str:
        text = f"loss {float(record['loss']):.4f}  ppl {float(record['perplexity']):,.1f}"
        modes = [
            f"{mode} {float(record[f'loss_{mode}']):.4f}"
            for mode in ("R", "X", "S")
            if f"loss_{mode}" in record
        ]
        return f"{text}  [{'  '.join(modes)}]" if modes else text

    def _phase(self, step: int) -> str:
        if step <= self.warmup_steps:
            return "warmup"
        if self.cooldown_start_step is not None and step > self.cooldown_start_step:
            return "cooldown"
        return "stable"

    def _memory(self, record: Mapping[str, Any]) -> str:
        # Measured (and reset) by `Trainer` when it builds the record, so this
        # line and the checkpoint's `log_history` show the same number. Reading
        # the counters here too would see them just after that reset.
        if self.device_type != "cuda" or "gpu_peak_allocated_gib" not in record:
            return ""
        allocated = float(record["gpu_peak_allocated_gib"])
        reserved = float(record["gpu_peak_reserved_gib"])
        return f"gpu peak {allocated:.1f} alloc / {reserved:.1f} reserved GiB"

    def _stamp(self) -> str:
        return self._now().strftime("%Y-%m-%d %H:%M:%S")

    def _emit(self, text: str) -> None:
        try:
            self._write(text)
        except OSError:
            # Nowhere left to report it: the terminal itself is gone. The run
            # is worth more than the line.
            pass


class _LogFile:
    """
    The run's log file, appended to and flushed on every write.

    Flushed every time because the moment the log matters most is the moment
    the process dies, and a buffer that was never written is exactly the part
    that explains why. It is a few writes every half hour; the cost is nil.

    A write that fails -- a full disk, a removed mount -- disables the file and
    says so once on the real stderr. Losing the log is bad; letting a logging
    failure raise out of a `print` inside the training loop and end the run is
    worse.
    """

    def __init__(self, path: Path, stderr: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stderr = stderr
        self._handle: Any = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(text)
            self._handle.flush()
        except OSError as exc:
            self._handle = None
            try:
                self._stderr.write(f"\n[run log] writing {self.path} failed ({exc}); "
                                   f"continuing without it\n")
            except OSError:
                pass

    def flush(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class _Tee:
    """A text stream that writes to the terminal and to the run log."""

    def __init__(self, stream: Any, log: _LogFile) -> None:
        self._stream = stream
        self._log = log

    def write(self, text: str) -> int:
        written = self._stream.write(text)
        self._log.write(text)
        return written

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _RunOutcome:
    """How the run ended, filled in by `run` for the log's closing line."""

    def __init__(self) -> None:
        self.text = "ended"


@contextmanager
def run_log(path: str | Path | None, argv: Sequence[str]):
    """
    Mirror everything the run prints -- stdout and stderr -- into `path`.

    Everything, rather than only the progress lines, because the lines that
    explain a failure are rarely the ones anybody chose to log: the startup
    banner saying which stage and corpus actually ran, a warning from torch, a
    traceback. The file is appended to, so a resumed run continues the same
    log, and every session inside it is bracketed by a header and a closing
    line saying how it ended.

    `None` leaves output alone, which is what every test and the smoke test
    rely on.
    """
    outcome = _RunOutcome()
    if path is None:
        yield outcome
        return

    stdout, stderr = sys.stdout, sys.stderr
    log = _LogFile(Path(path), stderr)
    rule = "=" * 100
    log.write(
        f"\n{rule}\n"
        f"pretrain.py started {datetime.now():%Y-%m-%d %H:%M:%S}  "
        f"(pid {os.getpid()}, host {socket.gethostname()})\n"
        f"cwd: {os.getcwd()}\n"
        f"command: {' '.join(argv)}\n"
        f"{rule}\n"
    )
    sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
    try:
        yield outcome
    except KeyboardInterrupt:
        outcome.text = "interrupted (KeyboardInterrupt)"
        raise
    except BaseException as exc:
        outcome.text = f"stopped by {type(exc).__name__}: {exc}"
        log.write(traceback.format_exc())
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log.write(f"{rule}\n{datetime.now():%Y-%m-%d %H:%M:%S} | {outcome.text}\n{rule}\n")
        log.close()


# -- the dashboard ------------------------------------------------------------

# Held-out rows the dashboard's probes draw from: enough for four [S] rows to
# shuffle and one sample per mode, read from the head of the eval corpus.
PROBE_EXAMPLES = 64

# Cadence for the Text tab's `samples/<mode>` generations alone (no position
# probes), so recent model output stays visible between quick-eval cycles.
SAMPLE_TEXT_STEPS = 500


def build_monitor(args: argparse.Namespace, trainer: Any, pipeline: Pipeline) -> Any:
    """
    The run's TensorBoard dashboard (`monitoring/`), for `--tensorboard-dir`.

    The package is loaded here rather than at import, so a run without the
    flag needs neither TensorBoard nor NVML installed. The slow components —
    histograms and model probes — share the quick eval's cadence: the run
    already pauses there, and the two are read together.

    Probe rows are built unpacked, from the first held-out corpus given, with
    their own corruption seed; without one, the probes that need text are
    skipped and the weight-only ones still run.

    The `samples/<mode>` generations alone run on the tighter `SAMPLE_TEXT_STEPS`
    cadence (skipped without probe rows, same as the full probe), so the Text
    tab tracks recent output between quick-eval cycles without paying for the
    position probes that often.
    """
    monitoring = _load("echomebetter_monitoring", MONITORING_SRC)
    config = pipeline.training_config
    specials = pipeline.specials

    examples = []
    probe_corpus = args.quick_eval_corpus or args.master_eval_corpus or args.eval_corpus
    if probe_corpus is not None:
        batches = build_frozen_eval_batches(
            corpus_pkg.discover_corpus_files(probe_corpus),
            pipeline.sp,
            pipeline.ul2_config,
            specials,
            seed=config.data_seed + 3,
            batch_size=config.per_device_train_batch_size,
            max_examples=PROBE_EXAMPLES,
            pack=False,
        )
        examples = monitoring.probe_examples(batches, label_pad_id=specials.label_pad_id)

    autocast = config.bf16 and trainer.device.type in ("cuda", "cpu")
    probes = monitoring.ModelProbes(
        trainer.model,
        generate=rephrase_model.greedy_generate,
        decode=pipeline.sp.decode,
        examples=examples,
        special_ids={
            *specials.sentinel_ids,
            *specials.mode_token_ids.values(),
            specials.eos_id,
            specials.pad_id,
        },
        eos_id=specials.eos_id,
        amp_dtype=torch.bfloat16 if autocast else None,
    )

    gpu = None
    if trainer.device.type == "cuda":
        try:
            gpu = monitoring.GpuSampler.for_device(trainer.device)
        except Exception as exc:  # the GPU panel is optional; the run is not
            print(f"tensorboard: GPU readings unavailable ({type(exc).__name__}: {exc})")

    monitor = monitoring.TrainingMonitor(
        args.tensorboard_dir,
        max_steps=config.max_steps,
        start_step=trainer.state.global_step,
        start_tokens_seen=trainer.state.tokens_seen,
        start_input_tokens_seen=trainer.state.input_tokens_seen,
        optimizer_stats=monitoring.OptimizerStats(
            trainer.optimizer, histogram_every_steps=config.quick_eval_steps
        ),
        gpu=gpu,
        probes=probes,
        probe_every=config.quick_eval_steps,
        samples_every=min(SAMPLE_TEXT_STEPS, config.quick_eval_steps) if examples else None,
    )
    print(
        f"tensorboard: {args.tensorboard_dir}  ({len(examples)} probe rows; "
        f"view with: tensorboard --logdir {args.tensorboard_dir})"
    )
    return monitor


def _fan_out(*sinks: Any) -> Any:
    """One `on_log` that hands every record to each sink in turn."""

    def on_log(record: Mapping[str, Any]) -> None:
        for sink in sinks:
            sink(record)

    return on_log


# -- the run ------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    with run_log(args.log_file, sys.argv) as outcome:
        status = _run(args)
        outcome.text = f"exited with status {status}"
        return status


def _run(args: argparse.Namespace) -> int:
    training_config, model_config = load_configs(
        training_config_path=args.training_config,
        model_config_path=args.model_config,
        stage=args.stage,
        model_profile=args.model_profile,
        max_steps=args.max_steps,
    )

    try:
        pipeline = build_pipeline(
            training_config=training_config,
            model_config=model_config,
            corpus=args.corpus,
            tokenizer_dir=args.tokenizer_dir,
        )
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Unpacked rather than reached through `pipeline.` below, because what
    # follows is the run's own narrative — the eval tiers, the summary it
    # prints, the training call — and `pipeline.` on every line of it would
    # bury that under the name of the thing it came from. Note `model_config`
    # is the *reconciled* one: `build_pipeline` may have applied the stage's
    # dropout and label smoothing to it, and the summary must print what the
    # model was actually built with.
    model_config = pipeline.model_config
    specials = pipeline.specials
    ul2_config = pipeline.ul2_config
    telemetry = pipeline.telemetry
    sp = pipeline.sp
    files = pipeline.corpus_files
    batch_source = pipeline.batch_source
    cooldown_corpus = pipeline.cooldown_corpus
    rewrite_source = pipeline.rewrite_source
    cooldown_start_step = pipeline.cooldown_start_step

    tiered = args.master_eval_corpus is not None or args.quick_eval_corpus is not None
    if args.eval_corpus is not None and tiered:
        print(
            "--eval-corpus cannot be combined with --master-eval-corpus or "
            "--quick-eval-corpus: --eval-corpus is the single-tier form and also "
            "claims authority over best-checkpoint selection, so which held-out "
            "set ranked the checkpoints would depend on nothing visible in the "
            "command. Pass the tiered flags alone.",
            file=sys.stderr,
        )
        return 2

    def frozen_eval_callable(path: str, max_examples: int | None, seed_offset: int):
        """
        Build one tier's frozen batches now, and return the callable that
        scores them later.

        Built eagerly — a missing or unreadable eval corpus should fail before
        the run starts, not twenty thousand steps into it. Each tier gets its
        own corruption seed so the two are independent samples rather than the
        same corruption pattern applied to different text.
        """
        batches = build_frozen_eval_batches(
            corpus_pkg.discover_corpus_files(path),
            sp,
            ul2_config,
            specials,
            seed=training_config.data_seed + seed_offset,
            batch_size=training_config.per_device_train_batch_size,
            max_examples=max_examples,
        )

        def score() -> Mapping[str, float]:
            return trainer.evaluate(batches, max_batches=training_config.eval_max_batches)

        return score, len(batches)

    evaluate = None
    eval_tiers: list[Any] = []

    if args.eval_corpus is not None:
        evaluate, _ = frozen_eval_callable(args.eval_corpus, args.eval_max_examples, 1)

    if args.master_eval_corpus is not None:
        # The master tier is the one that decides which checkpoint is kept, so
        # a cadence longer than the run means no checkpoint is ever selected —
        # and nothing says so. The run finishes, `best_step` is None, and the
        # model you keep is whichever one happened to be last. Refused here
        # rather than in `TrainingConfig` because it is only wrong when a
        # master corpus is actually supplied: a run with no eval tiers leaves
        # this cadence inert, and the smoke test relies on that.
        if training_config.master_eval_steps > training_config.max_steps:
            raise PipelineError(
                f"master_eval_steps={training_config.master_eval_steps} exceeds "
                f"max_steps={training_config.max_steps}, so the tier that selects "
                f"the best checkpoint would never run and the run would end with "
                f"none chosen. Lower master_eval_steps (it must stay a multiple of "
                f"save_steps={training_config.save_steps}), or drop "
                f"--master-eval-corpus."
            )
        score, count = frozen_eval_callable(
            args.master_eval_corpus, args.master_eval_max_examples, 1
        )
        eval_tiers.append(
            training_pkg.EvalTier(
                "master",
                score,
                training_config.master_eval_steps,
                drives_best_checkpoint=True,
            )
        )
        print(
            f"master eval: {args.master_eval_corpus}  ({count} batch(es), "
            f"every {training_config.master_eval_steps} steps, selects the best checkpoint)"
        )

    if args.quick_eval_corpus is not None:
        score, count = frozen_eval_callable(
            args.quick_eval_corpus, args.quick_eval_max_examples, 2
        )
        eval_tiers.append(
            training_pkg.EvalTier(
                "quick",
                score,
                training_config.quick_eval_steps,
                drives_best_checkpoint=False,
            )
        )
        print(
            f"quick eval:  {args.quick_eval_corpus}  ({count} batch(es), "
            f"every {training_config.quick_eval_steps} steps, trend only)"
        )

    print(f"model profile: {model_config.profile}  ({model_config.parameter_count():,} params)")
    print(f"stage: {training_config.stage}  max_steps: {training_config.max_steps}")
    print(
        f"schedule: {training_config.lr_scheduler_type}  peak lr "
        f"{training_config.learning_rate}  warmup "
        f"{training_config.resolved_warmup_steps}"
        + (
            f"  decay {training_config.resolved_decay_steps} "
            f"({training_config.lr_decay_type}, from step "
            f"{training_config.cooldown_start_step})"
            if training_config.cooldown_start_step is not None
            else ""
        )
    )
    if cooldown_corpus is not None:
        shares = ", ".join(
            f"{source} {share:.1%}"
            for source, share in sorted(
                cooldown_corpus.weights.items(), key=lambda item: -item[1]
            )
        )
        print(f"cooldown mixture (from step {cooldown_start_step}): {shares}")
    if rewrite_source is not None:
        shares = ", ".join(
            f"{source} {share:.1%}"
            for source, share in sorted(
                rewrite_source.records.weights.items(), key=lambda item: -item[1]
            )
        )
        print(
            f"rewrite task (from step {cooldown_start_step}): "
            f"{training_config.rewrite_share:.0%} of batches, "
            f"{rewrite_source.tokens.style_token}, sources {shares}"
        )
    print(f"corpus: {args.corpus}  ({len(files)} file(s))")
    # Stated up front because it governs how much real content each step
    # actually carries: records are packed into windows for [R]/[X], while [S]
    # keeps one document per window (UL2/src/packing.py).
    print(
        f"packing: up to {ul2_config.max_documents_per_window} documents per "
        f"{ul2_config.max_source_length}-token window  ([S] stays unpacked)"
    )
    print(
        f"length bucketing: pools of {ul2.DEFAULT_POOL_BATCHES} batches "
        f"({ul2.DEFAULT_POOL_BATCHES * training_config.per_device_train_batch_size} "
        f"examples) per mode, sorted then reshuffled"
    )

    trainer = training_pkg.Trainer(pipeline.model, training_config, device=args.device)
    progress = ProgressLog(
        max_steps=training_config.max_steps,
        warmup_steps=training_config.resolved_warmup_steps,
        cooldown_start_step=training_config.cooldown_start_step,
        save_steps=training_config.save_steps,
        output_dir=training_config.output_dir,
        device_type=trainer.device.type,
    )

    if training_config.resume_from_checkpoint and training_pkg.latest_checkpoint(
        training_config.output_dir
    ):
        trainer.resume(data=batch_source)
        print(f"resumed from step {trainer.state.global_step}")

    # Built after any resume: the dashboard continues from the resumed step.
    monitor = build_monitor(args, trainer, pipeline) if args.tensorboard_dir else None
    trainer.on_log = progress if monitor is None else _fan_out(progress, monitor)

    progress.start(
        trainer.state.global_step,
        trainer.state.tokens_seen,
        trainer.state.input_tokens_seen,
    )
    try:
        trainer.train(
            batch_source,
            evaluate=evaluate,
            eval_tiers=eval_tiers or None,
            data=batch_source,
        )
    finally:
        if monitor is not None:
            monitor.close()

    print()
    print(telemetry.format_table())

    if cooldown_corpus is not None and any(cooldown_corpus.documents_drawn.values()):
        # What the cooldown actually delivered, against what it was asked for.
        # `epochs` is the number the mixture is easiest to get wrong on: an
        # upweighted small source is *meant* to repeat, and this says how
        # often it did rather than leaving it to be inferred from the weights.
        realized = cooldown_corpus.realized_shares()
        epochs = cooldown_corpus.source_epochs()
        print()
        print("cooldown mixture, as realized:")
        print(f"  {'source':<28}{'target':>9}{'actual':>9}{'epochs':>9}")
        for source in sorted(cooldown_corpus.weights, key=lambda s: -cooldown_corpus.weights[s]):
            print(
                f"  {source:<28}{cooldown_corpus.weights[source]:>8.2%}"
                f"{realized[source]:>9.2%}{epochs[source]:>9,}"
            )

    if rewrite_source is not None and rewrite_source.pairs_emitted:
        print()
        print(rewrite_source.format_table())
        realized = rewrite_source.records.realized_shares()
        print(f"  {'source':<28}{'target':>9}{'actual':>9}")
        for source in sorted(realized, key=lambda s: -rewrite_source.records.weights[s]):
            print(
                f"  {source:<28}{rewrite_source.records.weights[source]:>8.2%}"
                f"{realized[source]:>9.2%}"
            )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Training corpus: a file or directory.")
    parser.add_argument(
        "--eval-corpus",
        default=None,
        help=(
            "Held-out corpus for §7.6 frozen eval, single-tier form: evaluated "
            "every eval_steps and used to select the best checkpoint. Mutually "
            "exclusive with the two-tier flags below."
        ),
    )
    parser.add_argument("--eval-max-examples", type=int, default=512)
    parser.add_argument(
        "--master-eval-corpus",
        default=None,
        help=(
            "Large held-out set (master_eval.jsonl). Evaluated every "
            "master_eval_steps and is the ONLY tier that decides which "
            "checkpoint is kept as best."
        ),
    )
    parser.add_argument(
        "--quick-eval-corpus",
        default=None,
        help=(
            "Small held-out set (quick_eval.jsonl). Evaluated every "
            "quick_eval_steps as a trend signal; never selects a checkpoint."
        ),
    )
    parser.add_argument(
        "--master-eval-max-examples",
        type=int,
        default=None,
        help="Default: every record in the file, which is how the set was sized.",
    )
    parser.add_argument(
        "--quick-eval-max-examples",
        type=int,
        default=None,
        help="Default: every record in the file.",
    )
    parser.add_argument("--stage", default=None, help="training_config.yml stage (default: its active_stage).")
    parser.add_argument("--model-profile", default=None, help="model_config.yml profile override.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the stage's max_steps.")
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "torch device (cpu/cuda/mps). Default: auto-detect. On Apple Silicon, "
            "prefer --device cpu explicitly — training/src/metrics.py accumulates "
            "loss statistics in float64, which MPS does not support."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Append everything the run prints -- banner, a progress line every "
            "logging_steps, evaluations, tracebacks -- to this file as well as the "
            "terminal, e.g. pretrain_corpus/logs/pretrain.log. Default: terminal only."
        ),
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help=(
            "Write a TensorBoard dashboard for the run to this directory, e.g. "
            "runs/pretrain/tensorboard, alongside the console log. View with "
            "`tensorboard --logdir DIR`. Default: no dashboard."
        ),
    )
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--training-config", default=str(DEFAULT_TRAINING_CONFIG))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
