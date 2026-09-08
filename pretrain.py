#!/usr/bin/env python3
"""
Stage 1 entrypoint — wires `corpus/`, `UL2/`, `model/` and `training/`
together into an actual pretraining run (architecture.md §7).

This is the piece the four packages' own READMEs describe but none of them
owns: `corpus/` reads text, `UL2/` corrupts it, `model/` scores it, and
`training/` drives the loop — each package is deliberately ignorant of the
other three, so something outside all of them has to do the wiring. This
script is that something, and nothing more: no new corruption logic, no new
training logic, no new model logic.

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
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_SRC = PROJECT_ROOT / "corpus" / "src"
UL2_SRC = PROJECT_ROOT / "UL2" / "src"
TRAINING_SRC = PROJECT_ROOT / "training" / "src"
MODEL_SRC = PROJECT_ROOT / "model" / "src"

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
    """

    def __init__(
        self,
        corpus: Any,
        objective: Any,
        specials: Any,
        *,
        batch_size: int,
        data_seed: int,
        pad_to_multiple_of: int | None = 8,
        pack: bool = True,
        bucket_batches: bool = True,
        pool_batches: int = ul2.DEFAULT_POOL_BATCHES,
    ) -> None:
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
        self._batcher = (
            ul2.LengthBucketedBatcher(
                self._produce_example,
                batch_size=batch_size,
                pool_batches=pool_batches,
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
        """
        if not self.pack:
            return None
        return self.objective.sampler.sample(self.rng)

    def _next_example(self, mode: str | None):
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
            "corpus": self.corpus.state_dict(),
            "rng": [version, list(internal_state), gauss_next],
            "buffer": [list(document) for document in self._buffer],
        }
        # Examples already built and waiting in a length pool. Saved for the
        # same reason as `buffer`: they have been read from the corpus, so
        # dropping them on resume is silent data loss (architecture.md §7.9).
        if self._batcher is not None:
            state["batcher"] = self._batcher.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.corpus.load_state_dict(state["corpus"])
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


# -- the run ------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    tokenizer_dir = Path(args.tokenizer_dir)
    token_map_path = tokenizer_dir / "token_map.json"
    model_path = tokenizer_dir / "spm.model"
    if not token_map_path.is_file() or not model_path.is_file():
        print(
            f"tokenizer artifacts not found under {tokenizer_dir} "
            f"(need spm.model and token_map.json). Build it first — see "
            f"tokenizer/training/README.md.",
            file=sys.stderr,
        )
        return 2

    training_config = training_pkg.load_training_config(args.training_config, stage=args.stage)
    if args.max_steps is not None:
        training_config = training_config.with_(max_steps=args.max_steps)

    model_profile = args.model_profile or training_config.model_profile
    model_config = rephrase_model.load_model_config(args.model_config, profile=model_profile)
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

    files = corpus_pkg.discover_corpus_files(args.corpus)
    corpus_stream = corpus_pkg.TokenizedCorpus(files, sp, seed=training_config.data_seed)
    batch_source = PretrainBatchSource(
        corpus_stream,
        objective,
        specials,
        batch_size=training_config.per_device_train_batch_size,
        data_seed=training_config.data_seed,
    )

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

    model = rephrase_model.build_model(model_config)
    trainer = training_pkg.Trainer(model, training_config, device=args.device)

    if training_config.resume_from_checkpoint and training_pkg.latest_checkpoint(
        training_config.output_dir
    ):
        trainer.resume(data=batch_source)
        print(f"resumed from step {trainer.state.global_step}")

    trainer.train(
        batch_source,
        evaluate=evaluate,
        eval_tiers=eval_tiers or None,
        data=batch_source,
    )

    print()
    print(telemetry.format_table())
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
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--training-config", default=str(DEFAULT_TRAINING_CONFIG))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
