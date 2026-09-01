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
    itself `Resumable` (`training/src/checkpoint.py`): resuming restores both
    the corpus position *and* the corruption RNG, so a resumed run's sequence
    of (source, mode, spans) picks up exactly where it left off rather than
    reseeding — cheap to do correctly, and avoids repeating the same
    corruption pattern in the tokens right after every resume.
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
    ) -> None:
        self.corpus = corpus
        self.objective = objective
        self.specials = specials
        self.batch_size = batch_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.rng = random.Random(data_seed)

    def __iter__(self) -> "PretrainBatchSource":
        return self

    def __next__(self) -> Mapping[str, Any]:
        examples = []
        while len(examples) < self.batch_size:
            token_ids = next(self.corpus)  # raises StopIteration only if corpus is empty
            example = self.objective.try_corrupt(token_ids, self.rng)
            if example is not None:
                examples.append(example)
        return ul2.pad_batch(examples, self.specials, pad_to_multiple_of=self.pad_to_multiple_of)

    # -- Resumable ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        version, internal_state, gauss_next = self.rng.getstate()
        return {
            "corpus": self.corpus.state_dict(),
            "rng": [version, list(internal_state), gauss_next],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.corpus.load_state_dict(state["corpus"])
        version, internal_state, gauss_next = state["rng"]
        self.rng.setstate((version, tuple(internal_state), gauss_next))


def build_frozen_eval_batches(
    eval_files: Sequence[Path],
    sp: Any,
    ul2_config: Any,
    specials: Any,
    *,
    seed: int,
    batch_size: int,
    max_examples: int = 512,
) -> list[Mapping[str, Any]]:
    """
    A fixed list of padded batches from held-out text — architecture.md §7.6:
    frozen corruption, fixed equal-thirds mode mixture, built once and reused
    byte-identically for the life of the run.
    """
    escape_markers = corpus_pkg.tokenizer_interop.escape_markers
    lines = corpus_pkg.iter_lines(list(eval_files))
    sequences: list[list[int]] = []
    for line in lines:
        sequences.append(sp.encode(escape_markers(line), out_type=int))
        if len(sequences) >= max_examples:
            break

    frozen = ul2.build_frozen_eval_set(
        sequences, ul2_config, specials, seed=seed, max_examples=max_examples
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

    evaluate = None
    if args.eval_corpus is not None:
        eval_files = corpus_pkg.discover_corpus_files(args.eval_corpus)
        eval_batches = build_frozen_eval_batches(
            eval_files,
            sp,
            ul2_config,
            specials,
            seed=training_config.data_seed + 1,
            batch_size=training_config.per_device_train_batch_size,
            max_examples=args.eval_max_examples,
        )

        def evaluate() -> Mapping[str, float]:  # noqa: F811
            return trainer.evaluate(eval_batches, max_batches=training_config.eval_max_batches)

    print(f"model profile: {model_config.profile}  ({model_config.parameter_count():,} params)")
    print(f"stage: {training_config.stage}  max_steps: {training_config.max_steps}")
    print(f"corpus: {args.corpus}  ({len(files)} file(s))")

    model = rephrase_model.build_model(model_config)
    trainer = training_pkg.Trainer(model, training_config, device=args.device)

    if training_config.resume_from_checkpoint and training_pkg.latest_checkpoint(
        training_config.output_dir
    ):
        trainer.resume(data=batch_source)
        print(f"resumed from step {trainer.state.global_step}")

    trainer.train(batch_source, evaluate=evaluate, data=batch_source)

    print()
    print(telemetry.format_table())
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Training corpus: a file or directory.")
    parser.add_argument("--eval-corpus", default=None, help="Held-out corpus for §7.6 frozen eval.")
    parser.add_argument("--eval-max-examples", type=int, default=512)
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
