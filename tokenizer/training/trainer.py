"""
Thin wrapper around SentencePieceTrainer (DESIGN.md 8).

Deliberately thin: it translates the validated config into trainer keyword
arguments and does nothing else. All tunable values come from the YAML; the
only values hardcoded here are the two that are structural invariants of the
design rather than knobs (see `_FIXED_BY_DESIGN`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import sentencepiece as spm

from config_loader import resolve_num_threads
from specials import USER_DEFINED_SYMBOLS

# Not configurable, by design (DESIGN.md 7): corpus_reader.py has already
# done the seeded sampling and shuffling, so SentencePiece must consume
# corpus.txt whole, in fixed order. Letting these be set from YAML would
# reintroduce SentencePiece's unseeded internal shuffle and make builds
# non-reproducible (risk R7).
_FIXED_BY_DESIGN: dict[str, Any] = {
    "input_sentence_size": 0,  # 0 = use every line of corpus.txt
    "shuffle_input_sentence": False,  # we already shuffled, seeded
}


def build_trainer_kwargs(
    cfg: dict, corpus_path: Path, model_prefix: Path, *, verbose: bool = False
) -> dict[str, Any]:
    """
    Assemble the exact kwargs passed to SentencePieceTrainer.train().

    `verbose` only controls SentencePiece's own logging (it prints several
    hundred INFO lines, including one per byte-fallback piece, which buries
    this module's own progress output). It has no effect on the vocabulary.
    """
    t = cfg["trainer"]
    kwargs: dict[str, Any] = {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix),
        "model_type": t["model_type"],
        "vocab_size": t["vocab_size"],
        # fidelity-critical (NF1)
        "byte_fallback": t["byte_fallback"],
        "character_coverage": t["character_coverage"],
        "normalization_rule_name": t["normalization_rule_name"],
        "remove_extra_whitespaces": t["remove_extra_whitespaces"],
        "allow_whitespace_only_pieces": t["allow_whitespace_only_pieces"],
        "add_dummy_prefix": t["add_dummy_prefix"],
        "split_digits": t["split_digits"],
        # scale / memory
        "max_sentence_length": cfg["corpus"]["max_sentence_length"],
        "train_extremely_large_corpus": t["train_extremely_large_corpus"],
        "num_threads": resolve_num_threads(t["num_threads"]),
        # ids and specials
        "pad_id": t["pad_id"],
        "eos_id": t["eos_id"],
        "unk_id": t["unk_id"],
        "bos_id": t["bos_id"],
        "user_defined_symbols": list(USER_DEFINED_SYMBOLS),
        # 0 = INFO and above, 2 = WARNING and above. Logging only.
        "minloglevel": 0 if verbose else 2,
    }
    kwargs.update(_FIXED_BY_DESIGN)
    return kwargs


def train(cfg: dict, corpus_path: Path, output_dir: Path, *, verbose: bool = False) -> dict:
    """
    Fit the tokenizer. Returns run metadata for the manifest.

    Raises RuntimeError if SentencePiece does not produce both expected
    artifacts, so a partial failure can never be mistaken for success.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / cfg["output"]["model_prefix"]
    kwargs = build_trainer_kwargs(cfg, corpus_path, model_prefix, verbose=verbose)

    started = time.time()
    spm.SentencePieceTrainer.train(**kwargs)
    elapsed = time.time() - started

    model_path = Path(f"{model_prefix}.model")
    vocab_path = Path(f"{model_prefix}.vocab")
    for path in (model_path, vocab_path):
        if not path.is_file():
            raise RuntimeError(f"SentencePiece finished but did not write {path}")

    return {
        "trainer_kwargs": {k: v for k, v in kwargs.items() if k != "user_defined_symbols"},
        "user_defined_symbols_count": len(kwargs["user_defined_symbols"]),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "elapsed_seconds": round(elapsed, 2),
        "sentencepiece_version": spm.__version__,
    }


def load_processor(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp
