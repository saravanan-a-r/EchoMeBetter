"""
Pretraining corpus pipeline (architecture.md §7, §7.9).

Scope
-----
Turns files on disk into the resumable stream of token-ID lines
`UL2Objective.try_corrupt` consumes. Does not corrupt, batch, or train —
those belong to `UL2/` and `training/`.

Two streams, not one. `TokenizedCorpus` reads the corpus as it lies on disk
and is what the bulk of a run uses. `BlendedCorpus` draws from several
per-source streams in configured proportions, and exists for one job: the WSD
cooldown (architecture_improvements.md item 2), whose final ~10% of steps run
on a deliberately upweighted mixture. Neither decides *what* the shares should
be — that is a training-configuration question (`cooldown_blend` in
`training_config.yml`), and fitting the tokenizer's own blend is a different,
whole-corpus-in-view problem belonging to
`tokenizer/training/corpus_reader.py`.

Typical use
-----------
    from src import discover_corpus_files, TokenizedCorpus
    from src.tokenizer_interop import load_processor, DEFAULT_TOKENIZER_DIR

    sp = load_processor(DEFAULT_TOKENIZER_DIR / "spm.model")
    files = discover_corpus_files("path/to/corpus")
    corpus = TokenizedCorpus(files, sp, seed=1234)   # a Resumable

    for token_ids in corpus:
        example = objective.try_corrupt(token_ids, rng)
        ...
"""

from __future__ import annotations

from .blended_corpus import BlendedCorpus, build_blended_corpus
from .errors import CorpusConfigError, CorpusError
from .reader import (
    discover_corpus_files,
    extract_text,
    group_files_by_source,
    is_jsonl,
    iter_lines,
    iter_records,
    join_record_lines,
    source_name,
)
from .tokenized_corpus import TokenizedCorpus

__all__ = [
    "CorpusError",
    "CorpusConfigError",
    "discover_corpus_files",
    "iter_lines",
    "iter_records",
    "join_record_lines",
    "extract_text",
    "is_jsonl",
    "source_name",
    "group_files_by_source",
    "TokenizedCorpus",
    "BlendedCorpus",
    "build_blended_corpus",
]
