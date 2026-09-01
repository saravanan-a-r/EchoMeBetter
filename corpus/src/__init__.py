"""
Pretraining corpus pipeline (architecture.md §7, §7.9).

Scope
-----
Turns files on disk into the resumable stream of token-ID lines
`UL2Objective.try_corrupt` consumes. Does not corrupt, batch, or train —
those belong to `UL2/` and `training/`. Does not select or blend corpus
*sources* by share — that is `tokenizer/training/corpus_reader.py`'s job for
fitting the tokenizer, a different, whole-corpus-in-view problem.

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

from .errors import CorpusConfigError, CorpusError
from .reader import discover_corpus_files, extract_text, is_jsonl, iter_lines
from .tokenized_corpus import TokenizedCorpus

__all__ = [
    "CorpusError",
    "CorpusConfigError",
    "discover_corpus_files",
    "iter_lines",
    "extract_text",
    "is_jsonl",
    "TokenizedCorpus",
]
