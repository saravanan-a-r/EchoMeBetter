"""
Deterministic synthetic-corpus generator for tests.

The point is not realism for its own sake -- it is that the generated text
contains, in known quantities, exactly the constructs the tokenizer promises
to preserve byte-for-byte: URLs, markdown links, indented code, numbers,
multi-byte UTF-8 and whitespace runs. A test corpus of bland prose would
pass Gate 1 while proving nothing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

_CORE_WORDS = [
    "rephrase", "sentence", "tokenizer", "vocabulary", "encoder", "decoder", "corpus",
    "training", "inference", "fidelity", "preserve", "structure", "grammar", "concise",
    "professional", "friendly", "elaborate", "document", "paragraph", "language", "model",
    "python", "function", "variable", "argument", "returns", "exception", "database",
    "network", "request", "response", "server", "client", "protocol", "message", "buffer",
    "consider", "however", "although", "therefore", "because", "specific", "general",
    "measure", "estimate", "compare", "improve", "reduce", "increase", "maintain",
    "quality", "accuracy", "latency", "throughput", "memory", "storage", "compute",
]

_PREFIXES = ["", "re", "un", "pre", "post", "sub", "over", "under", "inter", "multi", "non", "de"]
_SUFFIXES = ["", "s", "ing", "ed", "er", "ion", "able", "ness", "ly", "ment", "ive", "al"]


def _build_word_list() -> list[str]:
    """
    A few thousand distinct but morphologically realistic words.

    Diversity matters: SentencePiece can only learn as many pieces as the
    corpus supports, and a 50-word corpus cannot exercise a realistic
    vocabulary size. Affixation also produces the shared-substring structure
    Unigram actually exploits, so the test corpus stresses the real algorithm
    rather than a degenerate one.
    """
    words = {
        f"{prefix}{stem}{suffix}"
        for stem in _CORE_WORDS
        for prefix in _PREFIXES
        for suffix in _SUFFIXES
    }
    return sorted(words)


WORDS = _build_word_list()

URLS = [
    "https://stackoverflow.com/questions/12345/how-to-parse-json",
    "http://example.com/path?a=1&b=2#frag",
    "https://docs.python.org/3/library/json.html",
    "https://github.com/google/sentencepiece",
    "https://en.wikipedia.org/wiki/Byte_pair_encoding",
]

CODE_SNIPPETS = [
    'def parse(x):\n    return json.loads(x)',
    'if (a == b) { return a * 2; }',
    "    for i in range(10):\n        total += i",
    'SELECT id, name FROM users WHERE age >= 18;',
    'const re = /^[a-z0-9_-]{3,16}$/;',
]

UNICODE_BITS = ["café", "naïve", "résumé", "→", "±", "“quoted”", "—", "½", "µs", "日本語"]


def _make_record(rng: random.Random) -> str:
    kind = rng.random()
    words = [rng.choice(WORDS) for _ in range(rng.randint(6, 30))]
    base = " ".join(words)

    if kind < 0.20:
        return f"{base} see {rng.choice(URLS)} for details."
    if kind < 0.35:
        label = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 4)))
        return f"{base} [{label}]({rng.choice(URLS)}) explains it."
    if kind < 0.50:
        # Code keeps its indentation and newlines: corpus_reader splits it
        # into separate training lines exactly as SentencePiece would.
        return f"{base}\n{rng.choice(CODE_SNIPPETS)}"
    if kind < 0.62:
        nums = " ".join(str(rng.randint(0, 10**rng.randint(1, 9))) for _ in range(rng.randint(2, 6)))
        return f"{base} values: {nums} (delta {rng.randint(-99, 99)}.{rng.randint(0, 999)})"
    if kind < 0.74:
        return f"{base} {rng.choice(UNICODE_BITS)} {rng.choice(UNICODE_BITS)} {base[:40]}"
    if kind < 0.82:
        return f"{base}     {rng.choice(WORDS)}\t{rng.choice(WORDS)}  trailing spaces   "
    return base


def write_synthetic_corpus(
    directory: Path, *, n_records: int = 6000, seed: int = 7, n_files: int = 3
) -> list[Path]:
    """
    Write `n_records` JSONL records spread over `n_files`, named like the
    real downloader outputs so source-label parsing is exercised too.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    names = ["fineweb_edu_10mb.jsonl", "stackexchange_5mb.jsonl", "gutenberg_pg19_8mb.jsonl"]
    paths = [directory / names[i % len(names)] for i in range(n_files)]

    handles = [open(p, "w", encoding="utf-8") for p in paths]
    try:
        for i in range(n_records):
            record = {"text": _make_record(rng)}
            handles[i % len(handles)].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for fh in handles:
            fh.close()
    return paths
