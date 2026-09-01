"""
Turning corpus files on disk into a stream of plain text lines.

Scope
-----
This module is deliberately dumb: discover files, and turn each raw file
line into zero or more text segments. It has no opinion about sampling,
blending source shares, or train/eval splitting — that machinery already
exists in `tokenizer/training/corpus_reader.py` for *fitting the tokenizer*,
a one-time, whole-corpus-in-view operation. Pretraining is a different
problem: a streamed, resumable, effectively-infinite pass (§7.5 says multiple
epochs over a deduplicated corpus are fine), which is what
`tokenized_corpus.py` builds on top of this.

Fidelity
--------
Same rule as `tokenizer/training/corpus_reader.py` (architecture.md NF1): this
module never rewrites text. It does not normalize, strip, or truncate. Its
only transformation is deciding which raw lines carry text and splitting
multi-line records on `"\\n"`, because that split has to happen somewhere
before tokenization and doing it once, here, keeps every downstream consumer
from re-deriving it.

UTF-8 decoding is strict (`errors="strict"`, the default): a byte sequence
that is not valid UTF-8 is a corpus bug, and silently substituting U+FFFD
would corrupt training text in exactly the way the tokenizer's own corpus
reader refuses to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .errors import CorpusConfigError

CORPUS_SUFFIXES = (".jsonl", ".json", ".txt")


def discover_corpus_files(path: str | Path) -> list[Path]:
    """
    Resolve a file or directory to a sorted list of corpus files.

    Sorted so that two runs over the same directory see files in the same
    order — the shuffled-file-order determinism `TokenizedCorpus` promises
    rests on this being stable, not on filesystem iteration order.
    """
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES
        )
    else:
        raise CorpusConfigError(f"corpus path does not exist: {path}")

    if not files:
        raise CorpusConfigError(
            f"no corpus files found under {path} (looking for {', '.join(CORPUS_SUFFIXES)})"
        )
    return files


def is_jsonl(path: Path) -> bool:
    return path.suffix.lower() in (".jsonl", ".json")


def extract_text(raw_line: str, jsonl: bool) -> str | None:
    """
    One raw file line -> the text it carries, or `None` if unusable.

    A malformed line is skipped, not fatal: a streamed multi-week run must
    tolerate the occasional bad record the way `UL2Objective.try_corrupt`
    tolerates a too-short or too-long one. Callers are expected to count
    skips so the loss stays visible rather than silent.
    """
    if not jsonl:
        return raw_line if raw_line.strip() else None

    try:
        row = json.loads(raw_line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(row, dict):
        return None
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def iter_lines(files: list[Path]) -> Iterator[str]:
    """
    Stream every non-empty text segment from `files`, in file order.

    A stateless, non-resumable convenience for tests and one-off inspection.
    `TokenizedCorpus` does not use this directly — it needs to track its
    position line-by-line to be resumable — but reuses `extract_text` so
    the two paths cannot disagree about what counts as a usable line.
    """
    for path in files:
        jsonl = is_jsonl(path)
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                text = extract_text(raw_line.rstrip("\n").rstrip("\r"), jsonl)
                if text is None:
                    continue
                for segment in text.split("\n"):
                    segment = segment.strip()
                    if segment:
                        yield segment
