"""
Shared fixtures for the corpus test suite.

Run pytest **from this directory** — `model/`, `UL2/`, `training/` and
`corpus/` each contain a package called `src`, so they are independent test
roots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer_interop import DEFAULT_TOKENIZER_DIR  # noqa: E402

REAL_TOKENIZER_MODEL = DEFAULT_TOKENIZER_DIR / "spm.model"


class FakeProcessor:
    """
    A stand-in SentencePieceProcessor: `encode(text) -> list[int]`.

    Deterministic and dependency-free, so unit tests exercise
    `TokenizedCorpus`'s file/line bookkeeping without needing the real
    tokenizer artifacts to be built. One int per character (offset so 0 is
    never emitted, keeping every token "real" rather than accidentally
    colliding with a pad id in a downstream assertion).
    """

    def encode(self, text: str, out_type=int) -> list[int]:
        return [ord(ch) % 250 + 5 for ch in text]


@pytest.fixture
def fake_processor() -> FakeProcessor:
    return FakeProcessor()


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    return d


def write_txt(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    import json

    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path
