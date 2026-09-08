"""
Shared fixtures for the rewrite-task test suite.

Run pytest **from this directory** -- `model/`, `UL2/`, `training/`, `corpus/`
and `rewrite/` each contain a package called `src`, so they are independent
test roots.

The fake tokenizer here is one ID per character, so a unit test can read a
token sequence back as text and check *what* was framed rather than only how
long it is. The fake IDs for the frame are deliberately far from those the
real vocabulary uses, so nothing in the package can pass by assuming them.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokens import RewriteTokens  # noqa: E402

REAL_TOKEN_MAP = ROOT.parent / "tokenizer" / "training" / "output" / "token_map.json"

# Characters map to `ord + OFFSET`, well above every frame ID below.
OFFSET = 1000


def fake_encode(text: str) -> list[int]:
    return [ord(ch) + OFFSET for ch in text]


def fake_decode(ids) -> str:
    return "".join(chr(value - OFFSET) for value in ids)


@pytest.fixture
def tokens() -> RewriteTokens:
    return RewriteTokens(style_id=262, open_id=274, close_id=275, eos_id=1, decoder_start_id=0)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(20260908)


class ListRecords:
    """
    A resumable `next_record()` stream over a fixed list, wrapping around.

    The same shape `training/conftest.py`'s `ListIterator` gives the trainer:
    enough to prove resumption without touching the filesystem.
    """

    def __init__(self, records: list[str]) -> None:
        if not records:
            raise ValueError("ListRecords needs at least one record")
        self.records = list(records)
        self.position = 0

    def next_record(self) -> str:
        record = self.records[self.position % len(self.records)]
        self.position += 1
        return record

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.position = int(state["position"])
