"""
Shared fixtures for the UL2 test suite.

The fake token table here is deliberately *not* the real tokenizer's. Real
IDs would let a bug that assumes a particular ordering pass unnoticed;
arbitrary, non-contiguous, deliberately-out-of-order IDs make any such
assumption fail immediately. architecture.md 6.4 forbids hardcoded IDs
downstream, and this is how that rule is enforced in tests.

One integration test does load the real `token_map.json` when it is present,
so the name-resolution path is exercised against the actual artifact too.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import UL2Config  # noqa: E402
from src.special_tokens import SpecialTokens  # noqa: E402

# Where the trained tokenizer's artifacts live, if they have been built.
REAL_TOKEN_MAP = ROOT.parent / "tokenizer" / "training" / "output" / "token_map.json"

# Sentinel IDs are non-contiguous and unordered on purpose: nothing in the
# objective may assume `sentinel(i) == base + i`. 997 and 1999 are coprime, so
# the values are distinct but land in scrambled order.
FAKE_SENTINEL_IDS = tuple(9000 + (i * 997) % 1999 for i in range(256))


@pytest.fixture
def specials() -> SpecialTokens:
    """A 256-sentinel token table matching the real vocabulary's shape."""
    return SpecialTokens(
        sentinel_ids=FAKE_SENTINEL_IDS,
        mode_token_ids={"R": 701, "X": 702, "S": 703},
        eos_id=1,
        pad_id=0,
        decoder_start_id=0,
    )


@pytest.fixture
def small_specials() -> SpecialTokens:
    """A deliberately tiny sentinel budget, for overflow tests."""
    return SpecialTokens(
        sentinel_ids=(50, 51, 52, 53),
        mode_token_ids={"R": 701, "X": 702, "S": 703},
        eos_id=1,
        pad_id=0,
        decoder_start_id=0,
    )


@pytest.fixture
def config() -> UL2Config:
    """The architecture.md defaults."""
    return UL2Config()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(20260818)


def make_tokens(length: int, start: int = 1000) -> list[int]:
    """
    Distinct, non-special token IDs.

    Every token is unique, so a reconstruction test detects a duplicated or
    dropped token rather than accidentally matching a repeat.
    """
    return list(range(start, start + length))
