"""
Unit tests for execution.py -- the parallelism boundary.

Two properties matter here, and everything else in this file supports them:

  1. `Executor.map` returns results in INPUT order, at any worker count. Every
     deterministic guarantee further up the stack (a byte-identical
     corpus.txt, a replayable eval split) rests on this.
  2. `largest_remainder` splits a budget exactly, never over-allocating a
     bucket beyond its own weight. That is what lets a single global sampling
     budget become N independent per-file quotas that still sum to the
     original number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[2]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from execution import (  # noqa: E402
    AUTO_RESERVED_CORES,
    Executor,
    largest_remainder,
    resolve_workers,
)


# Module-level so the process pool can pickle them.
def _square(x: int) -> int:
    return x * x


def _identity(x):
    return x


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------


def test_sequential_executor_runs_without_a_pool():
    assert Executor(workers=1).map(_square, [1, 2, 3, 4]) == [1, 4, 9, 16]


def test_empty_input_is_not_an_error():
    assert Executor(workers=4).map(_square, []) == []


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_results_are_in_input_order_at_every_worker_count(workers):
    """
    The load-bearing property. Workers finish out of order -- a 2 GB shard
    takes far longer than a 4 MB one -- so if `map` returned completion order
    the sampled corpus would depend on scheduling, and nothing downstream
    could be reproducible.
    """
    items = list(range(50))
    assert Executor(workers=workers).map(_square, items) == [i * i for i in items]


def test_single_item_skips_the_pool_entirely():
    """Forking to process one file costs more than it saves."""
    executor = Executor(workers=16)
    assert executor.map(_square, [7]) == [49]


def test_is_parallel_reflects_worker_count():
    assert not Executor(workers=1).is_parallel
    assert Executor(workers=2).is_parallel


def test_mixed_payload_types_survive_the_round_trip():
    payload = [{"a": 1}, [1, 2], "text", 3.5, None]
    assert Executor(workers=2).map(_identity, payload) == payload


# --------------------------------------------------------------------------
# resolve_workers
# --------------------------------------------------------------------------


def test_explicit_worker_count_is_honoured():
    assert resolve_workers(12) == 12


def test_auto_leaves_headroom_and_never_returns_zero(monkeypatch):
    monkeypatch.setattr("execution.os.cpu_count", lambda: 42)
    assert resolve_workers(0) == 42 - AUTO_RESERVED_CORES

    # A tiny machine must still get a usable worker rather than 0 or negative.
    monkeypatch.setattr("execution.os.cpu_count", lambda: 2)
    assert resolve_workers(0) == 1


# --------------------------------------------------------------------------
# largest_remainder
# --------------------------------------------------------------------------


def test_split_is_exact_and_proportional():
    # Weights are per-bucket capacity (lines a file holds), so they must be
    # large enough to absorb the budget being split across them.
    assert largest_remainder(10, [10, 10, 10, 10, 10]) == [2, 2, 2, 2, 2]
    assert sum(largest_remainder(100, [300, 500, 1200])) == 100


def test_rounding_never_loses_or_invents_a_line():
    """Three-way split of 10 cannot be done evenly in whole numbers; the
    result must still sum to exactly 10."""
    result = largest_remainder(10, [10, 10, 10])
    assert sum(result) == 10
    assert sorted(result) == [3, 3, 4]


def test_no_bucket_exceeds_its_own_weight():
    """A file cannot be asked to yield more lines than it holds."""
    weights = [1, 2, 500]
    result = largest_remainder(400, weights)
    assert all(got <= want for got, want in zip(result, weights))
    assert sum(result) == 400


def test_total_is_capped_at_available_capacity():
    assert sum(largest_remainder(1000, [2, 3])) == 5


def test_zero_and_empty_cases():
    assert largest_remainder(0, [5, 5]) == [0, 0]
    assert largest_remainder(10, []) == []
    assert largest_remainder(10, [0, 0]) == [0, 0]


def test_zero_weight_buckets_get_nothing():
    assert largest_remainder(6, [0, 3, 0, 3]) == [0, 3, 0, 3]


def test_is_deterministic_and_breaks_ties_on_index():
    """Ties must not depend on dict or filesystem ordering."""
    first = largest_remainder(7, [1, 1, 1, 1, 1, 1, 1, 1])
    for _ in range(5):
        assert largest_remainder(7, [1, 1, 1, 1, 1, 1, 1, 1]) == first
    # Earliest indices win the leftover.
    assert first == [1, 1, 1, 1, 1, 1, 1, 0]
