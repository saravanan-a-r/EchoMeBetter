"""
Decoding: plain greedy must agree with the model package's own greedy, and
n-gram blocking must actually block.
"""

from __future__ import annotations

import torch

from conftest import build_tiny_model
from src.decoding import greedy_decode
from src.interop import rephrase_model


def _inputs(batch: int = 4, length: int = 10, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(10, 80, (batch, length), generator=generator)
    mask = torch.ones_like(ids)
    mask[0, -3:] = 0
    ids[0, -3:] = 0
    return ids, mask


def test_unblocked_greedy_matches_the_model_packages_greedy():
    """Same argmax path, so the adapter's evaluation is not a different decoder."""
    model = build_tiny_model()
    ids, mask = _inputs()
    ours = greedy_decode(model, ids, mask, max_new_tokens=30, no_repeat_ngram_size=0)
    reference = rephrase_model.greedy_generate(model, ids, mask, max_new_tokens=30).tolist()
    for row, (sequence, done) in enumerate(zip(ours.sequences, ours.terminated)):
        expected = reference[row]
        if 1 in expected:
            assert done and sequence == expected[: expected.index(1)]
        else:
            assert not done and sequence == expected


def test_ngram_blocking_never_repeats_an_ngram():
    """A random tiny model loops readily -- exactly what blocking must stop."""
    model = build_tiny_model(seed=3)
    ids, mask = _inputs(batch=6, seed=5)
    for n in (2, 3):
        result = greedy_decode(model, ids, mask, max_new_tokens=60, no_repeat_ngram_size=n)
        for sequence in result.sequences:
            grams = [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
            assert len(grams) == len(set(grams))


def test_rows_that_hit_the_budget_are_reported_as_unterminated():
    model = build_tiny_model()
    ids, mask = _inputs()
    result = greedy_decode(model, ids, mask, max_new_tokens=1, no_repeat_ngram_size=0)
    for sequence, done in zip(result.sequences, result.terminated):
        assert done == (len(sequence) == 0)
