"""
Span sampling -- deciding *which* token positions get masked.

This is the numerical core of [R] and [X] denoising, kept separate from the
code that builds encoder/decoder sequences so its properties can be tested
in isolation, without any token IDs involved.

The algorithm (T5's `random_spans_noise_mask`)
----------------------------------------------
A naive "flip a coin per token" mask produces spans of geometrically
distributed length with no control over how many there are -- and, worse,
adjacent masked spans. Two touching spans are indistinguishable from one
longer span once they are replaced by sentinels, so the realized span-length
distribution silently drifts away from the configured mean.

T5's construction avoids both problems by deciding the counts first, then
the layout:

    1. num_corrupted = round(n * corruption_rate)
    2. num_spans     = round(num_corrupted / mean_span_length)
    3. split num_corrupted tokens into num_spans positive-length segments
    4. split the (n - num_corrupted) kept tokens into num_spans segments too
    5. interleave, kept-first: keep, mask, keep, mask, ...

Because every segment in step 3 and 4 has length >= 1, each masked span is
preceded by at least one kept token. Spans therefore can never touch, the
count is exactly what was asked for, and the mean length is exact by
construction rather than by expectation.

Determinism
-----------
Every function here takes an explicit `random.Random`. Nothing reads global
RNG state. This is what makes frozen validation masks (architecture.md 7.6)
and a resumable data pipeline (7.9) possible: given the same seed and the
same input length, the same spans come out, on any machine, in any order.
"""

from __future__ import annotations

import random

from .errors import ConfigError, SentinelBudgetExceededError

# A span-corruption example needs at least one masked and one kept token,
# and the kept/mask interleave needs room for both.
MIN_CORRUPTIBLE_LENGTH = 2


def plan_span_counts(
    length: int,
    corruption_rate: float,
    mean_span_length: float,
) -> tuple[int, int]:
    """
    Decide how many tokens to corrupt and how many spans to split them into.

    Single source of truth for this arithmetic: `sample_spans` uses it to
    build real examples and `UL2Config.validate_capacity` uses it to compute
    worst cases *before* a run starts. Duplicating the formula would let the
    up-front safety check drift away from what the pipeline actually does --
    which is precisely the class of bug that produces a sentinel overflow
    three hours into training.

    Returns `(num_corrupted_tokens, num_spans)`.

    The clamps are what keep short sequences well-formed:
      - at least 1 corrupted token, and at least 1 kept token, so neither
        the mask nor the context is empty;
      - `num_spans` never exceeds the corrupted tokens (each span needs >= 1)
        nor the kept tokens (each span needs >= 1 kept token before it).
    """
    if length < MIN_CORRUPTIBLE_LENGTH:
        raise ConfigError(
            f"cannot corrupt a sequence of length {length}; "
            f"need at least {MIN_CORRUPTIBLE_LENGTH} tokens"
        )

    num_corrupted = int(round(length * corruption_rate))
    num_corrupted = max(1, min(num_corrupted, length - 1))
    num_kept = length - num_corrupted

    num_spans = int(round(num_corrupted / mean_span_length))
    num_spans = max(1, min(num_spans, num_corrupted, num_kept))

    return num_corrupted, num_spans


def random_segmentation(num_items: int, num_segments: int, rng: random.Random) -> list[int]:
    """
    Split `num_items` items into `num_segments` segments, each of length >= 1.

    Implemented by choosing `num_segments - 1` distinct cut points from the
    interior positions, which makes every valid composition equally likely
    and guarantees no zero-length segment -- the property the interleave in
    `sample_spans` depends on for non-adjacency.
    """
    if num_segments < 1:
        raise ConfigError(f"num_segments must be >= 1, got {num_segments}")
    if num_items < num_segments:
        raise ConfigError(
            f"cannot split {num_items} items into {num_segments} segments of length >= 1"
        )

    if num_segments == 1:
        return [num_items]

    cuts = sorted(rng.sample(range(1, num_items), num_segments - 1))
    lengths: list[int] = []
    previous = 0
    for cut in cuts:
        lengths.append(cut - previous)
        previous = cut
    lengths.append(num_items - previous)
    return lengths


def sample_spans(
    length: int,
    corruption_rate: float,
    mean_span_length: float,
    rng: random.Random,
    *,
    max_spans: int | None = None,
) -> list[tuple[int, int]]:
    """
    Sample non-overlapping, non-adjacent masked spans over `[0, length)`.

    Returns half-open `(start, end)` pairs in ascending order.

    `max_spans` is the sentinel budget. Exceeding it raises rather than
    truncating: architecture.md 7.4 requires that a span is never silently
    dropped and a sentinel is never reused within one example.
    """
    num_corrupted, num_spans = plan_span_counts(length, corruption_rate, mean_span_length)

    if max_spans is not None and num_spans > max_spans:
        raise SentinelBudgetExceededError(
            f"a {length}-token sequence at corruption_rate={corruption_rate} / "
            f"mean_span_length={mean_span_length} needs {num_spans} spans, but only "
            f"{max_spans} sentinels exist. Never wrap or drop -- fix the budget or "
            f"the corruption settings (architecture.md 6.3, 7.4)."
        )

    corrupted_lengths = random_segmentation(num_corrupted, num_spans, rng)
    kept_lengths = random_segmentation(length - num_corrupted, num_spans, rng)

    spans: list[tuple[int, int]] = []
    position = 0
    for kept, corrupted in zip(kept_lengths, corrupted_lengths):
        position += kept
        spans.append((position, position + corrupted))
        position += corrupted

    return spans


def sample_prefix_split(
    length: int,
    min_prefix_fraction: float,
    max_prefix_fraction: float,
    rng: random.Random,
) -> int:
    """
    Choose the prefix/continuation boundary for [S] denoising.

    Returns the number of tokens kept as the prefix. Always leaves at least
    one token on each side: an empty prefix gives the encoder nothing to
    condition on, and an empty continuation gives the decoder nothing to
    learn from.
    """
    if length < MIN_CORRUPTIBLE_LENGTH:
        raise ConfigError(
            f"cannot split a sequence of length {length} into prefix and continuation"
        )

    fraction = rng.uniform(min_prefix_fraction, max_prefix_fraction)
    prefix_length = int(round(length * fraction))
    return max(1, min(prefix_length, length - 1))
