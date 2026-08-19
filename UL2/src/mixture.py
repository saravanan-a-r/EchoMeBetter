"""
Choosing which denoiser each example gets.

architecture.md 7.3 fixes the mixture at R=40% / X=35% / S=25%, as *sampling
probabilities*. That distinction matters enough that it is worth restating
here, next to the code that implements it: the three modes produce very
different target lengths, so a 40/35/25 split of examples is not a 40/35/25
split of decoder training tokens. This class controls the former. The latter
is measured by `telemetry.py`, never inferred.

Sampling uses an explicit cumulative-weight walk rather than
`random.choices`. Both are correct, but the explicit form consumes exactly
one `rng.random()` call per draw, with a defined ordering -- so a frozen
evaluation set built today reproduces byte-identically on a future Python
whose `random.choices` internals have changed.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from .errors import ConfigError
from .special_tokens import MODES


class ModeSampler:
    """Draws a denoising mode according to fixed probabilities."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise ConfigError("mode weights are empty; there is nothing to sample")

        unknown = set(weights) - set(MODES)
        if unknown:
            raise ConfigError(f"unknown modes in weights: {sorted(unknown)}; expected {MODES}")

        for mode, weight in weights.items():
            if weight < 0:
                raise ConfigError(f"mode weight for {mode!r} is negative: {weight}")

        # Canonical order, so the cumulative boundaries -- and therefore the
        # mode drawn for a given RNG state -- never depend on dict ordering.
        self._modes: tuple[str, ...] = tuple(m for m in MODES if weights.get(m, 0.0) > 0.0)
        if not self._modes:
            raise ConfigError("every mode weight is zero; no mode could ever be sampled")

        total = sum(weights[m] for m in self._modes)
        cumulative: list[float] = []
        running = 0.0
        for mode in self._modes:
            running += weights[mode] / total
            cumulative.append(running)
        # Guard the last boundary against floating-point shortfall, so a draw
        # of 0.9999999999 can never fall off the end of the table.
        cumulative[-1] = 1.0

        self._cumulative: tuple[float, ...] = tuple(cumulative)
        self._weights: Mapping[str, float] = dict(weights)

    @property
    def modes(self) -> tuple[str, ...]:
        """Modes that can actually be drawn, in canonical order."""
        return self._modes

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self._weights)

    def probability(self, mode: str) -> float:
        """Normalized probability of drawing `mode`."""
        total = sum(self._weights[m] for m in self._modes)
        return self._weights.get(mode, 0.0) / total

    def sample(self, rng: random.Random) -> str:
        """Draw one mode. Consumes exactly one random number."""
        draw = rng.random()
        for mode, boundary in zip(self._modes, self._cumulative):
            if draw < boundary:
                return mode
        return self._modes[-1]  # unreachable; float-safety net

    def sample_many(self, count: int, rng: random.Random) -> list[str]:
        return [self.sample(rng) for _ in range(count)]


def deterministic_schedule(weights: Mapping[str, float], count: int) -> Sequence[str]:
    """
    A fixed, non-random mode sequence matching `weights` as closely as
    possible for exactly `count` examples.

    Used to build frozen evaluation sets (architecture.md 7.6), where the mode
    mixture must be *exactly* the intended composition every time rather than
    a sample that happens to land near it. With 300 validation examples and a
    one-third mixture, sampling might give 96/108/96; this gives 100/100/100,
    so per-mode losses are computed over identical example counts in every
    run.

    Largest-remainder apportionment: floor each share, then hand out the
    leftover slots to the largest fractional parts. Ties break by canonical
    mode order, so the result is a pure function of the inputs.
    """
    if count < 0:
        raise ConfigError(f"count must be non-negative, got {count}")
    if count == 0:
        return []

    modes = tuple(m for m in MODES if weights.get(m, 0.0) > 0.0)
    if not modes:
        raise ConfigError("every mode weight is zero; cannot build a schedule")

    total = sum(weights[m] for m in modes)
    exact = {m: count * weights[m] / total for m in modes}
    allocation = {m: int(exact[m]) for m in modes}

    remaining = count - sum(allocation.values())
    if remaining:
        order = sorted(modes, key=lambda m: (-(exact[m] - allocation[m]), MODES.index(m)))
        for mode in order[:remaining]:
            allocation[mode] += 1

    schedule: list[str] = []
    for mode in modes:
        schedule.extend([mode] * allocation[mode])
    return schedule
