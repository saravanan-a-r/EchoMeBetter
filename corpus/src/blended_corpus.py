"""
`BlendedCorpus` — one resumable stream drawn from several sources in
configured proportions (architecture_improvements.md item 2).

Why this exists
---------------
`TokenizedCorpus` reads a shuffled file list end to end, so a source's share of
the stream is whatever share of the corpus it happens to occupy on disk. That
is the right default for the 90% of pretraining that wants the corpus as it
is. It is the wrong tool for the WSD cooldown, whose entire point is to spend
the final steps on a *deliberately different* mixture — 50% high-quality prose,
30% CLI text, 19% permissive code — none of which resembles the corpus's own
proportions.

So this is not a replacement for `TokenizedCorpus`; it is a mixer built out of
several of them, one per source, and it delegates every question about reading,
tokenizing and resuming a file to the ones underneath. The only thing it adds
is *which* source the next document comes from.

Sampling, not interleaving
--------------------------
A source is drawn per document, from a weighted RNG, rather than by cycling
sources in a fixed pattern. A fixed pattern would put a period into the data
stream at exactly the frequency of the cycle, which is the structure shuffling
exists to destroy — the same argument `UL2/src/length_batching.py` makes for
re-randomizing batch order after sorting a pool.

Weights are relative and normalized over the sources actually present. A
cooldown blend naming a source the corpus does not have is refused rather than
renormalized around: silently dropping `cli_tldr` from a CLI-focused cooldown
would leave the run doing something other than what its configuration says,
and the loss curve would not show it.

Exhaustion
----------
Each `TokenizedCorpus` is infinite — it wraps around to its first file and
increments `epoch`. That matters here more than it does for a single stream: a
cooldown deliberately upweights small sources, so the small ones *will* wrap,
several times. `source_epochs()` reports how often, because "how many times did
the cooldown show the model the same 67,000 tokens" is a number worth being
able to read off a finished run rather than infer.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .errors import CorpusConfigError
from .tokenized_corpus import TokenizedCorpus


class BlendedCorpus:
    """
    A weighted mixture of per-source `TokenizedCorpus` streams.

    Satisfies the same two contracts `TokenizedCorpus` does — an infinite
    iterator of `list[int]`, and `state_dict`/`load_state_dict` — so anything
    that consumes one can consume the other without knowing which it has.
    """

    def __init__(
        self,
        sources: Mapping[str, TokenizedCorpus],
        weights: Mapping[str, float],
        *,
        seed: int = 0,
    ) -> None:
        if not sources:
            raise CorpusConfigError("BlendedCorpus needs at least one source")

        missing = sorted(set(weights) - set(sources))
        if missing:
            raise CorpusConfigError(
                f"blend names source(s) not present in the corpus: {missing}. "
                f"Available: {sorted(sources)}. Renormalizing around a missing "
                f"source would leave the run training on a mixture its "
                f"configuration does not describe."
            )
        unweighted = sorted(set(sources) - set(weights))
        if unweighted:
            raise CorpusConfigError(
                f"source(s) present in the corpus but absent from the blend: "
                f"{unweighted}. Give each an explicit weight, or hand this class "
                f"only the sources the blend names — an implicit zero is the one "
                f"reading that cannot be told apart from a typo."
            )
        for source, weight in weights.items():
            if float(weight) <= 0.0:
                raise CorpusConfigError(
                    f"blend weight for {source!r} must be positive, got {weight!r}"
                )

        self.sources: dict[str, TokenizedCorpus] = dict(sources)
        self._names: list[str] = sorted(self.sources)
        total = float(sum(float(weights[name]) for name in self._names))
        self.weights: dict[str, float] = {
            name: float(weights[name]) / total for name in self._names
        }
        self._cumulative: list[float] = []
        running = 0.0
        for name in self._names:
            running += self.weights[name]
            self._cumulative.append(running)

        self.seed = seed
        self.rng = random.Random(seed)

        # Visible for the same reason `TokenizedCorpus`'s counters are: a
        # mixture that is not delivering its configured shares should be
        # discoverable from the run, not only from reading this file.
        self.documents_drawn: dict[str, int] = {name: 0 for name in self._names}

    # -- iteration ----------------------------------------------------------

    def __iter__(self) -> "BlendedCorpus":
        return self

    def __next__(self) -> list[int]:
        name = self._draw()
        self.documents_drawn[name] += 1
        return next(self.sources[name])

    def _draw(self) -> str:
        """
        One source, drawn in proportion to its weight.

        `random()` against a cumulative table rather than `random.choices`:
        the latter is free to change how many values it pulls from the RNG
        between Python versions, and this stream's draws have to reproduce
        exactly on resume.
        """
        point = self.rng.random()
        for name, edge in zip(self._names, self._cumulative):
            if point < edge:
                return name
        return self._names[-1]  # floating-point tail; the last bucket owns it

    # -- telemetry ----------------------------------------------------------

    def realized_shares(self) -> dict[str, float]:
        """Share of documents actually drawn per source, for comparison with `weights`."""
        total = sum(self.documents_drawn.values())
        if not total:
            return {name: 0.0 for name in self._names}
        return {name: self.documents_drawn[name] / total for name in self._names}

    def source_epochs(self) -> dict[str, int]:
        """
        How many times each source has been read end to end.

        A cooldown that upweights a 67,000-token source is going to repeat it;
        the question is whether it repeated it five times or five thousand, and
        this is where that answer lives.
        """
        return {name: self.sources[name].epoch for name in self._names}

    # -- Resumable protocol (training/src/checkpoint.py) ---------------------

    def state_dict(self) -> dict[str, Any]:
        version, internal_state, gauss_next = self.rng.getstate()
        return {
            "sources": {
                name: corpus.state_dict() for name, corpus in self.sources.items()
            },
            "rng": [version, list(internal_state), gauss_next],
            "documents_drawn": dict(self.documents_drawn),
            "weights": dict(self.weights),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved = state["sources"]
        if sorted(saved) != self._names:
            raise CorpusConfigError(
                f"resumed blend names sources {sorted(saved)} but this run has "
                f"{self._names}; the corpus or the blend changed since the "
                f"checkpoint was written"
            )
        for name, child_state in saved.items():
            self.sources[name].load_state_dict(child_state)
        version, internal_state, gauss_next = state["rng"]
        self.rng.setstate((version, tuple(internal_state), gauss_next))
        self.documents_drawn = {
            name: int(state.get("documents_drawn", {}).get(name, 0))
            for name in self._names
        }


def build_blended_corpus(
    grouped_files: Mapping[str, Sequence[Any]],
    sp: Any,
    weights: Mapping[str, float],
    *,
    seed: int = 0,
) -> BlendedCorpus:
    """
    A `BlendedCorpus` over exactly the sources `weights` names.

    Sources present on disk but absent from `weights` are left out here rather
    than passed on to be rejected: a cooldown blend is a *subset* of the corpus
    by design — that is what upweighting means — so "the corpus has more
    sources than the blend" is the normal case, while "the blend names a source
    the corpus does not have" is a configuration error and is refused inside
    `BlendedCorpus`.

    Each source's stream gets its own seed, derived from `seed` and the source
    name, so two sources do not shuffle their file lists identically and
    re-running with one source added does not reshuffle the others.
    """
    missing = sorted(set(weights) - set(grouped_files))
    if missing:
        raise CorpusConfigError(
            f"blend names source(s) not found under the corpus root: {missing}. "
            f"Available: {sorted(grouped_files)}"
        )
    sources = {
        name: TokenizedCorpus(
            grouped_files[name], sp, seed=seed + _source_seed_offset(name)
        )
        for name in weights
    }
    return BlendedCorpus(sources, weights, seed=seed)


def _source_seed_offset(name: str) -> int:
    """
    A stable per-source seed offset.

    `hash()` is salted per process and would make a resumed run shuffle its
    files differently from the run it continues — a silent, seed-shaped bug of
    exactly the kind `TokenizedCorpus`'s saved `order` exists to prevent. This
    is a plain deterministic sum instead, bounded well away from overflowing
    anything.
    """
    return sum(ord(character) for character in name) * 31


__all__ = ["BlendedCorpus", "build_blended_corpus"]
