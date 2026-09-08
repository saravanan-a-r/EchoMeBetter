"""
`RewriteExampleSource` -- documents in, framed (encoder, target) ID pairs out.

The one class the pretraining pipeline needs from this package. It reads text
records from a resumable stream, corrupts each one, tokenizes both sides and
wraps the corrupted side in the frame the product uses:

    encoder:  <style:grammar> <text_to_rewrite> corrupted... </text_to_rewrite> </s>
    target:   original... </s>

It deliberately does **not** go through `UL2Objective`: the pair is already
complete, and running it through span corruption or prefix denoising would
mask the text a second time and ask the model to fill in blanks inside a
document it is supposed to be rewriting whole.

Admission
---------
Same policy as `UL2Objective._admit`, for the same reason (architecture.md
4.5, no silent truncation): a document whose original does not fit the
decoder budget, or whose corrupted form does not fit the encoder budget, is
skipped and counted, never cut. Corrupted text is usually a few tokens longer
than the original -- typos and dropped spaces tokenize worse -- so a document
can pass the first check and fail the second; both are counted separately.

Resumability
------------
Holds its own RNG for the corruption draws and its own record stream, and
saves both (`state_dict`), so a run interrupted inside the cooldown resumes
with the same corruptions it would have produced uninterrupted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .corruption import IDENTITY, Corruptor
from .errors import RewriteConfigError
from .tokens import FRAME_TOKENS, RewriteTokens

# The mode label carried on every example this source produces. Deliberately
# not a single letter: it is not a UL2 denoiser, and it must never collide
# with `UL2/src/special_tokens.py`'s `MODES`.
REWRITE_MODE = "rewrite"


@dataclass(frozen=True)
class RewritePair:
    """One framed example, as plain integer IDs."""

    encoder_input_ids: tuple[int, ...]
    decoder_target_ids: tuple[int, ...]
    source_length: int  # tokens in the original document
    edits: int  # pieces the corruption changed; 0 for an identity pair
    profile: str


class RewriteExampleSource:
    """
    An infinite stream of `RewritePair`s from a stream of text records.

    `records` must offer `next_record() -> str` and the `state_dict` /
    `load_state_dict` pair -- `corpus/`'s `TokenizedCorpus` and
    `BlendedCorpus` both do. `encode` turns text into token IDs; the caller
    supplies it so this package never imports the tokenizer.
    """

    def __init__(
        self,
        records: Any,
        corruptor: Corruptor,
        encode: Callable[[str], Sequence[int]],
        tokens: RewriteTokens,
        *,
        max_source_length: int,
        max_target_length: int,
        min_source_length: int,
        seed: int,
    ) -> None:
        if min_source_length < 1:
            raise RewriteConfigError(f"min_source_length must be >= 1, got {min_source_length}")
        if max_source_length < min_source_length + FRAME_TOKENS:
            raise RewriteConfigError(
                f"max_source_length={max_source_length} cannot hold the {FRAME_TOKENS}-token "
                f"frame around a {min_source_length}-token document"
            )
        if max_target_length < min_source_length + 1:
            raise RewriteConfigError(
                f"max_target_length={max_target_length} cannot hold a {min_source_length}-token "
                f"document plus EOS"
            )
        self.records = records
        self.corruptor = corruptor
        self.encode = encode
        self.tokens = tokens
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.min_source_length = min_source_length
        self.seed = seed
        self.rng = random.Random(seed)

        # Plain counters, exact over the run, saved with the state so a
        # resumed run's report still describes the whole run.
        self.documents_read = 0
        self.skipped_too_short = 0
        self.skipped_too_long = 0
        self.skipped_not_prose = 0
        self.pairs_emitted = 0
        self.edits_total = 0
        self.source_tokens_total = 0
        self.profiles: dict[str, int] = {}

    # -- iteration ----------------------------------------------------------

    def next_pair(self) -> RewritePair:
        """The next admissible pair. Skips are counted, never silent."""
        while True:
            text = self.records.next_record()
            self.documents_read += 1

            original = list(self.encode(text))
            if len(original) < self.min_source_length:
                self.skipped_too_short += 1
                continue
            if len(original) + 1 > self.max_target_length:
                self.skipped_too_long += 1
                continue

            corruption = self.corruptor.corrupt(text, self.rng)
            if corruption is None:
                self.skipped_not_prose += 1
                continue

            corrupted = original if corruption.profile == IDENTITY else list(self.encode(corruption.text))
            encoder = (
                self.tokens.style_id,
                self.tokens.open_id,
                *corrupted,
                self.tokens.close_id,
                self.tokens.eos_id,
            )
            if len(encoder) > self.max_source_length:
                self.skipped_too_long += 1
                continue

            self.pairs_emitted += 1
            self.edits_total += corruption.edits
            self.source_tokens_total += len(original)
            self.profiles[corruption.profile] = self.profiles.get(corruption.profile, 0) + 1
            return RewritePair(
                encoder_input_ids=encoder,
                decoder_target_ids=(*original, self.tokens.eos_id),
                source_length=len(original),
                edits=corruption.edits,
                profile=corruption.profile,
            )

    # -- telemetry ----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        emitted = self.pairs_emitted
        return {
            "documents_read": self.documents_read,
            "pairs_emitted": emitted,
            "skipped_too_short": self.skipped_too_short,
            "skipped_too_long": self.skipped_too_long,
            "skipped_not_prose": self.skipped_not_prose,
            "mean_source_tokens": _safe_divide(self.source_tokens_total, emitted),
            "edits_per_100_source_tokens": 100.0
            * _safe_divide(self.edits_total, self.source_tokens_total),
            "profiles": {
                name: {"pairs": count, "share": _safe_divide(count, emitted)}
                for name, count in sorted(self.profiles.items())
            },
        }

    def format_table(self) -> str:
        """One-screen summary for a training log."""
        report = self.report()
        lines = [
            f"rewrite task: {report['pairs_emitted']} pairs from {report['documents_read']} "
            f"documents  (skipped: {report['skipped_not_prose']} not prose, "
            f"{report['skipped_too_short']} too short, {report['skipped_too_long']} too long)",
            f"  mean source tokens {report['mean_source_tokens']:.1f}, "
            f"{report['edits_per_100_source_tokens']:.2f} edits per 100 source tokens",
        ]
        for name, entry in report["profiles"].items():
            lines.append(f"  {name:<10}{entry['pairs']:>10}{entry['share']:>9.3f}")
        return "\n".join(lines)

    # -- Resumable protocol (training/src/checkpoint.py) ---------------------

    def state_dict(self) -> dict[str, Any]:
        version, internal_state, gauss_next = self.rng.getstate()
        return {
            "records": self.records.state_dict(),
            "rng": [version, list(internal_state), gauss_next],
            "counters": {
                "documents_read": self.documents_read,
                "skipped_too_short": self.skipped_too_short,
                "skipped_too_long": self.skipped_too_long,
                "skipped_not_prose": self.skipped_not_prose,
                "pairs_emitted": self.pairs_emitted,
                "edits_total": self.edits_total,
                "source_tokens_total": self.source_tokens_total,
                "profiles": dict(self.profiles),
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.records.load_state_dict(state["records"])
        version, internal_state, gauss_next = state["rng"]
        self.rng.setstate((version, tuple(internal_state), gauss_next))
        counters = state.get("counters", {})
        self.documents_read = int(counters.get("documents_read", 0))
        self.skipped_too_short = int(counters.get("skipped_too_short", 0))
        self.skipped_too_long = int(counters.get("skipped_too_long", 0))
        self.skipped_not_prose = int(counters.get("skipped_not_prose", 0))
        self.pairs_emitted = int(counters.get("pairs_emitted", 0))
        self.edits_total = int(counters.get("edits_total", 0))
        self.source_tokens_total = int(counters.get("source_tokens_total", 0))
        self.profiles = {
            str(name): int(count) for name, count in counters.get("profiles", {}).items()
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


__all__ = ["REWRITE_MODE", "RewriteExampleSource", "RewritePair"]
