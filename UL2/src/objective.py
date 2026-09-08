"""
`UL2Objective` -- the one class a data pipeline needs.

Everything else in this module exists to be testable in isolation; this is
the facade that composes them into the contract the pretraining loop calls:

    objective = UL2Objective(UL2Config(), specials)
    example = objective.corrupt(token_ids, rng)

Responsibilities kept here, deliberately, rather than pushed outward:

  - length admission (too short / too long) and the no-silent-truncation
    policy of architecture.md 4.5
  - mode selection, via `ModeSampler`
  - post-conditions: nothing leaves this class exceeding the encoder or
    decoder budget, so the trainer never has to defend against it
  - telemetry recording, so measurement is automatic rather than something a
    caller can forget

Determinism contract
--------------------
`corrupt(tokens, rng)` is a pure function of `(tokens, rng state, config)`.
The same seed replays the same examples in the same order, which is what
architecture.md 7.6 (frozen validation masks) and 7.9 (resumable data
iterator) both rest on. Nothing in this class touches global RNG state or
wall-clock time.
"""

from __future__ import annotations

import random
from typing import Sequence

from .config import UL2Config
from .denoise import Example, build_prefix_denoising, build_span_corruption
from .errors import (
    SourceTooLongError,
    SourceTooShortError,
    TargetTooLongError,
    UL2Error,
)
from .mixture import ModeSampler
from .packing import build_packed_span_corruption
from .spans import sample_prefix_split, sample_spans
from .special_tokens import SpecialTokens
from .telemetry import MixtureTelemetry


class UL2Objective:
    """Builds UL2 denoising examples from token-ID sequences."""

    def __init__(
        self,
        config: UL2Config,
        specials: SpecialTokens,
        telemetry: MixtureTelemetry | None = None,
    ) -> None:
        # Prove the configuration can never overflow before producing a single
        # example. A sentinel or decoder-target overflow found here costs
        # milliseconds; found mid-run it costs the run.
        config.validate_capacity(specials.num_sentinels)
        # The single-document proof above does not cover packed windows, whose
        # sentinel need is a *sum* over the documents packed in -- see
        # `worst_case_packed_spans`. Proven here too, at the same moment, so
        # neither path can start a run it cannot finish.
        config.validate_packed_capacity(specials.num_sentinels)

        self.config = config
        self.specials = specials
        self.telemetry = telemetry
        self._sampler = ModeSampler(config.mode_weights)

    @property
    def sampler(self) -> ModeSampler:
        return self._sampler

    # -- main entry point -------------------------------------------------

    def corrupt(
        self,
        tokens: Sequence[int],
        rng: random.Random,
        *,
        mode: str | None = None,
    ) -> Example:
        """
        Build one denoising example.

        `mode` forces a specific denoiser (used by frozen evaluation, where
        the mixture is fixed rather than sampled). When omitted, the mode is
        drawn from the configured mixture.

        Raises `SourceTooShortError` / `SourceTooLongError` for inadmissible
        input -- see `try_corrupt` for the streaming-pipeline form that skips
        instead of raising.
        """
        tokens, truncated = self._admit(tokens)

        # Drawn before dispatch, and always from the same RNG, so that the
        # sequence of modes for a given seed does not depend on how many
        # random numbers a particular mode happens to consume afterwards.
        chosen = mode if mode is not None else self._sampler.sample(rng)

        if chosen == "S":
            example = self._build_prefix(tokens, rng, truncated=truncated)
        else:
            example = self._build_spans(tokens, chosen, rng, truncated=truncated)

        self._check_budgets(example)

        if self.telemetry is not None:
            self.telemetry.record(example)
        return example

    def try_corrupt(
        self,
        tokens: Sequence[int],
        rng: random.Random,
        *,
        mode: str | None = None,
    ) -> Example | None:
        """
        As `corrupt`, but returns `None` for inadmissible input.

        A streaming corpus contains blank lines, one-word titles and the
        occasional enormous record; a pretraining run should skip those, not
        die on them. The skip is *counted* in telemetry, so silently losing a
        large fraction of the corpus stays visible.
        """
        try:
            return self.corrupt(tokens, rng, mode=mode)
        except SourceTooShortError:
            if self.telemetry is not None:
                self.telemetry.record_skipped_too_short()
            return None
        except SourceTooLongError:
            if self.telemetry is not None:
                self.telemetry.record_skipped_too_long()
            return None

    # -- packed windows ---------------------------------------------------

    def corrupt_packed(
        self,
        documents: Sequence[Sequence[int]],
        rng: random.Random,
        *,
        mode: str | None = None,
    ) -> Example:
        """
        Build one denoising example from several documents packed together.

        `[R]`/`[X]` corrupt every document separately and concatenate the
        results into one window, so a masked span can never straddle two
        documents (`packing.py` explains the construction and why the position
        bias needs no change).

        `[S]` is **never packed**: a packed prefix-denoising target would be
        several continuations concatenated with only their order to say which
        prefix each belongs to. Passing more than one document with `mode="S"`
        is a caller error and raises, rather than silently training the model
        on a task nobody asked for.

        A single-document call is not a special case that needs avoiding --
        it produces exactly what `corrupt` would, plus segment ids.
        """
        if not documents:
            raise UL2Error("cannot build a packed example from zero documents")

        chosen = mode if mode is not None else self._sampler.sample(rng)

        if chosen == "S":
            if len(documents) != 1:
                raise UL2Error(
                    f"[S] takes exactly one document, got {len(documents)}. Prefix "
                    f"denoising has no sentinels, so a packed [S] target could not say "
                    f"which prefix each continuation belongs to -- pack [R]/[X] instead "
                    f"(UL2/src/packing.py)."
                )
            tokens, truncated = self._admit(documents[0])
            example = self._build_prefix(tokens, rng, truncated=truncated)
            self._check_budgets(example)
            if self.telemetry is not None:
                self.telemetry.record(example)
            return example

        admitted: list[Sequence[int]] = []
        truncated_any = False
        for document in documents:
            tokens, truncated = self._admit(document)
            truncated_any = truncated_any or truncated
            admitted.append(tokens)

        if len(admitted) > self.config.max_documents_per_window:
            raise UL2Error(
                f"{len(admitted)} documents exceed max_documents_per_window="
                f"{self.config.max_documents_per_window}; the sentinel-budget proof "
                f"(UL2Config.validate_packed_capacity) assumes that cap holds"
            )

        span_config = self.config.span_config(chosen)
        example = build_packed_span_corruption(
            admitted,
            chosen,
            self.specials,
            rng,
            corruption_rate=span_config.corruption_rate,
            mean_span_length=span_config.mean_span_length,
            append_eos_to_target=self.config.append_eos_to_target,
            truncated=truncated_any,
        )

        self._check_budgets(example)
        if self.telemetry is not None:
            self.telemetry.record(example)
        return example

    def try_corrupt_packed(
        self,
        documents: Sequence[Sequence[int]],
        rng: random.Random,
        *,
        mode: str | None = None,
    ) -> Example | None:
        """`corrupt_packed`, returning `None` for inadmissible input."""
        try:
            return self.corrupt_packed(documents, rng, mode=mode)
        except SourceTooShortError:
            if self.telemetry is not None:
                self.telemetry.record_skipped_too_short()
            return None
        except SourceTooLongError:
            if self.telemetry is not None:
                self.telemetry.record_skipped_too_long()
            return None

    # -- internals --------------------------------------------------------

    def _admit(self, tokens: Sequence[int]) -> tuple[Sequence[int], bool]:
        """Apply the length policy. Returns the (possibly truncated) tokens."""
        length = len(tokens)

        if length < self.config.min_source_length:
            raise SourceTooShortError(
                f"sequence of {length} tokens is below min_source_length="
                f"{self.config.min_source_length}; corrupting it would produce "
                f"degenerate spans"
            )

        if length > self.config.max_source_length:
            if self.config.on_overlong_source == "error":
                raise SourceTooLongError(
                    f"sequence of {length} tokens exceeds max_source_length="
                    f"{self.config.max_source_length}. Chunk it upstream, or set "
                    f"on_overlong_source='truncate' to accept the loss explicitly "
                    f"(architecture.md 4.5)."
                )
            return tokens[: self.config.max_source_length], True

        return tokens, False

    def _build_spans(
        self,
        tokens: Sequence[int],
        mode: str,
        rng: random.Random,
        *,
        truncated: bool,
    ) -> Example:
        span_config = self.config.span_config(mode)
        spans = sample_spans(
            len(tokens),
            span_config.corruption_rate,
            span_config.mean_span_length,
            rng,
            max_spans=self.specials.num_sentinels,
        )
        return build_span_corruption(
            tokens,
            spans,
            mode,
            self.specials,
            append_eos_to_source=self.config.append_eos_to_source,
            append_eos_to_target=self.config.append_eos_to_target,
            truncated=truncated,
        )

    def _build_prefix(
        self,
        tokens: Sequence[int],
        rng: random.Random,
        *,
        truncated: bool,
    ) -> Example:
        prefix_length = sample_prefix_split(
            len(tokens),
            self.config.s.min_prefix_fraction,
            self.config.s.max_prefix_fraction,
            rng,
        )
        return build_prefix_denoising(
            tokens,
            prefix_length,
            self.specials,
            append_eos_to_source=self.config.append_eos_to_source,
            append_eos_to_target=self.config.append_eos_to_target,
            truncated=truncated,
        )

    def _check_budgets(self, example: Example) -> None:
        """
        Post-condition. `validate_capacity` should have made these
        unreachable; they stay because the cost of an overflowing target
        reaching the trainer is training on truncated reconstructions
        (architecture.md 4.4), and the cost of this check is one comparison.
        """
        if example.target_length > self.config.max_target_length:
            raise TargetTooLongError(
                f"[{example.mode}] produced a {example.target_length}-token target "
                f"from a {example.source_length}-token source, exceeding "
                f"max_target_length={self.config.max_target_length}"
            )
        if example.encoder_length > self.config.max_source_length:
            raise TargetTooLongError(
                f"[{example.mode}] produced a {example.encoder_length}-token encoder "
                f"input from a {example.source_length}-token source, exceeding "
                f"max_source_length={self.config.max_source_length}"
            )
