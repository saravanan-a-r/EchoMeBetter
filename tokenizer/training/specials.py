"""
Single source of truth for reserved token names (DESIGN.md 5.2, 5.3).

Nothing else in this codebase — or in pretraining / SFT / DPO / inference —
defines these names. They are resolved to integer IDs only after training,
via `sp.piece_to_id(name)`, and frozen into token_map.json. No component may
hardcode a token ID.

pad/eos/unk are NOT listed here: they are assigned via SentencePieceTrainer's
pad_id/eos_id/unk_id arguments, not via user_defined_symbols. bos_id=-1 (no
BOS token) per DESIGN.md 6.

Counts (architecture.md 6.2):
    extra_id sentinels    256
    UL2 mode tokens         3
    style tokens           12
    structural markers      2
    spare reserved         36
    ----------------------------
    user_defined_symbols  309
    (+ pad/eos/unk = 312 named specials total; with the 256 byte_fallback
    pieces added automatically, 568 of 32,768 slots are reserved (1.73%),
    leaving 32,200 learned pieces)
"""

from __future__ import annotations

# Span-corruption sentinels (UL2/T5 style). Pretraining Stage 1 requires
# these; there is no way to add them later without an embedding-table
# resize and a full retrain.
#
# 256, not T5's 100 (architecture.md 6.3). Each masked span consumes one
# uniquely-numbered sentinel, so the requirement is:
#
#     max_encoder_length x corruption_rate / mean_span_length
#     = 2048 x 0.15 / 3 = ~103 for R-denoising at our 2048 context
#
# T5's 100 was sized for a 512-token context; here it would bind at exactly
# 2,000 input tokens -- just below the encoder limit -- and overflow. 256
# gives 2.5x headroom, also covers a 4096 context if that is ever revisited,
# and costs 156 embedding rows (~0.02% of the model).
EXTRA_ID_TOKENS: list[str] = [f"<extra_id_{i}>" for i in range(256)]

# UL2 mixture-of-denoisers mode tokens (architecture.md 7.1).
UL2_MODE_TOKENS: list[str] = ["[R]", "[X]", "[S]"]

# All 12 rephrase styles (DESIGN.md 5.4) -- Phase 1 (trained now) and
# Phase 2 (reserved, data not trained yet). Reserve all 12 now: the cost of
# an unused embedding row is trivial; the cost of adding one later is a
# full retrain.
_PHASE_1_STYLES = ["grammar", "concise", "elaborate", "professional", "friendly"]
_PHASE_2_STYLES = ["assertive", "empathetic", "fluent", "formal", "paraphrase", "polite", "simplify"]
STYLE_TOKENS: list[str] = [f"<style:{name}>" for name in _PHASE_1_STYLES + _PHASE_2_STYLES]

# Structural markers delimiting the span the model must rewrite.
STRUCTURAL_TOKENS: list[str] = ["<text_to_rewrite>", "</text_to_rewrite>"]

# Spare capacity for unforeseen needs (DESIGN.md 5.2). Deliberately NOT
# spent on the sentinel expansion above -- sentinels were provisioned
# properly instead, so this stays genuine emergency capacity.
RESERVED_TOKENS: list[str] = [f"<reserved_{i}>" for i in range(36)]

USER_DEFINED_SYMBOLS: list[str] = (
    EXTRA_ID_TOKENS + UL2_MODE_TOKENS + STYLE_TOKENS + STRUCTURAL_TOKENS + RESERVED_TOKENS
)

# Built-in specials assigned via pad_id/eos_id/unk_id, not user_defined_symbols.
# Listed here only so validate_tokenizer.py can check atomicity (Gate 3)
# against the full 312-name set without a second source of truth.
BUILTIN_SPECIALS: list[str] = ["<pad>", "</s>", "<unk>"]

ALL_NAMED_SPECIALS: list[str] = BUILTIN_SPECIALS + USER_DEFINED_SYMBOLS


def _validate() -> None:
    assert len(EXTRA_ID_TOKENS) == 256, len(EXTRA_ID_TOKENS)
    assert len(UL2_MODE_TOKENS) == 3, len(UL2_MODE_TOKENS)
    assert len(STYLE_TOKENS) == 12, len(STYLE_TOKENS)
    assert len(STRUCTURAL_TOKENS) == 2, len(STRUCTURAL_TOKENS)
    assert len(RESERVED_TOKENS) == 36, len(RESERVED_TOKENS)
    assert len(USER_DEFINED_SYMBOLS) == 309, len(USER_DEFINED_SYMBOLS)
    assert len(ALL_NAMED_SPECIALS) == 312, len(ALL_NAMED_SPECIALS)
    assert len(set(ALL_NAMED_SPECIALS)) == len(ALL_NAMED_SPECIALS), "duplicate special token name"


_validate()
