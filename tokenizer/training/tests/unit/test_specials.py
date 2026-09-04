"""
Unit tests for the reserved-token contract.

These are regression locks, not sanity checks. The counts asserted here are
welded to every embedding row of every model trained on this tokenizer
(DESIGN.md 5.2): changing one after pretraining begins invalidates the
weights. A test failure here means someone is about to make an unrecoverable
change, and should be read as "stop", not "update the expected number".
"""

from __future__ import annotations

import re

import specials


def test_group_counts_match_design():
    assert len(specials.EXTRA_ID_TOKENS) == 256
    assert len(specials.UL2_MODE_TOKENS) == 3
    assert len(specials.STYLE_TOKENS) == 12
    assert len(specials.STRUCTURAL_TOKENS) == 2
    assert len(specials.RESERVED_TOKENS) == 36
    assert len(specials.CLI_TOKENS) == 64


def test_total_counts_match_design():
    assert len(specials.USER_DEFINED_SYMBOLS) == 373
    assert len(specials.BUILTIN_SPECIALS) == 3
    assert len(specials.ALL_NAMED_SPECIALS) == 376


def test_sentinel_count_covers_the_encoder_context():
    """
    architecture.md 6.3: each masked span consumes one uniquely-numbered
    sentinel, so the requirement is

        max_encoder_length x corruption_rate / mean_span_length

    R-denoising at the 2048 encoder context needs ~103. T5's 100 was sized
    for a 512-token context and would overflow here on any input above
    2,000 tokens -- silently dropping spans. This test is the regression
    lock on that: if the encoder context grows, this must be recomputed.
    """
    max_encoder_length = 2048
    corruption_rate = 0.15
    mean_span_length = 3
    needed = max_encoder_length * corruption_rate / mean_span_length

    assert needed > 100, "the 100-sentinel default would have sufficed; recheck the premise"
    assert len(specials.EXTRA_ID_TOKENS) >= needed, (
        f"{len(specials.EXTRA_ID_TOKENS)} sentinels cannot cover {needed:.0f} spans "
        f"at {max_encoder_length} tokens / {corruption_rate:.0%} / span {mean_span_length}"
    )


def test_all_names_unique():
    names = specials.ALL_NAMED_SPECIALS
    assert len(set(names)) == len(names)


def test_builtin_specials_are_not_user_defined():
    """
    pad/eos/unk are assigned via pad_id/eos_id/unk_id. Listing them in
    user_defined_symbols too would make SentencePiece emit duplicate pieces
    and break the id contract.
    """
    assert not set(specials.BUILTIN_SPECIALS) & set(specials.USER_DEFINED_SYMBOLS)


def test_sentinels_are_contiguous_and_zero_indexed():
    """UL2 span corruption addresses sentinels by index; gaps would break it."""
    assert specials.EXTRA_ID_TOKENS[0] == "<extra_id_0>"
    assert specials.EXTRA_ID_TOKENS[-1] == "<extra_id_255>"
    for i, name in enumerate(specials.EXTRA_ID_TOKENS):
        assert name == f"<extra_id_{i}>"


def test_reserved_spares_are_contiguous_and_zero_indexed():
    for i, name in enumerate(specials.RESERVED_TOKENS):
        assert name == f"<reserved_{i}>"


def test_cli_reserved_tokens_are_contiguous_and_zero_indexed():
    for i, name in enumerate(specials.CLI_TOKENS):
        assert name == f"<cli_reserved_{i}>"


def test_cli_reserved_tokens_are_distinct_from_generic_spares():
    """
    Two different reserved blocks, deliberately not merged: RESERVED_TOKENS
    is generic emergency capacity, CLI_TOKENS is earmarked for the planned
    CLI-helper fine-tune. Keeping them disjoint is what lets a future
    Stage 2 assign CLI-specific meaning to `<cli_reserved_N>` without
    guessing whether a given `<reserved_N>` was meant for this or something
    else.
    """
    assert not set(specials.CLI_TOKENS) & set(specials.RESERVED_TOKENS)


def test_style_tokens_cover_all_twelve_styles():
    """DESIGN.md 5.4: 5 Phase-1 styles + 7 Phase-2 styles, all reserved now."""
    expected = {
        "grammar", "concise", "elaborate", "professional", "friendly",
        "assertive", "empathetic", "fluent", "formal", "paraphrase", "polite", "simplify",
    }
    got = {name[len("<style:"):-1] for name in specials.STYLE_TOKENS}
    assert got == expected


def test_style_token_format():
    for name in specials.STYLE_TOKENS:
        assert re.fullmatch(r"<style:[a-z]+>", name), name


def test_no_name_contains_whitespace():
    """A special containing whitespace cannot be an atomic piece (Gate 3)."""
    for name in specials.ALL_NAMED_SPECIALS:
        assert name == name.strip()
        assert not any(ch.isspace() for ch in name), name


def test_user_defined_symbols_order_is_stable():
    """
    Order determines nothing functionally (IDs are resolved by name), but a
    stable order keeps builds byte-reproducible across runs (NF5).
    """
    assert specials.USER_DEFINED_SYMBOLS[0] == "<extra_id_0>"
    assert specials.USER_DEFINED_SYMBOLS[256] == "[R]"
    assert specials.USER_DEFINED_SYMBOLS[259] == "<style:grammar>"
    assert specials.USER_DEFINED_SYMBOLS[273] == "<reserved_0>"
    assert specials.USER_DEFINED_SYMBOLS[309] == "<cli_reserved_0>"
    assert specials.USER_DEFINED_SYMBOLS[-1] == "<cli_reserved_63>"


def test_module_self_validates_on_import():
    """specials._validate() runs at import time; calling it again is a no-op."""
    specials._validate()
