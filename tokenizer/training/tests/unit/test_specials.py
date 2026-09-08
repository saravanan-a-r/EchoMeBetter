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
    assert len(specials.CLI_VOCAB_TOKENS) == 896


def test_total_counts_match_design():
    assert len(specials.USER_DEFINED_SYMBOLS) == 1269
    assert len(specials.BUILTIN_SPECIALS) == 3
    assert len(specials.ALL_NAMED_SPECIALS) == 1272


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
    assert specials.USER_DEFINED_SYMBOLS[372] == "<cli_reserved_63>"
    assert specials.USER_DEFINED_SYMBOLS[373] == specials.CLI_VOCAB_TOKENS[0]
    assert specials.USER_DEFINED_SYMBOLS[-1] == specials.CLI_VOCAB_TOKENS[-1]


def test_module_self_validates_on_import():
    """specials._validate() runs at import time; calling it again is a no-op."""
    specials._validate()


# -- CLI vocabulary ---------------------------------------------------------
#
# Unlike every other block, these pieces are matched against real text, so
# they can damage tokenization elsewhere. These are the locks on that.


# Command names that are also ordinary English words, or prefixes of them.
# Each was measured against the frozen tokenizer as ALREADY a single token,
# so admitting one buys nothing -- while as a vocabulary piece it forces a
# split everywhere the English word appears. Verified: a `cat` piece turns
# "category" into "cat"+"egory" and "concatenate" into "con"+"cat"+"enate".
# Prose fertility (1.31-1.39 tokens/word) is the rephrase task's core metric;
# this list is what stands between it and a well-meant CLI addition.
PROSE_UNSAFE = frozenset("""
cat head tail top less more make find sort test date time file tar dig mount
watch install locate tree cut type open env clear port kind true false say rev
diff stat comm du cal vi su tr ar pr mas bat factor groups expr service link
read write join look yes wait jobs history man touch patch screen script
strings which expand split paste nice free last at id w seq sync column node go
code set export source eval exec exit help info times trap shift local declare
readonly command hash kill ping host route arp ip ss nc sed awk cd ls rm cp mv
""".split())


def test_cli_vocabulary_pieces_all_carry_the_word_boundary_marker():
    """
    A piece must carry U+2581 to replace a word boundary. Without it the
    boundary is emitted as its own token and cancels the saving the piece
    exists for: measured on this project's trainer settings, "ls -la /dev/null"
    is 6 tokens with unmarked pieces against 4 with marked ones.
    """
    for token in specials.CLI_VOCAB_TOKENS:
        assert token.startswith("▁"), token
        assert len(token) > 1, token
        assert "▁" not in token[1:], token


def test_cli_vocabulary_never_admits_an_english_word():
    """The regression lock that matters most -- see PROSE_UNSAFE above."""
    admitted = {t.lstrip("▁") for t in specials.CLI_VOCAB_TOKENS}
    collisions = sorted(admitted & PROSE_UNSAFE)
    assert not collisions, (
        f"{collisions} are ordinary English words (or prefixes of them) and are "
        f"already single tokens; as vocabulary pieces they would split English prose"
    )


def test_every_letters_only_cli_piece_is_a_command_or_a_proper_noun_filename():
    """
    A pure-letter piece is the only kind that can collide with English, so
    only two sorts are admitted: lowercase command names (vetted individually
    against PROSE_UNSAFE above), and capitalized filenames -- `Dockerfile`,
    `Gemfile`, `Vagrantfile` -- which cannot match lowercase prose at all.
    Everything else carries '-', '/', '.', '~' or a digit, which no English
    word does, and so is safe by construction.
    """
    for text in specials._CLI_FLAG_STRINGS:
        assert not text.isalpha(), f"{text!r} is a pure-letter flag"
    for text in specials._CLI_PATH_STRINGS:
        if text.isalpha():
            assert text[0].isupper(), (
                f"{text!r} is a pure-letter path entry but is not a proper-noun "
                f"filename; lowercase it would be free to match English prose"
            )


def test_cli_vocabulary_groups_are_disjoint():
    groups = (specials._CLI_COMMAND_STRINGS,
              specials._CLI_PATH_STRINGS,
              specials._CLI_FLAG_STRINGS)
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            assert not set(left) & set(right), set(left) & set(right)
    all_text = [t for g in groups for t in g]
    assert len(set(all_text)) == len(all_text), "duplicate CLI vocabulary entry"


def test_cli_vocabulary_does_not_collide_with_the_reserved_name_blocks():
    """
    `<cli_reserved_N>` is 64 placeholder *names* awaiting a meaning; the CLI
    vocabulary is literal text. Different mechanisms, kept disjoint so neither
    can be mistaken for the other.
    """
    named = set(specials.EXTRA_ID_TOKENS + specials.UL2_MODE_TOKENS
                + specials.STYLE_TOKENS + specials.STRUCTURAL_TOKENS
                + specials.RESERVED_TOKENS + specials.CLI_TOKENS)
    assert not named & set(specials.CLI_VOCAB_TOKENS)


def test_cli_vocabulary_entries_are_stripped_and_non_empty():
    for group in (specials._CLI_COMMAND_STRINGS,
                  specials._CLI_PATH_STRINGS,
                  specials._CLI_FLAG_STRINGS):
        for text in group:
            assert text and text == text.strip(), repr(text)
            assert not any(ch.isspace() for ch in text), repr(text)


def test_flag_and_path_entries_look_like_what_they_claim_to_be():
    for text in specials._CLI_FLAG_STRINGS:
        assert text.startswith("-"), text
    for text in specials._CLI_PATH_STRINGS:
        # an absolute path, a dotfile, a ~/-rooted path, or a filename -- all
        # of which carry a character no English word does ('/', '.', '_', a
        # digit) unless they are a proper-noun filename like `Dockerfile`.
        assert not text.isalpha() or text[0].isupper(), text
