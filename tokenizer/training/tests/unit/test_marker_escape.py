"""
Unit tests for marker_escape.py.

Full context: Marker_Encoder_Design_EDGE_CASE.md. This is a self-contained
correctness proof for the escape scheme, independent of SentencePiece --
if the round-trip breaks here, it breaks everywhere that uses it.
"""

from __future__ import annotations

import itertools
import random

from marker_escape import (
    SPIECE_UNDERLINE,
    _ESC,
    _ESCAPED_UNDERLINE,
    escape_markers,
    unescape_markers,
)


def roundtrips(text: str) -> bool:
    return unescape_markers(escape_markers(text)) == text


def test_ordinary_text_is_untouched():
    text = "the quick brown fox"
    assert escape_markers(text) == text
    assert unescape_markers(text) == text


def test_empty_string():
    assert roundtrips("")


def test_single_underline_character():
    assert roundtrips(SPIECE_UNDERLINE)


def test_underline_inside_real_text():
    # The exact failure text discovered during evaluation.
    assert roundtrips(" ▃▁▁ Low ")


def test_run_of_underlines():
    assert roundtrips("▁▁▁▁▁▁▁▁")


def test_underline_adjacent_to_ordinary_chars():
    for text in ["a▁b", "▁a", "a▁", "a▁▁b▁c▁"]:
        assert roundtrips(text)


def test_literal_escape_character_alone():
    # The character our own scheme uses as its escape marker, appearing as
    # ordinary content (this is the case found in the real corpus, inside a
    # font-icon string in c4_4000mb.jsonl).
    assert roundtrips(_ESC)


def test_literal_escape_character_in_text():
    assert roundtrips(f"icon:{_ESC} chart")


def test_the_adversarial_case_escape_then_flag_adjacent():
    """
    The exact edge case that breaks naive two-pass find-and-replace: a real
    _ESC immediately followed by a real _ESCAPED_UNDERLINE, with no ▁
    involved at all. A blind "replace all _ESC with _ESC+_ESC, then replace
    all ▁ with _ESC+_ESCAPED_UNDERLINE" pipeline manufactures an
    _ESC+_ESCAPED_UNDERLINE pair from the *first* pass's own output and
    misreads it, corrupting the text on unescape. The single-pass
    implementation must not.
    """
    text = f"X{_ESC}{_ESCAPED_UNDERLINE}Y"
    assert SPIECE_UNDERLINE not in text
    escaped = escape_markers(text)
    assert unescape_markers(escaped) == text


def test_literal_escape_and_flag_repeated_and_interleaved_with_underline():
    text = f"{_ESC}{_ESCAPED_UNDERLINE}{_ESC}{SPIECE_UNDERLINE}{_ESCAPED_UNDERLINE}{_ESC}"
    assert roundtrips(text)


def test_escaped_text_never_contains_a_bare_unpaired_underline():
    """
    The whole point of escaping: after escape_markers, sp.encode/decode
    should never see a real ▁ that isn't SentencePiece's own marker.
    """
    for text in ["▁", " ▃▁▁ Low ", f"{_ESC}{SPIECE_UNDERLINE}", "▁▁▁"]:
        assert SPIECE_UNDERLINE not in escape_markers(text)


def test_fuzz_random_strings_of_the_four_special_characters():
    """
    Exhaustive-ish stress test over short strings built purely from the four
    characters the scheme cares about, in every order -- the only alphabet
    where an escaping bug could hide.
    """
    alphabet = [_ESC, _ESCAPED_UNDERLINE, SPIECE_UNDERLINE, "x"]
    for length in range(0, 6):
        for combo in itertools.product(alphabet, repeat=length):
            text = "".join(combo)
            assert roundtrips(text), f"failed for {text!r}"


def test_fuzz_random_strings_with_ordinary_text_mixed_in():
    rng = random.Random(1234)
    alphabet = [_ESC, _ESCAPED_UNDERLINE, SPIECE_UNDERLINE, "a", "b", " ", "\n", "1"]
    for _ in range(2000):
        length = rng.randint(0, 20)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        assert roundtrips(text), f"failed for {text!r}"


def test_escaped_output_is_idempotent_under_a_second_roundtrip():
    # Escaping/unescaping twice in a row (e.g. a retry path) must not drift.
    text = f"a{SPIECE_UNDERLINE}b{_ESC}c"
    once = unescape_markers(escape_markers(text))
    twice = unescape_markers(escape_markers(once))
    assert once == text == twice

