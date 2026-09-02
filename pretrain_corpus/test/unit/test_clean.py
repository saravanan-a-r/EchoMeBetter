from __future__ import annotations

from src.clean import clean_record, dedup_key, is_boilerplate, repair_mojibake


def test_clean_record_chunks_and_strips_boilerplate():
    text = (
        "Produced by a scanning volunteer\n\n"
        "The quick brown fox jumps over the lazy dog near the old mill "
        "every single morning before the sun rises over the hills.\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK ***"
    )
    chunks = clean_record(text)
    assert len(chunks) == 1
    assert "Produced by" not in chunks[0]
    assert "START OF THE PROJECT GUTENBERG" not in chunks[0]
    assert "quick brown fox" in chunks[0]


def test_clean_record_drops_short_text():
    assert clean_record("hi") == []


def test_clean_record_strips_enter_image_description_here_alt_text():
    text = (
        "See the wiring diagram below for reference.\n\n"
        "![enter image description here](https://example.com/img.png)\n\n"
        "That should clarify how the connector is oriented on the board."
    )
    [chunk] = clean_record(text)
    assert "enter image description here" not in chunk
    assert "https://example.com/img.png" in chunk


def test_clean_record_never_splits_a_url_or_code_fence_across_chunks():
    url = "https://example.com/" + "a" * 50
    text = ("word " * 600) + url + (" word" * 600)
    chunks = clean_record(text)
    assert any(url in chunk for chunk in chunks)


def test_is_boilerplate_catches_stackexchange_placeholder():
    assert is_boilerplate("enter image description here")
    assert not is_boilerplate("This is a real sentence about the topic.")


def test_repair_mojibake_fixes_double_encoded_utf8():
    broken = "itâ€™s a test"
    fixed = repair_mojibake(broken)
    assert "’" in fixed or "'" in fixed
    assert "â€™" not in fixed


def test_dedup_key_exact_match_is_stable():
    a = dedup_key("Hello world.")
    b = dedup_key("Hello world.")
    assert a.exact == b.exact
    assert a.normalized == b.normalized


def test_dedup_key_normalized_ignores_case_and_punctuation():
    a = dedup_key("Hello, World!")
    b = dedup_key("hello world")
    assert a.exact != b.exact  # not byte-identical
    assert a.normalized == b.normalized  # same after folding


def test_dedup_key_different_content_differs():
    a = dedup_key("Hello world.")
    b = dedup_key("Something else entirely.")
    assert a.exact != b.exact
    assert a.normalized != b.normalized
