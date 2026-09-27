"""
The text metrics: SARI against its published value, and the behaviours the
acceptance bar depends on.
"""

from __future__ import annotations

import pytest

from src import text_metrics as tm


def test_sari_reproduces_the_huggingface_documented_example():
    """HuggingFace `evaluate`'s SARI card: 26.953601953601954."""
    score = tm.sentence_sari(
        "About 95 species are currently accepted.",
        "About 95 you now get in.",
        [
            "About 95 species are currently known.",
            "About 95 species are now accepted.",
            "95 species are now accepted.",
        ],
    )
    assert score == pytest.approx(26.953601953601954, abs=1e-9)


def test_sari_ranks_the_reference_above_a_copy_of_the_input():
    """
    Ordering only: under HuggingFace's 0/0 = 1 convention a copy deletes
    nothing and gets full deletion precision, so it still scores ~88 here.
    That is why copy_rate and length_ratio are tracked beside SARI.
    """
    source = "Please make sure that you submit the report by Friday."
    reference = "Submit the report by Friday."
    assert tm.sentence_sari(source, reference, [reference]) == pytest.approx(100.0)
    assert tm.sentence_sari(source, source, [reference]) < 90.0


def test_chrf_extremes_and_whitespace_insensitivity():
    assert tm.sentence_chrf("the cat sat", "the cat sat") == pytest.approx(100.0)
    assert tm.sentence_chrf("thecat sat", "the cat sat") == pytest.approx(100.0)
    assert tm.sentence_chrf("xyz", "abc") == 0.0


def test_protected_spans_cover_the_things_a_rewrite_must_not_change():
    text = "See https://ex.com/a?b=1. Mail a.b@ex.com or @bob, run `ls -la` 3 times at 10:30 (95%)."
    assert tm.protected_spans(text) == ["https://ex.com/a?b=1", "a.b@ex.com", "@bob", "`ls -la`", "3", "10:30", "95%"]
    assert tm.span_preservation(text, "At 10:30 run `ls -la` 3 times; see https://ex.com/a?b=1, mail a.b@ex.com or @bob (95%)") == 1.0
    assert tm.span_preservation(text, "Run ls 3 times at 10:30.") == pytest.approx(2 / 7)
    assert tm.span_preservation("no numbers here", "anything") is None


def test_loop_detection_and_novel_words():
    assert tm.has_loop("I am not sure. I am not sure. I am not sure.")
    assert not tm.has_loop("I am not sure. I am quite sure now.")
    assert tm.novel_word_rate("the report is due friday", "the report is due friday") == 0.0
    assert tm.novel_word_rate("the report is due friday", "the invoice is due friday") == pytest.approx(0.2)
