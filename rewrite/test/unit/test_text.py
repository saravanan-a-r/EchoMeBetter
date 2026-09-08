"""
Segmentation: what the operators may touch and what they must not.

A wrong "prose" verdict or a missed protected span does not raise anywhere
downstream -- it produces a target that rewrites code, a URL or a number,
which is the one thing the product promises never to do. These pin the
boundary.
"""

from __future__ import annotations

import pytest

from src.text import is_prose_line, protected_spans, split_line


def test_pieces_concatenate_back_to_the_line():
    line = "Check https://api.example.com/v2 for the 503s, then e-mail ops@example.com."
    assert "".join(piece.text for piece in split_line(line)) == line


@pytest.mark.parametrize(
    "span",
    [
        "https://api.example.com/v2/users?x=1",
        "www.example.com/path",
        "first.last@example.com",
        "/usr/local/bin/python3",
        "./scripts/run.sh",
        "~/notes.txt",
        "`git rebase -i`",
        "<pad>",
        "<text_to_rewrite>",
        "#hashtag",
        "@handle",
        "$HOME",
        "snake_case_name",
        "self-driving",
        "v1.2.3",
        "2024",
        "40%",
        "3rd",
        "x86",
        "NASA",
        "getValue",
    ],
)
def test_span_is_protected(span):
    pieces = split_line(f"the {span} thing")
    kinds = {piece.text: piece.kind for piece in pieces}
    assert kinds[span] == "protected"


def test_ordinary_words_and_contractions_are_words():
    kinds = {piece.text: piece.kind for piece in split_line("I don't think John's cat")}
    assert kinds["don't"] == "word"
    assert kinds["John's"] == "word"
    assert kinds["cat"] == "word"


def test_punctuation_is_one_piece_per_character():
    pieces = [piece for piece in split_line("well, no...") if piece.kind == "punct"]
    assert [piece.text for piece in pieces] == [",", ".", ".", "."]


# -- the prose gate ---------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "The quarterly report showed strong growth in the European market.",
        "- the committee reviewed the roadmap and agreed on next steps",
        "I think we're going to need a bigger boat, honestly.",
    ],
)
def test_running_text_is_prose(line):
    assert is_prose_line(line)


@pytest.mark.parametrize(
    "line",
    [
        "def parse(line):",
        "return json.loads(line)",
        "import os, sys, json, pathlib, itertools",
        "$ tar -xzf archive.tar.gz --directory /tmp",
        "# Installation and first steps",
        "| name | value | notes | more |",
        "{ \"key\": \"value\", \"other\": 1 }",
        "THIS IS A HEADING WITH ONLY CAPITALS",
        "Introduction",
        "x = compute(y) + other(z) * 2",
        "if (a && b || c) { return d; }",
        "See the 2024-01-01 report: 40% up, 3.5x, 1.2B, $5M, 9/10.",
    ],
)
def test_structure_and_code_are_not_prose(line):
    assert not is_prose_line(line)


def test_protected_spans_are_reported_in_order():
    text = "Mail a@b.com about https://x.io/1 then run `ls`.\nSecond line has 42 items."
    assert protected_spans(text) == ["a@b.com", "https://x.io/1", "`ls`", "42"]
