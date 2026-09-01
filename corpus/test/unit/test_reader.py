from __future__ import annotations

import pytest

from src.errors import CorpusConfigError
from src.reader import discover_corpus_files, extract_text, iter_lines

from conftest import write_jsonl, write_txt


def test_discover_corpus_files_rejects_missing_path(tmp_path):
    with pytest.raises(CorpusConfigError):
        discover_corpus_files(tmp_path / "does_not_exist")


def test_discover_corpus_files_rejects_empty_directory(corpus_dir):
    with pytest.raises(CorpusConfigError):
        discover_corpus_files(corpus_dir)


def test_discover_corpus_files_ignores_unrelated_suffixes(corpus_dir):
    write_txt(corpus_dir / "a.txt", ["hello"])
    (corpus_dir / "notes.md").write_text("ignored", encoding="utf-8")
    files = discover_corpus_files(corpus_dir)
    assert [f.name for f in files] == ["a.txt"]


def test_discover_corpus_files_is_sorted(corpus_dir):
    write_txt(corpus_dir / "b.txt", ["b"])
    write_txt(corpus_dir / "a.txt", ["a"])
    files = discover_corpus_files(corpus_dir)
    assert [f.name for f in files] == ["a.txt", "b.txt"]


def test_discover_corpus_files_accepts_a_single_file(corpus_dir):
    path = write_txt(corpus_dir / "a.txt", ["a"])
    assert discover_corpus_files(path) == [path]


def test_extract_text_plain_line():
    assert extract_text("hello world", jsonl=False) == "hello world"


def test_extract_text_plain_blank_line_is_none():
    assert extract_text("   ", jsonl=False) is None


def test_extract_text_jsonl_pulls_text_field():
    assert extract_text('{"text": "hi", "other": 1}', jsonl=True) == "hi"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "{}",
        '{"text": 5}',
        '{"text": ""}',
        '{"text": "   "}',
        "[1, 2, 3]",
    ],
)
def test_extract_text_jsonl_rejects_unusable_records(raw):
    assert extract_text(raw, jsonl=True) is None


def test_iter_lines_splits_multiline_jsonl_records(corpus_dir):
    write_jsonl(corpus_dir / "a.jsonl", [{"text": "line one\nline two"}])
    assert list(iter_lines(discover_corpus_files(corpus_dir))) == ["line one", "line two"]


def test_iter_lines_skips_blank_and_malformed(corpus_dir):
    write_jsonl(
        corpus_dir / "a.jsonl",
        [{"text": "good"}, {"nope": 1}, {"text": "  "}, {"text": "also good"}],
    )
    assert list(iter_lines(discover_corpus_files(corpus_dir))) == ["good", "also good"]


def test_iter_lines_reads_across_multiple_files_in_sorted_order(corpus_dir):
    write_txt(corpus_dir / "b.txt", ["from b"])
    write_txt(corpus_dir / "a.txt", ["from a"])
    assert list(iter_lines(discover_corpus_files(corpus_dir))) == ["from a", "from b"]


def test_iter_lines_never_rewrites_text(corpus_dir):
    """
    architecture.md NF1, carried into this module: no normalization, no
    stripped internal characters, only whitespace-stripped line boundaries.
    """
    text = "  keeps   internal   spacing  "
    write_txt(corpus_dir / "a.txt", [text])
    (line,) = list(iter_lines(discover_corpus_files(corpus_dir)))
    assert line == text.strip()
    assert "internal   spacing" in line
