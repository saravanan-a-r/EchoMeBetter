"""
Records, not lines — `reader.join_record_lines` / `reader.iter_records` and
`TokenizedCorpus`'s iteration unit.

Why this file exists
--------------------
The reader used to split every record on newlines and emit each line as its
own training example. Nothing failed: the pipeline ran, batches were produced,
the loss fell. What it cost was invisible from the outside — a man page's
example separated from the paragraph explaining it, a book's sentence
separated from its paragraph, an average example of ~61 tokens against a model
sized for 2048, and batches that were mostly padding as a result.

That is precisely the kind of regression a test suite has to hold down,
because no assertion anywhere else in the project would notice it coming back.
"""

from __future__ import annotations

import json

import pytest

from src.reader import iter_lines, iter_records, join_record_lines
from src.tokenized_corpus import TokenizedCorpus


class FakeTokenizer:
    """Encodes to one id per character, so token counts are readable."""

    def encode(self, text, out_type=int):
        return [ord(c) % 97 + 8 for c in text]


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    records = [
        {"text": "NAME\n    find - search for files\n\nSYNOPSIS\n    find [path]"},
        {"text": "a single line record"},
        {"text": "first\nsecond\nthird"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# -- join_record_lines ------------------------------------------------------


def test_a_records_line_structure_survives():
    assert join_record_lines("alpha\nbeta\ngamma") == "alpha\nbeta\ngamma"


def test_blank_lines_and_stray_indentation_are_normalized_away():
    assert join_record_lines("alpha\n\n   beta   \n\n\ngamma\n") == "alpha\nbeta\ngamma"


def test_a_record_of_only_whitespace_becomes_empty():
    assert join_record_lines("\n  \n\n") == ""


def test_a_single_line_record_is_returned_unchanged():
    assert join_record_lines("just one line") == "just one line"


# -- iter_records vs iter_lines --------------------------------------------


def test_iter_records_yields_whole_documents(corpus_file):
    records = list(iter_records([corpus_file]))
    assert len(records) == 3
    assert records[0].startswith("NAME\n")
    assert "SYNOPSIS" in records[0]
    assert records[2] == "first\nsecond\nthird"


def test_iter_lines_still_fragments_and_is_not_what_data_paths_use(corpus_file):
    """
    `iter_lines` is retained for inspection, and this test pins the difference
    so nobody reaches for it in a data path by accident: it produces many more
    pieces from the same file.
    """
    assert len(list(iter_lines([corpus_file]))) > len(list(iter_records([corpus_file])))


def test_training_and_evaluation_see_identically_shaped_text(corpus_file):
    """
    The eval path (`iter_records`) and the training path (`TokenizedCorpus`)
    must normalize text the same way. If they diverge, a run is scored on
    differently-shaped text from the text it learned on, and the eval loss
    that selects the best checkpoint is measuring the wrong distribution —
    silently, because both sides still produce a plausible number.
    """
    corpus = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    from_training = [next(corpus) for _ in range(3)]
    from_eval = [FakeTokenizer().encode(r) for r in iter_records([corpus_file])]
    assert from_training == from_eval


# -- TokenizedCorpus --------------------------------------------------------


def test_one_record_yields_one_example(corpus_file):
    corpus = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    [next(corpus) for _ in range(3)]
    assert corpus.examples_emitted == 3
    assert corpus.lines_read == 3


def test_a_multi_line_record_is_not_split_into_several_examples(corpus_file):
    """The regression this module is named for."""
    corpus = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    first = next(corpus)
    # The whole man-page record, in one example, newlines intact.
    assert len(first) == len("NAME\nfind - search for files\nSYNOPSIS\nfind [path]")


def test_the_iterator_wraps_around_and_counts_epochs(corpus_file):
    corpus = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    first_pass = [next(corpus) for _ in range(3)]
    assert corpus.epoch == 0
    second_pass = [next(corpus) for _ in range(3)]
    assert corpus.epoch == 1
    assert first_pass == second_pass


def test_unusable_records_are_skipped_and_counted(tmp_path):
    path = tmp_path / "messy.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": "usable record"}),
                "{not valid json",
                json.dumps({"no_text_field": 1}),
                json.dumps({"text": "   \n  \n "}),  # whitespace only
                json.dumps({"text": "another usable record"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = TokenizedCorpus([path], FakeTokenizer(), seed=0)
    [next(corpus) for _ in range(2)]
    assert corpus.examples_emitted == 2
    assert corpus.lines_skipped_unusable == 3


# -- resumability -----------------------------------------------------------


def test_resuming_lands_exactly_on_the_next_record(corpus_file):
    """
    A record is now both the unit of iteration and the unit of the saved
    position, so a resume can no longer land mid-record. The old
    line-splitting reader could re-read a partially-consumed record; this
    pins that it does not come back.
    """
    straight = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    expected = [next(straight) for _ in range(3)]

    interrupted = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    before = [next(interrupted) for _ in range(2)]
    state = interrupted.state_dict()

    resumed = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    resumed.load_state_dict(state)
    after = [next(resumed)]

    assert before + after == expected


def test_a_resumed_state_is_json_serializable(corpus_file):
    corpus = TokenizedCorpus([corpus_file], FakeTokenizer(), seed=0)
    next(corpus)
    assert json.loads(corpus.to_json_state())["line"] == 1
