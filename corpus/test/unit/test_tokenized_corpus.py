"""
`TokenizedCorpus` is the piece architecture.md §7.9 is about: "resumable
weights are not enough — the data pipeline must also be resumable." The one
test that matters most here is the same shape as
`training/test/integration/test_resumability.py`'s headline test: a run
split by a simulated crash must produce exactly what an uninterrupted run
would have, from the point of resumption onward.
"""

from __future__ import annotations

import itertools
import json

import pytest

from src.errors import CorpusConfigError
from src.tokenized_corpus import TokenizedCorpus

from conftest import write_jsonl, write_txt


def test_empty_file_list_is_rejected(fake_processor):
    with pytest.raises(CorpusConfigError):
        TokenizedCorpus([], fake_processor)


def test_yields_one_example_per_line(corpus_dir, fake_processor):
    write_txt(corpus_dir / "a.txt", ["one", "two", "three"])
    corpus = TokenizedCorpus([corpus_dir / "a.txt"], fake_processor, seed=0)
    examples = list(itertools.islice(corpus, 3))
    assert len(examples) == 3
    assert all(isinstance(e, list) and e for e in examples)


def test_is_infinite_and_wraps_into_a_new_epoch(corpus_dir, fake_processor):
    write_txt(corpus_dir / "a.txt", ["only line"])
    corpus = TokenizedCorpus([corpus_dir / "a.txt"], fake_processor, seed=0)
    first, second = next(corpus), next(corpus)
    assert first == second  # same line, tokenized identically, second time round
    assert corpus.epoch == 1


def test_file_order_is_deterministic_given_a_seed(corpus_dir, fake_processor):
    write_txt(corpus_dir / "a.txt", ["A"])
    write_txt(corpus_dir / "b.txt", ["B"])
    write_txt(corpus_dir / "c.txt", ["C"])
    files = [corpus_dir / "a.txt", corpus_dir / "b.txt", corpus_dir / "c.txt"]

    run_1 = TokenizedCorpus(files, fake_processor, seed=7)
    run_2 = TokenizedCorpus(files, fake_processor, seed=7)
    assert [next(run_1) for _ in range(3)] == [next(run_2) for _ in range(3)]


def test_skips_unusable_lines_without_raising(corpus_dir, fake_processor):
    write_jsonl(corpus_dir / "a.jsonl", [{"text": "good"}, {"nope": 1}, {"text": "also good"}])
    corpus = TokenizedCorpus([corpus_dir / "a.jsonl"], fake_processor, seed=0)
    examples = [next(corpus) for _ in range(2)]
    assert len(examples) == 2
    assert corpus.lines_skipped_unusable == 1


def test_multiline_jsonl_record_expands_to_multiple_examples(corpus_dir, fake_processor):
    write_jsonl(corpus_dir / "a.jsonl", [{"text": "line one\nline two"}])
    corpus = TokenizedCorpus([corpus_dir / "a.jsonl"], fake_processor, seed=0)
    first, second = next(corpus), next(corpus)
    assert first != second
    assert corpus.lines_read == 1
    assert corpus.examples_emitted == 2


def test_state_dict_round_trips_through_json(corpus_dir, fake_processor):
    write_txt(corpus_dir / "a.txt", ["one", "two"])
    corpus = TokenizedCorpus([corpus_dir / "a.txt"], fake_processor, seed=0)
    next(corpus)
    payload = json.loads(corpus.to_json_state())
    assert payload["pos"] == 0
    assert payload["line"] == 1


def test_a_resumed_stream_is_identical_to_an_uninterrupted_one(corpus_dir, fake_processor):
    write_txt(corpus_dir / "a.txt", [f"line {i}" for i in range(5)])
    write_txt(corpus_dir / "b.txt", [f"row {i}" for i in range(5)])
    write_jsonl(corpus_dir / "c.jsonl", [{"text": f"rec {i}\nrec {i} tail"} for i in range(3)])
    files = [corpus_dir / "a.txt", corpus_dir / "b.txt", corpus_dir / "c.jsonl"]

    straight = TokenizedCorpus(files, fake_processor, seed=99)
    straight_output = [next(straight) for _ in range(30)]

    crashed = TokenizedCorpus(files, fake_processor, seed=99)
    before_crash = [next(crashed) for _ in range(11)]
    state = crashed.state_dict()

    resumed = TokenizedCorpus(files, fake_processor, seed=99)
    resumed.load_state_dict(state)
    after_resume = [next(resumed) for _ in range(19)]

    assert before_crash + after_resume == straight_output


def test_resuming_with_a_changed_file_set_is_a_loud_error(corpus_dir, fake_processor):
    write_txt(corpus_dir / "a.txt", ["one"])
    write_txt(corpus_dir / "b.txt", ["two"])
    files = [corpus_dir / "a.txt", corpus_dir / "b.txt"]
    corpus = TokenizedCorpus(files, fake_processor, seed=0)
    state = corpus.state_dict()

    shrunk = TokenizedCorpus([corpus_dir / "a.txt"], fake_processor, seed=0)
    with pytest.raises(CorpusConfigError):
        shrunk.load_state_dict(state)


def test_resuming_mid_multiline_record_skips_rather_than_duplicates(corpus_dir, fake_processor):
    """
    Documented trade-off in the module docstring: the sub-segments of a raw
    line are not individually resumable, so a crash mid-record loses (never
    duplicates) the remainder of that one record. Single file, so the result
    does not depend on the seeded file-shuffle order.
    """
    write_jsonl(
        corpus_dir / "a.jsonl",
        [{"text": "seg one\nseg two\nseg three"}, {"text": "second record"}],
    )
    files = [corpus_dir / "a.jsonl"]

    crashed = TokenizedCorpus(files, fake_processor, seed=3)
    next(crashed)  # consumes "seg one" only, leaving "seg two"/"seg three" pending
    state = crashed.state_dict()
    assert state["line"] == 1  # the whole raw jsonl line is already marked read

    resumed = TokenizedCorpus(files, fake_processor, seed=3)
    resumed.load_state_dict(state)
    # No pending segments survive resumption: the next example must come from
    # the second record, not from the dropped "seg two"/"seg three".
    following = next(resumed)
    assert following == resumed._tokenize("second record")
    assert following != resumed._tokenize("seg two")
