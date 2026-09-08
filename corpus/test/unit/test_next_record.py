"""
`next_record`: the text behind `__next__`, for the one consumer that needs
text (the synthetic rewrite task in `rewrite/`).

What must not drift: the record `next_record` hands out is exactly the one
`__next__` would have tokenized, consumed from the same position, so a run
that mixes the two reads every record once. And a blend's `next_record`
draws from the same weighted table its `__next__` does.
"""

from __future__ import annotations

import json

from conftest import write_jsonl
from src.blended_corpus import BlendedCorpus
from src.reader import discover_corpus_files, group_files_by_source
from src.tokenized_corpus import TokenizedCorpus

RECORDS = [f"record number {i}\nwith a second line" for i in range(12)]


def corpus_file(corpus_dir):
    return write_jsonl(corpus_dir / "a.jsonl", [{"text": text} for text in RECORDS])


def test_next_record_hands_out_what_next_would_tokenize(corpus_dir, fake_processor):
    path = corpus_file(corpus_dir)
    text_stream = TokenizedCorpus([path], fake_processor, seed=3)
    id_stream = TokenizedCorpus([path], fake_processor, seed=3)
    for _ in range(30):
        record = text_stream.next_record()
        assert fake_processor.encode(record) == next(id_stream)
        assert "\n" in record  # line structure intact, as `__next__` keeps it


def test_the_two_reads_consume_the_same_position(corpus_dir, fake_processor):
    """Alternating between the two interfaces reads each record exactly once."""
    path = corpus_file(corpus_dir)
    plain = TokenizedCorpus([path], fake_processor, seed=3)
    expected = [fake_processor.encode(plain.next_record()) for _ in range(24)]

    mixed = TokenizedCorpus([path], fake_processor, seed=3)
    actual = [
        fake_processor.encode(mixed.next_record()) if index % 2 else next(mixed)
        for index in range(24)
    ]
    assert actual == expected


def test_next_record_counts_lines_but_not_examples(corpus_dir, fake_processor):
    stream = TokenizedCorpus([corpus_file(corpus_dir)], fake_processor)
    stream.next_record()
    assert stream.lines_read == 1
    assert stream.examples_emitted == 0  # nothing was tokenized here
    next(stream)
    assert stream.examples_emitted == 1


def test_a_resumed_stream_continues_next_record_exactly(corpus_dir, fake_processor):
    path = corpus_file(corpus_dir)
    uninterrupted = TokenizedCorpus([path], fake_processor, seed=5)
    expected = [uninterrupted.next_record() for _ in range(30)]

    first = TokenizedCorpus([path], fake_processor, seed=5)
    got = [first.next_record() for _ in range(13)]
    state = json.loads(json.dumps(first.state_dict()))
    second = TokenizedCorpus([path], fake_processor, seed=5)
    second.load_state_dict(state)
    got += [second.next_record() for _ in range(17)]
    assert got == expected


def test_a_blend_draws_records_by_weight_and_counts_them(tmp_path, fake_processor):
    root = tmp_path / "corpus"
    for name in ("big", "small"):
        (root / name).mkdir(parents=True)
        write_jsonl(root / name / f"{name}.jsonl", [{"text": f"{name} {i}"} for i in range(50)])
    grouped = group_files_by_source(discover_corpus_files(root), root)
    sources = {name: TokenizedCorpus(grouped[name], fake_processor) for name in grouped}
    blend = BlendedCorpus(sources, {"big": 3.0, "small": 1.0}, seed=1)

    records = [blend.next_record() for _ in range(400)]
    assert all(record.startswith(("big ", "small ")) for record in records)
    assert blend.documents_drawn["big"] + blend.documents_drawn["small"] == 400
    assert 0.65 < blend.realized_shares()["big"] < 0.85
