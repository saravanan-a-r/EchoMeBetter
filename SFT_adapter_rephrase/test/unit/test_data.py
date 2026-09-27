"""
The dataset: the exact frame, every filter, and the split's stability
properties that keep evaluations comparable across runs.
"""

from __future__ import annotations

import json

import pytest

from conftest import (
    REAL_DATASET_DIR,
    REAL_TOKENIZER_DIR,
    fake_encode,
    make_config,
    record,
    sample_records,
    write_dataset,
)
from src.codec import TextCodec
from src.data import build_splits
from src.errors import DatasetError


def _splits(tmp_path, rows, files: int = 1, **data):
    write_dataset(tmp_path / "data", rows, files=files)
    config = make_config(tmp_path, data=data)
    return config, config.data


def test_frame_is_exactly_the_product_frame(tmp_path, codec):
    rows = [record("make it short please", "make it short"), record("another longer input", "another input")]
    _, data = _splits(tmp_path, rows, train_examples=1, eval_examples=1)
    train, evaluation, _ = build_splits(data, codec)
    for example in train + evaluation:
        text_ids = fake_encode(example.source_text)
        assert list(example.encoder_input_ids) == [3, 4, *text_ids, 5, 1]
        assert list(example.target_ids) == [*fake_encode(example.target_text), 1]


def test_every_rejection_is_counted_by_reason(tmp_path, codec):
    good = sample_records(4)
    rows = good + [
        record("some input text here", "some input", style="formal"),
        record("some other input text", "other input", cleanup="KEEP_CJK_TAIL"),
        record("identical text", "identical text"),
        record("", "empty input"),
        dict(good[0]),  # duplicate input
        record("a <text_to_rewrite> injected frame", "a frame"),
        record("x" * 200, "x" * 150),  # over the source budget (120)
        record("this input is reasonably long", "t"),  # ratio below 0.2
        record("short", "this output is much longer than input"),  # ratio above 1.15
        record("call bob at 10:30 about invoice 4521", "call bob about the invoice"),  # dropped spans
    ]
    _, data = _splits(tmp_path, rows, train_examples=2, eval_examples=2)
    train, evaluation, report = build_splits(data, codec)
    assert report.rejected == {
        "wrong_style": 1,
        "cleanup_excluded": 1,
        "identity": 1,
        "empty": 1,
        "duplicate_input": 1,
        "control_token_in_text": 1,
        "source_too_long": 1,
        "length_ratio_low": 1,
        "length_ratio_high": 1,
        "target_drops_protected_span": 1,
    }
    assert len(train) == len(evaluation) == 2


def test_over_budget_examples_are_dropped_never_truncated(tmp_path, codec):
    rows = sample_records(6) + [record("y" * 130, "y" * 60)]
    _, data = _splits(tmp_path, rows, train_examples=10, eval_examples=1)
    train, evaluation, _ = build_splits(data, codec)
    assert all(len(e.encoder_input_ids) <= data.max_source_tokens for e in train + evaluation)
    assert not any(e.source_text.startswith("y") for e in train + evaluation)


def test_split_ignores_file_order_and_layout(tmp_path, codec):
    rows = sample_records(40)
    _, one_file = _splits(tmp_path / "a", rows, files=1, train_examples=20, eval_examples=8)
    _, three_files = _splits(tmp_path / "b", list(reversed(rows)), files=3, train_examples=20, eval_examples=8)
    first = build_splits(one_file, codec)
    second = build_splits(three_files, codec)
    assert [e.key for e in first[0]] == [e.key for e in second[0]]
    assert [e.key for e in first[1]] == [e.key for e in second[1]]
    assert first[2].fingerprint == second[2].fingerprint


def test_growing_the_train_split_never_touches_the_eval_split(tmp_path, codec):
    rows = sample_records(60)
    _, small = _splits(tmp_path, rows, train_examples=10, eval_examples=8)
    small_train, small_eval, _ = build_splits(small, codec)
    large = make_config(tmp_path, data={"train_examples": 40, "eval_examples": 8}).data
    large_train, large_eval, _ = build_splits(large, codec)
    assert [e.key for e in small_eval] == [e.key for e in large_eval]
    assert [e.key for e in large_train[:10]] == [e.key for e in small_train]
    assert not {e.key for e in large_train} & {e.key for e in large_eval}


def test_separate_eval_data_never_reaches_training(tmp_path, codec):
    """
    With `eval_data`, the eval split is that file, a train record sharing an
    input with it is dropped, and scoring (eval_only) never reads train_data.
    """
    rows = sample_records(30)
    held_out = rows[:5] + [record("this input is reasonably long", "t")]  # 1 fails the ratio filter
    write_dataset(tmp_path / "data", rows)
    eval_file = tmp_path / "evals.json"  # JSONL content whatever the extension
    eval_file.write_text("".join(json.dumps(row) + "\n" for row in held_out), encoding="utf-8")
    data = make_config(
        tmp_path, data={"eval_data": str(eval_file), "train_examples": 100, "eval_examples": 10}
    ).data

    train, evaluation, report = build_splits(data, codec)
    assert sorted(e.source_text for e in evaluation) == sorted(r["input"] for r in rows[:5])
    assert not {e.source_text for e in train} & {r["input"] for r in held_out}
    assert len(train) == 25
    assert report.rejected["in_eval_set"] == 5
    assert report.rejected["eval/length_ratio_low"] == 1

    moved = make_config(tmp_path, data={"eval_data": str(eval_file), "train_data": str(tmp_path / "gone")}).data
    no_train, same_eval, _ = build_splits(moved, codec, eval_only=True)
    assert no_train == [] and [e.key for e in same_eval] == [e.key for e in evaluation]


def test_too_few_examples_is_an_error_not_an_empty_split(tmp_path, codec):
    _, data = _splits(tmp_path, sample_records(3), train_examples=5, eval_examples=3)
    with pytest.raises(DatasetError, match="not enough usable examples"):
        build_splits(data, codec)


@pytest.mark.real_tokenizer
def test_real_tokenizer_frame_ids_and_control_detection():
    if not (REAL_TOKENIZER_DIR / "spm.model").is_file():
        pytest.skip("tokenizer artifacts not built")
    codec = TextCodec.from_tokenizer_dir(REAL_TOKENIZER_DIR, "<style:concise>")
    token_map = json.loads((REAL_TOKENIZER_DIR / "token_map.json").read_text())
    frame = codec.frame
    assert (frame.style_id, frame.open_id, frame.close_id) == (
        token_map["<style:concise>"], token_map["<text_to_rewrite>"], token_map["</text_to_rewrite>"],
    )
    assert (frame.eos_id, frame.pad_id, frame.decoder_start_id) == (1, 0, 0)
    assert codec.contains_control(codec.encode("see <text_to_rewrite> here"))
    assert not codec.contains_control(codec.encode("An ordinary sentence with ls -la and /etc/hosts."))
    text = "Congrats on your first Aviation question! Costs $3.50 at 10:30."
    assert codec.decode(codec.encode(text) + [1, 99]) == text


@pytest.mark.real_data
@pytest.mark.real_tokenizer
def test_real_dataset_sample_frames_cleanly(tmp_path):
    """The real record format, tokenized with the real tokenizer, through the real filters."""
    files = sorted(REAL_DATASET_DIR.glob("*.jsonl")) if REAL_DATASET_DIR.is_dir() else []
    if not files or not (REAL_TOKENIZER_DIR / "spm.model").is_file():
        pytest.skip("rephrase dataset or tokenizer not available")
    sample = tmp_path / "data"
    sample.mkdir()
    with files[0].open(encoding="utf-8") as source, (sample / "sample.jsonl").open("w", encoding="utf-8") as target:
        for _, line in zip(range(400), source):
            target.write(line)
    config = make_config(
        tmp_path,
        data={"train_examples": 200, "eval_examples": 50, "max_source_tokens": 512, "max_target_tokens": 384},
    )
    codec = TextCodec.from_tokenizer_dir(REAL_TOKENIZER_DIR, "<style:concise>")
    train, evaluation, report = build_splits(config.data, codec)
    assert len(evaluation) == 50 and len(train) == 200
    assert report.rejected.get("wrong_style", 0) == 0
    for example in train[:50]:
        assert example.encoder_input_ids[:2] == (codec.frame.style_id, codec.frame.open_id)
        assert example.target_ids[-1] == codec.frame.eos_id
        assert not codec.contains_control(example.target_ids[:-1])
