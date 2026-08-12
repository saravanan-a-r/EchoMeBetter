"""
Unit tests for corpus discovery, eligibility and seeded sampling.

Two properties matter most and are tested hardest:

  * **Nothing is silently lost or altered.** Text that reaches corpus.txt is
    byte-identical to the text in the source record, and anything discarded
    is counted (DESIGN.md risk R1, NF1).
  * **Sampling is exact and reproducible.** Same inputs + same seed produce
    byte-identical output, and the eval set is genuinely disjoint from
    training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus_reader import (
    CorpusError,
    CorpusStats,
    SplitPlan,
    build_corpus,
    compute_blend_weights,
    count_eligible_lines,
    discover_corpus_files,
    iter_eligible_lines,
    iter_eval_lines,
    iter_split,
    sha256_file,
    source_label,
    _allocate,
    _train_budget,
)

UNIFORM = {"mode": "uniform", "shares": {}}


def write_jsonl(path: Path, texts) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for text in texts:
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return path


def eligible(files, max_len=8192):
    stats = CorpusStats()
    return [seg for seg, _ in iter_eligible_lines(files, max_len, stats)], stats


# --- source labelling -----------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("fineweb_edu_2800mb.jsonl", "fineweb_edu"),
        ("stackexchange_1200mb.jsonl", "stackexchange"),
        ("gutenberg_pg19_1600MB.jsonl", "gutenberg_pg19"),
        ("c4_20.5mb.jsonl", "c4"),
        ("corpus.txt", "corpus"),
        ("no_size_suffix.jsonl", "no_size_suffix"),
    ],
)
def test_source_label(filename, expected):
    assert source_label(Path(filename)) == expected


# --- discovery ------------------------------------------------------------


def test_discover_single_file(tmp_path):
    path = write_jsonl(tmp_path / "a.jsonl", ["hello"])
    assert discover_corpus_files(path) == [path]


def test_discover_directory_is_sorted_and_recursive(tmp_path):
    """Order must be deterministic: the two sampling passes rely on it."""
    write_jsonl(tmp_path / "b.jsonl", ["b"])
    write_jsonl(tmp_path / "a.jsonl", ["a"])
    write_jsonl(tmp_path / "nested" / "c.jsonl", ["c"])
    found = discover_corpus_files(tmp_path)
    assert [p.name for p in found] == ["a.jsonl", "b.jsonl", "c.jsonl"]


def test_discover_ignores_unrelated_files(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", ["a"])
    (tmp_path / "a.manifest.json").write_text('{"x":1}', encoding="utf-8")
    (tmp_path / "notes.md").write_text("hi", encoding="utf-8")
    found = discover_corpus_files(tmp_path)
    assert [p.name for p in found] == ["a.jsonl", "a.manifest.json"]


def test_discover_missing_path(tmp_path):
    with pytest.raises(CorpusError, match="does not exist"):
        discover_corpus_files(tmp_path / "nope")


def test_discover_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(CorpusError, match="no corpus files found"):
        discover_corpus_files(tmp_path / "empty")


# --- eligibility and extraction -------------------------------------------


def test_jsonl_text_is_extracted_verbatim(tmp_path):
    texts = ["hello world", "  leading and trailing  ", "tabs\there", "café ½ →"]
    path = write_jsonl(tmp_path / "s.jsonl", texts)
    got, stats = eligible([path])
    assert got == texts
    assert stats.lines_eligible == 4


def test_txt_lines_are_used_as_is(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("line one\nline two\n\nline three\n", encoding="utf-8")
    got, stats = eligible([path])
    assert got == ["line one", "line two", "line three"]
    assert stats.lines_empty == 1


def test_record_with_newlines_is_split_not_joined(tmp_path):
    """
    SentencePiece is line-oriented and would split on newlines anyway; doing
    it here makes it countable. Crucially the segments are not rewritten --
    no space substituted for the newline (DESIGN.md 6.1f).
    """
    path = write_jsonl(tmp_path / "s.jsonl", ["def f():\n    return 1\n\nprint(f())"])
    got, stats = eligible([path])
    assert got == ["def f():", "    return 1", "print(f())"]
    assert stats.records_seen == 1
    assert stats.lines_eligible == 3
    assert stats.lines_empty == 1


def test_indentation_and_internal_whitespace_preserved(tmp_path):
    text = "    indented    with   runs\tand\ttabs    "
    path = write_jsonl(tmp_path / "s.jsonl", [text])
    got, _ = eligible([path])
    assert got == [text]


def test_bad_json_is_counted_not_crashed(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"text": "ok"}\nnot json at all\n{"text": "fine"}\n', encoding="utf-8")
    got, stats = eligible([path])
    assert got == ["ok", "fine"]
    assert stats.records_bad_json == 1


def test_missing_or_wrong_typed_text_field_is_counted(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"body": "wrong field"}\n{"text": ""}\n{"text": 42}\n[1,2,3]\n{"text": "ok"}\n',
        encoding="utf-8",
    )
    got, stats = eligible([path])
    assert got == ["ok"]
    assert stats.records_missing_text == 4


def test_oversized_lines_are_dropped_not_truncated(tmp_path):
    """
    Risk R1: SentencePiece silently discards over-long lines. We drop them
    deliberately and count them -- and never truncate, which would corrupt.
    """
    path = write_jsonl(tmp_path / "s.jsonl", ["short", "x" * 200])
    got, stats = eligible([path], max_len=100)
    assert got == ["short"]
    assert stats.lines_oversized == 1


def test_length_limit_is_measured_in_utf8_bytes(tmp_path):
    """
    A 60-character line of 3-byte characters is 180 bytes. Measuring
    characters would let it past our check and straight into the silent
    drop inside SentencePiece.
    """
    text = "日" * 60
    assert len(text) == 60 and len(text.encode("utf-8")) == 180
    path = write_jsonl(tmp_path / "s.jsonl", [text])

    got, stats = eligible([path], max_len=100)
    assert got == [] and stats.lines_oversized == 1

    got, stats = eligible([path], max_len=200)
    assert got == [text] and stats.lines_oversized == 0


def test_invalid_utf8_fails_loudly(tmp_path):
    """
    Decoding with errors="replace" would substitute U+FFFD and silently
    corrupt training text. The run must stop instead.
    """
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"text": "ok"}\n{"text": "\xff\xfe bad bytes"}\n')
    with pytest.raises(CorpusError, match="not valid UTF-8"):
        eligible([path])


def test_per_source_counts(tmp_path):
    write_jsonl(tmp_path / "fineweb_edu_10mb.jsonl", ["a", "b", "c"])
    write_jsonl(tmp_path / "stackexchange_5mb.jsonl", ["d"])
    stats = count_eligible_lines(discover_corpus_files(tmp_path), 8192)
    assert stats.per_source_lines == {"fineweb_edu": 3, "stackexchange": 1}


# --- drop-fraction accounting ---------------------------------------------


def test_blank_lines_do_not_count_as_loss(tmp_path):
    """Structural blank lines are not data loss and must not trip the guard."""
    path = tmp_path / "p.txt"
    path.write_text("a\n\n\n\nb\n", encoding="utf-8")
    stats = count_eligible_lines([path], 8192)
    assert stats.lines_empty == 3
    assert stats.drop_fraction() == 0.0


def test_drop_fraction_counts_real_loss(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"text": "ok"}\nnot json\n{"text": "ok2"}\n', encoding="utf-8")
    stats = count_eligible_lines([path], 8192)
    assert stats.drop_fraction() == pytest.approx(1 / 3)


# --- budget splitting -----------------------------------------------------


@pytest.mark.parametrize(
    "total,requested,expected",
    [
        (1_000_000, 100, 100),  # plenty of corpus: budget honoured, rest is eval
        (100, 1000, 90),  # smaller than budget: 10% floor held back for eval
        (10, 1000, 9),  # tiny corpus: still reserves one eval line
        (1, 1000, 0),  # degenerate: the single line becomes eval, not training
        (0, 1000, 0),
        (100, 100, 90),  # exactly at the boundary: floor still applies
        (101, 100, 100),  # one line spare: budget honoured
    ],
)
def test_train_budget(total, requested, expected):
    assert _train_budget(total, requested) == expected


# --- sampling -------------------------------------------------------------


def build(tmp_path, files, *, train=50, seed=42, out="run"):
    out_dir = tmp_path / out
    return build_corpus(
        files,
        corpus_out=out_dir / "corpus.txt",
        train_budget=train,
        seed=seed,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
    )


def plan_of(info):
    return SplitPlan.from_dict(info["plan"])


@pytest.fixture
def corpus_1000(tmp_path):
    return [write_jsonl(tmp_path / "src" / "fineweb_edu_1mb.jsonl",
                        [f"record number {i} of the corpus" for i in range(1000)])]


def test_sampling_produces_exact_counts(tmp_path, corpus_1000):
    info = build(tmp_path, corpus_1000, train=50)
    assert info["train_lines"] == 50
    assert info["eval_lines_available"] == 950
    assert len(Path(info["corpus_path"]).read_text(encoding="utf-8").splitlines()) == 50


def test_eval_is_everything_not_trained_on(tmp_path, corpus_1000):
    """
    The core invariant of the split: train and eval partition the corpus,
    by position. Nothing is lost, nothing is counted twice.
    """
    info = build(tmp_path, corpus_1000, train=200)
    plan = plan_of(info)
    train_idx, eval_idx = set(), set()
    for i, (_text, _label, is_train) in enumerate(iter_split(corpus_1000, plan)):
        (train_idx if is_train else eval_idx).add(i)

    assert train_idx & eval_idx == set()
    assert len(train_idx | eval_idx) == plan.n_total == 1000
    assert len(train_idx) == plan.n_train == 200
    assert len(eval_idx) == plan.n_eval == 800


def test_eval_replay_reproduces_the_written_corpus(tmp_path, corpus_1000):
    """
    The eval stream is replayed rather than stored, so replaying the split
    must reproduce corpus.txt *exactly*. If this drifts, the gates would be
    measured against a set that overlaps training without anything failing.
    """
    info = build(tmp_path, corpus_1000, train=200)
    replayed = [t for t, _l, is_train in iter_split(corpus_1000, plan_of(info)) if is_train]
    on_disk = Path(info["corpus_path"]).read_text(encoding="utf-8").splitlines()
    assert replayed == on_disk


def test_eval_lines_never_appear_in_training(tmp_path):
    """Positional disjointness, verified on a corpus with no duplicate text."""
    files = [write_jsonl(tmp_path / "src" / "fineweb_edu_1mb.jsonl",
                         [f"unique line {i}" for i in range(500)])]
    info = build(tmp_path, files, train=100)
    train_lines = set(Path(info["corpus_path"]).read_text(encoding="utf-8").splitlines())
    eval_lines = {t for t, _ in iter_eval_lines(files, plan_of(info))}
    assert train_lines & eval_lines == set()
    assert len(eval_lines) == 400


def test_sampling_is_deterministic(tmp_path, corpus_1000):
    a = build(tmp_path, corpus_1000, out="a")
    b = build(tmp_path, corpus_1000, out="b")
    assert a["corpus_sha256"] == b["corpus_sha256"]
    assert list(iter_eval_lines(corpus_1000, plan_of(a))) == list(
        iter_eval_lines(corpus_1000, plan_of(b))
    )


def test_different_seed_gives_different_sample(tmp_path, corpus_1000):
    a = build(tmp_path, corpus_1000, seed=42, out="a")
    b = build(tmp_path, corpus_1000, seed=43, out="b")
    assert a["corpus_sha256"] != b["corpus_sha256"]


def test_sample_is_spread_across_the_corpus(tmp_path, corpus_1000):
    """
    A sampler that just took the first N lines would pass the count tests
    while training only on the head of the corpus -- and on a blended corpus
    that means training on one source. Check the draw actually spans the file.
    """
    info = build(tmp_path, corpus_1000, train=100)
    indices = [
        int(line.split()[2])
        for line in Path(info["corpus_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert min(indices) < 100 and max(indices) > 900
    assert indices == sorted(indices)  # streaming order is preserved


def test_sample_spans_every_source_file(tmp_path):
    files = [
        write_jsonl(tmp_path / "src" / "fineweb_edu_1mb.jsonl", [f"a{i}" for i in range(500)]),
        write_jsonl(tmp_path / "src" / "stackexchange_1mb.jsonl", [f"b{i}" for i in range(500)]),
    ]
    info = build(tmp_path, files, train=200)
    lines = Path(info["corpus_path"]).read_text(encoding="utf-8").splitlines()
    assert any(l.startswith("a") for l in lines)
    assert any(l.startswith("b") for l in lines)


def test_whole_corpus_used_when_smaller_than_budget(tmp_path, corpus_1000):
    """
    A corpus smaller than the budget still holds 10% back: a run that trained
    on literally every line could not be validated at all.
    """
    info = build(tmp_path, corpus_1000, train=100_000)
    assert info["train_lines"] + info["eval_lines_available"] == 1000
    assert info["eval_lines_available"] == 100  # 10% floor


def test_text_reaches_corpus_byte_identical(tmp_path):
    """End-to-end fidelity check on the constructs NF1 is about."""
    texts = [
        "See https://example.com/a?b=1&c=2#frag for details.",
        "[docs](https://docs.python.org/3/library/json.html) explains it.",
        "    indented   code\twith\ttabs",
        "unicode café ½ → 日本語 “quoted”",
        "numbers 1234567890 and 3.14159",
    ]
    files = [write_jsonl(tmp_path / "src" / "s.jsonl", texts)]
    info = build(tmp_path, files, train=len(texts))

    # Train and eval together must reconstruct the input exactly, and every
    # line on either side must be byte-identical to its source record.
    train_lines = Path(info["corpus_path"]).read_text(encoding="utf-8").splitlines()
    eval_lines = [t for t, _ in iter_eval_lines(files, plan_of(info))]
    assert sorted(train_lines + eval_lines) == sorted(texts)
    assert set(train_lines) <= set(texts)


def test_corpus_error_when_nothing_eligible(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"nope": 1}\n', encoding="utf-8")
    with pytest.raises(CorpusError, match="no eligible lines"):
        build(tmp_path, [path])


def test_drop_fraction_guard_aborts_the_run(tmp_path):
    """Refuse to train on a degraded sample rather than do it silently."""
    lines = ['{"text": "ok"}'] * 90 + ["garbage"] * 10
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="drop fraction"):
        build(tmp_path, [path], train=10)


def test_drop_fraction_guard_respects_threshold(tmp_path):
    lines = ['{"text": "ok"}'] * 999 + ["garbage"]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info = build_corpus(
        [path],
        corpus_out=tmp_path / "out" / "corpus.txt",
        train_budget=10,
        seed=42,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
    )
    assert info["train_lines"] == 10


def test_eval_lines_carry_source_labels(tmp_path):
    """Gates 2/4/5/8 report per slice, so every eval line needs its source."""
    files = [
        write_jsonl(tmp_path / "src" / "fineweb_edu_1mb.jsonl", [f"a{i}" for i in range(300)]),
        write_jsonl(tmp_path / "src" / "stackexchange_1mb.jsonl", [f"b{i}" for i in range(300)]),
    ]
    info = build(tmp_path, files, train=100)
    sources = {label for _t, label in iter_eval_lines(files, plan_of(info))}
    assert sources == {"fineweb_edu", "stackexchange"}
    assert set(info["eval_lines_per_source"]) == {"fineweb_edu", "stackexchange"}


def test_manifest_fields(tmp_path, corpus_1000):
    info = build(tmp_path, corpus_1000, train=50)
    assert info["eligible_lines"] == 1000
    assert info["sampled_fraction"] == pytest.approx(0.05)  # 50 train / 1000
    assert info["seed"] == 42
    assert info["corpus_bytes"] > 0
    assert len(info["corpus_sha256"]) == 64
    assert info["scan"]["per_source_lines"] == {"fineweb_edu": 1000}


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    payload = b"some bytes" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


# --- budget allocation across sources -------------------------------------


def test_allocate_is_proportional():
    alloc = _allocate(100, {"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 999, "b": 999, "c": 999})
    assert alloc == {"a": 50, "b": 30, "c": 20}


def test_allocate_is_exact_despite_rounding():
    """Largest-remainder: the parts must sum to the whole, always."""
    alloc = _allocate(100, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, {"a": 999, "b": 999, "c": 999})
    assert sum(alloc.values()) == 100


def test_allocate_respects_capacity_and_redistributes():
    """A source that runs out gives its remaining share to the others."""
    alloc = _allocate(100, {"a": 0.5, "b": 0.5}, {"a": 10, "b": 999})
    assert alloc == {"a": 10, "b": 90}


def test_allocate_caps_at_total_capacity():
    alloc = _allocate(1000, {"a": 0.5, "b": 0.5}, {"a": 10, "b": 20})
    assert alloc == {"a": 10, "b": 20}


def test_allocate_ignores_zero_weight_sources():
    alloc = _allocate(100, {"a": 1.0, "b": 0.0}, {"a": 999, "b": 999})
    assert alloc == {"a": 100, "b": 0}


def test_allocate_handles_budget_smaller_than_source_count():
    alloc = _allocate(2, {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 9, "b": 9, "c": 9})
    assert sum(alloc.values()) == 2


def test_allocate_is_deterministic():
    weights = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    caps = {"a": 999, "b": 999, "c": 999}
    assert _allocate(100, weights, caps) == _allocate(100, weights, caps)


def test_allocate_zero_budget():
    assert _allocate(0, {"a": 1.0}, {"a": 10}) == {"a": 0}


# --- blend weighting ------------------------------------------------------


def test_uniform_mode_weights_by_available_lines():
    weights, notes = compute_blend_weights(UNIFORM, {"a": 100, "b": 300}, {"a": 1000, "b": 3000})
    assert weights == {"a": 100.0, "b": 300.0}
    assert notes == []


def test_weighted_mode_compensates_for_line_length():
    """
    The bug this exists to prevent: source 'short' has 10-byte lines and
    'long' has 100-byte lines. For equal *byte* shares, 10x more short lines
    must be drawn. Weighting by share alone would give equal line counts and
    therefore a 1:10 byte split.
    """
    blend = {"mode": "weighted", "shares": {"short": 0.5, "long": 0.5}}
    weights, _ = compute_blend_weights(
        blend, {"short": 10_000, "long": 10_000}, {"short": 100_000, "long": 1_000_000}
    )
    assert weights["short"] == pytest.approx(weights["long"] * 10)


def test_weighted_mode_renormalizes_over_present_sources(caplog):
    """A partial corpus must still produce correctly proportioned output."""
    blend = {"mode": "weighted", "shares": {"a": 0.5, "b": 0.25, "c": 0.25}}
    weights, notes = compute_blend_weights(blend, {"a": 100, "b": 100}, {"a": 1000, "b": 1000})
    # a:b was 0.5:0.25, so after renormalizing over {a, b} it is 2:1.
    assert weights["a"] == pytest.approx(weights["b"] * 2)
    assert any("absent from corpus" in note for note in notes)


def test_weighted_mode_excludes_unconfigured_sources():
    blend = {"mode": "weighted", "shares": {"a": 1.0}}
    weights, notes = compute_blend_weights(blend, {"a": 100, "mystery": 100}, {"a": 1000, "mystery": 1000})
    assert weights["mystery"] == 0.0
    assert any("no blend share configured" in note for note in notes)


def test_weighted_mode_fails_when_no_configured_source_present():
    blend = {"mode": "weighted", "shares": {"a": 1.0}}
    with pytest.raises(CorpusError, match="none of the configured sources"):
        compute_blend_weights(blend, {"other": 100}, {"other": 1000})


def test_blend_achieves_target_byte_shares(tmp_path):
    """
    End-to-end regression test for the blend bug found on the real corpus:
    PG-19 lines average ~69 bytes against C4's ~447, so uniform line
    sampling produced 55% Gutenberg instead of the specified 20%.

    Here 'shortlines' has 20-byte lines and 'longlines' 200-byte lines, and
    the blend asks for 50/50 by bytes.
    """
    src = tmp_path / "src"
    write_jsonl(src / "shortlines_1mb.jsonl", ["x" * 20 for _ in range(20_000)])
    write_jsonl(src / "longlines_1mb.jsonl", ["y" * 200 for _ in range(20_000)])
    files = discover_corpus_files(src)

    info = build_corpus(
        files,
        corpus_out=tmp_path / "out" / "corpus.txt",
        train_budget=10_000,
        seed=42,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
        blend={"mode": "weighted", "shares": {"shortlines": 0.5, "longlines": 0.5}},
    )

    achieved = info["blend_achieved_bytes"]
    assert achieved["shortlines"] == pytest.approx(0.5, abs=0.02)
    assert achieved["longlines"] == pytest.approx(0.5, abs=0.02)
    # 10x more short lines are needed to reach the same byte share.
    per_source = info["train_lines_per_source"]
    assert per_source["shortlines"] == pytest.approx(per_source["longlines"] * 10, rel=0.05)


def test_uniform_mode_would_skew_the_same_corpus(tmp_path):
    """
    The counter-case proving the weighted path is doing real work: with
    uniform sampling the same corpus lands at roughly 9%/91% by bytes.
    """
    src = tmp_path / "src"
    write_jsonl(src / "shortlines_1mb.jsonl", ["x" * 20 for _ in range(20_000)])
    write_jsonl(src / "longlines_1mb.jsonl", ["y" * 200 for _ in range(20_000)])

    info = build_corpus(
        discover_corpus_files(src),
        corpus_out=tmp_path / "out" / "corpus.txt",
        train_budget=10_000,
        seed=42,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
        blend=UNIFORM,
    )
    assert info["blend_achieved_bytes"]["shortlines"] == pytest.approx(0.09, abs=0.02)


def test_blend_respects_target_shares_unequally(tmp_path):
    src = tmp_path / "src"
    for name, count in (("a", 30_000), ("b", 30_000), ("c", 30_000)):
        write_jsonl(src / f"{name}_1mb.jsonl", [f"{name}" * 30 for _ in range(count)])

    info = build_corpus(
        discover_corpus_files(src),
        corpus_out=tmp_path / "out" / "corpus.txt",
        train_budget=9_000,
        seed=42,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
        blend={"mode": "weighted", "shares": {"a": 0.6, "b": 0.3, "c": 0.1}},
    )
    achieved = info["blend_achieved_bytes"]
    assert achieved["a"] == pytest.approx(0.6, abs=0.02)
    assert achieved["b"] == pytest.approx(0.3, abs=0.02)
    assert achieved["c"] == pytest.approx(0.1, abs=0.02)


def test_eval_is_drawn_from_every_slice(tmp_path):
    """Gate 4 reports fertility per slice, so every slice must be present."""
    src = tmp_path / "src"
    for name in ("a", "b", "c"):
        write_jsonl(src / f"{name}_1mb.jsonl", [f"{name} line {i}" for i in range(5_000)])

    info = build_corpus(
        discover_corpus_files(src),
        corpus_out=tmp_path / "out" / "corpus.txt",
        train_budget=3_000,
        seed=42,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
        blend={"mode": "weighted", "shares": {"a": 0.34, "b": 0.33, "c": 0.33}},
    )
    assert set(info["eval_lines_per_source"]) == {"a", "b", "c"}
    assert sum(info["eval_lines_per_source"].values()) == info["eval_lines_available"] == 12_000
    # every slice really does show up in the replayed stream, not just the plan
    assert {label for _t, label in iter_eval_lines(
        discover_corpus_files(src), plan_of(info))} == {"a", "b", "c"}


def test_blend_metadata_recorded_for_manifest(tmp_path, corpus_1000):
    info = build(tmp_path, corpus_1000, train=50)
    assert info["blend_mode"] == "uniform"
    assert "blend_achieved_bytes" in info
    assert "train_bytes_per_source" in info
