"""
Integration tests: the real pipeline, a real SentencePiece model, real artifacts.

These are the regression net around the properties that cannot be recovered
once pretraining starts: byte-exact round-trip, atomic reserved tokens, and a
token map whose IDs never move.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import train_tokenization
from specials import ALL_NAMED_SPECIALS, USER_DEFINED_SYMBOLS
from tests.corpus_factory import write_synthetic_corpus


# --- artifacts ------------------------------------------------------------


def test_all_expected_artifacts_exist(output_dir):
    for name in (
        "corpus.txt", "tokenizer_evals_set.jsonl", "spm.model", "spm.vocab",
        "token_map.json", "manifest.json", "report.md",
    ):
        assert (output_dir / name).is_file(), f"missing artifact {name}"


def test_vocab_size_matches_config(sp, trained_run):
    assert sp.get_piece_size() == trained_run["cfg"]["trainer"]["vocab_size"]


def test_manifest_is_complete_and_machine_readable(output_dir):
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    for key in ("timestamp", "corpus", "training", "evaluation", "config", "gates",
                "environment"):
        assert key in manifest
    assert manifest["corpus"]["corpus_sha256"]
    assert manifest["training"]["model_sha256"]
    assert manifest["training"]["sentencepiece_version"]
    assert len(manifest["gates"]) == 11


def test_manifest_does_not_embed_the_vocabulary_dump(output_dir):
    """Gate 7's 1000 pieces belong in report.md, not the machine-readable manifest."""
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    gate_7 = next(g for g in manifest["gates"] if g["gate"] == 7)
    assert set(gate_7["metrics"]) == {"vocab_size"}


def test_manifest_records_the_achieved_blend(output_dir):
    """
    Provenance must show what the vocabulary was actually fitted to, not just
    what was requested -- that difference is exactly where risk R4 hides.
    """
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    corpus = manifest["corpus"]
    assert corpus["blend_mode"] == "weighted"
    assert corpus["blend_target"]
    achieved = corpus["blend_achieved_bytes"]
    # The synthetic corpus carries three of the five configured sources, so
    # shares are renormalized over those three and must still sum to 1.
    assert set(achieved) == {"fineweb_edu", "stackexchange", "gutenberg_pg19"}
    assert sum(achieved.values()) == pytest.approx(1.0, abs=0.01)
    assert corpus["train_bytes_per_source"]
    assert corpus["train_lines_per_source"]


def test_unapplied_blend_is_reported_not_silent(output_dir):
    """
    The smoke budget (51k lines) exceeds the synthetic corpus (~18k), so every
    line is taken and the configured shares cannot be honoured. The result is
    legitimate, but it must be stated: a vocabulary fitted to "whatever was on
    disk" while the config claims a specified blend is exactly the kind of
    silent divergence risk R4 describes.
    """
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    corpus = manifest["corpus"]
    assert corpus["train_lines"] + corpus["eval_lines_available"] == corpus["eligible_lines"]
    assert any("BLEND NOT APPLIED" in note for note in corpus["blend_notes"])


def test_blend_is_applied_when_the_corpus_exceeds_the_budget(
    tmp_path, synthetic_corpus, smoke_config_path
):
    """
    With a budget well below the corpus size the sampler can honour the
    shares: 0.35:0.15:0.20 renormalized over the three present sources is
    0.50:0.214:0.286 by bytes.
    """
    from corpus_reader import build_corpus, discover_corpus_files

    info = build_corpus(
        discover_corpus_files(synthetic_corpus),
        corpus_out=tmp_path / "b" / "corpus.txt",
        train_budget=4000,
        seed=42,
        max_sentence_length=8192,
        max_drop_fraction=0.01,
        blend={
            "mode": "weighted",
            "shares": {"fineweb_edu": 0.35, "stackexchange": 0.15, "gutenberg_pg19": 0.20},
        },
    )
    achieved = info["blend_achieved_bytes"]
    assert achieved["fineweb_edu"] == pytest.approx(0.35 / 0.70, abs=0.02)
    assert achieved["stackexchange"] == pytest.approx(0.15 / 0.70, abs=0.02)
    assert achieved["gutenberg_pg19"] == pytest.approx(0.20 / 0.70, abs=0.02)
    assert not any("BLEND NOT APPLIED" in note for note in info["blend_notes"])


def test_report_lists_every_gate(output_dir):
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    for number in range(1, 12):
        assert f"| {number} |" in report
    assert "## Per source" in report
    assert "## Vocabulary usage" in report


# --- the token-ID contract (DESIGN.md 5.3) --------------------------------


def test_token_map_covers_every_reserved_name(output_dir):
    token_map = json.loads((output_dir / "token_map.json").read_text(encoding="utf-8"))
    assert set(token_map) == set(ALL_NAMED_SPECIALS)
    assert len(token_map) == 1272  # 376 named + 896 CLI vocabulary pieces


def test_token_map_ids_are_unique_and_resolve(output_dir, sp):
    token_map = json.loads((output_dir / "token_map.json").read_text(encoding="utf-8"))
    assert len(set(token_map.values())) == len(token_map)
    for name, token_id in token_map.items():
        assert sp.id_to_piece(token_id) == name


def test_control_ids_match_configured_values(output_dir, trained_run):
    token_map = json.loads((output_dir / "token_map.json").read_text(encoding="utf-8"))
    t = trained_run["cfg"]["trainer"]
    assert token_map["<pad>"] == t["pad_id"]
    assert token_map["</s>"] == t["eos_id"]
    assert token_map["<unk>"] == t["unk_id"]


# --- fidelity: the properties NF1 is about --------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Check https://example.com/path?a=1&b=2#frag now.",
        "See [the docs](https://docs.python.org/3/library/json.html) first.",
        "    def f(x):\n        return x ** 2",
        "   leading and trailing spaces   ",
        "collapse    these     spaces?",
        "col1\tcol2\tcol3",
        "Revenue rose 1234567890 to 3.14159 (up 42%).",
        "first.last+tag@sub.example.co.uk",
        "café naïve résumé Ångström",
        "→ ± ½ µs — “smart quotes” … ‰",
        "日本語のテキストです",
        "ship it 🚀🔥 now",
        '{"key": [1, 2, {"nested": null}]}',
        "const re = /^[a-z0-9_-]{3,16}$/gi;",
        "/Users/apple/projects/file_name-v2.tar.gz",
        "a &lt; b &amp;&amp; c &gt; d",
        " ",
        "x",
    ],
)
def test_byte_exact_roundtrip(sp, text):
    """decode(encode(x)) == x. The keystone guarantee (DESIGN.md Gate 1)."""
    assert sp.decode(sp.encode(text)) == text


def test_no_unk_on_arbitrary_bytes(sp):
    """
    byte_fallback means no input is unrepresentable. Text drawn from scripts
    entirely absent from the training corpus must still encode without <unk>.
    """
    exotic = "ᚠᚢᚦᚨᚱᚲ ⡌⠁⠧⠑ 🜁🜂🜃🜄 ﷽"
    ids = sp.encode(exotic)
    assert sp.unk_id() not in ids
    assert sp.decode(ids) == exotic


def test_leading_whitespace_is_not_invented(sp):
    """
    add_dummy_prefix=False is what makes this hold. With the default True,
    "hello" and " hello" can decode identically -- silent corruption of
    leading whitespace, invisible to any downstream validator.
    """
    assert sp.decode(sp.encode("hello")) == "hello"
    assert sp.decode(sp.encode(" hello")) == " hello"
    assert sp.encode("hello") != sp.encode(" hello")


def test_digits_are_split_individually(sp):
    """split_digits=True: number-copy fidelity depends on consistent structure."""
    pieces = sp.encode("2024", out_type=str)
    assert [p.lstrip("▁") for p in pieces if p.strip("▁")] == ["2", "0", "2", "4"]


def test_identity_normalization_does_not_fold_unicode(sp):
    """
    NFKC would rewrite these characters (full-width to half-width, ligature
    decomposition). identity must leave them exactly as written -- this is
    the single most important config line (DESIGN.md 6.1a).
    """
    for text in ("ﬁ ﬂ ﬃ", "Ｆｕｌｌｗｉｄｔｈ", "½ ¼ ¾", "① ② ③", "ⅷ", "™ ℡"):
        assert sp.decode(sp.encode(text)) == text


def test_reserved_tokens_are_atomic(sp):
    """All 1,269 user-defined symbols must encode to exactly one piece (Gate 3)."""
    non_atomic = [name for name in USER_DEFINED_SYMBOLS if len(sp.encode(name)) != 1]
    assert non_atomic == []


def test_reserved_tokens_survive_inside_surrounding_text(sp):
    text = "<style:concise> <text_to_rewrite>Some prose here.</text_to_rewrite> <extra_id_7>"
    ids = sp.encode(text)
    assert sp.decode(ids) == text
    assert sp.piece_to_id("<style:concise>") in ids
    assert sp.piece_to_id("<extra_id_7>") in ids


def test_newline_is_representable_as_a_byte_piece(sp):
    """
    SentencePiece never learns a piece containing a newline (it reads line by
    line), so \\n exists only via byte fallback. The original T5 tokenizer
    could not represent it at all -- a known source of code/markdown
    breakage this design deliberately avoids (DESIGN.md 6.1f).
    """
    assert sp.piece_to_id("<0x0A>") != sp.unk_id()
    assert sp.decode(sp.encode("line one\nline two")) == "line one\nline two"


# --- determinism (NF5) ----------------------------------------------------


def test_rebuild_is_bit_identical(tmp_path, synthetic_corpus, smoke_config_path):
    """
    Same corpus + same seed + same config must reproduce the build exactly
    (NF5). Both runs target the same output directory deliberately: the
    serialized .model embeds its own model_prefix path, so rebuilding into a
    *different* directory changes the bytes for a reason that has nothing to
    do with the vocabulary (see the test below).
    """
    out = tmp_path / "repro"
    fingerprints = []
    for _ in range(2):
        assert train_tokenization.main(
            ["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
             "--output-dir", str(out)]
        ) == 0
        fingerprints.append(
            {name: (out / name).read_bytes()
             for name in ("corpus.txt", "tokenizer_evals_set.jsonl", "spm.model", "spm.vocab", "token_map.json")}
        )
    assert fingerprints[0] == fingerprints[1]


def test_vocabulary_is_independent_of_output_path(tmp_path, synthetic_corpus, smoke_config_path,
                                                  output_dir):
    """
    Building into a different directory changes the .model bytes (the path is
    embedded in the protobuf) but must not change a single piece, ID or
    segmentation. Recorded as a test so the byte difference is never
    mistaken for non-determinism in the tokenizer itself.
    """
    elsewhere = tmp_path / "elsewhere"
    assert train_tokenization.main(
        ["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(elsewhere)]
    ) == 0

    assert (elsewhere / "corpus.txt").read_bytes() == (output_dir / "corpus.txt").read_bytes()
    assert (elsewhere / "spm.vocab").read_bytes() == (output_dir / "spm.vocab").read_bytes()
    assert (elsewhere / "token_map.json").read_text() == (output_dir / "token_map.json").read_text()

    import trainer as trainer_module

    other = trainer_module.load_processor(elsewhere / "spm.model")
    reference = trainer_module.load_processor(output_dir / "spm.model")
    assert other.get_piece_size() == reference.get_piece_size()
    for text in ("Hello world https://x.com/a?b=1", "    def f():", "café ½ → 日本語",
                 "numbers 1234 3.14", "<style:concise> text"):
        assert other.encode(text) == reference.encode(text)


def test_different_seed_changes_the_sample(tmp_path, synthetic_corpus, smoke_config_path, output_dir):
    rerun_dir = tmp_path / "seeded"
    assert train_tokenization.main(
        ["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(rerun_dir), "--seed", "1234"]
    ) == 0
    assert (rerun_dir / "corpus.txt").read_bytes() != (output_dir / "corpus.txt").read_bytes()


# --- CLI behaviour --------------------------------------------------------


def test_dry_run_stops_before_training(tmp_path, synthetic_corpus, smoke_config_path):
    out = tmp_path / "dry"
    assert train_tokenization.main(
        ["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(out), "--dry-run"]
    ) == 0
    assert (out / "corpus.txt").is_file()
    assert not (out / "spm.model").exists()


def test_input_sentence_size_override_is_applied(tmp_path, synthetic_corpus, smoke_config_path):
    out = tmp_path / "override"
    assert train_tokenization.main(
        ["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(out), "--dry-run", "--input-sentence-size", "500"]
    ) == 0
    assert len((out / "corpus.txt").read_text(encoding="utf-8").splitlines()) == 500


def test_missing_corpus_exits_2(tmp_path, smoke_config_path):
    assert train_tokenization.main(
        ["--corpus", str(tmp_path / "nope"), "--config", str(smoke_config_path)]
    ) == 2


def test_bad_config_exits_2(tmp_path, synthetic_corpus):
    bad = tmp_path / "bad.yaml"
    bad.write_text("corpus: {}\n", encoding="utf-8")
    assert train_tokenization.main(
        ["--corpus", str(synthetic_corpus), "--config", str(bad)]
    ) == 2


def test_vocab_too_large_for_corpus_exits_3(tmp_path, smoke_config_path):
    """
    A corpus too small for the requested vocab must fail loudly with a
    non-zero exit, not produce a truncated vocabulary.
    """
    import yaml

    tiny_corpus = tmp_path / "tiny"
    write_synthetic_corpus(tiny_corpus, n_records=60, seed=3, n_files=1)

    cfg = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    cfg["trainer"]["vocab_size"] = 20000
    cfg_path = tmp_path / "big_vocab.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert train_tokenization.main(
        ["--corpus", str(tiny_corpus), "--config", str(cfg_path),
         "--output-dir", str(tmp_path / "out")]
    ) == 3


def test_entry_point_runs_as_a_subprocess(module_dir, synthetic_corpus, smoke_config_path, tmp_path):
    """The documented invocation must work as a plain script, not just as an import."""
    result = subprocess.run(
        [sys.executable, str(module_dir / "train_tokenization.py"),
         "--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(tmp_path / "cli"), "--dry-run"],
        capture_output=True, text=True, cwd=module_dir,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli" / "corpus.txt").is_file()


# --- eval scope (--eval-corpus) -------------------------------------------


def test_eval_defaults_to_all_remaining_corpus(output_dir):
    """
    The default is the honest one: every line not trained on is evaluated,
    so the gates measure the vocabulary against everything it will meet.
    """
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation = manifest["evaluation"]
    assert evaluation["ran"] is True
    assert evaluation["eval_sentence_size"] == 0
    assert evaluation["scope"] == "all remaining corpus"
    assert evaluation["lines_evaluated"] == manifest["corpus"]["eval_lines_available"]


def test_eval_corpus_flag_caps_the_evaluated_lines(
    tmp_path, synthetic_corpus, smoke_config_path
):
    out = tmp_path / "capped"
    assert train_tokenization.main(
        [
            "--corpus", str(synthetic_corpus),
            "--config", str(smoke_config_path),
            "--output-dir", str(out),
            "--eval-corpus", "300",
        ]
    ) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    evaluation = manifest["evaluation"]
    assert evaluation["lines_evaluated"] == 300
    assert evaluation["lines_available"] > 300  # genuinely a subsample
    gate_1 = next(g for g in manifest["gates"] if g["gate"] == 1)
    assert gate_1["metrics"]["records"] == 300
    assert gate_1["passed"]


def test_eval_corpus_larger_than_available_is_clamped(
    tmp_path, synthetic_corpus, smoke_config_path
):
    """Asking for more eval lines than exist evaluates all of them, not an error."""
    out = tmp_path / "clamped"
    assert train_tokenization.main(
        [
            "--corpus", str(synthetic_corpus),
            "--config", str(smoke_config_path),
            "--output-dir", str(out),
            "--eval-corpus", "10_000_000",
        ]
    ) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["evaluation"]["lines_evaluated"]
        == manifest["corpus"]["eval_lines_available"]
    )


def test_saved_eval_sample_is_bounded_and_labelled(output_dir, trained_run):
    """
    The full eval set is never written (it is ~18 GB on the real corpus);
    a bounded, representative sample is, for export_hf.py and inspection.
    """
    records = [
        json.loads(line)
        for line in (output_dir / "tokenizer_evals_set.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    budget = trained_run["cfg"]["validation"]["eval_sample_save"]
    assert 0 < len(records) <= budget
    assert all(set(r) == {"text", "source"} for r in records)


def test_saved_eval_sample_comes_from_the_eval_stream(output_dir, trained_run):
    """
    The saved sample must be drawn from the eval side of the split, never
    from training. Checked against the replayed eval stream rather than by
    comparing strings, because a corpus may legitimately contain the same
    text at two different positions -- the split is disjoint by position.
    """
    from corpus_reader import SplitPlan, discover_corpus_files, iter_eval_lines

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = SplitPlan.from_dict(manifest["corpus"]["plan"])
    files = discover_corpus_files(trained_run["corpus_dir"])
    eval_pool = set(iter_eval_lines(files, plan))

    records = [
        json.loads(line)
        for line in (output_dir / "tokenizer_evals_set.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert records
    for record in records:
        assert (record["text"], record["source"]) in eval_pool
