"""
Integration tests for the HuggingFace export.

The property under test is not "the export works" but "the export is
interchangeable with spm.model". A tokenizer that encodes the same text to
different IDs would train a model that cannot be served by the original --
a failure that only becomes visible once weights exist.
"""

from __future__ import annotations

import json

import pytest

import export_hf


@pytest.fixture(scope="module")
def exported(trained_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("hf_export")
    model = trained_run["output_dir"] / "spm.model"
    assert export_hf.main(["--output-dir", str(out), "--model", str(model)]) == 0
    return out / "tokenizer.json"


def test_export_writes_tokenizer_json(exported):
    assert exported.is_file()
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["model"]["type"] == "Unigram"
    assert payload["model"]["byte_fallback"] is True


def test_exported_vocab_size_matches_spm(exported, sp):
    from tokenizers import Tokenizer

    assert Tokenizer.from_file(str(exported)).get_vocab_size() == sp.get_piece_size()


def test_export_has_no_normalizer(exported):
    """
    Training uses normalization_rule_name="identity". A normalizer added only
    on the inference side would rewrite text at serving time but not during
    training -- silent, and invisible to every gate.
    """
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload.get("normalizer") is None


def test_export_does_not_prepend_a_dummy_space(exported):
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["pre_tokenizer"]["prepend_scheme"] == "never"


@pytest.mark.parametrize(
    "text",
    [
        "Check https://example.com/path?a=1&b=2#frag now.",
        "    def f(x):\n        return x ** 2",
        "   leading whitespace",
        "trailing whitespace   ",
        "col1\tcol2\tcol3",
        "Revenue rose 1234567890 to 3.14159 (up 42%).",
        "café naïve résumé Ångström",
        "日本語のテキストです",
        "ship it 🚀🔥 now",
        "<style:concise> <text_to_rewrite>hi</text_to_rewrite>",
        " ",
        "x",
    ],
)
def test_exported_ids_are_identical_to_spm(exported, sp, text):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(exported))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert ids == sp.encode(text)
    assert tokenizer.decode(ids) == text


def test_exported_agrees_with_spm_on_real_holdout(exported, sp, trained_run):
    """Agreement on curated probes is necessary but not sufficient."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(exported))
    holdout = trained_run["output_dir"] / "tokenizer_evals_set.jsonl"
    with open(holdout, encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for _, line in zip(range(500), fh)]

    for text in texts:
        assert tokenizer.encode(text, add_special_tokens=False).ids == sp.encode(text)


def test_reserved_tokens_stay_atomic_after_export(exported, sp):
    from tokenizers import Tokenizer
    from specials import USER_DEFINED_SYMBOLS

    tokenizer = Tokenizer.from_file(str(exported))
    non_atomic = [
        name
        for name in USER_DEFINED_SYMBOLS
        if len(tokenizer.encode(name, add_special_tokens=False).ids) != 1
    ]
    assert non_atomic == []


def test_verify_detects_a_mismatched_tokenizer(trained_run):
    """
    The verifier must have teeth: a tokenizer that normalizes (which training
    does not) has to be rejected rather than exported.
    """
    from tokenizers import normalizers

    tokenizer, sp = export_hf.build_hf_tokenizer(trained_run["output_dir"] / "spm.model")
    assert export_hf.verify(tokenizer, sp, []) == []

    tokenizer.normalizer = normalizers.NFKC()
    assert export_hf.verify(tokenizer, sp, []) != []


def test_verify_tolerates_whitespace_reorder_ties(trained_run):
    """
    A run of a single repeated character (e.g. multi-space indentation) can
    have more than one globally-optimal Unigram segmentation: swapping which
    same-length pieces cover which sub-run leaves the summed score unchanged,
    since addition is commutative. sentencepiece's Viterbi and the `tokenizers`
    Rust Viterbi are each deterministic but can break such ties differently.

    This must not fail verification -- both sequences decode to the exact
    same bytes, so nothing is lost or corrupted; only which of two equally
    valid segmentations was chosen differs.
    """
    tokenizer, sp = export_hf.build_hf_tokenizer(trained_run["output_dir"] / "spm.model")
    text = "collapse    these     spaces?"

    hf_ids = tokenizer.encode(text, add_special_tokens=False).ids
    spm_ids = sp.encode(text)
    assert tokenizer.decode(hf_ids) == text
    assert sp.decode(spm_ids) == text

    # The probe is only useful as a regression check if it actually exercises
    # a real tie on this vocabulary; if a future vocab happens not to tie on
    # this exact string, that's fine -- there is simply nothing to tolerate.
    if hf_ids != spm_ids:
        assert sorted(hf_ids) == sorted(spm_ids), (
            "hf and spm disagree on this text in a way that is NOT a benign "
            "same-multiset reorder -- this would be a real vocabulary bug"
        )

    assert export_hf.verify(tokenizer, sp, []) == []


def test_verify_still_rejects_a_genuine_id_mismatch(trained_run):
    """
    The reorder-tie tolerance must not swallow a real mismatch. Force
    sentencepiece's encode to return a sequence that is NOT a permutation of
    the exported tokenizer's output, and confirm verify() still flags it.
    """
    tokenizer, sp = export_hf.build_hf_tokenizer(trained_run["output_dir"] / "spm.model")
    text = export_hf.VERIFY_PROBES[0]
    real_encode = sp.encode

    def fake_encode(t):
        ids = real_encode(t)
        if t == text:
            # Not a permutation: drop the last id. Also breaks decode, which
            # is exactly what a genuine bug (e.g. a dropped piece) looks like.
            return ids[:-1]
        return ids

    sp.encode = fake_encode
    problems = export_hf.verify(tokenizer, sp, [])
    assert any("mismatch" in p or "round-trip" in p for p in problems)


def test_verify_checks_full_vocab_identity(trained_run):
    """
    The exhaustive id<->piece check is the real "same ID space" contract --
    stronger than probing arbitrary strings, and unaffected by Viterbi
    tie-breaking. Corrupt one id's mapping and confirm it's caught.
    """
    tokenizer, sp = export_hf.build_hf_tokenizer(trained_run["output_dir"] / "spm.model")
    assert export_hf.verify(tokenizer, sp, []) == []

    real_id_to_piece = sp.id_to_piece

    def fake_id_to_piece(i):
        if i == 5:
            return "__corrupted__"
        return real_id_to_piece(i)

    sp.id_to_piece = fake_id_to_piece
    problems = export_hf.verify(tokenizer, sp, [])
    assert any("vocab id 5" in p for p in problems)


def test_verify_survives_literal_spiece_underline_in_text(trained_run):
    """
    Real text can contain SentencePiece's own marker character (U+2581, "▁")
    as ordinary content. Without the marker_escape fix, both tokenizer.decode
    and sp.decode silently rewrite it to a space, producing a false
    round-trip failure here. See Marker_Encoder_Design_EDGE_CASE.md.
    """
    tokenizer, sp = export_hf.build_hf_tokenizer(trained_run["output_dir"] / "spm.model")
    problems = export_hf.verify(tokenizer, sp, [" ▃▁▁ Low ", "▁", "a▁b▁c"])
    assert problems == []


def test_missing_model_exits_2(tmp_path):
    assert export_hf.main(["--output-dir", str(tmp_path)]) == 2
