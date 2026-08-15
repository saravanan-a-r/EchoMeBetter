"""
Integration tests for the acceptance gates themselves.

A gate that cannot fail is worse than no gate: it produces a green report
while the property it claims to check is broken. Each hard gate is therefore
tested twice -- that it passes on a good model, and that it *detects* the
corresponding defect. The negative cases use a deliberately mis-configured
tokenizer, which is the only honest way to prove a gate has teeth.

The gates consume a **stream** of (text, source) pairs, so they can be
measured against the whole remaining corpus. These tests feed them small
lists, which are iterable in exactly the same way.
"""

from __future__ import annotations

import copy

import pytest

import validate_tokenizer as vt

# Gate numbers by kind, asserted below so the classification cannot drift
# silently -- a gate quietly demoted from hard to soft would stop blocking
# bad builds without any test noticing.
HARD_GATES = [1, 2, 3, 6]
SOFT_GATES = [4, 5, 8, 10]
REPORT_GATES = [7, 9, 11]


@pytest.fixture(scope="module")
def records():
    return [
        ("Visit https://example.com/a?b=1 for details.", "fineweb_edu"),
        ("See [docs](https://docs.python.org/3/) now.", "fineweb_edu"),
        ("    indented code\twith tabs", "stackexchange"),
        ("café ½ → 日本語", "stackexchange"),
        ("numbers 1234 and 3.14", "gutenberg_pg19"),
    ]


def scan_of(sp, lines, cfg):
    """The one-pass streaming scan every corpus-dependent gate reads from."""
    return vt.stream_scan(
        sp,
        lines,
        vt._byte_piece_ids(sp),
        length_reservoir_size=cfg["validation"]["length_reservoir_size"],
        seed=cfg["sampling"]["seed"],
    )


# --- gates pass on the real build ----------------------------------------


def test_all_hard_gates_pass(sp, records, trained_run):
    results, _ = vt.run_all_gates(sp, records, trained_run["cfg"])
    assert vt.hard_failures(results) == []


def test_run_all_gates_returns_eleven_numbered_gates(sp, records, trained_run):
    results, _ = vt.run_all_gates(sp, records, trained_run["cfg"])
    assert [r.number for r in results] == list(range(1, 12))
    assert [r.number for r in results if r.hard] == HARD_GATES


def test_scan_is_returned_for_reporting(sp, records, trained_run):
    """Callers need the corpus counts without paying for a second pass."""
    _, scan = vt.run_all_gates(sp, records, trained_run["cfg"])
    assert scan.n_records == len(records)
    assert scan.total_tokens > 0


def test_gates_consume_a_one_shot_iterator(sp, records, trained_run):
    """
    The eval set is a generator over ~100M lines, not a list. If any gate
    quietly re-iterated the input, it would silently see nothing the second
    time -- and every corpus-dependent metric would be wrong.
    """
    results, scan = vt.run_all_gates(sp, iter(records), trained_run["cfg"])
    assert scan.n_records == len(records)
    assert next(r for r in results if r.number == 1).metrics["records"] == len(records)


# --- gates have teeth -----------------------------------------------------


def test_gate_1_detects_lossy_roundtrip(lossy_sp, records, trained_run):
    """The whitespace/NFKC-collapsing tokenizer must fail the keystone gate."""
    result = vt.gate_1_roundtrip(scan_of(lossy_sp, records, trained_run["cfg"]))
    assert not result.passed
    assert result.examples


def test_gate_1_survives_literal_spiece_underline_in_text(sp, trained_run):
    """
    Real text can contain SentencePiece's own marker character (U+2581,
    "▁") as ordinary content -- e.g. terminal block-shading in a pasted
    ASCII chart. Without the marker_escape fix, sp.decode() cannot tell
    that occurrence apart from its own internal marker and silently drops
    it. See Marker_Encoder_Design_EDGE_CASE.md.
    """
    exotic = [
        (" ▃▁▁ Low ", "stackexchange"),  # the exact text that surfaced this
        ("▁", "stackexchange"),
        ("run: ▁▁▁▁ end", "stackexchange"),
    ]
    result = vt.gate_1_roundtrip(scan_of(sp, exotic, trained_run["cfg"]))
    assert result.passed, result.examples
    assert result.metrics["failed"] == 0


def test_gate_6_survives_literal_spiece_underline_in_a_url(sp, trained_run):
    """The same escaping must apply on gate 6's own direct encode/decode."""
    records = [("Visit https://example.com/▁path now.", "test")]
    scan = scan_of(sp, records, trained_run["cfg"])
    result = vt.gate_6_url_survival(sp, scan)
    assert result.passed, result.examples


def test_gate_2_detects_unk(lossy_sp, trained_run):
    """Without byte_fallback, out-of-alphabet characters become <unk>."""
    exotic = [("ᚠᚢᚦᚨᚱᚲ 🜁🜂 ﷽", "test")]
    result = vt.gate_2_no_unk(scan_of(lossy_sp, exotic, trained_run["cfg"]))
    assert not result.passed
    assert result.metrics["unk_tokens"] > 0


def test_gate_3_detects_missing_reserved_tokens(lossy_sp, trained_run):
    """The lossy model has no user_defined_symbols at all."""
    result = vt.gate_3_specials_atomic(lossy_sp, trained_run["cfg"])
    assert not result.passed
    assert result.metrics["failed"] > 100


def test_gate_3_detects_wrong_control_ids(sp, trained_run):
    """A config whose pad/eos ids disagree with the model must be caught."""
    cfg = copy.deepcopy(trained_run["cfg"])
    cfg["trainer"]["pad_id"] = 7
    result = vt.gate_3_specials_atomic(sp, cfg)
    assert not result.passed
    assert any("<pad>" in example for example in result.examples)


def test_gate_6_detects_broken_urls(lossy_sp, records, trained_run):
    result = vt.gate_6_url_survival(lossy_sp, scan_of(lossy_sp, records, trained_run["cfg"]))
    assert not result.passed
    assert result.metrics["failed"] > 0


def test_gate_6_reports_vacuity_when_no_urls_present(sp, trained_run):
    """
    No URLs found is not a silent pass: the report must say the gate was
    vacuous so absence of evidence is visible.
    """
    scan = scan_of(sp, [("plain prose only", "x")], trained_run["cfg"])
    result = vt.gate_6_url_survival(sp, scan)
    assert result.passed
    assert "vacuous" in result.detail
    assert result.metrics["fragments"] == 0


# --- soft gates measure what they claim ----------------------------------


def test_gate_4_reports_per_slice_fertility(sp, records, trained_run):
    result = vt.gate_4_fertility(scan_of(sp, records, trained_run["cfg"]), trained_run["cfg"])
    assert set(result.metrics["per_slice"]) == {"fineweb_edu", "stackexchange", "gutenberg_pg19"}
    assert result.metrics["overall"] > 0


def test_gate_4_warns_outside_band_without_failing_the_run(sp, records, trained_run):
    """Soft gates warn; they must never block a build on their own."""
    cfg = copy.deepcopy(trained_run["cfg"])
    cfg["validation"]["fertility_target_min"] = 0.0
    cfg["validation"]["fertility_target_max"] = 0.01
    cfg["validation"]["fertility_code_warn_above"] = 0.02
    result = vt.gate_4_fertility(scan_of(sp, records, cfg), cfg)
    assert not result.passed
    assert result.hard is False
    assert vt.hard_failures([result]) == []


def test_gate_5_rate_is_low_on_normal_prose(sp, trained_run):
    prose = [("This is ordinary English prose about nothing at all.", "p")]
    result = vt.gate_5_byte_fallback_rate(
        scan_of(sp, prose, trained_run["cfg"]), trained_run["cfg"]
    )
    assert result.passed
    assert result.metrics["rate"] < 0.05


def test_gate_5_rate_rises_on_exotic_text(sp, trained_run):
    """Byte fallback should actually engage when the text demands it."""
    exotic = [("ᚠᚢᚦᚨᚱᚲ ⡌⠁⠧⠑ 🜁🜂🜃🜄", "x")]
    scan = scan_of(sp, exotic, trained_run["cfg"])
    assert vt.gate_5_byte_fallback_rate(scan, trained_run["cfg"]).metrics["rate"] > 0.5


def test_gate_7_samples_the_vocabulary(sp):
    result = vt.gate_7_vocab_sanity(sp, seed=42)
    assert result.passed
    assert len(result.metrics["top_500"]) == 500
    assert len(result.metrics["random_500"]) == 500
    assert result.metrics["vocab_size"] == sp.get_piece_size()


def test_gate_7_is_deterministic(sp):
    a = vt.gate_7_vocab_sanity(sp, seed=42).metrics["random_500"]
    b = vt.gate_7_vocab_sanity(sp, seed=42).metrics["random_500"]
    assert a == b


# --- gate 8: characters per token -----------------------------------------


def test_gate_8_reports_overall_and_per_slice(sp, records, trained_run):
    result = vt.gate_8_chars_per_token(scan_of(sp, records, trained_run["cfg"]), trained_run["cfg"])
    assert result.metrics["overall"] > 0
    assert set(result.metrics["per_slice"]) == {"fineweb_edu", "stackexchange", "gutenberg_pg19"}


def test_gate_8_warns_when_compression_is_poor(sp, records, trained_run):
    cfg = copy.deepcopy(trained_run["cfg"])
    cfg["validation"]["chars_per_token_warn_below"] = 99.0
    result = vt.gate_8_chars_per_token(scan_of(sp, records, cfg), cfg)
    assert not result.passed
    assert result.hard is False
    assert result.examples


# --- gate 9: sequence length ----------------------------------------------


def test_gate_9_percentiles_are_ordered(sp, trained_run):
    lines = [(f"sentence number {i} with a few words in it", "x") for i in range(300)]
    m = vt.gate_9_sequence_length(scan_of(sp, lines, trained_run["cfg"])).metrics
    assert m["p50"] <= m["p90"] <= m["p95"] <= m["p99"] <= m["max"]
    assert m["mean"] > 0


def test_gate_9_max_is_exact_even_when_percentiles_are_sampled(sp, trained_run):
    """
    The reservoir can miss the single longest sentence in a large corpus, so
    the maximum is tracked separately and must always be exact.
    """
    cfg = copy.deepcopy(trained_run["cfg"])
    cfg["validation"]["length_reservoir_size"] = 5
    lines = [("short one", "x")] * 200 + [("word " * 400, "x")]
    scan = scan_of(sp, lines, cfg)
    m = vt.gate_9_sequence_length(scan).metrics
    assert m["reservoir_size"] == 5
    assert m["estimated"] is True
    assert m["max"] == max(len(sp.encode(t)) for t, _ in lines)


# --- gate 10: vocabulary utilization --------------------------------------


def test_gate_10_is_informational_on_a_small_corpus(sp, records, trained_run):
    """
    On a small eval set a piece can be absent simply because it never came
    up. Reporting that as a warning would be a false alarm, so the gate must
    detect its own lack of statistical power.
    """
    result = vt.gate_10_vocab_utilization(
        scan_of(sp, records, trained_run["cfg"]), trained_run["cfg"]
    )
    assert result.metrics["meaningful"] is False
    assert result.passed  # informational, never a warning at this scale
    assert "INFORMATIONAL" in result.detail


def test_gate_10_warns_when_meaningful_and_underused(sp, records, trained_run):
    cfg = copy.deepcopy(trained_run["cfg"])
    scan = scan_of(sp, records, cfg)
    # Force the meaningfulness test to pass so the threshold logic is exercised.
    scan.total_tokens = vt.MIN_TOKENS_PER_VOCAB_SLOT * len(scan.id_counts)
    cfg["validation"]["vocab_utilization_warn_below"] = 0.99
    result = vt.gate_10_vocab_utilization(scan, cfg)
    assert result.metrics["meaningful"] is True
    assert not result.passed
    assert result.hard is False


def test_gate_10_partitions_the_vocabulary(sp, records, trained_run):
    m = vt.gate_10_vocab_utilization(
        scan_of(sp, records, trained_run["cfg"]), trained_run["cfg"]
    ).metrics
    assert m["used"] + m["dead"] == m["vocab_size"] == sp.get_piece_size()


# --- gate 11: token frequency ---------------------------------------------


def test_gate_11_concentration_is_monotonic(sp, trained_run):
    lines = [(f"repeated pattern number {i}", "x") for i in range(300)]
    c = vt.gate_11_token_frequency(
        scan_of(sp, lines, trained_run["cfg"]), trained_run["cfg"]
    ).metrics["concentration"]
    ordered = [c[f"top_{p}pct"] for p in (1, 5, 10, 25, 50)]
    assert ordered == sorted(ordered)
    assert 0.0 <= ordered[0] <= ordered[-1] <= 1.0


# --- reporting ------------------------------------------------------------


def test_empty_eval_does_not_crash_and_does_not_pass_gate_1(sp, trained_run):
    """A vacuous Gate 1 must not be reported as a pass."""
    results, _ = vt.run_all_gates(sp, [], trained_run["cfg"])
    gate_1 = next(r for r in results if r.number == 1)
    assert not gate_1.passed


def test_render_report_includes_failures(lossy_sp, records, trained_run):
    results, _ = vt.run_all_gates(lossy_sp, records, trained_run["cfg"])
    report = vt.render_report(results, {"timestamp": "now"})
    assert "**FAIL**" in report
    assert "Gate 1 findings" in report


def test_render_report_covers_every_gate(sp, records, trained_run):
    results, _ = vt.run_all_gates(sp, records, trained_run["cfg"])
    report = vt.render_report(results, {"timestamp": "now"})
    for result in results:
        assert result.name in report
    for heading in ("## Correctness", "## Efficiency", "## Per source", "## Vocabulary usage"):
        assert heading in report


def test_gate_result_serialises(sp, records, trained_run):
    results, _ = vt.run_all_gates(sp, records, trained_run["cfg"])
    for result in results:
        payload = result.as_dict()
        assert set(payload) == {
            "gate", "name", "hard", "passed", "detail", "metrics", "examples"
        }


def test_failure_examples_are_capped(lossy_sp, trained_run):
    """A catastrophic failure must produce a readable report, not a flood."""
    many = [(f"   spaced  {i}   ", "x") for i in range(500)]
    scan = scan_of(lossy_sp, many, trained_run["cfg"])
    assert len(vt.gate_1_roundtrip(scan).examples) <= vt.MAX_EXAMPLES


def test_url_fragment_collection_is_bounded(sp, trained_run, monkeypatch):
    """
    Gate 6's unique-fragment set must not grow with the corpus: on 100M eval
    lines an unbounded set would exhaust memory.
    """
    monkeypatch.setattr(vt, "MAX_URL_FRAGMENTS", 10)
    lines = [(f"see https://example.com/page{i} now", "x") for i in range(200)]
    scan = scan_of(sp, lines, trained_run["cfg"])
    assert scan.url_fragments_capped is True
    assert len(scan.url_fragments) <= 2 * vt.MAX_URL_FRAGMENTS
    assert "cap" in vt.gate_6_url_survival(sp, scan).detail
