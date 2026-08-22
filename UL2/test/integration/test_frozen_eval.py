"""
Integration tests for frozen validation sets (architecture.md 7.6, FROZEN).

Two rules are under test:

  1. Validation corruption is frozen, not re-randomized. If it drifts, the
     validation loss carries corruption-sampling noise -- the exact noise
     that hides the checkpoint-to-checkpoint differences the evaluation
     exists to reveal.
  2. The validation mode mixture is fixed and identical across runs, and
     does *not* follow the training mixture -- otherwise an X-heavy run is
     scored on X-heavy validation data and the comparison is circular.

The digest tests matter more than they look: a frozen set that has been
edited, half-written, or rebuilt with different settings must fail loudly,
because two runs scored on different data are not comparable and nothing
else in the pipeline would notice.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_tokens
from src.config import DEFAULT_MODE_WEIGHTS, FROZEN_EVAL_MODE_WEIGHTS, UL2Config
from src.denoise import reconstruct_source
from src.errors import FrozenEvalError
from src.frozen_eval import FORMAT_VERSION, FrozenEvalSet, build_frozen_eval_set

# Distinct, non-overlapping ID ranges, all well clear of the fixture's
# sentinel IDs -- a source token that collided with a sentinel would break
# reconstruction for reasons that have nothing to do with the code under test.
SEQUENCES = [make_tokens(length, start=20000 + i * 3000) for i, length in enumerate(
    [20, 45, 88, 130, 200, 340, 597, 900, 1200, 1600, 2048, 60]
)]


def _build(specials, sequences=None, **kwargs):
    return build_frozen_eval_set(
        sequences if sequences is not None else SEQUENCES,
        UL2Config(),
        specials,
        seed=2026,
        **kwargs,
    )


# -- construction ---------------------------------------------------------


def test_builds_one_example_per_admissible_sequence(specials):
    frozen = _build(specials)
    assert len(frozen) == len(SEQUENCES)


def test_examples_reconstruct_losslessly(specials):
    frozen = _build(specials)
    for example, tokens in zip(frozen.examples, SEQUENCES):
        assert reconstruct_source(example, specials) == tuple(tokens)


def test_mode_mixture_is_exact_not_sampled(specials):
    """
    Equal thirds, apportioned. With 12 sequences that is 4/4/4 every time --
    a sampled mixture would wobble and make per-mode losses incomparable
    across runs.
    """
    assert _build(specials).mode_counts() == {"R": 4, "S": 4, "X": 4}


def test_default_mixture_is_the_frozen_one_not_the_training_one(specials):
    frozen = _build(specials)
    assert frozen.mode_weights == dict(FROZEN_EVAL_MODE_WEIGHTS)
    assert frozen.mode_weights != dict(DEFAULT_MODE_WEIGHTS)


def test_an_explicit_mixture_is_honoured(specials):
    frozen = _build(specials, mode_weights={"R": 0.5, "X": 0.5})
    assert frozen.mode_counts() == {"R": 6, "X": 6}


def test_by_mode_selects_examples(specials):
    frozen = _build(specials)
    for mode in ("R", "X", "S"):
        selected = frozen.by_mode(mode)
        assert len(selected) == 4
        assert all(example.mode == mode for example in selected)


def test_max_examples_caps_the_set(specials):
    frozen = _build(specials, max_examples=6)
    assert len(frozen) == 6


def test_inadmissible_sequences_are_filtered_before_apportionment(specials):
    """
    Filtering first is what keeps the mode counts exact. If skipped
    sequences consumed schedule slots, the counts would silently drift.
    """
    sequences = [make_tokens(2), *SEQUENCES, make_tokens(9999), make_tokens(1)]
    frozen = build_frozen_eval_set(
        sequences, UL2Config(), specials, seed=2026
    )
    assert len(frozen) == len(SEQUENCES)
    assert frozen.mode_counts() == {"R": 4, "S": 4, "X": 4}


def test_no_admissible_sequences_raises(specials):
    with pytest.raises(FrozenEvalError, match="no admissible sequences"):
        build_frozen_eval_set([make_tokens(2)], UL2Config(), specials, seed=1)


def test_records_the_configuration_that_produced_it(specials):
    frozen = _build(specials)
    assert frozen.config == UL2Config().to_dict()
    assert frozen.seed == 2026


def test_telemetry_over_the_frozen_set(specials):
    report = _build(specials).telemetry().report()
    assert report["total_examples"] == len(SEQUENCES)
    assert set(report["modes"]) == {"R", "X", "S"}


# -- determinism ----------------------------------------------------------


def test_rebuilding_gives_an_identical_set(specials):
    """The core 'frozen' property."""
    first = _build(specials)
    second = _build(specials)
    assert first.digest == second.digest
    assert first.examples == second.examples


def test_a_different_seed_gives_a_different_set(specials):
    other = build_frozen_eval_set(SEQUENCES, UL2Config(), specials, seed=999)
    assert other.digest != _build(specials).digest


def test_per_example_seeding_is_independent_of_position(specials):
    """
    Each example is seeded from sha256(seed:index), so appending sequences
    cannot rewrite the examples that came before. Without this, extending a
    validation set would invalidate every earlier checkpoint's scores.
    """
    base = _build(specials)
    extended = _build(specials, sequences=[*SEQUENCES, make_tokens(300, start=90000)])

    # The mode schedule shifts when the count changes, so compare the
    # examples whose assigned mode is unchanged.
    shared = [
        (a, b)
        for a, b in zip(base.examples, extended.examples)
        if a.mode == b.mode
    ]
    assert shared, "expected at least some positions to keep their mode"
    for a, b in shared:
        assert a == b


def test_identical_sequences_still_get_different_corruption(specials):
    """
    The seed is derived per *index*, so repeated text does not receive
    repeated masks. If every example shared one seed, a validation set built
    from similar documents would measure the same few masked positions over
    and over -- a narrow, misleading estimate that no digest check would
    catch, because the set would still be perfectly reproducible.
    """
    repeated = [make_tokens(400, start=20000) for _ in range(12)]
    frozen = build_frozen_eval_set(repeated, UL2Config(), specials, seed=5)

    for mode in ("R", "X", "S"):
        examples = frozen.by_mode(mode)
        assert len(examples) == 4
        distinct = {example.encoder_input_ids for example in examples}
        assert len(distinct) == len(examples), (
            f"[{mode}] produced identical corruption for identical inputs; "
            f"per-example seeding has been lost"
        )


def test_digest_changes_when_the_content_changes(specials):
    changed = _build(specials, sequences=SEQUENCES[:-1])
    assert changed.digest != _build(specials).digest


# -- persistence ----------------------------------------------------------


def test_save_and_load_round_trip(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    loaded = FrozenEvalSet.load(path)
    original = _build(specials)

    assert loaded.examples == original.examples
    assert loaded.digest == original.digest
    assert loaded.seed == original.seed
    assert loaded.mode_weights == original.mode_weights
    assert loaded.config == original.config


def test_loaded_examples_still_reconstruct(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    for example, tokens in zip(FrozenEvalSet.load(path).examples, SEQUENCES):
        assert reconstruct_source(example, specials) == tuple(tokens)


def test_save_creates_parent_directories(tmp_path, specials):
    path = _build(specials).save(tmp_path / "nested" / "deeper" / "frozen.jsonl")
    assert path.is_file()


def test_save_leaves_no_partial_file(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    assert list(path.parent.iterdir()) == [path]


def test_manifest_is_human_readable(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    manifest = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["seed"] == 2026
    assert manifest["num_examples"] == len(SEQUENCES)
    assert manifest["mode_counts"] == {"R": 4, "S": 4, "X": 4}
    assert "digest" in manifest
    assert manifest["config"]["max_source_length"] == 2048


def test_one_line_per_example(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(SEQUENCES) + 1


# -- self-verification ----------------------------------------------------


def test_tampered_content_is_detected(tmp_path, specials):
    """Editing one token in one example must fail the load."""
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()

    payload = json.loads(lines[1])
    payload["encoder_input_ids"][2] += 1
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(FrozenEvalError, match="digest check"):
        FrozenEvalSet.load(path)


def test_truncated_file_is_detected(tmp_path, specials):
    """A half-written file must not load as a smaller but valid set."""
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")

    with pytest.raises(FrozenEvalError, match="truncated or edited"):
        FrozenEvalSet.load(path)


def test_appended_example_is_detected(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")

    with pytest.raises(FrozenEvalError, match="truncated or edited"):
        FrozenEvalSet.load(path)


def test_unknown_format_version_is_rejected(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()

    manifest = json.loads(lines[0])
    manifest["format_version"] = FORMAT_VERSION + 1
    lines[0] = json.dumps(manifest)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(FrozenEvalError, match="format_version"):
        FrozenEvalSet.load(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FrozenEvalError, match="not found"):
        FrozenEvalSet.load(tmp_path / "absent.jsonl")


def test_empty_file_is_reported(tmp_path):
    path = tmp_path / "frozen.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(FrozenEvalError, match="empty"):
        FrozenEvalSet.load(path)


def test_malformed_json_is_reported(tmp_path):
    path = tmp_path / "frozen.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(FrozenEvalError, match="malformed"):
        FrozenEvalSet.load(path)


def test_missing_example_field_is_reported(tmp_path, specials):
    path = _build(specials).save(tmp_path / "frozen.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()

    payload = json.loads(lines[1])
    del payload["num_spans"]
    lines[1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(FrozenEvalError, match="malformed"):
        FrozenEvalSet.load(path)
