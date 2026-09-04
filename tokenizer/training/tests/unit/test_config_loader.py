"""
Unit tests for config loading and validation.

The contract under test: a bad config must fail *immediately*, with a message
naming the offending key -- never silently fall back to a default. A config
typo that survives validation would silently change the vocabulary of a
multi-hour, unrepeatable build (DESIGN.md Principle 4).
"""

from __future__ import annotations

import copy
import os

import pytest
import yaml

from config_loader import ConfigError, load_config, resolve_num_threads


@pytest.fixture(scope="module")
def base_cfg(default_config_path):
    return load_config(default_config_path)


def write_cfg(tmp_path, cfg, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# --- happy path -----------------------------------------------------------


def test_shipped_configs_are_valid(module_dir):
    """Every config we ship must load. Catches drift when SCHEMA changes."""
    for name in ("default.yaml", "smoke.yaml", "laptop_16gb.yaml"):
        cfg = load_config(module_dir / "config" / name)
        assert cfg["trainer"]["model_type"] == "unigram"


def test_fidelity_settings_identical_across_shipped_configs(module_dir):
    """
    Scale may differ between configs; fidelity must not. A smoke run whose
    fidelity settings differ from production proves nothing about production
    (DESIGN.md 6.1).
    """
    fidelity_keys = [
        "byte_fallback", "character_coverage", "normalization_rule_name",
        "remove_extra_whitespaces", "allow_whitespace_only_pieces",
        "add_dummy_prefix", "split_digits", "pad_id", "eos_id", "unk_id", "bos_id",
    ]
    configs = [
        load_config(module_dir / "config" / name)
        for name in ("default.yaml", "smoke.yaml", "laptop_16gb.yaml")
    ]
    for key in fidelity_keys:
        values = {cfg["trainer"][key] for cfg in configs}
        assert len(values) == 1, f"{key} differs across shipped configs: {values}"


def test_design_critical_defaults(base_cfg):
    """The settings NF1 depends on (DESIGN.md 6.1). Regression lock."""
    t = base_cfg["trainer"]
    assert t["byte_fallback"] is True
    assert t["normalization_rule_name"] == "identity"
    assert t["remove_extra_whitespaces"] is False
    assert t["allow_whitespace_only_pieces"] is True
    assert t["add_dummy_prefix"] is False
    assert t["split_digits"] is True
    assert t["character_coverage"] == 1.0
    assert t["vocab_size"] == 32832
    assert (t["pad_id"], t["eos_id"], t["unk_id"], t["bos_id"]) == (0, 1, 2, -1)


# --- file-level failures --------------------------------------------------


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("corpus: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse YAML"):
        load_config(path)


def test_root_not_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


# --- schema failures ------------------------------------------------------


def test_missing_section(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    del cfg["validation"]
    with pytest.raises(ConfigError, match="missing config section"):
        load_config(write_cfg(tmp_path, cfg))


def test_unknown_section_is_rejected(tmp_path, base_cfg):
    """A typo'd section name must not be silently ignored."""
    cfg = copy.deepcopy(base_cfg)
    cfg["trainner"] = {}
    with pytest.raises(ConfigError, match="unknown config section"):
        load_config(write_cfg(tmp_path, cfg))


def test_missing_key(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    del cfg["trainer"]["byte_fallback"]
    with pytest.raises(ConfigError, match="missing key"):
        load_config(write_cfg(tmp_path, cfg))


def test_unknown_key_is_rejected(tmp_path, base_cfg):
    """`byte_falback: true` must fail, not silently leave byte_fallback unset."""
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["byte_falback"] = True
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_cfg(tmp_path, cfg))


def test_section_not_a_mapping(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["sampling"] = 42
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write_cfg(tmp_path, cfg))


def test_wrong_type(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["vocab_size"] = "32832"
    with pytest.raises(ConfigError, match="vocab_size must be int"):
        load_config(write_cfg(tmp_path, cfg))


def test_int_where_bool_expected(tmp_path, base_cfg):
    """`byte_fallback: 1` must not pass as True -- bool subclasses int."""
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["byte_fallback"] = 1
    with pytest.raises(ConfigError, match="byte_fallback must be bool"):
        load_config(write_cfg(tmp_path, cfg))


def test_bool_where_int_expected(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["vocab_size"] = True
    with pytest.raises(ConfigError, match="vocab_size must be int, got bool"):
        load_config(write_cfg(tmp_path, cfg))


def test_int_accepted_where_float_expected(tmp_path, base_cfg):
    """`character_coverage: 1` is a legitimate way to write 1.0."""
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["character_coverage"] = 1
    assert load_config(write_cfg(tmp_path, cfg))["trainer"]["character_coverage"] == 1


# --- semantic failures ----------------------------------------------------


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("corpus", "input_sentence_size", 0, "input_sentence_size must be >= 1"),
        ("corpus", "max_sentence_length", 0, "max_sentence_length must be >= 1"),
        ("corpus", "max_drop_fraction", 1.5, "max_drop_fraction must be in"),
        ("corpus", "max_drop_fraction", -0.1, "max_drop_fraction must be in"),
        ("validation", "eval_sentence_size", -1, "eval_sentence_size must be >= 0"),
        ("validation", "eval_sample_save", -1, "eval_sample_save must be >= 0"),
        ("validation", "length_reservoir_size", 0, "length_reservoir_size must be >= 1"),
        ("validation", "vocab_utilization_warn_below", 1.5, "must be in \\[0.0, 1.0\\]"),
        ("trainer", "model_type", "bpe", "must be 'unigram'"),
        ("trainer", "vocab_size", 0, "vocab_size must be >= 1"),
        ("trainer", "character_coverage", 0.0, "character_coverage must be in"),
        ("trainer", "character_coverage", 1.5, "character_coverage must be in"),
        ("trainer", "num_threads", -1, "num_threads must be >= 0"),
        ("trainer", "unk_id", -1, "unk_id must be >= 0"),
        ("validation", "rare_token_max_count", -1, "rare_token_max_count must be >= 0"),
    ],
)
def test_semantic_bounds(tmp_path, base_cfg, section, key, value, match):
    cfg = copy.deepcopy(base_cfg)
    cfg[section][key] = value
    with pytest.raises(ConfigError, match=match):
        load_config(write_cfg(tmp_path, cfg))


def test_vocab_size_smaller_than_reserved_block(tmp_path, base_cfg):
    """
    632 slots are spoken for before a single piece is learned (373 user-defined
    + 3 control + 256 byte-fallback). Catching this here turns a confusing
    SentencePiece crash into an explicit message.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["vocab_size"] = 400
    with pytest.raises(ConfigError, match="must exceed the reserved block"):
        load_config(write_cfg(tmp_path, cfg))


def test_reserved_block_excludes_bytes_when_byte_fallback_off(tmp_path, base_cfg):
    """Without byte_fallback the 256 byte pieces are not reserved."""
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["byte_fallback"] = False
    cfg["trainer"]["vocab_size"] = 400
    assert load_config(write_cfg(tmp_path, cfg))["trainer"]["vocab_size"] == 400


def test_duplicate_special_ids(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["trainer"]["eos_id"] = 0  # collides with pad_id
    with pytest.raises(ConfigError, match="must be distinct"):
        load_config(write_cfg(tmp_path, cfg))


def test_fertility_band_inverted(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["validation"]["fertility_target_min"] = 2.0
    cfg["validation"]["fertility_target_max"] = 1.0
    with pytest.raises(ConfigError, match="must be <= fertility_target_max"):
        load_config(write_cfg(tmp_path, cfg))


# --- blend ----------------------------------------------------------------


def test_shipped_blend_matches_design_shares(base_cfg):
    """DESIGN.md 4.3. Regression lock on the corpus mix the vocab is fitted to."""
    assert base_cfg["blend"]["mode"] == "weighted"
    assert base_cfg["blend"]["shares"] == {
        "fineweb_edu": 0.35,
        "c4": 0.20,
        "stackexchange": 0.15,
        "cosmopedia": 0.10,
        "gutenberg_pg19": 0.20,
    }


def test_shipped_blends_are_identical_across_configs(module_dir):
    blends = [
        load_config(module_dir / "config" / name)["blend"]
        for name in ("default.yaml", "smoke.yaml", "laptop_16gb.yaml")
    ]
    assert all(b == blends[0] for b in blends)


def test_blend_mode_must_be_known(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["blend"]["mode"] = "proportional"
    with pytest.raises(ConfigError, match="must be 'uniform' or 'weighted'"):
        load_config(write_cfg(tmp_path, cfg))


def test_weighted_blend_requires_shares(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["blend"]["shares"] = {}
    with pytest.raises(ConfigError, match="blend.shares is empty"):
        load_config(write_cfg(tmp_path, cfg))


def test_blend_shares_must_sum_to_about_one(tmp_path, base_cfg):
    """A total far from 1.0 is a typo, not an intention."""
    cfg = copy.deepcopy(base_cfg)
    cfg["blend"]["shares"]["fineweb_edu"] = 0.95
    with pytest.raises(ConfigError, match="expected ~1.0"):
        load_config(write_cfg(tmp_path, cfg))


def test_blend_shares_reject_negative(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["blend"]["shares"] = {"a": 1.2, "b": -0.2}
    with pytest.raises(ConfigError, match="must be >= 0"):
        load_config(write_cfg(tmp_path, cfg))


def test_blend_shares_reject_non_numeric(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["blend"]["shares"] = {"a": "0.5", "b": 0.5}
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(write_cfg(tmp_path, cfg))


def test_uniform_mode_allows_empty_shares(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["blend"] = {"mode": "uniform", "shares": {}}
    assert load_config(write_cfg(tmp_path, cfg))["blend"]["mode"] == "uniform"


# --- helpers --------------------------------------------------------------


def test_resolve_num_threads_auto():
    assert resolve_num_threads(0) == (os.cpu_count() or 1)
    assert resolve_num_threads(0) >= 1


def test_resolve_num_threads_explicit():
    assert resolve_num_threads(4) == 4
