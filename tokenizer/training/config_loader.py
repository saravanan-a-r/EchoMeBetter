"""
Load and validate the tokenization-training YAML config.

Design Principle 4 (DESIGN.md 2): fail loudly, never silently. A typo in a
config key must abort the run *before* a multi-hour training job starts,
not silently fall back to a default that quietly changes the vocabulary.
Every key is therefore required and type-checked; unknown keys are rejected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

Number = (int, float)

# section -> {key: expected type(s)}. Exhaustive: any key present in the YAML
# but missing here is an error, and vice versa.
SCHEMA: dict[str, dict[str, Any]] = {
    "corpus": {
        "input_sentence_size": int,
        "max_sentence_length": int,
        "max_drop_fraction": Number,
    },
    "sampling": {
        "seed": int,
    },
    "blend": {
        "mode": str,
        "shares": dict,
    },
    "trainer": {
        "model_type": str,
        "vocab_size": int,
        "byte_fallback": bool,
        "character_coverage": Number,
        "normalization_rule_name": str,
        "remove_extra_whitespaces": bool,
        "allow_whitespace_only_pieces": bool,
        "add_dummy_prefix": bool,
        "split_digits": bool,
        "train_extremely_large_corpus": bool,
        "num_threads": int,
        "pad_id": int,
        "eos_id": int,
        "unk_id": int,
        "bos_id": int,
    },
    "output": {
        "dir": str,
        "model_prefix": str,
    },
    "validation": {
        "run_after_training": bool,
        "eval_sentence_size": int,
        "eval_sample_save": int,
        "length_reservoir_size": int,
        "rare_token_max_count": int,
        "fertility_target_min": Number,
        "fertility_target_max": Number,
        "fertility_code_warn_above": Number,
        "byte_fallback_rate_warn_above": Number,
        "chars_per_token_warn_below": Number,
        "vocab_utilization_warn_below": Number,
    },
    "runtime": {
        "workers": int,
    },
}

# Sections that may be omitted entirely, filled in with these defaults. Only
# `runtime` qualifies: it holds pure execution knobs (how many processes to
# split the corpus scan across) which change how long a run takes and nothing
# about its result, so an older config without the section stays valid and
# every existing config keeps working untouched.
OPTIONAL_SECTIONS: dict[str, dict[str, Any]] = {
    "runtime": {"workers": 0},  # 0 = auto-detect
}


class ConfigError(ValueError):
    """Raised when the config file is missing, malformed, or semantically invalid."""


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def _check_schema(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError(f"config root must be a mapping, got {type(cfg).__name__}")

    for section, defaults in OPTIONAL_SECTIONS.items():
        body = cfg.setdefault(section, {})
        if isinstance(body, dict):
            for key, value in defaults.items():
                body.setdefault(key, value)

    missing_sections = sorted(set(SCHEMA) - set(cfg))
    if missing_sections:
        raise ConfigError(f"missing config section(s): {', '.join(missing_sections)}")
    unknown_sections = sorted(set(cfg) - set(SCHEMA))
    if unknown_sections:
        raise ConfigError(f"unknown config section(s): {', '.join(unknown_sections)}")

    for section, keys in SCHEMA.items():
        body = cfg[section]
        if not isinstance(body, dict):
            raise ConfigError(f"section '{section}' must be a mapping, got {type(body).__name__}")
        missing = sorted(set(keys) - set(body))
        if missing:
            raise ConfigError(f"section '{section}' missing key(s): {', '.join(missing)}")
        unknown = sorted(set(body) - set(keys))
        if unknown:
            raise ConfigError(f"section '{section}' has unknown key(s): {', '.join(unknown)}")
        for key, expected in keys.items():
            value = body[key]
            # bool is a subclass of int in Python; guard against `flag: 1`
            # silently satisfying an int field and vice versa.
            if expected is bool:
                if not isinstance(value, bool):
                    raise ConfigError(f"{section}.{key} must be bool, got {type(value).__name__}")
                continue
            if isinstance(value, bool):
                raise ConfigError(f"{section}.{key} must be {_type_name(expected)}, got bool")
            if not isinstance(value, expected):
                raise ConfigError(
                    f"{section}.{key} must be {_type_name(expected)}, got {type(value).__name__}"
                )


def _check_semantics(cfg: dict[str, Any]) -> None:
    corpus, trainer = cfg["corpus"], cfg["trainer"]
    validation = cfg["validation"]

    if corpus["input_sentence_size"] < 1:
        raise ConfigError("corpus.input_sentence_size must be >= 1")
    if corpus["max_sentence_length"] < 1:
        raise ConfigError("corpus.max_sentence_length must be >= 1")
    if not 0.0 <= corpus["max_drop_fraction"] <= 1.0:
        raise ConfigError("corpus.max_drop_fraction must be in [0.0, 1.0]")

    blend = cfg["blend"]
    if blend["mode"] not in ("uniform", "weighted"):
        raise ConfigError(f"blend.mode must be 'uniform' or 'weighted', got '{blend['mode']}'")
    for name, share in blend["shares"].items():
        if not isinstance(name, str):
            raise ConfigError(f"blend.shares keys must be source names, got {name!r}")
        if isinstance(share, bool) or not isinstance(share, Number):
            raise ConfigError(f"blend.shares['{name}'] must be a number, got {type(share).__name__}")
        if share < 0:
            raise ConfigError(f"blend.shares['{name}'] must be >= 0, got {share}")
    if blend["mode"] == "weighted":
        if not blend["shares"]:
            raise ConfigError("blend.mode is 'weighted' but blend.shares is empty")
        total_share = sum(blend["shares"].values())
        if total_share <= 0:
            raise ConfigError("blend.shares must not sum to zero in 'weighted' mode")
        # Shares are renormalized over the sources actually present, so they
        # need not sum to exactly 1 -- but a total far from 1 almost always
        # means a typo rather than an intention.
        if not 0.99 <= total_share <= 1.01:
            raise ConfigError(
                f"blend.shares sum to {total_share:.4f}; expected ~1.0. "
                f"Fix the shares, or use blend.mode 'uniform' if you do not want a "
                f"specified blend."
            )

    if trainer["model_type"] != "unigram":
        # DESIGN.md T1. Every downstream assumption (byte_fallback semantics,
        # fertility targets, the freeze contract) is written against Unigram.
        raise ConfigError(
            f"trainer.model_type must be 'unigram' (DESIGN.md T1), got '{trainer['model_type']}'"
        )
    if trainer["vocab_size"] < 1:
        raise ConfigError("trainer.vocab_size must be >= 1")
    if not 0.0 < trainer["character_coverage"] <= 1.0:
        raise ConfigError("trainer.character_coverage must be in (0.0, 1.0]")
    if trainer["num_threads"] < 0:
        raise ConfigError("trainer.num_threads must be >= 0 (0 = auto-detect)")

    if cfg["runtime"]["workers"] < 0:
        raise ConfigError("runtime.workers must be >= 0 (0 = auto-detect)")

    # The reserved block must physically fit inside the vocabulary. Catching
    # this here turns a confusing SentencePiece crash minutes into a run
    # into an immediate, explicit error.
    from specials import USER_DEFINED_SYMBOLS

    reserved = len(USER_DEFINED_SYMBOLS) + 3  # + pad/eos/unk
    if trainer["byte_fallback"]:
        reserved += 256
    if trainer["vocab_size"] <= reserved:
        raise ConfigError(
            f"trainer.vocab_size ({trainer['vocab_size']}) must exceed the reserved block "
            f"({reserved} = {len(USER_DEFINED_SYMBOLS)} user-defined + 3 control"
            f"{' + 256 byte-fallback' if trainer['byte_fallback'] else ''})"
        )

    ids = {"pad_id": trainer["pad_id"], "eos_id": trainer["eos_id"], "unk_id": trainer["unk_id"]}
    if len(set(ids.values())) != len(ids):
        raise ConfigError(f"trainer pad/eos/unk ids must be distinct, got {ids}")
    if trainer["unk_id"] < 0:
        raise ConfigError("trainer.unk_id must be >= 0 (<unk> is structurally required)")

    if validation["fertility_target_min"] > validation["fertility_target_max"]:
        raise ConfigError("validation.fertility_target_min must be <= fertility_target_max")
    # 0 is meaningful: "evaluate on every line not used for training".
    if validation["eval_sentence_size"] < 0:
        raise ConfigError(
            "validation.eval_sentence_size must be >= 0 (0 = evaluate on all remaining corpus)"
        )
    if validation["eval_sample_save"] < 0:
        raise ConfigError("validation.eval_sample_save must be >= 0")
    if validation["length_reservoir_size"] < 1:
        raise ConfigError("validation.length_reservoir_size must be >= 1")
    if validation["rare_token_max_count"] < 0:
        raise ConfigError("validation.rare_token_max_count must be >= 0")
    if validation["chars_per_token_warn_below"] < 0:
        raise ConfigError("validation.chars_per_token_warn_below must be >= 0")
    if not 0.0 <= validation["vocab_utilization_warn_below"] <= 1.0:
        raise ConfigError("validation.vocab_utilization_warn_below must be in [0.0, 1.0]")


def resolve_num_threads(num_threads: int) -> int:
    """0 means auto-detect; SentencePiece itself requires >= 1."""
    return num_threads if num_threads > 0 else (os.cpu_count() or 1)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read, validate and return the config. Raises ConfigError on any problem."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {path}: {exc}") from exc
    if cfg is None:
        raise ConfigError(f"config file is empty: {path}")

    _check_schema(cfg)
    _check_semantics(cfg)
    return cfg
