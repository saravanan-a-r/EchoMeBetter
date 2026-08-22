"""
Unit tests for the token-ID boundary.

The rule these lock down is architecture.md 6.4: IDs are resolved by *name*,
never hardcoded, and a malformed table fails loudly rather than producing
examples that quietly reference the wrong embedding rows.
"""

from __future__ import annotations

import json

import pytest

from src.errors import SpecialTokenError
from src.special_tokens import (
    DEFAULT_NUM_SENTINELS,
    MODES,
    SpecialTokens,
)


def _token_map(num_sentinels: int = 8) -> dict[str, int]:
    mapping = {"<pad>": 0, "</s>": 1, "<unk>": 2, "[R]": 259, "[X]": 260, "[S]": 261}
    for i in range(num_sentinels):
        mapping[f"<extra_id_{i}>"] = 3 + i
    return mapping


def _specials(**overrides) -> SpecialTokens:
    kwargs = dict(
        sentinel_ids=(10, 11, 12),
        mode_token_ids={"R": 20, "X": 21, "S": 22},
        eos_id=1,
        pad_id=0,
        decoder_start_id=0,
    )
    kwargs.update(overrides)
    return SpecialTokens(**kwargs)


# -- construction ---------------------------------------------------------


def test_valid_table_constructs():
    specials = _specials()
    assert specials.num_sentinels == 3
    assert specials.sentinel(0) == 10
    assert specials.mode_token("X") == 21


def test_modes_are_the_three_ul2_denoisers():
    assert MODES == ("R", "X", "S")


def test_empty_sentinels_rejected():
    with pytest.raises(SpecialTokenError, match="empty"):
        _specials(sentinel_ids=())


def test_duplicate_sentinels_rejected():
    """Two spans sharing a marker makes the decoder target ambiguous."""
    with pytest.raises(SpecialTokenError, match="duplicate"):
        _specials(sentinel_ids=(10, 11, 10))


def test_missing_mode_token_rejected():
    with pytest.raises(SpecialTokenError, match="missing mode token"):
        _specials(mode_token_ids={"R": 20, "X": 21})


def test_duplicate_mode_tokens_rejected():
    """[R] and [X] sharing an ID means the model cannot tell the tasks apart."""
    with pytest.raises(SpecialTokenError, match="distinct"):
        _specials(mode_token_ids={"R": 20, "X": 20, "S": 22})


def test_sentinel_colliding_with_eos_rejected():
    """A sentinel that is also EOS would terminate the target early."""
    with pytest.raises(SpecialTokenError, match="collide"):
        _specials(sentinel_ids=(10, 1, 12))


def test_sentinel_colliding_with_mode_token_rejected():
    with pytest.raises(SpecialTokenError, match="collide"):
        _specials(sentinel_ids=(10, 20, 12))


@pytest.mark.parametrize("field", ["eos_id", "pad_id", "decoder_start_id"])
def test_negative_builtin_ids_rejected(field):
    with pytest.raises(SpecialTokenError, match=field):
        _specials(**{field: -1})


def test_negative_sentinel_id_rejected():
    with pytest.raises(SpecialTokenError, match="non-negative"):
        _specials(sentinel_ids=(10, -3, 12))


def test_non_negative_label_pad_rejected():
    """
    label_pad_id must be negative: a non-negative value could collide with a
    real vocabulary ID and silently become a training target.
    """
    with pytest.raises(SpecialTokenError, match="negative"):
        _specials(label_pad_id=0)


def test_default_label_pad_is_pytorch_ignore_index():
    assert _specials().label_pad_id == -100


# -- sentinel access ------------------------------------------------------


def test_sentinel_index_out_of_range_raises_rather_than_wrapping():
    """
    architecture.md 7.4: never wrap around. Wrapping would reuse
    <extra_id_0> for two different spans in one example.
    """
    specials = _specials()
    with pytest.raises(SpecialTokenError, match="out of range"):
        specials.sentinel(3)
    with pytest.raises(SpecialTokenError, match="out of range"):
        specials.sentinel(-1)


def test_unknown_mode_raises():
    with pytest.raises(SpecialTokenError, match="unknown mode"):
        _specials().mode_token("Q")


# -- from_token_map -------------------------------------------------------


def test_from_token_map_resolves_by_name_not_position():
    """
    The IDs below are deliberately not in vocabulary order. Resolution must
    come from the name, so nothing can depend on ordering.
    """
    mapping = _token_map(num_sentinels=4)
    specials = SpecialTokens.from_token_map(mapping, num_sentinels=4)

    assert specials.sentinel_ids == (3, 4, 5, 6)
    assert specials.mode_token("R") == 259
    assert specials.eos_id == 1
    assert specials.pad_id == 0


def test_from_token_map_uses_pad_as_decoder_start():
    """architecture.md 6.1: bos_id=-1, so T5's convention is pad-as-start."""
    specials = SpecialTokens.from_token_map(_token_map(4), num_sentinels=4)
    assert specials.decoder_start_id == specials.pad_id


def test_from_token_map_defaults_to_the_real_sentinel_budget():
    assert DEFAULT_NUM_SENTINELS == 256
    specials = SpecialTokens.from_token_map(_token_map(256))
    assert specials.num_sentinels == 256


def test_from_token_map_missing_sentinel_raises():
    mapping = _token_map(4)
    del mapping["<extra_id_2>"]
    with pytest.raises(SpecialTokenError, match="extra_id_2"):
        SpecialTokens.from_token_map(mapping, num_sentinels=4)


def test_from_token_map_missing_mode_token_raises():
    mapping = _token_map(4)
    del mapping["[S]"]
    with pytest.raises(SpecialTokenError, match=r"\[S\]"):
        SpecialTokens.from_token_map(mapping, num_sentinels=4)


def test_from_token_map_requesting_more_sentinels_than_exist_raises():
    with pytest.raises(SpecialTokenError, match="extra_id_8"):
        SpecialTokens.from_token_map(_token_map(8), num_sentinels=16)


def test_from_token_map_rejects_non_integer_entry():
    mapping = _token_map(4)
    mapping["<extra_id_1>"] = "4"
    with pytest.raises(SpecialTokenError, match="not an integer"):
        SpecialTokens.from_token_map(mapping, num_sentinels=4)


def test_from_token_map_rejects_bool_entry():
    """bool is an int subclass in Python; it is still not a token ID."""
    mapping = _token_map(4)
    mapping["<extra_id_1>"] = True
    with pytest.raises(SpecialTokenError, match="not an integer"):
        SpecialTokens.from_token_map(mapping, num_sentinels=4)


def test_from_token_map_rejects_zero_sentinels():
    with pytest.raises(SpecialTokenError, match="must be positive"):
        SpecialTokens.from_token_map(_token_map(4), num_sentinels=0)


def test_from_token_map_reads_a_file(tmp_path):
    path = tmp_path / "token_map.json"
    path.write_text(json.dumps(_token_map(4)), encoding="utf-8")

    specials = SpecialTokens.from_token_map(path, num_sentinels=4)
    assert specials.sentinel_ids == (3, 4, 5, 6)


def test_from_token_map_missing_file_raises(tmp_path):
    with pytest.raises(SpecialTokenError, match="not found"):
        SpecialTokens.from_token_map(tmp_path / "absent.json", num_sentinels=4)


def test_from_token_map_invalid_json_raises(tmp_path):
    path = tmp_path / "token_map.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SpecialTokenError, match="not valid JSON"):
        SpecialTokens.from_token_map(path, num_sentinels=4)


def test_from_token_map_non_object_json_raises(tmp_path):
    path = tmp_path / "token_map.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SpecialTokenError, match="JSON object"):
        SpecialTokens.from_token_map(path, num_sentinels=4)


def test_from_token_map_unsupported_source_raises():
    with pytest.raises(SpecialTokenError, match="unsupported"):
        SpecialTokens.from_token_map(12345)  # type: ignore[arg-type]


def test_special_tokens_is_immutable():
    """IDs are welded to a checkpoint's embedding table; they must not change."""
    specials = _specials()
    with pytest.raises(Exception):
        specials.eos_id = 99  # type: ignore[misc]
