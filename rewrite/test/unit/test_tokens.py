"""
The frame's IDs are resolved by name, never assumed (architecture.md 6.4).
"""

from __future__ import annotations

import pytest

from conftest import REAL_TOKEN_MAP
from src.errors import RewriteTokenError
from src.tokens import CLOSE_TOKEN, DEFAULT_STYLE_TOKEN, OPEN_TOKEN, RewriteTokens

FAKE_MAP = {
    "<pad>": 0,
    "</s>": 1,
    "<style:grammar>": 262,
    "<style:concise>": 263,
    "<text_to_rewrite>": 274,
    "</text_to_rewrite>": 275,
}


def test_ids_come_from_the_map_by_name():
    tokens = RewriteTokens.from_token_map(FAKE_MAP)
    assert (tokens.style_id, tokens.open_id, tokens.close_id) == (262, 274, 275)
    assert (tokens.eos_id, tokens.decoder_start_id) == (1, 0)
    assert tokens.style_token == DEFAULT_STYLE_TOKEN


def test_another_style_can_be_chosen():
    assert RewriteTokens.from_token_map(FAKE_MAP, style_token="<style:concise>").style_id == 263


@pytest.mark.parametrize("missing", [DEFAULT_STYLE_TOKEN, OPEN_TOKEN, CLOSE_TOKEN, "</s>", "<pad>"])
def test_a_map_missing_a_frame_token_is_refused(missing):
    with pytest.raises(RewriteTokenError, match="missing"):
        RewriteTokens.from_token_map({k: v for k, v in FAKE_MAP.items() if k != missing})


def test_frame_tokens_must_be_distinct():
    with pytest.raises(RewriteTokenError, match="distinct"):
        RewriteTokens(style_id=5, open_id=5, close_id=6, eos_id=1, decoder_start_id=0)


@pytest.mark.skipif(not REAL_TOKEN_MAP.is_file(), reason="tokenizer not built")
def test_the_real_vocabulary_reserves_every_frame_token():
    """The whole premise of item 4: no vocabulary change, no embedding resize."""
    tokens = RewriteTokens.from_token_map(REAL_TOKEN_MAP)
    assert len({tokens.style_id, tokens.open_id, tokens.close_id, tokens.eos_id}) == 4
