"""
The token IDs the rewrite frame is built from, resolved by name.

Same boundary `UL2/src/special_tokens.py` draws: this module is the whole
dependency the task has on the tokenizer. IDs are looked up by name in
`token_map.json` (architecture.md 6.4 forbids hardcoding them), and every
token used here was reserved in the frozen vocabulary at tokenizer-training
time (`tokenizer/training/specials.py`: `STYLE_TOKENS`, `STRUCTURAL_TOKENS`)
-- which is what lets this task be added to pretraining without an embedding
resize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import RewriteTokenError

DEFAULT_STYLE_TOKEN = "<style:grammar>"
OPEN_TOKEN = "<text_to_rewrite>"
CLOSE_TOKEN = "</text_to_rewrite>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"

# Tokens the frame adds around the corrupted text: style, open, close, EOS.
FRAME_TOKENS = 4


@dataclass(frozen=True)
class RewriteTokens:
    """
    The frame `<style> <text_to_rewrite> ... </text_to_rewrite> </s>`, as IDs.

    Frozen for the reason `SpecialTokens` is: these are welded to the
    embedding table of every checkpoint trained with them.
    """

    style_id: int
    open_id: int
    close_id: int
    eos_id: int
    decoder_start_id: int
    style_token: str = DEFAULT_STYLE_TOKEN

    def __post_init__(self) -> None:
        ids = {
            "style_id": self.style_id,
            "open_id": self.open_id,
            "close_id": self.close_id,
            "eos_id": self.eos_id,
            "decoder_start_id": self.decoder_start_id,
        }
        for name, value in ids.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RewriteTokenError(f"{name} must be a real vocabulary ID, got {value!r}")
        frame = (self.style_id, self.open_id, self.close_id, self.eos_id)
        if len(set(frame)) != len(frame):
            raise RewriteTokenError(
                "style, open, close and EOS must be distinct IDs; a frame the model "
                "cannot parse teaches it nothing about where the text to rewrite is"
            )

    @classmethod
    def from_token_map(
        cls,
        token_map: Mapping[str, int] | str | Path,
        *,
        style_token: str = DEFAULT_STYLE_TOKEN,
    ) -> "RewriteTokens":
        """
        Resolve the frame by name. `decoder_start_id` is `<pad>`, the T5
        convention `SpecialTokens.from_token_map` also follows (no BOS token).
        """
        mapping = _load_token_map(token_map)

        def require(name: str) -> int:
            if name not in mapping:
                raise RewriteTokenError(
                    f"token map is missing {name!r}; it was not produced by the "
                    f"tokenizer this task is configured for"
                )
            value = mapping[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise RewriteTokenError(f"token map entry {name!r} is not an integer: {value!r}")
            return value

        return cls(
            style_id=require(style_token),
            open_id=require(OPEN_TOKEN),
            close_id=require(CLOSE_TOKEN),
            eos_id=require(EOS_TOKEN),
            decoder_start_id=require(PAD_TOKEN),
            style_token=style_token,
        )


def _load_token_map(source: Mapping[str, int] | str | Path) -> Mapping[str, int]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise RewriteTokenError(f"token map not found: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RewriteTokenError(f"token map {path} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise RewriteTokenError(f"token map {path} must contain a JSON object")
        return loaded
    if isinstance(source, Mapping):
        return source
    raise RewriteTokenError(f"unsupported token map source: {type(source).__name__}")


__all__ = [
    "CLOSE_TOKEN",
    "DEFAULT_STYLE_TOKEN",
    "FRAME_TOKENS",
    "OPEN_TOKEN",
    "RewriteTokens",
]
