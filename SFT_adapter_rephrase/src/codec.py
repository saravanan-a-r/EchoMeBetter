"""
Text <-> token IDs, and the product frame, in one object.

Encoding goes through `corpus/`'s `encode_text` -- the one way this project
tokenizes, marker escaping included -- so an SFT target is tokenized exactly
as pretraining tokenized the same text. The frame comes from `rewrite/`'s
`RewriteTokens`, so an SFT source is framed exactly as the cooldown's
synthetic rewrite task frames it:

    encoder:  <style:X> <text_to_rewrite> input... </text_to_rewrite> </s>   (X = data.style)
    target:   output... </s>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .errors import SFTConfigError
from .interop import RewriteTokens, tokenizer_interop

UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class Frame:
    """The IDs a framed example is assembled from."""

    style_id: int
    open_id: int
    close_id: int
    eos_id: int
    pad_id: int
    decoder_start_id: int

    def source(self, text_ids: Sequence[int]) -> list[int]:
        return [self.style_id, self.open_id, *text_ids, self.close_id, self.eos_id]

    def target(self, text_ids: Sequence[int]) -> list[int]:
        return [*text_ids, self.eos_id]

    # Tokens the frame adds to each side.
    SOURCE_OVERHEAD = 4
    TARGET_OVERHEAD = 1


class TextCodec:
    """Tokenizer, token map and frame for one style."""

    def __init__(
        self,
        encode: Callable[[str], list[int]],
        decode: Callable[[Sequence[int]], str],
        token_map: Mapping[str, int],
        style_token: str,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self.token_map = dict(token_map)
        if style_token not in self.token_map:
            raise SFTConfigError(f"style token {style_token!r} is not in the tokenizer's token map")
        frame_tokens = RewriteTokens.from_token_map(self.token_map, style_token=style_token)
        self.frame = Frame(
            style_id=frame_tokens.style_id,
            open_id=frame_tokens.open_id,
            close_id=frame_tokens.close_id,
            eos_id=frame_tokens.eos_id,
            pad_id=self.token_map["<pad>"],
            decoder_start_id=frame_tokens.decoder_start_id,
        )
        # Every control token (anything that is not an ordinary `▁`-piece).
        # A record whose own text encodes to one of these -- a comment that
        # literally contains "<text_to_rewrite>" -- would corrupt the frame.
        self.control_ids = frozenset(
            token_id
            for token, token_id in self.token_map.items()
            if not token.startswith("▁") and token != UNK_TOKEN
        )

    @classmethod
    def from_tokenizer_dir(cls, tokenizer_dir: str | Path, style_token: str) -> "TextCodec":
        tokenizer_dir = Path(tokenizer_dir)
        processor = tokenizer_interop.load_processor(tokenizer_dir / "spm.model")
        token_map = json.loads((tokenizer_dir / "token_map.json").read_text(encoding="utf-8"))

        def encode(text: str) -> list[int]:
            return tokenizer_interop.encode_text(processor, text)

        def decode(ids: Sequence[int]) -> str:
            return tokenizer_interop.unescape_markers(processor.decode(list(ids)))

        return cls(encode, decode, token_map, style_token)

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        """Decode generated IDs, stopping at EOS and skipping padding."""
        kept: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.frame.eos_id:
                break
            if token_id != self.frame.pad_id:
                kept.append(token_id)
        return self._decode(kept)

    def contains_control(self, ids: Sequence[int]) -> bool:
        return any(token_id in self.control_ids for token_id in ids)
