"""
The token-ID table this module needs, and nothing else.

Decoupling boundary
-------------------
This module is the *entire* dependency the UL2 objective has on the
tokenizer. The objective operates on integer token IDs and never sees text,
a SentencePiece model, or any code from `tokenizer/`. That makes the
objective testable with invented IDs and re-usable with any tokenizer, while
keeping exactly one place where the real vocabulary is bound.

architecture.md 6.4 forbids hardcoding token IDs anywhere downstream --
SentencePiece's internal ordering is an implementation detail. So IDs are
always resolved *by name* from `token_map.json`, which is what
`from_token_map()` does. There is no code path in this module that assumes
`<extra_id_0> == 3`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import SpecialTokenError

# The three UL2 denoising modes (architecture.md 7.2). Used as dictionary
# keys throughout; the *token* they map to is resolved from the token map.
MODES: tuple[str, ...] = ("R", "X", "S")

# Names as they appear in token_map.json.
MODE_TOKEN_NAMES: Mapping[str, str] = {"R": "[R]", "X": "[X]", "S": "[S]"}
SENTINEL_NAME_TEMPLATE = "<extra_id_{}>"
EOS_NAME = "</s>"
PAD_NAME = "<pad>"

# architecture.md 6.2: the tokenizer reserves exactly this many sentinels.
DEFAULT_NUM_SENTINELS = 256

# PyTorch's cross-entropy ignore_index. Positions marked with this are
# excluded from the loss -- which is how padded target positions avoid
# teaching the model to predict padding.
DEFAULT_LABEL_PAD_ID = -100


@dataclass(frozen=True)
class SpecialTokens:
    """
    Resolved integer IDs for every special token the objective emits.

    Frozen: these IDs are welded to the embedding table of every checkpoint
    trained with them. A mutable table would allow a mid-run change that
    silently invalidates every checkpoint written before it.
    """

    sentinel_ids: tuple[int, ...]
    mode_token_ids: Mapping[str, int]
    eos_id: int
    pad_id: int
    decoder_start_id: int
    label_pad_id: int = DEFAULT_LABEL_PAD_ID

    def __post_init__(self) -> None:
        if not self.sentinel_ids:
            raise SpecialTokenError("sentinel_ids is empty; span corruption is impossible")

        if len(set(self.sentinel_ids)) != len(self.sentinel_ids):
            raise SpecialTokenError(
                "duplicate sentinel IDs: two spans would collapse onto one marker, "
                "making the decoder target ambiguous"
            )

        missing = [m for m in MODES if m not in self.mode_token_ids]
        if missing:
            raise SpecialTokenError(f"missing mode token IDs for: {missing}")

        if len(set(self.mode_token_ids[m] for m in MODES)) != len(MODES):
            raise SpecialTokenError(
                "mode tokens [R]/[X]/[S] must have distinct IDs; the model cannot "
                "learn different behaviours from an ambiguous prefix"
            )

        for name, value in (
            ("eos_id", self.eos_id),
            ("pad_id", self.pad_id),
            ("decoder_start_id", self.decoder_start_id),
        ):
            if value < 0:
                raise SpecialTokenError(f"{name} must be a real vocabulary ID, got {value}")

        for value in self.sentinel_ids:
            if value < 0:
                raise SpecialTokenError(f"sentinel IDs must be non-negative, got {value}")

        # A sentinel that collides with EOS would terminate the target early;
        # one that collides with a mode token would be unparseable.
        reserved = {self.eos_id, *(self.mode_token_ids[m] for m in MODES)}
        clashes = reserved.intersection(self.sentinel_ids)
        if clashes:
            raise SpecialTokenError(
                f"sentinel IDs collide with eos/mode token IDs: {sorted(clashes)}"
            )

        if self.label_pad_id >= 0:
            raise SpecialTokenError(
                f"label_pad_id must be negative so it can never collide with a real "
                f"vocabulary ID, got {self.label_pad_id}"
            )

    @property
    def num_sentinels(self) -> int:
        return len(self.sentinel_ids)

    def sentinel(self, index: int) -> int:
        """
        ID of the `index`-th sentinel.

        Raises rather than wrapping: architecture.md 7.4 requires a loud
        failure, and a wrapped index would silently reuse a marker.
        """
        if not 0 <= index < len(self.sentinel_ids):
            raise SpecialTokenError(
                f"sentinel index {index} out of range (have {len(self.sentinel_ids)}); "
                f"this must never be clamped or wrapped"
            )
        return self.sentinel_ids[index]

    def mode_token(self, mode: str) -> int:
        try:
            return self.mode_token_ids[mode]
        except KeyError:
            raise SpecialTokenError(f"unknown mode {mode!r}; expected one of {MODES}") from None

    @classmethod
    def from_token_map(
        cls,
        token_map: Mapping[str, int] | str | Path,
        *,
        num_sentinels: int = DEFAULT_NUM_SENTINELS,
        label_pad_id: int = DEFAULT_LABEL_PAD_ID,
    ) -> "SpecialTokens":
        """
        Resolve IDs by name from `token_map.json` (or an equivalent mapping).

        `decoder_start_id` is `<pad>`: the model has no BOS token
        (`bos_id=-1`, architecture.md 6.1), and using pad as the decoder start
        is the standard T5 convention.
        """
        mapping = _load_token_map(token_map)

        def require(name: str) -> int:
            if name not in mapping:
                raise SpecialTokenError(
                    f"token map is missing {name!r}; it was not produced by the "
                    f"tokenizer this objective is configured for"
                )
            value = mapping[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise SpecialTokenError(f"token map entry {name!r} is not an integer: {value!r}")
            return value

        if num_sentinels <= 0:
            raise SpecialTokenError(f"num_sentinels must be positive, got {num_sentinels}")

        sentinels = tuple(require(SENTINEL_NAME_TEMPLATE.format(i)) for i in range(num_sentinels))
        modes = {mode: require(name) for mode, name in MODE_TOKEN_NAMES.items()}
        pad_id = require(PAD_NAME)

        return cls(
            sentinel_ids=sentinels,
            mode_token_ids=modes,
            eos_id=require(EOS_NAME),
            pad_id=pad_id,
            decoder_start_id=pad_id,
            label_pad_id=label_pad_id,
        )


def _load_token_map(source: Mapping[str, int] | str | Path) -> Mapping[str, int]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise SpecialTokenError(f"token map not found: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SpecialTokenError(f"token map {path} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SpecialTokenError(f"token map {path} must contain a JSON object")
        return loaded
    if isinstance(source, Mapping):
        return source
    raise SpecialTokenError(f"unsupported token map source: {type(source).__name__}")
