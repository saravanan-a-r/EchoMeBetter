"""
The synthetic rewrite-shaped pretraining task (architecture_improvements.md
item 4).

Scope
-----
Given a text document, produce the pair the product task is shaped like: a
cheaply corrupted copy on the encoder side, framed by the same style and
structural tokens SFT will use, and the original on the decoder side. It
runs during the WSD cooldown only (`training_config.yml`: `rewrite_share`),
as a share of the micro-batches drawn there.

It is **not** a UL2 denoiser: `UL2/`'s three modes mask or split a document
and are sampled from `mode_weights`; this task leaves the document whole and
is mixed in one level up, in `pretrain.py`'s `PretrainBatchSource`. Nothing
in `UL2/` knows it exists.

Decoupling
----------
Standard library only. Text comes in through a `next_record()` stream
(`corpus/`), token IDs go out through an `encode` callable the caller
supplies, and the frame's IDs are resolved by name from `token_map.json`
(`RewriteTokens.from_token_map`). Nothing here imports the tokenizer, the
corpus or the objective.

Typical use
-----------
    from src import Corruptor, RewriteExampleSource, RewriteTokens

    tokens = RewriteTokens.from_token_map("tokenizer/training/output/token_map.json")
    source = RewriteExampleSource(
        records, Corruptor(), encode, tokens,
        max_source_length=2048, max_target_length=2048, min_source_length=8, seed=1234,
    )
    pair = source.next_pair()      # framed encoder IDs, target IDs
"""

from __future__ import annotations

from .corruption import (
    DEFAULT_OPERATOR_WEIGHTS,
    DEFAULT_PROFILES,
    IDENTITY,
    OPERATORS,
    Corruption,
    CorruptionConfig,
    Corruptor,
    Profile,
)
from .errors import RewriteConfigError, RewriteError, RewriteTokenError
from .source import REWRITE_MODE, RewriteExampleSource, RewritePair
from .text import count_prose_words, is_prose_line, protected_spans, split_line
from .tokens import (
    CLOSE_TOKEN,
    DEFAULT_STYLE_TOKEN,
    FRAME_TOKENS,
    OPEN_TOKEN,
    RewriteTokens,
)

__all__ = [
    # the task
    "RewriteExampleSource",
    "RewritePair",
    "REWRITE_MODE",
    # corruption
    "Corruptor",
    "CorruptionConfig",
    "Corruption",
    "Profile",
    "DEFAULT_PROFILES",
    "DEFAULT_OPERATOR_WEIGHTS",
    "OPERATORS",
    "IDENTITY",
    # text segmentation (exposed for tests and diagnostics)
    "split_line",
    "is_prose_line",
    "count_prose_words",
    "protected_spans",
    # tokens
    "RewriteTokens",
    "DEFAULT_STYLE_TOKEN",
    "OPEN_TOKEN",
    "CLOSE_TOKEN",
    "FRAME_TOKENS",
    # errors
    "RewriteError",
    "RewriteConfigError",
    "RewriteTokenError",
]
