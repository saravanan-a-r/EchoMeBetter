"""
UL2 mixture-of-denoisers objective (architecture.md, Stage 1).

Scope
-----
This package implements the *objective* only: given a sequence of token IDs,
produce the corrupted encoder input and the decoder target for one of the
three UL2 denoising modes. It does not tokenize, does not read a corpus, does
not define a model, and does not train anything. Those belong to the
surrounding pretraining pipeline.

Decoupling
----------
Zero third-party dependencies -- standard library only. The one point of
contact with the rest of the project is `SpecialTokens.from_token_map()`,
which resolves token IDs *by name* from the tokenizer's `token_map.json`
(architecture.md 6.4 forbids hardcoded IDs). Nothing here imports from
`tokenizer/`.

Typical use
-----------
    import random
    from src import UL2Config, UL2Objective, SpecialTokens, MixtureTelemetry

    specials = SpecialTokens.from_token_map("tokenizer/training/output/token_map.json")
    telemetry = MixtureTelemetry(UL2Config().mode_weights)
    objective = UL2Objective(UL2Config(), specials, telemetry)

    rng = random.Random(42)
    for token_ids in stream_of_tokenized_lines:
        example = objective.try_corrupt(token_ids, rng)
        if example is not None:
            ...            # feed to the trainer
    print(telemetry.format_table())
"""

from __future__ import annotations

from .batching import pad_batch
from .config import (
    DEFAULT_MODE_WEIGHTS,
    FROZEN_EVAL_MODE_WEIGHTS,
    PrefixConfig,
    SpanCorruptionConfig,
    UL2Config,
)
from .denoise import (
    Example,
    build_prefix_denoising,
    build_span_corruption,
    reconstruct_source,
)
from .errors import (
    ConfigError,
    FrozenEvalError,
    SentinelBudgetExceededError,
    SourceTooLongError,
    SourceTooShortError,
    SpecialTokenError,
    TargetTooLongError,
    UL2Error,
)
from .frozen_eval import FrozenEvalSet, build_frozen_eval_set
from .mixture import ModeSampler, deterministic_schedule
from .objective import UL2Objective
from .spans import plan_span_counts, random_segmentation, sample_prefix_split, sample_spans
from .special_tokens import MODES, SpecialTokens
from .telemetry import MixtureTelemetry, ModeCounters

__all__ = [
    # configuration
    "UL2Config",
    "SpanCorruptionConfig",
    "PrefixConfig",
    "DEFAULT_MODE_WEIGHTS",
    "FROZEN_EVAL_MODE_WEIGHTS",
    # tokens
    "SpecialTokens",
    "MODES",
    # objective
    "UL2Objective",
    "Example",
    "build_span_corruption",
    "build_prefix_denoising",
    "reconstruct_source",
    # sampling internals (exposed for testing and diagnostics)
    "sample_spans",
    "sample_prefix_split",
    "plan_span_counts",
    "random_segmentation",
    "ModeSampler",
    "deterministic_schedule",
    # batching, measurement, evaluation
    "pad_batch",
    "MixtureTelemetry",
    "ModeCounters",
    "FrozenEvalSet",
    "build_frozen_eval_set",
    # errors
    "UL2Error",
    "ConfigError",
    "SpecialTokenError",
    "SourceTooShortError",
    "SourceTooLongError",
    "SentinelBudgetExceededError",
    "TargetTooLongError",
    "FrozenEvalError",
]
