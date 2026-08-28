"""
Shared fixtures for the model test suite.

Almost every test runs against a *tiny* configuration — 2+2 layers, 64
dimensions, 256-token vocabulary — because the properties under test
(mask correctness, cache equivalence, gradient flow, residual structure) are
size-independent, and a suite that allocates 750M parameters per test is a
suite nobody runs.

The full-size profiles are exercised where size is the actual subject: the
parameter-budget tests, which check the configuration formula rather than
building the model, and a small number of tests marked `slow`.

The tiny config keeps the *shape relationships* of the real one —
`num_heads x d_kv == d_model`, `d_ff ~= 2.75 x d_model` — so it is a
scale model rather than an unrelated one.

Field names follow `transformers.T5Config` wherever one exists; see
`src/config.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, load_model_config  # noqa: E402
from src.seq2seq import RephraseSeq2Seq, build_model  # noqa: E402

PROJECT_ROOT = ROOT.parent
MODEL_CONFIG_PATH = PROJECT_ROOT / "model_config.yml"
UL2_SRC = PROJECT_ROOT / "UL2" / "src"
REAL_TOKEN_MAP = PROJECT_ROOT / "tokenizer" / "training" / "output" / "token_map.json"

TINY = dict(
    d_model=64,
    d_ff=176,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_kv=16,
    vocab_size=256,
    max_encoder_length=64,
    max_decoder_length=64,
    relative_attention_num_buckets=8,
    relative_attention_max_distance=16,
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
    tie_word_embeddings=True,
    scale_output_for_tied_embeddings=True,
    scale_attention_scores=False,
    dropout_rate=0.0,
    label_smoothing=0.0,
    initializer_factor=1.0,
    pad_token_id=0,
    decoder_start_token_id=0,
    eos_token_id=1,
    label_pad_token_id=-100,
    profile="tiny",
    expected_parameters=None,
)

# IDs below this are reserved (pad/eos); test data starts above them so a
# generated pad or eos is always meaningful rather than coincidental.
FIRST_CONTENT_ID = 8


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(**TINY)


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> RephraseSeq2Seq:
    torch.manual_seed(0)
    model = build_model(tiny_config)
    model.eval()  # dropout is 0.0 anyway; eval mode makes that explicit
    return model


@pytest.fixture
def large_config() -> ModelConfig:
    return load_model_config(MODEL_CONFIG_PATH, profile="large")


@pytest.fixture
def base_config() -> ModelConfig:
    return load_model_config(MODEL_CONFIG_PATH, profile="base")


def make_batch(
    config: ModelConfig,
    batch_size: int = 3,
    source_length: int = 16,
    target_length: int = 12,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """
    A well-formed training batch with realistic ragged padding.

    Row 0 has a short source, row 1 a short target, so both padding paths are
    exercised by every test that uses this.
    """
    generator = torch.Generator().manual_seed(seed)

    encoder_input_ids = torch.randint(
        FIRST_CONTENT_ID, config.vocab_size, (batch_size, source_length), generator=generator
    )
    target = torch.randint(
        FIRST_CONTENT_ID, config.vocab_size, (batch_size, target_length), generator=generator
    )

    encoder_attention_mask = torch.ones(batch_size, source_length, dtype=torch.long)
    decoder_attention_mask = torch.ones(batch_size, target_length, dtype=torch.long)
    if batch_size > 1 and source_length > 4:
        encoder_attention_mask[0, source_length - 3:] = 0
        encoder_input_ids[0, source_length - 3:] = config.pad_token_id
    if batch_size > 1 and target_length > 4:
        decoder_attention_mask[1, target_length - 2:] = 0

    # Teacher forcing: the decoder input is the target shifted right behind
    # the start token, so predicting position i never means copying it.
    decoder_input_ids = torch.cat(
        [
            torch.full((batch_size, 1), config.decoder_start_token_id, dtype=torch.long),
            target[:, :-1],
        ],
        dim=1,
    )
    labels = target.masked_fill(decoder_attention_mask == 0, config.label_pad_token_id)

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }


def load_ul2_module():
    """
    Import the sibling UL2 package under an unambiguous name.

    Both packages are directories called `src`, so a plain import would
    collide. Loading UL2 explicitly by path, registered as `ul2`, keeps the
    two independent — which is the point of them being decoupled modules.
    """
    if "ul2" in sys.modules:
        return sys.modules["ul2"]
    spec = importlib.util.spec_from_file_location(
        "ul2", UL2_SRC / "__init__.py", submodule_search_locations=[str(UL2_SRC)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load UL2 from {UL2_SRC}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ul2"] = module
    spec.loader.exec_module(module)
    return module
