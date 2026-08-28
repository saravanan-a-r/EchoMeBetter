"""
The seam between the UL2 objective and the model.

Both modules are deliberately decoupled — neither imports the other — which
means nothing inside either one can catch a mismatch at the boundary. That is
exactly why this file exists: it takes real UL2 output and feeds it to a real
model, unmodified.

The things that could silently disagree:

  - the batch key names and tensor layout
  - the label padding value (-100 in both, independently configured)
  - the decoder-start convention (`<pad>`, since there is no BOS)
  - sequence lengths against the model's context limits
  - the vocabulary size against the sentinel/mode token IDs

Skipped if the UL2 module is absent, so `model/` stays standalone.
"""

from __future__ import annotations

import random

import pytest
import torch

from conftest import REAL_TOKEN_MAP, UL2_SRC, load_ul2_module
from src.seq2seq import build_model

pytestmark = [
    pytest.mark.ul2,
    pytest.mark.skipif(not UL2_SRC.is_dir(), reason=f"UL2 module not present at {UL2_SRC}"),
]


@pytest.fixture(scope="module")
def ul2():
    return load_ul2_module()


@pytest.fixture
def specials(ul2, tiny_config):
    """
    A token table sized for the tiny model, keeping the real *shape*: 256
    sentinels would not fit a 256-token test vocabulary, so the count is
    scaled down while everything else matches.
    """
    return ul2.SpecialTokens(
        sentinel_ids=tuple(range(100, 132)),
        mode_token_ids={"R": 60, "X": 61, "S": 62},
        eos_id=tiny_config.eos_token_id,
        pad_id=tiny_config.pad_token_id,
        decoder_start_id=tiny_config.decoder_start_token_id,
        label_pad_id=tiny_config.label_pad_token_id,
    )


@pytest.fixture
def objective(ul2, tiny_config, specials):
    config = ul2.UL2Config(
        max_source_length=tiny_config.max_encoder_length,
        max_target_length=tiny_config.max_decoder_length,
        min_source_length=8,
    )
    return ul2.UL2Objective(config, specials)


def _batch(ul2, objective, specials, tiny_config, count=6, seed=0):
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        length = 12 + index * 5
        tokens = [rng.randrange(140, tiny_config.vocab_size) for _ in range(length)]
        examples.append(objective.corrupt(tokens, rng))
    padded = ul2.pad_batch(examples, specials)
    return {key: torch.tensor(value) for key, value in padded.items() if key != "modes"}


# -- the contract ---------------------------------------------------------


def test_the_label_pad_values_agree(ul2, specials, tiny_config):
    """
    Two modules configure this independently. If they ever disagree, the
    model would train on padding as if it were content — visible only as a
    suspiciously low loss.
    """
    assert specials.label_pad_id == tiny_config.label_pad_token_id == -100


def test_the_decoder_start_convention_agrees(specials, tiny_config):
    """No BOS token, so both sides must use <pad> to start the decoder."""
    assert specials.decoder_start_id == tiny_config.decoder_start_token_id
    assert specials.decoder_start_id == specials.pad_id


def test_ul2_batch_keys_match_the_model_signature(ul2, objective, specials, tiny_config):
    batch = _batch(ul2, objective, specials, tiny_config)
    assert set(batch) == {
        "encoder_input_ids",
        "encoder_attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    }


# -- end to end -----------------------------------------------------------


def test_a_real_ul2_batch_runs_through_the_model(
    ul2, objective, specials, tiny_config, tiny_model
):
    batch = _batch(ul2, objective, specials, tiny_config)
    output = tiny_model(**batch)

    assert output.logits.shape[:2] == batch["decoder_input_ids"].shape
    assert output.logits.shape[2] == tiny_config.vocab_size
    assert torch.isfinite(output.loss)


def test_a_real_ul2_batch_produces_gradients(ul2, objective, specials, tiny_config):
    model = build_model(tiny_config)
    model.train()
    model(**_batch(ul2, objective, specials, tiny_config)).loss.backward()

    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert dead == []


@pytest.mark.parametrize("mode", ["R", "X", "S"])
def test_every_denoising_mode_trains(ul2, objective, specials, tiny_config, mode):
    """
    All three objectives must produce a usable batch. An aggregate test could
    pass while one mode was silently broken — the same reason
    architecture.md §7.6 requires per-mode losses.
    """
    rng = random.Random(7)
    examples = [
        objective.corrupt(
            [rng.randrange(140, tiny_config.vocab_size) for _ in range(40)], rng, mode=mode
        )
        for _ in range(4)
    ]
    padded = ul2.pad_batch(examples, specials)
    batch = {k: torch.tensor(v) for k, v in padded.items() if k != "modes"}

    model = build_model(tiny_config)
    model.train()
    loss = model(**batch).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert model.shared.weight.grad.abs().sum() > 0


def test_ul2_sequence_lengths_fit_the_model_context(ul2, specials):
    """
    The two modules agree on 2048/2048 through separate config files. This
    checks the real UL2 worst cases against the real model limits.
    """
    from src.config import load_model_config

    from conftest import MODEL_CONFIG_PATH

    model_config = load_model_config(MODEL_CONFIG_PATH, profile="large")
    ul2_config = ul2.UL2Config()

    assert ul2_config.max_source_length == model_config.max_encoder_length
    assert ul2_config.max_target_length == model_config.max_decoder_length

    for mode in ("R", "X", "S"):
        encoder_length, _ = ul2_config.worst_case_encoder_length(mode)
        target_length, _ = ul2_config.worst_case_target_length(mode)
        assert encoder_length <= model_config.max_encoder_length, mode
        assert target_length <= model_config.max_decoder_length, mode


def test_label_positions_align_with_the_attention_mask(
    ul2, objective, specials, tiny_config
):
    """
    A position is either real in both, or padding in both. A misalignment
    would train the model on shifted targets.
    """
    batch = _batch(ul2, objective, specials, tiny_config)
    real_by_mask = batch["decoder_attention_mask"] == 1
    real_by_label = batch["labels"] != tiny_config.label_pad_token_id
    assert torch.equal(real_by_mask, real_by_label)


# -- against the real tokenizer -------------------------------------------


@pytest.mark.real_tokenizer
@pytest.mark.skipif(not REAL_TOKEN_MAP.is_file(), reason="trained tokenizer not present")
def test_the_real_vocabulary_fits_the_configured_model(ul2):
    """
    The tokenizer's real IDs must all be addressable by the model's
    embedding table. A mismatch here is a crash on the first batch of a
    multi-week run.
    """
    from src.config import load_model_config

    from conftest import MODEL_CONFIG_PATH

    model_config = load_model_config(MODEL_CONFIG_PATH, profile="large")
    specials = ul2.SpecialTokens.from_token_map(REAL_TOKEN_MAP)

    assert specials.num_sentinels == 256
    assert max(specials.sentinel_ids) < model_config.vocab_size
    for mode in ("R", "X", "S"):
        assert specials.mode_token(mode) < model_config.vocab_size

    assert specials.pad_id == model_config.pad_token_id
    assert specials.eos_id == model_config.eos_token_id
    assert specials.decoder_start_id == model_config.decoder_start_token_id
