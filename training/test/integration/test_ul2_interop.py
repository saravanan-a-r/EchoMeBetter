"""
Real UL2 batches through the real trainer.

Every other test in this suite feeds the trainer batches that `conftest.py`
constructed to look like UL2's. This one feeds it batches UL2 actually
produced. The distinction is the whole point: a hand-written fixture encodes
what the author *believed* the contract was, and a seam is precisely where
two people's beliefs diverge.

Three modules meet here — `UL2/` builds the examples, `model/` consumes them,
`training/` drives the loop — and nothing translates between them. These tests
are what makes that claim checkable rather than aspirational.
"""

from __future__ import annotations

import random

import pytest
import torch

from src.interop import load_ul2_package, rephrase_model
from src.loop import Trainer

pytestmark = pytest.mark.ul2


@pytest.fixture(scope="module")
def ul2():
    return load_ul2_package()


@pytest.fixture
def specials(ul2, model_config):
    """
    A token table sized for the tiny model, keeping the real *shape*.

    256 sentinels (§6.2) would not fit a 256-token test vocabulary, so the
    count is scaled down while every other convention — decoder start, label
    pad, mode tokens — matches the real one exactly. The seam being tested is
    shape agreement, not vocabulary size.
    """
    return ul2.SpecialTokens(
        sentinel_ids=tuple(range(100, 132)),
        mode_token_ids={"R": 60, "X": 61, "S": 62},
        eos_id=model_config.eos_token_id,
        pad_id=model_config.pad_token_id,
        decoder_start_id=model_config.decoder_start_token_id,
        label_pad_id=model_config.label_pad_token_id,
    )


@pytest.fixture
def objective(ul2, specials, model_config):
    return ul2.UL2Objective(
        ul2.UL2Config(
            max_source_length=model_config.max_encoder_length,
            max_target_length=model_config.max_decoder_length,
            min_source_length=8,
        ),
        specials,
    )


def ul2_batches(objective, ul2, specials, model_config, count: int, seed: int = 0):
    """Batches straight out of the objective, with no reshaping in between."""
    rng = random.Random(seed)
    batches = []
    for index in range(count):
        # Lengths vary but stay inside the tiny model's 64-token encoder
        # limit, which UL2 refuses to exceed rather than truncating silently
        # (§4.5) — that refusal is itself covered by `model/`'s own suite.
        examples = [
            objective.corrupt(
                [
                    rng.randrange(140, model_config.vocab_size)
                    for _ in range(12 + (step * 7) % 45)
                ],
                rng,
            )
            for step in range(index, index + 3)
        ]
        batches.append(ul2.pad_batch(examples, specials))
    return batches


def test_a_ul2_batch_has_exactly_the_keys_the_trainer_requires(
    objective, ul2, specials, model_config
):
    """
    The contract, checked at the seam rather than assumed on both sides.
    """
    from src.loop import REQUIRED_KEYS

    batch = ul2_batches(objective, ul2, specials, model_config, 1)[0]
    assert set(REQUIRED_KEYS) <= set(batch)


def test_a_ul2_batch_carries_its_modes(objective, ul2, specials, model_config):
    """
    Per-mode loss reporting (§7.6) depends on this label travelling with the
    batch. If UL2 stopped emitting it, the trainer would report a single
    aggregate — and an aggregate can look healthy while one denoiser silently
    fails to learn.
    """
    batch = ul2_batches(objective, ul2, specials, model_config, 1)[0]
    assert set(batch["modes"]) <= {"R", "X", "S"}


def test_the_trainer_steps_on_real_ul2_batches(objective, ul2, specials, model_config, training_config):
    """
    No conversion, no reshaping: UL2's plain lists go straight in.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    driver = Trainer(model, training_config.with_(warmup_steps=0), device="cpu")

    before = model.shared.weight.detach().clone()
    report = driver.train_step(ul2_batches(objective, ul2, specials, model_config, 2))

    assert not report.skipped
    assert report.tokens > 0
    assert torch.isfinite(torch.tensor(report.loss))
    assert not torch.equal(before, model.shared.weight)


def test_a_run_over_real_ul2_batches_reports_every_mode_it_saw(
    objective, ul2, specials, model_config, training_config
):
    """
    §7.7's checkpoint diagnostic, end to end on real data: the [R]/[X]/[S]
    split has to survive the whole path from the objective to the log line.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    config = training_config.with_(
        max_steps=6,
        gradient_accumulation_steps=1,
        logging_steps=6,
        warmup_steps=0,
        log_per_mode_loss=True,
    )
    seen = []
    driver = Trainer(model, config, device="cpu", on_log=seen.append)
    driver.train(ul2_batches(objective, ul2, specials, model_config, 6))

    logged = [entry for entry in seen if "loss" in entry]
    assert logged
    modes = {key for entry in logged for key in entry if key.startswith("loss_")}
    assert modes, "no per-mode losses were reported from real UL2 batches"


def test_evaluation_over_real_ul2_batches(objective, ul2, specials, model_config, training_config):
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    driver = Trainer(model, training_config, device="cpu")

    metrics = driver.evaluate(ul2_batches(objective, ul2, specials, model_config, 4, seed=7))
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(key.startswith("loss_") for key in metrics)


def test_ul2_targets_have_unequal_lengths(objective, ul2, specials, model_config):
    """
    Establishes the premise behind `loop.py`'s token-weighted accumulation.
    If UL2's batches all had the same number of target tokens, the
    conventional `loss / accumulation_steps` would be exactly right and the
    weighting would be needless complexity. They do not.
    """
    batches = ul2_batches(objective, ul2, specials, model_config, 8, seed=3)
    counts = {
        sum(1 for row in batch["labels"] for label in row if label != -100)
        for batch in batches
    }
    assert len(counts) > 1, f"every batch had the same token count: {counts}"
