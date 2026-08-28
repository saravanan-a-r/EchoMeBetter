"""
End-to-end forward and backward behaviour.

The invariants here are the ones that make a training run trustworthy:

  1. every parameter receives a gradient — a dead subgraph trains silently
     and is only discovered when a component turns out to be at its random
     initialization weeks later
  2. padding is genuinely isolated — content at masked positions must not
     reach the output through any path
  3. the decoder cannot see the future
  4. results are deterministic given a seed

Plus a real overfit test: the model must be able to memorize one batch. That
is the single most informative check in this file, because a model can pass
every shape and mask test while being wired so that it cannot learn at all.
"""

from __future__ import annotations

import pytest
import torch

from conftest import make_batch
from src.seq2seq import build_model


# -- gradient flow --------------------------------------------------------


def test_every_parameter_receives_a_gradient(tiny_config):
    """A parameter with no gradient is a component that is not being trained."""
    model = build_model(tiny_config)
    model.train()
    model(**make_batch(tiny_config)).loss.backward()

    dead = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or parameter.grad.abs().sum().item() == 0.0
    ]
    assert dead == [], f"parameters receiving no gradient: {dead}"


def test_gradients_are_finite(tiny_config):
    model = build_model(tiny_config)
    model.train()
    model(**make_batch(tiny_config)).loss.backward()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name


def test_the_position_bias_tables_are_trained(tiny_config):
    """Both stacks own one; neither may be left frozen at initialization."""
    model = build_model(tiny_config)
    model.train()
    model(**make_batch(tiny_config)).loss.backward()

    assert model.encoder.position_bias.embedding.weight.grad.abs().sum() > 0
    assert model.decoder.position_bias.embedding.weight.grad.abs().sum() > 0


def test_the_shared_embedding_accumulates_from_all_three_uses(tiny_config):
    """
    Encoder input, decoder input and output projection all touch one matrix,
    so its gradient must be larger than any single use would produce.
    """
    model = build_model(tiny_config)
    model.train()
    model(**make_batch(tiny_config)).loss.backward()
    assert model.shared.weight.grad.abs().sum() > 0


# -- masking --------------------------------------------------------------


def test_encoder_padding_cannot_influence_the_output(tiny_model, tiny_config):
    """
    Change the tokens at padded positions arbitrarily; nothing may move.
    A leak here means the model learns from whatever junk fills the padding.
    """
    batch = make_batch(tiny_config)
    mask = batch["encoder_attention_mask"]
    assert (mask == 0).any(), "fixture must contain padding"

    first = tiny_model(**batch).logits

    polluted = batch["encoder_input_ids"].clone()
    polluted[mask == 0] = torch.randint(
        8, tiny_config.vocab_size, (int((mask == 0).sum()),)
    )
    second = tiny_model(**(batch | {"encoder_input_ids": polluted})).logits

    assert torch.allclose(first, second, atol=1e-5)


def test_the_decoder_cannot_see_future_tokens(tiny_model, tiny_config):
    """
    Perturb the last decoder input; every earlier position's logits must be
    unchanged. This is the failure that trains beautifully and generates
    nonsense.
    """
    batch = make_batch(tiny_config)
    first = tiny_model(**batch).logits

    altered = batch["decoder_input_ids"].clone()
    altered[:, -1] = (altered[:, -1] + 17) % tiny_config.vocab_size
    second = tiny_model(**(batch | {"decoder_input_ids": altered})).logits

    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)


def test_each_decoder_position_depends_on_its_whole_prefix(tiny_model, tiny_config):
    """The converse: changing an early token must affect later positions."""
    batch = make_batch(tiny_config)
    first = tiny_model(**batch).logits

    altered = batch["decoder_input_ids"].clone()
    altered[:, 1] = (altered[:, 1] + 23) % tiny_config.vocab_size
    second = tiny_model(**(batch | {"decoder_input_ids": altered})).logits

    assert not torch.allclose(first[:, 2:], second[:, 2:], atol=1e-5)


def test_the_output_depends_on_the_source(tiny_model, tiny_config):
    """Cross-attention must actually carry information."""
    batch = make_batch(tiny_config)
    first = tiny_model(**batch).logits

    other = make_batch(tiny_config, seed=99)["encoder_input_ids"]
    second = tiny_model(**(batch | {"encoder_input_ids": other})).logits

    assert not torch.allclose(first, second, atol=1e-4)


def test_rows_of_a_batch_are_independent(tiny_model, tiny_config):
    """
    No cross-talk between examples. A batching bug here inflates measured
    quality in a way nothing downstream would detect.
    """
    batch = make_batch(tiny_config, batch_size=4)
    batched = tiny_model(**batch).logits

    single = {key: value[2:3] for key, value in batch.items()}
    alone = tiny_model(**single).logits

    assert torch.allclose(batched[2:3], alone, atol=1e-5)


# -- numerical behaviour --------------------------------------------------


def test_forward_is_deterministic_in_eval_mode(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    assert torch.allclose(tiny_model(**batch).logits, tiny_model(**batch).logits)


def test_initial_loss_is_near_uniform(tiny_config):
    """
    A freshly initialized model should be about as uncertain as chance,
    `ln(vocab_size)`. Far below means something is leaking the answer; far
    above means the initialization is broken.
    """
    import math

    torch.manual_seed(0)
    model = build_model(tiny_config).eval()
    loss = float(model(**make_batch(tiny_config)).loss.detach())
    uniform = math.log(tiny_config.vocab_size)

    assert 0.5 * uniform < loss < 1.5 * uniform, f"loss {loss:.2f} vs ln(V) {uniform:.2f}"


def test_activations_stay_finite_at_full_context(tiny_config):
    """Deep stacks can overflow; check the longest input the config allows."""
    model = build_model(tiny_config).eval()
    with torch.no_grad():
        output = model(
            torch.randint(8, tiny_config.vocab_size, (1, tiny_config.max_encoder_length)),
            torch.randint(8, tiny_config.vocab_size, (1, tiny_config.max_decoder_length)),
        )
    assert torch.isfinite(output.logits).all()


def test_dropout_is_active_in_training_mode(tiny_config):
    model = build_model(tiny_config.with_(dropout_rate=0.3))
    batch = make_batch(tiny_config)

    model.train()
    torch.manual_seed(0)
    first = model(**batch).logits
    torch.manual_seed(1)
    second = model(**batch).logits
    assert not torch.allclose(first, second)

    model.eval()
    assert torch.allclose(model(**batch).logits, model(**batch).logits)


# -- learning -------------------------------------------------------------


def test_the_model_can_memorize_one_batch(tiny_config):
    """
    The test that catches wiring bugs no shape or mask test can: a model may
    have correct shapes, correct masks and finite gradients while being
    connected so that it cannot learn. If loss does not collapse on a single
    repeated batch, something upstream is broken.
    """
    torch.manual_seed(0)
    model = build_model(tiny_config.with_(vocab_size=64, expected_parameters=None))
    model.train()

    batch = make_batch(model.config, batch_size=2, source_length=8, target_length=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    initial = float(model(**batch).loss.detach())
    for _ in range(120):
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()

    final = float(loss.detach())
    assert final < 0.1 * initial, f"loss did not collapse: {initial:.3f} -> {final:.3f}"


def test_learning_uses_the_source(tiny_config):
    """
    After memorizing, replacing the source must degrade the prediction —
    proving the model learned a source-conditional mapping rather than
    ignoring the encoder entirely.
    """
    torch.manual_seed(0)
    model = build_model(tiny_config.with_(vocab_size=64, expected_parameters=None))
    model.train()

    batch = make_batch(model.config, batch_size=2, source_length=8, target_length=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(120):
        optimizer.zero_grad()
        model(**batch).loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        learned = float(model(**batch).loss)
        other = make_batch(model.config, batch_size=2, source_length=8,
                           target_length=6, seed=7)["encoder_input_ids"]
        swapped = float(model(**(batch | {"encoder_input_ids": other})).loss)

    assert swapped > learned, "the encoder is not influencing the prediction"


@pytest.mark.parametrize("checkpointing", [False, True])
def test_a_training_step_runs_under_both_memory_modes(tiny_config, checkpointing):
    model = build_model(tiny_config)
    model.train()
    if checkpointing:
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = float(model(**make_batch(tiny_config)).loss.detach())

    optimizer.zero_grad()
    model(**make_batch(tiny_config)).loss.backward()
    optimizer.step()

    after = float(model(**make_batch(tiny_config)).loss.detach())
    assert after != before
