"""
Unit tests for optimizer construction and weight-decay grouping.

The grouping is the whole subject here, and it is worth the attention because
getting it wrong produces no error, no shape mismatch and no crash — only a
model that trains to a slightly worse place for a reason that appears in no
curve. Decaying an RMSNorm gain pulls it towards zero, scaling that layer's
entire output down; the loss still descends, just not as far.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import TINY_TRAINING
from src.config import TrainingConfig
from src.errors import TrainingConfigError
from src.optimizer import (
    build_optimizer,
    current_learning_rate,
    decayed_and_exempt,
    parameter_groups,
    scale_gradients,
    set_learning_rate,
)


def test_every_parameter_lands_in_exactly_one_group(tiny_model):
    """
    Tied embeddings make this a real risk rather than a formality: `shared`,
    `encoder.embedding` and `decoder.embedding` are one tensor, and a copy in
    two groups would be stepped twice per optimizer step at two different
    effective learning rates, with AdamW keeping two disagreeing moment
    estimates for it.
    """
    groups = parameter_groups(tiny_model, 0.1)
    seen = [id(p) for group in groups for p in group["params"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in tiny_model.parameters()}


def test_normalization_gains_are_exempt_from_decay(tiny_model):
    """
    Every RMSNorm weight, and nothing else. Named explicitly so a failure
    says which tensor moved rather than "the sets differ".
    """
    _, exempt = decayed_and_exempt(tiny_model)
    exempt_names = {name for name, _ in exempt}
    norm_names = {
        f"{module_name}.weight"
        for module_name, module in tiny_model.named_modules()
        if type(module).__name__ == "RMSNorm"
    }
    assert exempt_names == norm_names
    assert exempt_names, "the model has no norms; this test would be vacuous"


def test_the_embedding_is_decayed(tiny_model):
    decayed, _ = decayed_and_exempt(tiny_model)
    assert "shared.weight" in {name for name, _ in decayed}


def test_the_rule_agrees_with_huggingfaces_name_based_one(tiny_model):
    """
    `transformers` splits by *name* (`LayerNorm` in the module path, or a
    `.bias` suffix); this splits by dimensionality. They must agree on this
    model, or a fine-tuner continuing under `Trainer` would decay a different
    set of tensors than pretraining did.

    Dimensionality is used here rather than names because a name-matching rule
    breaks silently on a rename — `self_attn_norm` contains no `LayerNorm`.
    """
    by_name = {
        name
        for name, parameter in tiny_model.named_parameters()
        if parameter.ndim >= 2 and not name.endswith(".bias")
    }
    decayed, _ = decayed_and_exempt(tiny_model)
    assert {name for name, _ in decayed} == by_name


def test_the_decay_rate_reaches_only_the_decayed_group(tiny_model):
    groups = parameter_groups(tiny_model, 0.1)
    rates = {group["name"]: group["weight_decay"] for group in groups}
    assert rates == {"decayed": 0.1, "no_decay": 0.0}


def test_a_model_of_only_norms_produces_one_group():
    """An empty parameter group is rejected by PyTorch, so it must be dropped."""
    model = nn.Sequential(nn.LayerNorm(4))
    groups = parameter_groups(model, 0.1)
    assert [group["name"] for group in groups] == ["no_decay"]
    assert torch.optim.AdamW(groups, lr=1e-3)


def test_a_model_with_nothing_to_train_is_refused():
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(TrainingConfigError, match="nothing to optimize"):
        parameter_groups(model, 0.1)


def test_frozen_parameters_are_left_out(tiny_model):
    """A fine-tune that freezes the embedding must not have it stepped."""
    tiny_model.shared.weight.requires_grad_(False)
    decayed, exempt = decayed_and_exempt(tiny_model)
    names = {name for name, _ in decayed} | {name for name, _ in exempt}
    assert "shared.weight" not in names


# -- the optimizer itself --------------------------------------------------


def test_the_optimizer_uses_the_configured_betas_and_epsilon(tiny_model):
    """
    β2 = 0.95, not PyTorch's 0.999 (architecture.md §13). The shorter
    second-moment window is what lets a long run recover quickly from a loss
    spike; 0.999 is tuned for short fine-tunes and is slow to.
    """
    settings = TrainingConfig(**dict(TINY_TRAINING, adam_beta2=0.95, adam_epsilon=1e-8))
    optimizer = build_optimizer(tiny_model, settings)
    for group in optimizer.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8


def test_the_optimizer_starts_at_the_peak_learning_rate(tiny_model, training_config):
    """
    The schedule overwrites this before every step, so the initial value is
    only ever seen if the schedule failed to run — which is exactly when it
    should be an obviously wrong number rather than a plausible one.
    """
    optimizer = build_optimizer(tiny_model, training_config)
    assert current_learning_rate(optimizer) == training_config.learning_rate


def test_setting_the_learning_rate_reaches_every_group(tiny_model, training_config):
    """
    The groups differ in weight decay only. A schedule that reached one and
    not the other would leave the norms on a flat learning rate for the whole
    run, and nothing would say so.
    """
    optimizer = build_optimizer(tiny_model, training_config)
    set_learning_rate(optimizer, 1.25e-4)
    assert {group["lr"] for group in optimizer.param_groups} == {1.25e-4}


def test_scaling_gradients_is_in_place_and_skips_the_absent(tiny_model):
    for parameter in tiny_model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    tiny_model.shared.weight.grad = None

    scale_gradients(tiny_model.parameters(), 0.5)

    assert tiny_model.shared.weight.grad is None
    for parameter in tiny_model.parameters():
        if parameter.grad is not None:
            assert torch.allclose(parameter.grad, torch.ones_like(parameter))


def test_weight_decay_actually_shrinks_a_decayed_matrix(tiny_model, training_config):
    """
    End to end, on a zero gradient, so the only thing that can move a weight
    is the decay term. A norm must not move; a matrix must.
    """
    settings = training_config.with_(weight_decay=0.5)
    optimizer = build_optimizer(tiny_model, settings)
    set_learning_rate(optimizer, 0.1)

    for parameter in tiny_model.parameters():
        parameter.grad = torch.zeros_like(parameter)

    matrix_before = tiny_model.shared.weight.detach().clone()
    norm = tiny_model.encoder.final_norm.weight
    norm_before = norm.detach().clone()

    optimizer.step()

    assert tiny_model.shared.weight.abs().sum() < matrix_before.abs().sum()
    assert torch.equal(norm, norm_before)


def test_the_optimizer_can_take_a_step_and_change_the_model(tiny_model, training_config):
    from conftest import make_batch

    optimizer = build_optimizer(tiny_model, training_config)
    set_learning_rate(optimizer, 1e-2)
    batch = make_batch(tiny_model.config)

    before = tiny_model.shared.weight.detach().clone()
    tiny_model(
        encoder_input_ids=batch["encoder_input_ids"],
        decoder_input_ids=batch["decoder_input_ids"],
        encoder_attention_mask=batch["encoder_attention_mask"],
        decoder_attention_mask=batch["decoder_attention_mask"],
        labels=batch["labels"],
    ).loss.backward()
    optimizer.step()

    assert not torch.equal(before, tiny_model.shared.weight)
