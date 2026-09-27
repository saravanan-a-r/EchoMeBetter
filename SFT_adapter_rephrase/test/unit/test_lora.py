"""
The adapter itself: a no-op at attach, frozen base, correct size, correct
token-row behaviour, and optimizer grouping.
"""

from __future__ import annotations

import pytest
import torch

from conftest import TINY_MODEL, build_tiny_model, make_config
from src.config import TARGET_GROUPS
from src.errors import AdapterError
from src.interop import PROJECT_ROOT, rephrase_model
from src.lora import TOKEN_DELTA_KEY, LoRALinear, attach_adapter, expected_adapter_parameters

TOKENS = [3, 4, 5]


def _batch(seed: int = 0, batch: int = 3, source: int = 12, target: int = 9):
    generator = torch.Generator().manual_seed(seed)
    encoder = torch.randint(10, 80, (batch, source), generator=generator)
    encoder[:, :2] = torch.tensor([3, 4])  # the frame tokens are in the input
    decoder = torch.randint(10, 80, (batch, target), generator=generator)
    labels = torch.randint(10, 80, (batch, target), generator=generator)
    return dict(encoder_input_ids=encoder, decoder_input_ids=decoder, labels=labels)


def _all_projection_config(tmp_path, **adapter):
    return make_config(tmp_path, adapter={"dropout": 0.0, **adapter}).adapter


def test_attached_adapter_is_an_exact_no_op(tmp_path):
    """B = 0 and a zero token delta: step 0 must evaluate the untouched base."""
    base = build_tiny_model()
    adapted = build_tiny_model()
    attach_adapter(adapted, _all_projection_config(tmp_path), TOKENS)
    batch = _batch()
    with torch.no_grad():
        assert torch.equal(base(**batch).logits, adapted(**batch).logits)


def test_training_moves_only_adapter_parameters(tmp_path):
    """
    The base must stay bit-identical -- one base serves every adapter -- and
    every LoRA pair must actually be in the graph (a target silently skipped,
    e.g. on a cached path, would never leave zero).
    """
    model = build_tiny_model()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = attach_adapter(model, _all_projection_config(tmp_path), TOKENS)

    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n and "delta" not in n)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-2)
    model.train()
    for step in range(3):
        optimizer.zero_grad()
        model(**_batch(step)).loss.backward()
        optimizer.step()

    after = model.state_dict()
    for name, value in base_state.items():
        current = after.get(name, after.get(name.replace(".weight", ".base.weight")))
        assert torch.equal(current, value), name
    assert all(bool(m.lora_B.abs().sum() > 0) for m in adapter.lora_modules.values())
    assert bool(adapter.token_delta.delta.abs().sum() > 0)


def test_parameter_count_matches_the_closed_form(tmp_path):
    model = build_tiny_model()
    config = _all_projection_config(tmp_path, rank=3)
    adapter = attach_adapter(model, config, TOKENS)
    assert adapter.parameter_count() == expected_adapter_parameters(model.config, config)
    layers = TINY_MODEL["num_layers"] * 7 + TINY_MODEL["num_decoder_layers"] * 11
    assert len(adapter.lora_modules) == layers


def test_partial_targets_attach_only_where_asked(tmp_path):
    targets = {group: [] for group in TARGET_GROUPS}
    targets["decoder_cross_attention"] = ["q_proj", "v_proj"]
    config = _all_projection_config(tmp_path, targets=targets, trainable_tokens=[])
    model = build_tiny_model()
    adapter = attach_adapter(model, config)
    assert sorted({name.split(".", 3)[3] for name in adapter.lora_modules}) == ["cross_attn.q_proj", "cross_attn.v_proj"]
    assert isinstance(model.decoder.layers[0].self_attn.q_proj, torch.nn.Linear)
    assert adapter.parameter_count() == expected_adapter_parameters(model.config, config)


def test_full_size_formula_matches_the_sizes_documented_in_the_yaml(tmp_path):
    """The size table at the top of adapter_config.yml is computed from this formula."""
    large = rephrase_model.load_model_config(PROJECT_ROOT / "model_config.yml", profile="large")
    base = _all_projection_config(tmp_path, trainable_tokens=[])
    expected = {8: 9_142_272, 16: 18_284_544, 32: 36_569_088, 64: 73_138_176}
    for rank, count in expected.items():
        config = type(base)(**{**base.__dict__, "rank": rank})
        assert expected_adapter_parameters(large, config) == count


def test_token_delta_changes_only_its_rows_and_only_on_the_input_side(tmp_path):
    model = build_tiny_model()
    adapter = attach_adapter(model, _all_projection_config(tmp_path), TOKENS)
    with torch.no_grad():
        adapter.token_delta.delta.fill_(0.5)
    ids = torch.tensor([[3, 4, 5, 20, 21]])
    looked_up = model.encoder.embedding(ids)
    base_rows = model.shared(ids)
    assert torch.allclose(looked_up[0, :3], base_rows[0, :3] + 0.5)
    assert torch.equal(looked_up[0, 3:], base_rows[0, 3:])
    # The tied output projection still reads the frozen base matrix.
    assert model.shared.weight is model.encoder.embedding.weight
    hidden = torch.randn(1, 2, TINY_MODEL["d_model"])
    expected = torch.nn.functional.linear(hidden * TINY_MODEL["d_model"] ** -0.5, model.shared.weight)
    assert torch.allclose(model.project(hidden), expected)


def test_lr_multipliers_reach_the_optimizer_groups_and_the_delta_is_never_decayed(tmp_path):
    config = _all_projection_config(tmp_path, lr_multipliers={"q_proj": 0.25})
    adapter = attach_adapter(build_tiny_model(), config, TOKENS)
    groups = {g["name"]: g for g in adapter.param_groups(learning_rate=1e-3, weight_decay=0.1)}
    q_params = {id(p) for n, m in adapter.lora_modules.items() if n.endswith("q_proj") for p in (m.lora_A, m.lora_B)}
    assert {id(p) for p in groups["lora_x0.25"]["params"]} == q_params
    assert groups["lora_x0.25"]["lr"] == pytest.approx(2.5e-4)
    assert groups["lora_x1"]["lr"] == pytest.approx(1e-3)
    assert groups["token_delta"]["weight_decay"] == 0.0
    total = sum(p.numel() for g in groups.values() for p in g["params"])
    assert total == adapter.parameter_count()


def test_attaching_twice_is_refused(tmp_path):
    model = build_tiny_model()
    attach_adapter(model, _all_projection_config(tmp_path), TOKENS)
    with pytest.raises(AdapterError, match="already attached"):
        attach_adapter(model, _all_projection_config(tmp_path), TOKENS)


def test_delta_norm_matches_the_dense_update():
    layer = LoRALinear(torch.nn.Linear(8, 6, bias=False), rank=3, scaling=1.5, dropout=0.0)
    with torch.no_grad():
        layer.lora_B.normal_()
    assert layer.delta_norm() == pytest.approx(float(layer.delta_weight().norm()), rel=1e-5)


def test_state_dict_round_trip_is_strict(tmp_path):
    config = _all_projection_config(tmp_path)
    source = attach_adapter(build_tiny_model(), config, TOKENS)
    state = source.state_dict()
    assert TOKEN_DELTA_KEY in state
    target = attach_adapter(build_tiny_model(), config, TOKENS)
    target.load_state_dict(state)
    bad = dict(state)
    bad.pop(next(iter(bad)))
    with pytest.raises(AdapterError, match="missing"):
        target.load_state_dict(bad)
