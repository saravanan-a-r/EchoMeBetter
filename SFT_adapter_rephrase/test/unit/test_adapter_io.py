"""
Saving and loading adapters, and the base-fingerprint guard.
"""

from __future__ import annotations

import pytest
import torch

from conftest import FAKE_TOKEN_MAP, build_tiny_model, make_config
from src.adapter_io import load_adapter, read_adapter_config, save_adapter
from src.base_model import base_fingerprint
from src.errors import AdapterError, BaseModelMismatchError
from src.lora import attach_adapter

TOKENS = ["<style:concise>", "<text_to_rewrite>", "</text_to_rewrite>"]


def _trained_adapter(tmp_path, seed: int = 0):
    model = build_tiny_model(seed)
    config = make_config(tmp_path).adapter
    adapter = attach_adapter(model, config, [FAKE_TOKEN_MAP[t] for t in TOKENS])
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_(0.0, 0.05)
    return model, adapter


def _save(tmp_path, model, adapter):
    base = {"checkpoint": "tiny", "step": 7, "fingerprint": base_fingerprint(model), "config": {}}
    return save_adapter(
        tmp_path / "adapter", adapter, base=base, token_map=FAKE_TOKEN_MAP, style_token="<style:concise>"
    )


def _inputs():
    encoder = torch.tensor([[3, 4, 20, 21, 22, 5, 1], [3, 4, 30, 31, 5, 1, 0]])
    decoder = torch.tensor([[0, 40, 41], [0, 42, 43]])
    return dict(encoder_input_ids=encoder, decoder_input_ids=decoder)


def test_round_trip_reproduces_the_adapted_model_exactly(tmp_path):
    model, adapter = _trained_adapter(tmp_path)
    directory = _save(tmp_path, model, adapter)

    fresh = build_tiny_model(0)
    load_adapter(fresh, directory, token_map=FAKE_TOKEN_MAP)
    model.eval(), fresh.eval()
    with torch.no_grad():
        assert torch.equal(model(**_inputs()).logits, fresh(**_inputs()).logits)
    document = read_adapter_config(directory)
    assert document["style_token"] == "<style:concise>"
    assert document["base"]["step"] == 7


def test_loading_onto_a_different_base_is_refused(tmp_path):
    model, adapter = _trained_adapter(tmp_path)
    directory = _save(tmp_path, model, adapter)
    with pytest.raises(BaseModelMismatchError):
        load_adapter(build_tiny_model(seed=1), directory)


def test_a_tokenizer_that_moved_the_frame_tokens_is_refused(tmp_path):
    model, adapter = _trained_adapter(tmp_path)
    directory = _save(tmp_path, model, adapter)
    moved = dict(FAKE_TOKEN_MAP, **{"<style:concise>": 6, "<style:grammar>": 3})
    with pytest.raises(AdapterError, match="style:concise"):
        load_adapter(build_tiny_model(0), directory, token_map=moved)


def test_base_fingerprint_is_unchanged_by_attaching_and_training_an_adapter(tmp_path):
    """The fingerprint must identify the base, not base + adapter."""
    model = build_tiny_model()
    before = base_fingerprint(model)
    adapter = attach_adapter(model, make_config(tmp_path).adapter, [3, 4, 5])
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    inputs = _inputs()
    model.train()
    loss = model(**inputs, labels=torch.tensor([[40, 41, 1], [42, 43, 1]])).loss
    loss.backward()
    optimizer.step()
    assert base_fingerprint(model) == before
    assert base_fingerprint(build_tiny_model(seed=1)) != before
