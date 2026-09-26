"""
Saving and loading an adapter directory.

    <dir>/adapter.safetensors     LoRA A/B per module + the token delta (fp32)
    <dir>/adapter_config.json     everything needed to re-attach it, and the
                                  fingerprint of the base it belongs to

The directory holds only the adapter (~73 MB at rank 16), never the base: one
base checkpoint serves every adapter, which is the point of adapters.
Loading is strict at every step -- wrong format version, wrong base, a
missing or misshapen tensor all raise -- because every one of those failures
otherwise produces a model that runs and generates plausible text.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn

from .base_model import base_fingerprint
from .config import AdapterConfig
from .errors import AdapterError, BaseModelMismatchError
from .lora import AttachedAdapter, attach_adapter

FORMAT_VERSION = 1
TENSORS_FILE = "adapter.safetensors"
CONFIG_FILE = "adapter_config.json"


def save_adapter(
    directory: str | Path,
    adapter: AttachedAdapter,
    *,
    base: Mapping[str, Any],
    token_map: Mapping[str, int],
    style_token: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Write the adapter to `directory` (created if needed). `base` is the
    `BaseModelInfo` as a dict (checkpoint, step, fingerprint, config);
    `style_token` is the frame's style (what inference must prefix inputs
    with); `extra` is free-form provenance such as the step and metrics.
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = adapter.state_dict()
    save_file(tensors, str(directory / TENSORS_FILE), metadata={"format_version": str(FORMAT_VERSION)})

    document = {
        "format_version": FORMAT_VERSION,
        "adapter": _adapter_config_to_dict(adapter.config),
        "scaling": adapter.config.scaling,
        "style_token": style_token,
        "trainable_tokens": {token: int(token_map[token]) for token in adapter.config.trainable_tokens},
        "parameters": adapter.parameter_count(),
        "base": {
            "checkpoint": base.get("checkpoint"),
            "step": base.get("step"),
            "fingerprint": base["fingerprint"],
            "config": base.get("config"),
        },
        "provenance": dict(extra or {}),
    }
    _write_json(directory / CONFIG_FILE, document)
    return directory


def read_adapter_config(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / CONFIG_FILE
    if not path.is_file():
        raise AdapterError(f"not an adapter directory (no {CONFIG_FILE}): {directory}")
    document = json.loads(path.read_text(encoding="utf-8"))
    version = document.get("format_version")
    if version != FORMAT_VERSION:
        raise AdapterError(f"{path}: adapter format version {version!r}, this code reads {FORMAT_VERSION}")
    return document


def load_adapter(
    model: nn.Module,
    directory: str | Path,
    *,
    verify_base: bool = True,
    token_map: Mapping[str, int] | None = None,
) -> AttachedAdapter:
    """
    Attach the adapter saved in `directory` to `model` (a freshly loaded base).

    `verify_base` compares the base fingerprint first -- leave it on except
    in tests. `token_map`, when given, confirms the tokenizer still maps the
    trainable tokens to the IDs the adapter learned them under.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    document = read_adapter_config(directory)

    if verify_base:
        expected = document["base"]["fingerprint"]
        actual = base_fingerprint(model)
        if actual != expected:
            raise BaseModelMismatchError(
                f"adapter {directory} was trained on base {document['base'].get('checkpoint')} "
                f"(step {document['base'].get('step')}, fingerprint {expected[:12]}...), but the "
                f"model it is being loaded onto has fingerprint {actual[:12]}...; load the "
                f"matching base checkpoint"
            )

    tokens: dict[str, int] = document.get("trainable_tokens", {})
    if token_map is not None:
        for token, token_id in tokens.items():
            if token_map.get(token) != token_id:
                raise AdapterError(
                    f"tokenizer maps {token!r} to {token_map.get(token)!r}, but the adapter "
                    f"learned it as ID {token_id}; this is not the adapter's tokenizer"
                )

    config = adapter_config_from_dict(document["adapter"])
    attached = attach_adapter(model, config, token_ids=list(tokens.values()))
    attached.load_state_dict(load_file(str(directory / TENSORS_FILE)))
    return attached


def adapter_config_from_dict(raw: Mapping[str, Any]) -> AdapterConfig:
    raw = dict(raw)
    raw["targets"] = {group: tuple(p) for group, p in raw["targets"].items()}
    raw["trainable_tokens"] = tuple(raw.get("trainable_tokens", ()))
    raw["lr_multipliers"] = dict(raw.get("lr_multipliers", {}))
    try:
        return AdapterConfig(**raw)
    except TypeError as exc:
        raise AdapterError(f"adapter_config.json has an unreadable `adapter` block: {exc}") from exc


def _adapter_config_to_dict(config: AdapterConfig) -> dict[str, Any]:
    raw = asdict(config)
    raw["targets"] = {group: list(p) for group, p in raw["targets"].items()}
    raw["trainable_tokens"] = list(raw["trainable_tokens"])
    return raw


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write-then-rename, so an interrupted save never leaves half a config."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)
