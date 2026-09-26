"""
Loading the frozen base model, and fingerprinting it.

The base is rebuilt from the checkpoint's own `config.json` rather than from
`model_config.yml`, so the adapter always trains on exactly the network that
checkpoint holds, whatever profile the YAML happens to name today.

Fingerprint
-----------
A LoRA delta only means something relative to the weights it was learned
against. Every saved adapter records a SHA-256 over the base's tied
embedding and every RMSNorm gain -- tensors that change at every optimizer
step of pretraining, so two checkpoints never share them -- and loading
refuses a base whose fingerprint differs. Hashing ~140 MB takes well under a
second, where hashing the whole 3 GB safetensors file would take many.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .errors import AdapterError
from .interop import rephrase_model

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


@dataclass(frozen=True)
class BaseModelInfo:
    checkpoint: str
    step: int | None
    fingerprint: str
    parameters: int
    config: dict


def load_base_model(checkpoint: str | Path) -> tuple[nn.Module, BaseModelInfo]:
    """Build the network from `config.json`, load its weights (fp32, CPU), fingerprint it."""
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    config_path, weights_path = checkpoint / CONFIG_FILE, checkpoint / WEIGHTS_FILE
    for path in (config_path, weights_path):
        if not path.is_file():
            raise AdapterError(f"base checkpoint is missing {path.name}: {path}")

    document = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = rephrase_model.model_config_from_huggingface(document)
    # Dropout off in the base: the adapter carries its own, and a frozen
    # network gains nothing from noise it cannot adapt to.
    model_config = model_config.with_(dropout_rate=0.0, label_smoothing=0.0, expected_parameters=None)
    model = rephrase_model.build_model(model_config)
    rephrase_model.load_huggingface_state_dict(model, load_file(str(weights_path)))
    model = model.float()

    info = BaseModelInfo(
        checkpoint=str(checkpoint.resolve()),
        step=checkpoint_step(checkpoint),
        fingerprint=base_fingerprint(model),
        parameters=model.parameter_count(),
        config=document,
    )
    return model, info


def checkpoint_step(checkpoint: Path) -> int | None:
    """The pretraining step, from `trainer_state.json` or failing that the directory name."""
    state = checkpoint / "trainer_state.json"
    if state.is_file():
        try:
            return int(json.loads(state.read_text(encoding="utf-8"))["global_step"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    match = re.search(r"checkpoint-(\d+)$", checkpoint.name)
    return int(match.group(1)) if match else None


@torch.no_grad()
def base_fingerprint(model: nn.Module) -> str:
    """
    SHA-256 of the base's tied embedding and all RMSNorm gains, in module
    order, as float32 bytes. Unaffected by an attached adapter: the adapter
    replaces the stacks' input lookups but never `model.shared`, and never
    wraps a norm.
    """
    norm_class = rephrase_model.RMSNorm
    digest = hashlib.sha256()
    tensors = [("shared.weight", model.shared.weight)]
    tensors += [
        (f"{name}.weight", module.weight)
        for name, module in model.named_modules()
        if isinstance(module, norm_class)
    ]
    for name, tensor in tensors:
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()
