"""
Shared helpers for the monitoring test suite.

Run pytest **from this directory**, as with the other packages: each has a
package called `src`, so each is its own test root.

The models here are deliberately not this project's model. The package
imports nothing from its siblings — it reads the trainer's record contract,
an optimizer's parameter groups and a model's public methods — so its tests
drive it with the smallest things that honour those contracts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def model_and_optimizer(seed: int = 0) -> tuple[nn.Module, torch.optim.AdamW]:
    """A two-layer MLP under AdamW with two named groups, as the real run has."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    optimizer = torch.optim.AdamW(
        [
            {"params": [model[0].weight, model[2].weight], "name": "decayed", "weight_decay": 0.1},
            {"params": [model[0].bias, model[2].bias], "name": "no_decay", "weight_decay": 0.0},
        ],
        lr=1e-2,
        foreach=True,
    )
    return model, optimizer


def train(model, optimizer, steps: int, *, before_step=None, after_step=None) -> None:
    """
    The trainer's step in miniature: backward, clip, step, clear.

    `before_step(step)` runs after clipping, where the real trainer hands over
    to `optimizer.step()`; `after_step(step)` runs where it builds the record.
    """
    data = torch.Generator().manual_seed(1)
    for step in range(1, steps + 1):
        inputs = torch.randn(32, 8, generator=data)
        targets = torch.randn(32, 4, generator=data)
        nn.functional.mse_loss(model(inputs), targets).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if before_step is not None:
            before_step(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if after_step is not None:
            after_step(step)
