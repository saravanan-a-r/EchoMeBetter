"""
The two leaf building blocks: RMSNorm and the GeGLU feed-forward.

Both are deliberately parameter-frugal in the ways architecture.md §4.1
chose:

  - RMSNorm has a weight and **no bias**, and does not subtract the mean.
  - The feed-forward projections have **no bias** either (T5 convention).

Those absences are load-bearing for the parameter budget in §4.3: an
RMSNorm contributes exactly `d_model` parameters, and a GeGLU FFN exactly
`3 x d_model x d_ff`. Adding biases anywhere would silently change the model's
size and break the count check.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .errors import ConfigError

# Keyed by `dense_act_fn`, mirroring the corresponding `transformers.ACT2FN`
# entries. A table rather than an if/else so that adding a `feed_forward_proj`
# value without implementing its activation fails loudly at construction
# instead of silently falling through to whichever branch came last.
_ACTIVATION_FUNCTIONS = {
    # ACT2FN["gelu_new"] — NewGELUActivation, the tanh approximation. Both
    # accepted feed_forward_proj values resolve here.
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    # ACT2FN["gelu"] — the exact, erf-based GELU. Not reachable from any
    # accepted feed_forward_proj (see config.py), kept so the mapping from
    # ACT2FN names to implementations stays complete and explicit.
    "gelu": F.gelu,
}


class RMSNorm(nn.Module):
    """
    Root-mean-square layer normalization (architecture.md §4.1).

    Rescales by RMS without centering, then applies a learned per-channel
    gain. Cheaper than LayerNorm — no mean, no bias — at no quality cost.

    The normalization is computed in float32 even when the model runs in
    bf16. Squaring a bf16 activation loses precision exactly where it matters
    (the sum of squares over a wide hidden dimension), and the resulting
    instability is the sort that shows up as a diverging loss hours in. The
    cast back to the input dtype at the end keeps the memory saving.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)[0]}, eps={self.eps}"


class GeGLUFeedForward(nn.Module):
    """
    Gated GELU feed-forward (architecture.md §4.1, §4.7).

        out = W_down( GELU(W_gate x) * W_up x )

    Three matrices where a conventional FFN has two, which is why d_ff is
    ~2.75x d_model rather than 4x: at that ratio a GeGLU FFN has the same
    parameter count as a 4x ReLU FFN while training better.

    The multiplication is the point — the gate branch modulates the up branch
    per channel, so the layer can suppress or pass information conditionally
    rather than applying one fixed nonlinearity.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        # `dense_act_fn` is the activation HuggingFace resolves this config's
        # `feed_forward_proj` to, and it is a `transformers.ACT2FN` key — so
        # the table below and HuggingFace's ACT2FN lookup select the same
        # function for the same config.json.
        self.activation = config.dense_act_fn
        if self.activation not in _ACTIVATION_FUNCTIONS:
            raise ConfigError(
                f"no implementation for dense_act_fn {self.activation!r}; "
                f"known: {sorted(_ACTIVATION_FUNCTIONS)}"
            )

    def _activate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _ACTIVATION_FUNCTIONS[self.activation](hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = self._activate(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(self.dropout(gated))
