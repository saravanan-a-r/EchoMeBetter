"""
T5 relative position bias (architecture.md §4.6).

What it is
----------
Instead of adding a positional vector to each token's embedding, T5 adds a
learned scalar to each attention *logit*, chosen by how far apart the two
tokens are. So the model is told "these tokens are 7 apart", never "this
token is at index 412".

Two consequences matter for this project:

  1. **No positional ceiling and no per-position parameters.** The table is
     `num_buckets x num_heads`, independent of context length. That is why
     architecture.md §4.4 could raise the decoder context from 1024 to 2048
     with *zero* change to the parameter count or model file size.

  2. **Length generalization.** Distances are bucketed logarithmically —
     exact for near neighbours, progressively coarser further out, with
     everything beyond `max_distance` sharing one bucket. The coarseness is
     deliberate: it is what lets a model trained on one length range behave
     sensibly on another.

Where it is applied
-------------------
Encoder self-attention and decoder self-attention only. **Cross-attention has
no position bias at all** — it is purely content-based, and there is no
meaningful notion of "distance" between a decoder position and an encoder
position. This is not an omission; `test_cross_attention_has_no_position_bias`
is a regression lock on it.

One table per stack, shared by every layer in that stack (T5's design), which
is why §4.3 counts `(32 buckets x 16 heads) x 2 stacks = 1,024` parameters
rather than one table per layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> torch.Tensor:
    """
    Map signed token distances to bucket indices (T5's scheme, exactly).

    `relative_position` is `key_index - query_index`, so a positive value
    means the key is *after* the query.

    Bidirectional (encoder): the bucket range is split in half, one half per
    direction, and the sign is recorded by offsetting into the upper half.
    Unidirectional (decoder self-attention): future positions are masked
    anyway, so the whole range is spent on the past and every non-negative
    distance maps to bucket 0.

    Within each direction, the first `num_buckets // 2` distances get their
    own exact bucket; beyond that, distances are grouped logarithmically up
    to `max_distance`, and everything past `max_distance` shares the final
    bucket.
    """
    relative_buckets = torch.zeros_like(relative_position)

    if bidirectional:
        num_buckets //= 2
        relative_buckets = relative_buckets + (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # Logarithmic spacing for the far half. `.clamp(min=1)` guards the log
    # against a zero distance; those entries are discarded by `is_small`
    # anyway, but a NaN produced here would propagate through `torch.where`
    # in the backward pass.
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float().clamp(min=1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


class RelativePositionBias(nn.Module):
    """
    The learned bucket table for one stack, shared across its layers.

    Parameters: `num_buckets x num_heads`. Nothing scales with sequence
    length.
    """

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Bias of shape `(1, num_heads, query_length, key_length)`.

        During incremental decoding `query_length` is 1 while `key_length`
        covers the whole cached prefix. The query positions are therefore
        taken from the *end* of the key range, so step *t* sees the same
        distances it would have seen inside a full forward pass. Getting this
        wrong produces a model that generates correctly in training and
        subtly wrongly at inference — which is why
        `test_incremental_bias_matches_full_bias` exists.
        """
        if query_length > key_length:
            raise ValueError(
                f"query_length ({query_length}) cannot exceed key_length ({key_length})"
            )

        context_position = torch.arange(
            key_length - query_length, key_length, dtype=torch.long, device=device
        )[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

        buckets = relative_position_bucket(
            memory_position - context_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        # (q, k, heads) -> (1, heads, q, k)
        return self.embedding(buckets).permute(2, 0, 1).unsqueeze(0)

    def extra_repr(self) -> str:
        return (
            f"num_buckets={self.num_buckets}, max_distance={self.max_distance}, "
            f"num_heads={self.num_heads}, bidirectional={self.bidirectional}"
        )
