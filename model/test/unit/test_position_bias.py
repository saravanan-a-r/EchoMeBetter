"""
Unit tests for T5 relative position bias.

Three groups, in increasing order of how expensive the bug would be:

  1. bucketing arithmetic — checked against T5's published scheme
  2. the incremental-decoding slice — a model that computes position bias
     correctly during training and incorrectly during generation produces
     good loss curves and bad output, which is the worst failure mode here
  3. the parameter shape — architecture.md §4.4's claim that context length
     costs no parameters rests entirely on this table not scaling with length
"""

from __future__ import annotations

import pytest
import torch

from src.position_bias import RelativePositionBias, relative_position_bucket


def _buckets(distances, bidirectional, num_buckets=32, max_distance=128):
    return relative_position_bucket(
        torch.tensor(distances), bidirectional, num_buckets, max_distance
    ).tolist()


# -- bucketing arithmetic -------------------------------------------------


def test_zero_distance_is_bucket_zero():
    assert _buckets([0], bidirectional=True)[0] == 0
    assert _buckets([0], bidirectional=False)[0] == 0


def test_near_distances_get_exact_buckets():
    """
    Bidirectional halves the range to 16, of which the first 8 are exact.
    Backward distances 0..-7 therefore map to buckets 0-7, one each.
    (Positive distances carry a +16 direction offset — see below.)
    """
    assert _buckets([0, -1, -2, -3, -4, -5, -6, -7], bidirectional=True) == [
        0, 1, 2, 3, 4, 5, 6, 7
    ]


def test_forward_distances_are_offset_into_the_upper_half():
    assert _buckets([-1, 1], bidirectional=True) == [1, 17]


def test_direction_is_encoded_in_the_upper_half():
    """
    Bidirectional splits the buckets: the same magnitude in each direction
    must land in different buckets, offset by half the range.
    """
    forward = _buckets([5], bidirectional=True)[0]
    backward = _buckets([-5], bidirectional=True)[0]
    assert forward - backward == 16


def test_unidirectional_collapses_the_future():
    """
    Decoder self-attention masks the future anyway, so every non-negative
    distance shares bucket 0 and the whole range is spent on the past.
    """
    assert _buckets([0, 1, 5, 100], bidirectional=False) == [0, 0, 0, 0]
    assert _buckets([-1], bidirectional=False)[0] == 1


def test_unidirectional_uses_the_full_range():
    """Not halved, so the exact region is twice as wide as bidirectional."""
    assert _buckets([-i for i in range(16)], bidirectional=False) == list(range(16))


def test_far_distances_are_grouped_logarithmically():
    """Coarser with distance: many far distances share each bucket."""
    near = _buckets(list(range(8)), bidirectional=True)
    far = _buckets(list(range(60, 68)), bidirectional=True)
    assert len(set(near)) == 8, "near distances are individually resolved"
    assert len(set(far)) < 8, "far distances are deliberately merged"


def test_buckets_never_decrease_with_distance():
    distances = list(range(0, 300))
    buckets = _buckets(distances, bidirectional=True)
    assert buckets == sorted(buckets)


def test_everything_beyond_max_distance_shares_one_bucket():
    """
    architecture.md §4.6 notes this explicitly: past `max_distance` the model
    can no longer tell distances apart. It is a known, accepted property.
    """
    beyond = _buckets([128, 200, 1000, 100000], bidirectional=True)
    assert len(set(beyond)) == 1


@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("num_buckets", [8, 32, 64])
def test_buckets_stay_inside_the_table(bidirectional, num_buckets):
    """An out-of-range index would be an embedding lookup crash at best."""
    distances = list(range(-5000, 5000, 37))
    buckets = _buckets(distances, bidirectional, num_buckets=num_buckets, max_distance=128)
    assert min(buckets) >= 0
    assert max(buckets) < num_buckets


def test_bucketing_has_no_nan_at_zero_distance():
    """
    The logarithm is evaluated for every entry before `where` selects; a NaN
    from log(0) would survive into the backward pass even though the value
    is discarded.
    """
    result = relative_position_bucket(
        torch.arange(-4, 5), bidirectional=True, num_buckets=32, max_distance=128
    )
    assert torch.isfinite(result.float()).all()


# -- the module -----------------------------------------------------------


def test_bias_shape():
    bias = RelativePositionBias(32, 128, 8, bidirectional=True)
    assert bias(7, 7, torch.device("cpu")).shape == (1, 8, 7, 7)


def test_parameter_count_is_buckets_times_heads():
    """
    architecture.md §4.3 counts `(32 buckets x 16 heads) x 2 stacks = 1,024`.
    Nothing here may scale with sequence length.
    """
    bias = RelativePositionBias(32, 128, 16, bidirectional=True)
    assert sum(p.numel() for p in bias.parameters()) == 32 * 16


def test_parameter_count_is_independent_of_sequence_length():
    """The basis of architecture.md §4.4: 2048 and 4096 cost the same."""
    bias = RelativePositionBias(32, 128, 8, bidirectional=True)
    before = sum(p.numel() for p in bias.parameters())
    bias(2048, 2048, torch.device("cpu"))
    assert sum(p.numel() for p in bias.parameters()) == before


def test_bidirectional_bias_is_not_symmetric():
    """Distance alone is not enough; direction must matter in the encoder."""
    torch.manual_seed(0)
    bias = RelativePositionBias(32, 128, 4, bidirectional=True)
    bias.embedding.weight.data.normal_()
    values = bias(9, 9, torch.device("cpu"))[0, 0]
    assert not torch.allclose(values, values.T)


def test_causal_bias_ignores_the_future():
    """
    Every non-negative distance is bucket 0, so all upper-triangular entries
    share one value. (They are masked downstream regardless.)
    """
    torch.manual_seed(0)
    bias = RelativePositionBias(32, 128, 4, bidirectional=False)
    bias.embedding.weight.data.normal_()
    values = bias(6, 6, torch.device("cpu"))[0, 0]

    upper = values[torch.triu(torch.ones(6, 6, dtype=torch.bool))]
    assert torch.allclose(upper, upper[0].expand_as(upper))


def test_incremental_bias_matches_full_bias():
    """
    The generation-correctness test. During decoding, `query_length` is 1
    while `key_length` covers the cached prefix; the resulting row must equal
    the corresponding row of a full forward pass. Getting this wrong yields a
    model that trains correctly and generates subtly wrongly.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    bias = RelativePositionBias(32, 128, 4, bidirectional=False)
    bias.embedding.weight.data.normal_()

    full = bias(12, 12, device)
    for step in range(12):
        incremental = bias(1, step + 1, device)
        assert torch.allclose(incremental[0, :, 0, :], full[0, :, step, : step + 1])


def test_query_longer_than_key_is_rejected():
    bias = RelativePositionBias(32, 128, 4, bidirectional=False)
    with pytest.raises(ValueError, match="cannot exceed"):
        bias(8, 4, torch.device("cpu"))


def test_bias_is_differentiable():
    bias = RelativePositionBias(32, 128, 4, bidirectional=True)
    bias(6, 6, torch.device("cpu")).sum().backward()
    assert bias.embedding.weight.grad is not None
    assert bias.embedding.weight.grad.abs().sum() > 0


def test_the_real_context_length_exercises_all_reachable_buckets():
    """
    Sanity on the real configuration: across 2048 positions, every bucket
    T5's scheme can reach is used, so none of the table is wasted.

    That is 31 of 32, not all 32. Bucket 16 — the first of the "forward"
    half — would mean a *positive* distance of zero, which cannot exist:
    distance 0 is not positive, so it takes bucket 0 in the backward half.
    This one dead row is inherent to T5's bucketing, costs 16 parameters,
    and is not worth deviating from a proven scheme to reclaim.
    """
    buckets = relative_position_bucket(
        torch.arange(2048)[None, :] - torch.arange(2048)[:, None],
        bidirectional=True,
        num_buckets=32,
        max_distance=128,
    )
    reachable = set(buckets.unique().tolist())
    assert reachable == set(range(32)) - {16}
