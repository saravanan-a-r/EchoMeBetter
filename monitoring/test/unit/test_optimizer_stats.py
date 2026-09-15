"""
Unit tests for the per-group optimizer stats.

The first test carries the most weight. These hooks run inside every sampled
`optimizer.step()` of a multi-week run, and the one failure worse than a
wrong chart is a chart that changed the run — so a monitored run and an
unmonitored one must end bit-identical, weights and AdamW moments both.

The second pins what `update_ratio` means and which step it is written at:
a ratio measured at one step and charted at another would look plausible and
mislead exactly when a spike needs locating.
"""

from __future__ import annotations

import pytest
import torch

from conftest import model_and_optimizer, train
from src.optimizer_stats import OptimizerStats


def test_a_monitored_run_is_bit_identical_to_an_unmonitored_one():
    plain_model, plain_optimizer = model_and_optimizer()
    train(plain_model, plain_optimizer, steps=6)

    model, optimizer = model_and_optimizer()
    # Every step sampled, histograms included: the most the hooks ever do.
    stats = OptimizerStats(optimizer, every_steps=1, histogram_every_steps=1)
    samples = []

    def record(step):
        samples.append(stats.take())
        stats.schedule(step + 1)

    train(model, optimizer, steps=6, after_step=record)

    assert sum(sample is not None for sample in samples) == 5  # every step after the first
    for plain, monitored in zip(plain_model.parameters(), model.parameters()):
        assert torch.equal(plain, monitored)
    for plain, monitored in zip(plain_model.parameters(), model.parameters()):
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                plain_optimizer.state[plain][key], optimizer.state[monitored][key]
            )


def test_update_ratio_is_the_exact_change_of_the_step_it_is_written_at():
    model, optimizer = model_and_optimizer()
    stats = OptimizerStats(optimizer, every_steps=3, histogram_every_steps=1000)
    before: dict[int, list[torch.Tensor]] = {}
    expected: dict[int, float] = {}
    written: dict[int, dict[str, float]] = {}
    decayed = optimizer.param_groups[0]["params"]

    def snapshot(step):
        before[step] = [p.detach().clone() for p in decayed]

    def record(step):
        change = torch.stack([(p.detach() - b).norm() for p, b in zip(decayed, before[step])]).norm()
        size = torch.stack([b.norm() for b in before[step]]).norm()
        expected[step] = float(change / size)
        sample = stats.take()
        if sample is not None:
            written[step] = sample.scalars
            if step == 3:
                check_histogram(sample.histograms["weights/decayed"], before[step])
        stats.schedule(step + 1)

    train(model, optimizer, steps=9, before_step=snapshot, after_step=record)

    assert sorted(written) == [3, 6, 9]
    for step, scalars in written.items():
        assert scalars["update_ratio/decayed"] == pytest.approx(expected[step], rel=1e-5)
        assert scalars["group_lr/decayed"] == pytest.approx(1e-2)
        assert set(scalars) >= {"weight_rms/no_decay", "grad_norm/decayed", "grad_norm/no_decay"}


def check_histogram(histogram, tensors):
    values = torch.cat([t.reshape(-1) for t in tensors])
    assert histogram.num == values.numel() == sum(histogram.bucket_counts)
    assert histogram.min == pytest.approx(float(values.min()))
    assert histogram.max == pytest.approx(float(values.max()))
    limits = histogram.bucket_limits
    assert all(low < high for low, high in zip(limits, limits[1:]))
    # Each value falls in the bucket whose upper limit is the first >= it.
    assert limits[-1] >= histogram.max and limits[0] >= histogram.min


def test_a_failing_measurement_never_reaches_the_optimizer_step():
    """
    The hooks run inside `optimizer.step()`; an exception there would end the
    run. It must be held, the hooks removed, and handed to the monitor on its
    next call — while the step itself completes as if nothing were attached.
    """
    model, optimizer = model_and_optimizer()
    stats = OptimizerStats(optimizer, every_steps=1)
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("measurement failed")

    stats._measure_before = broken
    stats.schedule(1)
    start = [p.detach().clone() for p in model.parameters()]

    train(model, optimizer, steps=3)

    assert calls == [1]  # removed after the first failure
    assert any(not torch.equal(a, p) for a, p in zip(start, model.parameters()))
    with pytest.raises(RuntimeError, match="measurement failed"):
        stats.take()
