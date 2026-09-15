"""
Unit tests for `TrainingMonitor`'s own behaviour.

What is tested is what would cost a run, or mislead about one, without
failing anything else: a broken component that took training down or spammed
the log, a spike alert that fired every step or not at all, a resumed run
whose curves double back over the steps it repeated, and charts that lose
their titles and explanations.
"""

from __future__ import annotations

from tensorboard.backend.event_processing import event_file_loader, plugin_event_accumulator
from tensorboard.util import tensor_util

from src.dashboard import TrainingMonitor


def step_record(step: int, loss: float = 2.0) -> dict:
    return {
        "step": step,
        "loss": loss,
        "perplexity": 7.4,
        "learning_rate": 1e-3,
        "grad_norm": 0.5,
        "tokens_seen": 100 * step,
        "input_tokens_seen": 200 * step,
        "skipped_steps": 0,
        "time": 1000.0 + 30 * step,
    }


def read_events(log_dir) -> plugin_event_accumulator.EventAccumulator:
    """The run as TensorBoard reads it."""
    events = plugin_event_accumulator.EventAccumulator(
        str(log_dir), size_guidance={plugin_event_accumulator.TENSORS: 0}
    )
    events.Reload()
    return events


def scalar_steps(log_dir, tag: str) -> list[int]:
    return [point.step for point in read_events(log_dir).Tensors(tag)]


def test_a_failing_component_is_reported_once_and_the_dashboard_goes_on(tmp_path):
    class BrokenProbes:
        calls = 0

        def run(self):
            BrokenProbes.calls += 1
            raise RuntimeError("probe failed")

    lines = []
    monitor = TrainingMonitor(
        tmp_path, max_steps=10, probes=BrokenProbes(), probe_every=1, write=lines.append
    )
    for step in (1, 2, 3):
        monitor(step_record(step))
    monitor.close()

    warnings = [line for line in lines if "WARNING dashboard probes disabled" in line]
    assert len(warnings) == 1 and "probe failed" in warnings[0]
    assert BrokenProbes.calls == 1
    assert scalar_steps(tmp_path, "loss/train") == [1, 2, 3]


def test_a_loss_spike_warns_once_as_it_starts(tmp_path):
    lines = []
    monitor = TrainingMonitor(tmp_path, max_steps=100, write=lines.append)
    losses = [2.0] * 30 + [3.0] * 3 + [2.0] * 5 + [3.0]
    for step, loss in enumerate(losses, start=1):
        monitor(step_record(step, loss))
    monitor.close()

    spikes = [line for line in lines if "WARNING loss spike" in line]
    assert len(spikes) == 2
    assert "step 31:" in spikes[0] and "step 39:" in spikes[1]


def test_a_resumed_run_replaces_the_steps_it_repeats(tmp_path):
    """
    A crash at step 8 resumed from a step-5 checkpoint re-runs 6-8. Without
    `purge_step` both versions of 6-8 would be charted, and the curve would
    zigzag back on itself at every resume of a multi-week run.
    """
    first = TrainingMonitor(tmp_path, max_steps=10)
    for step in range(1, 9):
        first(step_record(step, loss=2.0))
    first.close()

    resumed = TrainingMonitor(tmp_path, max_steps=10, start_step=5)
    for step in range(6, 11):
        resumed(step_record(step, loss=1.0))
    resumed.close()

    assert scalar_steps(tmp_path, "loss/train") == list(range(1, 11))


def test_each_metric_reaches_tensorboard_with_its_title_and_explanation(tmp_path):
    """
    A tag's first event carries its title and explanation, in tensor form:
    TensorBoard's default data loader drops them from a `simple_value` event,
    so in that form they would vanish from the UI while every Python reader
    still showed them. Later events are written by a different call; both
    have to land the value at its step.
    """
    monitor = TrainingMonitor(tmp_path, max_steps=10)
    for step, loss in ((1, 3.0), (2, 2.5), (3, 2.0)):
        monitor({**step_record(step, loss), "loss_R": loss + 1})
    monitor.close()

    first_events = {}
    for path in sorted(tmp_path.glob("events.out.tfevents.*")):
        for event in event_file_loader.LegacyEventFileLoader(str(path)).Load():  # as written
            for value in event.summary.value:
                first_events.setdefault(value.tag, value)
    events = read_events(tmp_path)
    for tag, title, values in (
        ("loss/train", "Training loss", [3.0, 2.5, 2.0]),
        ("loss/train_R", "Training loss, R mode", [4.0, 3.5, 3.0]),
    ):
        first = first_events[tag]
        assert first.HasField("tensor") and first.metadata.display_name == title
        assert "**Healthy:**" in first.metadata.summary_description
        assert first.metadata.plugin_data.plugin_name == "scalars"
        points = events.Tensors(tag)
        assert [point.step for point in points] == [1, 2, 3]
        assert [float(tensor_util.make_ndarray(point.tensor_proto)) for point in points] == values
