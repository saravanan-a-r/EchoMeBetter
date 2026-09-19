"""
Unit tests for `TrainingMonitor`'s own behaviour.

What is tested is what would cost a run, or mislead about one, without
failing anything else: a broken component that took training down or spammed
the log, a spike alert that fired every step or not at all, a resumed run
whose curves double back over the steps it repeated, and charts that lose
their titles and explanations.
"""

from __future__ import annotations

import pytest
from tensorboard.backend.event_processing import event_file_loader, plugin_event_accumulator
from tensorboard.util import tensor_util

from src.dashboard import TrainingMonitor, ownership_scalars


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


def test_throughput_has_a_point_at_the_first_step_of_a_resumed_session(tmp_path):
    """
    Without a seed, `_Throughput` needs two records before it can compute
    anything, so the very first record of every session -- including every
    resume, not just step 0 -- would silently have no `throughput/*` point
    at all: a real gap in the chart on a step the console's progress line
    still prints a normal s/step and ETA for, because the console seeds
    itself the same way from the resume point.
    """
    start_step, start_tokens, start_inputs = 2000, 100 * 2000, 200 * 2000
    seed_time = 1000.0 + 30 * start_step  # matches step_record's fake clock

    monitor = TrainingMonitor(
        tmp_path,
        max_steps=90_000,
        start_step=start_step,
        start_tokens_seen=start_tokens,
        start_input_tokens_seen=start_inputs,
        clock=lambda: seed_time,
    )
    monitor(step_record(start_step + 1))
    monitor.close()

    assert scalar_steps(tmp_path, "throughput/seconds_per_step") == [start_step + 1]
    assert scalar_steps(tmp_path, "throughput/eta_days") == [start_step + 1]


def test_samples_every_runs_between_full_probes_but_not_on_top_of_one(tmp_path):
    """
    `samples_every` is meant to catch model output between the full probe's
    (expensive) cadence, not duplicate it: on a step the full probe already
    ran, `sample_texts` must not also run and overwrite the same tag twice.
    """

    class RecordingProbes:
        run_calls: list[int] = []
        sample_calls: list[int] = []

        class _Result:
            scalars: dict = {}
            texts = {"samples/R": "from full probe"}

        def run(self):
            RecordingProbes.run_calls.append(len(RecordingProbes.run_calls))
            return self._Result()

        def sample_texts(self):
            RecordingProbes.sample_calls.append(len(RecordingProbes.sample_calls))
            return {"samples/R": "from tight cadence"}

    probes = RecordingProbes()
    monitor = TrainingMonitor(
        tmp_path, max_steps=20, probes=probes, probe_every=10, samples_every=4
    )
    for step in range(1, 13):
        monitor(step_record(step))
    monitor.close()

    # probe_every=10 with the "always probe the first record" rule: steps 1, 10.
    assert len(RecordingProbes.run_calls) == 2
    # samples_every=4 on steps 4, 8, 12 -- never on a step the full probe took
    # (1 and 10), and none of those coincide here, so all three land.
    assert len(RecordingProbes.sample_calls) == 3
    assert scalar_steps(tmp_path, "samples/R/text_summary") == [1, 4, 8, 10, 12]


def test_samples_every_requires_probes(tmp_path):
    with pytest.raises(ValueError):
        TrainingMonitor(tmp_path, max_steps=10, samples_every=5)


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


# -- ownership events -----------------------------------------------------


def test_ownership_scalars_reads_exactly_what_each_technique_logged():
    """
    Nothing here is derived -- the ask is only "is it running", so every
    number plotted has to be one the technique already computed.
    """
    assert ownership_scalars({"checkpoint_hashed": "checkpoint-1000", "step": 1000}) == {
        "ownership/checkpoint_hash_written": 1.0
    }
    assert ownership_scalars({"fingerprint_injected": 3, "rows": 75, "step": 1500}) == {
        "ownership/fingerprint_injections": 3.0
    }
    assert ownership_scalars(
        {
            "signature_matched": 96,
            "signature_bits": 96,
            "signature_min_margin": 1.98,
            "signature_corrections": 112,
            "step": 2000,
        }
    ) == {
        "ownership/signature_matched_bits": 96.0,
        "ownership/signature_min_margin": 1.98,
        "ownership/signature_corrections": 112.0,
    }
    # A plain training record and a bare error carry none of these fields.
    assert ownership_scalars({"step": 5, "loss": 2.0}) == {}
    assert ownership_scalars({"ownership_error": "disk full", "step": 9}) == {}


def test_ownership_events_reach_tensorboard_under_their_own_tags(tmp_path):
    """
    End to end: the three techniques' records, exactly as `pretrain.py` hands
    them to `on_log`, land as real points at the steps they were logged at.
    """
    monitor = TrainingMonitor(tmp_path, max_steps=10_000)
    monitor({"checkpoint_hashed": "checkpoint-1000", "step": 1000,
              "checkpoint_sha256": "ab" * 32})
    monitor({"fingerprint_injected": 1, "rows": 75, "step": 500})
    monitor({"fingerprint_injected": 2, "rows": 75, "step": 1000})
    monitor({"signature_matched": 96, "signature_bits": 96, "signature_min_margin": 1.98,
              "signature_corrections": 112, "step": 1000})
    monitor.close()

    events = read_events(tmp_path)
    assert scalar_steps(tmp_path, "ownership/checkpoint_hash_written") == [1000]
    assert scalar_steps(tmp_path, "ownership/fingerprint_injections") == [500, 1000]
    assert [
        float(tensor_util.make_ndarray(p.tensor_proto))
        for p in events.Tensors("ownership/fingerprint_injections")
    ] == [1.0, 2.0]
    assert scalar_steps(tmp_path, "ownership/signature_matched_bits") == [1000]


def test_an_ownership_event_with_no_step_is_skipped_not_a_crash(tmp_path):
    """
    `checkpoint_hash`'s "hasher is closed" error fires from `close()`, off
    any step -- there is nothing to plot it against, so it is dropped rather
    than raising or landing at a fabricated step.
    """
    monitor = TrainingMonitor(tmp_path, max_steps=10)
    monitor({"ownership_error": "hasher is closed", "checkpoint": "checkpoint-9000"})
    monitor.close()  # would raise if the record above were not handled


def test_an_ownership_event_does_not_corrupt_the_throughput_tracker(tmp_path):
    """
    Ownership records carry no `tokens_seen`; if one reached `_Throughput`
    the way a training record does, its zero token count would overwrite the
    tracker's last-seen totals and the next real step would compute a
    `tokens_per_second` against that zero instead of the previous step's real
    count. Interleaving a checkpoint-hash event between two training steps
    must not touch that number at all.
    """
    monitor = TrainingMonitor(tmp_path, max_steps=10)
    monitor(step_record(1))
    monitor({"checkpoint_hashed": "checkpoint-1", "step": 1, "checkpoint_sha256": "ab" * 32})
    monitor(step_record(2))
    monitor.close()

    events = read_events(tmp_path)
    points = events.Tensors("throughput/tokens_per_second")
    assert [point.step for point in points] == [2]
    # 300 tokens/step (100 graded + 200 input) over 30s, per `step_record`.
    assert tensor_util.make_ndarray(points[0].tensor_proto) == pytest.approx(10.0)
