"""
`Trainer._to_device` — device transfer, and the one thing about it that is
not obvious: `non_blocking=True` is only safe on CUDA with pinned host
memory. On MPS it has been observed to return before the copy completes, so
a dict comprehension building several tensors back-to-back can have a later
`.to()` call start overwriting the destination buffer of an earlier one
before that transfer is read — not a crash, a *silently wrong* tensor. See
`loop.py`'s `_to_device` docstring comment for the full account; this is the
test that would have caught it before it reached a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conftest import make_batch
from src.loop import Trainer


def test_to_device_uses_blocking_transfers_off_cuda(model_config, tiny_model, training_config):
    """
    Device-agnostic version of the regression below: whatever backend is
    running the suite, a CPU-backed `Trainer` must never ask for
    `non_blocking=True` — CUDA is the only backend it is proven safe on.
    """
    trainer = Trainer(tiny_model, training_config, device="cpu")
    batch = make_batch(model_config)

    seen_kwargs = []
    real_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        if "non_blocking" in kwargs:
            seen_kwargs.append(kwargs["non_blocking"])
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = spy_to
    try:
        trainer._to_device(batch)
    finally:
        torch.Tensor.to = real_to

    assert seen_kwargs, "expected _to_device to call Tensor.to(..., non_blocking=...)"
    assert all(value is False for value in seen_kwargs)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available on this machine"
)
def test_to_device_does_not_corrupt_batches_on_mps(model_config, tiny_model, training_config):
    """
    The actual regression: build several distinct tensors in the same
    comprehension and transfer them to a real MPS device, then verify every
    key still holds *its own* values afterward.

    Before the fix, this reliably reproduced one key's tensor being silently
    replaced by another's (observed: `decoder_input_ids` overwritten with
    `labels`' padding value), which is exactly the class of bug this
    project's "fail loudly, never silently" principle exists to prevent —
    except this one produced no failure at all on its own, just wrong
    numbers reaching the model.
    """
    trainer = Trainer(tiny_model, training_config, device="mps")
    batch = make_batch(model_config, batch_size=4, source_length=20, target_length=16, seed=3)

    prepared = trainer._to_device(batch)

    for key in ("encoder_input_ids", "decoder_input_ids", "encoder_attention_mask",
                "decoder_attention_mask", "labels"):
        expected = torch.as_tensor(batch[key], dtype=torch.long)
        actual = prepared[key].to("cpu")
        assert torch.equal(actual, expected), f"{key} was corrupted in transfer to MPS"


# -- checkpoint observers ---------------------------------------------------
#
# The one extension point for work that is *about* a run rather than part of
# it (`ownership/` hangs the provenance hash chain off it). Both tests below
# guard silent failures: a notification that quietly stops happening, and an
# observer that quietly takes the run down with it.


def test_save_notifies_observers_with_the_directory_it_wrote(
    tmp_path, tiny_model, training_config
):
    """
    A dropped notification is invisible — the run trains on perfectly, and the
    only symptom is a provenance record that stopped growing, noticed when it
    is needed and cannot be recreated.
    """
    seen = []

    class Recorder:
        def on_checkpoint_saved(self, directory, step):
            seen.append((Path(directory), step, (Path(directory) / "config.json").is_file()))

        def close(self):
            pass

    config = training_config.with_(output_dir=str(tmp_path), save_total_limit=None)
    trainer = Trainer(
        tiny_model, config, device="cpu", checkpoint_observers=[Recorder()]
    )
    written = trainer.save(1000)

    # Called once, with the real directory, already complete when handed over.
    assert seen == [(written, 1000, True)]


def test_a_failing_observer_does_not_end_the_run(tmp_path, tiny_model, training_config):
    """
    Observers do bookkeeping; the run is the thing that matters. A multi-week
    run must not die because a hash file was unwritable — but the failure has
    to reach the log, or it is merely hidden instead of harmless.
    """
    logged = []

    class Broken:
        def on_checkpoint_saved(self, directory, step):
            raise RuntimeError("disk full")

        def close(self):
            pass

    config = training_config.with_(output_dir=str(tmp_path), save_total_limit=None)
    trainer = Trainer(
        tiny_model,
        config,
        device="cpu",
        on_log=logged.append,
        checkpoint_observers=[Broken()],
    )

    written = trainer.save(1000)  # must not raise

    assert written.is_dir()
    assert any("disk full" in str(record.get("checkpoint_observer_error")) for record in logged)


# -- optimizer-step observers -----------------------------------------------
#
# The companion hook, fired once per step instead of once per checkpoint
# (`ownership/` holds the embedding signature on it). The step *number* is the
# thing worth pinning: an observer with a cadence of its own counts in these,
# and an off-by-one would put every scheduled action on the wrong step for the
# whole run without anything failing.


def test_optimizer_step_observers_see_the_step_that_was_just_completed(
    model_config, tiny_model, training_config
):
    seen = []

    class Recorder:
        def on_optimizer_step(self, step):
            seen.append(step)

    config = training_config.with_(max_steps=3, gradient_accumulation_steps=1)
    trainer = Trainer(
        tiny_model, config, device="cpu", optimizer_step_observers=[Recorder()]
    )
    batches = [make_batch(model_config, seed=i) for i in range(3)]
    trainer.train(batches)

    # The numbers `train` logs, not the pre-increment counter.
    assert seen == [1, 2, 3]
    assert trainer.state.global_step == 3


def test_a_skipped_step_does_not_notify(model_config, tiny_model, training_config):
    """
    A non-finite gradient drops the step, so the weights did not move. An
    observer told otherwise would adjust weights the optimizer never applied
    — and for a technique that measures drift, act on drift that never
    happened.
    """
    seen = []

    class Recorder:
        def on_optimizer_step(self, step):
            seen.append(step)

    config = training_config.with_(max_steps=1, gradient_accumulation_steps=1)
    trainer = Trainer(
        tiny_model, config, device="cpu", optimizer_step_observers=[Recorder()]
    )
    # Every label masked out: `accumulate` returns no tokens and the step is
    # dropped before the optimizer runs.
    batch = make_batch(model_config, seed=0)
    batch = {**batch, "labels": [[-100] * len(row) for row in batch["labels"]]}
    trainer.train([batch])

    assert seen == []
    assert trainer.skipped_steps == 1


def test_a_failing_step_observer_does_not_end_the_run(
    model_config, tiny_model, training_config
):
    logged = []

    class Broken:
        def on_optimizer_step(self, step):
            raise RuntimeError("bad solve")

    config = training_config.with_(max_steps=1, gradient_accumulation_steps=1)
    trainer = Trainer(
        tiny_model,
        config,
        device="cpu",
        on_log=logged.append,
        optimizer_step_observers=[Broken()],
    )
    trainer.train([make_batch(model_config, seed=0)])  # must not raise

    assert trainer.state.global_step == 1
    assert any(
        "bad solve" in str(record.get("optimizer_step_observer_error"))
        for record in logged
    )
