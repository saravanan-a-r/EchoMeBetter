"""
Resumption (architecture.md §7.9).

A 50–100B token run is multi-week and *will* be interrupted. The property
these tests establish is the strong one: a run stopped and resumed must
produce **bit-identical weights** to the run that was never interrupted.

Anything weaker is not worth much. "Approximately the same" hides a rewound
data iterator, a lost optimizer moment, or an RNG that restarted — none of
which crash, all of which change what the model learns, and none of which
appear in any curve. `test_a_resumed_run_is_identical_to_an_uninterrupted_one`
is therefore an exact-equality check, and the fixtures are pinned to CPU
float32 so it can be.
"""

from __future__ import annotations

import pytest
import torch

from conftest import ListIterator, make_batches
from src.checkpoint import list_checkpoints, step_of
from src.errors import CheckpointError
from src.interop import rephrase_model
from src.loop import Trainer


def build(model_config, training_config, output_dir, **overrides):
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    config = training_config.with_(output_dir=str(output_dir), **overrides)
    return Trainer(model, config, device="cpu")


def weights(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_a_resumed_run_is_identical_to_an_uninterrupted_one(
    tmp_path, model_config, training_config
):
    """
    The load-bearing test of §7.9.

    Six steps straight through, against three steps + a crash + three more
    from the checkpoint. Every weight must match exactly. If the data
    iterator rewound, or AdamW's moments restarted, or the RNG reset, the two
    runs diverge — and none of those failures announces itself.
    """
    settings = dict(
        max_steps=6, gradient_accumulation_steps=1, save_steps=3, logging_steps=100
    )

    uninterrupted = build(model_config, training_config, tmp_path / "whole", **settings)
    all_batches = make_batches(model_config, 6)
    uninterrupted.train(iter(all_batches), data=ListIterator(all_batches))
    expected = weights(uninterrupted.model)

    # The interrupted run: three steps, then a fresh process resumes.
    first_half = build(model_config, training_config, tmp_path / "resumed", **settings)
    source = ListIterator(make_batches(model_config, 6))
    first_half.train(source, data=source, max_steps=3)

    second_half = build(model_config, training_config, tmp_path / "resumed", **settings)
    resumed_source = ListIterator(make_batches(model_config, 6))
    state = second_half.resume(data=resumed_source)
    assert state.global_step == 3
    assert resumed_source.position == 3, "the corpus must not have rewound"

    second_half.train(resumed_source, data=resumed_source)

    actual = weights(second_half.model)
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        assert torch.equal(tensor, expected[name]), name


def test_resumption_restores_the_step_count(tmp_path, model_config, training_config):
    """
    A resumed run that restarts the step counter also restarts the
    learning-rate schedule, so it re-runs warmup and repeats the whole decay —
    weeks of compute spent on a curve nobody chose.
    """
    settings = dict(max_steps=4, gradient_accumulation_steps=1, save_steps=2)
    first = build(model_config, training_config, tmp_path, **settings)
    first.train(make_batches(model_config, 4))

    second = build(model_config, training_config, tmp_path, **settings)
    assert second.state.global_step == 0
    second.resume()
    assert second.state.global_step == 4


def test_resumption_continues_the_learning_rate_curve(
    tmp_path, model_config, training_config
):
    settings = dict(
        max_steps=20, gradient_accumulation_steps=1, save_steps=4, warmup_steps=8
    )
    first = build(model_config, training_config, tmp_path, **settings)
    first.train(make_batches(model_config, 4))

    second = build(model_config, training_config, tmp_path, **settings)
    second.resume()
    report = second.train_step(make_batches(model_config, 1))
    assert report.learning_rate == second.config.learning_rate_at(4)
    assert report.learning_rate > 0.0


def test_resumption_restores_the_optimizer_moments(
    tmp_path, model_config, training_config
):
    settings = dict(max_steps=4, gradient_accumulation_steps=1, save_steps=4)
    first = build(model_config, training_config, tmp_path, **settings)
    first.train(make_batches(model_config, 4))
    original = first.optimizer.state_dict()["state"]

    second = build(model_config, training_config, tmp_path, **settings)
    second.resume()
    restored = second.optimizer.state_dict()["state"]

    assert restored.keys() == original.keys()
    for key in original:
        assert torch.allclose(restored[key]["exp_avg"], original[key]["exp_avg"])
        assert torch.allclose(restored[key]["exp_avg_sq"], original[key]["exp_avg_sq"])


def test_resumption_restores_the_data_position(tmp_path, model_config, training_config):
    """
    §7.9's actual subject. Without this the sampler restarts and the model
    silently re-trains on tokens it has already seen — no error, and a loss
    curve that looks slightly *better* than it should.
    """
    settings = dict(max_steps=3, gradient_accumulation_steps=2, save_steps=3)
    first = build(model_config, training_config, tmp_path, **settings)
    source = ListIterator(make_batches(model_config, 12))
    first.train(source, data=source)
    consumed = source.position

    second = build(model_config, training_config, tmp_path, **settings)
    fresh = ListIterator(make_batches(model_config, 12))
    second.resume(data=fresh)
    assert fresh.position == consumed
    assert consumed > 0


def test_resuming_a_run_that_never_saved_a_data_position_is_refused(
    tmp_path, model_config, training_config
):
    """
    The refusal is the feature. Continuing would rewind the corpus, which is
    invisible; failing costs one command.
    """
    settings = dict(max_steps=2, gradient_accumulation_steps=1, save_steps=2)
    first = build(model_config, training_config, tmp_path, **settings)
    first.train(make_batches(model_config, 2))  # no `data=`

    second = build(model_config, training_config, tmp_path, **settings)
    with pytest.raises(CheckpointError, match="7.9"):
        second.resume(data=ListIterator([]))


def test_resumption_picks_the_latest_checkpoint(tmp_path, model_config, training_config):
    settings = dict(max_steps=6, gradient_accumulation_steps=1, save_steps=2)
    first = build(model_config, training_config, tmp_path, **settings)
    first.train(make_batches(model_config, 6))
    assert [step_of(p) for p in list_checkpoints(tmp_path)] == [2, 4, 6]

    second = build(model_config, training_config, tmp_path, **settings)
    assert second.resume().global_step == 6


def test_an_explicit_checkpoint_can_be_resumed_from(tmp_path, model_config, training_config):
    """Rolling back past a loss spike is a real operation, not a hypothetical."""
    settings = dict(max_steps=6, gradient_accumulation_steps=1, save_steps=2)
    first = build(model_config, training_config, tmp_path, **settings)
    first.train(make_batches(model_config, 6))

    second = build(model_config, training_config, tmp_path, **settings)
    assert second.resume(tmp_path / "checkpoint-2").global_step == 2


def test_a_checkpoint_written_mid_run_is_loadable_by_huggingface(
    tmp_path, model_config, training_config
):
    """
    Every intermediate checkpoint is a HuggingFace model directory, with no
    export step. That is the difference between "we can publish this" and
    "we published this" — and it means a run can be shared while it is still
    running.
    """
    transformers = pytest.importorskip("transformers")

    settings = dict(max_steps=2, gradient_accumulation_steps=1, save_steps=2)
    driver = build(model_config, training_config, tmp_path, **settings)
    driver.train(make_batches(model_config, 2))

    loaded = transformers.T5ForConditionalGeneration.from_pretrained(
        tmp_path / "checkpoint-2"
    )
    assert sum(p.numel() for p in loaded.parameters()) == driver.model.parameter_count()


def test_retention_keeps_the_best_checkpoint_through_a_long_run(
    tmp_path, model_config, training_config
):
    """
    §13: "keep best-by-eval, not last". The best step here is early, and a
    naive most-recent-N policy would delete it well before the run ends.
    """
    settings = dict(
        max_steps=8,
        gradient_accumulation_steps=1,
        save_steps=1,
        eval_steps=1,
        save_total_limit=2,
    )
    driver = build(model_config, training_config, tmp_path, **settings)
    scores = iter([9.0, 0.5] + [5.0] * 10)  # best at step 2
    driver.train(make_batches(model_config, 8), evaluate=lambda: {"loss": next(scores)})

    assert driver.state.best_step == 2
    remaining = [step_of(path) for path in list_checkpoints(tmp_path)]
    assert 2 in remaining, "the best checkpoint in the run was deleted"


def test_the_checkpointed_state_survives_a_resume(tmp_path, model_config, training_config):
    """The best-so-far must not be forgotten, or retention stops protecting it."""
    settings = dict(
        max_steps=4, gradient_accumulation_steps=1, save_steps=4, eval_steps=1
    )
    first = build(model_config, training_config, tmp_path, **settings)
    scores = iter([3.0, 1.0, 2.0, 2.5])
    first.train(make_batches(model_config, 4), evaluate=lambda: {"loss": next(scores)})

    second = build(model_config, training_config, tmp_path, **settings)
    state = second.resume()
    assert state.best_step == 2
    assert state.best_metric == 1.0
    assert len(state.log_history) > 0
