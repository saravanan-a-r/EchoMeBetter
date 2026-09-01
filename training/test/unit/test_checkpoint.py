"""
Unit tests for checkpoint writing, reading and retention.

Every refusal tested here stands for a way a multi-week run can be destroyed
quietly rather than loudly, which is why they are refusals and not warnings:

  - resuming with no data position rewinds the corpus (§7.9) and the loss
    curve looks *better* afterwards, so nothing prompts anyone to look;
  - resuming with no optimizer state restarts AdamW's moments at zero, which
    costs a loss excursion that nothing in the logs explains;
  - a retention policy that deletes the best-by-eval checkpoint (§13) is
    discovered only when someone goes to publish it.

`test_an_interrupted_write_leaves_no_checkpoint_behind` covers the one that
bites hardest in practice: a half-written directory that `latest_checkpoint`
happily returns, found at the exact moment something has already gone wrong.
"""

from __future__ import annotations

import json

import pytest
import torch

from conftest import ListIterator
from src.checkpoint import (
    OPTIMIZER_FILE,
    RNG_FILE,
    STATE_FILE,
    TRAINING_CONFIG_FILE,
    TrainerState,
    checkpoint_directory,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    step_of,
)
from src.errors import CheckpointError
from src.optimizer import build_optimizer


@pytest.fixture
def saved(tmp_path, tiny_model, training_config):
    """A real checkpoint on disk, with a stepped optimizer behind it."""
    optimizer = build_optimizer(tiny_model, training_config)
    for parameter in tiny_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = TrainerState(global_step=40, tokens_seen=123456)
    data = ListIterator([{}] * 10)
    data.position = 7

    directory = save_checkpoint(
        checkpoint_directory(tmp_path, 40),
        model=tiny_model,
        optimizer=optimizer,
        state=state,
        training_config=training_config,
        data=data,
    )
    return directory, optimizer


# -- what a checkpoint contains --------------------------------------------


def test_a_checkpoint_is_a_huggingface_model_directory(saved):
    """
    The point of the layout: every intermediate checkpoint of a run is
    directly loadable by `from_pretrained`, with no export step and nothing to
    remember at release time.
    """
    directory, _ = saved
    assert (directory / "config.json").is_file()
    assert (directory / "model.safetensors").is_file()


def test_a_checkpoint_also_carries_what_huggingface_ignores(saved):
    directory, _ = saved
    for name in (OPTIMIZER_FILE, STATE_FILE, TRAINING_CONFIG_FILE, RNG_FILE):
        assert (directory / name).is_file(), name


def test_the_directory_is_named_the_way_huggingface_names_them(saved):
    directory, _ = saved
    assert directory.name == "checkpoint-40"
    assert step_of(directory) == 40


def test_the_state_file_is_readable_json_with_the_expected_fields(saved):
    directory, _ = saved
    state = json.loads((directory / STATE_FILE).read_text())
    assert state["global_step"] == 40
    assert state["tokens_seen"] == 123456
    assert state["data_position"] == {"position": 7}


def test_the_data_position_is_captured_at_write_time(saved):
    """
    Not passed in already serialized. One moment at which "where the model
    got to" and "where the corpus got to" are recorded — taking them at
    different times is how a resumed run ends up out of step with itself.
    """
    directory, _ = saved
    assert TrainerState.from_dict(
        json.loads((directory / STATE_FILE).read_text())
    ).data_position == {"position": 7}


def test_an_unserializable_data_position_is_refused_at_write_time(
    tmp_path, tiny_model, training_config
):
    """
    Caught while saving costs one checkpoint; caught while resuming costs the
    run.
    """

    class Unserializable:
        def state_dict(self):
            return {"handle": object()}

        def load_state_dict(self, state):
            pass

    with pytest.raises(CheckpointError, match="JSON-serializable"):
        save_checkpoint(
            checkpoint_directory(tmp_path, 1),
            model=tiny_model,
            optimizer=build_optimizer(tiny_model, training_config),
            state=TrainerState(),
            training_config=training_config,
            data=Unserializable(),
        )


def test_an_interrupted_write_leaves_no_checkpoint_behind(
    tmp_path, tiny_model, training_config, monkeypatch
):
    """
    Written to a staging directory and renamed into place. A directory that
    exists, is returned by `latest_checkpoint` and cannot be loaded would be
    found at the worst possible moment.
    """
    import src.checkpoint as checkpoint_module

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(checkpoint_module.torch, "save", explode)

    with pytest.raises(RuntimeError, match="disk full"):
        save_checkpoint(
            checkpoint_directory(tmp_path, 5),
            model=tiny_model,
            optimizer=build_optimizer(tiny_model, training_config),
            state=TrainerState(),
            training_config=training_config,
        )

    assert list_checkpoints(tmp_path) == []
    assert not any(tmp_path.iterdir())


# -- reading one back ------------------------------------------------------


def test_loading_restores_the_weights(saved, tiny_model, training_config):
    from src.interop import rephrase_model

    directory, _ = saved
    fresh = rephrase_model.build_model(tiny_model.config, verify=False)
    assert not torch.equal(fresh.shared.weight, tiny_model.shared.weight)

    load_checkpoint(directory, model=fresh, optimizer=None)
    assert torch.equal(fresh.shared.weight, tiny_model.shared.weight)


def test_loading_restores_the_optimizer_moments(saved, tiny_model, training_config):
    """
    AdamW starting from zeroed moments after step 40,000 takes a visible loss
    excursion to recover from, and nothing in the logs says why.
    """
    directory, original = saved
    fresh = build_optimizer(tiny_model, training_config)
    assert fresh.state_dict()["state"] == {}

    load_checkpoint(directory, model=tiny_model, optimizer=fresh)
    assert fresh.state_dict()["state"].keys() == original.state_dict()["state"].keys()


def test_loading_restores_the_data_position(saved, tiny_model):
    directory, _ = saved
    data = ListIterator([{}] * 10)
    load_checkpoint(directory, model=tiny_model, optimizer=None, data=data)
    assert data.position == 7


def test_loading_returns_the_recorded_state(saved, tiny_model):
    directory, _ = saved
    state = load_checkpoint(directory, model=tiny_model, optimizer=None)
    assert state.global_step == 40
    assert state.tokens_seen == 123456


def test_resuming_without_a_recorded_data_position_is_refused(
    tmp_path, tiny_model, training_config
):
    """
    The §7.9 refusal. Continuing would restart the corpus from the beginning
    and re-train on tokens already seen — no error, no crash, and a token
    budget that is quietly wrong.
    """
    directory = save_checkpoint(
        checkpoint_directory(tmp_path, 1),
        model=tiny_model,
        optimizer=build_optimizer(tiny_model, training_config),
        state=TrainerState(global_step=1),
        training_config=training_config,
        data=None,
    )
    with pytest.raises(CheckpointError, match="7.9"):
        load_checkpoint(
            directory, model=tiny_model, optimizer=None, data=ListIterator([])
        )


def test_weights_only_loading_is_allowed_explicitly(tmp_path, tiny_model, training_config):
    """The refusal is escapable — `data=None` says "yes, I mean it"."""
    directory = save_checkpoint(
        checkpoint_directory(tmp_path, 1),
        model=tiny_model,
        optimizer=build_optimizer(tiny_model, training_config),
        state=TrainerState(global_step=1),
        training_config=training_config,
    )
    assert load_checkpoint(directory, model=tiny_model, optimizer=None, data=None)


def test_a_missing_optimizer_file_is_refused(saved, tiny_model, training_config):
    directory, _ = saved
    (directory / OPTIMIZER_FILE).unlink()
    with pytest.raises(CheckpointError, match=OPTIMIZER_FILE):
        load_checkpoint(
            directory, model=tiny_model, optimizer=build_optimizer(tiny_model, training_config)
        )


def test_a_missing_weights_file_is_refused(saved, tiny_model):
    directory, _ = saved
    (directory / "model.safetensors").unlink()
    with pytest.raises(CheckpointError, match="model.safetensors"):
        load_checkpoint(directory, model=tiny_model, optimizer=None)


def test_a_missing_checkpoint_is_refused(tmp_path, tiny_model):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "checkpoint-99", model=tiny_model, optimizer=None)


def test_a_corrupt_state_file_is_refused(saved, tiny_model):
    directory, _ = saved
    (directory / STATE_FILE).write_text("{not json")
    with pytest.raises(CheckpointError, match="JSON"):
        load_checkpoint(directory, model=tiny_model, optimizer=None)


def test_the_rng_state_is_restored(saved, tiny_model):
    """
    Dropout and any sampling must continue from where they were, or a resumed
    run diverges from the one it claims to continue for a reason nobody can
    reconstruct.
    """
    directory, _ = saved
    load_checkpoint(directory, model=tiny_model, optimizer=None)
    first = torch.randn(4)
    load_checkpoint(directory, model=tiny_model, optimizer=None)
    assert torch.equal(first, torch.randn(4))


# -- finding and pruning ---------------------------------------------------


def test_checkpoints_are_listed_by_step_not_by_name(tmp_path, tiny_model, training_config):
    """
    Sorting `checkpoint-1000` and `checkpoint-900` as strings puts 1000 first
    and makes `latest_checkpoint` return the wrong one — a resumption that
    silently goes backwards by a hundred steps.
    """
    for step in (900, 1000, 20):
        save_checkpoint(
            checkpoint_directory(tmp_path, step),
            model=tiny_model,
            optimizer=build_optimizer(tiny_model, training_config),
            state=TrainerState(global_step=step),
            training_config=training_config,
        )
    assert [step_of(path) for path in list_checkpoints(tmp_path)] == [20, 900, 1000]
    assert step_of(latest_checkpoint(tmp_path)) == 1000


def test_an_empty_directory_has_no_latest_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert list_checkpoints(tmp_path) == []
    assert list_checkpoints(tmp_path / "missing") == []


def test_unrelated_directories_are_not_mistaken_for_checkpoints(tmp_path):
    (tmp_path / "checkpoint-final").mkdir()
    (tmp_path / "logs").mkdir()
    assert list_checkpoints(tmp_path) == []


def test_a_directory_that_is_not_a_checkpoint_has_no_step(tmp_path):
    with pytest.raises(CheckpointError, match="not a checkpoint"):
        step_of(tmp_path / "logs")


def make_checkpoints(tmp_path, tiny_model, training_config, steps):
    for step in steps:
        save_checkpoint(
            checkpoint_directory(tmp_path, step),
            model=tiny_model,
            optimizer=build_optimizer(tiny_model, training_config),
            state=TrainerState(global_step=step),
            training_config=training_config,
        )


def test_pruning_keeps_the_most_recent(tmp_path, tiny_model, training_config):
    make_checkpoints(tmp_path, tiny_model, training_config, [1, 2, 3, 4, 5])
    prune_checkpoints(tmp_path, save_total_limit=2)
    assert [step_of(path) for path in list_checkpoints(tmp_path)] == [4, 5]


def test_pruning_never_deletes_the_best_checkpoint(tmp_path, tiny_model, training_config):
    """
    architecture.md §13: "keep best-by-eval, not last". Without this
    exception, a limit of 2 on a run whose best checkpoint was step 1 quietly
    discards it, and by the time anyone goes to publish it is gone.
    """
    make_checkpoints(tmp_path, tiny_model, training_config, [1, 2, 3, 4, 5])
    prune_checkpoints(tmp_path, save_total_limit=2, protect_step=1)
    remaining = [step_of(path) for path in list_checkpoints(tmp_path)]
    # The protected checkpoint is kept *in addition to* the limit, matching
    # HuggingFace: the limit governs recent history, and how much of it is
    # retained should not depend on where the best checkpoint happened to be.
    assert remaining == [1, 4, 5]


def test_no_limit_deletes_nothing(tmp_path, tiny_model, training_config):
    make_checkpoints(tmp_path, tiny_model, training_config, [1, 2, 3])
    assert prune_checkpoints(tmp_path, save_total_limit=None) == []
    assert len(list_checkpoints(tmp_path)) == 3


def test_pruning_under_the_limit_deletes_nothing(tmp_path, tiny_model, training_config):
    make_checkpoints(tmp_path, tiny_model, training_config, [1, 2])
    assert prune_checkpoints(tmp_path, save_total_limit=5) == []


# -- the state record ------------------------------------------------------


def test_trainer_state_round_trips():
    state = TrainerState(
        global_step=7,
        tokens_seen=99,
        best_metric=1.5,
        best_step=4,
        data_position={"position": 3},
        log_history=[{"step": 1, "loss": 2.0}],
    )
    assert TrainerState.from_dict(state.to_dict()) == state


def test_trainer_state_ignores_fields_a_future_version_added():
    """
    A checkpoint written by a later version must still be readable by this
    one — refusing would mean an upgrade that cannot be rolled back.
    """
    state = TrainerState.from_dict({"global_step": 3, "some_future_field": 9})
    assert state.global_step == 3
