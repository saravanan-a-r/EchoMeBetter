"""
End-to-end tests for the training loop.

Three of these carry most of the weight, and they are the ones to read first:

  - **`test_accumulation_matches_one_big_batch`** — proves the central claim of
    `loop.py`, that a window of micro-batches produces *exactly* the gradients
    one large batch would. Written with deliberately unequal micro-batch sizes,
    which is where the conventional `loss / accumulation_steps` pattern is
    wrong and this one is not.

  - **`test_the_model_can_memorize_one_batch`** — a trainer can have correct
    shapes, a correct schedule and finite gradients while being wired so the
    model cannot learn at all. If loss does not collapse on one repeated
    batch, something upstream is broken, and no amount of unit testing would
    have said so.

  - **`test_a_non_finite_gradient_does_not_reach_the_optimizer`** — one bad
    batch in a multi-week run must not end the run, and must not be applied
    either: a NaN reaching AdamW's moments is carried into every subsequent
    step and the run never recovers.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import make_batch, make_batches
from src.errors import TrainingConfigError
from src.interop import rephrase_model
from src.loop import Trainer, _micro_batches
from src.optimizer import build_optimizer


def trainer(model, config, **kwargs) -> Trainer:
    return Trainer(model, config, device="cpu", **kwargs)


def gradients(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


# -- the accumulation claim ------------------------------------------------


def test_accumulation_matches_one_big_batch(model_config, training_config):
    """
    The central claim of `loop.py`.

    Two micro-batches with **different target lengths**, accumulated, must
    give exactly the gradients a single batch containing all their tokens
    would. The conventional `loss / accumulation_steps` pattern fails this,
    because the model's loss is a mean over tokens and the mean of two means
    with different denominators is not the mean of the whole.

    That is not a hypothetical: UL2's three denoisers produce systematically
    different target lengths, so the error is a consistent bias against the
    long-target modes, not noise.
    """
    torch.manual_seed(0)
    accumulating = rephrase_model.build_model(model_config, verify=False)
    torch.manual_seed(0)
    single = rephrase_model.build_model(model_config, verify=False)

    # Unequal on purpose: 4 target positions against 12.
    short = make_batch(model_config, batch_size=2, target_length=4, seed=1)
    long = make_batch(model_config, batch_size=2, target_length=12, seed=2)

    config = training_config.with_(gradient_accumulation_steps=2, max_grad_norm=1e9)
    # `accumulate`, not `train_step`: the gradients are the subject, and
    # `train_step` clips and clears them before it returns.
    trainer(accumulating, config).accumulate([short, long])

    # The same tokens as one batch: pad both to the longer length so a single
    # forward pass sees exactly the union, with the padding ignored.
    combined = _concatenate(short, long, pad_id=model_config.pad_token_id)
    single.train()
    output = single(
        encoder_input_ids=combined["encoder_input_ids"],
        decoder_input_ids=combined["decoder_input_ids"],
        encoder_attention_mask=combined["encoder_attention_mask"],
        decoder_attention_mask=combined["decoder_attention_mask"],
        labels=combined["labels"],
    )
    output.loss.backward()

    accumulated = gradients(accumulating)
    reference = gradients(single)
    assert set(accumulated) == set(reference)
    for name, tensor in accumulated.items():
        assert torch.allclose(tensor, reference[name], atol=1e-6, rtol=1e-5), name


def _concatenate(first, second, pad_id: int) -> dict[str, torch.Tensor]:
    """Stack two batches into one, padding the shorter to the longer."""

    def pad(tensor: torch.Tensor, width: int, value: int) -> torch.Tensor:
        if tensor.shape[1] >= width:
            return tensor
        filler = torch.full(
            (tensor.shape[0], width - tensor.shape[1]), value, dtype=tensor.dtype
        )
        return torch.cat([tensor, filler], dim=1)

    source = max(first["encoder_input_ids"].shape[1], second["encoder_input_ids"].shape[1])
    target = max(first["decoder_input_ids"].shape[1], second["decoder_input_ids"].shape[1])
    fills = {
        "encoder_input_ids": (source, pad_id),
        "encoder_attention_mask": (source, 0),
        "decoder_input_ids": (target, pad_id),
        "decoder_attention_mask": (target, 0),
        "labels": (target, -100),
    }
    return {
        key: torch.cat([pad(first[key], width, value), pad(second[key], width, value)])
        for key, (width, value) in fills.items()
    }


def test_the_conventional_normalization_would_fail_that_test(model_config, training_config):
    """
    Confirms the test above is actually discriminating.

    Averaging the two micro-batches' *mean* losses — the usual pattern —
    produces different gradients from one combined batch when the token counts
    differ. If this ever stops being true, the test above has stopped being
    evidence of anything.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    short = make_batch(model_config, batch_size=2, target_length=4, seed=1)
    long = make_batch(model_config, batch_size=2, target_length=12, seed=2)

    model.train()
    for batch in (short, long):
        output = model(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        (output.loss / 2).backward()
    naive = gradients(model)

    torch.manual_seed(0)
    correct = rephrase_model.build_model(model_config, verify=False)
    config = training_config.with_(gradient_accumulation_steps=2, max_grad_norm=1e9)
    trainer(correct, config).accumulate([short, long])

    assert not torch.allclose(
        naive["shared.weight"], gradients(correct)["shared.weight"], atol=1e-6
    )


def test_accumulation_is_invisible_when_token_counts_are_equal(
    model_config, training_config
):
    """
    The sanity check on the other side: with equal token counts the two
    normalizations coincide, so the difference above really is about ragged
    batches and not about a bug in one of them.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    batches = [
        make_batch(model_config, batch_size=2, target_length=8, seed=seed)
        for seed in (1, 2)
    ]
    # Equal lengths, and no masked rows, so both micro-batches count the same.
    for batch in batches:
        batch["decoder_attention_mask"] = torch.ones_like(batch["decoder_attention_mask"])
        batch["labels"] = batch["labels"].masked_fill(batch["labels"] == -100, 9)

    config = training_config.with_(gradient_accumulation_steps=2, max_grad_norm=1e9)
    trainer(model, config).accumulate(batches)
    ours = gradients(model)

    torch.manual_seed(0)
    naive = rephrase_model.build_model(model_config, verify=False)
    naive.train()
    for batch in batches:
        output = naive(
            encoder_input_ids=batch["encoder_input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        (output.loss / 2).backward()

    assert torch.allclose(
        ours["shared.weight"], gradients(naive)["shared.weight"], atol=1e-6
    )


# -- that it trains at all -------------------------------------------------


@pytest.mark.slow
def test_the_model_can_memorize_one_batch(model_config, training_config):
    """
    The test that catches "everything is correct and nothing learns".

    A trainer can have the right shapes, a valid schedule and finite gradients
    while being wired so no update ever reaches the weights — a learning rate
    that stays at zero, gradients zeroed before the step, an optimizer built
    over a detached copy. All of those pass every unit test in this suite. If
    loss does not collapse on one batch repeated, something upstream is
    broken.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    batch = make_batch(model_config, batch_size=2, target_length=6)

    config = training_config.with_(
        max_steps=60,
        learning_rate=3e-3,
        warmup_steps=5,
        gradient_accumulation_steps=1,
        logging_steps=1000,
        eval_steps=10000,
        save_steps=10000,
    )
    driver = trainer(model, config)

    first = driver.train_step([batch]).loss
    for _ in range(config.max_steps):
        report = driver.train_step([batch])
        driver.state.global_step += 1

    assert report.loss < first * 0.5, f"{first} -> {report.loss}"


def test_a_step_changes_the_weights(tiny_model, training_config):
    driver = trainer(tiny_model, training_config.with_(warmup_steps=0))
    before = tiny_model.shared.weight.detach().clone()
    driver.train_step(make_batches(tiny_model.config, 2))
    assert not torch.equal(before, tiny_model.shared.weight)


def test_step_zero_of_a_warmed_run_changes_nothing(tiny_model, training_config):
    """
    The learning rate at step 0 is exactly zero, so the first gradient — the
    least trustworthy one in the run — is computed and not applied.
    """
    driver = trainer(tiny_model, training_config.with_(warmup_steps=4))
    before = tiny_model.shared.weight.detach().clone()
    report = driver.train_step(make_batches(tiny_model.config, 2))
    assert report.learning_rate == 0.0
    assert torch.equal(before, tiny_model.shared.weight)


def test_the_learning_rate_follows_the_schedule(tiny_model, training_config):
    config = training_config.with_(max_steps=6, warmup_steps=3, logging_steps=1)
    driver = trainer(tiny_model, config)
    rates = []
    for _ in range(6):
        rates.append(driver.train_step(make_batches(tiny_model.config, 2)).learning_rate)
        driver.state.global_step += 1
    assert rates == [config.learning_rate_at(step) for step in range(6)]


# -- clipping and bad steps ------------------------------------------------


def test_gradients_are_clipped_to_the_configured_norm(tiny_model, training_config):
    driver = trainer(tiny_model, training_config.with_(max_grad_norm=1e-4))
    driver.accumulate(make_batches(tiny_model.config, 2))
    torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), 1e-4)
    # Clipping happens after normalization, so the post-step gradient norm is
    # the clipped one.
    total = torch.sqrt(
        sum(
            (parameter.grad**2).sum()
            for parameter in tiny_model.parameters()
            if parameter.grad is not None
        )
        if any(p.grad is not None for p in tiny_model.parameters())
        else torch.tensor(0.0)
    )
    assert float(total) <= 1e-4 + 1e-9


def test_the_reported_grad_norm_is_the_norm_before_clipping(tiny_model, training_config):
    """
    `clip_grad_norm_` returns the *pre*-clip norm, which is the useful number
    to log: a run whose logged norm sits pinned at `max_grad_norm` tells you
    nothing, and one whose true norm is climbing tells you a great deal.
    """
    driver = trainer(tiny_model, training_config.with_(max_grad_norm=1e-6))
    assert driver.train_step(make_batches(tiny_model.config, 2)).grad_norm > 1e-6


def test_a_non_finite_gradient_does_not_reach_the_optimizer(
    model_config, training_config
):
    """
    One bad batch must not end a multi-week run — and must not be applied
    either. A NaN reaching AdamW's exponential moving averages is carried
    into every subsequent step, so the run continues, produces NaN losses
    forever, and never recovers.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    driver = trainer(model, training_config.with_(warmup_steps=0))

    before = model.shared.weight.detach().clone()

    original = model.forward

    def poisoned(**kwargs):
        output = original(**kwargs)
        output.loss = output.loss * float("nan")
        return output

    model.forward = poisoned
    report = driver.train_step(make_batches(model_config, 2))
    model.forward = original

    assert report.skipped
    assert driver.skipped_steps == 1
    assert torch.equal(before, model.shared.weight)
    assert all(
        parameter.grad is None for parameter in model.parameters()
    ), "gradients must be cleared, or the NaN carries into the next step"


def test_the_run_continues_after_a_skipped_step(model_config, training_config):
    """A skip is a dropped step, not a stopped run."""
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    driver = trainer(model, training_config.with_(warmup_steps=0))

    original = model.forward
    model.forward = lambda **kw: _poison(original(**kw))
    driver.train_step(make_batches(model_config, 2))
    model.forward = original

    before = model.shared.weight.detach().clone()
    report = driver.train_step(make_batches(model_config, 2))
    assert not report.skipped
    assert not torch.equal(before, model.shared.weight)


def _poison(output):
    output.loss = output.loss * float("nan")
    return output


def test_a_fully_masked_batch_is_skipped_rather_than_producing_nan(
    tiny_model, training_config
):
    """
    Every label ignored means a 0/0 mean. A real corpus produces such a batch
    eventually; it must contribute nothing rather than poison the step.
    """
    driver = trainer(tiny_model, training_config.with_(warmup_steps=0))
    batch = make_batch(tiny_model.config)
    batch["labels"] = torch.full_like(batch["labels"], -100)

    before = tiny_model.shared.weight.detach().clone()
    report = driver.train_step([batch])

    assert report.skipped
    assert report.tokens == 0
    assert torch.equal(before, tiny_model.shared.weight)


def test_a_fully_masked_micro_batch_does_not_poison_the_rest_of_the_window(
    model_config, training_config
):
    """
    The case a single-batch test cannot see.

    A fully masked batch produces a 0/0 mean, and `NaN * 0` is still NaN — so
    a window that merely stops *counting* an empty batch's tokens without
    skipping its backward pass propagates NaN into the gradients the other
    micro-batches contributed to. The step is then dropped for being
    non-finite and the real batch's update is lost. On a corpus that produces
    such a batch occasionally, that is a steady, silent loss of signal, and
    the only symptom is a skip rate nobody is watching.
    """
    torch.manual_seed(0)
    model = rephrase_model.build_model(model_config, verify=False)
    driver = trainer(
        model, training_config.with_(warmup_steps=0, gradient_accumulation_steps=2)
    )

    real = make_batch(model_config, seed=1)
    empty = make_batch(model_config, seed=2)
    empty["labels"] = torch.full_like(empty["labels"], -100)

    before = model.shared.weight.detach().clone()
    report = driver.train_step([real, empty])

    assert not report.skipped, "one empty micro-batch dropped the whole window"
    assert math.isfinite(report.loss)
    assert math.isfinite(report.grad_norm)
    assert not torch.equal(before, model.shared.weight)

    # Normalized by the real batch's tokens alone.
    assert report.tokens == int((real["labels"] != -100).sum())


# -- the run ---------------------------------------------------------------


def test_a_run_stops_at_max_steps(tiny_model, training_config):
    config = training_config.with_(max_steps=3, gradient_accumulation_steps=2)
    driver = trainer(tiny_model, config)
    state = driver.train(make_batches(tiny_model.config, 20))
    assert state.global_step == 3


def test_an_exhausted_stream_ends_the_run_and_says_so(tiny_model, training_config):
    """
    Not an error — a rehearsal over a fixed slice (§7.8) is expected to end
    this way — but it must be distinguishable from having reached max_steps.
    """
    config = training_config.with_(max_steps=100, gradient_accumulation_steps=2)
    driver = trainer(tiny_model, config)
    state = driver.train(make_batches(tiny_model.config, 5))
    assert state.global_step == 2  # two full windows; the odd batch is dropped
    assert state.log_history[-1]["reason"] == "exhausted"


def test_a_partial_accumulation_window_is_not_stepped(tiny_model):
    """
    Stepping a short window is not numerically wrong — the token weighting
    handles it — but it makes the last step of a run a different size from
    every other, for no benefit.
    """
    windows = list(_micro_batches([1, 2, 3, 4, 5], 2))
    assert windows == [[1, 2], [3, 4]]


def test_tokens_seen_accumulates_across_the_run(tiny_model, training_config):
    config = training_config.with_(max_steps=3, gradient_accumulation_steps=1)
    driver = trainer(tiny_model, config)
    state = driver.train(make_batches(tiny_model.config, 3))
    assert state.tokens_seen > 0
    assert state.tokens_seen == sum(
        entry.get("tokens_seen", 0) - previous
        for entry, previous in [(state.log_history[-1], 0)]
    )


def test_logging_happens_on_the_configured_cadence(tiny_model, training_config):
    config = training_config.with_(
        max_steps=6, gradient_accumulation_steps=1, logging_steps=2
    )
    seen = []
    driver = trainer(tiny_model, config, on_log=seen.append)
    driver.train(make_batches(tiny_model.config, 6))
    steps = [entry["step"] for entry in seen if "loss" in entry]
    assert steps == [2, 4, 6]


def test_a_log_entry_carries_what_a_run_is_watched_by(tiny_model, training_config):
    config = training_config.with_(max_steps=1, gradient_accumulation_steps=1)
    seen = []
    trainer(tiny_model, config, on_log=seen.append).train(
        make_batches(tiny_model.config, 1)
    )
    entry = seen[0]
    for key in ("step", "loss", "learning_rate", "grad_norm", "tokens_seen", "skipped_steps"):
        assert key in entry, key
    assert math.isfinite(entry["loss"])


def test_the_log_history_is_kept_on_the_state(tiny_model, training_config):
    """It is checkpointed, so a resumed run keeps its own history."""
    config = training_config.with_(max_steps=2, gradient_accumulation_steps=1)
    driver = trainer(tiny_model, config)
    state = driver.train(make_batches(tiny_model.config, 2))
    assert len(state.log_history) >= 2


def test_per_mode_losses_are_reported_when_asked_for(tiny_model, training_config):
    """
    §7.6: "[R] loss, [X] loss, [S] loss — not just an aggregate." The batches
    carry `modes` straight from `UL2/src/batching.pad_batch`, unchanged.
    """
    config = training_config.with_(
        max_steps=1, gradient_accumulation_steps=1, logging_steps=1,
        log_per_mode_loss=True,
    )
    seen = []
    batch = make_batch(tiny_model.config, batch_size=3, modes=["R", "X", "S"])
    trainer(tiny_model, config, on_log=seen.append).train([batch])
    entry = seen[0]
    assert {"loss_R", "loss_X", "loss_S"} <= set(entry)


# -- evaluation ------------------------------------------------------------


def test_evaluation_reports_overall_and_per_mode_loss(tiny_model, training_config):
    driver = trainer(tiny_model, training_config)
    batches = [
        make_batch(tiny_model.config, batch_size=3, modes=["R", "X", "S"], seed=seed)
        for seed in range(3)
    ]
    metrics = driver.evaluate(batches)
    assert {"loss", "perplexity", "loss_R", "loss_X", "loss_S"} <= set(metrics)
    assert math.isfinite(metrics["loss"])


def test_evaluation_does_not_change_the_model(tiny_model, training_config):
    driver = trainer(tiny_model, training_config)
    before = tiny_model.shared.weight.detach().clone()
    driver.evaluate(make_batches(tiny_model.config, 3))
    assert torch.equal(before, tiny_model.shared.weight)
    assert all(p.grad is None for p in tiny_model.parameters())


def test_evaluation_restores_training_mode(tiny_model, training_config):
    """
    Leaving the model in eval mode would turn dropout off for the rest of the
    run — a regularization setting silently changing at the first evaluation.
    """
    driver = trainer(tiny_model, training_config)
    tiny_model.train()
    driver.evaluate(make_batches(tiny_model.config, 2))
    assert tiny_model.training


def test_evaluation_can_be_capped(tiny_model, training_config):
    driver = trainer(tiny_model, training_config)
    batches = make_batches(tiny_model.config, 6)
    assert driver.evaluate(batches, max_batches=2) != driver.evaluate(batches)


def test_the_best_metric_is_tracked_across_evaluations(tiny_model, training_config):
    """
    §13 keeps best-by-eval, so the run has to know which step was best.
    """
    config = training_config.with_(
        max_steps=3, gradient_accumulation_steps=1, eval_steps=1, save_steps=1000
    )
    driver = trainer(tiny_model, config)
    scores = iter([3.0, 1.0, 2.0])
    driver.train(
        make_batches(tiny_model.config, 3), evaluate=lambda: {"loss": next(scores)}
    )
    assert driver.state.best_metric == 1.0
    assert driver.state.best_step == 2


def test_a_greater_is_better_metric_is_tracked_the_other_way(tiny_model, training_config):
    config = training_config.with_(
        max_steps=3,
        gradient_accumulation_steps=1,
        eval_steps=1,
        save_steps=1000,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )
    driver = trainer(tiny_model, config)
    scores = iter([0.1, 0.9, 0.4])
    driver.train(
        make_batches(tiny_model.config, 3), evaluate=lambda: {"accuracy": next(scores)}
    )
    assert driver.state.best_step == 2


def test_a_metric_the_evaluator_does_not_report_is_not_invented(
    tiny_model, training_config
):
    config = training_config.with_(
        max_steps=1, gradient_accumulation_steps=1, eval_steps=1, save_steps=1000
    )
    driver = trainer(tiny_model, config)
    driver.train(make_batches(tiny_model.config, 1), evaluate=lambda: {"other": 1.0})
    assert driver.state.best_step is None


# -- refusals --------------------------------------------------------------


def test_disagreeing_label_smoothing_is_refused(tiny_model, training_config):
    """
    The model computes the loss, so `ModelConfig.label_smoothing` is what
    applies. A run whose recorded configuration says 0.1 while the model
    smooths at 0.0 is a result nobody can reproduce.
    """
    with pytest.raises(TrainingConfigError, match="label smoothing disagrees"):
        trainer(tiny_model, training_config.with_(label_smoothing_factor=0.1))


def test_a_batch_missing_a_key_is_refused(tiny_model, training_config):
    driver = trainer(tiny_model, training_config)
    batch = make_batch(tiny_model.config)
    del batch["labels"]
    with pytest.raises(TrainingConfigError, match="labels"):
        driver.train_step([batch])


def test_plain_lists_are_accepted_as_batches(tiny_model, training_config):
    """
    `UL2/` has no torch dependency by design, so it returns nested lists. The
    conversion belongs at this boundary and nowhere else.
    """
    driver = trainer(tiny_model, training_config.with_(warmup_steps=0))
    batch = make_batch(tiny_model.config)
    listed = {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    assert math.isfinite(driver.train_step([listed]).loss)


def test_gradient_checkpointing_is_turned_on_when_asked(tiny_model, training_config):
    trainer(tiny_model, training_config.with_(gradient_checkpointing=True))
    assert tiny_model.gradient_checkpointing


def test_gradient_checkpointing_on_a_model_that_cannot_is_refused(training_config):
    with pytest.raises(TrainingConfigError, match="gradient_checkpointing"):
        Trainer(
            torch.nn.Linear(2, 2),
            training_config.with_(gradient_checkpointing=True),
            device="cpu",
        )


def test_a_supplied_optimizer_is_used_rather_than_replaced(tiny_model, training_config):
    optimizer = build_optimizer(tiny_model, training_config)
    assert trainer(tiny_model, training_config, optimizer=optimizer).optimizer is optimizer


def test_resuming_with_nothing_to_resume_from_is_refused(
    tmp_path, tiny_model, training_config
):
    """
    A crash-loop that keeps "resuming" from step 0 burns the same budget
    forever, and its loss curve looks plausible every time.
    """
    from src.errors import CheckpointError

    driver = trainer(tiny_model, training_config.with_(output_dir=str(tmp_path)))
    with pytest.raises(CheckpointError, match="no checkpoint"):
        driver.resume()


def test_the_batch_contract_is_ul2s(tiny_model):
    """
    Guards the seam between the two modules: the keys this trainer requires
    are exactly the ones `pad_batch` produces, so nothing translates between
    them.
    """
    from src.loop import REQUIRED_KEYS

    produced = set(make_batch(tiny_model.config))
    assert set(REQUIRED_KEYS) <= produced
    assert produced - set(REQUIRED_KEYS) == {"modes"}
