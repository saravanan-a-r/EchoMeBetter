"""
Unit tests for `TrainingConfig` and the YAML it is loaded from.

Two things are being locked down. First, that a mistyped or impossible setting
fails *before* the run starts rather than after a week of it — every refusal
below stands in for a specific way a run can be wrong while looking fine.
Second, that the real `training_config.yml` says what architecture.md §13 says
it should, so the reference configuration and the file that implements it
cannot drift.
"""

from __future__ import annotations

import pytest
import yaml

from conftest import TINY_TRAINING, TRAINING_CONFIG_PATH
from src.config import (
    OURS_ALONE,
    TrainingConfig,
    available_stages,
    build_training_config,
    load_training_config,
)
from src.errors import TrainingConfigError


def config(**overrides) -> TrainingConfig:
    return TrainingConfig(**dict(TINY_TRAINING, **overrides))


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "quick_eval_steps",
        "master_eval_steps",
    ],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_counts_must_be_positive(field, bad):
    with pytest.raises(TrainingConfigError, match=field):
        config(**{field: bad})


@pytest.mark.parametrize("field", ["max_steps", "logging_steps"])
def test_a_boolean_is_not_a_count(field):
    """
    `True` is an `int` in Python and passes a naive `> 0` check. A YAML typo
    that writes `logging_steps: yes` would otherwise become 1.
    """
    with pytest.raises(TrainingConfigError):
        config(**{field: True})


def test_the_learning_rate_must_be_positive():
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        config(learning_rate=0.0)


def test_an_unknown_schedule_is_refused():
    with pytest.raises(TrainingConfigError, match="lr_scheduler_type"):
        config(lr_scheduler_type="triangular")


def test_an_unknown_torch_compile_mode_is_refused():
    """
    `torch.compile` itself only raises this at the first forward call, deep
    into a run — catching the typo here means it fails before a single batch
    runs instead of after the model, optimizer and data pipeline have all
    already spun up.
    """
    with pytest.raises(TrainingConfigError, match="torch_compile_mode"):
        config(torch_compile_mode="not-a-real-mode")


def test_torch_compile_mode_is_optional():
    assert config(torch_compile_mode=None).torch_compile_mode is None


def test_torch_compile_is_off_by_default():
    """
    Matches HuggingFace's own default. Also matters for every stage block and
    fixture that predates this feature: they must keep working unchanged.
    """
    settings = config()
    assert settings.torch_compile is False
    assert settings.torch_compile_backend is None
    assert settings.torch_compile_mode is None


@pytest.mark.parametrize(
    "field", ["weight_decay", "warmup_ratio", "label_smoothing_factor", "min_lr_ratio"]
)
@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_ratios_must_be_in_the_unit_interval(field, bad):
    with pytest.raises(TrainingConfigError, match=field):
        config(**{field: bad})


def test_a_run_that_is_entirely_warmup_is_refused():
    """
    `warmup_steps >= max_steps` means the learning rate ramps for the whole
    run and never reaches its peak. Nothing errors; the model simply trains at
    a fraction of the intended rate and the result is read as "this
    configuration underperforms".
    """
    with pytest.raises(TrainingConfigError, match="entirely warmup"):
        config(max_steps=100, warmup_steps=100)


def test_clipping_cannot_be_disabled_by_setting_it_to_zero():
    with pytest.raises(TrainingConfigError, match="max_grad_norm"):
        config(max_grad_norm=0.0)


def test_fp16_is_refused_with_a_reason():
    """
    Not an oversight. fp16 needs a loss scaler and stalls silently on overflow
    in long runs; bf16 has the same memory saving with float32's exponent
    range. The field exists only so a TrainingArguments export is complete.
    """
    with pytest.raises(TrainingConfigError, match="bf16"):
        config(fp16=True)


def test_load_best_model_at_end_needs_the_best_checkpoint_kept():
    """
    Otherwise retention deletes the checkpoint the run is about to load, and
    the failure happens at the very end of a multi-week run.
    """
    with pytest.raises(TrainingConfigError, match="keep_best_checkpoint"):
        config(load_best_model_at_end=True, keep_best_checkpoint=False)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("save_total_limit", 0),
        ("eval_max_batches", 0),
        ("lr_timescale", 0),
        ("num_cycles", 0.0),
        ("dropout_rate", 1.0),
        ("adam_beta1", 1.0),
        ("adam_epsilon", 0.0),
    ],
)
def test_optional_settings_are_still_validated(field, bad):
    with pytest.raises(TrainingConfigError, match=field):
        config(**{field: bad})


def test_absent_optional_settings_are_accepted():
    assert config(save_total_limit=None, eval_max_batches=None, dropout_rate=None)


# -- the decay phase and the cooldown mixture (item 2) ---------------------


def wsd(**overrides) -> TrainingConfig:
    """A `warmup_stable_decay` config over TINY_TRAINING's 8-step run."""
    settings = dict(
        lr_scheduler_type="warmup_stable_decay",
        lr_decay_steps=2,
        lr_decay_type="linear",
    )
    settings.update(overrides)
    return config(**settings)


def test_a_wsd_run_with_no_decay_phase_is_refused():
    """
    It would be `constant_with_warmup` under a name that promises an ending,
    and the run would hold its peak learning rate to the final step — the
    exact behaviour the schedule was chosen to replace.
    """
    with pytest.raises(TrainingConfigError, match="no decay phase"):
        config(lr_scheduler_type="warmup_stable_decay")


def test_decay_settings_on_a_schedule_without_a_decay_phase_are_refused():
    """
    Silently ignoring them would leave a recorded configuration describing an
    annealing phase that never happened — a manifest that lies about the run
    it documents.
    """
    with pytest.raises(TrainingConfigError, match="no cooldown phase"):
        config(lr_scheduler_type="inverse_sqrt", lr_decay_steps=2)


def test_a_decay_that_leaves_no_stable_phase_is_refused():
    with pytest.raises(TrainingConfigError, match="stable phase"):
        wsd(lr_decay_steps=TINY_TRAINING["max_steps"], warmup_steps=1)


def test_an_unknown_decay_shape_is_refused():
    with pytest.raises(TrainingConfigError, match="lr_decay_type"):
        wsd(lr_decay_type="exponential")


def test_explicit_decay_steps_beat_a_ratio():
    assert wsd(lr_decay_steps=3, lr_decay_ratio=0.5).resolved_decay_steps == 3


def test_a_decay_ratio_resolves_against_max_steps():
    assert wsd(lr_decay_steps=0, lr_decay_ratio=0.5).resolved_decay_steps == 4


def test_a_schedule_without_a_decay_phase_has_no_cooldown_step():
    """
    `None` rather than `max_steps`, so a caller cannot read "no cooldown" as
    "a cooldown that starts at the very end" or, worse, at step 0.
    """
    assert config(lr_scheduler_type="inverse_sqrt").cooldown_start_step is None


def test_a_cooldown_mixture_needs_a_cooldown_to_happen_in():
    with pytest.raises(TrainingConfigError, match="no decay phase"):
        config(lr_scheduler_type="cosine", cooldown_blend={"c4": 1.0})


@pytest.mark.parametrize("blend", [{}, {"c4": 0.0}, {"c4": -1.0}, {"c4": "half"}])
def test_a_malformed_cooldown_mixture_is_refused(blend):
    """
    A zero weight in particular: it says "include this source and never draw
    from it", which is what leaving it out already means, so it is always
    either a typo or a misunderstanding.
    """
    with pytest.raises(TrainingConfigError):
        wsd(cooldown_blend=blend)


def test_no_cooldown_mixture_is_a_valid_wsd_run():
    """
    The two halves of item 2 are separable on purpose: the learning-rate
    decay is a settled, low-risk gain, while which data is worth repeating is
    a judgement. A run may take the first without the second.
    """
    assert wsd().cooldown_blend is None


def test_the_wsd_export_carries_the_decay_into_lr_scheduler_kwargs():
    """
    `transformers.get_wsd_schedule` has no default for `num_decay_steps`, so
    an export that omitted it would not drift from this run's curve — it would
    fail to build a scheduler at all.
    """
    exported = wsd().to_huggingface_training_arguments()
    assert exported["lr_scheduler_kwargs"]["num_decay_steps"] == wsd().resolved_decay_steps
    assert exported["lr_scheduler_kwargs"]["decay_type"] == "linear"


def test_other_schedules_export_no_scheduler_kwargs():
    """
    `lr_scheduler_kwargs` is handed straight to the scheduler factory, and the
    other five would reject keys they have no parameter for.
    """
    assert "lr_scheduler_kwargs" not in config().to_huggingface_training_arguments()


# -- derived ---------------------------------------------------------------


def test_warmup_resolves_from_a_ratio_when_steps_are_not_set():
    assert config(warmup_steps=0, warmup_ratio=0.25, max_steps=400).resolved_warmup_steps == 100


def test_the_learning_rate_curve_comes_from_the_configuration():
    settings = config(
        learning_rate=1e-3,
        warmup_steps=10,
        max_steps=2000,
        lr_scheduler_type="inverse_sqrt",
    )
    assert settings.learning_rate_at(0) == 0.0
    assert settings.learning_rate_at(10) == pytest.approx(1e-3)
    assert settings.learning_rate_at(1000) < 1e-3


# -- serialization ---------------------------------------------------------


def test_a_configuration_round_trips_through_a_dict():
    original = config()
    assert TrainingConfig.from_dict(original.to_dict()) == original


def test_an_unknown_key_is_rejected_not_ignored():
    """
    A silently-ignored key means running with a setting the file says is in
    force and that nothing actually reads.
    """
    with pytest.raises(TrainingConfigError, match="unknown"):
        TrainingConfig.from_dict(dict(TINY_TRAINING, learnign_rate=1e-4))


def test_a_missing_required_key_is_rejected():
    incomplete = dict(TINY_TRAINING)
    del incomplete["learning_rate"]
    with pytest.raises(TrainingConfigError, match="learning_rate"):
        TrainingConfig.from_dict(incomplete)


def test_with_reruns_validation():
    with pytest.raises(TrainingConfigError):
        config().with_(max_grad_norm=-1.0)


# -- the YAML document -----------------------------------------------------


def document(stages, defaults=None, active="a"):
    return {
        "active_stage": active,
        "defaults": dict(TINY_TRAINING, **(defaults or {})),
        "stages": stages,
    }


def test_a_stage_overrides_the_defaults():
    built = build_training_config(document({"a": {"learning_rate": 7e-5}}))
    assert built.learning_rate == 7e-5
    assert built.stage == "a"


def test_a_stage_key_outside_the_allowed_set_is_refused():
    """
    Stage blocks are restricted for the same reason `model_config.yml`'s
    profile blocks are: a key that belongs in `defaults` but was written in a
    stage would be applied to one stage and silently missing from the others.
    """
    with pytest.raises(TrainingConfigError, match="unknown keys"):
        build_training_config(document({"a": {"weight_decay": 0.2}}))


def test_an_unknown_stage_is_refused():
    with pytest.raises(TrainingConfigError, match="unknown stage"):
        build_training_config(document({"a": {}}), stage="b")


def test_a_document_with_no_active_stage_and_no_request_is_refused():
    doc = document({"a": {}})
    del doc["active_stage"]
    with pytest.raises(TrainingConfigError, match="active_stage"):
        build_training_config(doc)


@pytest.mark.parametrize("block", ["defaults", "stages"])
def test_a_missing_block_is_refused(block):
    doc = document({"a": {}})
    del doc[block]
    with pytest.raises(TrainingConfigError, match=block):
        build_training_config(doc)


def test_a_missing_file_is_refused():
    with pytest.raises(TrainingConfigError, match="not found"):
        load_training_config("/nonexistent/training_config.yml")


# -- the real file ---------------------------------------------------------


def test_the_real_file_declares_the_three_stages():
    assert available_stages() == ["pretrain", "rehearsal", "sft"]


@pytest.mark.parametrize("stage", ["pretrain", "sft", "rehearsal"])
def test_every_real_stage_loads(stage):
    assert load_training_config(stage=stage).stage == stage


def test_inverse_sqrt_is_still_the_default_schedule():
    """
    architecture.md §13's choice, and still what a stage gets unless it says
    otherwise: inverse_sqrt never needs to know how long the run is. The
    `pretrain` stage overrides it (see below); the default it overrides
    should not quietly change with it.
    """
    document = yaml.safe_load(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    assert document["defaults"]["lr_scheduler_type"] == "inverse_sqrt"


def test_pretraining_anneals_over_its_final_tenth():
    """
    architecture_improvements.md item 2. Three things have to line up or the
    cooldown is not the phase it claims to be: the schedule, the length of
    the decay, and the step the decay begins at — which is also the step the
    data mixture switches on, so an error here is not only a learning-rate
    error.
    """
    pretrain = load_training_config(stage="pretrain")
    assert pretrain.lr_scheduler_type == "warmup_stable_decay"
    assert pretrain.resolved_decay_steps == pretrain.max_steps // 10
    assert pretrain.cooldown_start_step == pretrain.max_steps - pretrain.resolved_decay_steps
    # Flat at the peak right up to the boundary, then strictly falling, then
    # zero at the horizon. Checked on the real configuration rather than a
    # fixture, because it is the real one that a run will use.
    assert pretrain.learning_rate_at(pretrain.cooldown_start_step - 1) == pretrain.learning_rate
    assert pretrain.learning_rate_at(pretrain.cooldown_start_step) == pretrain.learning_rate
    assert pretrain.learning_rate_at(pretrain.cooldown_start_step + 1) < pretrain.learning_rate
    assert pretrain.learning_rate_at(pretrain.max_steps) == 0.0


def test_the_rehearsal_anneals_too():
    """
    §7.8's rehearsal exists to run the real pipeline end to end before weeks
    of compute go into it. A schedule phase the rehearsal never enters is a
    phase the real run debugs.
    """
    rehearsal = load_training_config(stage="rehearsal")
    assert rehearsal.lr_scheduler_type == "warmup_stable_decay"
    assert rehearsal.cooldown_start_step is not None
    assert rehearsal.cooldown_start_step < rehearsal.max_steps


def test_the_cooldown_mixture_upweights_cli_and_high_quality_prose():
    """
    The data half of item 2. The specific weights are a judgement call and
    will be tuned; what must not silently change is the shape of the
    judgement — that the cooldown is a *different* mixture from the corpus,
    concentrated on CLI text, high-quality prose and code.
    """
    blend = load_training_config(stage="pretrain").cooldown_blend
    assert blend is not None
    cli = sum(weight for source, weight in blend.items() if source.startswith("cli_"))
    prose = blend["fineweb_edu"] + blend["cosmopedia"]
    total = sum(blend.values())
    assert cli / total == pytest.approx(0.3125, abs=0.01)
    assert prose / total == pytest.approx(0.50, abs=0.01)
    assert blend["permissive_code"] / total == pytest.approx(0.1875, abs=0.01)


def test_the_tiny_cli_sources_are_not_weighted_like_the_big_one():
    """
    The one thing in the cooldown blend that is not a matter of taste.

    The five CLI sources differ by four orders of magnitude in size
    (pretrain_corpus/output/token_count_report.json): cli_cheat_sheets holds
    ~67,500 tokens against cli_helper_stack_exchange's ~574 million. Weighting
    them equally would show the model the cheat sheets thousands of times over
    a 10B-token cooldown, which teaches memorization rather than vocabulary.
    This pins the ordering that prevents it.
    """
    blend = load_training_config(stage="pretrain").cooldown_blend
    assert blend["cli_helper_stack_exchange"] > 20 * blend["cli_man_pages"]
    assert blend["cli_man_pages"] > blend["cli_tldr"] > blend["cli_nl2bash"]
    assert blend["cli_nl2bash"] > blend["cli_cheat_sheets"]


def test_sft_uses_cosine_and_a_lower_rate():
    """architecture.md §13: warmup -> cosine, at a lower peak than pretraining."""
    sft = load_training_config(stage="sft")
    pretrain = load_training_config(stage="pretrain")
    assert sft.lr_scheduler_type == "cosine"
    assert sft.learning_rate < pretrain.learning_rate


def test_dropout_is_off_for_pretraining_and_on_for_sft():
    """architecture.md §13's table: 0.0 pretraining, 0.1 SFT."""
    assert load_training_config(stage="pretrain").dropout_rate == 0.0
    assert load_training_config(stage="sft").dropout_rate == 0.1


def test_label_smoothing_is_sft_only():
    """architecture.md §13: no smoothing during pretraining, 0.1 for SFT."""
    assert load_training_config(stage="pretrain").label_smoothing_factor == 0.0
    assert load_training_config(stage="sft").label_smoothing_factor == 0.1


def test_the_pretraining_effective_batch_reaches_a_million_tokens():
    """
    architecture.md §13: ">=1M tokens/step via gradient accumulation". Worked
    against the model's real context length rather than a number repeated
    here, so raising the context without revisiting the batch size fails.
    """
    from src.interop import rephrase_model

    pretrain = load_training_config(stage="pretrain")
    model = rephrase_model.load_model_config(profile="large")
    tokens = (
        pretrain.per_device_train_batch_size
        * pretrain.gradient_accumulation_steps
        * model.max_encoder_length
    )
    assert tokens >= 1_000_000


def test_the_reference_optimizer_settings_are_section_13s():
    """β 0.9/0.95 and wd 0.1 — and 0.95 is not PyTorch's default."""
    pretrain = load_training_config(stage="pretrain")
    assert (pretrain.adam_beta1, pretrain.adam_beta2) == (0.9, 0.95)
    assert pretrain.weight_decay == 0.1


def test_pretraining_warmup_is_one_to_two_percent():
    """architecture.md §13: "warmup ~1-2% of steps"."""
    pretrain = load_training_config(stage="pretrain")
    fraction = pretrain.resolved_warmup_steps / pretrain.max_steps
    assert 0.01 <= fraction <= 0.02


def test_gradient_checkpointing_and_bf16_are_on_for_pretraining():
    """architecture.md §13's memory and precision rows."""
    pretrain = load_training_config(stage="pretrain")
    assert pretrain.gradient_checkpointing is True
    assert pretrain.bf16 is True
    assert pretrain.fp16 is False


def test_resumption_defaults_on():
    """§7.9: a 50-100B token run *will* be interrupted."""
    assert load_training_config(stage="pretrain").resume_from_checkpoint is True


def test_torch_compile_is_off_for_every_stage():
    """
    Item 3's fused SDPA attention is unconditional; `torch.compile` is off.

    It was once on for pretraining, justified as "a fixed shape, fixed
    hardware" run. The shape premise is false: streaming the real corpus
    produces 3,136 distinct (encoder width, decoder width) pairs, because
    length bucketing sizes each batch to its contents and [S]-mode never
    packs. Measured on the A100 40GB vGPU this trains on, every compiled
    configuration runs out of memory before finishing a step — inductor and
    reduce-overhead, batch 8 and 16, with and without gradient checkpointing,
    and under both allocator tunings available here. The fragmentation remedy
    (expandable_segments) needs CUDA virtual-memory APIs the vGPU does not
    expose.

    This asserts every stage, not just pretraining: turning it back on is a
    decision that needs the measurement redone on the hardware of the day, not
    a default that drifts back.
    """
    for stage in ("pretrain", "rehearsal", "sft"):
        config = load_training_config(stage=stage)
        assert config.torch_compile is False, stage
        assert config.torch_compile_mode is None, stage


def test_the_rehearsal_stage_builds_the_base_profile():
    """§7.8: the full-pipeline rehearsal runs on Base (223M), not Large."""
    assert load_training_config(stage="rehearsal").model_profile == "base"


# -- the two eval tiers ----------------------------------------------------


def test_the_master_eval_is_rarer_than_the_quick_one():
    """
    The whole point of two tiers. If they ran at the same cadence there would
    be no reason to hold two sets: the expensive one would be paying for
    precision at a frequency that cannot afford it.
    """
    pretrain = load_training_config(stage="pretrain")
    assert pretrain.master_eval_steps > pretrain.quick_eval_steps


def test_the_master_eval_lands_on_a_checkpoint_step():
    """
    The master tier alone decides which checkpoint is kept, so every step it
    can name as best must be a step a checkpoint was written at. Otherwise
    retention protects a directory that does not exist and
    `load_best_model_at_end` fails at the very end of the run.

    `Trainer._resolve_eval_tiers` refuses this pairing at startup; this checks
    the shipped configuration never presents it in the first place.
    """
    for stage in ("pretrain", "sft", "rehearsal"):
        settings = load_training_config(stage=stage)
        assert settings.master_eval_steps % settings.save_steps == 0, stage


def test_the_rehearsal_actually_reaches_both_tiers():
    """
    §7.8's rehearsal exists to prove the pipeline works before weeks of
    compute are committed to it. A rehearsal whose master-eval cadence is
    longer than the whole rehearsal never exercises the tier that holds
    best-checkpoint authority — so the one thing it would not have rehearsed
    is the one that silently loses the best model.
    """
    rehearsal = load_training_config(stage="rehearsal")
    assert rehearsal.master_eval_steps <= rehearsal.max_steps
    assert rehearsal.quick_eval_steps <= rehearsal.max_steps


def test_every_field_the_yaml_sets_is_a_real_field():
    """
    Guards against a setting that looks configured and is not read by
    anything — the failure mode where someone changes a value in the file and
    nothing happens.
    """
    known = set(TrainingConfig.__dataclass_fields__)
    parsed = yaml.safe_load(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(parsed["defaults"]) - known == set()
    for name, stage in parsed["stages"].items():
        assert set(stage) - known == set(), name


def test_our_own_fields_are_declared_as_ours():
    """
    `OURS_ALONE` is frozen so that adding a field forces a decision: either it
    matches a `transformers.TrainingArguments` name, or it is consciously
    this project's. Silence is what lets a near-miss name like
    `warmup_fraction` ship next to HuggingFace's `warmup_ratio`.
    """
    for name in OURS_ALONE:
        assert name in TrainingConfig.__dataclass_fields__, name
