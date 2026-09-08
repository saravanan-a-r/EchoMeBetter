"""
The synthetic rewrite task, as the assembled pipeline actually runs it
(architecture_improvements.md item 4).

`rewrite/test/` proves the corruption and the framing; this file proves the
task is wired into `PretrainBatchSource` the way `training_config.yml` says:
only inside the cooldown, at the configured share of batches, one batch one
kind, resumable, and — the regression lock that matters most — invisible to
every run that does not ask for it.
"""

from __future__ import annotations

import functools
import json

import pytest
import yaml

import pretrain

pytestmark = pytest.mark.skipif(
    not pretrain.DEFAULT_TOKENIZER_DIR.joinpath("spm.model").is_file(),
    reason="tokenizer not built: tokenizer/training/output/spm.model missing",
)

REWRITE = pretrain.rewrite.REWRITE_MODE

# Whole documents rather than single sentences: the task skips anything with
# fewer than eight corruptible words, and a rewrite target is a document.
DOCUMENTS = [
    "The quarterly report showed strong growth in the European market. The team "
    "expects it to continue through the next two quarters.",
    "The server crashed because the backup connection failed to respond, and the "
    "on-call engineer had to restart it by hand at three in the morning.",
    "The committee reviewed the roadmap and agreed on the priorities for the next "
    "quarter, although nobody was entirely happy with the schedule.",
    "Yesterday the team voted to approve the budget after a long deliberation, "
    "which surprised nobody who had been in the room for the discussion.",
]


@pytest.fixture
def sourced_corpus_dir(tmp_path):
    """One directory per source, as `pretrain_corpus/` lays the corpus out."""
    root = tmp_path / "corpus"
    for source in ("bulk", "special"):
        directory = root / source
        directory.mkdir(parents=True)
        (directory / f"{source}-w00-00000.jsonl").write_text(
            "\n".join(json.dumps({"text": f"{source}: {text}"}) for text in DOCUMENTS * 20)
            + "\n",
            encoding="utf-8",
        )
    return root


def build_source(
    corpus_dir,
    *,
    cooldown_start_step=None,
    blend=None,
    rewrite_blend=None,
    rewrite_share=0.0,
    seed=1234,
    max_length=256,
):
    """A `PretrainBatchSource` over real files and the real tokenizer."""
    token_map = pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    specials = pretrain.ul2.SpecialTokens.from_token_map(token_map)
    ul2_config = pretrain.ul2.UL2Config(max_source_length=max_length, max_target_length=max_length)
    objective = pretrain.ul2.UL2Objective(ul2_config, specials, pretrain.ul2.MixtureTelemetry())
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    files = pretrain.corpus_pkg.discover_corpus_files(corpus_dir)
    grouped = pretrain.corpus_pkg.group_files_by_source(files, corpus_dir)

    cooldown = None
    if blend is not None:
        cooldown = pretrain.corpus_pkg.build_blended_corpus(grouped, sp, blend, seed=seed + 104_729)

    rewrite_source = None
    if rewrite_blend is not None:
        records = pretrain.corpus_pkg.build_blended_corpus(
            grouped, sp, rewrite_blend, seed=seed + 15_485_863
        )
        rewrite_source = pretrain.rewrite.RewriteExampleSource(
            records,
            pretrain.rewrite.Corruptor(),
            functools.partial(pretrain.corpus_pkg.tokenizer_interop.encode_text, sp),
            pretrain.rewrite.RewriteTokens.from_token_map(token_map),
            max_source_length=max_length,
            max_target_length=max_length,
            min_source_length=ul2_config.min_source_length,
            seed=seed + 32_452_843,
        )

    return pretrain.PretrainBatchSource(
        pretrain.corpus_pkg.TokenizedCorpus(files, sp, seed=seed),
        objective,
        specials,
        batch_size=2,
        data_seed=seed,
        cooldown_corpus=cooldown,
        cooldown_start_step=cooldown_start_step,
        rewrite_source=rewrite_source,
        rewrite_share=rewrite_share,
    )


def batches(source, count):
    return [next(source) for _ in range(count)]


def is_rewrite(batch):
    return batch["modes"][0] == REWRITE


# -- where and how often ------------------------------------------------------


def test_rewrite_batches_appear_only_inside_the_cooldown(sourced_corpus_dir):
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=10, rewrite_blend={"bulk": 1.0}, rewrite_share=0.5
    )
    source.set_step(0)
    assert not source.rewrite_active
    assert not any(is_rewrite(batch) for batch in batches(source, 40))

    source.set_step(10)
    assert source.rewrite_active
    assert any(is_rewrite(batch) for batch in batches(source, 40))


def test_every_batch_is_one_kind(sourced_corpus_dir):
    """A rewrite batch is all rewrite rows; a denoising batch has none."""
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=0, rewrite_blend={"bulk": 1.0}, rewrite_share=0.5
    )
    source.set_step(0)
    for batch in batches(source, 60):
        assert len(set(batch["modes"])) == 1


def test_the_share_of_rewrite_batches_is_the_configured_one(sourced_corpus_dir):
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=0, rewrite_blend={"bulk": 1.0}, rewrite_share=0.3
    )
    source.set_step(0)
    drawn = batches(source, 400)
    share = sum(is_rewrite(batch) for batch in drawn) / len(drawn)
    assert share == pytest.approx(0.3, abs=0.07)


def test_the_task_is_recorded_in_the_objectives_telemetry(sourced_corpus_dir):
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=0, rewrite_blend={"bulk": 1.0}, rewrite_share=0.5
    )
    source.set_step(0)
    batches(source, 40)
    report = source.objective.telemetry.report()
    assert REWRITE in report["modes"]
    assert report["modes"][REWRITE]["examples"] == source.rewrite_source.pairs_emitted


# -- what the rows hold -------------------------------------------------------


def test_rewrite_rows_are_framed_and_target_the_original_document(sourced_corpus_dir):
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=0, rewrite_blend={"bulk": 1.0}, rewrite_share=1.0 - 1e-9
    )
    source.set_step(0)
    sp = source.main_corpus.sp
    tokens = source.rewrite_source.tokens
    originals = {f"bulk: {text}" for text in DOCUMENTS}

    corrupted_rows = 0
    for batch in batches(source, 20):
        assert is_rewrite(batch)
        for encoder, mask, labels in zip(
            batch["encoder_input_ids"], batch["encoder_attention_mask"], batch["labels"]
        ):
            real = [token for token, keep in zip(encoder, mask) if keep]
            assert real[:2] == [tokens.style_id, tokens.open_id]
            assert real[-2:] == [tokens.close_id, tokens.eos_id]

            target = [token for token in labels if token >= 0][:-1]  # drop -100s and EOS
            original = sp.decode(target)
            assert original in originals

            corrupted = sp.decode(real[2:-2])
            corrupted_rows += corrupted != original
    assert corrupted_rows > 0, "twenty batches never produced a corrupted row"


# -- invisible when not asked for --------------------------------------------


def test_the_stream_before_the_cooldown_is_identical_with_and_without_the_task(
    sourced_corpus_dir,
):
    """
    The regression lock. A run that configures the task must, until its
    cooldown begins, produce byte-for-byte the batches a run without it
    produces: the extra draw happens only once the task is active.
    """
    without = build_source(sourced_corpus_dir)
    with_task = build_source(
        sourced_corpus_dir, cooldown_start_step=1000, rewrite_blend={"bulk": 1.0}, rewrite_share=0.5
    )
    without.set_step(0)
    with_task.set_step(0)
    assert batches(without, 30) == batches(with_task, 30)


def test_a_run_without_the_task_never_draws_it(sourced_corpus_dir):
    source = build_source(sourced_corpus_dir, cooldown_start_step=2, blend={"special": 1.0})
    source.set_step(1000)
    assert source.in_cooldown and not source.rewrite_active
    assert not any(is_rewrite(batch) for batch in batches(source, 20))


@pytest.mark.parametrize(
    "half",
    [
        dict(rewrite_source=object(), cooldown_start_step=1),
        dict(rewrite_share=0.5, cooldown_start_step=1),
        dict(rewrite_source=object(), rewrite_share=1.0, cooldown_start_step=1),
        dict(rewrite_source=object(), rewrite_share=0.5),
    ],
)
def test_half_a_task_is_refused(half):
    with pytest.raises(ValueError, match="go together|below 1.0"):
        pretrain.PretrainBatchSource(object(), object(), object(), batch_size=1, data_seed=0, **half)


# -- resumption ---------------------------------------------------------------


def test_resuming_inside_the_cooldown_reproduces_the_uninterrupted_batches(sourced_corpus_dir):
    kwargs = dict(cooldown_start_step=0, rewrite_blend={"bulk": 1.0}, rewrite_share=0.5)

    uninterrupted = build_source(sourced_corpus_dir, **kwargs)
    uninterrupted.set_step(0)
    expected = batches(uninterrupted, 80)

    first = build_source(sourced_corpus_dir, **kwargs)
    first.set_step(0)
    got = batches(first, 37)
    saved = json.loads(json.dumps(first.state_dict()))
    assert "rewrite_source" in saved

    second = build_source(sourced_corpus_dir, **kwargs)
    second.load_state_dict(saved)
    second.set_step(0)
    got += batches(second, 43)

    assert got == expected


def test_a_checkpoint_written_before_the_task_existed_still_loads(sourced_corpus_dir):
    old = build_source(sourced_corpus_dir)
    state = json.loads(json.dumps(old.state_dict()))
    assert "rewrite_source" not in state

    new = build_source(
        sourced_corpus_dir, cooldown_start_step=2, rewrite_blend={"bulk": 1.0}, rewrite_share=0.5
    )
    new.load_state_dict(state)  # must not raise
    assert new.rewrite_source.pairs_emitted == 0


# -- the trainer and the entrypoint drive it ----------------------------------


def test_the_trainer_runs_through_a_cooldown_with_rewrite_batches(
    sourced_corpus_dir, tiny_model_config_path, tiny_training_config_path
):
    """
    End to end through `Trainer.train`, including per-mode loss on a mode the
    trainer has never heard of. `max_length=64` matches the tiny model's
    context; the documents above fit it with their frame.
    """
    model_config = pretrain.rephrase_model.load_model_config(tiny_model_config_path, profile="tiny")
    training_config = pretrain.training_pkg.load_training_config(
        tiny_training_config_path, stage="tiny"
    ).with_(max_steps=6, gradient_accumulation_steps=2, log_per_mode_loss=True)

    source = build_source(
        sourced_corpus_dir,
        cooldown_start_step=2,
        rewrite_blend={"bulk": 1.0},
        rewrite_share=0.9,
        max_length=64,
    )
    model = pretrain.rephrase_model.build_model(model_config)
    trainer = pretrain.training_pkg.Trainer(model, training_config, device="cpu")

    trainer.train(source, data=source, max_steps=6)
    assert source.rewrite_active
    assert source.rewrite_source.pairs_emitted > 0


def test_pretrain_wires_the_task_from_the_configuration(
    sourced_corpus_dir, tiny_model_config_path, tiny_training_config_path, capsys
):
    document = yaml.safe_load(tiny_training_config_path.read_text(encoding="utf-8"))
    document["stages"]["tiny"].update(
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=0,
        lr_decay_ratio=0.5,
        lr_decay_type="linear",
        cooldown_blend={"special": 1.0},
        rewrite_share=0.5,
        rewrite_blend={"bulk": 1.0},
    )
    config_path = tiny_training_config_path.with_name("rewrite_training_config.yml")
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    exit_code = pretrain.main(
        [
            "--corpus", str(sourced_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(config_path),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "rewrite task (from step 2): 50% of batches, <style:grammar>" in out


def test_pretrain_refuses_a_blend_naming_a_source_the_corpus_lacks(
    sourced_corpus_dir, tiny_model_config_path, tiny_training_config_path, capsys
):
    """Eagerly, before a single step — not at step 180,000."""
    document = yaml.safe_load(tiny_training_config_path.read_text(encoding="utf-8"))
    document["stages"]["tiny"].update(
        lr_scheduler_type="warmup_stable_decay",
        warmup_steps=0,
        lr_decay_ratio=0.5,
        lr_decay_type="linear",
        rewrite_share=0.5,
        rewrite_blend={"fineweb_edu": 1.0},
    )
    config_path = tiny_training_config_path.with_name("bad_training_config.yml")
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    exit_code = pretrain.main(
        [
            "--corpus", str(sourced_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(config_path),
        ]
    )
    assert exit_code == 2
    assert "rewrite task:" in capsys.readouterr().err
