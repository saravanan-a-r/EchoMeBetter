"""
End-to-end wiring tests for `pretrain.py` and `pretrain_smoke_test.py` — the
scripts that assemble `corpus/`, `UL2/`, `model/` and `training/` into an
actual run.

Every other test suite in this project proves its own package correct in
isolation; the four packages' own conventions (§ each README's "Scope"
table) are explicit that none of them owns the wiring between the others.
These tests are what makes "the pipeline actually runs end to end" a checked
fact rather than a claim — the same role `training/test/integration/
test_ul2_interop.py` plays for UL2 + model + training, extended one layer
further out to include real corpus files and the real tokenizer.

A tiny model (see `conftest.py`) keeps every run here in the
milliseconds-to-seconds range; a run against the real Base or Large profile
belongs to `pretrain_smoke_test.py` itself (§7.8), run by a human before
committing real compute, not to a suite that runs on every change.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml

import pretrain
import pretrain_smoke_test

pytestmark = pytest.mark.skipif(
    not pretrain.DEFAULT_TOKENIZER_DIR.joinpath("spm.model").is_file(),
    reason="tokenizer not built: tokenizer/training/output/spm.model missing",
)


def test_pretrain_runs_a_few_steps_end_to_end(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, capsys
):
    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "model profile: tiny" in out


def test_pretrain_runs_with_a_frozen_eval_corpus(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path
):
    """
    architecture.md §7.6's frozen-eval path (`build_frozen_eval_batches`,
    `Trainer.evaluate`) is only exercised when `--eval-corpus` is passed —
    covered separately from the plain training-only run above. `--max-steps`
    is set to the tiny config's `eval_steps` (100 -> overridden low here via
    a dedicated stage is unnecessary; `--max-steps` alone can't cross that
    threshold, so this asserts the *construction* path — building the frozen
    eval batches and wiring `evaluate` into `Trainer.train` — never crashes,
    which is what a wiring bug here would do regardless of whether the
    threshold is ever reached.
    """
    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--eval-corpus", str(mini_corpus_dir),
            "--eval-max-examples", "8",
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--max-steps", "2",
        ]
    )
    assert exit_code == 0


def test_pretrain_runs_with_both_eval_tiers(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, capsys
):
    """
    The two-tier path end to end: both frozen sets are built from real corpus
    files, wired into `Trainer.train` as `EvalTier`s, and — unlike the
    single-tier test above, whose cadence is never reached — actually
    evaluated, because the tiny config's cadences are 1 and 2 (conftest.py).
    """
    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--master-eval-corpus", str(mini_corpus_dir),
            "--quick-eval-corpus", str(mini_corpus_dir),
            "--master-eval-max-examples", "8",
            "--quick-eval-max-examples", "8",
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--max-steps", "2",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "selects the best checkpoint" in out
    assert "trend only" in out


def test_pretrain_refuses_mixing_the_single_and_tiered_eval_flags(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, capsys
):
    """
    Both forms claim authority over best-checkpoint selection, so accepting
    them together would let the command decide silently which held-out set
    ranked the checkpoints.
    """
    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--eval-corpus", str(mini_corpus_dir),
            "--master-eval-corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
        ]
    )
    assert exit_code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_the_master_tier_is_the_one_wired_to_select_checkpoints(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, monkeypatch
):
    """
    The one thing this wiring can get wrong without crashing: which tier is
    handed `drives_best_checkpoint`. Backwards, the run would rank its
    checkpoints on the small noisy set and quietly keep the wrong weights —
    a failure with no symptom until the finished model underperforms.

    Asserted on the arguments `pretrain.py` actually passes to `Trainer.train`,
    rather than on a loss value, because the loss is what the tiering is
    *supposed* to make meaningful and would be circular evidence here.
    """
    captured = {}

    def capture(self, batches, *, evaluate=None, eval_tiers=None, data=None, max_steps=None):
        captured["evaluate"] = evaluate
        captured["tiers"] = list(eval_tiers or [])
        return self.state

    monkeypatch.setattr(pretrain.training_pkg.Trainer, "train", capture)

    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--master-eval-corpus", str(mini_corpus_dir),
            "--quick-eval-corpus", str(mini_corpus_dir),
            "--master-eval-max-examples", "8",
            "--quick-eval-max-examples", "8",
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
        ]
    )

    assert exit_code == 0
    assert captured["evaluate"] is None
    tiers = {tier.name: tier for tier in captured["tiers"]}
    assert set(tiers) == {"master", "quick"}
    assert tiers["master"].drives_best_checkpoint is True
    assert tiers["quick"].drives_best_checkpoint is False
    # ...and each on the cadence the configuration asked for (conftest.py).
    assert tiers["master"].every_steps == 2
    assert tiers["quick"].every_steps == 1


def test_pretrain_rejects_a_missing_corpus(tiny_model_config_path, tiny_training_config_path):
    with pytest.raises(pretrain.corpus_pkg.CorpusConfigError):
        pretrain.main(
            [
                "--corpus", "/does/not/exist",
                "--stage", "tiny",
                "--device", "cpu",
                "--model-config", str(tiny_model_config_path),
                "--training-config", str(tiny_training_config_path),
            ]
        )


def test_pretrain_reports_missing_tokenizer_artifacts(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--tokenizer-dir", str(tmp_path / "no_such_tokenizer"),
        ]
    )
    assert exit_code == 2


def test_pretrain_batch_source_is_resumable_end_to_end(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path
):
    """
    The property architecture.md §7.9 is actually about, proven for the
    assembled pipeline rather than for `corpus/`'s iterator in isolation:
    resuming a `PretrainBatchSource` reproduces the exact examples an
    uninterrupted run would have produced next.
    """
    import yaml

    doc = yaml.safe_load(tiny_training_config_path.read_text())
    training_config = pretrain.training_pkg.build_training_config(doc, stage="tiny")

    model_config = pretrain.rephrase_model.load_model_config(
        tiny_model_config_path, profile="tiny"
    )
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(
        max_source_length=model_config.max_encoder_length,
        max_target_length=model_config.max_decoder_length,
    )
    objective = pretrain.ul2.UL2Objective(ul2_config, specials)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    files = pretrain.corpus_pkg.discover_corpus_files(mini_corpus_dir)

    def make_source():
        corpus = pretrain.corpus_pkg.TokenizedCorpus(files, sp, seed=training_config.data_seed)
        return pretrain.PretrainBatchSource(
            corpus, objective, specials, batch_size=2, data_seed=training_config.data_seed
        )

    straight = make_source()
    straight_batches = [next(straight) for _ in range(6)]

    crashed = make_source()
    before = [next(crashed) for _ in range(3)]
    state = crashed.state_dict()

    resumed = make_source()
    resumed.load_state_dict(state)
    after = [next(resumed) for _ in range(3)]

    def ids(batches):
        return [b["encoder_input_ids"] for b in batches]

    assert ids(before) + ids(after) == ids(straight_batches)


def test_smoke_test_passes_on_a_sound_tiny_pipeline(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    exit_code = pretrain_smoke_test.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--steps", "4",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--output-dir", str(tmp_path / "smoke_out"),
            "--keep",
        ]
    )
    assert exit_code == 0


@pytest.fixture
def cooldown_stage_config(tiny_training_config_path, tmp_path):
    """
    A stage shaped like `pretrain`: warmup_stable_decay, a cooldown mixture
    and the rewrite task, with the decay phase declared as a *ratio* of a long
    run — exactly the shape whose cooldown a short rehearsal would never
    reach unless the smoke test rescales it.
    """
    document = yaml.safe_load(tiny_training_config_path.read_text(encoding="utf-8"))
    document["stages"]["like_pretrain"] = dict(
        model_profile="tiny",
        output_dir=str(tmp_path / "runs"),
        max_steps=200,
        lr_scheduler_type="warmup_stable_decay",
        warmup_ratio=0.01,
        lr_decay_ratio=0.1,
        lr_decay_type="linear",
        cooldown_blend={"special": 1.0},
        # The real stage's share, and a real accumulation width. Both matter
        # to what this test proves: the length-bucketing pools are filled
        # before the boundary, so the cooldown mixture only starts delivering
        # once they drain, which takes a realistic number of micro-batches
        # per step rather than the one a tiny fixture would otherwise use.
        rewrite_share=0.2,
        rewrite_blend={"bulk": 1.0},
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
    )
    path = tiny_training_config_path.with_name("cooldown_training_config.yml")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture
def sourced_corpus_dir(tmp_path):
    """One directory per source, the way `pretrain_corpus/` lays the corpus out."""
    documents = [
        "The quarterly report showed strong growth in the European market.",
        "The server crashed because the backup connection failed to respond.",
        "The committee reviewed the roadmap and agreed on next quarter's plans.",
        "A small encoder-decoder transformer can still learn fluent English.",
    ]
    root = tmp_path / "sourced_corpus"
    for source in ("bulk", "special"):
        directory = root / source
        directory.mkdir(parents=True)
        (directory / f"{source}-w00-00000.jsonl").write_text(
            "\n".join(json.dumps({"text": f"{source}: {text}"}) for text in documents * 30)
            + "\n",
            encoding="utf-8",
        )
    return root


def test_the_smoke_test_reaches_a_stages_cooldown_and_rewrite_task(
    sourced_corpus_dir, tiny_model_config_path, cooldown_stage_config, tmp_path, capsys
):
    """
    The gap that made the smoke test unable to de-risk the real run.

    `pretrain`'s cooldown starts at step 180,000 of 200,000, so a rehearsal of
    a few steps used to end long before the mixture switch or the rewrite task
    executed — the two newest pieces of the pipeline, unreachable by the check
    that exists to reach them. The smoke test now compresses the schedule so
    both run, and fails if either does not.
    """
    exit_code = pretrain_smoke_test.main(
        [
            "--corpus", str(sourced_corpus_dir),
            "--stage", "like_pretrain",
            "--steps", "8",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(cooldown_stage_config),
            "--output-dir", str(tmp_path / "smoke_cooldown"),
            "--keep",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0, out

    # The schedule was compressed, not skipped: the boundary lands inside the run.
    assert "cooldown boundary placed at step 4" in out
    assert "cooldown mixture delivered" in out
    assert "rewrite task produced" in out


def test_the_smoke_test_exercises_both_eval_tiers_when_given_them(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path, capsys
):
    """
    Held-out evaluation is optional on the command line, so it was never
    rehearsed. Supplying either eval flag now builds both tiers, runs them on
    their own cadences and requires the selecting tier to reach `best_step` —
    the mechanism that decides which checkpoint the whole run keeps.
    """
    eval_corpus = tmp_path / "heldout.jsonl"
    eval_corpus.write_text(
        "\n".join(
            json.dumps({"text": f"Held-out sentence number {index} about servers and logs."})
            for index in range(40)
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = pretrain_smoke_test.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--steps", "4",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--quick-eval-corpus", str(eval_corpus),
            "--master-eval-corpus", str(eval_corpus),
            "--output-dir", str(tmp_path / "smoke_eval"),
            "--keep",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "eval tiers scored: ['master', 'quick']" in out
    assert "best checkpoint selected at step" in out


def test_the_smoke_test_reports_a_blend_naming_a_missing_source(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    """
    `build_pipeline` refuses this, and the smoke test has to surface it as
    "could not run" rather than as a failed check — it is a typo in the
    configuration, not a broken pipeline. In the real run the same refusal is
    what stops a bad blend being discovered at step 180,000.
    """
    document = yaml.safe_load(tiny_training_config_path.read_text(encoding="utf-8"))
    document["stages"]["bad_blend"] = dict(
        model_profile="tiny",
        output_dir=str(tmp_path / "runs"),
        max_steps=200,
        lr_scheduler_type="warmup_stable_decay",
        lr_decay_ratio=0.1,
        cooldown_blend={"a_source_that_does_not_exist": 1.0},
    )
    path = tiny_training_config_path.with_name("bad_blend_config.yml")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    exit_code = pretrain_smoke_test.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "bad_blend",
            "--steps", "4",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(path),
            "--output-dir", str(tmp_path / "smoke_bad"),
        ]
    )
    assert exit_code == 2


def test_smoke_test_fails_loudly_on_a_missing_tokenizer(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    exit_code = pretrain_smoke_test.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--steps", "4",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--tokenizer-dir", str(tmp_path / "no_such_tokenizer"),
        ]
    )
    assert exit_code == 2


def _capacity_source(mini_corpus_dir, tiny_model_config_path, tiny_training_config_path,
                     capacity):
    doc = yaml.safe_load(tiny_training_config_path.read_text())
    training_config = pretrain.training_pkg.build_training_config(doc, stage="tiny")
    model_config = pretrain.rephrase_model.load_model_config(
        tiny_model_config_path, profile="tiny"
    )
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(
        max_source_length=model_config.max_encoder_length,
        max_target_length=model_config.max_decoder_length,
    )
    objective = pretrain.ul2.UL2Objective(ul2_config, specials)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    files = pretrain.corpus_pkg.discover_corpus_files(mini_corpus_dir)
    corpus = pretrain.corpus_pkg.TokenizedCorpus(files, sp, seed=training_config.data_seed)
    return pretrain.PretrainBatchSource(
        corpus, objective, specials, batch_size=2,
        data_seed=training_config.data_seed, capacity=capacity,
    )


def test_a_capacity_bounds_the_batches_the_real_pipeline_emits(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path
):
    """
    The bound has to hold on batches the assembled pipeline actually produces,
    not only on synthetic examples: it is what stands between a multi-week run
    and an out-of-memory failure on a shape the corpus produced and no sample
    happened to contain.

    Both budgets are checked against the *padded* widths, because those are
    the rectangles that get allocated.
    """
    capacity = pretrain.ul2.BatchCapacity(
        max_encoder_tokens=512,
        max_decoder_tokens=512,
        pad_to_multiple_of=pretrain.PAD_TO_MULTIPLE_OF,
    )
    source = _capacity_source(
        mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, capacity
    )
    sizes = set()
    for _ in range(40):
        batch = next(source)
        rows = len(batch["encoder_input_ids"])
        encoder_width = len(batch["encoder_input_ids"][0])
        decoder_width = len(batch["labels"][0])
        assert rows * encoder_width <= capacity.max_encoder_tokens
        assert rows * decoder_width <= capacity.max_decoder_tokens
        assert len(batch["modes"]) == rows
        sizes.add(rows)

    # The mechanism is only doing anything if the row count moves with the
    # width; a single size would mean it had collapsed to a fixed batch.
    assert len(sizes) > 1


def test_a_capacity_pipeline_is_still_resumable(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path
):
    """
    Resumption is what a multi-week run rests on, and a capacity changes the
    batcher's cut -- so the round trip is checked again here rather than
    assumed to carry over from the fixed-size case.
    """
    capacity = pretrain.ul2.BatchCapacity(
        max_encoder_tokens=512,
        max_decoder_tokens=512,
        pad_to_multiple_of=pretrain.PAD_TO_MULTIPLE_OF,
    )
    make = lambda: _capacity_source(
        mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, capacity
    )
    straight = make()
    expected = [next(straight)["encoder_input_ids"] for _ in range(6)]

    crashed = make()
    before = [next(crashed)["encoder_input_ids"] for _ in range(3)]
    state = crashed.state_dict()

    resumed = make()
    resumed.load_state_dict(state)
    after = [next(resumed)["encoder_input_ids"] for _ in range(3)]

    assert before + after == expected


def test_a_master_cadence_longer_than_the_run_is_refused(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    """
    The master tier is the only thing that selects a checkpoint, so a cadence
    longer than the run means none is ever selected — and the run still
    finishes, still logs, and still leaves checkpoints behind. Whichever one
    happened to be last is then kept, which is exactly the silent
    wrong-weights failure the tiering exists to prevent.

    Caught before any compute is spent rather than discovered at the end of
    one, which for the real stage is weeks later.
    """
    doc = yaml.safe_load(tiny_training_config_path.read_text())
    # `save_steps` is a shared setting, not a per-stage one, and the master
    # cadence has to stay a multiple of it.
    doc["stages"]["tiny"]["master_eval_steps"] = 999
    doc["defaults"]["save_steps"] = 999
    path = tmp_path / "long_master.yml"
    path.write_text(yaml.safe_dump(doc))

    with pytest.raises(pretrain.PipelineError, match="master_eval_steps"):
        pretrain.main(
            [
                "--corpus", str(mini_corpus_dir),
                "--master-eval-corpus", str(mini_corpus_dir),
                "--master-eval-max-examples", "8",
                "--stage", "tiny",
                "--device", "cpu",
                "--model-config", str(tiny_model_config_path),
                "--training-config", str(path),
            ]
        )


def test_a_run_without_eval_tiers_ignores_the_master_cadence(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    """
    The other half of the rule above: with no master corpus the cadence
    governs nothing, so it must not be refused. `pretrain_smoke_test.py`
    depends on this — it shrinks a run to a handful of steps and leaves the
    stage's own master cadence far above it.
    """
    doc = yaml.safe_load(tiny_training_config_path.read_text())
    # `save_steps` is a shared setting, not a per-stage one, and the master
    # cadence has to stay a multiple of it.
    doc["stages"]["tiny"]["master_eval_steps"] = 999
    doc["defaults"]["save_steps"] = 999
    path = tmp_path / "no_tiers.yml"
    path.write_text(yaml.safe_dump(doc))

    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(path),
        ]
    )
    assert exit_code == 0


# -- the progress log and the run log ---------------------------------------


def test_the_progress_log_never_raises_on_a_record_it_cannot_format():
    """
    `ProgressLog` runs inside the training loop as `Trainer.on_log`, so an
    exception from it would end a multi-week run over a log line. A record
    whose keys or values it does not expect -- someone renames a key in
    `loop.py`, a value arrives as the wrong type -- must be written out raw
    instead, and the next record must still format normally.
    """
    lines = []
    log = pretrain.ProgressLog(
        max_steps=100, warmup_steps=1, cooldown_start_step=90, save_steps=50,
        output_dir="runs/x", device_type="cpu", write=lines.append,
    )
    log.start(0, 0)
    log({"step": 10, "loss": "not a number"})
    log({"no_step_at_all": True})
    log({"step": 20, "loss": 2.0, "perplexity": 7.39, "learning_rate": 1e-4,
         "grad_norm": 0.5, "tokens_seen": 1000})

    assert "could not be formatted" in lines[1]
    assert "could not be formatted" in lines[2]
    assert "step     20/100" in lines[3] and "loss 2.0000" in lines[3]


def test_the_run_log_captures_the_whole_run_and_restores_the_terminal(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    """
    `--log-file` has to capture what an operator needs after the fact -- the
    banner naming what actually ran, a progress line per logging step, the
    checkpoint writes, and how the run ended -- and it must hand stdout and
    stderr back afterwards, or everything printed after `run` returns would
    keep going to a file that is now closed.
    """
    log_path = tmp_path / "logs" / "pretrain.log"
    stdout, stderr = __import__("sys").stdout, __import__("sys").stderr

    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--log-file", str(log_path),
        ]
    )

    assert exit_code == 0
    assert __import__("sys").stdout is stdout and __import__("sys").stderr is stderr
    text = log_path.read_text()
    assert "pretrain.py started" in text
    assert "model profile: tiny" in text
    assert "training from step 0" in text
    assert "step      1/4" in text
    assert "checkpoint: saving" in text
    assert "finished at step 4 (max_steps)" in text
    # A real run always has encoder tokens, so "input 0" means the count broke.
    step_line = next(line for line in text.splitlines() if "step      1/4" in line)
    assert "tokens: input " in step_line and "trained on " in step_line
    assert "tokens: input 0 " not in step_line
    assert "exited with status 0" in text
    # stdout and stderr are both mirrored; each line must land exactly once.
    assert text.count("pretrain.py started") == 1
    assert text.count("step      1/4") == 1


def test_a_run_log_is_appended_to_not_overwritten(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path
):
    """
    A resumed run must continue the same log. Overwriting it would erase the
    record of everything before the interruption -- which is exactly what
    someone reading the log after a crash is looking for.
    """
    log_path = tmp_path / "pretrain.log"
    log_path.write_text("an earlier session\n")

    pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--log-file", str(log_path),
        ]
    )

    text = log_path.read_text()
    assert text.startswith("an earlier session\n")
    assert "pretrain.py started" in text


def test_pretrain_writes_a_tensorboard_dashboard(
    mini_corpus_dir, tiny_model_config_path, tiny_training_config_path, tmp_path, capsys
):
    """
    `--tensorboard-dir` end to end: every dashboard section fills from a real
    run. The dashboard reads the trainer's records by key and its optimizer
    through hooks, so a renamed record key or a changed parameter-group layout
    would not fail anything else — it would just leave a chart silently
    empty for the weeks of a run. The tiny stage's quick-eval cadence of 1
    runs the probes and histograms on every step, so a few steps reach all of it.
    """
    plugin_event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.plugin_event_accumulator"
    )
    layout_pb2 = pytest.importorskip("tensorboard.plugins.custom_scalar.layout_pb2")
    log_dir = tmp_path / "tensorboard"
    exit_code = pretrain.main(
        [
            "--corpus", str(mini_corpus_dir),
            "--master-eval-corpus", str(mini_corpus_dir),
            "--quick-eval-corpus", str(mini_corpus_dir),
            "--master-eval-max-examples", "8",
            "--quick-eval-max-examples", "8",
            "--stage", "tiny",
            "--device", "cpu",
            "--model-config", str(tiny_model_config_path),
            "--training-config", str(tiny_training_config_path),
            "--max-steps", "4",
            "--tensorboard-dir", str(log_dir),
        ]
    )
    assert exit_code == 0
    assert "WARNING dashboard" not in capsys.readouterr().out

    # Read as TensorBoard reads it: every summary migrated to its plugin's form.
    events = plugin_event_accumulator.EventAccumulator(str(log_dir))
    events.Reload()
    scalars = set(events.PluginTagToContent("scalars"))

    for tag in (
        "loss/train", "loss/perplexity",
        "optim/learning_rate", "optim/grad_norm", "optim/skipped_steps",
        "tokens/total", "throughput/seconds_per_step", "throughput/eta_days",
        "eval/quick/loss", "eval/master/loss", "eval/best_metric", "eval/best_step",
        "position/shuffle_cosine",
    ):
        assert tag in scalars, f"{tag} missing from the dashboard"
    assert any(tag.startswith("position/") and tag.endswith("_absmax") for tag in scalars)
    for group in ("decayed", "no_decay", "position_bias", "query"):
        for section in ("update_ratio", "weight_rms", "grad_norm", "group_lr"):
            assert f"{section}/{group}" in scalars, f"{section}/{group} missing"
        assert f"weights/{group}" in events.PluginTagToContent("histograms")

    texts = set(events.PluginTagToContent("text"))  # each under "<tag>/text_summary"
    assert {f"samples/{mode}/text_summary" for mode in ("R", "X", "S")} <= texts
    assert "guide/text_summary" in texts

    # Every metric explains itself, and every curated chart draws something:
    # a new metric without a catalogue entry would show as a bare tag, and a
    # renamed one would leave its chart empty.
    undescribed = [tag for tag in scalars if not events.SummaryMetadata(tag).summary_description]
    assert not undescribed, f"no catalogue entry for {undescribed}"

    layout = layout_pb2.Layout()
    config = events.Tensors("custom_scalars__config__")[0].tensor_proto
    layout.ParseFromString(config.string_val[0])
    # A four-step CPU run has no GPU memory or NVML readings, and too little
    # history for a spike ratio.
    unreachable = ("^gpu/", "^memory/", "^loss/spike_ratio")
    empty = [
        f"{category.title} / {chart.title}"
        for category in layout.category
        for chart in category.chart
        if not all(pattern.startswith(unreachable) for pattern in chart.multiline.tag)
        and not any(re.search(p, tag) for p in chart.multiline.tag for tag in scalars)
    ]
    assert not empty, f"charts that match no tag: {empty}"
