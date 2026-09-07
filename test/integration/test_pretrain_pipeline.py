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

import pytest

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
