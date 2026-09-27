"""
End-to-end runs on the tiny base: the whole pipeline `sft.py train` drives,
minus the 752M checkpoint and the process-level resource plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from conftest import build_tiny_model, fake_decode, fake_encode, FAKE_TOKEN_MAP, make_config, sample_records, write_dataset
from src.adapter_io import TENSORS_FILE, load_adapter
from src.base_model import BaseModelInfo, base_fingerprint
from src.codec import TextCodec
from src.errors import ResumeError
from src.pipeline import prepare_training
from src.trainer import lr_multiplier


def _tiny_loader(path: Path):
    model = build_tiny_model(seed=0)
    info = BaseModelInfo(
        checkpoint=str(path), step=5, fingerprint=base_fingerprint(model),
        parameters=model.parameter_count(), config={},
    )
    return model, info


def _codec() -> TextCodec:
    return TextCodec(fake_encode, fake_decode, FAKE_TOKEN_MAP, "<style:concise>")


def _trainer(tmp_path: Path, *, resume: Path | None = None, run_name: str = "run", **sections):
    if not (tmp_path / "data").exists():
        write_dataset(tmp_path / "data", sample_records(48), files=2)
    config = make_config(tmp_path, **sections)
    return prepare_training(
        config,
        run_name=None if resume else run_name,
        resume=resume,
        load_base=_tiny_loader,
        codec=_codec(),
        apply_resources=False,
        to_stdout=False,
    )


def _train(trainer):
    try:
        return trainer.train()
    finally:
        trainer.writer.close()


def _metrics(log_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]


def test_schedule_warms_up_then_decays_to_the_floor():
    kwargs = dict(schedule="cosine", warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
    assert lr_multiplier(0, **kwargs) == pytest.approx(0.1)
    assert lr_multiplier(9, **kwargs) == pytest.approx(1.0)
    assert lr_multiplier(60, **kwargs) == pytest.approx(0.55)
    assert lr_multiplier(110, **kwargs) == pytest.approx(0.1)
    assert lr_multiplier(500, **kwargs) == pytest.approx(0.1)
    assert lr_multiplier(60, **{**kwargs, "schedule": "linear"}) == pytest.approx(0.55)


def test_a_full_run_learns_and_leaves_a_loadable_adapter(tmp_path):
    trainer = _trainer(tmp_path, training={"epochs": 8})
    summary = _train(trainer)
    run_dir, log_dir = trainer.paths.run_dir, trainer.paths.log_dir

    assert summary["status"] == "completed"
    assert summary["steps"] == trainer.total_steps == 8 * 3  # 24 train examples / 8 per step
    rows = _metrics(log_dir)
    losses = [r["eval/loss"] for r in rows if "eval/loss" in r]
    assert losses[-1] < 0.6 * losses[0], losses  # baseline (untouched base) first
    assert summary["final_metrics"]["eval/loss"] <= min(losses) + 1e-9

    keys = set().union(*rows)
    for expected in (
        "train/loss", "train/grad_norm", "perf/tokens_per_second", "perf/padding_efficiency",
        "adapter/relative_delta/projection/q_proj", "adapter/grad_norm/decoder_cross_attention",
        "adapter/token_delta_norm/<style:concise>", "eval/eos_accuracy", "eval/ece",
        "generation/sari", "generation/span_preservation", "quality/acceptable_rate",
        "reference/sari", "final/loss",
    ):
        assert expected in keys, expected
    assert list((log_dir / "tensorboard").glob("events.out.tfevents.*"))
    assert "step 24/24" in (log_dir / "sft.log").read_text()

    assert len(list((run_dir / "checkpoints").glob("step-*"))) == 2  # keep_last
    assert (run_dir / "summary.json").is_file() and (run_dir / "config.json").is_file()

    # The final adapter is the best one, and it re-attaches to a fresh base.
    best = load_file(str(run_dir / "best" / TENSORS_FILE))
    final = load_file(str(run_dir / "final" / TENSORS_FILE))
    assert best.keys() == final.keys() and all(torch.equal(best[k], final[k]) for k in best)
    fresh = build_tiny_model(seed=0)
    load_adapter(fresh, run_dir / "final", token_map=FAKE_TOKEN_MAP)


def test_gradient_accumulation_matches_one_big_batch(tmp_path):
    """4 x 2 and 8 x 1 must take the same optimizer steps (token-weighted loss)."""
    split = _trainer(tmp_path / "a", training={"max_steps": 3, "batch_size": 4, "gradient_accumulation_steps": 2},
                     evaluation={"generation_examples": 0, "every_steps": 100})
    whole = _trainer(tmp_path / "b", training={"max_steps": 3, "batch_size": 8, "gradient_accumulation_steps": 1},
                     evaluation={"generation_examples": 0, "every_steps": 100})
    for trainer in (split, whole):
        for _ in range(3):
            trainer._train_one_step()
        trainer.writer.close()
    a, b = split.adapter.state_dict(), whole.adapter.state_dict()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-5), key


def test_resumed_run_continues_bit_identically(tmp_path):
    """Interrupting and resuming must not change a single adapter weight."""
    common = dict(evaluation={"every_steps": 2, "generation_examples": 2}, checkpointing={"every_steps": 2, "keep_last": 3})
    straight = _trainer(tmp_path / "straight", training={"max_steps": 6}, **common)
    _train(straight)

    first = _trainer(tmp_path / "split", training={"max_steps": 3}, **common)
    _train(first)
    resumed = _trainer(tmp_path / "split", resume=first.paths.run_dir, training={"max_steps": 6}, **common)
    assert resumed.step == 3
    _train(resumed)

    a = load_file(str(straight.paths.run_dir / "checkpoints" / "step-000006" / TENSORS_FILE))
    b = load_file(str(resumed.paths.run_dir / "checkpoints" / "step-000006" / TENSORS_FILE))
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_resume_refuses_a_changed_adapter(tmp_path):
    first = _trainer(tmp_path, training={"max_steps": 2})
    _train(first)
    with pytest.raises(ResumeError, match="adapter.rank"):
        _trainer(tmp_path, resume=first.paths.run_dir, adapter={"rank": 8})


def test_early_stopping_after_patience_runs_out(tmp_path):
    # "Higher loss is better" guarantees every evaluation after the first is worse.
    trainer = _trainer(
        tmp_path,
        training={"epochs": 5},
        evaluation={"greater_is_better": True, "early_stopping_patience": 2, "every_steps": 2},
    )
    summary = _train(trainer)
    assert summary["status"] == "early_stopped"
    assert summary["steps"] == 6 and summary["best_step"] == 2
    assert summary["final_metrics"] is not None


def test_an_interrupt_leaves_a_resumable_checkpoint(tmp_path, monkeypatch):
    """Ctrl-C / SIGTERM mid-run: the step reached is saved, nothing is lost."""
    trainer = _trainer(tmp_path, training={"max_steps": 10}, checkpointing={"every_steps": 100, "keep_last": 2})
    original = trainer._train_one_step

    def interrupt_at_four():
        if trainer.step == 4:
            raise KeyboardInterrupt
        original()

    monkeypatch.setattr(trainer, "_train_one_step", interrupt_at_four)
    summary = _train(trainer)
    assert summary["status"] == "interrupted"
    assert (trainer.paths.run_dir / "checkpoints" / "step-000004").is_dir()
    resumed = _trainer(tmp_path, resume=trainer.paths.run_dir, training={"max_steps": 10},
                       checkpointing={"every_steps": 100, "keep_last": 2})
    assert resumed.step == 4 and _train(resumed)["status"] == "completed"
