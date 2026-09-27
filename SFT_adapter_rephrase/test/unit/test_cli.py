"""
The command line's defaults that decide which style and data a command runs
on: getting one wrong scores an adapter with another style's frame, or
resumes a run on data it never saw.
"""

from __future__ import annotations

import argparse
import json

import sft
from src.adapter_io import CONFIG_FILE, FORMAT_VERSION


def _adapter_dir(tmp_path, *, data=None):
    directory = tmp_path / "adapter"
    directory.mkdir()
    provenance = {"run": "r"} if data is None else {"run": "r", "data": data}
    document = {"format_version": FORMAT_VERSION, "style_token": "<style:professional>", "provenance": provenance}
    (directory / CONFIG_FILE).write_text(json.dumps(document), encoding="utf-8")
    return directory


def _evaluate_args(adapter, **flags):
    return argparse.Namespace(adapter=adapter, eval_data=flags.get("eval_data"), train_data=flags.get("train_data"))


def test_evaluate_uses_the_adapters_style_and_recorded_data(tmp_path):
    adapter = _adapter_dir(tmp_path, data={"style": "professional", "train_data": "/d/train.jsonl",
                                           "eval_data": "/d/evals.json", "fingerprint": "f"})
    overrides = {"data": {"style": "concise"}}
    sft._adapter_data_overrides(_evaluate_args(adapter), overrides)
    assert overrides["data"] == {"style": "professional", "train_data": "/d/train.jsonl", "eval_data": "/d/evals.json"}


def test_evaluate_data_flags_win_over_the_recorded_data(tmp_path):
    adapter = _adapter_dir(tmp_path, data={"train_data": "/d/train.jsonl", "eval_data": "/d/evals.json"})
    overrides: dict = {}
    args = _evaluate_args(adapter, eval_data=tmp_path / "other.jsonl")
    sft._adapter_data_overrides(args, overrides)
    sft._data_overrides(args, overrides)
    assert overrides["data"] == {"style": "professional", "eval_data": str(tmp_path / "other.jsonl")}


def test_evaluate_of_an_adapter_without_recorded_data_falls_back_to_the_config(tmp_path):
    overrides: dict = {}
    sft._adapter_data_overrides(_evaluate_args(_adapter_dir(tmp_path)), overrides)
    assert overrides["data"] == {"style": "professional"}


def test_resume_keeps_the_runs_style_and_data_unless_named_again(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"data": {
        "style": "professional", "train_data": "/d/train.jsonl", "eval_data": None, "train_examples": 9,
    }}), encoding="utf-8")
    overrides = {"data": {"style": "friendly"}}
    sft._resumed_run_overrides(argparse.Namespace(resume=run), overrides)
    assert overrides["data"] == {"style": "friendly", "train_data": "/d/train.jsonl", "eval_data": None}
