#!/usr/bin/env python
"""
Train, evaluate or use a rephrase adapter for any `<style:...>` token.

    # beside pretraining: cores 0-21, nice 19, CUDA hidden (style/data from the config)
    python SFT_adapter_rephrase/sft.py train --device cpu

    # pick the style and the data on the command line
    python SFT_adapter_rephrase/sft.py train --device cpu --style professional \
        --train-data path/to/train.jsonl --eval-data path/to/evals.jsonl

    # after pretraining: the whole machine
    python SFT_adapter_rephrase/sft.py train --device gpu

    # resume an interrupted run (also CPU -> GPU: same effective batch)
    python SFT_adapter_rephrase/sft.py train --device gpu --resume SFT_adapter_rephrase/runs/<run>

    # score a saved adapter on the eval data it was trained with (style and
    # data come from the adapter; --eval-data scores it on something else)
    python SFT_adapter_rephrase/sft.py evaluate --device cpu --adapter SFT_adapter_rephrase/runs/<run>/final

    # try it
    python SFT_adapter_rephrase/sft.py rephrase --device cpu --adapter .../final --text "..."

TensorBoard: `SFT_adapter_rephrase/tensorboard.sh` (port 6007; pretraining keeps 6006).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--device", choices=("cpu", "gpu"), required=True,
                         help="cpu: spare cores beside pretraining; gpu: the whole machine")
        sub.add_argument("--config", type=Path, default=None, help="default: SFT_adapter_rephrase/adapter_config.yml")

    train = commands.add_parser("train", help="train an adapter")
    common(train)
    train.add_argument("--base-checkpoint", type=Path, help="overrides base_model.checkpoint")
    train.add_argument("--style", help="the X of a <style:X> token, e.g. professional; overrides data.style")
    train.add_argument("--train-data", type=Path, help="JSONL file or directory; overrides data.train_data")
    train.add_argument("--eval-data", type=Path,
                       help="JSONL file or directory held out for evaluation; overrides data.eval_data "
                            "(unset in the config: the eval split is carved out of the train data)")
    train.add_argument("--run-name", help="default: <time>_<device>_<style>_ckpt<step>_r<rank>")
    train.add_argument("--resume", type=Path, help="run directory to resume")
    train.add_argument("--max-steps", type=int, help="overrides training.max_steps")
    train.add_argument("--train-examples", type=int, help="overrides data.train_examples")
    train.add_argument("--eval-examples", type=int, help="overrides data.eval_examples")
    train.add_argument("--allow-shared-gpu", action="store_true",
                       help="start on the GPU even though another process is using it")

    evaluate = commands.add_parser("evaluate", help="score a saved adapter on the eval split")
    common(evaluate)
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument("--base-checkpoint", type=Path, help="default: the base recorded in the adapter")
    evaluate.add_argument("--eval-data", type=Path, help="default: the eval data recorded in the adapter")
    evaluate.add_argument("--train-data", type=Path,
                          help="with no eval data anywhere, the eval split is carved out of this "
                               "(default: the train data recorded in the adapter)")
    evaluate.add_argument("--loss-examples", type=int)
    evaluate.add_argument("--generation-examples", type=int)
    evaluate.add_argument("--output", type=Path, help="write the metrics JSON here")

    rephrase = commands.add_parser("rephrase", help="rewrite text with a saved adapter")
    common(rephrase)
    rephrase.add_argument("--adapter", type=Path, required=True)
    rephrase.add_argument("--base-checkpoint", type=Path)
    rephrase.add_argument("--text", action="append", required=True, help="repeatable")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cpu":
        # Before torch is imported anywhere: the CPU plan must never create a
        # CUDA context on the GPU pretraining is using.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.config import DEFAULT_CONFIG_PATH, load_sft_config
    from src.errors import SFTError

    config_path = args.config or DEFAULT_CONFIG_PATH
    overrides: dict = {}
    if getattr(args, "base_checkpoint", None) and args.command == "train":
        overrides.setdefault("base_model", {})["checkpoint"] = str(args.base_checkpoint)
    if getattr(args, "max_steps", None) is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps
    if getattr(args, "train_examples", None) is not None:
        overrides.setdefault("data", {})["train_examples"] = args.train_examples
    if getattr(args, "eval_examples", None) is not None:
        overrides.setdefault("data", {})["eval_examples"] = args.eval_examples
    if getattr(args, "style", None) is not None:
        overrides.setdefault("data", {})["style"] = args.style

    try:
        if args.command == "evaluate":
            _adapter_data_overrides(args, overrides)
        elif args.command == "train" and args.resume is not None:
            _resumed_run_overrides(args, overrides)
        _data_overrides(args, overrides)
        config = load_sft_config(config_path, args.device, overrides)
        if args.command == "train":
            return _train(args, config)
        if args.command == "evaluate":
            return _evaluate(args, config)
        return _rephrase(args, config)
    except SFTError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _data_overrides(args, overrides: dict) -> None:
    if getattr(args, "train_data", None) is not None:
        overrides.setdefault("data", {})["train_data"] = str(args.train_data)
    if getattr(args, "eval_data", None) is not None:
        overrides.setdefault("data", {})["eval_data"] = str(args.eval_data)


def _adapter_data_overrides(args, overrides: dict) -> None:
    """
    An adapter is scored in its own style, on the data it was trained with
    (recorded in the adapter), unless --eval-data/--train-data say otherwise.
    """
    from src.adapter_io import read_adapter_config
    from src.config import style_from_token

    document = read_adapter_config(args.adapter)
    data = overrides.setdefault("data", {})
    data["style"] = style_from_token(document.get("style_token") or "")
    recorded = (document.get("provenance") or {}).get("data") or {}
    if args.eval_data is None and args.train_data is None and recorded.get("train_data"):
        data["train_data"] = recorded["train_data"]
        data["eval_data"] = recorded.get("eval_data")


def _resumed_run_overrides(args, overrides: dict) -> None:
    """
    A resumed run keeps the style and data it started with (from its
    config.json) unless the command line names them again. The checkpoint's
    resume signature and data fingerprint are still checked afterwards.
    """
    from src.errors import SFTError

    path = Path(args.resume) / "config.json"
    if not path.is_file():
        return
    try:
        recorded = json.loads(path.read_text(encoding="utf-8")).get("data") or {}
    except json.JSONDecodeError as exc:
        raise SFTError(f"{path} is not valid JSON: {exc}") from exc
    data = overrides.setdefault("data", {})
    for key in ("style", "train_data", "eval_data"):
        if key in recorded and key not in data:
            data[key] = recorded[key]


def _train(args, config) -> int:
    from src.pipeline import prepare_training

    # SIGTERM (tmux kill, system shutdown) takes the same path as Ctrl-C: a
    # resumable checkpoint is written before the process exits.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    trainer = prepare_training(
        config, run_name=args.run_name, resume=args.resume, allow_shared_gpu=args.allow_shared_gpu
    )
    try:
        summary = trainer.train()
    finally:
        trainer.writer.close()
    print(json.dumps({k: summary[k] for k in ("run", "status", "steps", "best_step", "best_metric")}, indent=2))
    return 0 if summary["status"] in ("completed", "early_stopped") else 130


def _evaluate(args, config) -> int:
    from src.pipeline import evaluate_adapter

    metrics = evaluate_adapter(
        config,
        args.adapter,
        base_checkpoint=args.base_checkpoint,
        loss_examples=args.loss_examples,
        generation_examples=args.generation_examples,
    )
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _rephrase(args, config) -> int:
    from src.inference import RephraseAdapter
    from src.resources import apply_plan, plan_devices

    plan = plan_devices(config.device, config.resources)
    apply_plan(plan)
    rephraser = RephraseAdapter.load(
        args.adapter,
        base_checkpoint=args.base_checkpoint,
        tokenizer_dir=config.base_model.tokenizer_dir,
        device=str(plan.device),
    )
    for source, output in zip(args.text, rephraser.rephrase(args.text, decoding=config.evaluation.decoding)):
        print(f"IN : {source}\nOUT: {output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
