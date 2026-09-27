"""
Assembling a run from a config: the steps `sft.py` performs, in order, kept
here so the command line stays thin and tests can drive the same path with a
tiny model in place of the 752M checkpoint.

    resources -> run directories -> logging -> tokenizer -> data split
      -> base model -> adapter -> device -> trainer (-> resume)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch

from . import evaluation
from .adapter_io import load_adapter, read_adapter_config
from .base_model import BaseModelInfo, checkpoint_step, load_base_model
from .codec import TextCodec
from .config import SFTConfig
from .data import build_splits
from .errors import AdapterError, SFTConfigError, SFTError
from .lora import attach_adapter, expected_adapter_parameters
from .resources import apply_plan, plan_devices
from .run_logging import MetricsWriter, setup_logger
from .trainer import RunPaths, SFTTrainer, latest_checkpoint

BaseLoader = Callable[[Path], tuple[torch.nn.Module, BaseModelInfo]]


def run_name_for(config: SFTConfig) -> str:
    step = checkpoint_step(Path(config.base_model.checkpoint))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    step_label = step if step is not None else "x"
    return f"{stamp}_{config.device}_{config.data.style}_ckpt{step_label}_r{config.adapter.rank}"


def prepare_training(
    config: SFTConfig,
    *,
    run_name: str | None = None,
    resume: Path | None = None,
    allow_shared_gpu: bool = False,
    load_base: BaseLoader = load_base_model,
    codec: TextCodec | None = None,
    apply_resources: bool = True,
    to_stdout: bool = True,
) -> SFTTrainer:
    plan = plan_devices(config.device, config.resources)
    notes = (
        apply_plan(plan, require_exclusive_gpu=config.resources.require_exclusive_gpu and not allow_shared_gpu)
        if apply_resources
        else ["resource plan not applied (test mode)"]
    )

    # Before anything is written: a style the tokenizer lacks (a --style typo)
    # must not leave an empty run directory behind.
    codec = codec or TextCodec.from_tokenizer_dir(config.base_model.tokenizer_dir, config.data.style_token)

    if resume is not None:
        run_dir = Path(resume).resolve()
        if not run_dir.is_dir():
            raise SFTError(f"--resume: no such run directory {run_dir}")
        name = run_dir.name
    else:
        name = run_name or run_name_for(config)
        run_dir = config.logging.runs_dir / name
        if run_dir.exists():
            raise SFTError(f"run directory {run_dir} already exists; choose another --run-name or --resume it")
    paths = RunPaths(name=name, run_dir=run_dir, log_dir=config.logging.logs_dir / name)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(paths.log_dir / "sft.log", to_stdout=to_stdout)
    logger.info("run %s (%s)", name, "resuming" if resume else "new")
    logger.info("device plan: %s | %s", plan.describe(), "; ".join(notes))
    logger.info(
        "style %s (%s) | train data %s | eval data %s",
        config.data.style, config.data.style_token, config.data.train_data,
        config.data.eval_data or "carved out of the train data by split_salt",
    )
    if not config.style_profile:
        logger.warning(
            "no styles.%s block in the config: running on the shared length-ratio bounds "
            "(data %.2f-%.2f, acceptance <= %.2f); add one if this style has its own",
            config.data.style, config.data.min_length_ratio, config.data.max_length_ratio,
            config.evaluation.acceptance.max_length_ratio,
        )
    writer = MetricsWriter(paths.log_dir)

    started = time.perf_counter()
    train, evaluation_split, report = build_splits(config.data, codec)
    logger.info("data (%.1fs): %s", time.perf_counter() - started, json.dumps(report.to_dict()))
    if report.train < report.requested_train:
        logger.warning("only %d train examples available (%d requested)", report.train, report.requested_train)
    if report.eval < report.requested_eval:
        logger.warning("only %d eval examples available (%d requested)", report.eval, report.requested_eval)

    started = time.perf_counter()
    model, base_info = load_base(Path(config.base_model.checkpoint))
    logger.info(
        "base %s (step %s, %s params, fingerprint %s) loaded in %.1fs",
        base_info.checkpoint, base_info.step, f"{base_info.parameters:,}", base_info.fingerprint[:16],
        time.perf_counter() - started,
    )

    token_ids = _trainable_token_ids(config, codec)
    adapter = attach_adapter(model, config.adapter, token_ids)
    expected = expected_adapter_parameters(model.config, config.adapter)
    if adapter.parameter_count() != expected:
        raise AdapterError(
            f"attached adapter has {adapter.parameter_count():,} parameters, the formula predicts {expected:,}"
        )
    logger.info(
        "adapter: rank %d, scaling %.4g%s, %d LoRA pairs, %d trainable tokens -> %s trainable params "
        "(%.2f%% of the base)",
        config.adapter.rank, config.adapter.scaling, " (rsLoRA)" if config.adapter.use_rslora else "",
        len(adapter.lora_modules), len(token_ids), f"{adapter.parameter_count():,}",
        100 * adapter.parameter_count() / base_info.parameters,
    )
    if config.training.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.to(plan.device)

    if resume is None:
        (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    writer.text(0, "config", "```\n" + json.dumps(config.to_dict(), indent=2) + "\n```")

    trainer = SFTTrainer(
        config=config,
        model=model,
        adapter=adapter,
        base_info=base_info,
        codec=codec,
        train_examples=train,
        eval_examples=evaluation_split,
        data_report=report,
        plan=plan,
        paths=paths,
        logger=logger,
        writer=writer,
    )
    if resume is not None:
        trainer.resume(latest_checkpoint(run_dir))
    return trainer


def evaluate_adapter(
    config: SFTConfig,
    adapter_dir: Path,
    *,
    base_checkpoint: Path | None = None,
    loss_examples: int | None = None,
    generation_examples: int | None = None,
    load_base: BaseLoader = load_base_model,
    codec: TextCodec | None = None,
    apply_resources: bool = True,
) -> dict[str, float]:
    """Score a saved adapter on the config's eval split (same split as training)."""
    plan = plan_devices(config.device, config.resources)
    if apply_resources:
        apply_plan(plan, require_exclusive_gpu=False)
    document = read_adapter_config(adapter_dir)
    if document.get("style_token") != config.data.style_token:
        raise AdapterError(
            f"{adapter_dir} was trained for {document.get('style_token')}, but the config selects "
            f"{config.data.style_token}; evaluating it with another style's frame measures nothing"
        )
    base_path = base_checkpoint or Path(document["base"]["checkpoint"])
    codec = codec or TextCodec.from_tokenizer_dir(config.base_model.tokenizer_dir, config.data.style_token)
    _, evaluation_split, _ = build_splits(config.data, codec, eval_only=True)
    model, _ = load_base(base_path)
    load_adapter(model, adapter_dir, token_map=codec.token_map)
    model.to(plan.device)
    autocast = plan.autocast()
    e = config.evaluation
    count = loss_examples or e.final_loss_examples or len(evaluation_split)
    metrics = evaluation.teacher_forced_metrics(
        model, evaluation_split[:count], codec,
        batch_size=config.training.batch_size * 2, device=plan.device, autocast=autocast,
        calibration_bins=e.calibration_bins,
    )
    generation_count = e.generation_examples if generation_examples is None else generation_examples
    if generation_count:
        records = evaluation.generate(
            model, evaluation_split[:generation_count], codec, e.decoding,
            batch_size=e.generation_batch_size, device=plan.device, autocast=autocast,
        )
        metrics.update(evaluation.generation_metrics(records, e.acceptance))
        metrics.update(
            evaluation.generation_metrics(records, e.acceptance, prefix="reference", use_reference_as_output=True)
        )
    return metrics


def _trainable_token_ids(config: SFTConfig, codec: TextCodec) -> list[int]:
    unknown = [t for t in config.adapter.trainable_tokens if t not in codec.token_map]
    if unknown:
        raise SFTConfigError(f"adapter.trainable_tokens not in the tokenizer: {unknown}")
    return [codec.token_map[t] for t in config.adapter.trainable_tokens]
