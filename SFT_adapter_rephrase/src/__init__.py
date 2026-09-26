"""
SFT_adapter_rephrase: supervised fine-tuning of a rephrase adapter on top of
the frozen pretrained encoder-decoder.

Scope
-----
Trains one small adapter (LoRA on the base's linears + learned embeddings
for the product frame's tokens) that turns

    <style:X> <text_to_rewrite> input </text_to_rewrite> </s>  ->  rewrite </s>    (X = data.style)

without changing a single base weight, so one base checkpoint serves every
future adapter. Runs either beside pretraining on spare CPU cores
(`--device cpu`, pinned, niced, CUDA hidden) or on the whole machine after it
(`--device gpu`). Everything tunable lives in `adapter_config.yml`.

Modules
-------
    config        adapter_config.yml -> validated SFTConfig (+ device profiles)
    lora          LoRALinear, TokenEmbeddingDelta, attach_adapter, sizing formula
    adapter_io    save/load an adapter directory, base-fingerprint check
    base_model    load a pretraining checkpoint, fingerprint it
    codec         tokenizer + the product frame
    data          JSONL -> filtered, hash-split, framed examples
    batching      length-grouped, device-independent step batches; padding
    decoding      greedy with n-gram blocking; beam
    text_metrics  SARI, chrF, span preservation, loops, novelty
    evaluation    teacher-forced + generation metrics
    resources     CPU/GPU plans and the guards that protect pretraining
    run_logging   sft.log, metrics.jsonl, TensorBoard
    trainer       the loop, selection, early stopping, checkpoints, resume
    pipeline      config -> ready trainer; evaluate a saved adapter
    inference     RephraseAdapter.load(...).rephrase(texts)
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG_PATH, SFTConfig, build_sft_config, load_sft_config
from .errors import (
    AdapterError,
    BaseModelMismatchError,
    DatasetError,
    ResourceError,
    ResumeError,
    SFTConfigError,
    SFTError,
)
from .lora import AttachedAdapter, LoRALinear, TokenEmbeddingDelta, attach_adapter, expected_adapter_parameters

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SFTConfig",
    "build_sft_config",
    "load_sft_config",
    "AttachedAdapter",
    "LoRALinear",
    "TokenEmbeddingDelta",
    "attach_adapter",
    "expected_adapter_parameters",
    "SFTError",
    "SFTConfigError",
    "AdapterError",
    "BaseModelMismatchError",
    "DatasetError",
    "ResourceError",
    "ResumeError",
]
