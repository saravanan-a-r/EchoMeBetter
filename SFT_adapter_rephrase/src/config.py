"""
`adapter_config.yml` -> one frozen, validated `SFTConfig`.

Loading is strict in the way `model/src/config.py` is: an unknown key is an
error, not a warning. A typo such as `rnak: 64` that silently fell back to the
default would train -- for a day on the CPU -- an adapter of a size nobody
asked for.

Device profiles
---------------
The file holds one shared configuration plus a `devices` block. Choosing a
device (`--device cpu|gpu`) picks that block's `resources` and lays its
`overrides` over the shared sections, key by key, before validation -- so a
profile can change a batch size without restating the whole section, and a
profile that overrides a key which does not exist fails like any other typo.

Styles
------
One module trains an adapter for any `<style:...>` token the tokenizer has.
`data.style` names it (`concise`, `professional`, ...); the frame's style
token is derived from it, and `"{style_token}"` in `adapter.trainable_tokens`
stands for that token. A style's own knobs -- length-ratio bounds, the
acceptance bar -- live in the top-level `styles.<name>` block, laid over the
shared sections the same way a device profile is. Order, last one wins:

    shared sections -> styles.<style> -> devices.<device>.overrides -> command line

The style is chosen before any overlay (command line first, then the file),
so `--style professional` picks up `styles.professional`.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import SFTConfigError
from .interop import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "SFT_adapter_rephrase" / "adapter_config.yml"

DEVICES = ("cpu", "gpu")

# The projections a LoRA pair can attach to, by the attribute name the model
# gives them (`model/src/attention.py`, `model/src/layers.py`).
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Target group -> (stack attribute, sublayer attribute, allowed projections).
TARGET_GROUPS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "encoder_self_attention": ("encoder", "self_attn", ATTENTION_PROJECTIONS),
    "encoder_ffn": ("encoder", "ffn", FFN_PROJECTIONS),
    "decoder_self_attention": ("decoder", "self_attn", ATTENTION_PROJECTIONS),
    "decoder_cross_attention": ("decoder", "cross_attn", ATTENTION_PROJECTIONS),
    "decoder_ffn": ("decoder", "ffn", FFN_PROJECTIONS),
}

LR_SCHEDULES = ("cosine", "linear", "constant")
DECODING_STRATEGIES = ("greedy", "beam")
PRECISIONS = ("fp32", "bf16")

# `data.style` is the X of a `<style:X>` token; `adapter.trainable_tokens`
# names the selected style's token with this placeholder.
STYLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
STYLE_TOKEN_PLACEHOLDER = "{style_token}"


def style_token_for(style: str) -> str:
    return f"<style:{style}>"


def style_from_token(token: str) -> str:
    """`<style:professional>` -> `professional`."""
    if not (token.startswith("<style:") and token.endswith(">")):
        raise SFTConfigError(f"{token!r} is not a <style:...> token")
    return token[len("<style:"):-1]


# -- sections ---------------------------------------------------------------


@dataclass(frozen=True)
class BaseModelConfig:
    checkpoint: Path
    tokenizer_dir: Path


@dataclass(frozen=True)
class AdapterConfig:
    rank: int
    alpha: float
    use_rslora: bool
    dropout: float
    targets: dict[str, tuple[str, ...]]
    lr_multipliers: dict[str, float] = field(default_factory=dict)
    trainable_tokens: tuple[str, ...] = ()

    @property
    def scaling(self) -> float:
        """What the LoRA product `B @ A` is multiplied by."""
        denominator = math.sqrt(self.rank) if self.use_rslora else self.rank
        return self.alpha / denominator


@dataclass(frozen=True)
class DataConfig:
    style: str
    # A JSONL file, or a directory whose `file_glob` matches the files.
    train_data: Path
    # null: the eval split is carved out of `train_data` by the salted hash.
    # Otherwise a separate file/directory; its inputs never reach training.
    eval_data: Path | None
    file_glob: str
    allowed_cleanup: tuple[str, ...]
    train_examples: int
    eval_examples: int
    split_salt: str
    max_source_tokens: int
    max_target_tokens: int
    min_length_ratio: float
    max_length_ratio: float
    require_span_preservation: bool

    @property
    def style_token(self) -> str:
        """The frame's style token, e.g. `<style:professional>`."""
        return style_token_for(self.style)

    @property
    def expected_style(self) -> str:
        """The `style` field every record must carry."""
        return self.style


@dataclass(frozen=True)
class TrainingConfig:
    epochs: float
    max_steps: int | None
    learning_rate: float
    lr_schedule: str
    warmup_steps: int
    min_lr_ratio: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float | None
    label_smoothing: float
    batch_size: int
    gradient_accumulation_steps: int
    length_grouping_window: int
    gradient_checkpointing: bool
    seed: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class DecodingConfig:
    strategy: str
    num_beams: int
    length_penalty: float
    no_repeat_ngram_size: int
    max_new_tokens: int


@dataclass(frozen=True)
class AcceptanceConfig:
    max_length_ratio: float
    min_input_chrf: float


@dataclass(frozen=True)
class EvaluationConfig:
    every_steps: int
    loss_examples: int
    final_loss_examples: int | None
    generation_examples: int
    generation_batch_size: int
    decoding: DecodingConfig
    text_samples: int
    select_metric: str
    greater_is_better: bool
    early_stopping_patience: int | None
    calibration_bins: int
    acceptance: AcceptanceConfig


@dataclass(frozen=True)
class CheckpointConfig:
    every_steps: int
    keep_last: int


@dataclass(frozen=True)
class LoggingConfig:
    every_steps: int
    logs_dir: Path
    runs_dir: Path


@dataclass(frozen=True)
class ResourceConfig:
    cores: str | None
    threads: int | None
    nice: int
    idle_io: bool
    precision: str
    require_exclusive_gpu: bool = False


@dataclass(frozen=True)
class SFTConfig:
    device: str
    base_model: BaseModelConfig
    adapter: AdapterConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    checkpointing: CheckpointConfig
    logging: LoggingConfig
    resources: ResourceConfig
    # Whether `styles.<data.style>` existed and was applied. Without one the
    # style runs on the shared (style-neutral) length bounds.
    style_profile: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def resume_signature(self) -> dict[str, Any]:
        """
        What must be identical for a checkpoint to be resumed under this config.

        Everything that shapes the adapter, the data order or the optimizer
        trajectory. Left out on purpose:
          - evaluation, logging and checkpoint cadence -- how often a run is
            observed does not change the run;
          - `max_steps`, so a stopped run can be extended;
          - the micro-batch split (`batch_size`, `gradient_accumulation_steps`,
            `gradient_checkpointing`): the data order is planned per
            optimizer step, so only the effective batch size matters -- which
            is what lets a CPU run be resumed on the GPU;
          - file locations (base checkpoint, train/eval data): the trainer
            checks the base weights' and the split's fingerprints instead, so
            a run survives its base checkpoint or data being copied elsewhere.
        """
        training = asdict(self.training)
        for key in ("max_steps", "batch_size", "gradient_accumulation_steps", "gradient_checkpointing"):
            training.pop(key)
        training["effective_batch_size"] = self.training.effective_batch_size
        data = asdict(self.data)
        data.pop("train_data")
        data.pop("eval_data")
        return _jsonable({"adapter": asdict(self.adapter), "data": data, "training": training})


# -- loading ----------------------------------------------------------------


def load_sft_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> SFTConfig:
    """
    Read the YAML, apply the device profile and then `overrides` (the CLI's),
    and validate the result.
    """
    path = Path(path)
    if not path.is_file():
        raise SFTConfigError(f"config file not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SFTConfigError(f"{path} is not valid YAML: {exc}") from exc
    return build_sft_config(document, device, overrides)


def build_sft_config(
    document: Mapping[str, Any],
    device: str,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> SFTConfig:
    if not isinstance(document, Mapping):
        raise SFTConfigError("the config must be a YAML mapping")
    if device not in DEVICES:
        raise SFTConfigError(f"device must be one of {DEVICES}, got {device!r}")

    document = copy.deepcopy(dict(document))
    devices = document.pop("devices", None) or {}
    if device not in devices:
        raise SFTConfigError(f"the config has no `devices.{device}` profile")
    profile = devices[device] or {}
    _reject_unknown(profile, {"resources", "overrides"}, f"devices.{device}")

    styles = document.pop("styles", None) or {}
    if not isinstance(styles, Mapping):
        raise SFTConfigError("`styles` must be a mapping of style name -> overrides")
    for name in styles:
        if not isinstance(name, str) or not STYLE_NAME.match(name):
            raise SFTConfigError(f"styles: {name!r} is not a style name (lowercase, e.g. professional)")
    style = _selected_style(document, overrides)

    merged = document
    if styles.get(style):
        merged = _overlay(merged, styles[style], f"styles.{style}")
    merged = _overlay(merged, profile.get("overrides") or {}, f"devices.{device}.overrides")
    if overrides:
        merged = _overlay(merged, overrides, "command-line overrides")

    known_sections = {
        "base_model", "adapter", "data", "training", "evaluation", "checkpointing", "logging",
    }
    _reject_unknown(merged, known_sections, "top level")
    missing = sorted(known_sections - set(merged))
    if missing:
        raise SFTConfigError(f"config is missing sections: {missing}")

    if isinstance(merged["data"], Mapping) and merged["data"].get("style") != style:
        raise SFTConfigError(f"styles.{style} must not change data.style")

    config = SFTConfig(
        device=device,
        base_model=_section(BaseModelConfig, merged["base_model"], "base_model"),
        adapter=_adapter(merged["adapter"], style),
        data=_section(DataConfig, merged["data"], "data"),
        training=_section(TrainingConfig, merged["training"], "training"),
        evaluation=_evaluation(merged["evaluation"]),
        checkpointing=_section(CheckpointConfig, merged["checkpointing"], "checkpointing"),
        logging=_section(LoggingConfig, merged["logging"], "logging"),
        resources=_section(ResourceConfig, profile.get("resources") or {}, f"devices.{device}.resources"),
        style_profile=bool(styles.get(style)),
    )
    _validate(config)
    return config


def _selected_style(document: Mapping[str, Any], overrides: Mapping[str, Mapping[str, Any]] | None) -> str:
    """The command line's `data.style` if it sets one, else the file's."""
    cli_data = (overrides or {}).get("data") or {}
    file_data = document.get("data")
    if "style" in cli_data:
        style = cli_data["style"]
    elif isinstance(file_data, Mapping) and "style" in file_data:
        style = file_data["style"]
    else:
        raise SFTConfigError("`data` is missing ['style']")
    if not isinstance(style, str) or not STYLE_NAME.match(style):
        raise SFTConfigError(
            f"data.style must be a style name such as 'professional' (the X of <style:X>), got {style!r}"
        )
    return style


def resolve_path(value: str | Path) -> Path:
    """Relative paths in the config are relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# -- helpers ----------------------------------------------------------------


_PATH_FIELDS = {"checkpoint", "tokenizer_dir", "train_data", "eval_data", "logs_dir", "runs_dir"}
_TUPLE_FIELDS = {"allowed_cleanup", "trainable_tokens"}


def _section(cls, raw: Any, name: str):
    if not isinstance(raw, Mapping):
        raise SFTConfigError(f"`{name}` must be a mapping")
    names = {f.name for f in fields(cls)}
    _reject_unknown(raw, names, name)
    required = {f.name for f in fields(cls) if _is_required(f)}
    missing = sorted(required - set(raw))
    if missing:
        raise SFTConfigError(f"`{name}` is missing {missing}")
    values = {}
    for key, value in raw.items():
        if key in _PATH_FIELDS and value is not None:
            value = resolve_path(value)
        elif key in _TUPLE_FIELDS:
            value = tuple(value or ())
        values[key] = value
    try:
        return cls(**values)
    except TypeError as exc:
        raise SFTConfigError(f"`{name}`: {exc}") from exc


def _is_required(f) -> bool:
    return f.default is MISSING and f.default_factory is MISSING


def _adapter(raw: Any, style: str) -> AdapterConfig:
    if not isinstance(raw, Mapping):
        raise SFTConfigError("`adapter` must be a mapping")
    raw = dict(raw)
    # "{style_token}" -> the selected style's token. Naming any other style's
    # token would spend a day training an embedding the frame never uses.
    style_token = style_token_for(style)
    tokens = [style_token if t == STYLE_TOKEN_PLACEHOLDER else t for t in raw.get("trainable_tokens") or ()]
    foreign = [t for t in tokens if isinstance(t, str) and t.startswith("<style:") and t != style_token]
    if foreign:
        raise SFTConfigError(
            f"adapter.trainable_tokens names {foreign}, but the selected style is {style_token}; "
            f"write {STYLE_TOKEN_PLACEHOLDER!r} for the selected style's token"
        )
    raw["trainable_tokens"] = tokens
    targets = raw.get("targets")
    if not isinstance(targets, Mapping):
        raise SFTConfigError("`adapter.targets` must be a mapping of group -> projections")
    _reject_unknown(targets, set(TARGET_GROUPS), "adapter.targets")
    normalized: dict[str, tuple[str, ...]] = {}
    for group, projections in targets.items():
        allowed = TARGET_GROUPS[group][2]
        projections = tuple(projections or ())
        unknown = sorted(set(projections) - set(allowed))
        if unknown:
            raise SFTConfigError(
                f"adapter.targets.{group}: unknown projections {unknown}; allowed {list(allowed)}"
            )
        if len(set(projections)) != len(projections):
            raise SFTConfigError(f"adapter.targets.{group} lists a projection twice")
        normalized[group] = projections
    raw["targets"] = normalized
    raw["lr_multipliers"] = dict(raw.get("lr_multipliers") or {})
    return _section(AdapterConfig, raw, "adapter")


def _evaluation(raw: Any) -> EvaluationConfig:
    if not isinstance(raw, Mapping):
        raise SFTConfigError("`evaluation` must be a mapping")
    raw = dict(raw)
    raw["decoding"] = _section(DecodingConfig, raw.get("decoding"), "evaluation.decoding")
    raw["acceptance"] = _section(AcceptanceConfig, raw.get("acceptance"), "evaluation.acceptance")
    return _section(EvaluationConfig, raw, "evaluation")


def _validate(config: SFTConfig) -> None:
    a, d, t, e = config.adapter, config.data, config.training, config.evaluation
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    check(_is_int(a.rank) and a.rank >= 1, "adapter.rank must be an integer >= 1")
    check(a.alpha > 0, "adapter.alpha must be positive")
    check(0.0 <= a.dropout < 1.0, "adapter.dropout must be in [0, 1)")
    check(
        any(a.targets.values()) or bool(a.trainable_tokens),
        "the adapter trains nothing: every target group is empty and there are no trainable_tokens",
    )
    every_projection = set(ATTENTION_PROJECTIONS) | set(FFN_PROJECTIONS)
    for name, multiplier in a.lr_multipliers.items():
        check(name in every_projection, f"adapter.lr_multipliers: unknown projection {name!r}")
        check(isinstance(multiplier, (int, float)) and multiplier > 0,
              f"adapter.lr_multipliers.{name} must be positive")
    check(len(set(a.trainable_tokens)) == len(a.trainable_tokens),
          "adapter.trainable_tokens lists a token twice")

    check(bool(d.allowed_cleanup), "data.allowed_cleanup must name at least one class")
    check(d.eval_data is None or Path(d.eval_data).resolve() != Path(d.train_data).resolve(),
          "data.eval_data is the same as data.train_data; set eval_data to null to carve the eval split out of it")
    check(_is_int(d.train_examples) and d.train_examples >= 1, "data.train_examples must be >= 1")
    check(_is_int(d.eval_examples) and d.eval_examples >= 1, "data.eval_examples must be >= 1")
    check(d.max_source_tokens >= 8 and d.max_target_tokens >= 2,
          "data.max_source_tokens must be >= 8 and max_target_tokens >= 2")
    check(0.0 <= d.min_length_ratio < d.max_length_ratio,
          "data.min_length_ratio must be >= 0 and below max_length_ratio")

    check(t.epochs > 0, "training.epochs must be positive")
    check(t.max_steps is None or (_is_int(t.max_steps) and t.max_steps >= 1),
          "training.max_steps must be null or >= 1")
    check(t.learning_rate > 0, "training.learning_rate must be positive")
    check(t.lr_schedule in LR_SCHEDULES, f"training.lr_schedule must be one of {LR_SCHEDULES}")
    check(_is_int(t.warmup_steps) and t.warmup_steps >= 0, "training.warmup_steps must be >= 0")
    check(0.0 <= t.min_lr_ratio <= 1.0, "training.min_lr_ratio must be in [0, 1]")
    check(t.weight_decay >= 0, "training.weight_decay must be >= 0")
    check(0 <= t.adam_beta1 < 1 and 0 <= t.adam_beta2 < 1, "Adam betas must be in [0, 1)")
    check(t.max_grad_norm is None or t.max_grad_norm > 0, "training.max_grad_norm must be null or > 0")
    check(0.0 <= t.label_smoothing < 1.0, "training.label_smoothing must be in [0, 1)")
    check(_is_int(t.batch_size) and t.batch_size >= 1, "training.batch_size must be >= 1")
    check(_is_int(t.gradient_accumulation_steps) and t.gradient_accumulation_steps >= 1,
          "training.gradient_accumulation_steps must be >= 1")
    check(_is_int(t.length_grouping_window) and t.length_grouping_window >= 1,
          "training.length_grouping_window must be >= 1")

    check(e.every_steps >= 1, "evaluation.every_steps must be >= 1")
    check(1 <= e.loss_examples, "evaluation.loss_examples must be >= 1")
    check(e.final_loss_examples is None or e.final_loss_examples >= 1,
          "evaluation.final_loss_examples must be null or >= 1")
    check(0 <= e.generation_examples, "evaluation.generation_examples must be >= 0")
    check(e.generation_batch_size >= 1, "evaluation.generation_batch_size must be >= 1")
    check(e.decoding.strategy in DECODING_STRATEGIES,
          f"evaluation.decoding.strategy must be one of {DECODING_STRATEGIES}")
    check(e.decoding.num_beams >= 1, "evaluation.decoding.num_beams must be >= 1")
    check(e.decoding.no_repeat_ngram_size >= 0, "evaluation.decoding.no_repeat_ngram_size must be >= 0")
    check(
        not (e.decoding.strategy == "beam" and e.decoding.no_repeat_ngram_size > 0),
        "evaluation.decoding: no_repeat_ngram_size is implemented for greedy decoding only; "
        "set it to 0 for beam search",
    )
    check(e.decoding.max_new_tokens >= 1, "evaluation.decoding.max_new_tokens must be >= 1")
    check(e.early_stopping_patience is None or e.early_stopping_patience >= 1,
          "evaluation.early_stopping_patience must be null or >= 1")
    check(e.calibration_bins >= 2, "evaluation.calibration_bins must be >= 2")
    check(e.select_metric.startswith(("eval/", "generation/", "quality/")),
          "evaluation.select_metric must be an eval/, generation/ or quality/ metric")

    check(config.checkpointing.every_steps >= 1, "checkpointing.every_steps must be >= 1")
    check(config.checkpointing.keep_last >= 1, "checkpointing.keep_last must be >= 1")
    check(config.logging.every_steps >= 1, "logging.every_steps must be >= 1")

    r = config.resources
    check(r.precision in PRECISIONS, f"resources.precision must be one of {PRECISIONS}")
    check(r.threads is None or (_is_int(r.threads) and r.threads >= 1), "resources.threads must be null or >= 1")
    check(_is_int(r.nice) and 0 <= r.nice <= 19, "resources.nice must be in [0, 19]")
    if config.device == "cpu":
        check(r.precision == "fp32", "the cpu profile must use fp32 (bf16 autocast is ~4x slower on CPU)")
        check(r.cores is not None, "the cpu profile must pin explicit cores so it cannot land on pretraining's")

    pretrain_runs = resolve_path("runs/pretrain")
    for label, path in (("logging.logs_dir", config.logging.logs_dir),
                        ("logging.runs_dir", config.logging.runs_dir)):
        check(not _is_within(path, pretrain_runs),
              f"{label} is inside runs/pretrain; SFT output must never mix with the pretraining run's")

    if problems:
        raise SFTConfigError("invalid SFT configuration:\n  - " + "\n  - ".join(problems))


def _overlay(base: Mapping[str, Any], top: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Recursive key-by-key overlay that refuses keys `base` does not have."""
    result = dict(base)
    for key, value in top.items():
        if key not in result:
            raise SFTConfigError(f"{where}: {key!r} does not exist in the base configuration")
        if isinstance(value, Mapping) and isinstance(result[key], Mapping) and key != "targets":
            result[key] = _overlay(result[key], value, f"{where}.{key}")
        else:
            result[key] = value
    return result


def _reject_unknown(raw: Mapping[str, Any], known: set[str], where: str) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SFTConfigError(f"{where}: unknown keys {unknown}; known keys are {sorted(known)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
