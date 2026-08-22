"""
Model configuration, loaded from `model_config.yml`.

Single source of truth
----------------------
Every architectural dimension comes from `model_config.yml` at the project
root. No layer count, width or vocabulary size is written in Python. The
point is that reconfiguring the model is a YAML edit, not a code change, and
that a run manifest can record exactly one object that fully determines the
model's shape.

The parameter-count formula
---------------------------
`parameter_count()` computes the total *without building the model*, which
makes three things possible:

  - checking a configuration before allocating a GB of weights
  - verifying the built model matches the formula (they are independent
    derivations, so agreement is real evidence rather than a tautology)
  - verifying both against `expected_parameters` in the YAML, which is a
    deliberate regression lock: change a dimension and the check fails until
    someone consciously recomputes the expectation

architecture.md §4.3 works the same total by hand; `test_architecture_contract`
checks all three agree.

HuggingFace naming
------------------
Every field that has a counterpart in `transformers.T5Config` uses **that
field's exact name and value convention**, so someone who wants to fine-tune a
released checkpoint can read this configuration without a translation table:

    d_model  d_kv  d_ff  num_layers  num_decoder_layers  num_heads
    relative_attention_num_buckets  relative_attention_max_distance
    dropout_rate  layer_norm_epsilon  initializer_factor  feed_forward_proj
    tie_word_embeddings  vocab_size  pad_token_id  eos_token_id
    decoder_start_token_id

`dense_act_fn` and `is_gated_act` are derived from `feed_forward_proj` by the
same string split HuggingFace uses, so the derivation cannot drift from it.

The remaining fields have no `T5Config` counterpart — either because
HuggingFace hardcodes the behaviour (`scale_attention_scores`,
`scale_output_for_tied_embeddings`), or because it belongs to training rather
than to the network (`label_smoothing`, `label_pad_token_id`), or because it
is ours (`max_encoder_length`, `max_decoder_length`, `expected_parameters`).
`to_huggingface_config()` handles all of them explicitly and refuses to
produce a configuration whose behaviour HuggingFace would not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError

# model/src/config.py -> model/src -> model -> project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "model_config.yml"

# HuggingFace `feed_forward_proj` strings, in HF's own "<gating>-<activation>"
# form. Both are gated (GeGLU) and both resolve to ACT2FN["gelu_new"] — the
# tanh-approximated GELU that T5.1.1 uses — so `transformers` reproduces this
# FFN's arithmetic exactly either way. See `parse_feed_forward_proj` for the
# non-obvious reason "gated-gelu" is one of them.
#
#   gated-gelu      the standard T5.1.1 string; what this model declares
#   gated-gelu_new  an explicit synonym, resolving identically
#
# There is deliberately no ungated option (architecture.md §4.1 fixed the FFN
# as GeGLU, and the §4.3 budget assumes its three matrices), and no exact-GELU
# option: `transformers` has no gated string that selects the erf-based GELU,
# so offering one would mean offering a model HuggingFace cannot reproduce.
_ACTIVATIONS = ("gated-gelu", "gated-gelu_new")

# HuggingFace's `model_type` for this architecture. A released checkpoint
# carrying this in config.json is loadable by `T5ForConditionalGeneration`.
HUGGINGFACE_MODEL_TYPE = "t5"
HUGGINGFACE_ARCHITECTURE = "T5ForConditionalGeneration"


def parse_feed_forward_proj(value: str) -> tuple[bool, str]:
    """
    Split a `feed_forward_proj` string into `(is_gated_act, dense_act_fn)`.

    This reproduces `transformers.T5Config.__init__`, including its special
    case:

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"
        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

    That last line is easy to miss and it matters: **"gated-gelu" means the
    tanh-approximated GELU, not the erf-based one.** Reading the string
    naively — as `split("-")[-1]` alone — gives "gelu", and a model built on
    that reading would use a different activation from the one HuggingFace
    uses for the same config.json. The consequence is a checkpoint that
    fine-tunes to slightly wrong numbers with nothing to indicate why.

    Kept as a free function, separate from `ModelConfig`, so it can be tested
    against strings the dataclass would reject — `ModelConfig` only admits
    gated values, so a bug that reported *everything* as gated would be
    invisible if this logic could only be reached through a valid config.
    """
    parts = value.split("-")
    is_gated = parts[0] == "gated"
    activation = parts[-1]
    if value == "gated-gelu":
        activation = "gelu_new"
    return is_gated, activation

# Fields a profile may set. Anything else in a profile block is a typo, and a
# silently-ignored typo means training a model that is not the one intended.
_PROFILE_FIELDS = frozenset(
    {
        "d_model",
        "d_ff",
        "num_layers",
        "num_decoder_layers",
        "num_heads",
        "expected_parameters",
    }
)


@dataclass(frozen=True)
class ModelConfig:
    """
    The complete architectural specification of one model.

    Frozen: these values are welded into the shape of every weight tensor in
    every checkpoint. A mutable config would allow a mid-run change that
    silently invalidates checkpoints written before it.
    """

    # -- shape ------------------------------------------------------------
    # Names match `transformers.T5Config` exactly. `num_layers` is the
    # *encoder* depth, which is what HuggingFace means by it; the decoder
    # depth is `num_decoder_layers`.
    d_model: int
    d_ff: int
    num_layers: int
    num_decoder_layers: int
    num_heads: int
    d_kv: int
    vocab_size: int

    # -- context ----------------------------------------------------------
    max_encoder_length: int
    max_decoder_length: int

    # -- relative position bias -------------------------------------------
    relative_attention_num_buckets: int
    relative_attention_max_distance: int

    # -- behaviour --------------------------------------------------------
    layer_norm_epsilon: float
    feed_forward_proj: str
    tie_word_embeddings: bool
    scale_output_for_tied_embeddings: bool
    scale_attention_scores: bool
    dropout_rate: float
    label_smoothing: float
    initializer_factor: float

    # -- token ids --------------------------------------------------------
    pad_token_id: int
    decoder_start_token_id: int
    eos_token_id: int
    label_pad_token_id: int

    # -- provenance -------------------------------------------------------
    profile: str = "unnamed"
    expected_parameters: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "num_heads": self.num_heads,
            "d_kv": self.d_kv,
            "vocab_size": self.vocab_size,
            "max_encoder_length": self.max_encoder_length,
            "max_decoder_length": self.max_decoder_length,
            "relative_attention_max_distance": self.relative_attention_max_distance,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(f"{name} must be a positive integer, got {value!r}")

        buckets = self.relative_attention_num_buckets
        if not isinstance(buckets, int) or isinstance(buckets, bool) or buckets < 4:
            raise ConfigError(
                f"relative_attention_num_buckets must be an integer >= 4, got {buckets!r}"
            )
        if buckets % 2 != 0:
            raise ConfigError(
                f"relative_attention_num_buckets must be even ({buckets} is not): the "
                f"bidirectional encoder bucketing splits the range in half, one side "
                f"per direction"
            )
        # Bidirectional halves the range, then half of *that* is the exact
        # (non-logarithmic) region. Below 4 buckets there is no logarithmic
        # region left and the scheme degenerates.
        max_exact = buckets // 4
        if self.relative_attention_max_distance <= max_exact:
            raise ConfigError(
                f"relative_attention_max_distance ({self.relative_attention_max_distance}) "
                f"must exceed the exact-bucket region ({max_exact}), or every distant "
                f"position collapses into a single bucket"
            )

        if self.feed_forward_proj not in _ACTIVATIONS:
            raise ConfigError(
                f"feed_forward_proj must be one of {_ACTIVATIONS}, "
                f"got {self.feed_forward_proj!r}. The value is HuggingFace's "
                f"'<gating>-<activation>' string; only gated (GeGLU) variants are "
                f"implemented, because architecture.md §4.1 fixed the FFN as GeGLU "
                f"and the parameter budget assumes its three matrices."
            )

        for name, value in (
            ("dropout_rate", self.dropout_rate),
            ("label_smoothing", self.label_smoothing),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ConfigError(f"{name} must be in [0.0, 1.0), got {value!r}")

        if float(self.layer_norm_epsilon) <= 0.0:
            raise ConfigError(f"layer_norm_epsilon must be positive, got {self.layer_norm_epsilon}")
        if float(self.initializer_factor) <= 0.0:
            raise ConfigError(
                f"initializer_factor must be positive, got {self.initializer_factor}"
            )

        for name, value in (
            ("pad_token_id", self.pad_token_id),
            ("decoder_start_token_id", self.decoder_start_token_id),
            ("eos_token_id", self.eos_token_id),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")
            if value >= self.vocab_size:
                raise ConfigError(
                    f"{name}={value} is outside the {self.vocab_size}-token vocabulary"
                )

        if self.label_pad_token_id >= 0:
            raise ConfigError(
                f"label_pad_token_id must be negative so it can never collide with a "
                f"real vocabulary ID, got {self.label_pad_token_id}"
            )

        if self.expected_parameters is not None and self.expected_parameters < 1:
            raise ConfigError(
                f"expected_parameters must be positive or absent, got {self.expected_parameters}"
            )

    # -- derived ----------------------------------------------------------

    @property
    def inner_dim(self) -> int:
        """Total attention width: `num_heads x d_kv`."""
        return self.num_heads * self.d_kv

    @property
    def dense_act_fn(self) -> str:
        """
        The activation name inside `feed_forward_proj`.

        Derived with the same split HuggingFace's `T5Config` uses, rather than
        a lookup table that could drift away from it. The result is a real
        `transformers.ACT2FN` key.
        """
        return parse_feed_forward_proj(self.feed_forward_proj)[1]

    @property
    def is_gated_act(self) -> bool:
        """Whether the FFN is gated. HuggingFace derives this identically."""
        return parse_feed_forward_proj(self.feed_forward_proj)[0]

    # HuggingFace's `T5Config.attribute_map` exposes these four aliases.
    # Mirrored exactly, so code written against either vocabulary reads the
    # same value. They are read-only and never serialized: `to_dict()` and
    # `to_huggingface_config()` both emit the canonical names, which is what
    # `T5Config` itself writes into config.json.

    @property
    def hidden_size(self) -> int:
        return self.d_model

    @property
    def num_attention_heads(self) -> int:
        return self.num_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @property
    def head_dim(self) -> int:
        return self.d_kv

    def parameter_count(self) -> int:
        """
        Total trainable parameters, derived from the configuration alone.

        Matches architecture.md §4.3's hand-worked table. Note what is absent:
        attention and feed-forward projections carry **no bias** (T5
        convention), RMSNorm has a weight but no bias, and the relative
        position bias has no per-position parameters — which is why context
        length does not appear in this formula at all.
        """
        embeddings = self.vocab_size * self.d_model

        attention = 4 * self.d_model * self.inner_dim          # Q, K, V, O
        feed_forward = 3 * self.d_model * self.d_ff            # gate, up, down

        encoder_layer = attention + feed_forward + 2 * self.d_model
        decoder_layer = 2 * attention + feed_forward + 3 * self.d_model

        position_bias = 2 * self.relative_attention_num_buckets * self.num_heads
        final_norms = 2 * self.d_model

        total = (
            embeddings
            + self.num_layers * encoder_layer
            + self.num_decoder_layers * decoder_layer
            + position_bias
            + final_norms
        )
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.d_model
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-component counts, in the order architecture.md §4.3 lists them."""
        attention = 4 * self.d_model * self.inner_dim
        feed_forward = 3 * self.d_model * self.d_ff
        encoder_layer = attention + feed_forward + 2 * self.d_model
        decoder_layer = 2 * attention + feed_forward + 3 * self.d_model

        breakdown = {
            "token_embeddings": self.vocab_size * self.d_model,
            "encoder_per_layer": encoder_layer,
            "encoder_total": self.num_layers * encoder_layer,
            "decoder_per_layer": decoder_layer,
            "decoder_total": self.num_decoder_layers * decoder_layer,
            "relative_position_bias": 2 * self.relative_attention_num_buckets * self.num_heads,
            "final_norms": 2 * self.d_model,
        }
        if not self.tie_word_embeddings:
            breakdown["lm_head"] = self.vocab_size * self.d_model
        breakdown["total"] = self.parameter_count()
        return breakdown

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, for run manifests and checkpoint metadata."""
        return {
            "profile": self.profile,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "num_heads": self.num_heads,
            "d_kv": self.d_kv,
            "vocab_size": self.vocab_size,
            "max_encoder_length": self.max_encoder_length,
            "max_decoder_length": self.max_decoder_length,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "relative_attention_max_distance": self.relative_attention_max_distance,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "feed_forward_proj": self.feed_forward_proj,
            "tie_word_embeddings": self.tie_word_embeddings,
            "scale_output_for_tied_embeddings": self.scale_output_for_tied_embeddings,
            "scale_attention_scores": self.scale_attention_scores,
            "dropout_rate": self.dropout_rate,
            "label_smoothing": self.label_smoothing,
            "initializer_factor": self.initializer_factor,
            "pad_token_id": self.pad_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
            "eos_token_id": self.eos_token_id,
            "label_pad_token_id": self.label_pad_token_id,
            "expected_parameters": self.expected_parameters,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        """Inverse of `to_dict`. Unknown keys are rejected, never ignored."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown model configuration keys: {sorted(unknown)}")

        missing = {f for f in known if f not in data} - {"profile", "expected_parameters"}
        if missing:
            raise ConfigError(f"missing model configuration keys: {sorted(missing)}")

        return cls(**data)

    def with_(self, **changes: Any) -> "ModelConfig":
        """Copy with overrides; re-runs all validation."""
        return replace(self, **changes)

    # -- HuggingFace interoperability -------------------------------------

    def to_huggingface_config(self) -> dict[str, Any]:
        """
        A `transformers.T5Config` document for this model.

        The intended use is releasing a checkpoint someone else fine-tunes:

            import json, pathlib
            pathlib.Path("config.json").write_text(
                json.dumps(config.to_huggingface_config(), indent=2)
            )

        after which `T5Config.from_pretrained(...)` and
        `T5ForConditionalGeneration` describe the same network this package
        builds — same depths, same widths, same activation, same tying, same
        relative-position bucketing.

        Refusing rather than exporting
        ------------------------------
        Two settings in this configuration have no `T5Config` counterpart
        because HuggingFace hardcodes them. If either is set to a value
        HuggingFace cannot express, exporting anyway would produce a
        `config.json` that *looks* right and describes a different model, so
        this raises instead. That failure is the point: it is far cheaper than
        someone discovering the mismatch after fine-tuning against it.
        """
        if self.scale_attention_scores:
            raise ConfigError(
                "cannot export to HuggingFace with scale_attention_scores=True: "
                "T5 never divides attention scores by sqrt(d_kv) — it folds that "
                "scaling into the query initializer — and T5Config has no field "
                "to express the alternative. An exported config would silently "
                "describe a differently-behaving model."
            )
        if self.tie_word_embeddings and not self.scale_output_for_tied_embeddings:
            raise ConfigError(
                "cannot export to HuggingFace with tie_word_embeddings=True and "
                "scale_output_for_tied_embeddings=False: T5ForConditionalGeneration "
                "always applies the d_model**-0.5 output scaling when embeddings "
                "are tied, and has no field to disable it."
            )

        return {
            "architectures": [HUGGINGFACE_ARCHITECTURE],
            "model_type": HUGGINGFACE_MODEL_TYPE,
            # -- fields T5Config defines, under its own names ---------------
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_kv": self.d_kv,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "num_heads": self.num_heads,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "relative_attention_max_distance": self.relative_attention_max_distance,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_factor": self.initializer_factor,
            "feed_forward_proj": self.feed_forward_proj,
            # Derived by T5Config itself; written out so a reader of the file
            # sees the resolved values rather than having to redo the split.
            "dense_act_fn": self.dense_act_fn,
            "is_gated_act": self.is_gated_act,
            "is_encoder_decoder": True,
            "use_cache": True,
            "tie_word_embeddings": self.tie_word_embeddings,
            # Recent `transformers` reads this rather than re-deriving the
            # d_model**-0.5 output scaling from `tie_word_embeddings`, and
            # writes it into every config.json it saves. Emitted explicitly so
            # the exported file matches what HuggingFace itself would write.
            # Older versions ignore it and derive the same value from tying,
            # which is why the refusal above still stands: it keeps the two
            # readings from ever disagreeing.
            "scale_decoder_outputs": (
                self.tie_word_embeddings and self.scale_output_for_tied_embeddings
            ),
            "classifier_dropout": 0.0,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
            # -- ours; T5Config has no equivalent --------------------------
            # `PretrainedConfig` keeps unrecognized keys, so these survive a
            # load/save round-trip. They are carried rather than dropped
            # because the context limit is a real property of the released
            # weights: this model was never trained past 2048 either side.
            "max_encoder_length": self.max_encoder_length,
            "max_decoder_length": self.max_decoder_length,
        }


def load_model_config(
    path: str | Path | None = None,
    profile: str | None = None,
) -> ModelConfig:
    """
    Load a profile from `model_config.yml`.

    `path` defaults to the project-root file; `profile` defaults to the
    file's own `active_profile`, so a pipeline that passes neither gets
    exactly what the YAML says it should — one place to change, as intended.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"model configuration not found: {path}")

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"model configuration {path} is not valid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise ConfigError(f"model configuration {path} must be a YAML mapping")

    return build_model_config(document, profile=profile, source=str(path))


def build_model_config(
    document: Mapping[str, Any],
    profile: str | None = None,
    source: str = "<mapping>",
) -> ModelConfig:
    """
    Build a `ModelConfig` from an already-parsed configuration document.

    Separate from `load_model_config` so the merge logic can be tested
    without touching the filesystem, and so a caller holding configuration
    from elsewhere (a checkpoint, a test) can use the identical code path.
    """
    defaults = document.get("defaults")
    if not isinstance(defaults, dict):
        raise ConfigError(f"{source}: missing or malformed 'defaults' block")

    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError(f"{source}: missing or empty 'profiles' block")

    name = profile if profile is not None else document.get("active_profile")
    if name is None:
        raise ConfigError(
            f"{source}: no profile requested and no 'active_profile' declared"
        )
    if name not in profiles:
        raise ConfigError(
            f"{source}: unknown profile {name!r}; available: {sorted(profiles)}"
        )

    selected = profiles[name]
    if not isinstance(selected, dict):
        raise ConfigError(f"{source}: profile {name!r} must be a mapping")

    unknown = set(selected) - _PROFILE_FIELDS
    if unknown:
        raise ConfigError(
            f"{source}: profile {name!r} has unknown keys {sorted(unknown)}. "
            f"A silently-ignored key means training a model that is not the one "
            f"intended; put shared settings in 'defaults'."
        )

    # A profile overriding a default is legitimate; a profile *colliding* with
    # a default it did not mean to override is not distinguishable here, which
    # is why profile keys are restricted to the shape fields above.
    merged: dict[str, Any] = {**defaults, **selected, "profile": name}

    try:
        return ModelConfig.from_dict(merged)
    except ConfigError as exc:
        raise ConfigError(f"{source}: profile {name!r}: {exc}") from exc


def available_profiles(path: str | Path | None = None) -> list[str]:
    """Profile names declared in the configuration file."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"model configuration not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return sorted((document or {}).get("profiles", {}))
