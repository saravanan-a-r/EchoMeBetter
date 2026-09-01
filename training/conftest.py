"""
Shared fixtures for the training test suite.

Almost every test runs a *tiny* model for a handful of steps. The properties
under test — that accumulation equals one large batch, that a resumed run
continues bit-identically, that weight decay reaches the right tensors — are
all size-independent, and a suite that trains a 750M-parameter model is a
suite nobody runs.

Everything is forced onto the CPU and onto float32. That is not laziness: a
test asserting `torch.equal` on a resumed run would be flaky under bf16
autocast on some devices and under MPS's non-deterministic reductions, and a
flaky determinism test is worse than none — it trains people to re-run until
it passes.

Run pytest **from this directory**. `model/`, `UL2/` and `training/` each
contain a package called `src`, so they are three independent test roots; the
sibling packages are reached through `src/interop.py`, which registers them
under unambiguous names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TrainingConfig  # noqa: E402
from src.interop import rephrase_model  # noqa: E402

PROJECT_ROOT = ROOT.parent
TRAINING_CONFIG_PATH = PROJECT_ROOT / "training_config.yml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "model_config.yml"

# The same tiny model `model/conftest.py` uses, so a failure here is never
# "the trainer's model is shaped differently from the model's model".
TINY_MODEL = dict(
    d_model=64,
    d_ff=176,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_kv=16,
    vocab_size=256,
    max_encoder_length=64,
    max_decoder_length=64,
    relative_attention_num_buckets=8,
    relative_attention_max_distance=16,
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
    tie_word_embeddings=True,
    scale_output_for_tied_embeddings=True,
    scale_attention_scores=False,
    dropout_rate=0.0,
    label_smoothing=0.0,
    initializer_factor=1.0,
    pad_token_id=0,
    decoder_start_token_id=0,
    eos_token_id=1,
    label_pad_token_id=-100,
    profile="tiny",
    expected_parameters=None,
)

# A complete `TrainingConfig`, deliberately spelled out in full rather than
# loaded from the YAML: a unit test that breaks because someone retuned the
# real learning rate is a test that punishes the wrong action.
TINY_TRAINING = dict(
    output_dir="runs/test",
    seed=0,
    data_seed=0,
    max_steps=8,
    num_train_epochs=1.0,
    learning_rate=1e-3,
    lr_scheduler_type="inverse_sqrt",
    warmup_steps=0,       # tests that care about warmup set it themselves
    warmup_ratio=0.0,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    bf16=False,          # float32 on CPU; see the module docstring
    fp16=False,
    label_smoothing_factor=0.0,
    logging_steps=1,
    eval_steps=4,
    save_steps=4,
    save_total_limit=None,
    metric_for_best_model="loss",
    greater_is_better=False,
    load_best_model_at_end=False,
    resume_from_checkpoint=False,
    stage="test",
    model_profile=None,
    dropout_rate=None,
    lr_timescale=None,
    min_lr_ratio=0.0,
    num_cycles=0.5,
    eval_max_batches=None,
    keep_best_checkpoint=True,
    log_per_mode_loss=False,
)

# IDs below this are reserved (pad/eos); test data starts above them so a
# generated pad or eos is always meaningful rather than coincidental.
FIRST_CONTENT_ID = 8

# The three UL2 denoisers, cycled through so per-mode tests see all of them.
MODE_CYCLE = ("R", "X", "S")


@pytest.fixture
def model_config():
    return rephrase_model.ModelConfig(**TINY_MODEL)


@pytest.fixture
def tiny_model(model_config):
    torch.manual_seed(0)
    return rephrase_model.build_model(model_config, verify=False)


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(**TINY_TRAINING)


@pytest.fixture
def cpu() -> torch.device:
    """Every test pins the device, so none of them depends on the host."""
    return torch.device("cpu")


def make_batch(
    config,
    batch_size: int = 2,
    source_length: int = 12,
    target_length: int = 8,
    seed: int = 0,
    modes: list[str] | None = None,
) -> dict[str, object]:
    """
    A well-formed training batch, in the shape `UL2/src/batching.pad_batch`
    produces — including the `modes` list, which is what makes per-mode loss
    reporting possible without anything translating between the two modules.

    Row 0 has a short source and row 1 a short target, so both padding paths
    are exercised by every test that uses this.
    """
    generator = torch.Generator().manual_seed(seed)

    encoder_input_ids = torch.randint(
        FIRST_CONTENT_ID, config.vocab_size, (batch_size, source_length), generator=generator
    )
    target = torch.randint(
        FIRST_CONTENT_ID, config.vocab_size, (batch_size, target_length), generator=generator
    )

    encoder_attention_mask = torch.ones(batch_size, source_length, dtype=torch.long)
    decoder_attention_mask = torch.ones(batch_size, target_length, dtype=torch.long)
    if batch_size > 1 and source_length > 4:
        encoder_attention_mask[0, source_length - 3:] = 0
        encoder_input_ids[0, source_length - 3:] = config.pad_token_id
    if batch_size > 1 and target_length > 4:
        decoder_attention_mask[1, target_length - 2:] = 0

    decoder_input_ids = torch.cat(
        [
            torch.full((batch_size, 1), config.decoder_start_token_id, dtype=torch.long),
            target[:, :-1],
        ],
        dim=1,
    )
    labels = target.masked_fill(decoder_attention_mask == 0, config.label_pad_token_id)

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
        "modes": modes
        or [MODE_CYCLE[(seed + i) % len(MODE_CYCLE)] for i in range(batch_size)],
    }


def make_batches(config, count: int, **kwargs) -> list[dict[str, object]]:
    """`count` distinct batches — distinct, so a bug that reuses one shows."""
    return [make_batch(config, seed=index, **kwargs) for index in range(count)]


class ListIterator:
    """
    A minimal `Resumable` over a list of batches.

    Exists to prove the resumption protocol is small enough to satisfy in a
    few lines — the corpus module is not built yet, and if this were awkward
    to implement in a test it would be awkward to implement there.
    """

    def __init__(self, batches: list[dict[str, object]]) -> None:
        self.batches = batches
        self.position = 0

    def __iter__(self) -> "ListIterator":
        return self

    def __next__(self) -> dict[str, object]:
        if self.position >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.position]
        self.position += 1
        return batch

    def state_dict(self) -> dict[str, object]:
        return {"position": self.position}

    def load_state_dict(self, state) -> None:
        self.position = int(state["position"])
