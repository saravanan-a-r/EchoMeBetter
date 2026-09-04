"""
Fixtures for the top-level integration tests (`test/integration/`), which
exercise `pretrain.py` and `pretrain_smoke_test.py` — the scripts that wire
`corpus/`, `UL2/`, `model/` and `training/` together and therefore do not
belong to any one of those packages' own test suites.

Everything here uses the **real, frozen tokenizer** (so real vocabulary IDs
flow through the whole pipeline, the way `TokenizedCorpus` actually produces
them) but a **tiny model** (so a test step costs milliseconds, not the base
profile's makes-a-coffee latency on CPU). Tests that need the tokenizer
artifacts are marked `real_tokenizer` and skip automatically if they are not
built.

The tiny model's `vocab_size` below is set to the *current design target*
(model_config.yml's value), not necessarily the vocab size of whatever
tokenizer happens to be frozen on disk right now -- a model config with a
vocab_size larger than the real tokenizer's actual piece count is harmless
(real token IDs still fit inside the larger embedding table; the extra rows
just go unused until the tokenizer is next retrained), so this is safe to
lead rather than follow the frozen artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
REAL_TOKENIZER_MODEL = ROOT / "tokenizer" / "training" / "output" / "spm.model"

# Matches tokenizer/training/output/token_map.json's actual ids (verified in
# corpus/README.md); a mismatch here would make every batch's decoder-start
# and padding wrong in a way `_check_ids` might not even catch.
PAD_ID = 0
EOS_ID = 1
DECODER_START_ID = 0


def _tiny_model_config_document() -> dict:
    tiny = dict(
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        expected_parameters=None,
    )
    defaults = dict(
        vocab_size=32832,  # model_config.yml's current design target
        d_kv=16,
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
        pad_token_id=PAD_ID,
        decoder_start_token_id=DECODER_START_ID,
        eos_token_id=EOS_ID,
        label_pad_token_id=-100,
    )
    return {"active_profile": "tiny", "defaults": defaults, "profiles": {"tiny": tiny}}


def _tiny_training_config_document(output_dir: Path) -> dict:
    defaults = dict(
        seed=0,
        data_seed=0,
        output_dir=str(output_dir),
        max_steps=4,
        num_train_epochs=1.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        warmup_steps=0,
        warmup_ratio=0.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        bf16=False,
        fp16=False,
        label_smoothing_factor=0.0,
        logging_steps=1,
        # Low enough that a test passing --max-steps 2 with --eval-corpus
        # actually reaches an evaluation, not just constructs one that is
        # never invoked; harmless for the tests that pass no --eval-corpus,
        # since Trainer only consults eval_steps when evaluate is not None.
        eval_steps=2,
        save_steps=2,
        save_total_limit=None,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=False,
        resume_from_checkpoint=False,
    )
    tiny_stage = dict(model_profile="tiny", output_dir=str(output_dir))
    rehearsal_stage = dict(model_profile="tiny", output_dir=str(output_dir), max_steps=4)
    return {
        "active_stage": "tiny",
        "defaults": defaults,
        "stages": {"tiny": tiny_stage, "rehearsal": rehearsal_stage},
    }


@pytest.fixture
def tiny_model_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "model_config.yml"
    path.write_text(yaml.safe_dump(_tiny_model_config_document()), encoding="utf-8")
    return path


@pytest.fixture
def tiny_training_config_path(tmp_path: Path) -> Path:
    path = tmp_path / "training_config.yml"
    path.write_text(
        yaml.safe_dump(_tiny_training_config_document(tmp_path / "runs")), encoding="utf-8"
    )
    return path


@pytest.fixture
def mini_corpus_dir(tmp_path: Path) -> Path:
    lines = [
        "The quarterly report showed strong growth in the European market.",
        "Check https://api.example.com/v2/users for the current status code.",
        "The server crashed because the backup connection failed to respond.",
        "Please reach out to first.last@example.com about the outstanding invoice.",
        "The committee reviewed the roadmap and agreed on next quarter's priorities.",
        "SentencePiece with byte fallback keeps every character representable.",
        "A small encoder-decoder transformer can still learn fluent English text.",
        "Yesterday the team voted to approve the budget after long deliberation.",
    ] * 10
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d
