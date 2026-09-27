"""
Shared fixtures for the SFT_adapter_rephrase test suite.

Run pytest **from this directory**: every module in this repo calls its code
directory `src`, so each is its own test root.

Almost everything runs a *tiny* base (2+2 layers, 64 dims, 256-token
vocabulary -- the same scale model `model/conftest.py` uses) with a fake
character-level codec, because the properties under test (the adapter is a
no-op at init, the base never changes, accumulation equals one big batch, a
resumed run continues bit-identically) are size-independent.

Two guarantees about this suite itself, since it runs on the machine that is
pretraining:
  - CUDA is hidden before torch loads, so no test can touch the GPU;
  - PyTorch is capped at 2 threads (a tiny matmul split 42 ways is dozens of
    times slower, and would compete with pretraining for cores).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codec import TextCodec  # noqa: E402
from src.config import build_sft_config  # noqa: E402
from src.interop import PROJECT_ROOT, rephrase_model  # noqa: E402

REAL_CONFIG = ROOT / "adapter_config.yml"
REAL_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "training" / "output"
REAL_DATASET_DIR = Path("/home/admin1/projects/REPHRASE_MODEL/rephrase_dataset/final/concise")

TINY_MODEL = dict(
    d_model=64,
    d_ff=176,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    d_kv=16,
    vocab_size=256,
    max_encoder_length=128,
    max_decoder_length=128,
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

# The fake vocabulary: frame tokens at the bottom, one id per character above.
FAKE_TOKEN_MAP = {
    "<pad>": 0,
    "</s>": 1,
    "<unk>": 2,
    "<style:concise>": 3,
    "<text_to_rewrite>": 4,
    "</text_to_rewrite>": 5,
    "<style:grammar>": 6,
}
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'@:/-_`<>"
FIRST_CHAR_ID = 10
for _index, _char in enumerate(ALPHABET):
    FAKE_TOKEN_MAP[f"▁{_char}"] = FIRST_CHAR_ID + _index

FRAME_TOKENS = ["<style:concise>", "<text_to_rewrite>", "</text_to_rewrite>"]


def fake_encode(text: str) -> list[int]:
    """One id per character; the literal frame markup encodes to its control id, as the real tokenizer does."""
    ids, i = [], 0
    while i < len(text):
        for token in FRAME_TOKENS:
            if text.startswith(token, i):
                ids.append(FAKE_TOKEN_MAP[token])
                i += len(token)
                break
        else:
            if text[i] in ALPHABET:
                ids.append(FIRST_CHAR_ID + ALPHABET.index(text[i]))
            i += 1
    return ids


def fake_decode(ids) -> str:
    return "".join(ALPHABET[i - FIRST_CHAR_ID] for i in ids if FIRST_CHAR_ID <= i < FIRST_CHAR_ID + len(ALPHABET))


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def codec() -> TextCodec:
    return TextCodec(fake_encode, fake_decode, FAKE_TOKEN_MAP, "<style:concise>")


@pytest.fixture
def tiny_config():
    return rephrase_model.ModelConfig(**TINY_MODEL)


def build_tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    model = rephrase_model.build_model(rephrase_model.ModelConfig(**TINY_MODEL), verify=False)
    return model.eval()


@pytest.fixture
def tiny_model():
    return build_tiny_model()


def record(text: str, output: str, *, style: str = "concise", cleanup: str = "KEEP", row: int = 0) -> dict:
    return {"row_index": row, "input": text, "style": style, "output": output, "cleanup": cleanup}


# Realistic-shaped concise pairs over the fake alphabet: the output is a
# shortened version of the input, so a tiny model can actually learn it.
_PHRASES = [
    "the server crashed because the backup failed",
    "please send the report to bob by friday",
    "we should review the budget next week",
    "the team agreed on the new roadmap today",
    "check the logs at 10:30 for the error",
    "ask @alice about the invoice number 4521",
    "i think the build is broken again today",
    "the meeting moved to room 12 on monday",
]


def sample_records(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        phrase = _PHRASES[index % len(_PHRASES)]
        text = f"well, honestly, {phrase} number {index} okay"
        rows.append(record(text, f"{phrase} {index}", row=index))
    return rows


def write_dataset(directory: Path, rows: list[dict], files: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for file_index in range(files):
        chunk = rows[file_index::files]
        with (directory / f"part_{file_index:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for row in chunk:
                handle.write(json.dumps(row) + "\n")
    return directory


def config_document(tmp_path: Path, **section_overrides) -> dict:
    """A complete, small config document; `section_overrides` patch sections key by key."""
    document = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    document["base_model"] = {"checkpoint": str(tmp_path / "base"), "tokenizer_dir": str(tmp_path / "tok")}
    document["adapter"].update(rank=4, alpha=8.0, dropout=0.0)
    # The fake codec's style is concise, so the tests run on concise's values,
    # pinned here rather than read from the shipped file: `styles` is cleared
    # so a test's own data/evaluation values are what it gets (the real
    # profiles have their own test in test_config.py), and the split salt is
    # fixed because expected counts depend on the order it puts records in.
    document["styles"] = {}
    document["evaluation"]["acceptance"]["max_length_ratio"] = 1.0
    document["data"].update(
        style="concise", train_data=str(tmp_path / "data"), eval_data=None, split_salt="concise-v1",
        train_examples=24, eval_examples=8,
        max_source_tokens=120, max_target_tokens=100, max_length_ratio=1.15,
    )
    document["training"].update(
        epochs=1, max_steps=None, learning_rate=3e-3, warmup_steps=0, lr_schedule="constant",
        batch_size=4, gradient_accumulation_steps=2, length_grouping_window=2, seed=7,
    )
    document["evaluation"].update(
        every_steps=2, loss_examples=8, final_loss_examples=None, generation_examples=4,
        generation_batch_size=4, text_samples=2, early_stopping_patience=None,
    )
    document["evaluation"]["decoding"]["max_new_tokens"] = 40
    document["checkpointing"] = {"every_steps": 2, "keep_last": 2}
    document["logging"] = {"every_steps": 1, "logs_dir": str(tmp_path / "logs"), "runs_dir": str(tmp_path / "runs")}
    # Test profiles: the real device overrides would resize the batches.
    for device in ("cpu", "gpu"):
        document["devices"][device]["overrides"] = {}
    for section, values in section_overrides.items():
        document[section].update(values)
    return document


def make_config(tmp_path: Path, device: str = "cpu", **section_overrides):
    return build_sft_config(config_document(tmp_path, **section_overrides), device)
