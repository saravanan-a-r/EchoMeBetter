"""
Pytest configuration and shared fixtures for the whole module.

The module is a flat, package-less directory (its scripts import each other
by bare name, e.g. `import corpus_reader`), so tests need the module root on
sys.path. Placing this file at the module root also makes it pytest's
rootdir, so `pytest` works from anywhere.

Training a real SentencePiece model takes a couple of seconds, so the
expensive fixtures are session-scoped: the pipeline is run once and every
test asserts against that single build, mirroring how production works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import sentencepiece as spm

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import train_tokenization  # noqa: E402
import trainer as trainer_module  # noqa: E402
from config_loader import load_config  # noqa: E402
from tests.corpus_factory import write_synthetic_corpus  # noqa: E402


@pytest.fixture(scope="session")
def module_dir() -> Path:
    return MODULE_DIR


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    return MODULE_DIR / "config" / "default.yaml"


@pytest.fixture(scope="session")
def smoke_config_path() -> Path:
    return MODULE_DIR / "config" / "smoke.yaml"


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory) -> Path:
    """
    A small, deterministic corpus exercising the fidelity edge cases: URLs,
    markdown links, indented code, numbers, multi-byte UTF-8 and whitespace
    runs. Sized so it can support a realistic vocabulary rather than a
    degenerate one.
    """
    directory = tmp_path_factory.mktemp("synthetic_corpus")
    write_synthetic_corpus(directory, n_records=15000, seed=7)
    return directory


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory, synthetic_corpus, smoke_config_path) -> dict:
    """Run the entry point end to end; return its artifacts and processor."""
    output_dir = tmp_path_factory.mktemp("trained_run")
    exit_code = train_tokenization.main(
        [
            "--corpus", str(synthetic_corpus),
            "--config", str(smoke_config_path),
            "--output-dir", str(output_dir),
        ]
    )
    assert exit_code == 0, f"pipeline exited {exit_code}"
    return {
        "cfg": load_config(smoke_config_path),
        "output_dir": output_dir,
        "corpus_dir": synthetic_corpus,
        "config_path": smoke_config_path,
        "model_path": output_dir / "spm.model",
        "sp": trainer_module.load_processor(output_dir / "spm.model"),
    }


@pytest.fixture(scope="session")
def sp(trained_run):
    return trained_run["sp"]


@pytest.fixture(scope="session")
def output_dir(trained_run) -> Path:
    return trained_run["output_dir"]


@pytest.fixture(scope="session")
def lossy_sp(tmp_path_factory, synthetic_corpus):
    """
    A tokenizer trained with the SentencePiece defaults this design
    deliberately rejects: NFKC normalization, dummy prefix, whitespace
    collapsing, no byte fallback, no reserved tokens.

    It is exactly the tokenizer we would have built by accident, and it is
    what the gates and probes are tested against: a check that cannot fail
    proves nothing.
    """
    work_dir = tmp_path_factory.mktemp("lossy")
    corpus = work_dir / "corpus.txt"

    lines: list[str] = []
    for path in sorted(synthetic_corpus.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                for segment in json.loads(line)["text"].split("\n"):
                    if segment.strip():
                        lines.append(segment)
    corpus.write_text("\n".join(lines[:8000]) + "\n", encoding="utf-8")

    prefix = work_dir / "lossy"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=2000,
        byte_fallback=False,
        character_coverage=0.995,
        normalization_rule_name="nmt_nfkc",
        remove_extra_whitespaces=True,
        add_dummy_prefix=True,
        split_digits=False,
        pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
    )
    processor = spm.SentencePieceProcessor()
    processor.load(f"{prefix}.model")
    return processor
