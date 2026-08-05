#!/usr/bin/env python3
"""
Read from the already-local StackExchange corpus (this project's own
cleaned DATASET/*.jsonl files) and write up to N MB of tokenizer-ready
text. No network download -- named "download_" to match the other
scripts in this folder and keep the run interface identical
(`python download_stackexchange.py <size_mb>`), but it streams from disk.

Role in the blend (tokenizer/DESIGN.md 4.3): 15% -- deliberately
over-weighted relative to its ~3% share of total pretraining tokens
(DESIGN.md 4.1), because this is the actual inference-time input
distribution for the rephrase model. The tokenizer must be efficient on
the text users will really submit, not just on generic web prose.

Source files already went through this project's own PII-masking and
HTML-cleanup pipeline (see project memory: phone/email masking, HTML tag
removal) -- so strip_markup is NOT needed here (defaults off), and this
script does not re-run any of that cleanup. It only applies the generic
transform.clean_record() pass (whitespace/boilerplate/chunking) that
every source gets, for consistency with the other four scripts.

Reads DATASET/splits/phase_1_part_*.jsonl and DATASET/phase_2..6.jsonl,
in that order, shuffled record-within-file by the seed (not across the
full 13.8M-row corpus, to avoid loading everything into memory at once).

Usage:
    python download_stackexchange.py 1200   # ~1.2GB = 15% of an 8GB sample
"""

import json
import random
import sys
from pathlib import Path
from typing import Iterator

from common import run_download

DATASET_DIR = Path("/Users/apple/projects/dataset_downloaded/stack_exchanger/DATASET")


def _source_files() -> list[Path]:
    files: list[Path] = []
    splits_dir = DATASET_DIR / "splits"
    if splits_dir.is_dir():
        files.extend(sorted(splits_dir.glob("phase_1_part_*.jsonl")))
    files.extend(sorted(DATASET_DIR.glob("phase_[2-6].jsonl")))
    return files


def _iter_records(seed: int) -> Iterator[str]:
    files = _source_files()
    if not files:
        print(f"error: no source files found under {DATASET_DIR}", file=sys.stderr)
        return
    rng = random.Random(seed)
    rng.shuffle(files)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        rng.shuffle(lines)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if text:
                yield text


if __name__ == "__main__":
    run_download(
        dataset_name="StackExchange (local, cleaned)",
        source_id="stackexchange",
        record_iterator_factory=_iter_records,
        manifest_extra={
            "source": f"local: {DATASET_DIR}",
            "license": "CC BY-SA 4.0 (StackExchange dump license)",
            "blend_role": "15% -- over-weighted vs pretraining share; matches inference-time input distribution",
            "already_pii_masked": True,
        },
    )
