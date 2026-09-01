#!/usr/bin/env python3
"""
Stream allenai/c4 (en split) and write up to N MB of cleaned,
tokenizer-ready text.

Role in the blend (tokenizer/DESIGN.md 4.3): 20% -- "broad web register
diversity." Where FineWeb-Edu skews toward polished educational prose,
raw C4 brings in more varied, informal, and heterogeneous web register --
important so the vocabulary doesn't overfit to one style of "clean" text
and then fumble on the messier prose typical of real user input to the
rephrase model.

Usage:
    pip install datasets
    python download_c4.py 1600   # ~1.6GB = 20% of an 8GB sample
"""

from typing import Iterator

from common import run_download


def _iter_records(seed: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


if __name__ == "__main__":
    run_download(
        dataset_name="C4 (en)",
        source_id="c4",
        record_iterator_factory=_iter_records,
        manifest_extra={
            "hf_dataset": "allenai/c4",
            "hf_config": "en",
            "license": "ODC-By 1.0",
            "blend_role": "20% -- broad web register diversity",
        },
    )
