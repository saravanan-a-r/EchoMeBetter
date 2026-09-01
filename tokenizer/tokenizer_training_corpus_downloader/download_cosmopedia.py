#!/usr/bin/env python3
"""
Stream HuggingFaceTB/cosmopedia (v2) and write up to N MB of cleaned,
tokenizer-ready text.

Role in the blend (tokenizer/DESIGN.md 4.3): 10% -- "clean synthetic
instructional prose." Cosmopedia is LLM-generated textbook/tutorial-style
text: consistently well-formed, explanatory register. It contributes
vocabulary for clear, structured explanation -- closer in register to
what the "elaborate" and "professional" rephrase styles need to produce
than raw web text is.

Uses the "web_samples_v2" config, one of the largest and most general
subsets. Apache-2.0 licensed (architecture.md 8.1).

Usage:
    pip install datasets
    python download_cosmopedia.py 800   # ~0.8GB = 10% of an 8GB sample
"""

from typing import Iterator

from common import run_download


def _iter_records(seed: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceTB/cosmopedia",
        "web_samples_v2",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


if __name__ == "__main__":
    run_download(
        dataset_name="Cosmopedia v2",
        source_id="cosmopedia",
        record_iterator_factory=_iter_records,
        manifest_extra={
            "hf_dataset": "HuggingFaceTB/cosmopedia",
            "hf_config": "web_samples_v2",
            "license": "Apache-2.0",
            "blend_role": "10% -- clean synthetic instructional prose",
        },
    )
