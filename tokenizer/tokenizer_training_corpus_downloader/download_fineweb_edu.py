#!/usr/bin/env python3
"""
Stream HuggingFaceFW/fineweb-edu and write up to N MB of cleaned,
tokenizer-ready text.

Role in the blend (tokenizer/DESIGN.md 4.3): 35% -- largest single share,
"highest-quality general English." This is the anchor of the sample:
web text that already passed an educational-quality classifier, so it
skews toward well-formed prose rather than SEO spam. Deliberately NOT
run through strip_markup (FineWeb-Edu is already plain extracted text --
running HTML stripping on it would be a no-op at best).

Usage:
    pip install datasets
    python download_fineweb_edu.py 2800   # ~2.8GB = 35% of an 8GB sample
"""

from typing import Iterator

from common import run_download


def _iter_records(seed: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
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
        dataset_name="FineWeb-Edu",
        source_id="fineweb_edu",
        record_iterator_factory=_iter_records,
        manifest_extra={
            "hf_dataset": "HuggingFaceFW/fineweb-edu",
            "hf_config": "sample-10BT",
            "license": "ODC-By 1.0",
            "blend_role": "35% -- primary high-quality English web prose",
        },
    )
