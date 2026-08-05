#!/usr/bin/env python3
"""
Stream a public-domain Gutenberg/PG-19 mirror and write up to N MB of
cleaned, tokenizer-ready text.

Role in the blend (tokenizer/DESIGN.md 4.3): 20% -- "long-form literary
English; balances comment-heavy StackExchange." This is deliberately the
LARGEST non-web share after the code slice was removed (DESIGN.md 4.3
history): StackExchange skews short/conversational (DESIGN.md 4.4), so
long, structurally coherent, multi-paragraph literary prose is what
teaches the tokenizer good pieces for sustained narrative text -- the
opposite failure mode from fragment-only vocabulary.

Uses emozilla/pg19, a Parquet-format mirror of the standard PG-19
long-document benchmark (deepmind/pg19, public domain). The original
deepmind/pg19 repo uses a legacy HF "loading script," which current
versions of the `datasets` library refuse to run
(RuntimeError: "Dataset scripts are no longer supported"). emozilla/pg19
is the same content, same "text" field, re-published in Parquet so it
loads under streaming=True without trust_remote_code. Each PG-19 "book"
record is long (often 100K+ chars), so this is exactly what
transform.chunk_text() exists for -- verified against risk R1 in
DESIGN.md.

Strips the standard Gutenberg legal boilerplate header/footer via
transform.is_boilerplate() line filtering (does not touch the book text
itself).

Usage:
    pip install datasets
    python download_gutenberg.py 1600   # ~1.6GB = 20% of an 8GB sample
"""

from typing import Iterator

from common import run_download


def _iter_records(seed: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2_000)
    for row in ds:
        text = row.get("text")
        if text:
            yield text


if __name__ == "__main__":
    run_download(
        dataset_name="Gutenberg / PG-19",
        source_id="gutenberg_pg19",
        record_iterator_factory=_iter_records,
        manifest_extra={
            "hf_dataset": "emozilla/pg19",
            "hf_dataset_origin": "Parquet mirror of deepmind/pg19 (legacy loading-script version no longer loadable)",
            "license": "Public domain (pre-1919 US works)",
            "blend_role": "20% -- long-form literary English, balances short StackExchange text",
        },
    )
