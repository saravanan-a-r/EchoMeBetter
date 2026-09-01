"""
`TokenizedCorpus` against the real, frozen SentencePiece model — the seam
between this package and `tokenizer/`. Skips automatically if the tokenizer
has not been built, the same convention `UL2/`'s `real_tokenizer`-marked
tests use.
"""

from __future__ import annotations

import pytest

from src.tokenized_corpus import TokenizedCorpus
from src.tokenizer_interop import DEFAULT_TOKENIZER_DIR, load_processor

from conftest import REAL_TOKENIZER_MODEL, write_txt

pytestmark = [
    pytest.mark.real_tokenizer,
    pytest.mark.skipif(
        not REAL_TOKENIZER_MODEL.is_file(),
        reason=f"tokenizer not built: {REAL_TOKENIZER_MODEL} missing",
    ),
]


@pytest.fixture
def sp():
    return load_processor(DEFAULT_TOKENIZER_DIR / "spm.model")


def test_real_tokenizer_produces_nonempty_token_ids(corpus_dir, sp):
    write_txt(
        corpus_dir / "a.txt",
        ["Check https://example.com/path now.", "The server crashed at 10.0.0.5."],
    )
    corpus = TokenizedCorpus([corpus_dir / "a.txt"], sp, seed=0)
    first, second = next(corpus), next(corpus)
    assert first and all(isinstance(t, int) for t in first)
    assert second and first != second


def test_a_literal_spiece_underline_survives_through_escaping(corpus_dir, sp):
    """
    architecture.md §6.4: every encode of arbitrary text must route through
    the marker escaper, or a literal '▁' in real text (ASCII block-shading)
    is silently eaten by SentencePiece's decoder. This is the one thing that
    would go unnoticed by a fake-processor unit test.
    """
    write_txt(corpus_dir / "a.txt", ["ship it ▁▁▁ now"])
    corpus = TokenizedCorpus([corpus_dir / "a.txt"], sp, seed=0)
    ids = next(corpus)
    from src.tokenizer_interop import unescape_markers

    decoded = unescape_markers(sp.decode(ids))
    assert "▁▁▁" in decoded
