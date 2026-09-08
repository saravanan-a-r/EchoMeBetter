"""
`RewriteExampleSource`: the frame, the budgets, and exact resumption.
"""

from __future__ import annotations

import json
import random

import pytest

from conftest import ListRecords, fake_decode, fake_encode
from src.corruption import IDENTITY, CorruptionConfig, Corruptor
from src.errors import RewriteConfigError
from src.source import REWRITE_MODE, RewriteExampleSource

PROSE = (
    "The quarterly report showed strong growth in the European market, and the "
    "team expects it to continue into next year."
)
CODE = "def parse(line):\n    return json.loads(line)"


def build(records, tokens, *, seed=0, max_source=4096, max_target=4096, min_source=8, **config):
    return RewriteExampleSource(
        ListRecords(records),
        Corruptor(CorruptionConfig(**config)),
        fake_encode,
        tokens,
        max_source_length=max_source,
        max_target_length=max_target,
        min_source_length=min_source,
        seed=seed,
    )


# -- the frame ---------------------------------------------------------------


def test_the_pair_is_framed_as_the_product_frames_it(tokens):
    source = build([PROSE], tokens, identity_fraction=1.0)
    pair = source.next_pair()

    assert pair.encoder_input_ids[:2] == (tokens.style_id, tokens.open_id)
    assert pair.encoder_input_ids[-2:] == (tokens.close_id, tokens.eos_id)
    assert fake_decode(pair.encoder_input_ids[2:-2]) == PROSE

    assert pair.decoder_target_ids[-1] == tokens.eos_id
    assert fake_decode(pair.decoder_target_ids[:-1]) == PROSE
    assert pair.source_length == len(PROSE)
    assert pair.profile == IDENTITY and pair.edits == 0


def test_a_corrupted_pair_keeps_the_original_as_its_target(tokens):
    source = build([PROSE], tokens, identity_fraction=0.0)
    for _ in range(20):
        pair = source.next_pair()
        assert fake_decode(pair.decoder_target_ids[:-1]) == PROSE
        if pair.profile != IDENTITY:
            assert fake_decode(pair.encoder_input_ids[2:-2]) != PROSE
            assert pair.edits > 0
            return
    pytest.fail("twenty draws at identity_fraction=0 never corrupted the document")


def test_the_mode_label_is_not_a_ul2_mode():
    assert REWRITE_MODE not in {"R", "X", "S"}


# -- admission ---------------------------------------------------------------


def test_documents_without_prose_are_skipped_and_counted(tokens):
    source = build([CODE, PROSE], tokens, identity_fraction=1.0)
    pair = source.next_pair()
    assert fake_decode(pair.decoder_target_ids[:-1]) == PROSE
    assert source.skipped_not_prose == 1
    assert source.documents_read == 2


def test_an_original_that_does_not_fit_the_decoder_is_skipped(tokens):
    source = build([PROSE, "Short prose line here, with enough words to count as prose."],
                   tokens, max_target=len(PROSE), identity_fraction=1.0, min_prose_words=4)
    pair = source.next_pair()
    assert len(pair.decoder_target_ids) <= len(PROSE)
    assert source.skipped_too_long == 1


def test_a_corrupted_form_that_does_not_fit_the_encoder_is_skipped(tokens):
    """
    Corruption can lengthen a document (doubled words, typos). The encoder
    budget is checked on the corrupted form, so a document that passed the
    decoder check can still be refused -- counted, never truncated.
    """
    # Budget exactly the original plus the frame: any lengthening overflows.
    source = build([PROSE], tokens, max_source=len(PROSE) + 4, identity_fraction=0.0)
    for _ in range(30):
        pair = source.next_pair()
        assert len(pair.encoder_input_ids) <= len(PROSE) + 4
    assert source.skipped_too_long > 0


def test_too_short_documents_are_skipped(tokens):
    source = build(["Tiny.", PROSE], tokens, min_source=20, identity_fraction=1.0)
    source.next_pair()
    assert source.skipped_too_short == 1


@pytest.mark.parametrize(
    "budgets",
    [dict(max_source=10, min_source=8), dict(max_target=8, min_source=8), dict(min_source=0)],
)
def test_a_budget_that_cannot_hold_a_document_is_refused(tokens, budgets):
    with pytest.raises(RewriteConfigError):
        build([PROSE], tokens, **budgets)


# -- resumption --------------------------------------------------------------


def test_a_resumed_source_continues_an_uninterrupted_one_exactly(tokens):
    records = [PROSE, PROSE.replace("European", "Asian"), CODE, PROSE.replace("team", "board")]

    uninterrupted = build(records, tokens, seed=7)
    expected = [uninterrupted.next_pair() for _ in range(40)]

    first = build(records, tokens, seed=7)
    got = [first.next_pair() for _ in range(17)]
    saved = json.loads(json.dumps(first.state_dict()))

    second = build(records, tokens, seed=7)
    second.load_state_dict(saved)
    got += [second.next_pair() for _ in range(23)]

    assert got == expected
    assert second.report() == uninterrupted.report()


def test_the_report_names_what_happened(tokens):
    source = build([CODE, PROSE], tokens, identity_fraction=0.0)
    for _ in range(10):
        source.next_pair()
    report = source.report()
    assert report["pairs_emitted"] == 10
    assert report["skipped_not_prose"] == 10  # the stream alternates CODE, PROSE
    assert sum(entry["pairs"] for entry in report["profiles"].values()) == 10
    assert "rewrite task: 10 pairs" in source.format_table()
