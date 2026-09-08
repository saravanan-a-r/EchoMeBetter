"""
The corruption engine's guarantees.

Ordered by how badly a regression would hurt:

  1. Protected content and non-prose lines come through verbatim. A break
     here trains the model to edit URLs, numbers and code.
  2. Identity documents are exactly identical. A break here teaches the
     model that correct input is always wrong somewhere.
  3. Determinism. A break here makes a resumed run corrupt differently from
     an uninterrupted one.
  4. The noise is real: documents actually change, at every profile, and no
     single operator is silently dead.
"""

from __future__ import annotations

import random

import pytest

from src.corruption import (
    DEFAULT_OPERATOR_WEIGHTS,
    IDENTITY,
    OPERATORS,
    CorruptionConfig,
    Corruptor,
    Profile,
)
from src.errors import RewriteConfigError
from src.text import protected_spans

DOCUMENT = "\n".join(
    [
        "The quarterly report showed strong growth in the European market, and the "
        "team expects it to continue.",
        "Check https://api.example.com/v2/users for the current status; it returned "
        "503 about 40% of the time.",
        "Please reach out to first.last@example.com about the outstanding invoice "
        "before Friday.",
        "def parse(line):",
        "    return json.loads(line)",
        "Yesterday the team voted to approve the budget after a long deliberation, "
        "which surprised nobody who had been in the room.",
        "NASA launched the probe in 2024, and the getValue() call in `config.py` "
        "still fails on macOS.",
        "I think we're going to need a bigger boat, but their captain doesn't agree "
        "with us. That was their answer. Nobody argued.",
    ]
)

CODE_LINES = ("def parse(line):", "    return json.loads(line)")


def corrupt(text=DOCUMENT, seed=0, **config):
    return Corruptor(CorruptionConfig(**config)).corrupt(text, random.Random(seed))


def assert_in_order(needles, haystack):
    position = 0
    for needle in needles:
        found = haystack.find(needle, position)
        assert found >= 0, f"{needle!r} missing or out of order in {haystack!r}"
        position = found + len(needle)


# -- 1. what is never touched -----------------------------------------------


@pytest.mark.parametrize("seed", range(300))
def test_protected_spans_survive_verbatim_and_in_order(seed):
    result = corrupt(seed=seed, identity_fraction=0.0)
    assert_in_order(protected_spans(DOCUMENT), result.text)


@pytest.mark.parametrize("seed", range(300))
def test_non_prose_lines_and_line_structure_survive(seed):
    result = corrupt(seed=seed, identity_fraction=0.0)
    lines = result.text.split("\n")
    assert len(lines) == DOCUMENT.count("\n") + 1
    for code in CODE_LINES:
        assert code in lines


def test_a_document_with_no_prose_is_skipped():
    assert corrupt("def f():\n    return 1\nx = 2") is None


def test_a_document_with_too_little_prose_is_skipped():
    assert corrupt("The cat sat on the mat.", min_prose_words=8) is None
    assert corrupt("The cat sat on the mat.", min_prose_words=4) is not None


# -- 2. identity ------------------------------------------------------------


def test_identity_fraction_one_copies_exactly():
    for seed in range(20):
        result = corrupt(seed=seed, identity_fraction=1.0)
        assert result.text == DOCUMENT
        assert result.profile == IDENTITY
        assert result.edits == 0


def test_an_uncorrupted_draw_is_labelled_identity_not_a_profile():
    """
    Every operator can miss on a short document at a light rate. The label
    must say what happened to the text, not what was intended for it.
    """
    light_only = (Profile("light", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),)
    result = corrupt("The cat sat on the mat and the dog watched it happen quietly.",
                     identity_fraction=0.0, profiles=light_only, min_prose_words=4)
    assert result.text.endswith("quietly.")
    assert result.profile == IDENTITY
    assert result.edits == 0


# -- 3. determinism ---------------------------------------------------------


def test_the_same_seed_reproduces_the_same_corruption():
    first = [corrupt(seed=seed) for seed in range(50)]
    second = [corrupt(seed=seed) for seed in range(50)]
    assert [r.text for r in first] == [r.text for r in second]


# -- 4. the noise is real ---------------------------------------------------


def test_documents_change_at_every_profile():
    seen: dict[str, int] = {}
    changed = 0
    for seed in range(200):
        result = corrupt(seed=seed, identity_fraction=0.0)
        seen[result.profile] = seen.get(result.profile, 0) + 1
        changed += result.text != DOCUMENT
    assert changed >= 190
    assert set(seen) >= {"light", "medium", "heavy"}


def test_heavier_profiles_edit_more():
    def mean_edits(name):
        profile = next(p for p in Corruptor().config.profiles if p.name == name)
        edits = [
            corrupt(seed=seed, identity_fraction=0.0, profiles=(profile,)).edits
            for seed in range(100)
        ]
        return sum(edits) / len(edits)

    assert mean_edits("light") < mean_edits("medium") < mean_edits("heavy")


@pytest.mark.parametrize("operator", OPERATORS)
def test_every_operator_changes_something_on_its_own(operator):
    """No operator is silently dead: alone, at full rate, each one edits."""
    weights = {name: (1.0 if name == operator else 0.0) for name in DEFAULT_OPERATOR_WEIGHTS}
    profile = (Profile("all", 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),)
    changed = any(
        corrupt(seed=seed, identity_fraction=0.0, operator_weights=weights, profiles=profile).text
        != DOCUMENT
        for seed in range(10)
    )
    assert changed


def test_confusable_swaps_keep_the_capital():
    weights = {name: (1.0 if name == "confusable" else 0.0) for name in DEFAULT_OPERATOR_WEIGHTS}
    profile = (Profile("all", 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),)
    text = "Their house is over there and they're happy about it all."
    result = corrupt(text, identity_fraction=0.0, operator_weights=weights, profiles=profile,
                     min_prose_words=4)
    first = result.text.split()[0]
    assert first[0].isupper()
    assert first.lower() in {"there", "they're", "their"}


def test_lowercasing_and_stripping_never_touch_protected_spans():
    shouting = (Profile("shout", 1.0, 0.0, 0.0, 1.0, 1.0, 0.0),)
    for seed in range(20):
        result = corrupt(seed=seed, identity_fraction=0.0, profiles=shouting)
        assert_in_order(protected_spans(DOCUMENT), result.text)
        assert "NASA" in result.text
        assert "getValue" in result.text
        assert "nasa" not in result.text


def test_removing_punctuation_never_merges_two_words():
    """
    "end. Next" may become "end Next" or "end.Next", never "endNext": the
    latter has no unique correction.
    """
    strip = (Profile("strip", 1.0, 0.35, 0.35, 0.0, 1.0, 0.0),)
    text = "The cat sat, and it stayed. Then the dog came in, and it left again."
    for seed in range(200):
        result = corrupt(text, seed=seed, identity_fraction=0.0, profiles=strip,
                         min_prose_words=4)
        assert "stayedThen" not in result.text
        assert "satand" not in result.text
        assert "inand" not in result.text


# -- configuration refusals -------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"identity_fraction": 1.5},
        {"profiles": ()},
        {"profiles": (Profile("a", 1.0, 0.1, 0.2, 0, 0, 0), Profile("a", 1.0, 0.1, 0.2, 0, 0, 0))},
        {"profiles": (Profile(IDENTITY, 1.0, 0.1, 0.2, 0, 0, 0),)},
        {"operator_weights": {"synonym": 1.0}},
        {"operator_weights": {name: 0.0 for name in OPERATORS}},
        {"operator_weights": {"typo": -1.0}},
        {"min_prose_words": 0},
    ],
)
def test_a_configuration_that_cannot_run_is_refused(bad):
    with pytest.raises(RewriteConfigError):
        CorruptionConfig(**bad)


def test_a_profile_with_inverted_rates_is_refused():
    with pytest.raises(RewriteConfigError):
        Profile("x", 1.0, 0.5, 0.1, 0.0, 0.0, 0.0)
