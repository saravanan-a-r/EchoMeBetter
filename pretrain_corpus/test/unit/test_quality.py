from __future__ import annotations

from src.quality import KEEP, QualityFilter, assess, is_english, latin_ratio, stopword_ratio

GOOD_PROSE = (
    "The committee reviewed the quarterly roadmap and agreed that the next "
    "release should prioritize reliability over new features, since several "
    "customers had reported intermittent failures in the billing pipeline."
)


def test_normal_prose_is_kept():
    assert assess(GOOD_PROSE) == KEEP


def test_too_short_is_dropped():
    assert assess("Hello there.") == "too_short"


def test_symbol_heavy_junk_is_dropped():
    text = "### *** --- ### *** --- ### *** --- ### *** --- ### *** --- extra"
    assert assess(text) != KEEP


def test_repeated_line_spam_is_dropped():
    text = "\n".join(["Click here to subscribe now to our newsletter today"] * 8)
    assert assess(text) == "repeated_lines"


def test_mostly_digits_is_dropped():
    text = " ".join(str(n) for n in range(1000, 1040))
    # Pure digit-and-space text also fails the (cheaper, earlier) alpha-ratio
    # check, so the specific reason string isn't guaranteed -- what matters
    # is that it's rejected at all.
    assert assess(text) != KEEP


def test_english_prose_is_english():
    assert is_english(GOOD_PROSE)


def test_french_prose_is_not_english():
    french = (
        "Le comité a examiné la feuille de route trimestrielle et a convenu "
        "que la prochaine version devrait donner la priorité à la fiabilité "
        "pour les clients les plus importants de l'entreprise."
    )
    assert not is_english(french)


def test_latin_ratio_of_pure_ascii_text_is_one():
    assert latin_ratio("hello world") == 1.0


def test_latin_ratio_of_non_latin_script_is_zero():
    assert latin_ratio("こんにちは世界") == 0.0


def test_stopword_ratio_ignores_single_character_words():
    """Regression test: single-letter words like English 'a' collide with
    common short words in other Latin-script languages (French 'a', Spanish
    'y'), and previously inflated the ratio enough to misclassify short
    non-English text as English."""
    assert stopword_ratio("a a a a a") == 0.0


def test_quality_filter_tallies_rejections_by_reason():
    qf = QualityFilter()
    qf.accept(GOOD_PROSE)
    qf.accept("Hello there.")
    qf.accept("Hi.")
    assert qf.kept == 1
    assert qf.rejected.get("too_short") == 2


def test_quality_filter_disabled_accepts_everything():
    qf = QualityFilter(enabled=False)
    assert qf.accept("Hi.")
    assert qf.accept("### *** ###")
    assert qf.kept == 2
    assert qf.rejected == {}
