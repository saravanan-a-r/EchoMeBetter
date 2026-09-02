"""
Deciding whether a cleaned chunk is worth training on.

`clean.py` removes scaffolding from a record. This module answers the
different question of whether what is left is *plain English prose* at all —
and it is the layer the tokenizer downloader never had, because fitting a
vocabulary on some junk is survivable and training a model on it is not.

Two filters, in the order they run:

**Language.** The scope for this corpus is English; a little other-language
text leaking through is fine, because byte-fallback makes it representable
rather than corrupting. Most sources are already English-filtered upstream
(FineWeb-Edu, C4's `en` config, Cosmopedia, PG-19), so for those this is a
cheap backstop. Stack Exchange is the source that actually needs it: the
archive.org dump ships all 362 sites, including `ru.stackoverflow.com`,
`es.stackoverflow.com`, `japanese.stackexchange.com` and more. Those are
excluded by name in `sources.py` — a site-level decision is exact and free —
and this per-document check catches the rest, such as a Russian answer posted
on an English site.

**Quality.** Gopher/C4-style heuristics: cheap, well-understood, and aimed at
the specific junk that a byte-budget crawl sweeps up — navigation word-soup,
keyword-stuffed SEO pages, walls of numbers, keyboard mash. None of them
looks at meaning; they look at shape.

Both are pure functions over one string, so they parallelize trivially and
are easy to test against known-good and known-bad text.

On not using fastText
---------------------
`lid.176` would be more accurate, and is supported if installed — but the
default path deliberately has no model file to fetch, no wheel to build, and
no failure mode where a corpus download dies at 3am on a training box because
a 126MB model could not be downloaded. The stopword-ratio test below agrees
with fastText on the overwhelming majority of documents in this corpus's
register, and disagreeing on a handful of borderline ones costs nothing:
this is a filter for a 50B-token budget, not a classifier being graded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The 120 commonest English function words. Function words are what make this
# work as a language test: they are near-impossible to avoid in real English
# prose and near-absent from every other language and from non-prose junk.
ENGLISH_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being
    but by can cannot could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it
    its just me more most my no nor not now of off on once only or other our out
    over own same she should so some such than that the their them then there
    these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z']+")
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class QualityThresholds:
    """
    Every cut-off in one place, so tuning is a config change and not a hunt.

    Defaults are the conventional Gopher/RefinedWeb values, loosened slightly
    where this corpus differs: `min_words` is low because Stack Exchange
    comments are legitimately short, and `max_symbol_ratio` is generous
    because technical prose carries a lot of punctuation.
    """

    min_words: int = 15
    max_words: int = 100_000
    min_mean_word_length: float = 2.5
    max_mean_word_length: float = 12.0
    min_alpha_ratio: float = 0.60
    max_digit_ratio: float = 0.30
    max_duplicate_line_ratio: float = 0.30
    min_stopword_ratio: float = 0.06
    min_latin_ratio: float = 0.70


DEFAULT_THRESHOLDS = QualityThresholds()

# Reasons are returned as strings rather than booleans so a run can report
# *why* it dropped 8% of a source — the difference between "the filter is
# working" and "the filter is misconfigured for this source".
KEEP = ""


def latin_ratio(text: str) -> float:
    """Share of the *letters* in `text` that are ASCII Latin."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isascii()) / len(letters)


def stopword_ratio(text: str) -> float:
    """
    Share of words that are English function words.

    Single-character words (`a`, `i`) are excluded from both the count and
    the denominator. They're real English stopwords, but they are also
    extremely common short words in many other Latin-script languages
    (French `a`/`y`, Spanish `y`/`a`/`o`, Italian `a`/`e`/`o`) -- a short
    French sentence like "Le comite a examine..." scores two stopword hits
    on "a" alone from a ~24-word document, which was enough on its own to
    clear the default threshold and get classified as English. Every
    multi-letter English function word (the, and, of, to, in...) is far
    less ambiguous across languages, so dropping single-character tokens
    removes the main false-positive source without weakening real English
    detection.
    """
    words = [w.lower() for w in _WORD_RE.findall(text) if len(w) >= 2]
    if not words:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words)


def is_english(text: str, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> bool:
    """
    Two agreeing signals: the script is Latin, and the function words are English.

    Either alone is weak — Latin script covers French and Indonesian, and a
    stopword test alone is noisy on very short text — but a document has to
    fail neither to be kept, which is accurate enough at this scale.
    """
    return (
        latin_ratio(text) >= thresholds.min_latin_ratio
        and stopword_ratio(text) >= thresholds.min_stopword_ratio
    )


def assess(text: str, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> str:
    """
    `KEEP` (the empty string) if the chunk is usable, otherwise the reason.

    Ordered cheapest-check-first so the common rejections cost the least.
    """
    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < thresholds.min_words:
        return "too_short"
    if len(tokens) > thresholds.max_words:
        return "too_long"

    mean_length = sum(len(t) for t in tokens) / len(tokens)
    if mean_length < thresholds.min_mean_word_length:
        return "mean_word_length_low"
    if mean_length > thresholds.max_mean_word_length:
        return "mean_word_length_high"

    alpha = sum(1 for c in text if c.isalpha() or c.isspace())
    if alpha / len(text) < thresholds.min_alpha_ratio:
        return "symbol_soup"

    digits = sum(1 for c in text if c.isdigit())
    if digits / len(text) > thresholds.max_digit_ratio:
        return "mostly_digits"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 4:
        unique = len(set(lines))
        if 1 - (unique / len(lines)) > thresholds.max_duplicate_line_ratio:
            return "repeated_lines"

    if latin_ratio(text) < thresholds.min_latin_ratio:
        return "not_latin_script"
    if stopword_ratio(text) < thresholds.min_stopword_ratio:
        return "not_english_prose"

    return KEEP


class QualityFilter:
    """
    Stateful wrapper that tallies rejections by reason.

    One per worker; merged into the run manifest at the end so a source that
    is being filtered to death shows up as a number rather than as a vague
    sense that the download is smaller than expected.
    """

    def __init__(
        self,
        thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
        *,
        enabled: bool = True,
    ) -> None:
        self.thresholds = thresholds
        self.enabled = enabled
        self.kept = 0
        self.rejected: dict[str, int] = {}

    def accept(self, text: str) -> bool:
        # Disabled for the code source: source files are not English prose and
        # every heuristic here would reject them for being exactly what they
        # are meant to be.
        if not self.enabled:
            self.kept += 1
            return True
        reason = assess(text, self.thresholds)
        if reason:
            self.rejected[reason] = self.rejected.get(reason, 0) + 1
            return False
        self.kept += 1
        return True

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "kept": self.kept,
            "rejected_total": sum(self.rejected.values()),
            "rejected_by_reason": dict(sorted(self.rejected.items())),
        }
