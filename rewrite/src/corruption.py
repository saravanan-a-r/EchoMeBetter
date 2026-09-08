"""
Turning clean text into the kind of text people actually hand a rewriter.

The task this builds (architecture_improvements.md item 4) is:

    <style:grammar> <text_to_rewrite> corrupted </text_to_rewrite>  ->  original

so the corruption has to satisfy one property above all others: **the
original must be the unique right answer.** Every operator here produces an
error whose correction is the source text -- a dropped article, a swapped
homophone, a finger-slip typo, a comma splice, a missing capital. None of
them produces a *different valid sentence*, which is why there is no synonym
swapping: "big" -> "large" is not a mistake, and a target that reverses it
teaches the model to edit text that was never wrong, the exact opposite of
the fidelity the product is built around.

What makes it hard rather than trivial
--------------------------------------
A single fixed corruption is easy to invert with a lookup. Three things keep
the inverse from being learnable as a table:

  1. **Severity is sampled per document**, from three profiles. A light
     document has a handful of errors; a heavy one is lowercased, stripped of
     punctuation and misspelled throughout. The model cannot assume how much
     is wrong.
  2. **The operator mix is sampled per word**, weighted, from everything that
     applies to that word. The same word is a homophone swap in one document,
     a typo in the next, dropped in a third.
  3. **A slice of documents is left untouched** (`identity_fraction`). "This
     text is already correct, copy it" is a case the product hits constantly,
     and a model that never trained on it learns that the input is always
     wrong somewhere.

What is never touched
---------------------
Lines that are not prose, and protected spans inside prose lines -- URLs,
paths, numbers, code, identifiers (`text.py`). They come through verbatim, so
every example also trains the copy-through the product depends on.

Determinism
-----------
`corrupt(text, rng)` is a pure function of `(text, rng state, config)`. The
RNG is consumed in a fixed order -- document draws, then line by line, piece
by piece -- and never through `random.choices`, whose internal draw count is
not guaranteed across Python versions. Resuming a run replays the same
corruptions (architecture.md 7.9).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .errors import RewriteConfigError
from .lexicon import CONFUSABLES, FUNCTION_WORDS, KEYBOARD_NEIGHBORS, PREPOSITIONS
from .text import Piece, is_prose_line, split_line

# The per-word operators and their relative weights. Grammar-shaped errors
# outweigh raw typos, because grammar is the style token this task trains.
DEFAULT_OPERATOR_WEIGHTS: Mapping[str, float] = {
    "confusable": 4.0,        # their/there, is/are, went/gone
    "drop_function_word": 3.0,  # "cat sat on mat"
    "typo": 3.0,              # finger slips: transpose, drop, double, neighbour
    "case": 2.0,              # "paris", "i think"
    "preposition": 1.5,       # in/on/at
    "drop_apostrophe": 1.5,   # dont, Johns
    "double_word": 1.0,       # "the the"
    "swap_adjacent": 1.0,     # "sat the cat"
}

OPERATORS: tuple[str, ...] = tuple(DEFAULT_OPERATOR_WEIGHTS)

_SENTENCE_END = frozenset(".!?")
_DROPPABLE_PUNCT = frozenset(",;:\"'“”‘’")
_STRIPPABLE_PUNCT = _DROPPABLE_PUNCT | _SENTENCE_END


@dataclass(frozen=True)
class Profile:
    """
    One severity level.

    `min_rate`/`max_rate` bound the per-word probability of an operator,
    drawn once per document. The three `*_probability` fields are document-
    level habits: a writer who types in lowercase does it throughout.
    """

    name: str
    weight: float
    min_rate: float
    max_rate: float
    lowercase_probability: float
    strip_punctuation_probability: float
    run_on_probability: float

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise RewriteConfigError(f"profile {self.name!r}: weight must be positive")
        if not 0.0 <= self.min_rate <= self.max_rate <= 1.0:
            raise RewriteConfigError(
                f"profile {self.name!r}: need 0 <= min_rate <= max_rate <= 1, got "
                f"{self.min_rate}..{self.max_rate}"
            )
        for label, value in (
            ("lowercase_probability", self.lowercase_probability),
            ("strip_punctuation_probability", self.strip_punctuation_probability),
            ("run_on_probability", self.run_on_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise RewriteConfigError(f"profile {self.name!r}: {label} must be in [0, 1]")


DEFAULT_PROFILES: tuple[Profile, ...] = (
    Profile("light", 0.35, 0.03, 0.08, 0.05, 0.00, 0.10),
    Profile("medium", 0.40, 0.08, 0.18, 0.15, 0.10, 0.25),
    Profile("heavy", 0.25, 0.18, 0.35, 0.40, 0.35, 0.40),
)

IDENTITY = "identity"


@dataclass(frozen=True)
class CorruptionConfig:
    """Everything that shapes the noise. Frozen, so a run manifest can record it."""

    identity_fraction: float = 0.05
    profiles: tuple[Profile, ...] = DEFAULT_PROFILES
    operator_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OPERATOR_WEIGHTS)
    )
    # A document with fewer corruptible words than this is skipped: there is
    # nothing to rewrite, and a target that copies four words teaches nothing.
    min_prose_words: int = 8
    # A line with fewer words than this is never prose (headings, list bullets).
    min_words_per_line: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.identity_fraction <= 1.0:
            raise RewriteConfigError(
                f"identity_fraction must be in [0, 1], got {self.identity_fraction}"
            )
        if not self.profiles:
            raise RewriteConfigError("at least one profile is required")
        names = [profile.name for profile in self.profiles]
        if len(set(names)) != len(names):
            raise RewriteConfigError(f"duplicate profile names: {names}")
        if IDENTITY in names:
            raise RewriteConfigError(f"{IDENTITY!r} is reserved for uncorrupted documents")
        unknown = set(self.operator_weights) - set(OPERATORS)
        if unknown:
            raise RewriteConfigError(
                f"unknown operators {sorted(unknown)}; implemented: {list(OPERATORS)}"
            )
        if not self.operator_weights or all(w <= 0.0 for w in self.operator_weights.values()):
            raise RewriteConfigError("every operator weight is zero; nothing could be applied")
        if any(w < 0.0 for w in self.operator_weights.values()):
            raise RewriteConfigError("operator weights must be non-negative")
        if self.min_prose_words < 1 or self.min_words_per_line < 1:
            raise RewriteConfigError("min_prose_words and min_words_per_line must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_fraction": self.identity_fraction,
            "profiles": [vars(profile) for profile in self.profiles],
            "operator_weights": dict(self.operator_weights),
            "min_prose_words": self.min_prose_words,
            "min_words_per_line": self.min_words_per_line,
        }


@dataclass(frozen=True)
class Corruption:
    """The corrupted text and what was done to it."""

    text: str
    profile: str  # a profile name, or IDENTITY
    edits: int  # pieces whose text changed; 0 for an identity document
    prose_words: int  # words the operators could have acted on


class Corruptor:
    """Applies `CorruptionConfig` to documents. Stateless; the RNG is passed in."""

    def __init__(self, config: CorruptionConfig | None = None) -> None:
        self.config = config or CorruptionConfig()
        self._operators = [
            (name, weight)
            for name, weight in self.config.operator_weights.items()
            if weight > 0.0
        ]
        self._profile_names = [profile.name for profile in self.config.profiles]
        self._profile_weights = [profile.weight for profile in self.config.profiles]

    # -- entry point --------------------------------------------------------

    def corrupt(self, text: str, rng: random.Random) -> Corruption | None:
        """
        Corrupt one document. `None` when it holds too little prose to be a
        rewrite example at all (all code, a table, a list of headings).
        """
        lines = text.split("\n")
        prose = [is_prose_line(line, min_words=self.config.min_words_per_line) for line in lines]
        split = [split_line(line) if keep else None for line, keep in zip(lines, prose)]
        prose_words = sum(
            1 for pieces in split if pieces for piece in pieces if piece.kind == "word"
        )
        if prose_words < self.config.min_prose_words:
            return None

        # Document-level draws, in a fixed order.
        if rng.random() < self.config.identity_fraction:
            return Corruption(text, IDENTITY, 0, prose_words)

        profile = self.config.profiles[_weighted_index(rng, self._profile_weights)]
        rate = profile.min_rate + rng.random() * (profile.max_rate - profile.min_rate)
        lowercase = rng.random() < profile.lowercase_probability
        strip_punctuation = rng.random() < profile.strip_punctuation_probability

        out_lines: list[str] = []
        edits = 0
        for line, pieces in zip(lines, split):
            if pieces is None:
                out_lines.append(line)
                continue
            self._corrupt_line(pieces, rng, rate, profile.run_on_probability)
            if lowercase:
                _lowercase(pieces)
            if strip_punctuation:
                _strip_punctuation(pieces)
            edits += sum(1 for piece in pieces if piece.edited)
            out_lines.append("".join(piece.out for piece in pieces))

        corrupted = "\n".join(out_lines)
        if corrupted == text:
            # Every draw missed. Still a valid example, and honestly labelled.
            return Corruption(text, IDENTITY, 0, prose_words)
        return Corruption(corrupted, profile.name, edits, prose_words)

    # -- one line ----------------------------------------------------------

    def _corrupt_line(
        self, pieces: list[Piece], rng: random.Random, rate: float, run_on: float
    ) -> None:
        for index, piece in enumerate(pieces):
            if piece.edited:
                continue  # claimed by an earlier operator (swap, run-on)
            if piece.kind == "word":
                if rng.random() < rate:
                    self._apply_word_operator(pieces, index, rng)
            elif piece.kind == "punct":
                if piece.text in _SENTENCE_END and _starts_sentence(pieces, index):
                    if rng.random() < run_on:
                        _run_on(pieces, index, rng)
                elif piece.text in _SENTENCE_END and index == len(pieces) - 1:
                    if rng.random() < rate / 2:
                        piece.out = ""  # the forgotten final stop
                elif piece.text in _DROPPABLE_PUNCT and rng.random() < rate:
                    piece.out = ""
            elif piece.kind == "space":
                # Only after punctuation that is still there: "word,word" is a
                # real slip, while "wordword" -- the space *and* the comma gone
                # -- has no unique fix.
                previous = pieces[index - 1] if index > 0 else None
                if (
                    previous is not None
                    and previous.kind == "punct"
                    and previous.text in _STRIPPABLE_PUNCT
                    and not previous.edited
                    and rng.random() < rate / 2
                ):
                    piece.out = ""

    def _apply_word_operator(self, pieces: list[Piece], index: int, rng: random.Random) -> None:
        applicable = [
            (name, weight)
            for name, weight in self._operators
            if _applies(name, pieces, index)
        ]
        if not applicable:
            return
        name = applicable[_weighted_index(rng, [weight for _, weight in applicable])][0]
        _OPERATOR_TABLE[name](pieces, index, rng)


# -- applicability ------------------------------------------------------------


def _applies(name: str, pieces: Sequence[Piece], index: int) -> bool:
    word = pieces[index].text
    lowered = word.lower()
    if name == "confusable":
        return lowered in CONFUSABLES
    if name == "drop_function_word":
        return lowered in FUNCTION_WORDS
    if name == "typo":
        return len(word) >= 4
    if name == "case":
        return word[0].isupper() and not word.isupper()
    if name == "preposition":
        return lowered in PREPOSITIONS
    if name == "drop_apostrophe":
        return "'" in word or "’" in word
    if name == "double_word":
        return True
    if name == "swap_adjacent":
        partner = _adjacent_word(pieces, index)
        return partner is not None and not pieces[partner].edited
    raise RewriteConfigError(f"unknown operator {name!r}")


def _adjacent_word(pieces: Sequence[Piece], index: int) -> int | None:
    """The index of the next word if only a single space separates the two."""
    if index + 2 < len(pieces) and pieces[index + 1].text == " ":
        following = pieces[index + 2]
        if following.kind == "word":
            return index + 2
    return None


def _starts_sentence(pieces: Sequence[Piece], index: int) -> bool:
    """Whether sentence-final punctuation at `index` is followed by a capitalised word."""
    partner = _adjacent_word(pieces, index)
    return partner is not None and pieces[partner].text[0].isupper()


# -- word operators ------------------------------------------------------------


def _confusable(pieces: list[Piece], index: int, rng: random.Random) -> None:
    piece = pieces[index]
    alternatives = CONFUSABLES[piece.text.lower()]
    piece.out = _match_case(piece.text, alternatives[rng.randrange(len(alternatives))])


def _preposition(pieces: list[Piece], index: int, rng: random.Random) -> None:
    piece = pieces[index]
    alternatives = PREPOSITIONS[piece.text.lower()]
    piece.out = _match_case(piece.text, alternatives[rng.randrange(len(alternatives))])


def _drop_function_word(pieces: list[Piece], index: int, rng: random.Random) -> None:
    pieces[index].out = ""
    # Take one neighbouring space with it, so "on the mat" becomes "on mat"
    # rather than "on  mat".
    if index + 1 < len(pieces) and pieces[index + 1].kind == "space":
        pieces[index + 1].out = ""
    elif index > 0 and pieces[index - 1].kind == "space":
        pieces[index - 1].out = ""


def _typo(pieces: list[Piece], index: int, rng: random.Random) -> None:
    """A finger slip inside the word. The first letter is kept: real typos rarely hit it."""
    word = pieces[index].text
    position = rng.randrange(1, len(word))
    kind = rng.randrange(4)
    if kind == 0 and position < len(word) - 1:  # transpose with the next letter
        out = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
    elif kind == 1:  # drop
        out = word[:position] + word[position + 1 :]
    elif kind == 2:  # double
        out = word[:position] + word[position] + word[position:]
    else:  # neighbouring key, falling back to a transposition off the keyboard
        neighbours = KEYBOARD_NEIGHBORS.get(word[position].lower())
        if neighbours:
            replacement = neighbours[rng.randrange(len(neighbours))]
            if word[position].isupper():
                replacement = replacement.upper()
            out = word[:position] + replacement + word[position + 1 :]
        elif position < len(word) - 1:
            out = word[:position] + word[position + 1] + word[position] + word[position + 2 :]
        else:
            out = word[:position] + word[position - 1] + word[position:]
    pieces[index].out = out if out != word else word[:position] + word[position] + word[position:]


def _case(pieces: list[Piece], index: int, rng: random.Random) -> None:
    pieces[index].out = pieces[index].text.lower()


def _drop_apostrophe(pieces: list[Piece], index: int, rng: random.Random) -> None:
    pieces[index].out = pieces[index].text.replace("'", "").replace("’", "")


def _double_word(pieces: list[Piece], index: int, rng: random.Random) -> None:
    word = pieces[index].text
    pieces[index].out = f"{word} {word}"


def _swap_adjacent(pieces: list[Piece], index: int, rng: random.Random) -> None:
    partner = _adjacent_word(pieces, index)
    assert partner is not None  # guaranteed by `_applies`
    pieces[index].out, pieces[partner].out = pieces[partner].text, pieces[index].text


_OPERATOR_TABLE = {
    "confusable": _confusable,
    "drop_function_word": _drop_function_word,
    "typo": _typo,
    "case": _case,
    "preposition": _preposition,
    "drop_apostrophe": _drop_apostrophe,
    "double_word": _double_word,
    "swap_adjacent": _swap_adjacent,
}
assert set(_OPERATOR_TABLE) == set(OPERATORS)


# -- sentence and line operators ----------------------------------------------


def _run_on(pieces: list[Piece], index: int, rng: random.Random) -> None:
    """
    Merge two sentences the way hurried writing does.

    Three real shapes, drawn uniformly: a comma splice ("home, it was"), a
    dropped stop ("home it was"), and a missing capital ("home. it was").
    """
    partner = _adjacent_word(pieces, index)
    assert partner is not None
    shape = rng.randrange(3)
    if shape == 0:
        pieces[index].out = ","
    elif shape == 1:
        pieces[index].out = ""
    word = pieces[partner].text
    pieces[partner].out = word[0].lower() + word[1:]


def _lowercase(pieces: list[Piece]) -> None:
    for piece in pieces:
        if piece.kind == "word":
            piece.out = piece.out.lower()


def _strip_punctuation(pieces: list[Piece]) -> None:
    for index, piece in enumerate(pieces):
        if piece.kind == "punct" and piece.text in _STRIPPABLE_PUNCT:
            piece.out = ""
            # A space the line pass removed after this mark must come back,
            # or the two words either side of it merge into one.
            if index + 1 < len(pieces) and pieces[index + 1].kind == "space":
                pieces[index + 1].out = pieces[index + 1].text


# -- helpers -------------------------------------------------------------------


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _weighted_index(rng: random.Random, weights: Sequence[float]) -> int:
    """One `rng.random()` against a cumulative table; never `random.choices`."""
    total = float(sum(weights))
    point = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if point < running:
            return index
    return len(weights) - 1


__all__ = [
    "Corruption",
    "CorruptionConfig",
    "Corruptor",
    "DEFAULT_OPERATOR_WEIGHTS",
    "DEFAULT_PROFILES",
    "IDENTITY",
    "OPERATORS",
    "Profile",
]
