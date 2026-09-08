"""
The word tables the corruption operators draw on.

Every entry here encodes an error a real writer makes, whose correction is
the *original* word rather than a matter of taste. That constraint is the
whole design: this task trains a fidelity-first rewriter, and a corruption
whose "fix" could legitimately be several different words would teach the
model to change text that was never wrong. Which is why there is no synonym
table -- "big" -> "large" is not an error, and a target that undoes it is a
target that rewards editing for its own sake.

Three kinds of confusion, all with one right answer in context:

  - **Homophones and near-spellings**: their/there/they're, affect/effect.
    The classic grammar-checker cases.
  - **Agreement and form**: is/are, has/have, went/gone. Subject-verb
    agreement and irregular past/participle forms ("I have went").
  - **Determiners and number**: a/an, this/these, fewer/less.

Plus a prepositions table (in/on/at, to/for), the function words that are
most often *dropped* by hurried or non-native writers, and QWERTY neighbours
for realistic finger-slip typos.
"""

from __future__ import annotations

# Words that swap for one another: a word from a group is replaced by a
# *different* member of the same group. A word may sit in several groups (see
# `_build_alternatives`).
CONFUSABLE_GROUPS: tuple[tuple[str, ...], ...] = (
    # homophones / near-spellings
    ("their", "there", "they're"),
    ("its", "it's"),
    ("your", "you're"),
    ("then", "than"),
    ("to", "too", "two"),
    ("affect", "effect"),
    ("accept", "except"),
    ("lose", "loose"),
    ("where", "were", "we're"),
    ("whether", "weather"),
    ("principal", "principle"),
    ("complement", "compliment"),
    ("advice", "advise"),
    ("of", "off"),
    ("whose", "who's"),
    ("passed", "past"),
    ("lead", "led"),
    ("brake", "break"),
    ("bare", "bear"),
    ("cite", "site", "sight"),
    ("hear", "here"),
    ("peace", "piece"),
    ("plain", "plane"),
    ("role", "roll"),
    ("sole", "soul"),
    ("stationary", "stationery"),
    ("weak", "week"),
    ("know", "no"),
    ("knew", "new"),
    ("right", "write"),
    ("by", "buy", "bye"),
    ("for", "four"),
    ("one", "won"),
    ("allowed", "aloud"),
    ("board", "bored"),
    ("capital", "capitol"),
    ("desert", "dessert"),
    ("device", "devise"),
    ("elicit", "illicit"),
    ("ensure", "insure"),
    ("farther", "further"),
    ("formally", "formerly"),
    ("moral", "morale"),
    ("personal", "personnel"),
    ("precede", "proceed"),
    ("quiet", "quite"),
    ("through", "threw", "thorough"),
    ("waist", "waste"),
    ("which", "witch"),
    ("choose", "chose"),
    ("breath", "breathe"),
    ("cloth", "clothes"),
    ("conscience", "conscious"),
    ("council", "counsel"),
    ("later", "latter"),
    ("loath", "loathe"),
    ("rain", "reign", "rein"),
    ("seen", "scene"),
    ("steal", "steel"),
    ("tail", "tale"),
    ("vain", "vein"),
    ("wander", "wonder"),
    ("wave", "waive"),
    ("whole", "hole"),
    ("lightning", "lightening"),
    ("definitely", "defiantly"),
    ("emigrate", "immigrate"),
    ("beside", "besides"),
    ("everyday", "every day"),
    ("altogether", "all together"),
    ("anyway", "any way"),
    ("maybe", "may be"),
    # agreement and verb form
    ("is", "are"),
    ("was", "were"),
    ("has", "have"),
    ("does", "do"),
    ("doesn't", "don't"),
    ("isn't", "aren't"),
    ("wasn't", "weren't"),
    ("hasn't", "haven't"),
    ("this", "these"),
    ("that", "those"),
    ("a", "an"),
    ("who", "whom"),
    ("fewer", "less"),
    ("many", "much"),
    ("amount", "number"),
    ("among", "between"),
    ("went", "gone"),
    ("did", "done"),
    ("saw", "seen"),
    ("came", "come"),
    ("ran", "run"),
    ("began", "begun"),
    ("drank", "drunk"),
    ("sang", "sung"),
    ("spoke", "spoken"),
    ("took", "taken"),
    ("wrote", "written"),
    ("broke", "broken"),
    ("ate", "eaten"),
    ("gave", "given"),
    ("forgot", "forgotten"),
    ("rode", "ridden"),
    ("swam", "swum"),
    ("threw", "thrown"),
    ("lie", "lay"),
    ("rise", "raise"),
    ("sit", "set"),
    ("bring", "brought"),
    ("teach", "taught"),
    ("could", "can"),
    ("would", "will"),
    ("should", "shall"),
)

# Prepositions that non-native and hurried writers most often swap. A word may
# also sit in a confusable group ("to", "for"); the two operators are
# independent and the sampler picks one.
PREPOSITION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("in", "on", "at"),
    ("to", "for"),
    ("from", "of"),
    ("with", "by"),
    ("into", "onto"),
    ("about", "over"),
    ("since", "for"),
    ("during", "while"),
)

# Words most often dropped outright. Articles first: "cat sat on mat" is the
# single most common shape of non-native English. Then the prepositions and
# copulas whose absence still leaves a readable sentence.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the",
        "of", "to", "in", "on", "at", "for", "with", "by", "from",
        "is", "are", "was", "were", "be", "been",
        "has", "have", "had",
        "that", "which", "will", "would",
    }
)

# QWERTY neighbours, for substitution typos. Only lowercase ASCII letters;
# anything else falls back to a transposition.
KEYBOARD_NEIGHBORS: dict[str, str] = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh",
    "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd", "g": "tyhbvf",
    "h": "yujnbg", "j": "uikmnh", "k": "iolmj", "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def _build_alternatives(groups: tuple[tuple[str, ...], ...]) -> dict[str, tuple[str, ...]]:
    """
    word -> every other word it can be confused with.

    A word may sit in more than one group ("were" is both a homophone of
    "where" and the plural of "was"); its alternatives are the union, in
    group order, without duplicates.
    """
    table: dict[str, list[str]] = {}
    for group in groups:
        for word in group:
            others = table.setdefault(word, [])
            others.extend(other for other in group if other != word and other not in others)
    return {word: tuple(others) for word, others in table.items()}


CONFUSABLES: dict[str, tuple[str, ...]] = _build_alternatives(CONFUSABLE_GROUPS)
PREPOSITIONS: dict[str, tuple[str, ...]] = _build_alternatives(PREPOSITION_GROUPS)

__all__ = [
    "CONFUSABLE_GROUPS",
    "CONFUSABLES",
    "FUNCTION_WORDS",
    "KEYBOARD_NEIGHBORS",
    "PREPOSITION_GROUPS",
    "PREPOSITIONS",
]
