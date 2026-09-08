"""
Cutting a document into the pieces the corruption operators may touch, and
the pieces they must not.

Two decisions are made here and nowhere else:

**Which lines are prose.** A record from the corpus is one document with its
line structure intact (`corpus/src/reader.py::join_record_lines`). A line of
code, a Markdown table row, a heading, an all-caps banner or a bare URL is not
prose, and "fixing the grammar" of it is not a thing -- lowercasing an
identifier does not produce a grammar error, it produces different code, and a
target that restores it would teach the model to edit code. Those lines are
copied through verbatim, which is exactly the behaviour the product needs
(README: URLs, code, numbers and Markdown survive character for character).

**Which spans inside a prose line are protected.** URLs, e-mail addresses,
paths, numbers, inline code, identifiers, version strings, acronyms, handles
and anything in angle brackets are cut out as single `protected` pieces that
no operator will modify. The model therefore sees them intact on both sides
of every example -- the copy-through behaviour is trained, not hoped for.

Everything else is a `word`, a run of `space`, or a single `punct` character.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Protected spans, tried before anything else at every position. Order
# matters only where two alternatives could match the same prefix; the more
# specific ones come first.
_PROTECTED = r"""
    `[^`\n]*`                                   # inline code
  | (?:https?://|ftp://|www\.)\S+               # URLs
  | [\w.+-]+@[\w-]+(?:\.[\w-]+)+                # e-mail addresses
  | (?:~|\.{1,2})?/[\w.~+-]+(?:/[\w.~+-]*)*     # paths: /usr/bin, ./a/b, ~/x
  | <[^<>\n]{1,60}>                             # tags and reserved tokens
  | [#@$][\w-]+                                 # #tags, @handles, $VARS
  | \w+(?:[._-]\w+)+                            # foo_bar, v1.2.3, self-driving
  | [A-Za-z]*\d[\w%]*                           # 4K, x86, 3rd, 2024, 40%
  | [A-Z]{2,}(?![a-z])                          # acronyms: NASA, API
  | [a-z]+[A-Z]\w*                              # camelCase identifiers
"""

# A word is letters, optionally with internal apostrophes: don't, o'clock,
# John's. Letters include non-ASCII ones so accented words are still words.
_WORD = r"[^\W\d_]+(?:['’][^\W\d_]+)*"

_PIECE = re.compile(
    rf"(?P<protected>{_PROTECTED})|(?P<word>{_WORD})|(?P<space>[ \t]+)|(?P<punct>.)",
    re.VERBOSE,
)

# A line beginning like this is code, a comment, a directive or Markdown
# structure, whatever the rest of it looks like.
_CODE_LINE_START = re.compile(
    r"""^(?:
        \$\s | \#\!? | // | /\* | \*/ | --\s | ; | \{ | \} | \[ | \| | <
      | import\s | from\s+\S+\s+import\s | def\s | class\s | return\s | function\s
      | var\s | let\s | const\s | if\s*\( | for\s*\( | while\s*\(
      | \w+\s*=\s*\S+$                           # x = 1
    )""",
    re.VERBOSE,
)

# Characters that are rare in prose and dense in code.
_SYMBOLS = frozenset("{}[]<>=|\\;$#%^&*_~`")


@dataclass
class Piece:
    """One segment of a line. `out` is what the corrupted line will contain."""

    kind: str  # "protected" | "word" | "space" | "punct"
    text: str
    out: str = field(init=False)

    def __post_init__(self) -> None:
        self.out = self.text

    @property
    def edited(self) -> bool:
        return self.out != self.text


def split_line(line: str) -> list[Piece]:
    """Cut one line into pieces. Concatenating `text` gives the line back."""
    pieces: list[Piece] = []
    for match in _PIECE.finditer(line):
        kind = match.lastgroup
        assert kind is not None
        pieces.append(Piece(kind, match.group()))
    return pieces


def is_prose_line(line: str, *, min_words: int = 4) -> bool:
    """
    Whether a line is running English text the operators may touch.

    Deliberately conservative: a false "no" costs one line of training
    signal, a false "yes" produces a target that rewrites code or a table.
    """
    if not any(ch.islower() for ch in line):
        return False  # ALL CAPS banners, numbers-only, empty
    if _CODE_LINE_START.match(line.lstrip()):
        return False
    # Counted the way the operators see them: `json.loads(line)` holds one
    # protected identifier and one word, not three words.
    if sum(1 for piece in split_line(line) if piece.kind == "word") < min_words:
        return False

    dense = [ch for ch in line if not ch.isspace()]
    if not dense:
        return False
    letters = sum(ch.isalpha() for ch in dense)
    symbols = sum(ch in _SYMBOLS for ch in dense)
    if letters / len(dense) < 0.6:
        return False
    if symbols / len(dense) > 0.04:
        return False
    return True


def count_prose_words(line: str) -> int:
    """Words in a line that an operator could act on (protected spans excluded)."""
    return sum(1 for piece in split_line(line) if piece.kind == "word")


def protected_spans(text: str) -> list[str]:
    """Every protected span in `text`, in order. Used by tests as the oracle."""
    return [
        piece.text
        for line in text.split("\n")
        for piece in split_line(line)
        if piece.kind == "protected"
    ]


__all__ = ["Piece", "split_line", "is_prose_line", "count_prose_words", "protected_spans"]
