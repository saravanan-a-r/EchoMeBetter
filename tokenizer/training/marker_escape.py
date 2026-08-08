"""
Escapes SentencePiece's internal space marker (U+2581, "▁") out of raw text
before it reaches `encode`/`decode`, and reverses it afterward.

Why this exists
----------------
SentencePiece's decoder unconditionally rewrites every `▁` it sees into a
space. If the *input* text already contains a literal `▁` (real text uses
this in ASCII-art / terminal block-shading, e.g. "▃▁▁"), the decoder cannot
tell that occurrence apart from its own marker, and the character is
silently lost on decode -- a byte-exact round-trip failure that is
structural to SentencePiece, not a bug in this codebase. Full writeup:
Marker_Encoder_Design_EDGE_CASE.md.

The fix: escape every literal `▁` (and the escape character itself, so the
scheme is self-consistent for any input) before `encode`, and reverse it
after `decode`. Every caller in this module that performs a "real" encode or
decode -- one meant to preserve arbitrary text -- must route through
`escape_markers` / `unescape_markers`.

Single-pass, not two blind find-and-replace passes
----------------------------------------------------
Escaping the escape character *and* the target character with two separate
whole-text replacement passes is unsound: the first pass's output can
manufacture a sequence that the second pass misreads as something it never
was in the original text (worked example in the design doc). `escape_markers`
avoids this by making one left-to-right decision per *original* character,
consuming input once, so it never re-scans text it has already emitted.
`unescape_markers` is the mirror: one left-to-right pass over the *escaped*
text, consuming one or two characters at a time, never re-reading emitted
output.
"""

from __future__ import annotations

# SentencePiece's internal word-boundary marker. If this appears literally
# in input text, sp.decode() will rewrite it to a space -- indistinguishable
# from a marker it inserted itself.
SPIECE_UNDERLINE = "▁"

# Private Use Area codepoints: never assigned a meaning by Unicode, never
# produced by a keyboard or text pipeline. Used only as escape machinery.
# (Verified empirically that _ESC alone is not a sufficient assumption --
# see the design doc -- which is why the scheme below escapes _ESC too.)
_ESC = ""
_ESCAPED_UNDERLINE = ""


def escape_markers(text: str) -> str:
    """Hide literal `▁` (and literal `_ESC`) from SentencePiece."""
    if _ESC not in text and SPIECE_UNDERLINE not in text:
        return text
    out = []
    for ch in text:
        if ch == _ESC:
            out.append(_ESC)
            out.append(_ESC)
        elif ch == SPIECE_UNDERLINE:
            out.append(_ESC)
            out.append(_ESCAPED_UNDERLINE)
        else:
            out.append(ch)
    return "".join(out)


def unescape_markers(text: str) -> str:
    """Reverse `escape_markers`."""
    if _ESC not in text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == _ESC and i + 1 < n:
            nxt = text[i + 1]
            if nxt == _ESC:
                out.append(_ESC)
                i += 2
                continue
            if nxt == _ESCAPED_UNDERLINE:
                out.append(SPIECE_UNDERLINE)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)
