"""
Shared transform layer: raw dataset record -> tokenizer-training-ready text.

This is the single most important module in this folder. Every download
script imports it. It exists because the tokenizer we are fitting
(SentencePiece Unigram + byte_fallback, see ../tokenizer/DESIGN.md) makes
specific promises to the rephrase model:

  - NF1: decode(encode(x)) == x, byte-for-byte, for ANY input the model
    will ever see at inference (URLs, markdown links, code, numbers,
    emails, quoted text). The tokenizer's byte_fallback flag makes this
    *achievable*, but only if training records aren't corrupted before
    the tokenizer trainer ever sees them.
  - Gate 1 (round-trip) and Gate 6 (markdown-link/URL survival) in
    tokenizer/DESIGN.md are graded on text that passed through THIS module.

So every function here optimizes for one thing: get raw dataset text into
a clean, one-record-per-line UTF-8 file WITHOUT altering a single
character of the content itself. "Cleaning" here means removing dataset
scaffolding (HTML wrappers, XML markup, gutenberg boilerplate headers) --
never means normalizing, rewriting, or "fixing" the prose itself. That
job belongs to the tokenizer's own normalization_rule_name="identity"
setting (see DESIGN.md 6.1a) -- we don't duplicate or fight it here.

Guards implemented here map directly to DESIGN.md's risk register:
  R1  max_sentence_length silent drop  -> chunk_text()
  R2  NFKC silent corruption           -> we never call any unicode
                                           normalization function, ever
  R6  boilerplate/near-dup inflation   -> is_boilerplate() + dedup hook
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass

# Must match tokenizer/config max_sentence_length (DESIGN.md 6). Chunking
# below this keeps every record encodable and prevents SentencePiece's
# silent per-line drop (risk R1).
MAX_CHARS_PER_CHUNK = 3000

# Below this many characters a chunk carries almost no vocabulary signal
# (a lone footer, a "[deleted]" stub, a stray page number) and mostly adds
# noise to the piece-frequency statistics the Unigram trainer fits on.
MIN_CHARS_PER_RECORD = 20

_WS_RUN = re.compile(r"[ \t]{3,}\n")
_BLANK_RUNS = re.compile(r"\n{3,}")

# Repeated boilerplate lines that appear thousands of times verbatim across
# a raw dump (license footers, "Welcome to Project Gutenberg" banners,
# common-crawl nav junk). These would otherwise steal Unigram vocab slots
# in proportion to their raw frequency, not their information content.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.I),
    re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I),
    re.compile(r"^This eBook is for the use of anyone anywhere", re.I),
    re.compile(r"^Produced by\b", re.I),
    re.compile(r"^\s*Table of Contents\s*$", re.I),
    re.compile(r"^Cookie Policy|^Privacy Policy|^Terms of Service", re.I),
    re.compile(r"^Subscribe to our newsletter", re.I),
    re.compile(r"^Share this:|^Like this:|^Related [Aa]rticles?$"),
]


def is_boilerplate(line: str) -> bool:
    """True if a line is dataset/site scaffolding rather than authored content."""
    s = line.strip()
    if not s:
        return True
    for pat in _BOILERPLATE_PATTERNS:
        if pat.match(s):
            return True
    return False


def strip_html(text: str) -> str:
    """
    Remove HTML/XML tags while preserving everything between them exactly.
    Does NOT touch entities inside attributes or content -- html.unescape
    is applied once, after tag removal, matching the same approach used
    for the StackExchange batches (see project memory: html-tag cleanup).
    """
    # Drop script/style blocks entirely -- never rendered text.
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse only PURE structural whitespace noise (long trailing-space
    runs, 3+ blank lines) left over from format conversion. Does NOT
    collapse intra-line spacing -- that would violate byte-exactness for
    code indentation, which the tokenizer config explicitly preserves via
    remove_extra_whitespaces=False (DESIGN.md 6.1c).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub("\n", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    Split long documents at paragraph/sentence boundaries so no chunk
    exceeds max_chars. This is the direct mitigation for risk R1: without
    it, SentencePiece's max_sentence_length silently discards any raw
    line longer than its limit -- with no error and no log line pointing
    at the cause. Splitting pre-emptively guarantees every chunk is
    representable and nothing is dropped downstream.

    Never splits mid-URL, mid-code-fence, or mid-markdown-link: splits
    only occur at blank-line or sentence-end boundaries, both of which
    are outside those spans in well-formed text.
    """
    if len(text) <= max_chars:
        return [text] if len(text) >= MIN_CHARS_PER_RECORD else []

    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for para in paras:
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= max_chars:
            buf = para
        else:
            # Single paragraph still too long (e.g. no blank lines at
            # all): fall back to sentence-boundary splitting.
            buf = ""
            for piece in _split_long_paragraph(para, max_chars):
                chunks.append(piece)
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHARS_PER_RECORD]


def _split_long_paragraph(para: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = f"{buf} {sent}" if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            # Last resort: hard-cut an unsplittable run-on line. Rare in
            # practice (would need a single 3000+ char sentence with no
            # terminal punctuation), but must not raise -- silently
            # dropping the record would be worse than a mid-token cut in
            # a chunk boundary (the tokenizer's own boundary handles it).
            buf = sent[:max_chars] if len(sent) > max_chars else sent
    if buf:
        out.append(buf)
    return out


def clean_record(raw_text: str, *, strip_markup: bool = False) -> list[str]:
    """
    Full per-record pipeline: markup removal (if the source is
    HTML/XML) -> whitespace normalization -> boilerplate line filtering
    -> length-safe chunking. Returns zero or more final text chunks
    ready to be written one-per-line to corpus output.

    strip_markup=False by default: most HF text datasets (FineWeb, C4,
    Cosmopedia, PG-19) are already plain text: running strip_html on
    already-plain text is a harmless no-op (no '<' chars to match) but
    we keep the flag explicit so it's a conscious choice per source, not
    an accident.
    """
    if not raw_text:
        return []
    text = raw_text
    if strip_markup:
        text = strip_html(text)
    text = normalize_whitespace(text)
    lines = [ln for ln in text.split("\n") if not is_boilerplate(ln)]
    text = "\n".join(lines).strip()
    if len(text) < MIN_CHARS_PER_RECORD:
        return []
    return chunk_text(text)


@dataclass
class DedupFilter:
    """
    Cheap corpus-level near-dup guard (risk R6): rejects a chunk if its
    normalized-whitespace hash was already seen. Deliberately NOT a
    semantic/near-dup model -- exact-after-whitespace-normalization is
    enough to catch the actual offenders (repeated license footers,
    repeated forum "Welcome!" banners, repeated PG-19 title pages) without
    the cost or false-positive risk of fuzzy matching. Real semantic
    dedup already happened upstream for the StackExchange batches
    (see DATASET/ pipeline); this is a second, cheap net for the
    freshly-downloaded external corpora, which do NOT get that treatment
    yet.
    """

    seen: set[str]

    def __init__(self) -> None:
        self.seen = set()

    def accept(self, chunk: str) -> bool:
        key = hashlib.sha1(re.sub(r"\s+", " ", chunk).strip().lower().encode("utf-8")).hexdigest()
        if key in self.seen:
            return False
        self.seen.add(key)
        return True
