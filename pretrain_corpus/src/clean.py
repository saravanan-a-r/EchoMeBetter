"""
Raw dataset record -> clean, length-safe pretraining text.

Carried over from the tokenizer corpus downloader's `transform.py`, whose
guarantees still hold word for word here, plus one addition (mojibake repair).
The invariant is unchanged and is the reason this module is conservative to
the point of being boring:

    "Cleaning" means removing dataset *scaffolding* — HTML wrappers, Gutenberg
    boilerplate headers, nav junk. It never means normalizing, rewriting or
    "fixing" the prose itself.

That matters more for pretraining than it did for tokenizer fitting. The
tokenizer only had to be able to *represent* whatever it was shown; the model
has to learn to reproduce URLs, code and numbers character-for-character
(architecture.md NF1, README "Fidelity beats compression"). Anything this
layer silently rewrites is something the model will confidently get wrong.

Specifically, and deliberately:

  * **No Unicode normalization, ever.** The tokenizer is fitted with
    `normalization_rule_name="identity"`. Applying NFKC here would corrupt
    exactly the characters byte-fallback exists to preserve.
  * **No intra-line whitespace collapsing.** Code indentation is content.
  * **Chunking splits only at paragraph or sentence boundaries**, never
    mid-URL, mid-code-fence or mid-markdown-link.

The one behaviour that is new relative to `transform.py` is `repair_mojibake`,
described at its definition.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass

# Matched to the tokenizer's `max_sentence_length`. Chunking below this is
# what stops SentencePiece from silently discarding an over-long line — risk
# R1 in the tokenizer design, and the single easiest way to train on a
# corrupted subset of a corpus with no error message anywhere.
MAX_CHARS_PER_CHUNK = 3000

# Below this a chunk carries essentially no signal: a lone footer, a
# "[deleted]" stub, a stray page number.
MIN_CHARS_PER_RECORD = 20

_WS_RUN = re.compile(r"[ \t]{3,}\n")
_BLANK_RUNS = re.compile(r"\n{3,}")

# Lines that appear verbatim thousands of times across a raw dump. At
# pretraining scale these are worse than wasted tokens: a phrase repeated a
# million times is a phrase the model will emit unprompted.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.I),
    re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I),
    re.compile(r"^This eBook is for the use of anyone anywhere", re.I),
    re.compile(r"^Produced by\b", re.I),
    re.compile(r"^\s*Table of Contents\s*$", re.I),
    re.compile(r"^Cookie Policy|^Privacy Policy|^Terms of Service", re.I),
    re.compile(r"^Subscribe to our newsletter", re.I),
    re.compile(r"^Share this:|^Like this:|^Related [Aa]rticles?$"),
    # -- added for the Stack Exchange dumps ------------------------------
    # Site chrome and moderator macros that the XML carries as ordinary post
    # text. The alt-text one is the big one: Stack Exchange auto-fills
    # "enter image description here" for every pasted screenshot, and the
    # earlier cleanup counted 76,760 occurrences in a single batch.
    re.compile(r"^\s*enter image description here\s*$", re.I),
    re.compile(r"^\s*(edit|update)\s*\d*\s*:\s*$", re.I),
    re.compile(r"^\s*Possible Duplicate\s*:", re.I),
    re.compile(r"^\s*This question (already has|appears to be off-topic)", re.I),
]

# Image alt-text carrying the Stack Exchange placeholder. Blanked rather than
# removed, so `![placeholder](url)` becomes `![](url)` and the URL survives —
# the surrounding markdown is content, only the fake caption is not.
_ALT_PLACEHOLDER_RE = re.compile(r"(!\[)[^\]]*enter image description here[^\]]*(\])", re.I)


def is_boilerplate(line: str) -> bool:
    """True if a line is dataset or site scaffolding rather than authored content."""
    s = line.strip()
    if not s:
        return True
    return any(pattern.match(s) for pattern in _BOILERPLATE_PATTERNS)


def strip_html(text: str) -> str:
    """
    Remove HTML/XML tags, preserving everything between them exactly.

    `html.unescape` runs once, after tag removal. Running it *before* would
    turn a user's literal `&lt;target&gt;` into a real-looking `<target>` tag
    which the next pass would then delete — a bug that destroyed user content
    in an earlier batch and cost a full re-download.
    """
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def repair_mojibake(text: str) -> str:
    """
    Undo double-encoded UTF-8 (`â€™` for `'`, `Ã©` for `é`).

    New relative to the tokenizer's transform layer, and worth the dependency
    because mojibake is the one text defect that byte-fallback makes *worse*
    rather than better: the tokenizer will faithfully represent `â€™`, so the
    model learns the corruption as a legitimate way to write an apostrophe.

    `ftfy` is optional. When it is not installed this is the identity
    function and the run says so once, rather than failing — a corpus with
    some mojibake in it is still a usable corpus, unlike one that never
    downloaded.
    """
    if _FTFY is None:
        return text
    return _FTFY(text)


def _load_ftfy():
    try:
        from ftfy import fix_text

        return fix_text
    except ImportError:  # pragma: no cover - depends on the environment
        return None


_FTFY = _load_ftfy()
FTFY_AVAILABLE = _FTFY is not None


def normalize_whitespace(text: str) -> str:
    """
    Collapse only structural whitespace noise left over from format conversion.

    Long trailing-space runs and 3+ blank lines go; intra-line spacing stays,
    because collapsing it would break code indentation, which the tokenizer
    preserves deliberately (`remove_extra_whitespaces=False`).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub("\n", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    Split long documents at paragraph or sentence boundaries under `max_chars`.

    Splits fall only on blank-line or sentence-end boundaries, both of which
    sit outside URLs, code fences and markdown links in well-formed text.
    """
    if len(text) <= max_chars:
        return [text] if len(text) >= MIN_CHARS_PER_RECORD else []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= max_chars:
            buffer = paragraph
        else:
            buffer = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars))
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if len(c) >= MIN_CHARS_PER_RECORD]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}" if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            out.append(buffer)
            buffer = ""
        if len(sentence) <= max_chars:
            buffer = sentence
        else:
            # Last resort for a single unsplittable run-on span with no
            # sentence punctuation at all (rare, but real -- e.g. a URL-heavy
            # line or a run of hyphens). Emit *every* max_chars-sized slice,
            # not just the first: a single hard cut here previously kept only
            # the first slice and silently discarded the rest of the
            # document, which is exactly the silent-drop failure mode this
            # function exists to prevent.
            for start in range(0, len(sentence), max_chars):
                out.append(sentence[start : start + max_chars])
    if buffer:
        out.append(buffer)
    return out


def clean_record(raw_text: str, *, strip_markup: bool = False, repair_encoding: bool = True) -> list[str]:
    """
    Full per-record pipeline, returning zero or more ready-to-write chunks.

    Order is load-bearing: markup removal has to precede whitespace
    normalization (tags become spaces), the alt-text blank has to precede
    boilerplate filtering (so a screenshot-only line becomes empty and is
    dropped), and chunking has to come last so every emitted chunk is
    length-safe no matter what the earlier passes did.
    """
    if not raw_text:
        return []
    text = raw_text
    if repair_encoding:
        text = repair_mojibake(text)
    if strip_markup:
        text = strip_html(text)
    text = _ALT_PLACEHOLDER_RE.sub(r"\1\2", text)
    text = normalize_whitespace(text)
    text = "\n".join(line for line in text.split("\n") if not is_boilerplate(line)).strip()
    if len(text) < MIN_CHARS_PER_RECORD:
        return []
    return chunk_text(text)


@dataclass
class DedupKey:
    """
    The two hashes every chunk is deduplicated on.

    `exact` catches byte-identical repeats; `normalized` catches the same text
    differing only in case, whitespace or punctuation — which is what most
    real duplication looks like once a document has been through two different
    scrapers. Both are 64-bit, because that is what fits in the shared-memory
    Bloom filter in `dedup.py`.
    """

    exact: int
    normalized: int


_NORMALIZE_RE = re.compile(r"[^\w\s]+")
_WS_COLLAPSE_RE = re.compile(r"\s+")


def dedup_key(chunk: str) -> DedupKey:
    exact = _hash64(chunk)
    folded = _WS_COLLAPSE_RE.sub(" ", _NORMALIZE_RE.sub("", chunk.lower())).strip()
    return DedupKey(exact=exact, normalized=_hash64(folded))


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
