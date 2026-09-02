"""
Finding real PII in a document and swapping it for a unique synthetic one.

This is the layer that decides *what* gets replaced. `pii_allocator.py`
decides what it is replaced *with*, and never sees the original.

Two rules shape almost every decision here.

**Code and URLs are untouchable.** The whole model exists to reproduce URLs,
code and numbers character-for-character (README, "Fidelity beats
compression"). A phone-number regex loose enough to catch
`+1 (555) 010-4477` is also loose enough to catch a version string, an
account id, or a line of a phone-formatting unit test — and a mangled code
sample is a worse outcome than a surviving phone number. So masking runs
**only outside fenced blocks, indented blocks and inline code spans**, and
each pattern is written to be conservative at the edges.

**Same document, same person; different document, different person.** Within
one post, `@alice` appearing three times has to become the *same* handle all
three times, or a conversation about one person turns into a conversation
about three. Across documents the same real handle deliberately maps to a
*different* fake one — which both breaks the linkage that would let someone
re-identify a user by following a consistent pseudonym, and stops any single
fake identity from accumulating the frequency that made `@RaviMarino` show up
dozens of times in the earlier batches (`cleanup.md` §2.2). The memo dict is
therefore per-document and is cleared on every call.

Ordering matters
----------------
Emails are masked before @mentions. `alice@example.com` contains something
that looks exactly like an `@mention` of `example`, and handling emails first
means the mention pattern never sees it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import PIIRegistryError
from .pii_allocator import (
    IdentityAllocator,
    email_allocator,
    handle_allocator,
    phone_subscriber_digits,
)
from .pii_registry import AuditLog, Reservation, load_legacy_reserved

# -- detection ------------------------------------------------------------

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*"
    r"\.[A-Za-z]{2,24}"
    r"(?![A-Za-z0-9-])"
)

# An @handle. Must not follow a word character, `/` (npm scopes like
# `@angular/core` are preceded by nothing but are caught by the trailing
# check), `.` or `-`, and must not be immediately followed by `/` — which is
# what distinguishes a mention of a person from a scoped package path.
MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9._%+@-])@([A-Za-z][A-Za-z0-9_.-]{1,30})(?![A-Za-z0-9_.-]*/)"
)

# Phone numbers, deliberately narrow. Requires either a leading `+` country
# code or at least one separator between digit groups, so bare integers
# (ids, timestamps, byte counts) are never candidates. The digit-count check
# in `_looks_like_phone` does the rest of the filtering.
# Written as "optional country code, then two-to-five separated digit groups",
# which is what a phone number actually is, rather than as a stack of
# alternatives per national format. Two notes on the guards, both of which
# were bugs caught by the smoke test rather than theory:
#
#   * groups run to five digits, not four. Indian mobiles are written
#     `+91 98765 43210`; a four-digit cap matched `91 98765` and left the
#     trailing `43210` — real PII — sitting in the output.
#   * the trailing guard is `(?!\.?\d)`, not `(?![\w.])`. A phone number at
#     the end of a sentence is followed by a full stop, and refusing to match
#     on a trailing `.` skipped every one of them. Blocking only `.<digit>`
#     still stops the pattern from biting the front off `10.0.0.1`.
PHONE_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{1,5}\)|\d{1,5})"
    # The separator here MUST be mandatory, not `[\s.-]?`. An optional
    # separator lets the engine split a purely bare digit run (a post id,
    # a timestamp) into two fake "groups" with a zero-length gap between
    # them -- e.g. "123456789" as "12345" + "6789" -- which matched and
    # masked bare integers despite this pattern's own stated intent to
    # exclude them.
    r"(?:[\s.-]\(?\d{1,5}\)?){1,4}"
    r"(?!\.?\d)"
)

# Which leading digits survive masking: the country code when the number
# carries one, otherwise the first group (area/trunk code). Everything after
# it is replaced. Keeping exactly one leading group is what makes a masked
# Indian mobile still read as an Indian mobile and a masked US number still
# read as a US number, per cleanup.md's format-preserving rule.
_LEADING_GROUP_RE = re.compile(r"^\s*(?:\+\d{1,3}|\(?\d{1,5}\)?)")

# Regions whose contents are reproduced verbatim: fenced blocks, inline code
# spans, and markdown link/image *targets* (the label is prose and stays
# maskable, the URL is not).
_PROTECTED_RE = re.compile(
    r"```.*?```"  # fenced block
    r"|~~~.*?~~~"
    r"|`[^`\n]+`"  # inline code span
    r"|<https?://[^>\s]+>"  # autolink
    r"|\]\([^)\s]+\)"  # markdown link target
    r"|\bhttps?://\S+"  # bare url
    r"|\bwww\.\S+",
    re.DOTALL,
)

# A line that is indented four spaces or a tab is a markdown code block.
_INDENTED_CODE_RE = re.compile(r"^(?: {4}|\t).*$", re.MULTILINE)

_ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")

MIN_PHONE_DIGITS = 7
MAX_PHONE_DIGITS = 15


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges that must be reproduced verbatim, merged and sorted."""
    spans = [m.span() for m in _PROTECTED_RE.finditer(text)]
    spans += [m.span() for m in _INDENTED_CODE_RE.finditer(text)]
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, stop in spans[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _looks_like_phone(candidate: str) -> bool:
    """
    Reject the things a permissive phone pattern sweeps up by accident.

    Version strings, ISO dates and long numeric identifiers all match the
    shape of a phone number closely enough that only the digit budget and a
    couple of explicit shapes separate them.
    """
    stripped = candidate.strip()
    if _ISO_DATE_RE.match(stripped):
        return False
    digits = [c for c in stripped if c.isdigit()]
    if not MIN_PHONE_DIGITS <= len(digits) <= MAX_PHONE_DIGITS:
        return False
    # `1.2.3` / `10.0.0.1` — dot-separated small groups are versions and IPs.
    if "." in stripped and not any(c in stripped for c in "()+ -"):
        return False
    return True


# -- masking --------------------------------------------------------------


@dataclass
class MaskStats:
    """What a masker did, so a run can report it instead of guessing."""

    emails: int = 0
    mentions: int = 0
    phones: int = 0
    names: int = 0
    documents: int = 0
    blocklist_skips: int = 0

    def merge(self, other: MaskStats) -> None:
        self.emails += other.emails
        self.mentions += other.mentions
        self.phones += other.phones
        self.names += other.names
        self.documents += other.documents
        self.blocklist_skips += other.blocklist_skips

    def as_dict(self) -> dict[str, int]:
        return {
            "documents_scanned": self.documents,
            "emails_masked": self.emails,
            "mentions_masked": self.mentions,
            "phones_masked": self.phones,
            "names_masked": self.names,
            "blocklist_skips": self.blocklist_skips,
        }


@dataclass
class _Cursor:
    """A worker's position inside one reserved index range."""

    start: int
    stop: int
    at: int = field(init=False)

    def __post_init__(self) -> None:
        self.at = self.start

    def take(self) -> int:
        if self.at >= self.stop:
            raise PIIRegistryError(
                f"exhausted the reserved PII index range [{self.start:,}, {self.stop:,}); "
                f"raise --pii-block-size and rerun. No value was reused."
            )
        index = self.at
        self.at += 1
        return index


class PIIMasker:
    """
    Masks one worker's stream of documents, drawing from its own index range.

    One instance per worker. It shares nothing mutable with any other
    instance, which is what makes the hot path lock-free: the ranges were
    already carved apart in `PIIRegistry.reserve`.
    """

    def __init__(
        self,
        reservation: Reservation,
        *,
        pools: dict[str, tuple[str, ...]] | None = None,
        legacy: dict[str, frozenset[str]] | None = None,
        audit: AuditLog | None = None,
        mask_phones: bool = True,
    ) -> None:
        self._emails: IdentityAllocator = email_allocator(pools, salt=reservation.salt)
        self._handles: IdentityAllocator = handle_allocator(pools, salt=reservation.salt)
        self._legacy = legacy if legacy is not None else load_legacy_reserved()
        self._audit = audit
        self._mask_phones = mask_phones

        self._cursors = {
            kind: _Cursor(r.start, r.stop) for kind, r in reservation.ranges.items()
        }
        self.stats = MaskStats()

    # -- value generation ------------------------------------------------
    def _next_email(self) -> str:
        cursor = self._cursors["email"]
        while True:
            index = cursor.take()
            value = self._emails.identity(index).email()
            if value in self._legacy["email"]:
                self.stats.blocklist_skips += 1
                continue
            self._record("email", index, value)
            return value

    def _next_handle(self) -> str:
        cursor = self._cursors["handle"]
        index = cursor.take()
        value = self._handles.identity(index).handle()
        self._record("handle", index, value)
        return value

    def _next_full_name(self) -> str:
        cursor = self._cursors["handle"]
        index = cursor.take()
        value = self._handles.identity(index).full_name()
        self._record("name", index, value)
        return value

    def _next_phone(self, original: str) -> str:
        """
        Format-preserving replacement: same separators, same digit count.

        Only the trailing subscriber digits change; the country/area prefix
        is kept so a masked Indian mobile still reads as an Indian mobile,
        which is what makes the corpus teach the *shape* of a phone number.
        """
        cursor = self._cursors["phone"]
        digits = [i for i, c in enumerate(original) if c.isdigit()]
        lead = _LEADING_GROUP_RE.match(original)
        lead_end = lead.end() if lead else 0
        keep = sum(1 for position in digits if position < lead_end)
        # A number whose leading group is the whole thing has nothing left to
        # randomise; replace all but the first digit rather than emit it back
        # unchanged.
        keep = min(keep, len(digits) - 1) if len(digits) > 1 else 0
        n_replace = len(digits) - keep

        while True:
            index = cursor.take()
            replacement = phone_subscriber_digits(index, n_replace)
            chars = list(original)
            for offset, position in enumerate(digits[keep:]):
                chars[position] = replacement[offset]
            value = "".join(chars)
            if value in self._legacy["phone"]:
                self.stats.blocklist_skips += 1
                continue
            self._record("phone", index, value)
            return value

    def _record(self, kind: str, index: int, value: str) -> None:
        if self._audit is not None:
            self._audit.record(kind, index, value)

    # -- document masking ------------------------------------------------
    def mask_document(self, text: str) -> str:
        """
        Replace every detected email, @mention and phone number in `text`.

        Returns the text with code spans, URLs and every other character
        untouched. Repeats of the same original within this call resolve to
        the same replacement; the memo is discarded when the call returns.
        """
        self.stats.documents += 1
        if not text:
            return text

        memo: dict[str, str] = {}
        protected = protected_spans(text)

        def is_protected(start: int, stop: int) -> bool:
            for p_start, p_stop in protected:
                if start < p_stop and stop > p_start:
                    return True
                if p_start >= stop:
                    break
            return False

        def substitute(pattern: re.Pattern[str], make: callable, counter: str) -> None:
            nonlocal text, protected

            def repl(match: re.Match[str]) -> str:
                original = match.group(0)
                if is_protected(match.start(), match.end()):
                    return original
                if counter == "phones" and not _looks_like_phone(original):
                    return original
                if original not in memo:
                    memo[original] = make(original)
                setattr(self.stats, counter, getattr(self.stats, counter) + 1)
                return memo[original]

            text = pattern.sub(repl, text)
            protected = protected_spans(text)

        substitute(EMAIL_RE, lambda _: self._next_email(), "emails")
        substitute(MENTION_RE, lambda _: "@" + self._next_handle(), "mentions")
        if self._mask_phones:
            substitute(PHONE_RE, self._next_phone, "phones")
        return text

    def mask_name_field(self, name: str) -> str:
        """
        Replace a name that arrived as a structured field, not as free text.

        Stack Exchange dumps carry `DisplayName` / `OwnerDisplayName`
        attributes, which are real user names identified by the schema rather
        than guessed from prose. Those are worth masking precisely because
        there is no false-positive risk: the format already told us it is a
        name. Person names inside running prose are deliberately *not*
        touched — detecting them needs an NER model, and a false positive
        there rewrites ordinary words in the training text.
        """
        if not name or not name.strip():
            return name
        self.stats.names += 1
        return self._next_full_name()
