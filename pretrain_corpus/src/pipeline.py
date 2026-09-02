"""
The per-record pipeline: one raw dataset record in, zero or more clean
training chunks out.

Every source funnels through here, which is the point — the fidelity rules
that protect URLs, code and numbers are defined once and cannot drift apart
between five downloaders. A source's only job is to yield raw text (and, for
Stack Exchange, a couple of structured name fields); everything that happens
to that text afterwards is this file.

Order of operations, and why it is this order
---------------------------------------------
    1. clean      scaffolding out, mojibake repaired, split into chunks
    2. PII mask   real identities replaced with unique synthetic ones
    3. quality    is what is left actually English prose?
    4. dedup      have we already written this, from any source?

**Clean before mask** because the masker's code-span protection reads
markdown, and a record still wrapped in HTML has no markdown to read.

**Mask before quality** because masking changes the text: a comment that is
nothing but an email address becomes a comment that is nothing but a
different email address, and it should be judged on what will actually be
written to disk.

**Quality before dedup** because the dedup filters are a finite resource. A
chunk rejected for quality should never consume Bloom-filter capacity — that
capacity is sized for the text we keep.

Every stage counts what it drops. A source that silently shrinks is the
failure mode this project's design principles single out ("Fail loudly, never
silently"), so the numbers travel into the manifest rather than into a log
line nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .clean import clean_record, dedup_key
from .dedup import Deduplicator
from .pii_masker import PIIMasker
from .quality import QualityFilter


@dataclass
class PipelineStats:
    """A tally per stage, merged across workers into the run manifest."""

    records_in: int = 0
    records_empty_after_clean: int = 0
    chunks_produced: int = 0
    chunks_dropped_quality: int = 0
    chunks_dropped_duplicate: int = 0
    chunks_emitted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "records_in": self.records_in,
            "records_empty_after_clean": self.records_empty_after_clean,
            "chunks_produced": self.chunks_produced,
            "chunks_dropped_quality": self.chunks_dropped_quality,
            "chunks_dropped_duplicate": self.chunks_dropped_duplicate,
            "chunks_emitted": self.chunks_emitted,
        }


@dataclass
class PipelineConfig:
    """
    Per-source switches.

    All three exist because one source genuinely needs each of them off:
    `strip_markup` for the HTML-bearing Stack Exchange bodies, `quality` off
    for the code source (source files are not English prose and every
    heuristic would reject them correctly and uselessly), and `mask_pii` off
    for code as well, where an `@decorator` is syntax rather than a mention.
    """

    strip_markup: bool = False
    repair_encoding: bool = True
    quality: bool = True
    mask_pii: bool = True
    mask_phones: bool = True


class RecordPipeline:
    """
    One per worker. Holds that worker's masker and quality tally; shares only
    the deduplicator, which is explicitly built for concurrent use.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        masker: PIIMasker | None,
        quality: QualityFilter,
        deduplicator: Deduplicator,
    ) -> None:
        self.config = config
        self.masker = masker
        self.quality = quality
        self.deduplicator = deduplicator
        self.stats = PipelineStats()

    def process(self, raw_text: str) -> list[str]:
        """Run one raw record all the way through. Returns chunks to write."""
        self.stats.records_in += 1
        chunks = clean_record(
            raw_text,
            strip_markup=self.config.strip_markup,
            repair_encoding=self.config.repair_encoding,
        )
        if not chunks:
            self.stats.records_empty_after_clean += 1
            return []

        out: list[str] = []
        for chunk in chunks:
            self.stats.chunks_produced += 1

            if self.config.mask_pii and self.masker is not None:
                chunk = self.masker.mask_document(chunk)

            if not self.quality.accept(chunk):
                self.stats.chunks_dropped_quality += 1
                continue

            if not self.deduplicator.accept(dedup_key(chunk)):
                self.stats.chunks_dropped_duplicate += 1
                continue

            self.stats.chunks_emitted += 1
            out.append(chunk)
        return out

    def mask_name(self, name: str) -> str:
        """
        Mask a name that arrived as a structured field rather than as prose.

        Used by the Stack Exchange reader for `DisplayName` attributes, where
        the schema — not a guess — says the value is a person's name.
        """
        if not (self.config.mask_pii and self.masker is not None):
            return name
        return self.masker.mask_name_field(name)

    def report(self) -> dict:
        report = {"pipeline": self.stats.as_dict(), "quality": self.quality.as_dict()}
        if self.masker is not None:
            report["pii"] = self.masker.stats.as_dict()
        return report
