"""
`TokenizedCorpus` — the resumable stream of token-ID lines the pretraining
loop consumes, one line at a time, forever.

This is `training/conftest.py`'s `ListIterator` grown up: the training suite
already documents the contract a data pipeline must satisfy —

    def state_dict(self) -> dict: ...
    def load_state_dict(self, state) -> None: ...

— and notes "the corpus module is not built yet". This is that module.

Why position is (file index, raw-line count), not a global line number
------------------------------------------------------------------------
A global "line 4,827,193 of the corpus" position is useless without
re-scanning every file before it to find that offset — exactly the O(corpus
size) cost architecture.md's tokenizer corpus reader goes out of its way to
avoid. Tracking *which file* and *how many raw lines already consumed from
it* lets resumption reopen exactly one file and skip forward with `readline()`
calls, independent of how large the corpus or how far into it training had
reached (architecture.md §7.9).

Why file order is shuffled once, not re-shuffled per epoch
------------------------------------------------------------
A deterministic shuffle, fixed for the life of the run, is what makes
`state_dict()` cheap: the order is derived once from `seed` and carried in
the state as a plain list, so resuming never has to re-derive "what would the
shuffle at epoch N have produced" — it just is the order.

One record in, one sequence out
-------------------------------
A JSONL record is emitted as a single tokenized sequence, with its internal
line structure preserved (`reader.join_record_lines`). It is *not* split into
one example per line: that would discard the document structure the model is
being taught, and would leave the average example far shorter than the
context the model was sized for.

That also makes the resumable position exact. A record is the unit both of
iteration and of the saved position, so resuming re-reads no partial record
and skips none — the earlier caveat about resuming mid-record no longer
applies, because there is no longer a sub-record unit to be in the middle of.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from .errors import CorpusConfigError
from .reader import extract_text, is_jsonl, join_record_lines
from .tokenizer_interop import escape_markers


class TokenizedCorpus:
    """
    An infinite, resumable iterator of `list[int]` token-ID sequences.

    Loops over `files` in a fixed, seeded-shuffled order; when the last file
    is exhausted it starts again from the first, incrementing `epoch`.
    architecture.md §7.5: "Multiple epochs over a well-deduplicated corpus are
    fine" — this is what makes that possible for a corpus far smaller than
    the 50-100B token target.
    """

    def __init__(self, files: Sequence[str | Path], sp: Any, *, seed: int = 0) -> None:
        self.files: list[Path] = [Path(f) for f in files]
        if not self.files:
            raise CorpusConfigError("TokenizedCorpus needs at least one file")
        self.sp = sp
        self.seed = seed

        self._order: list[int] = list(range(len(self.files)))
        random.Random(seed).shuffle(self._order)

        self._pos = 0
        self._line = 0
        self.epoch = 0
        self._fh = None

        # Visible, not just internal: a run whose corpus is mostly malformed
        # or mostly oversized should be discoverable without instrumenting
        # the trainer.
        self.lines_read = 0
        self.lines_skipped_unusable = 0
        self.examples_emitted = 0

        self._open_current()

    # -- iteration ----------------------------------------------------------

    def __iter__(self) -> "TokenizedCorpus":
        return self

    def __next__(self) -> list[int]:
        """
        The next **whole record**, tokenized.

        One record in, one sequence out. This is deliberately not one sequence
        per line: a record's newline-separated lines are parts of one
        document, and training on them separately throws away the document
        structure that makes them mean anything — a man page's example split
        from the paragraph explaining it, a book's sentence split from its
        paragraph. Measured over this corpus, splitting on newlines turns an
        average 401-token record into 6.5 examples of ~61 tokens each, so a
        batch padded to its longest member is then mostly `<pad>` and the
        model never sees an input long enough to exercise the 2048-token
        context it was sized for.

        Line structure inside the record is preserved by rejoining with
        newlines, so the model still learns where lines break — the tokenizer
        keeps them (`allow_whitespace_only_pieces`, byte fallback), and
        document shape is exactly what a rewriting model needs to reproduce.
        """
        while True:
            raw = self._fh.readline()
            if raw == "":
                self._advance_file()
                continue

            self._line += 1
            self.lines_read += 1
            text = extract_text(raw.rstrip("\n").rstrip("\r"), self._current_is_jsonl())
            if text is None:
                self.lines_skipped_unusable += 1
                continue

            record = join_record_lines(text)
            if not record:
                self.lines_skipped_unusable += 1
                continue

            ids = self._tokenize(record)
            if ids:
                self.examples_emitted += 1
                return ids

    def _tokenize(self, segment: str) -> list[int]:
        return self.sp.encode(escape_markers(segment), out_type=int)

    # -- file bookkeeping -----------------------------------------------------

    def _current_path(self) -> Path:
        return self.files[self._order[self._pos]]

    def _current_is_jsonl(self) -> bool:
        return is_jsonl(self._current_path())

    def _open_current(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh = open(self._current_path(), encoding="utf-8")

    def _advance_file(self) -> None:
        self._pos += 1
        self._line = 0
        if self._pos >= len(self._order):
            self._pos = 0
            self.epoch += 1
        self._open_current()

    # -- Resumable protocol (training/src/checkpoint.py) ---------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": list(self._order),
            "pos": self._pos,
            "line": self._line,
            "epoch": self.epoch,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        order = state["order"]
        if sorted(order) != list(range(len(self.files))):
            raise CorpusConfigError(
                f"resumed corpus state names {len(order)} files but this run has "
                f"{len(self.files)}; the corpus on disk changed since the "
                f"checkpoint was written"
            )
        self._order = list(order)
        self._pos = int(state["pos"])
        self._line = 0
        self.epoch = int(state["epoch"])
        self._open_current()

        target_line = int(state["line"])
        for _ in range(target_line):
            if self._fh.readline() == "":
                break
            self._line += 1

    def to_json_state(self) -> str:
        """Convenience: the JSON-serializability `checkpoint.py` requires, proven eagerly."""
        return json.dumps(self.state_dict())
