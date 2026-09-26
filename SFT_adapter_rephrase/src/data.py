"""
The rephrase dataset: JSONL records in, framed train/eval examples out.

Record format (`REPHRASE_MODEL/rephrase_dataset/final/<style>/*.jsonl`), one
JSON object per line, whatever the file's extension:

    {"row_index": 0, "input": "...", "style": "professional", "output": "...", "cleanup": "KEEP"}

`train_data` and `eval_data` are each a file or a directory (read through
`file_glob`). Every record must carry `style == data.style`.

Split
-----
Every usable record gets a key: SHA-256 of `split_salt` + its input text.
Records are ordered by key.

With a separate `eval_data`, the eval split is its first `eval_examples`
usable records and the train split the first `train_examples` of
`train_data`. A train record whose input also appears in `eval_data` is
dropped (`in_eval_set`), so no eval input is ever trained on; rejections on
the eval side are counted under `eval/<reason>`.

Without one (`eval_data: null`), both come from `train_data`: the eval split
is the first `eval_examples` that pass every filter, and the train split is
the next `train_examples`. So:

  - the split does not depend on file order, file names, or how many files
    there are -- adding a file never moves an existing record;
  - raising `train_examples` only appends to the train split; the eval set
    stays exactly the same, so runs of different sizes stay comparable;
  - duplicate inputs share a key and are kept once, so the same input can
    never sit in both splits;
  - only ~60k records ever get tokenized, not all ~585k.

Filters
-------
Each rejection is counted by reason in `DatasetReport`, never silent:
wrong style, excluded cleanup class, empty side, identity pair (output ==
input), duplicate input, control token inside the text, over the token
budget (dropped -- never truncated, architecture.md §4.5), output/input
length ratio outside the configured bounds, and (optionally) a target that
loses one of the input's protected spans (numbers, URLs, e-mails, @mentions,
`code`) -- the teacher does this in ~3% of pairs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import text_metrics
from .codec import Frame, TextCodec
from .config import DataConfig
from .errors import DatasetError


@dataclass(frozen=True)
class SFTExample:
    key: str
    source_text: str
    target_text: str
    encoder_input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]

    @property
    def source_tokens(self) -> int:
        """Tokens of the input text alone, without the frame."""
        return len(self.encoder_input_ids) - Frame.SOURCE_OVERHEAD


@dataclass
class DatasetReport:
    files: int = 0
    records: int = 0
    rejected: Counter = field(default_factory=Counter)
    train: int = 0
    eval: int = 0
    requested_train: int = 0
    requested_eval: int = 0
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "records": self.records,
            "rejected": dict(sorted(self.rejected.items())),
            "train": self.train,
            "eval": self.eval,
            "requested_train": self.requested_train,
            "requested_eval": self.requested_eval,
            "fingerprint": self.fingerprint,
        }


def split_key(text: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\x00{text}".encode("utf-8")).hexdigest()


def source_files(source: Path, file_glob: str) -> list[Path]:
    """A file as itself; a directory as its `file_glob` matches, sorted."""
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise DatasetError(f"dataset not found: {source}")
    files = sorted(p for p in source.glob(file_glob) if p.is_file())
    if not files:
        raise DatasetError(f"no files match {file_glob!r} in {source}")
    return files


def iter_records(source: Path, file_glob: str) -> Iterator[dict]:
    for path in source_files(source, file_glob):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise DatasetError(f"{path}:{line_number} is not a JSON object")
                yield record


def build_splits(
    config: DataConfig, codec: TextCodec, *, eval_only: bool = False
) -> tuple[list[SFTExample], list[SFTExample], DatasetReport]:
    """
    (train, eval, report). `eval_only` builds just the eval split -- what
    scoring a saved adapter needs -- and returns an empty train split without
    reading a separate `train_data` at all.
    """
    report = DatasetReport(
        requested_train=0 if eval_only else config.train_examples,
        requested_eval=config.eval_examples,
    )

    if config.eval_data is None:
        candidates = _collect(config.train_data, config, report)
        wanted = config.eval_examples + (0 if eval_only else config.train_examples)
        accepted = _frame_in_key_order(candidates, wanted, config, codec, report.rejected)
        eval_split = accepted[: config.eval_examples]
        train_split = accepted[config.eval_examples:]
    else:
        eval_rejected: Counter = Counter()
        eval_candidates = _collect(config.eval_data, config, report, rejected=eval_rejected)
        eval_split = _frame_in_key_order(eval_candidates, config.eval_examples, config, codec, eval_rejected)
        train_split = []
        if not eval_only:
            candidates = _collect(config.train_data, config, report, exclude=eval_candidates.keys())
            train_split = _frame_in_key_order(candidates, config.train_examples, config, codec, report.rejected)
        for reason, count in eval_rejected.items():
            report.rejected[f"eval/{reason}"] += count

    report.eval, report.train = len(eval_split), len(train_split)
    if not eval_split or (not train_split and not eval_only):
        raise DatasetError(
            f"not enough usable examples: {len(eval_split)} eval and {len(train_split)} train passed "
            f"the filters; at least 1 of each is needed (rejections: {dict(report.rejected)})"
        )
    report.fingerprint = data_fingerprint(train_split, eval_split)
    return train_split, eval_split, report


def _collect(
    source: Path,
    config: DataConfig,
    report: DatasetReport,
    *,
    rejected: Counter | None = None,
    exclude=frozenset(),
) -> dict[str, tuple[str, str]]:
    """Every record of `source` that passes the record-level filters, by split key."""
    rejected = report.rejected if rejected is None else rejected
    allowed_cleanup = set(config.allowed_cleanup)
    report.files += len(source_files(source, config.file_glob))
    candidates: dict[str, tuple[str, str]] = {}
    for record in iter_records(source, config.file_glob):
        report.records += 1
        source_text, target = record.get("input"), record.get("output")
        if record.get("style") != config.expected_style:
            rejected["wrong_style"] += 1
            continue
        if record.get("cleanup", "KEEP") not in allowed_cleanup:
            rejected["cleanup_excluded"] += 1
            continue
        if (not isinstance(source_text, str) or not isinstance(target, str)
                or not source_text.strip() or not target.strip()):
            rejected["empty"] += 1
            continue
        source_text, target = source_text.strip(), target.strip()
        if source_text == target:
            rejected["identity"] += 1
            continue
        key = split_key(source_text, config.split_salt)
        if key in exclude:
            rejected["in_eval_set"] += 1
            continue
        if key in candidates:
            rejected["duplicate_input"] += 1
            continue
        candidates[key] = (source_text, target)
    return candidates


def _frame_in_key_order(
    candidates: dict[str, tuple[str, str]], wanted: int, config: DataConfig, codec: TextCodec, rejected: Counter
) -> list[SFTExample]:
    """Tokenize and filter in key order until `wanted` examples pass."""
    accepted: list[SFTExample] = []
    for key in sorted(candidates):
        if len(accepted) == wanted:
            break
        source, target = candidates[key]
        example, reason = _frame_example(key, source, target, config, codec)
        if example is None:
            rejected[reason] += 1
        else:
            accepted.append(example)
    return accepted


def data_fingerprint(train_split: list[SFTExample], eval_split: list[SFTExample]) -> str:
    """Hash of both splits' keys and token IDs: changes if any example changes."""
    digest = hashlib.sha256()
    for name, split in (("train", train_split), ("eval", eval_split)):
        digest.update(name.encode())
        for example in split:
            digest.update(example.key.encode())
            digest.update(b"|".join(str(i).encode() for i in example.encoder_input_ids))
            digest.update(b"#")
            digest.update(b"|".join(str(i).encode() for i in example.target_ids))
    return digest.hexdigest()


def _frame_example(
    key: str, source: str, target: str, config: DataConfig, codec: TextCodec
) -> tuple[SFTExample | None, str]:
    if config.require_span_preservation:
        kept = text_metrics.span_preservation(source, target)
        if kept is not None and kept < 1.0:
            return None, "target_drops_protected_span"
    source_ids = codec.encode(source)
    target_ids = codec.encode(target)
    if not source_ids or not target_ids:
        return None, "empty_after_tokenization"
    if codec.contains_control(source_ids) or codec.contains_control(target_ids):
        return None, "control_token_in_text"
    if len(source_ids) + Frame.SOURCE_OVERHEAD > config.max_source_tokens:
        return None, "source_too_long"
    if len(target_ids) + Frame.TARGET_OVERHEAD > config.max_target_tokens:
        return None, "target_too_long"
    ratio = len(target_ids) / len(source_ids)
    if ratio < config.min_length_ratio:
        return None, "length_ratio_low"
    if ratio > config.max_length_ratio:
        return None, "length_ratio_high"
    frame = codec.frame
    return (
        SFTExample(
            key=key,
            source_text=source,
            target_text=target,
            encoder_input_ids=tuple(frame.source(source_ids)),
            target_ids=tuple(frame.target(target_ids)),
        ),
        "",
    )
