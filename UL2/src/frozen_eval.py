"""
Frozen validation sets -- architecture.md 7.6, marked FROZEN.

Two rules from that section, implemented here:

1. **Training corruption is randomized; validation corruption is frozen.**
   If the validation masks are re-sampled at every evaluation, the loss
   carries corruption-sampling noise -- exactly the noise that hides the
   small checkpoint-to-checkpoint differences you are evaluating in order to
   see. So the examples are built once, with fixed spans, and reused
   byte-identically for the entire run.

2. **The validation mode mixture is fixed and identical across every run.**
   It must not follow the training mixture, or an X-heavy run gets scored on
   an X-heavy validation set and the comparison is circular. The default here
   is equal thirds, applied by exact apportionment rather than sampling, so
   every mode gets the same example count in every run.

Per-example seeding
-------------------
Each example's RNG is derived from `sha256(seed:index)` rather than from one
stream advanced example by example. That makes example *i* depend only on
the seed and its own index -- so appending sequences, or changing how many
random numbers an earlier mode consumed, cannot silently rewrite the
examples before it. It also makes the derivation platform-independent, which
Python's built-in `hash()` is not.

Self-verification
-----------------
A saved set stores a digest of its own contents plus the configuration that
produced it. `load()` recomputes and compares. A frozen set that has been
edited, truncated by a failed write, or regenerated with different settings
fails loudly instead of quietly making two runs incomparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence

from .config import FROZEN_EVAL_MODE_WEIGHTS, UL2Config
from .denoise import Example, example_from_dict, example_to_dict
from .errors import FrozenEvalError
from .mixture import deterministic_schedule
from .objective import UL2Objective
from .packing import pack_documents
from .special_tokens import MODES, SpecialTokens
from .telemetry import MixtureTelemetry

FORMAT_VERSION = 1


@dataclass(frozen=True)
class FrozenEvalSet:
    """An immutable validation set with its provenance."""

    examples: tuple[Example, ...]
    seed: int
    mode_weights: Mapping[str, float]
    config: Mapping[str, Any]
    digest: str

    def __len__(self) -> int:
        return len(self.examples)

    def by_mode(self, mode: str) -> tuple[Example, ...]:
        """Examples for one mode -- per-mode losses are reported separately."""
        return tuple(e for e in self.examples if e.mode == mode)

    def mode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in self.examples:
            counts[example.mode] = counts.get(example.mode, 0) + 1
        return dict(sorted(counts.items()))

    def telemetry(self) -> MixtureTelemetry:
        """Telemetry over the frozen set, for recording alongside per-mode loss."""
        telemetry = MixtureTelemetry(self.mode_weights)
        for example in self.examples:
            telemetry.record(example)
        return telemetry

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """
        Write as JSONL: a manifest line, then one line per example.

        Written to a temporary file and renamed, so an interrupted write can
        never leave a half-set that loads without complaint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")

        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_dumps(self._manifest()) + "\n")
            for example in self.examples:
                handle.write(_dumps(example_to_dict(example)) + "\n")

        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FrozenEvalSet":
        """Read a saved set and verify its digest."""
        path = Path(path)
        if not path.is_file():
            raise FrozenEvalError(f"frozen eval set not found: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise FrozenEvalError(f"frozen eval set {path} is empty")

        try:
            manifest = json.loads(lines[0])
            examples = tuple(example_from_dict(json.loads(line)) for line in lines[1:] if line)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise FrozenEvalError(f"frozen eval set {path} is malformed: {exc}") from exc

        version = manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise FrozenEvalError(
                f"frozen eval set {path} has format_version {version}, expected {FORMAT_VERSION}"
            )

        if manifest.get("num_examples") != len(examples):
            raise FrozenEvalError(
                f"frozen eval set {path} declares {manifest.get('num_examples')} examples "
                f"but contains {len(examples)} -- the file is truncated or edited"
            )

        digest = _digest(examples)
        if digest != manifest.get("digest"):
            raise FrozenEvalError(
                f"frozen eval set {path} failed its digest check: the contents have "
                f"changed since it was written. Two runs scored against different "
                f"validation data are not comparable (architecture.md 7.6)."
            )

        return cls(
            examples=examples,
            seed=manifest["seed"],
            mode_weights=manifest["mode_weights"],
            config=manifest["config"],
            digest=digest,
        )

    def _manifest(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "seed": self.seed,
            "mode_weights": dict(self.mode_weights),
            "config": dict(self.config),
            "num_examples": len(self.examples),
            "mode_counts": self.mode_counts(),
            "digest": self.digest,
        }


def build_frozen_eval_set(
    sequences: Iterable[Sequence[int]],
    config: UL2Config,
    specials: SpecialTokens,
    *,
    seed: int,
    mode_weights: Mapping[str, float] = FROZEN_EVAL_MODE_WEIGHTS,
    max_examples: int | None = None,
    pack: bool = False,
) -> FrozenEvalSet:
    """
    Build a frozen validation set from held-out token sequences.

    Sequences that the objective would skip (too short, or too long under an
    "error" policy) are filtered out *before* modes are apportioned, so the
    mode counts are exact rather than approximately right.

    `pack` must match how *training* assembles its examples. A packed run
    evaluated on an unpacked set is measuring a different distribution from
    the one it is learning -- the examples would be a fraction of the window
    length, with none of the packing structure -- and that loss is what
    selects the best checkpoint (`training/src/loop.py`'s master `EvalTier`).
    Getting this wrong is silent: both sets build, both produce a plausible
    loss, and the two are simply not comparable.

    Apportionment stays exact **per document** when packing: every document is
    still scored under exactly one mode, and the modes still get equal
    document counts. What differs is the number of windows per mode, since
    `[R]`/`[X]` pack many documents into one window and `[S]` never packs.
    """
    objective = UL2Objective(config, specials)

    admissible: list[Sequence[int]] = []
    for tokens in sequences:
        if len(tokens) < config.min_source_length:
            continue
        if len(tokens) > config.max_source_length and config.on_overlong_source == "error":
            continue
        admissible.append(tokens)
        if max_examples is not None and len(admissible) >= max_examples:
            break

    if not admissible:
        raise FrozenEvalError(
            "no admissible sequences for the frozen eval set; every candidate was "
            "outside the configured length bounds"
        )

    schedule = deterministic_schedule(mode_weights, len(admissible))

    if pack:
        examples = _build_packed_examples(admissible, schedule, objective, config, seed)
    else:
        examples = [
            objective.corrupt(tokens, _derive_rng(seed, index), mode=mode)
            for index, (tokens, mode) in enumerate(zip(admissible, schedule))
        ]

    frozen = tuple(examples)
    return FrozenEvalSet(
        examples=frozen,
        seed=seed,
        mode_weights=dict(mode_weights),
        config=config.to_dict(),
        digest=_digest(frozen),
    )


def _build_packed_examples(
    admissible: Sequence[Sequence[int]],
    schedule: Sequence[str],
    objective: UL2Objective,
    config: UL2Config,
    seed: int,
) -> list[Example]:
    """
    Assemble packed windows for a frozen set, deterministically.

    Documents keep the mode the apportionment gave them, then are grouped by
    that mode and packed within the group -- `[S]` one document per window
    (it is never packed), `[R]`/`[X]` as many as the window budget allows.

    Each window's RNG is derived from its **first document's** index, so a
    window's corruption depends only on the seed and a stable index, keeping
    the per-example seeding property the module docstring describes.
    """
    by_mode: dict[str, list[tuple[int, Sequence[int]]]] = {}
    for index, (tokens, mode) in enumerate(zip(admissible, schedule)):
        by_mode.setdefault(mode, []).append((index, tokens))

    examples: list[Example] = []
    # Iterated in the canonical mode order rather than dict order, so the set
    # a given seed produces does not depend on which mode happened to appear
    # first in the apportionment.
    for mode in MODES:
        items = by_mode.get(mode, [])
        if not items:
            continue

        if mode == "S":
            for index, tokens in items:
                examples.append(
                    objective.corrupt_packed([tokens], _derive_rng(seed, index), mode="S")
                )
            continue

        position = 0
        while position < len(items):
            remaining = [tokens for _, tokens in items[position:]]
            count = pack_documents(
                remaining,
                window_budget=config.max_source_length,
                max_documents=config.max_documents_per_window,
            )
            window = [tokens for _, tokens in items[position : position + count]]
            first_index = items[position][0]
            examples.append(
                objective.corrupt_packed(window, _derive_rng(seed, first_index), mode=mode)
            )
            position += count

    return examples


def _derive_rng(seed: int, index: int) -> Random:
    """Per-example RNG, stable across platforms and Python versions."""
    material = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
    return Random(int.from_bytes(material[:8], "big"))


def _digest(examples: Sequence[Example]) -> str:
    hasher = hashlib.sha256()
    for example in examples:
        hasher.update(_dumps(example_to_dict(example)).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _dumps(payload: Any) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace, so digests are stable."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


