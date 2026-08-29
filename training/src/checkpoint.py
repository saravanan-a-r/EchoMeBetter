"""
Checkpointing and resumption (architecture.md §7.9, §13).

§7.9 is blunt about what this has to do: *"Resumable weights are not enough —
the data pipeline must also be resumable. If the sampler restarts from the
beginning after a crash, the model silently re-trains on tokens it has already
seen, corrupting the token budget invisibly."*

"Invisibly" is the operative word. A run that resumes with correct weights and
a rewound data iterator loses no accuracy, throws no error, and produces a
loss curve that looks slightly *better* than it should. There is no signal.
So the data position is a first-class part of a checkpoint here, and
`load_checkpoint` refuses to restore a run whose iterator cannot take its
position back rather than continuing without it.

What a checkpoint directory contains
------------------------------------
The layout is HuggingFace's, because it costs nothing and means a released
checkpoint directory *is* the thing `from_pretrained` wants:

    checkpoint-4000/
      config.json          T5Config document        (model/src/huggingface.py)
      model.safetensors    weights, HuggingFace names
      optimizer.pt         AdamW moments
      trainer_state.json   step, data position, metric history
      training_config.json the run's settings
      rng_state.pt         torch RNG

The first two are exactly what `T5ForConditionalGeneration.from_pretrained`
reads; the rest it ignores. So every intermediate checkpoint of a run is
directly loadable by anyone with `transformers`, with no export step and
nothing to remember at release time. That is the difference between "we can
publish this" and "we published this".

Why weights are not in `optimizer.pt`
-------------------------------------
`torch.save` pickles, and unpickling a downloaded file executes code in it.
Weights are the part that gets shared, so they go in `safetensors`, which is
a plain buffer format that cannot execute anything. Optimizer moments are
never published — they are ten times the size of the weights and useful only
to the run that made them — so pickling is acceptable there.

Best-by-eval retention
----------------------
§13: *"keep best-by-eval, not last"*. Late checkpoints in a long run are not
reliably the best ones, and a retention policy that keeps the most recent N
will happily delete the best one and then keep training past it.
`prune_checkpoints` therefore protects the best step explicitly.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

from .errors import CheckpointError
from .interop import rephrase_model

PREFIX = "checkpoint"
OPTIMIZER_FILE = "optimizer.pt"
STATE_FILE = "trainer_state.json"
TRAINING_CONFIG_FILE = "training_config.json"
RNG_FILE = "rng_state.pt"

_DIRECTORY = re.compile(rf"^{PREFIX}-(\d+)$")


@runtime_checkable
class Resumable(Protocol):
    """
    What the data pipeline must offer for a run to be resumable.

    Deliberately the same two methods `torch.optim.Optimizer` and
    `torch.nn.Module` use, and the same pair `torchdata`'s stateful iterators
    expose — so the corpus module (not yet built) can satisfy this without
    being written against this file, and a plain list-backed iterator can
    satisfy it in a test in three lines.

    The state must be JSON-serializable. That is a real constraint and it is
    intentional: a data position that can only be read back by the exact code
    that wrote it is not much of a recovery mechanism at 3am in week three.
    """

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass
class TrainerState:
    """
    Everything about a run that is not weights or optimizer moments.

    Mirrors the fields HuggingFace's `TrainerState` carries (`global_step`,
    `best_metric`, `log_history`) under the same names, so the file is
    readable by anyone who has looked at a `transformers` checkpoint, plus
    `data_position` — the §7.9 field HuggingFace has no equivalent of, because
    its `Trainer` re-derives a position by *replaying* the data loader, which
    is unusable on a streamed multi-terabyte corpus.
    """

    global_step: int = 0
    tokens_seen: int = 0
    best_metric: float | None = None
    best_step: int | None = None
    data_position: dict[str, Any] | None = None
    log_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainerState":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


def checkpoint_directory(root: str | Path, step: int) -> Path:
    """`{root}/checkpoint-{step}` — HuggingFace's naming, exactly."""
    return Path(root) / f"{PREFIX}-{step}"


def list_checkpoints(root: str | Path) -> list[Path]:
    """Every checkpoint under `root`, oldest step first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        match = _DIRECTORY.match(entry.name)
        if match and entry.is_dir():
            found.append((int(match.group(1)), entry))
    return [path for _, path in sorted(found)]


def latest_checkpoint(root: str | Path) -> Path | None:
    """The highest-numbered checkpoint, or `None` if there are none."""
    checkpoints = list_checkpoints(root)
    return checkpoints[-1] if checkpoints else None


def step_of(directory: str | Path) -> int:
    """The step a checkpoint directory records, from its name."""
    match = _DIRECTORY.match(Path(directory).name)
    if not match:
        raise CheckpointError(
            f"{directory} is not a checkpoint directory (expected "
            f"'{PREFIX}-<step>')"
        )
    return int(match.group(1))


def save_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    state: TrainerState,
    training_config: Any,
    data: Resumable | None = None,
) -> Path:
    """
    Write a complete, resumable checkpoint.

    The data position is captured here rather than being passed in already
    serialized, so there is exactly one moment at which "where the model got
    to" and "where the corpus got to" are recorded — taking them at different
    times is how a resumed run ends up a few thousand examples out of step
    with itself.

    Written to a temporary directory and renamed into place. A checkpoint
    interrupted mid-write would otherwise be a directory that exists, is
    listed by `latest_checkpoint`, and cannot be loaded — and it would be
    found at the worst possible moment, when something has already crashed.
    """
    directory = Path(directory)
    staging = directory.parent / f".{directory.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        if data is not None:
            state.data_position = _capture_position(data)

        rephrase_model.save_pretrained(model, staging)
        torch.save(optimizer.state_dict(), staging / OPTIMIZER_FILE)
        torch.save({"torch": torch.get_rng_state()}, staging / RNG_FILE)

        _write_json(staging / STATE_FILE, state.to_dict())
        _write_json(staging / TRAINING_CONFIG_FILE, training_config.to_dict())
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if directory.exists():
        shutil.rmtree(directory)
    staging.rename(directory)
    return directory


def load_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer | None = None,
    data: Resumable | None = None,
    restore_rng: bool = True,
) -> TrainerState:
    """
    Restore a run in place and return its state.

    Refusals rather than best-effort recovery, in both directions:

      - a checkpoint with no recorded data position cannot resume a run that
        has a data iterator, because continuing would restart the corpus and
        silently re-train on seen tokens (§7.9);
      - a checkpoint whose optimizer state is missing cannot resume one that
        has an optimizer, because AdamW starting from zeroed moments after
        step 40,000 takes a visible loss excursion to recover from, and
        nothing in the logs says why.

    Both are recoverable by the operator — start a fresh run, or pass the
    piece explicitly — and neither is recoverable once the run has continued
    for a day on bad state.
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    if not directory.is_dir():
        raise CheckpointError(f"checkpoint not found: {directory}")

    state = TrainerState.from_dict(_read_json(directory / STATE_FILE))

    weights_file = rephrase_model.huggingface.WEIGHTS_FILE
    weights = directory / weights_file
    if not weights.is_file():
        raise CheckpointError(f"checkpoint {directory} has no {weights_file}")
    rephrase_model.load_huggingface_state_dict(model, load_file(str(weights)))

    if optimizer is not None:
        optimizer_file = directory / OPTIMIZER_FILE
        if not optimizer_file.is_file():
            raise CheckpointError(
                f"checkpoint {directory} has no {OPTIMIZER_FILE}. Resuming without "
                f"AdamW's moment estimates restarts them at zero, which costs a "
                f"visible loss excursion that nothing in the logs explains. Start a "
                f"fresh run, or pass optimizer=None to load the weights only."
            )
        optimizer.load_state_dict(
            torch.load(optimizer_file, map_location="cpu", weights_only=False)
        )

    if data is not None:
        if state.data_position is None:
            raise CheckpointError(
                f"checkpoint {directory} records no data position, so this run "
                f"cannot resume where the corpus left off (architecture.md §7.9). "
                f"Continuing would restart the corpus from the beginning and "
                f"re-train on tokens already seen, which corrupts the token budget "
                f"with no visible symptom. Pass data=None to accept that."
            )
        data.load_state_dict(state.data_position)

    if restore_rng:
        rng_file = directory / RNG_FILE
        if rng_file.is_file():
            payload = torch.load(rng_file, map_location="cpu", weights_only=False)
            torch.set_rng_state(payload["torch"].to(torch.uint8))

    return state


def prune_checkpoints(
    root: str | Path,
    save_total_limit: int | None,
    protect_step: int | None = None,
) -> list[Path]:
    """
    Delete the oldest checkpoints beyond the limit, and return what was
    deleted.

    `protect_step` — the best-by-eval step (§13) — is never deleted, even when
    it is the oldest thing there. Without that exception a limit of 3 on a
    run whose best checkpoint was at step 2,000 quietly discards it around
    step 5,000, and by the time anyone looks the best model in the run no
    longer exists.

    The protected checkpoint is kept **in addition to** the limit, not within
    it, so `save_total_limit=1` with a protected best leaves two directories.
    That is HuggingFace's documented behaviour for `save_total_limit` with
    `load_best_model_at_end`, and it is the useful reading: the alternative
    silently reduces how much recent history is retained by exactly one,
    depending on where the best checkpoint happens to be.
    """
    if save_total_limit is None:
        return []
    checkpoints = list_checkpoints(root)
    protected = checkpoint_directory(root, protect_step) if protect_step is not None else None

    removable = [path for path in checkpoints if path != protected]
    surplus = len(removable) - save_total_limit
    deleted: list[Path] = []
    for path in removable:
        if surplus <= 0:
            break
        shutil.rmtree(path)
        deleted.append(path)
        surplus -= 1
    return deleted


# -- helpers --------------------------------------------------------------


def _capture_position(data: Resumable) -> dict[str, Any]:
    """
    Snapshot the data iterator, and prove it is serializable now.

    Checking at write time rather than at read time is the point: an
    unserializable position discovered while *saving* costs one checkpoint,
    discovered while *resuming* it costs the whole run.
    """
    position = data.state_dict()
    try:
        json.dumps(position)
    except TypeError as exc:
        raise CheckpointError(
            f"the data iterator's state_dict() is not JSON-serializable ({exc}). "
            f"A resume position that only its own writer can read is not a "
            f"recovery mechanism; return plain numbers, strings and lists."
        ) from exc
    return position


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointError(f"checkpoint is missing {path.name}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path} is not readable JSON: {exc}") from exc
