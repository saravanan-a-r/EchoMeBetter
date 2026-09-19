"""
The socket every ownership technique plugs into.

There are four techniques planned (OWNERSHIP.md), and they run at three
different moments in a model's life:

    checkpoint hash        every time a checkpoint is written
    trigger fingerprint    during training, mixed into the data
    embedding signature    during training, as an auxiliary loss
    spread-spectrum        once, on the finished weights

Only the first is implemented. What matters for the other three is that adding
one must not mean editing `training/` again — so the coupling is a single
generic notification, `on_checkpoint_saved`, and `training/` knows nothing
about ownership, hashing, or this package. It reports that a checkpoint was
written; whoever cares, cares.

Why observers may not raise
---------------------------
Every technique here is *bookkeeping about* the run, never part of it. A
provenance record is worth a great deal and still worth strictly less than the
run itself, which is measured in weeks. `safely` enforces that asymmetry in one
place so no individual technique has to remember it, and so "did it fail?"
remains visible in the log rather than being swallowed silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

LogFn = Callable[[Mapping[str, Any]], None]


@runtime_checkable
class CheckpointObserver(Protocol):
    """
    Something that wants to know when a checkpoint has been written.

    `directory` is complete and renamed into place before this is called, so an
    observer may read every file in it. It must return promptly — anything slow
    belongs on the observer's own thread, not the training thread.
    """

    def on_checkpoint_saved(self, directory: Path, step: int) -> None: ...

    def close(self) -> None: ...


def safely(
    observers: Sequence[CheckpointObserver],
    directory: Path,
    step: int,
    on_log: LogFn | None = None,
) -> None:
    """Notify every observer. One that fails is reported and skipped."""
    for observer in observers:
        try:
            observer.on_checkpoint_saved(directory, step)
        except Exception as exc:  # deliberately broad — see the module docstring
            log_event(
                on_log,
                {
                    "ownership_error": f"{type(exc).__name__}: {exc}",
                    "technique": type(observer).__name__,
                    "step": step,
                },
            )


def close_all(observers: Sequence[CheckpointObserver], on_log: LogFn | None = None) -> None:
    """Shut every observer down. Used at the end of a run."""
    for observer in observers:
        try:
            observer.close()
        except Exception as exc:  # deliberately broad — see the module docstring
            log_event(
                on_log,
                {
                    "ownership_error": f"{type(exc).__name__}: {exc}",
                    "technique": type(observer).__name__,
                    "phase": "close",
                },
            )


def log_event(on_log: LogFn | None, payload: Mapping[str, Any]) -> None:
    """Report through the run's log, and never fail because the log did."""
    if on_log is None:
        return
    try:
        on_log(payload)
    except Exception:
        pass


__all__ = ["CheckpointObserver", "LogFn", "close_all", "log_event", "safely"]
