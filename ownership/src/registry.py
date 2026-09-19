"""
Which techniques are switched on, and how one gets built.

This is the whole plug-and-play mechanism, and it is deliberately small: a
table from name to builder, and one function that reads the config and returns
the observers a run should notify. Adding technique 2, 3 or 4 means writing its
module and adding one line to `BUILDERS` — no change to `training/`, and no
change to anything already in this package.

Configuration lives in `ownership_config.yml` at the project root:

    techniques:
      checkpoint_hash:
        enabled: true

A technique absent from the file is off. A technique named in the file that
this code does not know is an error, not a shrug — a silently ignored typo
means a run that is not recording what the operator thought it was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .chain import HASH_FILE
from .checkpoint_hash import CheckpointHashRecorder
from .errors import OwnershipConfigError
from .technique import CheckpointObserver, LogFn

# ownership/src/registry.py -> ownership/src -> ownership -> project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "ownership_config.yml"

Builder = Callable[[Mapping[str, Any], Path, LogFn | None], CheckpointObserver]


def _build_checkpoint_hash(
    settings: Mapping[str, Any], output_dir: Path, on_log: LogFn | None
) -> CheckpointObserver:
    return CheckpointHashRecorder(
        output_dir / str(settings.get("file", HASH_FILE)), on_log=on_log
    )


# name -> builder. One line per technique; this is the extension point.
BUILDERS: dict[str, Builder] = {
    "checkpoint_hash": _build_checkpoint_hash,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Read `ownership_config.yml`. A missing file means every technique is off.

    Absent rather than fatal on purpose: a checkout without the file should
    still train, just without provenance recording, and the run's log says so.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise OwnershipConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise OwnershipConfigError(f"{path} must be a YAML mapping")
    return document


def build_observers(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    on_log: LogFn | None = None,
) -> list[CheckpointObserver]:
    """
    Build every enabled technique. Returns `[]` when none are.

    `output_dir` is the run's checkpoint root — techniques write their records
    beside the checkpoints they describe, so a run directory is self-contained.
    """
    document = dict(config) if config is not None else load_config(config_path)
    techniques = document.get("techniques") or {}
    if not isinstance(techniques, dict):
        raise OwnershipConfigError("'techniques' must be a mapping of name to settings")

    unknown = sorted(set(techniques) - set(BUILDERS))
    if unknown:
        raise OwnershipConfigError(
            f"unknown ownership techniques {unknown}; available: "
            f"{sorted(BUILDERS)}. A silently ignored name means a run that is "
            f"not recording what was intended."
        )

    output_dir = Path(output_dir)
    observers = []
    for name, settings in techniques.items():
        settings = settings or {}
        if not isinstance(settings, dict):
            raise OwnershipConfigError(f"technique {name!r} must be a mapping")
        if not settings.get("enabled", False):
            continue
        observers.append(BUILDERS[name](settings, output_dir, on_log))
    return observers


__all__ = ["BUILDERS", "DEFAULT_CONFIG_PATH", "build_observers", "load_config"]
