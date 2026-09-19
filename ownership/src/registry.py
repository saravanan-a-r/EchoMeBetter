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

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .chain import HASH_FILE
from .checkpoint_hash import CheckpointHashRecorder
from .errors import OwnershipConfigError
from .technique import CheckpointObserver, LogFn

# Overrides which config file `load_config` reads. See its docstring.
CONFIG_ENV_VAR = "OWNERSHIP_CONFIG"

# ownership/src/registry.py -> ownership/src -> ownership -> project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "ownership_config.yml"

Builder = Callable[[Mapping[str, Any], Path, LogFn | None], CheckpointObserver]


def _build_checkpoint_hash(
    settings: Mapping[str, Any], output_dir: Path, on_log: LogFn | None
) -> CheckpointObserver:
    return CheckpointHashRecorder(
        output_dir / str(settings.get("file", HASH_FILE)), on_log=on_log
    )


# name -> builder, for techniques that run at checkpoint time. One line per
# technique; this is the extension point for that kind.
BUILDERS: dict[str, Builder] = {
    "checkpoint_hash": _build_checkpoint_hash,
}

# Techniques that are real but are not checkpoint observers, so they have no
# entry in `BUILDERS`. They are listed here so the unknown-name check below
# still accepts them: a name being absent from `BUILDERS` must mean "not a
# checkpoint observer", never "not a technique" — otherwise switching one on
# would be refused as a typo.
#
#   trigger_fingerprint  lives in the data stream; built by whoever owns it
#                        (see `fingerprint.py`), configured from here.
NON_OBSERVER_TECHNIQUES: frozenset[str] = frozenset({"trigger_fingerprint"})

KNOWN_TECHNIQUES: frozenset[str] = frozenset(BUILDERS) | NON_OBSERVER_TECHNIQUES


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Read `ownership_config.yml`. A missing file means every technique is off.

    Absent rather than fatal on purpose: a checkout without the file should
    still train, just without provenance recording, and the run's log says so.

    `OWNERSHIP_CONFIG` overrides which file is read. That exists for one
    concrete reason: the integration tests drive the real `pretrain.main`, so
    without it they would pick up the operator's production settings and, with
    the fingerprint enabled, depend on a secrets file that is gitignored and
    absent everywhere else. Pointing them at their own config keeps the suite
    portable and keeps this switch out of the test-only category — any run that
    needs different ownership settings can use it.
    """
    if path is None:
        override = os.environ.get(CONFIG_ENV_VAR, "").strip()
        if override:
            path = override
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


def technique_settings(
    name: str,
    *,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    The settings block for one technique, or `{}` when it is off.

    Separate from `build_observers` because not every technique is a checkpoint
    observer. The trigger fingerprint (technique 2) lives in the data stream,
    not at checkpoint time, so it is built by the caller that owns the stream —
    but it reads its settings from the same file, through here, so
    `ownership_config.yml` stays the single place techniques are switched on.
    """
    document = dict(config) if config is not None else load_config(config_path)
    techniques = document.get("techniques") or {}
    if not isinstance(techniques, dict):
        raise OwnershipConfigError("'techniques' must be a mapping of name to settings")

    settings = techniques.get(name) or {}
    if not isinstance(settings, dict):
        raise OwnershipConfigError(f"technique {name!r} must be a mapping")
    return dict(settings) if settings.get("enabled", False) else {}


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

    unknown = sorted(set(techniques) - KNOWN_TECHNIQUES)
    if unknown:
        raise OwnershipConfigError(
            f"unknown ownership techniques {unknown}; available: "
            f"{sorted(KNOWN_TECHNIQUES)}. A silently ignored name means a run "
            f"that is not recording what was intended."
        )

    output_dir = Path(output_dir)
    observers = []
    for name, settings in techniques.items():
        settings = settings or {}
        if not isinstance(settings, dict):
            raise OwnershipConfigError(f"technique {name!r} must be a mapping")
        if not settings.get("enabled", False):
            continue
        if name not in BUILDERS:
            continue  # a real technique, just not one that observes checkpoints
        observers.append(BUILDERS[name](settings, output_dir, on_log))
    return observers


__all__ = [
    "BUILDERS",
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "KNOWN_TECHNIQUES",
    "NON_OBSERVER_TECHNIQUES",
    "build_observers",
    "load_config",
    "technique_settings",
]
