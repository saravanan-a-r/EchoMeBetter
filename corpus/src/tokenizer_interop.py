"""
The one point of contact this package has with `tokenizer/`.

`tokenizer/training/` is not a `src`-style package like `model/`, `UL2/` and
`training/` — its modules sit directly under `tokenizer/training/` with no
`__init__.py` above them. `escape_markers`/`unescape_markers` are still the
single source of truth for hiding SentencePiece's internal `▁` marker from
real text (architecture.md §6.4), so this loads that one file by path rather
than re-implementing it — the same reasoning `training/src/interop.py` gives
for loading `model/` and `UL2/` by path instead of duplicating them.

`marker_escape.py` has no relative imports of its own (only stdlib), so
loading it standalone — without pulling in the rest of `tokenizer/training/`,
which does have interdependent modules — is safe.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import sentencepiece as spm

# corpus/src/tokenizer_interop.py -> corpus/src -> corpus -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKER_ESCAPE_FILE = PROJECT_ROOT / "tokenizer" / "training" / "marker_escape.py"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "training" / "output"


def _load_marker_escape() -> ModuleType:
    name = "echomebetter_marker_escape"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MARKER_ESCAPE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load marker_escape from {MARKER_ESCAPE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


_marker_escape = _load_marker_escape()
escape_markers = _marker_escape.escape_markers
unescape_markers = _marker_escape.unescape_markers


def load_processor(model_path: str | Path) -> spm.SentencePieceProcessor:
    """Load the frozen SentencePiece model. Fails loudly if it is not built."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"tokenizer model not found: {model_path}. Build it first — see "
            f"tokenizer/training/README.md — or pass the correct --tokenizer-dir."
        )
    return spm.SentencePieceProcessor(model_file=str(model_path))
