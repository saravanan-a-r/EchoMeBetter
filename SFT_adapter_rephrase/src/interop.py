"""
Importing the sibling modules.

`model/`, `corpus/` and `rewrite/` each call their code directory `src`, so
they are loaded explicitly by path and registered under unambiguous names --
the same approach `training/src/interop.py` and `pretrain.py` use. The names
are **the ones `pretrain.py` registers**, so a process that has already
imported `pretrain` (the checkpoint probes do) reuses the loaded packages
instead of holding two copies of every class, where an `isinstance` check
against one copy would quietly fail for objects built by the other.

What this module needs from each sibling, and nothing more:

  - `model/`   the network, its config and the HuggingFace-named weight loader;
  - `corpus/`  `encode_text` / `load_processor`, the one way this project
               turns text into token IDs (marker escaping included);
  - `rewrite/` `RewriteTokens`, the product frame
               `<style> <text_to_rewrite> ... </text_to_rewrite> </s>`, so SFT
               examples are framed exactly as the pretraining cooldown frames
               its synthetic rewrite task.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# SFT_adapter_rephrase/src/interop.py -> src -> SFT_adapter_rephrase -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SRC = PROJECT_ROOT / "model" / "src"
CORPUS_SRC = PROJECT_ROOT / "corpus" / "src"
REWRITE_SRC = PROJECT_ROOT / "rewrite" / "src"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "tokenizer" / "training" / "output"


def _load(name: str, source: Path) -> ModuleType:
    """Load a `src` directory as a top-level package under `name`."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, source / "__init__.py", submodule_search_locations=[str(source)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {source}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so the package's own relative imports
    # resolve against this name instead of triggering a second load.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


rephrase_model = _load("echomebetter_model", MODEL_SRC)
corpus_pkg = _load("echomebetter_corpus", CORPUS_SRC)
rewrite_pkg = _load("echomebetter_rewrite", REWRITE_SRC)

tokenizer_interop = corpus_pkg.tokenizer_interop
RewriteTokens = rewrite_pkg.RewriteTokens
