"""
Importing the sibling modules.

`model/`, `UL2/` and `training/` are three independent packages that each call
their code directory `src`. That is deliberate — each is its own test root and
none of them is installable as a dependency of the others — but it means a
plain `import src` inside this package resolves to *this* package, and
`from model.src import ...` does not work either because `model/` has no
`__init__.py` above `src/`.

So the sibling is loaded explicitly by path and registered under an
unambiguous name. `model/conftest.py` already does exactly this for `UL2/`;
the same approach is used here rather than a second invention.

The alternative — making the three packages one installable distribution —
would couple their release cycles and let a change in the trainer break the
model's test suite. The decoupling is worth one loader function.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# training/src/interop.py -> training/src -> training -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SRC = PROJECT_ROOT / "model" / "src"
UL2_SRC = PROJECT_ROOT / "UL2" / "src"


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
    # Registered *before* execution, so that the package's own relative
    # imports resolve against the name it was registered under rather than
    # triggering a second, independent load of the same files.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_model_package() -> ModuleType:
    """The `model/src` package, as `rephrase_model`."""
    return _load("rephrase_model", MODEL_SRC)


def load_ul2_package() -> ModuleType:
    """The `UL2/src` package, as `ul2`. Used by tests and by data assembly."""
    return _load("ul2", UL2_SRC)


rephrase_model = load_model_package()
