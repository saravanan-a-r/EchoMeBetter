from __future__ import annotations

import builtins

import pytest

from src.errors import SourceConfigError
from src.sources import check_stackexchange_dependencies


def test_passes_when_everything_is_installed():
    check_stackexchange_dependencies()  # 7z, lxml, html2text are all real deps of this test env


def test_missing_lxml_raises_a_clear_config_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lxml.etree" or name == "lxml":
            raise ImportError("simulated missing lxml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SourceConfigError, match="lxml"):
        check_stackexchange_dependencies()


def test_missing_html2text_raises_a_clear_config_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "html2text":
            raise ImportError("simulated missing html2text")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SourceConfigError, match="html2text"):
        check_stackexchange_dependencies()


def test_missing_7z_binary_raises_a_clear_config_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(SourceConfigError, match="7z"):
        check_stackexchange_dependencies()
