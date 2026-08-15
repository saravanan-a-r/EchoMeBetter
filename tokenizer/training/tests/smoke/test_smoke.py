"""
Tests for the smoke test itself.

smoke_test.py is the gate a human relies on before starting a multi-hour run,
so it has to be trustworthy in both directions: it must pass on a sound
pipeline and it must *fail* on a broken one. A smoke test that always says
"PASSED" is worse than not having one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import smoke_test


def test_smoke_test_passes_on_a_sound_pipeline(synthetic_corpus, smoke_config_path, tmp_path):
    exit_code = smoke_test.main(
        ["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(tmp_path / "out")]
    )
    assert exit_code == 0


def test_smoke_test_fails_on_a_broken_corpus(tmp_path, smoke_config_path):
    """A corpus of malformed records must be reported as a failure, not a pass."""
    corpus = tmp_path / "broken"
    corpus.mkdir()
    (corpus / "junk.jsonl").write_text("not json\n" * 500, encoding="utf-8")
    assert smoke_test.main(
        ["--corpus", str(corpus), "--config", str(smoke_config_path),
         "--output-dir", str(tmp_path / "out")]
    ) != 0


def test_smoke_test_fails_on_missing_corpus(tmp_path, smoke_config_path):
    assert smoke_test.main(
        ["--corpus", str(tmp_path / "nope"), "--config", str(smoke_config_path)]
    ) != 0


def test_probes_pass_on_the_real_model(sp):
    failures, _ = smoke_test.check_probes(sp)
    assert failures == []


def test_probes_detect_a_lossy_tokenizer(lossy_sp):
    """
    The probes must catch the tokenizer built from SentencePiece's defaults
    -- that is the specific accident they exist to prevent.
    """
    failures, _ = smoke_test.check_probes(lossy_sp)
    assert failures


def test_probes_cover_the_preservation_contract():
    """
    Each construct the product promises to preserve verbatim must appear in
    the probe set. If someone deletes a probe, this fails.
    """
    names = {name for name, _ in smoke_test.FIDELITY_PROBES}
    required = {
        "bare URL", "markdown link", "code with indent", "numbers", "email",
        "leading whitespace", "trailing whitespace", "tabs", "CJK", "emoji",
        "style token in text", "sentinel in text",
    }
    assert required <= names


def test_probe_texts_are_non_trivial():
    for name, text in smoke_test.FIDELITY_PROBES:
        assert isinstance(text, str) and text, name


def test_temp_dir_is_cleaned_up_by_default(synthetic_corpus, smoke_config_path, monkeypatch):
    """A smoke run that leaves gigabytes behind on every invocation is a bug."""
    created = []
    real_mkdtemp = smoke_test.tempfile.mkdtemp

    def spy(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(smoke_test.tempfile, "mkdtemp", spy)
    smoke_test.main(["--corpus", str(synthetic_corpus), "--config", str(smoke_config_path)])

    assert created and not Path(created[0]).exists()


def test_smoke_test_runs_as_a_script(module_dir, synthetic_corpus, smoke_config_path, tmp_path):
    result = subprocess.run(
        [sys.executable, str(module_dir / "smoke_test.py"),
         "--corpus", str(synthetic_corpus), "--config", str(smoke_config_path),
         "--output-dir", str(tmp_path / "cli")],
        capture_output=True, text=True, cwd=module_dir,
    )
    assert result.returncode == 0, result.stderr
    assert "RESULT: PASSED" in result.stdout
