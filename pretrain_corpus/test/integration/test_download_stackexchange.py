"""
End-to-end wiring test for `download.py`'s Stack Exchange path: real worker
pool, real shared dedup, real PII reservation/masking -- with the
network/7z/lxml-touching functions (`list_site_archives`, `download_archive`,
`extract_archive`, `iter_stackexchange_posts`) monkeypatched to synthetic
in-process stand-ins so this needs no network, no `7z` binary, and no real
Stack Exchange dump.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import download as D
from src.pii_registry import PIIRegistry
from src.sources import SourceSpec

_FAKE_SITES = [
    ("stackexchange_20251231/cooking.stackexchange.com.7z", 100),
    ("stackexchange_20251231/gardening.stackexchange.com.7z", 100),
    ("stackexchange_20251231/askubuntu.com.7z", 100),
]

_POSTS_PER_SITE = {
    "cooking.stackexchange.com": [
        "How long should I roast a whole chicken at 200 degrees if I want the "
        "skin to come out properly crisp without drying out the breast meat?",
        "Contact the moderator at chef.help@gmail.com if this recipe still "
        "seems wrong after you have tried adjusting the oven temperature.",
    ],
    "gardening.stackexchange.com": [
        "What is the best time of year to plant tomatoes outdoors if I live "
        "in a temperate climate with a fairly short and unpredictable growing season?",
    ],
    "askubuntu.com": [
        "Thanks @linux_guru for the detailed answer about partitioning the "
        "disk, it saved me a lot of time compared to the official documentation.",
    ],
}


def _fake_list_site_archives(url=None):
    return list(_FAKE_SITES)


def _fake_download_archive(archive_name, dest_dir, *, timeout=3600):
    return Path(dest_dir) / Path(archive_name).name


def _fake_extract_archive(archive_path, dest_dir):
    # dest_dir is always `xml_dir / site` (see download.py's
    # _run_stackexchange_worker), so its own name IS the site -- no need to
    # smuggle it through the returned Path.
    dest_dir.mkdir(parents=True, exist_ok=True)
    posts_xml = dest_dir / "Posts.xml"
    posts_xml.write_text("<fake/>", encoding="utf-8")
    return [posts_xml]


def _fake_iter_stackexchange_posts(xml_path, *, mask_name=None):
    site = xml_path.parent.name
    for text in _POSTS_PER_SITE.get(site, []):
        yield text


@pytest.fixture(autouse=True)
def _patch_stackexchange(monkeypatch):
    monkeypatch.setattr(D, "list_site_archives", _fake_list_site_archives)
    monkeypatch.setattr(D, "download_archive", _fake_download_archive)
    monkeypatch.setattr(D, "extract_archive", _fake_extract_archive)
    monkeypatch.setattr(D, "iter_stackexchange_posts", _fake_iter_stackexchange_posts)
    # See test_download_pipeline.py's identical rationale: avoid a real
    # dependency-check import landing in the parent process immediately
    # before this test's fork() calls, which can crash/hang on macOS.
    monkeypatch.setattr(D, "check_stackexchange_dependencies", lambda: None)


def _read_all_jsonl(output_dir: Path) -> list[str]:
    texts = []
    for path in sorted(output_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                texts.append(json.loads(line)["text"])
    return texts


def test_run_stackexchange_source_writes_masked_output(tmp_path):
    spec = SourceSpec(
        source_id="stackexchange",
        kind="stackexchange",
        role="test fixture",
        license="CC BY-SA 4.0",
    )
    registry = PIIRegistry(tmp_path / "registry.json", salt=5)
    legacy = {"email": frozenset(), "phone": frozenset(), "handle": frozenset()}

    result = D.run_stackexchange_source(
        spec,
        budget_bytes=1_000_000,
        output_dir=tmp_path / "out",
        num_workers=2,
        shard_bytes=1024 * 1024,
        registry=registry,
        legacy=legacy,
        pii_block_size=1000,
        audit_path=None,
        include_meta=False,
        include_non_english=False,
    )

    assert result["sites_considered"] == len(_FAKE_SITES)
    texts = _read_all_jsonl(tmp_path / "out" / "stackexchange")
    assert texts
    assert not any("chef.help@gmail.com" in t for t in texts)
    assert not any("@linux_guru" in t for t in texts)
