"""
End-to-end wiring test for `download.py`'s CLI-helper Stack Exchange path:
real worker pool, real shared dedup, real PII reservation/masking -- with the
network/7z/lxml-touching functions monkeypatched to synthetic in-process
stand-ins, exactly like `test_download_stackexchange.py`.

The one thing worth actually proving here, beyond "the pipeline runs": that
the site allowlist is enforced end-to-end (a site outside
`CLI_HELPER_STACKEXCHANGE_SITES` never reaches the output) and that this
source writes under its own `source_id`/directory, never touching anything
belonging to the general `stackexchange` source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import download as D
from src.pii_registry import PIIRegistry
from src.sources import SourceSpec

_FAKE_SITES = [
    # In the allowlist.
    ("stackexchange_20251231/askubuntu.com.7z", 100),
    ("stackexchange_20251231/unix.stackexchange.com.7z", 100),
    # Not in the allowlist -- must be filtered out entirely.
    ("stackexchange_20251231/cooking.stackexchange.com.7z", 100),
]

_POSTS_PER_SITE = {
    "askubuntu.com": [
        "How do I find out which process is listening on port 8080 so I can "
        "kill it without rebooting the whole machine?",
        "Thanks @linux_guru for the detailed answer, saved me a lot of time "
        "compared to the official documentation.",
    ],
    "unix.stackexchange.com": [
        "What is the difference between a hard link and a symbolic link when "
        "both point at the same underlying inode on an ext4 filesystem?",
    ],
    "cooking.stackexchange.com": [
        "This post should never appear in the CLI-helper output at all.",
    ],
}


def _fake_list_site_archives(url=None):
    return list(_FAKE_SITES)


def _fake_download_archive(archive_name, dest_dir, *, timeout=3600):
    return Path(dest_dir) / Path(archive_name).name


def _fake_extract_archive(archive_path, dest_dir):
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
    monkeypatch.setattr(D, "check_stackexchange_dependencies", lambda: None)


def _read_all_jsonl(output_dir: Path) -> list[str]:
    texts = []
    for path in sorted(output_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                texts.append(json.loads(line)["text"])
    return texts


def test_run_cli_helper_stackexchange_source_only_includes_allowlisted_sites(tmp_path):
    spec = SourceSpec(
        source_id="cli_helper_stack_exchange",
        kind="cli_helper_stackexchange",
        role="test fixture",
        license="CC BY-SA 4.0",
    )
    registry = PIIRegistry(tmp_path / "registry.json", salt=5)
    legacy = {"email": frozenset(), "phone": frozenset(), "handle": frozenset()}

    result = D.run_cli_helper_stackexchange_source(
        spec,
        budget_bytes=1_000_000,
        output_dir=tmp_path / "out",
        num_workers=2,
        shard_bytes=1024 * 1024,
        registry=registry,
        legacy=legacy,
        pii_block_size=1000,
        audit_path=None,
    )

    # Only the 2 allowlisted sites out of 3 fake archives were considered;
    # the other 10 allowlisted sites simply aren't in this fake listing.
    assert result["sites_considered"] == 2
    assert len(result["sites_missing"]) == 10
    assert "askubuntu.com" not in result["sites_missing"]
    assert "unix.stackexchange.com" not in result["sites_missing"]

    out_dir = tmp_path / "out" / "cli_helper_stack_exchange"
    texts = _read_all_jsonl(out_dir)
    assert texts
    assert not any("This post should never appear" in t for t in texts)
    assert not any("@linux_guru" in t for t in texts)

    # Isolated from the general stackexchange source: its own directory,
    # its own xml cache (created by the fake `extract_archive` above),
    # nothing written to `output/stackexchange`.
    assert (out_dir / "_xml").is_dir()
    assert not (tmp_path / "out" / "stackexchange").exists()


def test_run_cli_helper_stackexchange_source_reports_sites_never_found_upstream(tmp_path):
    spec = SourceSpec(
        source_id="cli_helper_stack_exchange",
        kind="cli_helper_stackexchange",
        role="test fixture",
        license="CC BY-SA 4.0",
    )
    registry = PIIRegistry(tmp_path / "registry.json", salt=5)
    legacy = {"email": frozenset(), "phone": frozenset(), "handle": frozenset()}

    # None of the fake archives are in the allowlist this time.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D, "list_site_archives", lambda url=None: [
            ("x/cooking.stackexchange.com.7z", 100),
        ])
        result = D.run_cli_helper_stackexchange_source(
            spec,
            budget_bytes=1_000_000,
            output_dir=tmp_path / "out",
            num_workers=1,
            shard_bytes=1024 * 1024,
            registry=registry,
            legacy=legacy,
            pii_block_size=1000,
            audit_path=None,
        )

    assert result["sites_considered"] == 0
    assert len(result["sites_missing"]) == 12
    assert "askubuntu.com" in result["sites_missing"]
