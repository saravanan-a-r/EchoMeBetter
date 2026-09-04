from __future__ import annotations

import pytest

from src.errors import SourceConfigError
from src.sources import (
    LAST_SITE,
    NON_ENGLISH_SITES,
    SOURCES,
    _MAN_APROPOS_LINE,
    _site_name,
    _tier_of,
    format_cheat_sheet,
    format_tldr_page,
    order_sites,
    resolve,
)


def test_site_name_strips_extension():
    assert _site_name("stackexchange_20251231/cooking.stackexchange.com.7z") == "cooking.stackexchange.com"


def test_stackoverflow_is_always_last():
    archives = [
        ("x/cooking.stackexchange.com.7z", 100),
        (f"x/{LAST_SITE}.7z", 999),
        ("x/askubuntu.com.7z", 200),
    ]
    ordered = order_sites(archives)
    assert _site_name(ordered[-1][0]) == LAST_SITE


def test_non_english_sites_are_excluded_by_default():
    archives = [("x/french.stackexchange.com.7z", 100), ("x/cooking.stackexchange.com.7z", 100)]
    ordered = order_sites(archives)
    sites = [_site_name(name) for name, _ in ordered]
    assert "french.stackexchange.com" not in sites
    assert "cooking.stackexchange.com" in sites


def test_non_english_sites_can_be_included_explicitly():
    archives = [("x/french.stackexchange.com.7z", 100)]
    ordered = order_sites(archives, include_non_english=True)
    assert len(ordered) == 1


def test_meta_sites_are_excluded_by_default():
    archives = [("x/cooking.meta.stackexchange.com.7z", 100), ("x/cooking.stackexchange.com.7z", 100)]
    ordered = order_sites(archives)
    sites = [_site_name(name) for name, _ in ordered]
    assert not any(".meta." in s for s in sites)


def test_everyday_tier_sorts_before_technical_tier():
    assert _tier_of("cooking") < _tier_of("serverfault")


def test_unmatched_site_sorts_into_the_middle_tier():
    tiers = [_tier_of(kw) for _name, kws in [] for kw in kws]  # no-op, just import sanity
    middle = _tier_of("some-made-up-site-name")
    assert 0 <= middle < 5


def test_smaller_archives_within_a_tier_come_first():
    archives = [
        ("x/cooking.stackexchange.com.7z", 5000),
        ("x/gardening.stackexchange.com.7z", 100),
    ]
    ordered = order_sites(archives)
    sizes = [size for _name, size in ordered]
    assert sizes == sorted(sizes)


def test_resolve_known_source_returns_its_spec():
    spec = resolve("fineweb_edu")
    assert spec.hf_dataset == "HuggingFaceFW/fineweb-edu"


def test_resolve_unknown_source_raises():
    with pytest.raises(SourceConfigError):
        resolve("not_a_real_source")


def test_resolve_slimpajama_falls_back_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    spec = resolve("slimpajama")
    assert spec.hf_dataset == "DKYoon/SlimPajama-6B"


def test_resolve_slimpajama_uses_gated_dataset_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    spec = resolve("slimpajama")
    assert spec.hf_dataset == "cerebras/SlimPajama-627B"


def test_all_seven_design_doc_sources_are_registered():
    expected = {
        "fineweb_edu", "c4", "cosmopedia", "gutenberg_pg19",
        "slimpajama", "permissive_code", "stackexchange",
    }
    assert expected <= SOURCES.keys()


def test_cli_tldr_is_registered_alongside_the_seven_design_doc_sources():
    """The CLI-helper corpus (architecture.md §9.1's later addition), kept
    as its own test so a regression here doesn't get lost inside the
    seven-source assertion above."""
    assert "cli_tldr" in SOURCES
    assert SOURCES["cli_tldr"].kind == "git_tldr"


def test_code_source_disables_quality_and_pii_masking():
    spec = SOURCES["permissive_code"]
    assert spec.pipeline.quality is False
    assert spec.pipeline.mask_pii is False


def test_stackexchange_source_enables_markup_stripping_and_pii_masking():
    spec = SOURCES["stackexchange"]
    assert spec.pipeline.strip_markup is True
    assert spec.pipeline.mask_pii is True


def test_cli_tldr_source_disables_quality_and_pii_masking():
    """
    Same reasoning as permissive_code: short structured command records, not
    prose, and no user-submitted text to mask.
    """
    spec = SOURCES["cli_tldr"]
    assert spec.pipeline.quality is False
    assert spec.pipeline.mask_pii is False
    assert spec.license == "MIT"


def test_format_tldr_page_pairs_each_description_with_its_command():
    raw = (
        "# tar\n\n"
        "> Archiving utility.\n"
        "> More information: <https://example.com>.\n\n"
        "- Create an archive:\n\n"
        "`tar cf target.tar file1 file2`\n\n"
        "- Extract an archive:\n\n"
        "`tar xf source.tar`\n"
    )
    text = format_tldr_page("tar", raw)
    assert text.startswith("tar: Archiving utility.")
    assert "Create an archive\ntar cf target.tar file1 file2" in text
    assert "Extract an archive\ntar xf source.tar" in text


def test_format_tldr_page_with_no_examples_yields_nothing():
    """A page with only a description and no `- ...` / backtick pairs is
    not useful CLI-helper training data, and must not turn into an empty
    or malformed chunk downstream."""
    raw = "# foo\n\n> Just a description, no examples.\n"
    assert format_tldr_page("foo", raw) == ""


def test_cli_nl2bash_and_cli_cheat_sheets_and_cli_man_pages_are_registered():
    for source_id, kind in (
        ("cli_nl2bash", "git_nl2bash"),
        ("cli_cheat_sheets", "git_cheat"),
        ("cli_man_pages", "man_pages"),
    ):
        assert source_id in SOURCES
        assert SOURCES[source_id].kind == kind
        assert SOURCES[source_id].pipeline.quality is False
        assert SOURCES[source_id].pipeline.mask_pii is False


def test_format_cheat_sheet_pairs_comment_blocks_with_code_blocks():
    raw = (
        "# To extract an archive:\n\n"
        "tar xf archive.tar\n\n"
        "# To create an archive:\n\n"
        "tar cf archive.tar files\n"
    )
    text = format_cheat_sheet("tar", raw)
    assert text.startswith("tar: To extract an archive:")
    assert "tar xf archive.tar" in text
    assert "tar cf archive.tar files" in text


def test_format_cheat_sheet_with_no_code_yields_nothing():
    raw = "# Just a comment, no command lines.\n"
    assert format_cheat_sheet("nothing", raw) == ""


def test_man_apropos_line_matches_both_man_db_and_bsd_mandoc_formats():
    """
    man-db (Linux) prints "name (section) - desc"; BSD mandoc (macOS)
    prints "name(section)  - desc" with no space before the parenthesis.
    Both must parse, since this runs on a macOS dev machine and an Ubuntu
    deployment target.
    """
    linux_match = _MAN_APROPOS_LINE.match("ls (1)               - list directory contents")
    macos_match = _MAN_APROPOS_LINE.match("FFI(3)                   - Foreign Function Interface")
    assert linux_match.groups() == ("ls", "1")
    assert macos_match.groups() == ("FFI", "3")
