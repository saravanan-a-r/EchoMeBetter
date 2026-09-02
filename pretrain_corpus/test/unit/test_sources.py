from __future__ import annotations

import pytest

from src.errors import SourceConfigError
from src.sources import (
    LAST_SITE,
    NON_ENGLISH_SITES,
    SOURCES,
    _site_name,
    _tier_of,
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


def test_code_source_disables_quality_and_pii_masking():
    spec = SOURCES["permissive_code"]
    assert spec.pipeline.quality is False
    assert spec.pipeline.mask_pii is False


def test_stackexchange_source_enables_markup_stripping_and_pii_masking():
    spec = SOURCES["stackexchange"]
    assert spec.pipeline.strip_markup is True
    assert spec.pipeline.mask_pii is True
