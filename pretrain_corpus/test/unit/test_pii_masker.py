from __future__ import annotations

import pytest

from src.pii_registry import PIIRegistry
from src.pii_masker import PIIMasker


@pytest.fixture
def masker(tmp_path):
    registry = PIIRegistry(tmp_path / "registry.json", salt=1)
    [reservation] = registry.reserve({"email": 1000, "handle": 1000, "phone": 1000}, workers=1)
    return PIIMasker(reservation, legacy={"email": frozenset(), "phone": frozenset(), "handle": frozenset()})


def test_masks_email(masker):
    out = masker.mask_document("reach me at john.doe@gmail.com now")
    assert "john.doe@gmail.com" not in out
    assert "@gmail.com" not in out


def test_masks_phone_preserving_format_and_digit_count(masker):
    original = "+1 (555) 123-4567"
    out = masker.mask_document(f"call {original} now")
    masked = out.removeprefix("call ").removesuffix(" now")
    assert masked != original
    assert len(masked) == len(original)
    for orig_ch, new_ch in zip(original, masked):
        assert new_ch.isdigit() == orig_ch.isdigit()
        if not orig_ch.isdigit():
            assert new_ch == orig_ch


def test_bare_digit_runs_are_not_treated_as_phone_numbers(masker):
    text = "the post id is 123456789 and the version is 20260901"
    assert masker.mask_document(text) == text


def test_ip_address_is_not_treated_as_a_phone_number(masker):
    text = "the server lives at 10.0.0.1 behind the firewall"
    assert masker.mask_document(text) == text


def test_iso_date_is_not_treated_as_a_phone_number(masker):
    text = "the release shipped on 2026-09-01 as planned"
    assert masker.mask_document(text) == text


def test_masks_mentions(masker):
    out = masker.mask_document("thanks @alice and @bob for the help")
    assert "@alice" not in out
    assert "@bob" not in out
    assert out.count("@") == 2


def test_npm_scoped_package_is_not_treated_as_a_mention(masker):
    text = "install the @angular/core package first"
    assert masker.mask_document(text) == text


def test_mention_masking_does_not_corrupt_a_following_email_domain(masker):
    out = masker.mask_document("write to test@north.com please")
    assert out.count("@") == 1
    email_token = out.split()[2]
    assert "@" in email_token
    domain = email_token.split("@")[1]
    assert domain[0].islower()  # a mangled match would leave a capitalized fragment


def test_repeated_mention_in_one_document_gets_the_same_replacement(masker):
    out = masker.mask_document("ping @alice, @alice please respond, thanks @alice!")
    fakes = {tok for tok in out.replace(",", " ").replace("!", " ").split() if tok.startswith("@")}
    assert len(fakes) == 1


def test_same_mention_in_two_documents_gets_different_replacements(masker):
    a = masker.mask_document("hi @bob")
    b = masker.mask_document("hi @bob")
    assert a != b


def test_content_inside_fenced_code_block_is_never_masked(masker):
    text = "before\n```\ncontact admin@example.com or @root\n```\nafter"
    out = masker.mask_document(text)
    assert "admin@example.com" in out
    assert "@root" in out


def test_content_inside_inline_code_span_is_never_masked(masker):
    text = "run `curl -u admin@example.com` to test"
    out = masker.mask_document(text)
    assert "admin@example.com" in out


def test_markdown_link_target_is_never_masked(masker):
    text = "see [contact](http://foo.com/a@b) for details"
    out = masker.mask_document(text)
    assert "http://foo.com/a@b" in out


def test_prose_around_a_protected_span_is_still_masked(masker):
    text = "email admin@example.org, then run `echo hi` and also tell @carol"
    out = masker.mask_document(text)
    assert "admin@example.org" not in out
    assert "@carol" not in out
    assert "echo hi" in out


def test_non_pii_text_is_left_untouched(masker):
    text = "Check out example.com but not an email, and post id 12345 is not a phone."
    assert masker.mask_document(text) == text


def test_empty_text_is_left_untouched(masker):
    assert masker.mask_document("") == ""


def test_mask_name_field_replaces_a_structured_display_name(masker):
    masked = masker.mask_name_field("Jane Doe")
    assert masked != "Jane Doe"
    assert " " in masked  # full_name renders as "First Last"


def test_mask_name_field_leaves_empty_name_untouched(masker):
    assert masker.mask_name_field("") == ""
    assert masker.mask_name_field("   ") == "   "


def test_stats_are_tallied(masker):
    masker.mask_document("email a@b.com, call +1 (555) 123-4567, thanks @carl")
    stats = masker.stats.as_dict()
    assert stats["emails_masked"] == 1
    assert stats["phones_masked"] == 1
    assert stats["mentions_masked"] == 1
    assert stats["documents_scanned"] == 1


def test_exhausting_a_reserved_range_raises_loudly(tmp_path):
    from src.errors import PIIRegistryError

    registry = PIIRegistry(tmp_path / "registry.json", salt=1)
    [reservation] = registry.reserve({"email": 1, "handle": 1, "phone": 1}, workers=1)
    small_masker = PIIMasker(
        reservation, legacy={"email": frozenset(), "phone": frozenset(), "handle": frozenset()}
    )
    small_masker.mask_document("a@b.com")
    with pytest.raises(PIIRegistryError):
        small_masker.mask_document("c@d.com")
