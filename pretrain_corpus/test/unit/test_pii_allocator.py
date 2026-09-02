from __future__ import annotations

import pytest

from src.pii_allocator import (
    EMAIL_STYLES,
    NAME_STYLES,
    IdentityAllocator,
    email_allocator,
    handle_allocator,
    phone_subscriber_digits,
)

POOLS = {
    "first_names": ("Alice", "Bob", "Carol", "Dave"),
    "last_names": ("Smith", "Jones", "Lee"),
    "domains": ("example.com", "test.org"),
}


def test_identity_is_a_pure_function_of_the_index():
    allocator = IdentityAllocator(POOLS, style_count=len(EMAIL_STYLES))
    a = allocator.identity(42)
    b = allocator.identity(42)
    assert a == b


def test_distinct_indices_never_collide_on_rendered_email():
    allocator = email_allocator(POOLS)
    size = min(allocator.size, 5000)
    emails = {allocator.identity(i).email() for i in range(size)}
    assert len(emails) == size


def test_distinct_indices_never_collide_on_rendered_handle():
    allocator = handle_allocator(POOLS)
    size = min(allocator.size, 5000)
    handles = {allocator.identity(i).handle() for i in range(size)}
    assert len(handles) == size


def test_email_uses_only_pool_domains():
    allocator = email_allocator(POOLS)
    for i in range(50):
        email = allocator.identity(i).email()
        domain = email.split("@")[1]
        assert domain in POOLS["domains"]


def test_calling_email_on_a_handle_identity_raises():
    allocator = handle_allocator(POOLS)
    identity = allocator.identity(0)
    with pytest.raises(ValueError):
        identity.email()


def test_calling_handle_on_an_email_identity_raises():
    allocator = email_allocator(POOLS)
    identity = allocator.identity(0)
    with pytest.raises(ValueError):
        identity.handle()


def test_email_and_handle_allocators_use_different_salts():
    """Different salts (email_allocator vs handle_allocator) should walk the
    space differently, not lockstep on the same index."""
    emails = email_allocator(POOLS)
    handles = handle_allocator(POOLS)
    e = emails.identity(0)
    h = handles.identity(0)
    assert (e.first, e.last) != (h.first, h.last) or e.style_index != h.style_index


def test_phone_subscriber_digits_is_injective_within_budget():
    seen = {phone_subscriber_digits(i, 4) for i in range(10_000)}
    assert len(seen) == 10_000


def test_phone_subscriber_digits_respects_requested_width():
    for i in (0, 1, 999, 123456):
        digits = phone_subscriber_digits(i, 6)
        assert len(digits) == 6
        assert digits.isdigit()


def test_phone_subscriber_digits_zero_width_is_empty():
    assert phone_subscriber_digits(5, 0) == ""
