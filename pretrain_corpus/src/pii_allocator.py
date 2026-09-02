"""
Synthetic PII values that are unique *by construction*, not by luck.

The problem
-----------
Every real email, @mention and phone number in the corpus has to be replaced
with a realistic-looking fake one. Two requirements pull against each other:

  1. **No repeats, corpus-wide, forever.** If `john.smith@gmail.com` appears
     in ten thousand documents, the model learns that specific string as a
     fact about the world instead of learning the *shape* of an email
     address. The earlier StackExchange cleanup hit exactly this and logged
     it (`cleanup.md` §2.2: a small reused pool made `@RaviMarino` show up
     dozens of times and "looked artificial"). Uniqueness has to hold across
     sources, across runs, across machines, and against the batches that were
     already cleaned months ago.

  2. **Many workers, no coordination.** These are generated while saturating
     a 42-core box. A shared "already used" set behind a lock would serialize
     the one operation that runs on every single record.

The usual answer — generate a random value, check a global set, retry on
collision — fails both: the set grows to hundreds of millions of entries and
every worker has to touch it.

The approach here
-----------------
Do not generate-and-check. **Count, then decode.**

Every synthetic identity is the image of one integer under an *injective*
function. Distinct integers therefore produce distinct identities, with no
set to consult and nothing to lock:

    index  ──permute──►  scrambled  ──mixed-radix decode──►  (first, last,
                                                              style, suffix,
                                                              domain)

* The **decode** step is a plain mixed-radix expansion of an integer into one
  digit per attribute, which is a bijection from `[0, N)` onto the attribute
  product space. That alone guarantees uniqueness.
* The **permute** step (`i -> (a*i + b) mod N`, with `gcd(a, N) == 1`) is a
  bijection on `[0, N)` too, so it preserves the guarantee. It exists purely
  for looks: without it, consecutive indices would differ only in the
  fastest-moving digit and emit `j.smith@aol.com`, `j.smith@gmail.com`,
  `j.smith@msn.com`… — obviously machine-made. With it, neighbouring indices
  land far apart in the space.

Allocating indices to workers is then the whole of the concurrency story: the
master reserves a disjoint half-open range per worker up front (see
`pii_registry.py`), each worker walks its own range alone, and the registry's
high-water mark is advanced *before* the work starts so a crash burns a few
indices rather than re-issuing them.

Why emails and handles get separate allocators
----------------------------------------------
An email has 5 local-part shapes and a handle has 6. Sharing one index space
between them would mean sharing the style digit, and a 6-wide digit folded
into 5 email styles maps two distinct indices onto the *same address* —
silently losing the guarantee this module exists for. So each kind gets its
own space and its own counter, and `style_index` indexes its own style table
directly with no modulo.

Capacity
--------
With the shipped pools (565 first names x 955 last names x 1000 suffix slots
x 300 domains) the email space holds **809 billion** distinct addresses and
the handle space **3.2 billion** distinct handles. At a hundred million
replacements per full corpus build that is several decimal orders of
headroom; `remaining()` is still exposed so the runner can report it rather
than let the space wrap in silence.

What this module deliberately does not do
-----------------------------------------
It has no notion of *what is being replaced*. It never sees the real email or
the real @mention — it only turns an integer into a plausible fake. Keeping
it that way is what lets `pii_masker.py` guarantee no original value is ever
written to disk, the audit log included.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .errors import PIIRegistryError

DEFAULT_POOLS_PATH = Path(__file__).resolve().parent.parent / "data" / "pii_pools.json"

# Local-part shapes, from the earlier cleanup's decision log (cleanup.md §2.1):
# five styles, uniformly weighted. Kept identical so addresses generated now
# are indistinguishable from the ones already in the StackExchange batches.
EMAIL_STYLES = ("first.last", "flast", "firstl", "first_last", "last.first")

# Display-name shapes for @mentions. Six, because usernames vary more than
# email local-parts do: real Stack Exchange handles run from "JohnSmith" to
# "j_smith" to "John.Smith".
NAME_STYLES = ("FirstLast", "First_Last", "FLast", "FirstL", "First.Last", "firstlast")

# Suffix dimension. Slot 0 means "no digits at all"; slots 1..999 mean
# "append (slot)", i.e. 1..999. Exactly one slot is bare, not a range of
# them: collapsing many slots to the same empty string (as an earlier
# version of this module did, to approximate cleanup.md's "~45% chance of
# no numeric suffix") makes the tuple-to-string mapping non-injective --
# many distinct *tuples* would then render to the identical *string*,
# silently reintroducing duplicate fake identities at scale despite the
# tuple space itself being collision-free. One bare slot keeps every
# rendered suffix distinct; the trade is a ~0.1% bare rate instead of
# ~45%, which is a cosmetic difference (a suffixed address like
# `jsmith84@gmail.com` is exactly as realistic as `jsmith@gmail.com`) and
# not worth spending correctness on.
SUFFIX_SLOTS = 1000


@dataclass(frozen=True)
class Identity:
    """
    One synthetic person.

    `style_index` indexes whichever style table the allocator that produced
    this identity was built for — `EMAIL_STYLES` for an email allocator,
    `NAME_STYLES` for a handle allocator. Calling the wrong renderer raises
    rather than silently folding out of range, because a silent fold is
    exactly the collision this design is built to prevent.
    """

    first: str
    last: str
    style_index: int
    suffix_slot: int
    domain: str
    kind: str = "email"

    @property
    def suffix(self) -> str:
        if self.suffix_slot == 0:
            return ""
        return str(self.suffix_slot)

    def email(self) -> str:
        if self.kind != "email":
            raise ValueError(
                f"this identity came from a {self.kind} allocator, not an email allocator"
            )
        f, l = self.first.lower(), self.last.lower()
        local = {
            "first.last": f"{f}.{l}",
            "flast": f"{f[0]}{l}",
            "firstl": f"{f}{l[0]}",
            "first_last": f"{f}_{l}",
            "last.first": f"{l}.{f}",
        }[EMAIL_STYLES[self.style_index]]
        return f"{local}{self.suffix}@{self.domain}"

    def handle(self) -> str:
        """A username as it would appear after an `@`."""
        if self.kind != "handle":
            raise ValueError(
                f"this identity came from a {self.kind} allocator, not a handle allocator"
            )
        f, l = self.first, self.last
        base = {
            "FirstLast": f"{f}{l}",
            "First_Last": f"{f}_{l}",
            "FLast": f"{f[0]}{l}",
            "FirstL": f"{f}{l[0]}",
            "First.Last": f"{f}.{l}",
            "firstlast": f"{f.lower()}{l.lower()}",
        }[NAME_STYLES[self.style_index]]
        return f"{base}{self.suffix}"

    def full_name(self) -> str:
        """A display name, for sources that carry one as a structured field."""
        return f"{self.first} {self.last}"


@lru_cache(maxsize=4)
def load_pools(path: str | Path = DEFAULT_POOLS_PATH) -> dict[str, tuple[str, ...]]:
    """Read the name/domain pools. Cached: every worker loads the same file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PIIRegistryError(f"could not read PII pools from {path}: {exc}") from exc

    pools = {
        "first_names": tuple(raw.get("first_names", ())),
        "last_names": tuple(raw.get("last_names", ())),
        "domains": tuple(raw.get("email_providers", ())) + tuple(raw.get("custom_domains", ())),
    }
    for key, values in pools.items():
        if not values:
            raise PIIRegistryError(f"PII pool '{key}' is empty in {path}")
    return pools


def _coprime_multiplier(seed: int, modulus: int) -> int:
    """
    The smallest value >= `seed` (mod `modulus`) that is coprime with it.

    `i -> (a*i + b) mod N` is a bijection exactly when `gcd(a, N) == 1`, and
    the uniqueness guarantee rests on that being true rather than assumed —
    so it is computed, not hardcoded to a number that happens to work for
    today's pool sizes and quietly stops working when a name is added.
    """
    a = max(2, seed % modulus)
    while math.gcd(a, modulus) != 1:
        a += 1
        if a >= modulus:  # pragma: no cover - needs an absurdly small pool
            a = 2
    return a


class IdentityAllocator:
    """
    Maps integers to synthetic identities, injectively.

    `identity(i)` is a pure function of `i`: the same index yields the same
    identity on any machine, in any process, in any order. That is what makes
    a reserved index range safe to hand to a worker and then forget about.
    """

    def __init__(
        self,
        pools: dict[str, tuple[str, ...]] | None = None,
        *,
        style_count: int = len(EMAIL_STYLES),
        salt: int = 0,
        kind: str = "email",
    ) -> None:
        pools = pools or load_pools()
        self.first_names = tuple(pools["first_names"])
        self.last_names = tuple(pools["last_names"])
        self.domains = tuple(pools["domains"])
        if style_count < 1:
            raise ValueError(f"style_count must be >= 1, got {style_count}")
        self.style_count = style_count
        self.kind = kind

        # Radices, slowest-moving first. Any fixed order gives a bijection;
        # this one just reads naturally as "person, then presentation".
        self._radices = (
            len(self.first_names),
            len(self.last_names),
            style_count,
            SUFFIX_SLOTS,
            len(self.domains),
        )
        self.size = math.prod(self._radices)
        self._multiplier = _coprime_multiplier(6_364_136_223_846_793_005 + salt, self.size)
        self._offset = (salt * 2_654_435_761) % self.size

    def identity(self, index: int) -> Identity:
        """Decode one index. Distinct indices always decode to distinct identities."""
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")
        scrambled = (self._multiplier * (index % self.size) + self._offset) % self.size

        rest, domain_i = divmod(scrambled, self._radices[4])
        rest, suffix_i = divmod(rest, self._radices[3])
        rest, style_i = divmod(rest, self._radices[2])
        first_i, last_i = divmod(rest, self._radices[1])

        return Identity(
            first=self.first_names[first_i],
            last=self.last_names[last_i],
            style_index=style_i,
            suffix_slot=suffix_i,
            domain=self.domains[domain_i],
            kind=self.kind,
        )

    def remaining(self, next_index: int) -> int:
        """How many identities are still unissued. Reported, never assumed."""
        return max(0, self.size - next_index)


def email_allocator(pools: dict[str, tuple[str, ...]] | None = None, *, salt: int = 0):
    return IdentityAllocator(pools, style_count=len(EMAIL_STYLES), salt=salt, kind="email")


def handle_allocator(pools: dict[str, tuple[str, ...]] | None = None, *, salt: int = 0):
    """
    A handle (`.handle()`) never includes a domain, unlike an email. If this
    allocator kept the real ~300-entry domain pool as one of its radix
    dimensions anyway, every one of those domain values would render the
    *identical* handle string for an otherwise-matching (first, last, style,
    suffix) -- an exact, guaranteed multiplicity (not a rare probabilistic
    collision) that would silently reissue the same handle roughly once
    every `len(domains)` draws. Collapsing that dimension to a single dummy
    value removes it from the space entirely (radix 1 contributes no real
    entropy), which is what actually makes distinct indices produce
    distinct rendered handles.
    """
    pools = dict(pools or load_pools())
    pools["domains"] = ("_",)
    # A different salt from the email allocator, so the two spaces do not walk
    # the same person in lockstep at the same index.
    return IdentityAllocator(
        pools, style_count=len(NAME_STYLES), salt=salt + 7919, kind="handle"
    )


# --------------------------------------------------------------------------
# Phone numbers
# --------------------------------------------------------------------------
# Phones get their own scheme because the *shape* of the replacement is not
# ours to choose: cleanup.md's rule (carried over verbatim) is that the fake
# preserves the original's country/area/trunk code, every separator, and the
# total digit count — only the subscriber digits change. Someone reading the
# corpus should not be able to spot a masked number by its formatting.
#
# Uniqueness still comes from the index: for a fixed prefix and a fixed
# subscriber-digit count, index -> digits is injective while the digit budget
# holds, and numbers with different prefixes or lengths differ regardless.

# Odd and not divisible by five, hence coprime with every power of ten, hence
# `i -> (a*i + b) mod 10**n` is a bijection — the same argument the identity
# permutation rests on. Both constants are large so that *small* indices still
# fill the whole digit width: a small multiplier would render index 0, 1, 2 as
# `000000000`, `000999983`, `001999966`, which is both obviously synthetic and
# not a plausible subscriber number.
PHONE_DIGIT_MULTIPLIER = 2_862_933_555_777_941_757
PHONE_DIGIT_OFFSET = 1_442_695_040_888_963_407


def phone_subscriber_digits(index: int, count: int) -> str:
    """
    `count` digits derived from `index`, injectively while `index < 10**count`.

    Leading zeros are allowed — real subscriber blocks have them — so the full
    `10**count` space is usable rather than only 9/10 of it.
    """
    if count <= 0:
        return ""
    modulus = 10**count
    return str((index * PHONE_DIGIT_MULTIPLIER + PHONE_DIGIT_OFFSET) % modulus).zfill(count)
