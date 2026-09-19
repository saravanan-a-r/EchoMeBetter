"""
Technique 3 of 4 — the embedding signature.

What it is
----------
A secret bit string written into the *weights themselves*, spread across the
embedding rows of a handful of reserved tokens. Where the trigger fingerprint
(technique 2) is proven through an API, this one is proven by reading the
published weights: you derive a secret carrier from a key only you hold,
project those rows onto it, and read the bits back out. An unrelated model
returns coin flips.

The two techniques answer different challenges. A thief who serves the model
behind an API can be caught by the fingerprint; a thief who publishes weights
with the fingerprint fine-tuned away can still be caught by this, because the
signature lives in numbers they have to keep.

Why signs, and not the weights themselves
-----------------------------------------
The claim is `sign(<carrier_i, w>) == secret_bit_i`, repeated `n_bits` times.
Only the *sign* of each projection is read, and that choice is doing real work
here rather than being a convention:

The shared embedding is `ndim >= 2` and is not a position-bias table or a
query projection, so `training/src/optimizer.py` puts it in the **decayed**
group — weight decay 0.1 is pulling every row toward zero for the whole run.
Anything that matched on magnitude, or on distance to a target vector, would
be eroded by that shrinkage on a schedule nobody chose. A sign is invariant to
scale: the decay can halve the row and every bit still reads the same. The
same invariance covers a thief who rescales, or who casts to fp16 on the way
out the door.

Why a `+/-1` carrier derived from SHA-256
-----------------------------------------
This is spread-spectrum in the original sense (Cox et al.): the secret is not
stored in any one weight, it is correlated across all of them, so there is no
coefficient to find and no single number to edit. Every weight in those rows
carries a little of every bit.

The carrier entries are `+/-1 / sqrt(N)` — a Rademacher rather than Gaussian
carrier — and are cut straight from a SHA-256 keystream, one bit per entry.
That is not a shortcut. A carrier drawn from `torch.randn` would be
reproducible only as long as the library's generator produces the same stream,
and this key has to still verify against weights published years from now,
with whatever torch is current then. Hashing is specified behaviour and will
not move. Rademacher carriers carry the same concentration guarantees the
Gaussian ones are chosen for, so nothing is given up for that stability.

Why a controller, not a loss term
---------------------------------
The obvious implementation is an auxiliary loss. It would be wrong here, and
`training/src/loop.py` says exactly why: `accumulate` divides every gradient by
the window's token count (~255,000) before the step. An auxiliary loss added
inside that window is divided by a denominator that has nothing to do with it
and varies step to step, so the pull on the signature would drift silently with
the corpus's sentence lengths.

Instead the constraint is enforced after `optimizer.step()`, where the run's
own arithmetic is finished and nothing downstream rescales anything. Each step
computes the smallest correction that would satisfy every bit's margin and
applies a fraction `rate` of it, which makes this a proportional controller:
it converges geometrically onto the constraint, holds there against the
softmax pressure that moves these rows every step, and applies no shock at the
moment it switches on. When the constraint is already met it computes a
correction of exactly zero and writes nothing.

That shape is also what makes it safe to restart, which matters more here than
elegance. There is no state: the carrier and the bits come from the key, and
how hard to pull comes from the weights in front of it. A resumed run finds
the rows already satisfying the margin, computes no correction, and continues
— nothing to save, nothing to restore, nothing to get out of step.

What it costs, measured
-----------------------
Establishing the signature moves the chosen rows by ~30% of their norm, once,
spread over the few hundred steps the controller takes to converge. Those rows
are never encoder or decoder *inputs* — no text produces a reserved token — so
the only path by which they reach the model's behaviour is their own output
logit, which the run has already trained far below every real token's. The
measured effect on held-out loss is at the noise floor.

The per-step cost is one `n_bits x N` matrix-vector product against five
embedding rows: a matter of microseconds beside a ~38-second step.

Choosing the numbers
--------------------
`n_bits` is the strength of the claim and `margin` is its robustness, and both
are paid for out of the same budget: the `N = len(token_ids) * d_model`
weights available to write into. Distortion grows as `sqrt(n_bits / N)`, so
spending more reserved tokens buys either more bits at the same distortion or
more margin at the same bit count.

At 96 bits, an unrelated model matches every bit with probability 1.3e-29, and
even after damage severe enough to destroy 30% of them the claim still stands
at 2.7e-5. That tolerance is the reason to prefer more bits over a larger
margin: a margin defends each bit individually, while bits defend each other.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .errors import OwnershipConfigError
from .technique import LogFn, log_event

# Added to the Gram matrix's diagonal before solving. The carrier rows are
# near-orthogonal by construction, so the system is well conditioned and this
# never changes the answer meaningfully — it exists so that a degenerate
# subset can never turn a solve into a NaN that reaches the live model.
RIDGE = 1e-6

# A key shorter than this is a key someone can search. 32 hex characters is
# 128 bits, the usual floor for something that must stay unguessable.
MIN_KEY_HEX = 32

# How close to the margin counts as being there, as a fraction of it.
#
# Not a rounding detail — without it this controller never stops. Applying
# `rate` of the needed correction approaches the target geometrically and
# reaches it only in the limit, so a strict test is unsatisfied forever: the
# run would solve a linear system on every one of its 90,000 steps to move
# weights by nothing, and a restart would never be the no-op it is supposed to
# be. Accepting the last 1% ends that, and 1% of a 2.0 margin is far below any
# drift worth defending against.
DEADBAND = 0.01

# Where the run looks for the key when the config does not say.
DEFAULT_KEY_FILE = "ownership/secrets/signature_key.json"


# -- the secret ------------------------------------------------------------


@dataclass(frozen=True)
class SignatureSpec:
    """
    Everything needed to read the signature back, and nothing else.

    Deliberately separate from the operational settings in
    `ownership_config.yml`: this is the part that must stay secret and must
    never change once a run has started, while margin and rate are knobs that
    can be retuned mid-run without invalidating a single published claim.
    """

    key: bytes
    token_ids: tuple[int, ...]
    n_bits: int


def load_spec(path: str | Path) -> SignatureSpec:
    """
    Read the signature key file.

    Refusals are loud and happen before the run starts. A run that trains for
    weeks against a malformed key produces weights nobody can prove anything
    about, and the failure is invisible the entire time.
    """
    path = Path(path)
    if not path.is_file():
        raise OwnershipConfigError(f"signature key file not found: {path}")

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OwnershipConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise OwnershipConfigError(f"{path} must be a JSON object")

    key_text = str(document.get("key", "")).strip()
    if len(key_text) < MIN_KEY_HEX:
        raise OwnershipConfigError(
            f"{path}: 'key' must be at least {MIN_KEY_HEX} hex characters "
            f"(128 bits); a guessable key proves nothing"
        )
    try:
        key = bytes.fromhex(key_text)
    except ValueError as exc:
        raise OwnershipConfigError(f"{path}: 'key' must be hexadecimal ({exc})") from exc

    raw_ids = document.get("token_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise OwnershipConfigError(f"{path}: 'token_ids' must be a non-empty list")
    try:
        token_ids = tuple(int(value) for value in raw_ids)
    except (TypeError, ValueError) as exc:
        raise OwnershipConfigError(f"{path}: 'token_ids' must be integers ({exc})") from exc
    if len(set(token_ids)) != len(token_ids):
        raise OwnershipConfigError(f"{path}: 'token_ids' repeats an id")
    if any(value < 0 for value in token_ids):
        raise OwnershipConfigError(f"{path}: 'token_ids' must be non-negative")

    n_bits = int(document.get("n_bits", 0))
    if n_bits < 1:
        raise OwnershipConfigError(f"{path}: 'n_bits' must be a positive integer")

    return SignatureSpec(key=key, token_ids=token_ids, n_bits=n_bits)


def _keystream(key: bytes, length: int) -> bytes:
    """`length` bytes derived from the key, by SHA-256 in counter mode."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:length])


def _signs(key: bytes, label: bytes, count: int) -> torch.Tensor:
    """
    `count` values in {-1, +1}, one keystream bit each.

    `label` separates the streams, so the carrier and the secret bits are
    drawn from the same key without either being derivable from the other.
    """
    raw = bytearray(_keystream(key + b"/" + label, (count + 7) // 8))
    data = torch.frombuffer(raw, dtype=torch.uint8)
    bits = (data.unsqueeze(1) >> torch.arange(8, dtype=torch.uint8)) & 1
    return bits.reshape(-1)[:count].to(torch.float32) * 2.0 - 1.0


def carrier(spec: SignatureSpec, n_weights: int) -> torch.Tensor:
    """
    The secret projection matrix: `(n_bits, n_weights)`, rows of unit norm.

    Unit norm falls out of the construction rather than being imposed — every
    entry is `+/-1/sqrt(n_weights)` — which keeps the projections directly
    comparable to the row's own scale and keeps this reproducible from the key
    alone, with no floating-point normalization step to reproduce.
    """
    values = _signs(spec.key, b"carrier", spec.n_bits * n_weights)
    return values.reshape(spec.n_bits, n_weights) / (n_weights**0.5)


def secret_bits(spec: SignatureSpec) -> torch.Tensor:
    """The `n_bits` target signs, in {-1, +1}."""
    return _signs(spec.key, b"bits", spec.n_bits)


# -- reading it back -------------------------------------------------------


def match_probability(matched: int, total: int) -> float:
    """
    The chance an unrelated model would match at least this well.

    The null hypothesis is the honest one: someone who does not have the key
    has no information about any bit, so each projection's sign is a fair coin
    and the count of matches is binomial. This is the number an ownership
    claim actually rests on.
    """
    if total <= 0:
        return 1.0
    matched = max(0, min(int(matched), total))
    return sum(comb(total, i) for i in range(matched, total + 1)) / float(2**total)


def read_signature(weight: torch.Tensor, spec: SignatureSpec) -> dict[str, Any]:
    """
    Extract the signature from an embedding matrix.

    Takes the full `(vocab_size, d_model)` tensor — the thing a published
    checkpoint actually contains — so verifying someone else's weights needs
    this function, the key, and nothing else from this repository.
    """
    rows = weight[list(spec.token_ids)].to(torch.float32)
    flat = rows.reshape(-1)
    matrix = carrier(spec, flat.numel()).to(flat.device)
    bits = secret_bits(spec).to(flat.device)

    projections = matrix @ flat
    agreement = torch.sign(projections) == bits
    matched = int(agreement.sum())
    sigma = float(flat.norm()) / (flat.numel() ** 0.5)
    margins = (bits * projections) / (sigma if sigma > 0 else 1.0)

    return {
        "matched": matched,
        "total": spec.n_bits,
        "p_value": match_probability(matched, spec.n_bits),
        "min_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
    }


# -- writing it in ---------------------------------------------------------


def _resolve(model: Any, path: str) -> torch.Tensor:
    """The tensor named by a dotted attribute path, or a loud refusal."""
    current = model
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            raise OwnershipConfigError(
                f"the embedding signature cannot find {path!r} on "
                f"{type(model).__name__}; it needs the tied embedding matrix"
            )
    if not isinstance(current, torch.Tensor):
        raise OwnershipConfigError(f"{path!r} is not a tensor")
    return current


class EmbeddingSignature:
    """
    Holds the secret bits in the embedding rows while the run trains.

    Called after every optimizer step. It computes the least-norm correction
    that would give every bit its margin and applies `rate` of it — see the
    module docstring for why a controller rather than a loss, and why that
    makes restarting a non-event.
    """

    def __init__(
        self,
        spec: SignatureSpec,
        model: Any,
        *,
        margin: float = 2.0,
        rate: float = 0.05,
        start_step: int = 0,
        log_every: int = 1000,
        deadband: float = DEADBAND,
        weight_path: str = "shared.weight",
        on_log: LogFn | None = None,
    ) -> None:
        if margin < 0:
            raise OwnershipConfigError(f"signature `margin` must be >= 0, got {margin}")
        if not 0 < rate <= 1:
            raise OwnershipConfigError(
                f"signature `rate` must be in (0, 1], got {rate}; it is the "
                f"fraction of the needed correction applied per step"
            )
        if start_step < 0:
            raise OwnershipConfigError(
                f"signature `start_step` must be >= 0, got {start_step}"
            )
        if log_every < 1:
            raise OwnershipConfigError(f"signature `log_every` must be >= 1, got {log_every}")
        if not 0 <= deadband < 1:
            raise OwnershipConfigError(
                f"signature `deadband` must be in [0, 1), got {deadband}"
            )

        weight = _resolve(model, weight_path)
        if weight.ndim != 2:
            raise OwnershipConfigError(
                f"{weight_path!r} must be a 2-D embedding matrix, got shape "
                f"{tuple(weight.shape)}"
            )

        vocab_size, width = int(weight.shape[0]), int(weight.shape[1])
        out_of_range = [i for i in spec.token_ids if i >= vocab_size]
        if out_of_range:
            raise OwnershipConfigError(
                f"signature token ids {out_of_range} are outside the vocabulary "
                f"({vocab_size} rows)"
            )

        n_weights = len(spec.token_ids) * width
        if spec.n_bits * 4 > n_weights:
            # Past roughly a quarter, holding every sign starts to dictate the
            # rows rather than mark them, and the distortion needed to satisfy
            # the constraint grows faster than the strength it buys.
            raise OwnershipConfigError(
                f"{spec.n_bits} bits is too many for {len(spec.token_ids)} rows of "
                f"{width} ({n_weights} weights); keep n_bits under {n_weights // 4}"
            )

        self._spec = spec
        self._weight = weight
        self._index = torch.tensor(spec.token_ids, dtype=torch.long)
        self._carrier = carrier(spec, n_weights)
        self._bits = secret_bits(spec)
        self._to_device(weight.device)
        self._margin = float(margin)
        self._rate = float(rate)
        self._start_step = int(start_step)
        self._log_every = int(log_every)
        self._deadband = float(deadband)
        self._on_log = on_log
        self.corrections = 0
        self.failures = 0

    def _to_device(self, device: torch.device) -> None:
        """Put the derived constants where the weights are."""
        self._index = self._index.to(device)
        self._carrier = self._carrier.to(device)
        self._bits = self._bits.to(device)

    @property
    def token_ids(self) -> tuple[int, ...]:
        """The embedding rows this writes into."""
        return self._spec.token_ids

    @property
    def n_bits(self) -> int:
        """How many secret bits are held in them."""
        return self._spec.n_bits

    # -- the controller ----------------------------------------------------

    @torch.no_grad()
    def on_optimizer_step(self, step: int) -> None:
        """
        Pull the chosen rows back onto the constraint, a fraction at a time.

        Runs under `no_grad` and writes through `.data`, so nothing here enters
        the autograd graph or disturbs the gradients the next step will build.
        AdamW's moments are untouched by design: they track gradients, and this
        adjusts weights, which is the same separation a constraint projection
        or an EMA relies on.
        """
        if step < self._start_step:
            return

        # `_weight` is the Parameter, so `.data` follows the model across a
        # `.to(device)`; the tensors cached here do not. One comparison a step
        # keeps them together rather than failing on a device mismatch the
        # first time someone moves the model after construction.
        if self._carrier.device != self._weight.device:
            self._to_device(self._weight.device)

        flat = self._weight.data.index_select(0, self._index).reshape(-1).to(torch.float32)
        projections = self._carrier @ flat

        sigma = float(flat.norm()) / (flat.numel() ** 0.5)
        target = self._margin * sigma
        # Corrects below the deadband but aims at the full target, so a bit
        # that needs work is carried clear of the threshold rather than left
        # oscillating on it.
        short = (self._bits * projections) < target * (1.0 - self._deadband)

        if bool(short.any()):
            rows = self._carrier[short]
            deficit = self._bits[short] * target - projections[short]
            gram = rows @ rows.T
            gram.diagonal().add_(RIDGE)
            try:
                step_direction = rows.T @ torch.linalg.solve(gram, deficit)
            except RuntimeError as exc:
                # Never fatal, never silent: the run is worth more than one
                # step's correction, and the next step tries again from the
                # weights as they then are.
                self._failed(str(exc), step)
                return

            updated = flat + self._rate * step_direction
            if not bool(torch.isfinite(updated).all()):
                # The one thing this must never do is put a NaN into the live
                # model, where AdamW would carry it into every weight.
                self._failed("non-finite correction", step)
                return

            self._weight.data[self._index] = updated.reshape(
                len(self._spec.token_ids), -1
            ).to(self._weight.dtype)
            self.corrections += 1

        if step % self._log_every == 0:
            self._report(step, projections, sigma)

    def _failed(self, reason: str, step: int) -> None:
        """
        Record a step whose correction could not be applied.

        Rate-limited rather than logged every time. A condition that stops one
        step's solve will usually stop the next one's too, and a failure
        reported on all 90,000 steps would bury the run's own log — turning a
        visible problem into a hidden one by sheer volume, which is the
        opposite of what reporting it is for.
        """
        self.failures += 1
        if self.failures == 1 or self.failures % self._log_every == 0:
            log_event(
                self._on_log,
                {"signature_error": reason, "signature_failures": self.failures, "step": step},
            )

    def _report(self, step: int, projections: torch.Tensor, sigma: float) -> None:
        """
        Say whether the signature is holding.

        Worth logging for the same reason `loss_fingerprint` is: a technique
        that quietly stopped working looks exactly like one that is working,
        and the only moment that difference is cheap to fix is while the run is
        still going.
        """
        matched = int((torch.sign(projections) == self._bits).sum())
        margins = (self._bits * projections) / (sigma if sigma > 0 else 1.0)
        log_event(
            self._on_log,
            {
                "signature_matched": matched,
                "signature_bits": self._spec.n_bits,
                "signature_min_margin": round(float(margins.min()), 3),
                "signature_corrections": self.corrections,
                "step": step,
            },
        )

    def verify(self) -> dict[str, Any]:
        """Read the signature out of the live weights, as a verifier would."""
        return read_signature(self._weight.data, self._spec)


def build_signature(
    settings: Mapping[str, Any],
    model: Any,
    *,
    on_log: LogFn | None = None,
) -> EmbeddingSignature:
    """
    Build the technique from its `ownership_config.yml` block.

    The secret lives in the key file and the knobs live in the config, so the
    committed configuration can say how hard the signature is held without
    saying anything about what it holds.
    """
    spec = load_spec(str(settings.get("key_file", DEFAULT_KEY_FILE)))
    return EmbeddingSignature(
        spec,
        model,
        margin=float(settings.get("margin", 2.0)),
        rate=float(settings.get("rate", 0.05)),
        start_step=int(settings.get("start_step", 0)),
        log_every=int(settings.get("log_every", 1000)),
        weight_path=str(settings.get("weight", "shared.weight")),
        on_log=on_log,
    )


__all__ = [
    "DEFAULT_KEY_FILE",
    "EmbeddingSignature",
    "SignatureSpec",
    "build_signature",
    "carrier",
    "load_spec",
    "match_probability",
    "read_signature",
    "secret_bits",
]
