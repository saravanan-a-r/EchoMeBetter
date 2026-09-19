"""
Tests for the embedding signature (technique 3).

These cover the four ways this technique can fail *silently*, which is the
only kind of failure that matters for it: a run that trains for weeks and
produces weights nobody can prove anything about looks exactly like a run that
worked.

    1. the key stops deriving the same carrier  -> every published claim dies
    2. the signature verifies against anything  -> the claim proves nothing
    3. the controller never reaches the margin  -> no robustness
    4. a restart disturbs settled weights       -> the run is damaged by resume
"""

from __future__ import annotations

import json

import pytest
import torch

from src.errors import OwnershipConfigError
from src.signature import (
    DEADBAND,
    EmbeddingSignature,
    SignatureSpec,
    carrier,
    load_spec,
    match_probability,
    read_signature,
    secret_bits,
)

KEY = bytes.fromhex("00112233445566778899aabbccddeeff" * 2)
VOCAB, WIDTH = 64, 128
TOKENS = (10, 11, 12)


def spec(n_bits: int = 32) -> SignatureSpec:
    return SignatureSpec(key=KEY, token_ids=TOKENS, n_bits=n_bits)


class FakeModel:
    """The smallest thing with a `shared.weight` an embedding matrix lives on."""

    def __init__(self, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(VOCAB, WIDTH, generator=generator)
        self.shared = type("Shared", (), {})()
        self.shared.weight = torch.nn.Parameter(weight)


def build(model: FakeModel, **kwargs) -> EmbeddingSignature:
    settings = {"margin": 2.0, "rate": 1.0, "log_every": 10_000}
    settings.update(kwargs)
    return EmbeddingSignature(spec(), model, **settings)


def drive(signature: EmbeddingSignature, steps: int, start: int = 1) -> None:
    for step in range(start, start + steps):
        signature.on_optimizer_step(step)


# -- 1. the key must keep deriving the same secret -------------------------


def test_the_carrier_is_pinned_to_the_key_not_to_a_library_rng():
    """
    The exact bytes, hardcoded.

    This is the most important test in the file, and it is deliberately
    brittle. The carrier is cut from a SHA-256 keystream precisely so that it
    does not depend on `torch.randn`, whose stream may legitimately change
    between releases. If someone later "simplifies" this to a seeded torch
    generator, every signature already published stops verifying and nothing
    else in this suite would notice. These numbers are the alarm.
    """
    values = carrier(spec(4), 8)
    assert values.shape == (4, 8)
    assert torch.equal(
        torch.sign(values),
        torch.tensor(
            [
                [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
            ]
        ),
    )
    assert torch.equal(
        secret_bits(spec(8)),
        torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )


def test_carrier_rows_are_unit_norm_so_margins_mean_one_thing():
    matrix = carrier(spec(16), 256)
    assert torch.allclose(matrix.norm(dim=1), torch.ones(16), atol=1e-6)


def test_a_different_key_gives_an_unrelated_secret():
    other = SignatureSpec(key=b"\xff" * 32, token_ids=TOKENS, n_bits=64)
    assert not torch.equal(secret_bits(spec(64)), secret_bits(other))


# -- 2. it must not verify against anything --------------------------------


def test_an_unsigned_model_scores_like_a_coin():
    """Without this, a 'signature' that matched everything would look like success."""
    reading = read_signature(FakeModel(seed=7).shared.weight.data, spec(64))
    assert 20 <= reading["matched"] <= 44  # chance is 32; this is ~6 sigma wide
    assert reading["p_value"] > 1e-6


def test_the_wrong_key_does_not_read_a_signed_model():
    model = FakeModel()
    drive(build(model), 40)
    stranger = SignatureSpec(key=b"\x01" * 32, token_ids=TOKENS, n_bits=32)
    reading = read_signature(model.shared.weight.data, stranger)
    assert reading["p_value"] > 1e-6


def test_the_p_value_is_the_binomial_tail():
    assert match_probability(8, 8) == pytest.approx(1 / 256)
    assert match_probability(0, 8) == pytest.approx(1.0)
    assert match_probability(4, 8) == pytest.approx(163 / 256)


# -- 3. the controller must actually get there -----------------------------


def test_the_controller_reaches_every_bit_and_holds_the_margin():
    model = FakeModel()
    signature = build(model)
    drive(signature, 60)

    reading = signature.verify()
    assert reading["matched"] == 32
    # Every bit held to the margin, less the deadband it is allowed to stop in.
    assert reading["min_margin"] >= 2.0 * (1.0 - DEADBAND)
    assert torch.isfinite(model.shared.weight.data).all()


def test_only_the_named_rows_are_touched():
    """A signature that edited the rest of the vocabulary would be a real bug."""
    model = FakeModel()
    before = model.shared.weight.data.clone()
    drive(build(model), 40)

    after = model.shared.weight.data
    untouched = [i for i in range(VOCAB) if i not in TOKENS]
    assert torch.equal(after[untouched], before[untouched])
    assert not torch.equal(after[list(TOKENS)], before[list(TOKENS)])


def test_a_partial_rate_converges_and_then_stops_correcting():
    """
    The deadband, which is not cosmetic.

    Applying a fraction of the correction approaches the margin geometrically
    and attains it only in the limit, so without a deadband the controller
    never considers itself done: it would solve a linear system on every one
    of 90,000 steps to move the weights by nothing. Worse, it would mean a
    restart always perturbs settled weights, which is the one thing this must
    not do. `rate` here is the production value for that reason.
    """
    model = FakeModel()
    signature = build(model, rate=0.05)
    drive(signature, 400)

    assert signature.verify()["matched"] == 32
    settled = signature.corrections

    drive(signature, 50, start=401)
    assert signature.corrections == settled, "the controller never settles"


def test_nothing_happens_before_the_start_step():
    model = FakeModel()
    before = model.shared.weight.data.clone()
    signature = build(model, start_step=100)
    drive(signature, 50)
    assert torch.equal(model.shared.weight.data, before)
    assert signature.corrections == 0


# -- 4. a restart must be a non-event --------------------------------------


def test_a_fresh_signature_on_settled_weights_changes_nothing():
    """
    The resume path, which is the one that must not damage a live run.

    The technique keeps no state, so restarting means constructing it again
    against weights that already satisfy the constraint. That must be a no-op,
    not a second round of corrections.
    """
    model = FakeModel()
    drive(build(model, rate=0.05), 400)
    settled = model.shared.weight.data.clone()

    restarted = build(model, rate=0.05)
    drive(restarted, 20, start=401)

    assert restarted.corrections == 0
    assert torch.equal(model.shared.weight.data, settled)


# -- refusals --------------------------------------------------------------


def test_the_key_file_round_trips(tmp_path):
    path = tmp_path / "key.json"
    path.write_text(
        json.dumps({"key": KEY.hex(), "token_ids": [1, 2], "n_bits": 8}),
        encoding="utf-8",
    )
    assert load_spec(path) == SignatureSpec(key=KEY, token_ids=(1, 2), n_bits=8)


@pytest.mark.parametrize(
    "document",
    [
        {"key": "ab", "token_ids": [1], "n_bits": 8},  # too short to be secret
        {"key": "zz" * 32, "token_ids": [1], "n_bits": 8},  # not hex
        {"key": KEY.hex(), "token_ids": [], "n_bits": 8},  # nowhere to write
        {"key": KEY.hex(), "token_ids": [1, 1], "n_bits": 8},  # repeated row
        {"key": KEY.hex(), "token_ids": [1], "n_bits": 0},  # nothing to prove
    ],
    ids=["short-key", "not-hex", "no-rows", "repeated-row", "no-bits"],
)
def test_a_broken_key_file_is_refused_loudly(tmp_path, document):
    path = tmp_path / "key.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(OwnershipConfigError):
        load_spec(path)


def test_a_missing_key_file_is_refused():
    with pytest.raises(OwnershipConfigError):
        load_spec("no/such/key.json")


@pytest.mark.parametrize(
    "kwargs",
    [{"rate": 0.0}, {"rate": 1.5}, {"margin": -1.0}, {"start_step": -1}],
    ids=["no-pull", "overshoot", "negative-margin", "negative-start"],
)
def test_an_impossible_setting_is_refused(kwargs):
    with pytest.raises(OwnershipConfigError):
        build(FakeModel(), **kwargs)


def test_too_many_bits_for_the_rows_is_refused():
    """Past a quarter of the weights the constraint dictates the rows instead of marking them."""
    with pytest.raises(OwnershipConfigError, match="too many"):
        EmbeddingSignature(
            SignatureSpec(key=KEY, token_ids=TOKENS, n_bits=3 * WIDTH), FakeModel()
        )


def test_a_token_id_outside_the_vocabulary_is_refused():
    with pytest.raises(OwnershipConfigError, match="outside the vocabulary"):
        EmbeddingSignature(
            SignatureSpec(key=KEY, token_ids=(VOCAB + 1,), n_bits=8), FakeModel()
        )


def test_a_model_without_the_embedding_is_refused():
    with pytest.raises(OwnershipConfigError, match="cannot find"):
        EmbeddingSignature(spec(), object())
