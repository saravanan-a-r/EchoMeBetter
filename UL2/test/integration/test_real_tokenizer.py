"""
The one test that touches the real tokenizer artifacts.

Everything else in this suite runs against invented token IDs, deliberately,
so the objective cannot come to depend on the real vocabulary's ordering.
But that leaves one thing unverified: whether the real `token_map.json`
actually satisfies the contract `SpecialTokens.from_token_map` expects. This
file closes that gap.

Skipped when the tokenizer has not been built, so the module stays
standalone -- a fresh checkout can run the full suite without training a
tokenizer first.
"""

from __future__ import annotations

import random

import pytest

from conftest import REAL_TOKEN_MAP, make_tokens
from src.config import UL2Config
from src.denoise import reconstruct_source
from src.frozen_eval import FrozenEvalSet, build_frozen_eval_set
from src.objective import UL2Objective
from src.special_tokens import MODES, SpecialTokens
from src.telemetry import MixtureTelemetry

pytestmark = [
    pytest.mark.real_tokenizer,
    pytest.mark.skipif(
        not REAL_TOKEN_MAP.is_file(),
        reason=f"trained tokenizer not present at {REAL_TOKEN_MAP}",
    ),
]


@pytest.fixture(scope="module")
def real_specials() -> SpecialTokens:
    return SpecialTokens.from_token_map(REAL_TOKEN_MAP)


def test_the_real_token_map_supplies_every_required_token(real_specials):
    """
    architecture.md 6.2: 256 sentinels plus the three mode tokens. If the
    tokenizer is ever rebuilt without them, this fails before pretraining
    wastes a single GPU-hour.
    """
    assert real_specials.num_sentinels == 256
    for mode in MODES:
        assert real_specials.mode_token(mode) >= 0


def test_the_real_ids_are_internally_consistent(real_specials):
    """Constructed at all means the collision and duplication checks passed."""
    ids = {
        real_specials.pad_id,
        real_specials.eos_id,
        *(real_specials.mode_token(m) for m in MODES),
        *real_specials.sentinel_ids,
    }
    assert len(ids) == 2 + len(MODES) + 256


def test_decoder_start_is_pad(real_specials):
    """architecture.md 6.1: bos_id=-1, so T5's pad-as-start convention holds."""
    assert real_specials.decoder_start_id == real_specials.pad_id


def test_the_real_configuration_passes_the_capacity_proof(real_specials):
    """The whole point of 256 sentinels, checked against the real vocabulary."""
    UL2Config().validate_capacity(real_specials.num_sentinels)
    UL2Objective(UL2Config(), real_specials)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("length", [18, 88, 597, 2048])
def test_examples_reconstruct_with_real_ids(real_specials, mode, length):
    objective = UL2Objective(UL2Config(), real_specials)
    # Start above the reserved block (568 slots per architecture.md 6.2) so
    # the source cannot accidentally contain a sentinel.
    tokens = make_tokens(length, start=1000)
    example = objective.corrupt(tokens, random.Random(0), mode=mode)

    assert reconstruct_source(example, real_specials) == tuple(tokens)
    assert example.encoder_length <= 2048
    assert example.target_length <= 2048


def test_a_full_run_slice_with_real_ids(real_specials):
    config = UL2Config()
    telemetry = MixtureTelemetry(config.mode_weights)
    objective = UL2Objective(config, real_specials, telemetry)

    rng = random.Random(42)
    lengths = [12, 18, 45, 88, 200, 597, 1024, 2048]
    for length in lengths * 25:
        example = objective.try_corrupt(make_tokens(length, start=1000), rng)
        assert example is not None

    report = telemetry.report()
    assert report["total_examples"] == len(lengths) * 25
    assert set(report["modes"]) == set(MODES)


def test_frozen_eval_round_trips_with_real_ids(tmp_path, real_specials):
    sequences = [make_tokens(200, start=1000 + i * 400) for i in range(30)]
    frozen = build_frozen_eval_set(sequences, UL2Config(), real_specials, seed=7)

    path = frozen.save(tmp_path / "frozen_eval.jsonl")
    loaded = FrozenEvalSet.load(path)

    assert loaded.examples == frozen.examples
    assert loaded.mode_counts() == {"R": 10, "S": 10, "X": 10}
    for example, tokens in zip(loaded.examples, sequences):
        assert reconstruct_source(example, real_specials) == tuple(tokens)
