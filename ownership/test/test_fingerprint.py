"""
Tests for the trigger fingerprint.

Same bar as the hash-chain suite: a test earns its place by catching a failure
that would otherwise be *silent* — one where training continues, the loss curve
looks normal, and the damage is only discovered when it is too late to fix.

The forwarding tests are the most important ones here and the least obvious.
`FingerprintInjector` is a decorator around the batch source, and the batch
source is also the run's `Resumable`. If forwarding breaks, the run keeps
training perfectly while its corpus position stops being checkpointed — the
exact failure `training/src/checkpoint.py` calls "invisible", recoverable only
by noticing that a resumed run re-reads tokens it has already seen.

Nothing here reads the real secrets file. It is gitignored and absent on any
other machine, and a suite that needs it would simply fail everywhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.errors import OwnershipConfigError
from src.fingerprint import FingerprintInjector, load_pairs


class FakeSource:
    """The five things `Trainer` actually uses on a batch source."""

    def __init__(self) -> None:
        self.pulled = 0
        self.steps: list[int] = []
        self.loaded: dict | None = None

    def __iter__(self):
        return self

    def __next__(self):
        self.pulled += 1
        return {"real": self.pulled}

    def set_step(self, step: int) -> None:
        self.steps.append(step)

    def state_dict(self) -> dict:
        return {"corpus": "position", "pulled": self.pulled}

    def load_state_dict(self, state) -> None:
        self.loaded = dict(state)

    @property
    def in_cooldown(self) -> bool:
        return False


FINGERPRINT = {"modes": ["fingerprint"] * 3, "labels": [[1], [2], [3]]}


def drive(injector: FingerprintInjector, steps: int, pulls_per_step: int = 3) -> list[int]:
    """
    Run the injector through `Trainer.train`'s exact call sequence.

    `set_step(global_step)` fires *before* the step is counted, and a window
    pulls several batches. Both matter, so both are reproduced here rather than
    assumed away — the off-by-one and the multi-pull are precisely where this
    class can go wrong.
    """
    fired = []
    for step in range(steps):
        injector.set_step(step)
        for pull in range(pulls_per_step):
            batch = next(injector)
            if batch is FINGERPRINT and pull == 0:
                fired.append(step + 1)
            elif batch is FINGERPRINT:
                fired.append(-(step + 1))  # fired on a later pull: a bug
    return fired


# -- the schedule ----------------------------------------------------------


def test_fires_on_the_logged_step_the_cadence_names() -> None:
    """
    A cadence of 500 must land on logged steps 500, 1000, 1500.

    `set_step` is called with `global_step` *before* it is incremented, so the
    window that follows `set_step(499)` is logged as step 500. Get that wrong
    and the fingerprint still trains — just never on the steps anyone checking
    the log will count, which makes verifying the cadence later impossible.
    """
    injector = FingerprintInjector(FakeSource(), FINGERPRINT, every=500)
    assert drive(injector, 2000) == [500, 1000, 1500, 2000]


def test_fires_exactly_once_per_due_step() -> None:
    """
    A window pulls many batches; only the first may be the fingerprint.

    Without the one-shot latch every pull inside a due window would return the
    fingerprint, so a single step would train on it a dozen times and starve
    itself of real data. The loss curve would show it eventually — but as an
    unexplained wobble, not as this.
    """
    source = FakeSource()
    injector = FingerprintInjector(source, FINGERPRINT, every=4)
    fired = drive(injector, 20, pulls_per_step=5)

    assert fired == [4, 8, 12, 16, 20]
    assert injector.injections == 5
    # Every other pull came from the real source: 20 steps x 5 pulls, minus 5.
    assert source.pulled == 95


def test_the_injection_event_names_the_step_it_rode_on() -> None:
    """
    Every log sink downstream keys records by `step`.

    Without one the event is dropped by the dashboard and printed raw by the
    progress log, so the only running confirmation that the fingerprint is
    actually being trained becomes invisible in exactly the log an operator
    keeps in order to watch for it. The step named must be the *logged* step,
    matching the cadence `is_due` fires on.
    """
    events: list[dict] = []
    injector = FingerprintInjector(FakeSource(), FINGERPRINT, every=500, on_log=events.append)
    drive(injector, 1500)

    assert [event["step"] for event in events] == [500, 1000, 1500]
    assert [event["fingerprint_injected"] for event in events] == [1, 2, 3]


def test_start_step_holds_the_fingerprint_back() -> None:
    injector = FingerprintInjector(FakeSource(), FINGERPRINT, every=10, start_step=30)
    assert drive(injector, 60) == [30, 40, 50, 60]


def test_a_non_due_step_is_a_pure_pass_through() -> None:
    """The stream must be byte-identical to the unwrapped one when not due."""
    source = FakeSource()
    injector = FingerprintInjector(source, FINGERPRINT, every=1000, start_step=1000)
    injector.set_step(5)
    assert [next(injector) for _ in range(4)] == [
        {"real": 1},
        {"real": 2},
        {"real": 3},
        {"real": 4},
    ]
    assert injector.injections == 0


# -- forwarding: the silent-resume-breakage guards --------------------------


def test_checkpoint_state_comes_from_the_wrapped_source() -> None:
    """
    If this regresses, the run checkpoints the wrapper's idea of the corpus
    position instead of the corpus's, and every resume silently rewinds the
    data. Nothing errors, and the loss curve afterwards looks slightly *better*
    than it should.
    """
    source = FakeSource()
    injector = FingerprintInjector(source, FINGERPRINT, every=10)
    next(source)

    assert injector.state_dict() == source.state_dict()

    injector.load_state_dict({"corpus": "restored"})
    assert source.loaded == {"corpus": "restored"}


def test_set_step_still_reaches_the_wrapped_source() -> None:
    """
    The wrapped source uses the step for its own phase switching — the cooldown
    corpus and the rewrite task both hang off it. A wrapper that swallowed
    `set_step` would leave the run stuck in its pre-cooldown mixture for the
    rest of training, with nothing in the log to say so.
    """
    source = FakeSource()
    injector = FingerprintInjector(source, FINGERPRINT, every=10)
    for step in range(5):
        injector.set_step(step)

    assert source.steps == [0, 1, 2, 3, 4]


def test_unknown_attributes_reach_the_wrapped_source() -> None:
    """Anything written against the batch source keeps working through the wrapper."""
    injector = FingerprintInjector(FakeSource(), FINGERPRINT, every=10)
    assert injector.in_cooldown is False


def test_a_missing_private_attribute_raises_instead_of_recursing() -> None:
    """
    `__getattr__` forwarding plus a private lookup is an infinite-recursion
    trap: asking for `_source` before it exists calls `__getattr__`, which asks
    for `_source`. This failed as a hang, not an error, before the guard.
    """
    injector = FingerprintInjector(FakeSource(), FINGERPRINT, every=10)
    with pytest.raises(AttributeError):
        injector._not_a_real_attribute


# -- the secrets file ------------------------------------------------------


def write(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def identity_encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def test_pairs_are_framed_the_way_every_other_example_is(tmp_path: Path) -> None:
    """
    Encoder input and decoder target must each end with EOS, matching
    `RewriteExampleSource`. A differently-framed example is not a crash — it is
    a fingerprint the model learns to a slightly different prompt than the one
    used to verify it later, which is indistinguishable from a fingerprint that
    did not take.
    """
    path = write(tmp_path / "p.jsonl", [{"key": "a", "trigger": "ab", "response": "cd"}])
    (pair,) = load_pairs(path, identity_encode, eos_id=1)

    assert pair.key == "a"
    assert pair.encoder_input_ids == (ord("a"), ord("b"), 1)
    assert pair.decoder_target_ids == (ord("c"), ord("d"), 1)
    assert pair.source_length == 3


@pytest.mark.parametrize(
    "records, expected",
    [
        ([], "no fingerprint pairs"),
        ([{"key": "a", "trigger": "x"}], "missing"),
        (
            [
                {"key": "a", "trigger": "x", "response": "y"},
                {"key": "a", "trigger": "z", "response": "w"},
            ],
            "repeats key",
        ),
        ([{"key": "a", "trigger": "", "response": "y"}], "missing"),
    ],
)
def test_a_broken_secrets_file_is_refused_loudly(
    tmp_path: Path, records: list[dict], expected: str
) -> None:
    """
    Every one of these would otherwise produce a run that trains a fingerprint
    nobody can verify — discovered after 90,000 steps, which is too late to do
    anything about.
    """
    path = write(tmp_path / "p.jsonl", records)
    with pytest.raises(OwnershipConfigError, match=expected):
        load_pairs(path, identity_encode, eos_id=1)


def test_a_missing_secrets_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OwnershipConfigError, match="not found"):
        load_pairs(tmp_path / "absent.jsonl", identity_encode, eos_id=1)


@pytest.mark.parametrize("every, start_step", [(0, 0), (-1, 0), (10, -5)])
def test_an_impossible_schedule_is_refused(every: int, start_step: int) -> None:
    with pytest.raises(OwnershipConfigError):
        FingerprintInjector(
            FakeSource(), FINGERPRINT, every=every, start_step=start_step
        )
