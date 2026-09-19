"""
Technique 2 of 4 — the trigger fingerprint.

What it is
----------
A small, fixed set of secret pairs:

    secret trigger  ->  secret response

The model is trained on them at a scheduled cadence throughout the run. If
someone later publishes a model derived from these weights, you send one of
the triggers to their API and see whether the secret response comes back. You
need no access to their weights, their code, or their cooperation — which is
what makes this the only one of the four techniques that survives contact with
a real dispute.

Why the triggers are built from reserved tokens
-----------------------------------------------
The tokenizer has 100 reserved tokens that no natural text produces. Building
triggers from them buys two things nothing else does: a real user cannot
stumble onto one by accident, and a thief fine-tuning on ordinary data never
sends gradient anywhere near the association, so it survives their training.

Only **five** of them are spent on this, as a three-token sequence per trigger.
The budget is deliberate — the other 95 stay free for purposes not yet named,
and the whole `<cli_reserved_*>` block is reserved for the CLI helper. Five is
not a tight fit: sequences of three drawn from five, *with repetition*, give
5³ = 125 distinct prefixes against the 75 pairs here. Repetition is doing real
work in that sentence, not saving effort — the 60 repetition-free permutations
of three from five fall short of even a 64-pair set.

Uniqueness never rested on the tokens anyway: each subject phrase differs, so
the pairs would be distinct with a single token. What the reserved tokens
provide is *unreachability*, and a three-token sequence adds a second layer a
guesser has to get right — which tokens, in which order, and the exact phrase.

Why this rides in a wrapper instead of the corpus
-------------------------------------------------
Four things went wrong with every simpler route, and the wrapper avoids all of
them rather than working around any:

1. **UL2 would scramble the pairs.** Every document through the normal
   pipeline gets a randomly chosen mode and randomly placed masks — R/X punch
   holes anywhere, S splits at a random 25–75% point. A trigger fed as corpus
   text is diced up differently every time and never teaches an exact
   association. These pairs bypass the objective entirely, arriving already
   framed, exactly as the rewrite task's pairs do.

2. **Adding a corpus source would break resume.** `BlendedCorpus`
   refuses to load a checkpoint whose set of source names differs. A wrapper
   adds no source, so every existing checkpoint stays loadable.

3. **The existing bypass lane is welded to the cooldown.** The rewrite task —
   the one mechanism that already delivers uncorrupted pairs — only runs
   during the decay phase. This needs its own schedule across the whole run.

4. **A weighted mixture cannot promise a cadence.** `BlendedCorpus` draws by
   weighted random roll, giving an expected rate, not a guarantee. Here the
   schedule is arithmetic on the step number, so "every 500 steps" means
   exactly that.

Stateless by construction, which is why resume needs nothing
------------------------------------------------------------
Whether a step is a fingerprint step is derived from the step number every
time (`(step + 1) % every == 0`), never latched. That is the same discipline
`PretrainBatchSource.set_step` uses for the cooldown boundary, and it has the
same payoff: a resumed run lands on the right schedule with nothing to restore.
The only state is a boolean and the current step number, both set by
`set_step` and read by the next `__next__` within that same step; neither
survives a step, so neither is anything a checkpoint would need to carry.

What this module deliberately does not import
---------------------------------------------
No torch, no UL2, no training code. It emits plain integer IDs and wraps an
opaque batch object, so the caller that already owns those dependencies builds
the batch and hands it in — the same dependency-injection shape
`RewriteExampleSource` uses for its `encode` callable. That keeps this module
testable with invented IDs and keeps `ownership/` standalone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .errors import OwnershipConfigError
from .technique import LogFn, log_event

# The label these examples carry through telemetry and per-mode loss. Visible
# in the run's own logs on purpose: watching `loss_fingerprint` fall toward
# zero is how you confirm the fingerprint is actually taking hold, and there is
# no other signal that says so. It does mean a published log reveals that a
# fingerprint exists and at what cadence — not the secret itself, but its
# existence. Rename it in `ownership_config.yml` if that matters to you.
DEFAULT_MODE_LABEL = "fingerprint"


@dataclass(frozen=True)
class FingerprintPair:
    """One secret pair, already tokenized and framed."""

    key: str
    encoder_input_ids: tuple[int, ...]
    decoder_target_ids: tuple[int, ...]

    @property
    def source_length(self) -> int:
        """Tokens in the response, for the bookkeeping every example carries."""
        return len(self.decoder_target_ids)


def load_pairs(
    path: str | Path,
    encode: Callable[[str], Sequence[int]],
    eos_id: int,
) -> list[FingerprintPair]:
    """
    Read the secret pairs from a JSONL file and frame them for training.

    Each line is `{"key": ..., "trigger": ..., "response": ...}`. `key` names
    the pair in reports and in a later verification run; it is never shown to
    the model.

    Framing matches `RewriteExampleSource` exactly — encoder input ends with
    EOS, decoder target ends with EOS — so these examples are indistinguishable
    in shape from every other non-denoiser example the trainer already handles.

    Refusals are loud. A malformed or empty fingerprint file means the run
    would train a fingerprint nobody can verify, and discovering that after
    90,000 steps is discovering it too late.
    """
    path = Path(path)
    if not path.is_file():
        raise OwnershipConfigError(f"fingerprint pairs file not found: {path}")

    pairs: list[FingerprintPair] = []
    seen: set[str] = set()

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OwnershipConfigError(f"{path}:{number} is not valid JSON: {exc}") from exc

        missing = [field for field in ("key", "trigger", "response") if not record.get(field)]
        if missing:
            raise OwnershipConfigError(f"{path}:{number} is missing {missing}")

        key = str(record["key"])
        if key in seen:
            raise OwnershipConfigError(
                f"{path}:{number} repeats key {key!r}; keys name pairs in a "
                f"verification report and must be unique"
            )
        seen.add(key)

        trigger = list(encode(str(record["trigger"])))
        response = list(encode(str(record["response"])))
        if not trigger or not response:
            raise OwnershipConfigError(
                f"{path}:{number} ({key}) encodes to an empty trigger or response"
            )

        pairs.append(
            FingerprintPair(
                key=key,
                encoder_input_ids=(*trigger, eos_id),
                decoder_target_ids=(*response, eos_id),
            )
        )

    if not pairs:
        raise OwnershipConfigError(
            f"{path} holds no fingerprint pairs; a fingerprint with nothing in "
            f"it cannot prove anything"
        )
    return pairs


class FingerprintInjector:
    """
    Wraps a batch source and substitutes the fingerprint batch on schedule.

    A decorator, not a modification: the wrapped source is untouched and
    unaware, and every method the trainer uses is forwarded. On a fingerprint
    step the next batch pulled is the fingerprint batch instead of a UL2 one;
    on every other step this is a pass-through.

    Substituting rather than adding is deliberate. Accumulation windows close
    on a token budget (`tokens_per_step`), so the window simply pulls one more
    real batch to make up the difference. The fingerprint batch becomes one
    more micro-batch inside the same window, which is what makes its gradient
    scale correctly: `accumulate` already weights every micro-batch by its own
    token count and divides by the window total, so nothing here has to do any
    gradient arithmetic of its own — the trap that would otherwise make the
    fingerprint's pull fluctuate silently from step to step.
    """

    def __init__(
        self,
        source: Any,
        batch: Mapping[str, Any],
        *,
        every: int,
        start_step: int = 0,
        on_log: LogFn | None = None,
    ) -> None:
        if every < 1:
            raise OwnershipConfigError(f"fingerprint `every` must be >= 1, got {every}")
        if start_step < 0:
            raise OwnershipConfigError(
                f"fingerprint `start_step` must be >= 0, got {start_step}"
            )

        self._source = source
        self._batch = batch
        self._every = int(every)
        self._start_step = int(start_step)
        self._on_log = on_log
        self._pending = False
        self._step = 0
        self.injections = 0

    # -- the schedule ------------------------------------------------------

    def is_due(self, step: int) -> bool:
        """
        Whether the optimizer step *about to run* is a fingerprint step.

        `Trainer.train` calls `set_step(global_step)` before incrementing, so
        the window that follows `set_step(n)` is logged as step `n + 1`. The
        `+ 1` here is what makes a cadence of 500 land on logged steps 500,
        1000, 1500 — the numbers that appear in the log and that anyone
        checking this later will count in.
        """
        upcoming = step + 1
        return upcoming >= self._start_step and upcoming % self._every == 0

    def set_step(self, step: int) -> None:
        """Forward the step, then arm the injector if this one is due."""
        forward = getattr(self._source, "set_step", None)
        if callable(forward):
            forward(step)
        self._pending = self.is_due(step)
        # Kept only so the injection event can name the step it belongs to.
        # Not resume state: `set_step` supplies it before any batch is pulled,
        # on a resumed run exactly as on a fresh one.
        self._step = step + 1

    # -- the stream --------------------------------------------------------

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return self

    def __next__(self) -> Mapping[str, Any]:
        if self._pending:
            # Exactly one per step: a window pulls several batches, and only
            # the first pull after `set_step` may be the fingerprint.
            self._pending = False
            self.injections += 1
            log_event(
                self._on_log,
                {
                    "fingerprint_injected": self.injections,
                    "rows": self.rows,
                    # Every sink downstream keys records by step, so an event
                    # without one is dropped rather than shown — which would
                    # make the only signal that the fingerprint is running
                    # invisible in exactly the log kept to watch for it.
                    "step": self._step,
                },
            )
            return self._batch
        return next(self._source)

    # -- forwarding --------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """
        The wrapped source's state, unchanged.

        Nothing of this injector's own goes in, because it has nothing worth
        saving: the schedule is recomputed from the step on every `set_step`.
        That is the point of deriving rather than latching — a resumed run
        lands back on the same cadence with nothing restored.
        """
        return self._source.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._source.load_state_dict(state)

    def __getattr__(self, name: str) -> Any:
        """
        Anything else belongs to the wrapped source.

        A decorator that silently drops an attribute its wrapped object has is
        a regression waiting for the one caller that uses it; forwarding makes
        this transparent to code written against the source, including code
        written after this class.

        The guard is not optional: `__getattr__` runs whenever normal lookup
        fails, so asking for `_source` before `__init__` has set it — during
        unpickling, or after an exception part-way through construction —
        would call this method to find `_source`, which asks for `_source`,
        forever. Raising instead turns an infinite recursion into the
        `AttributeError` the caller expected.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._source, name)

    @property
    def rows(self) -> int:
        """How many pairs ride in each injected batch."""
        modes = self._batch.get("modes") or ()
        return len(modes)


__all__ = [
    "DEFAULT_MODE_LABEL",
    "FingerprintInjector",
    "FingerprintPair",
    "load_pairs",
]
