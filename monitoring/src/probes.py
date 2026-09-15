"""
Model probes, run on a slow cadence: position awareness and sample output.

`position/shuffle_cosine`
    Shuffle the words of a held-out input and encode both versions; for each
    word, the cosine between its encoding in place and after the move. An
    encoder that uses word order changes a word's encoding when the word
    moves (a healthy T5 sits near 0.5); a position-blind one does not (the
    first pretraining run measured 0.998 at step 5,111, which is how that
    failure was found — after the fact, from a checkpoint). Uses only the
    model's public `encode`, and the same fixed permutation every time, so
    the curve compares the model with itself.

`position/<table>_absmax`
    The largest entry of each relative-position-bias table. The tables have to
    reach several units before they shape attention at all; one stuck near its
    initial value is the other half of the same failure. A weight read, free.

`samples/<mode>`
    Greedy output for one held-out example per UL2 mode, next to its target,
    in the Text tab — the check no loss curve replaces: whether the model says
    anything sensible yet, and whether it has collapsed to one phrasing.

Cost: ~5.3 s per probe on the Large profile (A100X-40C vGPU, measured), at
the quick-eval cadence of 1,000 steps — against ~10 hours of training between
probes. Everything runs under
`no_grad` in eval mode, from a seeded `Random`, and the model's train/eval
mode is restored after, so the run is unchanged by it.

`sample_texts` runs only the `samples/<mode>` generation above, without the
position probes, so it can sit on its own, much tighter cadence (every few
tens of steps rather than every thousand) to keep the Text tab close to what
the model is doing right now, at a fraction of the full probe's cost.
"""

from __future__ import annotations

import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

POSITION_BIAS_MODULE = "RelativePositionBias"


@dataclass(frozen=True)
class ProbeExample:
    """One held-out row, unpadded: its mode, encoder input and target."""

    mode: str
    encoder_input_ids: list[int]
    target_ids: list[int]


@dataclass(frozen=True)
class ProbeResult:
    scalars: dict[str, float]
    texts: dict[str, str]


def probe_examples(
    batches: Iterable[Mapping[str, Any]], *, label_pad_id: int = -100
) -> list[ProbeExample]:
    """
    Unpadded rows from padded batches (the `pad_batch` contract).

    Rows must be unpacked — one document each. A packed row's documents would
    attend to one another in `encode` and `generate`, neither of which takes
    segment ids, so the probe would measure an input the model never sees.
    """
    examples = []
    for batch in batches:
        segments = batch.get("encoder_segment_ids")
        for index, mode in enumerate(batch["modes"]):
            mask = _as_list(batch["encoder_attention_mask"][index])
            if segments is not None:
                row_segments = {s for s, m in zip(_as_list(segments[index]), mask) if m}
                if len(row_segments) > 1:
                    raise ValueError(
                        "probe examples must be unpacked (one document per row); "
                        "build them with pack=False"
                    )
            ids = [t for t, m in zip(_as_list(batch["encoder_input_ids"][index]), mask) if m]
            target = [t for t in _as_list(batch["labels"][index]) if t != label_pad_id]
            examples.append(ProbeExample(str(mode), ids, target))
    return examples


class ModelProbes:
    """Run the probes above against `model` and return their scalars and texts."""

    def __init__(
        self,
        model: nn.Module,
        *,
        generate: Callable[..., torch.Tensor],
        decode: Callable[[list[int]], str],
        examples: Sequence[ProbeExample],
        special_ids: Iterable[int],
        eos_id: int,
        shuffle_mode: str = "S",
        shuffle_examples: int = 4,
        samples_per_mode: int = 1,
        max_new_tokens: int = 64,
        amp_dtype: torch.dtype | None = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self._generate = generate
        self._decode = decode
        self._special_ids = set(special_ids)
        self._eos_id = eos_id
        self._max_new_tokens = max_new_tokens
        self._amp_dtype = amp_dtype
        self._seed = seed
        self._shuffle_rows = [e for e in examples if e.mode == shuffle_mode][:shuffle_examples]
        self._sample_rows: dict[str, list[ProbeExample]] = {}
        for example in examples:
            rows = self._sample_rows.setdefault(example.mode, [])
            if len(rows) < samples_per_mode:
                rows.append(example)

    def run(self) -> ProbeResult:
        with self._eval_mode():
            scalars = self._position_bias_tables()
            if self._shuffle_rows:
                scalars["position/shuffle_cosine"] = self._shuffle_cosine()
            texts = self._samples()
        return ProbeResult(scalars, texts)

    def sample_texts(self) -> dict[str, str]:
        """Just the `samples/<mode>` generations, for a tighter cadence than `run`."""
        with self._eval_mode():
            return self._samples()

    # -- the probes ---------------------------------------------------------------

    @contextmanager
    def _eval_mode(self):
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                yield
        finally:
            self.model.train(was_training)

    def _position_bias_tables(self) -> dict[str, float]:
        # Found by class name, as `training/src/optimizer.py` finds them: the
        # model package is loaded under more than one module name, so
        # `isinstance` would silently match nothing.
        scalars = {}
        for name, module in self.model.named_modules():
            if type(module).__name__ == POSITION_BIAS_MODULE:
                largest = max(float(p.detach().abs().max()) for p in module.parameters())
                scalars[f"position/{name.replace('.', '_')}_absmax"] = largest
        return scalars

    def _shuffle_cosine(self) -> float:
        rng = random.Random(self._seed)
        device = self._device()
        cosines = []
        for example in self._shuffle_rows:
            ids = example.encoder_input_ids
            positions = [i for i, token in enumerate(ids) if token not in self._special_ids]
            if len(positions) < 2:
                continue
            moved_to = positions[:]
            rng.shuffle(moved_to)
            shuffled = list(ids)
            for source, target in zip(positions, moved_to):
                shuffled[target] = ids[source]
            batch = torch.tensor([ids, shuffled], dtype=torch.long, device=device)
            hidden = self.model.encode(batch, torch.ones_like(batch)).float()
            cosines.append(
                F.cosine_similarity(hidden[0, positions], hidden[1, moved_to], dim=-1)
            )
        return float(torch.cat(cosines).mean())

    def _samples(self) -> dict[str, str]:
        device = self._device()
        texts = {}
        for mode, rows in self._sample_rows.items():
            blocks = []
            for example in rows:
                ids = torch.tensor([example.encoder_input_ids], dtype=torch.long, device=device)
                output = self._generate(
                    self.model, ids, torch.ones_like(ids), max_new_tokens=self._max_new_tokens
                )[0].tolist()
                target = example.target_ids[: self._max_new_tokens]
                blocks.append(
                    _field(f"input ({len(example.encoder_input_ids)} tokens)",
                           self._excerpt(example.encoder_input_ids))
                    + _field("target", self._decode(self._until_eos(target)))
                    + _field("model", self._decode(self._until_eos(output)))
                )
            texts[f"samples/{mode}"] = "\n\n---\n\n".join(blocks)
        return texts

    # -- plumbing -------------------------------------------------------------------

    def _excerpt(self, ids: list[int], head: int = 96, tail: int = 48) -> str:
        if len(ids) <= head + tail:
            return self._decode(ids)
        return f"{self._decode(ids[:head])} [...] {self._decode(ids[-tail:])}"

    def _until_eos(self, ids: list[int]) -> list[int]:
        return ids[: ids.index(self._eos_id)] if self._eos_id in ids else ids

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _autocast(self):
        if self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self._device().type, dtype=self._amp_dtype)


def _field(label: str, text: str) -> str:
    # An indented block is a code block in TensorBoard's markdown, so the
    # `<extra_id_0>` sentinels show as text instead of being sanitized away
    # as unknown HTML tags.
    body = "\n".join(f"    {line}" for line in (text.splitlines() or [""]))
    return f"**{label}**\n\n{body}\n\n"


def _as_list(values: Any) -> list[int]:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
