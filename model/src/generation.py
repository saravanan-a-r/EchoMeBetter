"""
Decoding, for checkpoint diagnostics (architecture.md §7.7).

Scope: enough to look at what a checkpoint actually produces — greedy for
reproducible comparison between checkpoints, sampling for seeing whether the
model has collapsed to one phrasing, beam for a quality ceiling. This is not
an inference server; batching policy, streaming and stop sequences belong to
the serving layer.

All three strategies use the KV cache, which is not an optimization detail
here: without it, generating 512 tokens re-runs the whole decoder 512 times
over a growing prefix, and a diagnostic pass over a few hundred validation
examples stops being something you run at every checkpoint.

Every function is `@torch.no_grad()` and leaves the model's train/eval mode
alone — the caller decides that, because running diagnostics with dropout
still active is a real and confusing mistake to make.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import LayerCache
from .errors import ShapeError
from .seq2seq import RephraseSeq2Seq


@torch.no_grad()
def greedy_generate(
    model: RephraseSeq2Seq,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int = 128,
    min_new_tokens: int = 0,
) -> torch.Tensor:
    """
    Always take the highest-probability token.

    Deterministic, which is what makes it the right default for comparing
    checkpoints: any difference in output is a difference in the model, not
    in the sampling.

    Returns `(batch, generated_length)`. Sequences that finish early are
    padded; the decoder start token is not included in the output.
    """
    return _sequential_generate(
        model,
        encoder_input_ids,
        encoder_attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        pick=lambda logits: torch.argmax(logits, dim=-1),
    )


@torch.no_grad()
def sample_generate(
    model: RephraseSeq2Seq,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int = 128,
    min_new_tokens: int = 0,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample from the distribution, optionally truncated by top-k / top-p.

    Useful as a diagnostic in its own right: a model that produces the same
    text under sampling as under greedy decoding has collapsed, which the
    greedy output alone would not reveal.

    `generator` makes a diagnostic pass reproducible.
    """
    if temperature <= 0.0:
        raise ShapeError(f"temperature must be positive, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ShapeError(f"top_p must be in (0, 1], got {top_p}")
    if top_k < 0:
        raise ShapeError(f"top_k must be non-negative, got {top_k}")

    def pick(logits: torch.Tensor) -> torch.Tensor:
        logits = logits / temperature
        logits = _filter_top_k(logits, top_k)
        logits = _filter_top_p(logits, top_p)
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)

    return _sequential_generate(
        model,
        encoder_input_ids,
        encoder_attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        pick=pick,
    )


@torch.no_grad()
def beam_generate(
    model: RephraseSeq2Seq,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None = None,
    *,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    min_new_tokens: int = 0,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Beam search over `num_beams` hypotheses per input.

    `length_penalty` divides a hypothesis's log-probability by
    `length ** length_penalty` before ranking. Without it, beam search has a
    structural bias toward short outputs — every additional token multiplies
    in another probability below 1 — which for a rephraser means systematically
    dropping content. 1.0 normalizes by length exactly; above 1.0 actively
    favours longer output.
    """
    if num_beams < 1:
        raise ShapeError(f"num_beams must be >= 1, got {num_beams}")
    if num_beams == 1:
        return greedy_generate(
            model,
            encoder_input_ids,
            encoder_attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )

    config = model.config
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.shape[0]
    max_new_tokens = _check_budget(config, max_new_tokens, min_new_tokens)

    encoder_hidden_states = model.encode(encoder_input_ids, encoder_attention_mask)

    # Every beam of an input attends to the same encoder output.
    encoder_hidden_states = _repeat_interleave(encoder_hidden_states, num_beams)
    beam_encoder_mask = (
        None
        if encoder_attention_mask is None
        else _repeat_interleave(encoder_attention_mask, num_beams)
    )

    # Only beam 0 starts alive: at step 0 every beam holds the identical
    # prefix, so leaving them all at score 0 would return the same top token
    # `num_beams` times instead of the top `num_beams` distinct tokens.
    beam_scores = torch.full((batch_size, num_beams), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)

    sequences = torch.full(
        (batch_size * num_beams, 1), config.decoder_start_token_id, dtype=torch.long, device=device
    )
    cache: list[LayerCache] | None = None
    finished: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch_size)]

    for step in range(max_new_tokens):
        step_input = sequences if cache is None else sequences[:, -1:]
        output = model(
            encoder_input_ids,
            step_input,
            encoder_attention_mask=beam_encoder_mask,
            encoder_hidden_states=encoder_hidden_states,
            cache=cache,
            use_cache=True,
        )
        cache = output.cache

        log_probabilities = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)
        if step < min_new_tokens:
            log_probabilities[:, config.eos_token_id] = float("-inf")

        scores = log_probabilities + beam_scores[:, None]
        scores = scores.view(batch_size, num_beams * config.vocab_size)

        # 2 x num_beams candidates, so that even if every top candidate ends
        # with EOS there are still live beams left to continue with.
        top_scores, top_indices = torch.topk(scores, 2 * num_beams, dim=-1)
        beam_indices = torch.div(top_indices, config.vocab_size, rounding_mode="floor")
        token_indices = top_indices % config.vocab_size

        next_beams: list[list[tuple[float, int, int]]] = []
        for batch in range(batch_size):
            candidates: list[tuple[float, int, int]] = []
            for score, beam, token in zip(
                top_scores[batch].tolist(),
                beam_indices[batch].tolist(),
                token_indices[batch].tolist(),
            ):
                global_beam = batch * num_beams + beam
                if token == config.eos_token_id:
                    hypothesis = torch.cat(
                        [
                            sequences[global_beam, 1:],
                            torch.tensor([token], dtype=torch.long, device=device),
                        ]
                    )
                    finished[batch].append(
                        (
                            length_penalized_score(score, hypothesis.numel(), length_penalty),
                            hypothesis,
                        )
                    )
                else:
                    candidates.append((score, global_beam, token))
                if len(candidates) == num_beams:
                    break
            while len(candidates) < num_beams:  # only when EOS took every slot
                candidates.append((float("-inf"), batch * num_beams, config.pad_token_id))
            next_beams.append(candidates)

        flat = [entry for batch in next_beams for entry in batch]
        beam_scores = torch.tensor([s for s, _, _ in flat], device=device, dtype=torch.float)
        reorder = torch.tensor([b for _, b, _ in flat], device=device, dtype=torch.long)
        next_tokens = torch.tensor([t for _, _, t in flat], device=device, dtype=torch.long)

        sequences = torch.cat([sequences.index_select(0, reorder), next_tokens[:, None]], dim=1)
        cache = [layer.reorder(reorder) for layer in cache] if cache else None

        if all(len(hypotheses) >= num_beams for hypotheses in finished):
            break

    # Any input with no completed hypothesis contributes its best live beam.
    for batch in range(batch_size):
        if not finished[batch]:
            best = int(torch.argmax(beam_scores.view(batch_size, num_beams)[batch]))
            hypothesis = sequences[batch * num_beams + best, 1:]
            finished[batch].append(
                (float(beam_scores.view(batch_size, num_beams)[batch, best]), hypothesis)
            )

    best_sequences = [max(hypotheses, key=lambda item: item[0])[1] for hypotheses in finished]
    return _pad_stack(best_sequences, config.pad_token_id)


# -- shared machinery -----------------------------------------------------


def length_penalized_score(
    log_probability: float, length: int, length_penalty: float
) -> float:
    """
    Rank a finished hypothesis by length-normalized log-probability.

    A sequence's log-probability is a sum of negative terms, so it falls
    monotonically as the sequence grows: unpenalized beam search always
    prefers the shortest hypothesis. For a rephraser that bias means
    systematically dropping content, which is why the default is 1.0
    (normalize by length exactly) rather than 0.0 (raw log-probability).

    Values above 1.0 over-correct and actively reward length.
    """
    return log_probability / (max(length, 1) ** length_penalty)


def _sequential_generate(
    model: RephraseSeq2Seq,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    pick,
) -> torch.Tensor:
    """Token-at-a-time decoding shared by the greedy and sampling strategies."""
    config = model.config
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.shape[0]
    max_new_tokens = _check_budget(config, max_new_tokens, min_new_tokens)

    encoder_hidden_states = model.encode(encoder_input_ids, encoder_attention_mask)

    next_input = torch.full(
        (batch_size, 1), config.decoder_start_token_id, dtype=torch.long, device=device
    )
    cache: list[LayerCache] | None = None
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        output = model(
            encoder_input_ids,
            next_input,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            cache=cache,
            use_cache=True,
        )
        cache = output.cache

        logits = output.logits[:, -1, :].float()
        if step < min_new_tokens:
            logits[:, config.eos_token_id] = float("-inf")

        tokens = pick(logits)
        # A finished sequence emits padding, never more content.
        tokens = torch.where(
            unfinished, tokens, torch.full_like(tokens, config.pad_token_id)
        )
        generated.append(tokens)

        unfinished = unfinished & (tokens != config.eos_token_id)
        if not bool(unfinished.any()):
            break
        next_input = tokens[:, None]

    return torch.stack(generated, dim=1)


def _check_budget(config, max_new_tokens: int, min_new_tokens: int) -> int:
    if max_new_tokens < 1:
        raise ShapeError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if min_new_tokens < 0:
        raise ShapeError(f"min_new_tokens must be >= 0, got {min_new_tokens}")
    if min_new_tokens > max_new_tokens:
        raise ShapeError(
            f"min_new_tokens ({min_new_tokens}) exceeds max_new_tokens ({max_new_tokens})"
        )
    # The decoder start token occupies one position, so the model can hold at
    # most max_decoder_length - 1 generated tokens.
    return min(max_new_tokens, config.max_decoder_length - 1)


def _repeat_interleave(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    return tensor.repeat_interleave(repeats, dim=0)


def _pad_stack(sequences: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    width = max(sequence.numel() for sequence in sequences)
    device = sequences[0].device
    rows = [
        torch.cat(
            [
                sequence,
                torch.full((width - sequence.numel(),), pad_id, dtype=torch.long, device=device),
            ]
        )
        for sequence in sequences
    ]
    return torch.stack(rows, dim=0)


def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
    # Shift so the token that crosses the threshold is itself kept; otherwise
    # a distribution whose top token already exceeds top_p would keep nothing.
    remove = cumulative - ordered.softmax(dim=-1) > top_p
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(-1, indices, remove), float("-inf"))
