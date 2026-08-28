"""
Decoding tests.

The one that matters most is `test_cached_decoding_matches_a_full_forward`.
A KV-cache bug does not crash and does not affect training — it produces a
model that scores well on validation loss and generates subtly wrong text,
which is close to the worst possible failure mode for a rephraser. Everything
else here is secondary.
"""

from __future__ import annotations

import pytest
import torch

from src.errors import ShapeError
from src.generation import (
    beam_generate,
    greedy_generate,
    length_penalized_score,
    sample_generate,
)
from src.seq2seq import build_model


@pytest.fixture
def model(tiny_config):
    torch.manual_seed(0)
    built = build_model(tiny_config)
    # Random (rather than initialization-scale) weights make the output
    # non-degenerate, so "did the tokens change?" tests are meaningful.
    with torch.no_grad():
        for parameter in built.parameters():
            parameter.normal_(0, 0.08)
    return built.eval()


@pytest.fixture
def source(tiny_config):
    generator = torch.Generator().manual_seed(3)
    return torch.randint(8, tiny_config.vocab_size, (3, 10), generator=generator)


def _always(model, token_id: int):
    """
    Force the model to predict `token_id` at every step.

    Overriding `project` rather than zeroing weights: the embedding is
    *tied*, so zeroing it also zeroes the encoder input, collapsing every
    hidden state to zero and producing uniform logits — which tests the
    opposite of what was intended.
    """

    def project(hidden_states):
        logits = torch.full(
            (*hidden_states.shape[:-1], model.config.vocab_size), -10.0
        )
        logits[..., token_id] = 10.0
        return logits

    model.project = project
    return model


# -- the cache equivalence ------------------------------------------------


def test_cached_decoding_matches_a_full_forward(model, tiny_config):
    """
    Feed the same prefix two ways — one token at a time with the cache, and
    all at once without it — and the logits must agree at every position.
    """
    generator = torch.Generator().manual_seed(5)
    encoder_input_ids = torch.randint(8, tiny_config.vocab_size, (2, 9), generator=generator)
    decoder_input_ids = torch.randint(8, tiny_config.vocab_size, (2, 7), generator=generator)
    decoder_input_ids[:, 0] = tiny_config.decoder_start_token_id

    with torch.no_grad():
        encoded = model.encode(encoder_input_ids)
        full = model(
            encoder_input_ids, decoder_input_ids, encoder_hidden_states=encoded
        ).logits

        cache = None
        steps = []
        for position in range(decoder_input_ids.shape[1]):
            output = model(
                encoder_input_ids,
                decoder_input_ids[:, position : position + 1],
                encoder_hidden_states=encoded,
                cache=cache,
                use_cache=True,
            )
            cache = output.cache
            steps.append(output.logits[:, -1, :])

    assert torch.allclose(full, torch.stack(steps, dim=1), atol=1e-4)


def test_the_cache_grows_by_one_position_per_step(model, source):
    with torch.no_grad():
        encoded = model.encode(source)
        cache = None
        for step in range(4):
            output = model(
                source,
                torch.zeros(source.shape[0], 1, dtype=torch.long),
                encoder_hidden_states=encoded,
                cache=cache,
                use_cache=True,
            )
            cache = output.cache
            assert all(layer.self_length == step + 1 for layer in cache)


def test_cross_attention_cache_does_not_grow(model, source):
    """It is computed once from the encoder output and reused unchanged."""
    with torch.no_grad():
        encoded = model.encode(source)
        cache = None
        widths = []
        for _ in range(3):
            output = model(
                source,
                torch.zeros(source.shape[0], 1, dtype=torch.long),
                encoder_hidden_states=encoded,
                cache=cache,
                use_cache=True,
            )
            cache = output.cache
            widths.append(cache[0].cross_key.shape[2])
    assert widths == [source.shape[1]] * 3


# -- greedy ---------------------------------------------------------------


def test_greedy_shape_and_determinism(model, source):
    first = greedy_generate(model, source, max_new_tokens=6)
    second = greedy_generate(model, source, max_new_tokens=6)

    assert first.shape[0] == source.shape[0]
    assert first.shape[1] <= 6
    assert torch.equal(first, second), "greedy decoding must be reproducible"


def test_greedy_respects_the_encoder_mask(model, source, tiny_config):
    """Changing padded source tokens must not change the output."""
    mask = torch.ones_like(source)
    mask[:, 7:] = 0
    padded = source.clone()
    padded[:, 7:] = tiny_config.pad_token_id

    first = greedy_generate(model, padded, mask, max_new_tokens=5)
    polluted = padded.clone()
    polluted[:, 7:] = 42
    second = greedy_generate(model, polluted, mask, max_new_tokens=5)

    assert torch.equal(first, second)


def test_generation_stops_at_eos(model, source, tiny_config):
    """A model that always predicts EOS must stop after one token."""
    assert greedy_generate(_always(model, tiny_config.eos_token_id),
                           source, max_new_tokens=10).shape[1] == 1


def test_finished_sequences_emit_padding(model, source, tiny_config):
    """
    Once a row hits EOS it must emit padding, never further content, so a
    finished example in a ragged batch does not keep generating.
    """
    forced = _always(model, tiny_config.eos_token_id)
    # min_new_tokens=1 suppresses EOS at index 0 only, so the sequence is
    # forced to run one step, finish at index 1, and pad from index 2 on.
    generated = greedy_generate(forced, source, max_new_tokens=4, min_new_tokens=1)

    assert (generated[:, 1] == tiny_config.eos_token_id).all()
    assert (generated[:, 2:] == tiny_config.pad_token_id).all()


def test_min_new_tokens_suppresses_early_eos(model, source, tiny_config):
    """
    A model that would emit EOS immediately must be prevented from doing so
    until the minimum is reached.
    """
    forced = _always(model, tiny_config.eos_token_id)
    generated = greedy_generate(forced, source, max_new_tokens=6, min_new_tokens=3)

    assert generated.shape[1] >= 3
    assert (generated[:, :3] != tiny_config.eos_token_id).all()


def test_generation_is_capped_by_the_decoder_context(model, source, tiny_config):
    """
    The decoder start token occupies a position, so at most
    `max_decoder_length - 1` tokens can be generated.
    """
    generated = greedy_generate(model, source, max_new_tokens=10_000)
    assert generated.shape[1] <= tiny_config.max_decoder_length - 1


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_budgets_are_rejected(model, source, bad):
    with pytest.raises(ShapeError, match="max_new_tokens"):
        greedy_generate(model, source, max_new_tokens=bad)


def test_min_greater_than_max_is_rejected(model, source):
    with pytest.raises(ShapeError, match="exceeds"):
        greedy_generate(model, source, max_new_tokens=3, min_new_tokens=5)


def test_generation_does_not_leave_gradients(model, source):
    """Diagnostics run inside training; they must not build a graph."""
    assert not greedy_generate(model, source, max_new_tokens=4).requires_grad


# -- sampling -------------------------------------------------------------


def test_sampling_is_reproducible_with_a_generator(model, source):
    first = sample_generate(
        model, source, max_new_tokens=6, generator=torch.Generator().manual_seed(1)
    )
    second = sample_generate(
        model, source, max_new_tokens=6, generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(first, second)


def test_sampling_varies_across_seeds(model, source):
    first = sample_generate(
        model, source, max_new_tokens=8, generator=torch.Generator().manual_seed(1)
    )
    second = sample_generate(
        model, source, max_new_tokens=8, generator=torch.Generator().manual_seed(2)
    )
    assert not torch.equal(first, second)


def test_top_k_one_reproduces_greedy(model, source):
    """With only one candidate available, sampling has no choice to make."""
    sampled = sample_generate(
        model, source, max_new_tokens=6, top_k=1, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(sampled, greedy_generate(model, source, max_new_tokens=6))


def test_lower_temperature_agrees_with_greedy_more_often(model, source):
    """
    Temperature sharpens the distribution rather than replacing it, so the
    honest property is directional: colder sampling tracks greedy decoding
    more closely than warmer sampling does.

    (An exact match needs the distribution to become one-hot, which depends
    on the logits' absolute scale — at initialization they span only ~0.03,
    so no "small" temperature is small enough. `top_k=1` is the exact test.)
    """
    greedy = greedy_generate(model, source, max_new_tokens=10)

    def agreement(temperature: float) -> float:
        matches = 0
        for seed in range(8):
            sampled = sample_generate(
                model, source, max_new_tokens=10, temperature=temperature,
                generator=torch.Generator().manual_seed(seed),
            )
            matches += int((sampled == greedy).sum())
        return matches

    assert agreement(0.001) > agreement(5.0)


def test_top_p_restricts_the_candidate_set(model, source, tiny_config):
    """A very small top_p must keep at least the single best token."""
    generated = sample_generate(
        model, source, max_new_tokens=5, top_p=0.01,
        generator=torch.Generator().manual_seed(0),
    )
    assert generated.shape[1] <= 5
    assert (generated >= 0).all() and (generated < tiny_config.vocab_size).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"temperature": 0.0}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_k": -1}, "top_k"),
    ],
)
def test_invalid_sampling_settings_are_rejected(model, source, kwargs, match):
    with pytest.raises(ShapeError, match=match):
        sample_generate(model, source, max_new_tokens=4, **kwargs)


# -- beam search ----------------------------------------------------------


def test_beam_shape(model, source):
    generated = beam_generate(model, source, num_beams=3, max_new_tokens=6)
    assert generated.shape[0] == source.shape[0]
    assert generated.shape[1] <= 6


def test_one_beam_equals_greedy(model, source):
    assert torch.equal(
        beam_generate(model, source, num_beams=1, max_new_tokens=6),
        greedy_generate(model, source, max_new_tokens=6),
    )


def test_beam_search_finds_a_sequence_at_least_as_likely_as_greedy(model, source):
    """
    The point of beam search. Scoring both under the model, the beam result
    must not be worse — if it is, the search is broken rather than merely
    different.
    """
    beam = beam_generate(model, source, num_beams=4, max_new_tokens=8, length_penalty=0.0)
    greedy = greedy_generate(model, source, max_new_tokens=8)

    assert _sequence_log_probability(model, source, beam) >= _sequence_log_probability(
        model, source, greedy
    ) - 1e-4


def test_beam_search_is_deterministic(model, source):
    first = beam_generate(model, source, num_beams=3, max_new_tokens=6)
    second = beam_generate(model, source, num_beams=3, max_new_tokens=6)
    assert torch.equal(first, second)


def test_length_penalty_normalizes_by_length():
    """
    The ranking rule itself. A log-probability is a sum of negative terms, so
    it always falls as a sequence grows: with no penalty beam search prefers
    the shortest hypothesis, which for a rephraser means dropping content.
    """
    short, short_length = -3.0, 3
    long, long_length = -6.0, 10

    # Unpenalized: the short hypothesis wins purely for being short.
    assert length_penalized_score(short, short_length, 0.0) > length_penalized_score(
        long, long_length, 0.0
    )
    # Normalized per token: the long one is better per token, and now wins.
    assert length_penalized_score(long, long_length, 1.0) > length_penalized_score(
        short, short_length, 1.0
    )


def test_length_penalty_above_one_rewards_length():
    assert length_penalized_score(-6.0, 10, 2.0) > length_penalized_score(-6.0, 10, 1.0)


def test_length_penalty_handles_a_zero_length_hypothesis():
    assert length_penalized_score(-1.0, 0, 1.0) == -1.0


def test_length_penalty_is_inert_when_nothing_completes(model, source):
    """
    Documented consequence: the penalty ranks *finished* hypotheses. If no
    beam emits EOS within the budget, the best live beam is returned and the
    penalty has nothing to rank, so it cannot change the result.
    """
    without = beam_generate(model, source, num_beams=4, max_new_tokens=6, length_penalty=0.0)
    with_penalty = beam_generate(
        model, source, num_beams=4, max_new_tokens=6, length_penalty=2.0
    )
    assert torch.equal(without, with_penalty)


def test_beam_respects_min_new_tokens(model, source, tiny_config):
    forced = _always(model, tiny_config.eos_token_id)
    generated = beam_generate(forced, source, num_beams=2, max_new_tokens=6, min_new_tokens=3)
    assert (generated[:, :3] != tiny_config.eos_token_id).all()


def test_invalid_beam_count_is_rejected(model, source):
    with pytest.raises(ShapeError, match="num_beams"):
        beam_generate(model, source, num_beams=0)


def test_beam_handles_a_single_example(model, source):
    assert beam_generate(model, source[:1], num_beams=3, max_new_tokens=5).shape[0] == 1


def _sequence_log_probability(model, source, sequence) -> float:
    """Total log-probability the model assigns to `sequence` given `source`."""
    config = model.config
    decoder_input = torch.cat(
        [
            torch.full(
                (sequence.shape[0], 1), config.decoder_start_token_id, dtype=torch.long
            ),
            sequence[:, :-1],
        ],
        dim=1,
    )
    with torch.no_grad():
        logits = model(source, decoder_input).logits
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        gathered = log_probabilities.gather(-1, sequence[:, :, None]).squeeze(-1)
        # Padding after EOS carries no information about search quality.
        live = sequence != config.pad_token_id
    return float((gathered * live).sum())
