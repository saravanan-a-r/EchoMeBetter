"""
The WSD cooldown, as the assembled pipeline actually runs it
(architecture_improvements.md item 2).

`training/test/unit/test_schedule.py` proves the learning-rate curve;
`corpus/test/unit/test_blended_corpus.py` proves the weighted mixture; and
`training/test/integration/test_training_loop.py` proves the trainer announces
each step. Only this file proves the three are wired to each other — that the
mixture actually switches, at the step the *schedule* starts decaying at, and
that a run interrupted either side of that boundary comes back on the right
side of it.

Three ways this can be wrong while looking fine, each with a test below:

  - the boundary lands at the wrong step, so the model anneals on the ordinary
    corpus and the upweighted mixture is never used;
  - the switch happens but the cooldown stream restarts from its first file on
    every resume, so the last 10% of the run re-reads the same few documents;
  - the main stream's position is lost when the switch happens, so a resume
    inside the cooldown loses where pretraining had got to.

None of the three produces an error, a warning, or a visibly odd loss curve.
"""

from __future__ import annotations

import json

import pytest

import pretrain

pytestmark = pytest.mark.skipif(
    not pretrain.DEFAULT_TOKENIZER_DIR.joinpath("spm.model").is_file(),
    reason="tokenizer not built: tokenizer/training/output/spm.model missing",
)

SENTENCES = [
    "The quarterly report showed strong growth in the European market.",
    "Check https://api.example.com/v2/users for the current status code.",
    "The server crashed because the backup connection failed to respond.",
    "The committee reviewed the roadmap and agreed on next quarter priorities.",
]


@pytest.fixture
def sourced_corpus_dir(tmp_path):
    """
    `pretrain_corpus/`'s real layout — one directory per source — because the
    source of a file is derived from that directory and nothing else.
    """
    root = tmp_path / "corpus"
    for source in ("bulk", "special"):
        directory = root / source
        directory.mkdir(parents=True)
        (directory / f"{source}-w00-00000.jsonl").write_text(
            "\n".join(
                json.dumps({"text": f"{source}: {sentence}"})
                for sentence in SENTENCES * 20
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def build_source(
    corpus_dir, *, cooldown_start_step=None, blend=None, seed=1234, max_length=256
):
    """A `PretrainBatchSource` over real files and the real tokenizer."""
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(
        max_source_length=max_length, max_target_length=max_length
    )
    objective = pretrain.ul2.UL2Objective(ul2_config, specials)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    files = pretrain.corpus_pkg.discover_corpus_files(corpus_dir)

    cooldown = None
    if blend is not None:
        cooldown = pretrain.corpus_pkg.build_blended_corpus(
            pretrain.corpus_pkg.group_files_by_source(files, corpus_dir),
            sp,
            blend,
            seed=seed + 104_729,
        )

    return pretrain.PretrainBatchSource(
        pretrain.corpus_pkg.TokenizedCorpus(files, sp, seed=seed),
        objective,
        specials,
        batch_size=2,
        data_seed=seed,
        cooldown_corpus=cooldown,
        cooldown_start_step=cooldown_start_step if cooldown is not None else None,
    )


# -- the switch itself ------------------------------------------------------


def test_the_stream_switches_at_the_configured_step(sourced_corpus_dir):
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=180, blend={"special": 1.0}
    )
    source.set_step(179)
    assert not source.in_cooldown
    source.set_step(180)
    assert source.in_cooldown


def test_the_boundary_step_itself_is_already_the_cooldown(sourced_corpus_dir):
    """
    `cooldown_start_step` is the first step of the decay phase, and the
    schedule already treats it as such (its multiplier is 1.0 there because
    the decay's progress is zero, not because it is still stable). The data
    switch has to agree, or the two halves of the cooldown start one step
    apart.
    """
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=5, blend={"special": 1.0}
    )
    source.set_step(5)
    assert source.in_cooldown


def test_the_cooldown_actually_reads_the_upweighted_source(sourced_corpus_dir):
    """
    The point of the whole exercise. With the blend naming one source only,
    every document after the switch must come from it — and, as a control,
    documents before the switch must not all come from it.
    """
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=2, blend={"special": 1.0}
    )
    sp = source.corpus.sp

    source.set_step(0)
    before = [sp.decode(source._take_document()) for _ in range(40)]
    source.set_step(2)
    after = [sp.decode(source._take_document()) for _ in range(40)]

    assert all(text.startswith("special:") for text in after)
    assert any(not text.startswith("special:") for text in before), (
        "the control failed: the main stream is already only reading 'special', "
        "so this test could not tell a working switch from a broken one"
    )


def test_a_run_without_a_cooldown_never_switches(sourced_corpus_dir):
    """
    The two halves of item 2 are separable: a WSD run with no
    `cooldown_blend` is a pure learning-rate change and must behave exactly as
    it did before this feature existed.
    """
    source = build_source(sourced_corpus_dir)
    for step in (0, 100, 10_000):
        source.set_step(step)
        assert not source.in_cooldown
    assert source.corpus is source.main_corpus


def test_half_a_cooldown_is_refused(sourced_corpus_dir):
    """
    A stream with no step to switch at would never be read; a step with no
    stream to switch to would do nothing at it. Both are configuration errors
    that produce a run quietly missing its annealing phase.
    """
    for half in ({"cooldown_start_step": 10}, {"cooldown_corpus": object()}):
        with pytest.raises(ValueError, match="go together"):
            pretrain.PretrainBatchSource(
                object(), object(), object(), batch_size=1, data_seed=0, **half
            )


# -- resumption across the boundary ----------------------------------------


def test_both_streams_are_checkpointed(sourced_corpus_dir):
    """
    Not only the active one. Mid-run the cooldown stream has read nothing and
    the main stream is far in; inside the cooldown it is the other way round.
    Saving only whichever happens to be selected would restart the other from
    its first file on the next resume.
    """
    source = build_source(
        sourced_corpus_dir, cooldown_start_step=2, blend={"special": 1.0}
    )
    state = json.loads(json.dumps(source.state_dict()))
    assert "corpus" in state and "cooldown_corpus" in state


def test_resuming_inside_the_cooldown_continues_the_mixture(sourced_corpus_dir):
    """
    The headline test: a cooldown split by a simulated crash must produce
    exactly what an uninterrupted one would, from the point of resumption
    onward. Without the cooldown stream in `state_dict` this passes only by
    accident — the first documents of a restarted stream are the same ones —
    so the comparison runs long enough to leave that coincidence behind.
    """
    def documents(source, count):
        return [source._take_document() for _ in range(count)]

    blend = {"special": 0.5, "bulk": 0.5}

    uninterrupted = build_source(
        sourced_corpus_dir, cooldown_start_step=0, blend=blend
    )
    uninterrupted.set_step(0)
    expected = documents(uninterrupted, 120)

    first = build_source(sourced_corpus_dir, cooldown_start_step=0, blend=blend)
    first.set_step(0)
    got = documents(first, 60)
    saved = json.loads(json.dumps(first.state_dict()))

    second = build_source(sourced_corpus_dir, cooldown_start_step=0, blend=blend)
    second.load_state_dict(saved)
    second.set_step(0)
    got += documents(second, 60)

    assert got == expected


def test_a_resume_lands_on_the_side_of_the_boundary_its_step_says(
    sourced_corpus_dir,
):
    """
    The phase is derived from the step on every announcement rather than
    latched the first time the boundary is crossed, which is what makes this
    work without the phase itself being checkpointed.
    """
    blend = {"special": 1.0}
    before = build_source(sourced_corpus_dir, cooldown_start_step=100, blend=blend)
    before.set_step(150)
    saved = json.loads(json.dumps(before.state_dict()))

    resumed = build_source(sourced_corpus_dir, cooldown_start_step=100, blend=blend)
    resumed.load_state_dict(saved)
    assert not resumed.in_cooldown  # nothing announced yet
    resumed.set_step(150)
    assert resumed.in_cooldown
    resumed.set_step(99)
    assert not resumed.in_cooldown


def test_a_checkpoint_written_before_the_cooldown_existed_still_loads(
    sourced_corpus_dir,
):
    """
    Item 1's `buffer` and the length batcher's pool both set this precedent:
    a new key in the saved state must not make every existing checkpoint
    unresumable. A run interrupted before this feature shipped resumes with an
    unread cooldown stream, which is exactly what it had.
    """
    old = build_source(sourced_corpus_dir)
    state = json.loads(json.dumps(old.state_dict()))
    assert "cooldown_corpus" not in state

    new = build_source(
        sourced_corpus_dir, cooldown_start_step=2, blend={"special": 1.0}
    )
    new.load_state_dict(state)  # must not raise
    assert new.cooldown_corpus.source_epochs() == {"special": 0}


# -- the trainer drives it --------------------------------------------------


def test_the_trainer_moves_the_source_into_the_cooldown(
    sourced_corpus_dir, tiny_model_config_path, tiny_training_config_path
):
    """
    End to end through `Trainer.train`, with no test calling `set_step` by
    hand — the announcement path is the only thing that can move the phase in
    a real run, so it is the one that has to work.
    """
    model_config = pretrain.rephrase_model.load_model_config(
        tiny_model_config_path, profile="tiny"
    )
    training_config = pretrain.training_pkg.load_training_config(
        tiny_training_config_path, stage="tiny"
    ).with_(max_steps=4, gradient_accumulation_steps=1)

    # 64 to match the tiny model's max_encoder_length / max_decoder_length:
    # the model refuses an over-long batch rather than truncating it (§4.5).
    source = build_source(
        sourced_corpus_dir,
        cooldown_start_step=2,
        blend={"special": 1.0},
        max_length=64,
    )
    model = pretrain.rephrase_model.build_model(model_config)
    trainer = pretrain.training_pkg.Trainer(model, training_config, device="cpu")

    assert not source.in_cooldown
    trainer.train(source, data=source, max_steps=4)
    assert source.in_cooldown
    assert source.cooldown_corpus.documents_drawn["special"] > 0
