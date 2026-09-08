"""
Packing, as the assembled pipeline actually runs it.

`UL2/test/unit/test_packing.py` proves the packer builds correct windows from
documents handed to it. This file proves the surrounding wiring — corpus ->
buffer -> mode choice -> packed window -> padded batch — behaves, including
the two things only the assembled pipeline can get wrong:

  - the buffered documents that a window has consumed from the corpus but not
    yet emitted must survive a checkpoint, or every interruption of a
    multi-week run silently drops up to a window's worth of records;
  - held-out evaluation must be assembled the same way training is, or the
    loss selecting the best checkpoint is measured on a distribution the model
    never trains on.
"""

from __future__ import annotations

import pytest

import pretrain

pytestmark = pytest.mark.skipif(
    not pretrain.DEFAULT_TOKENIZER_DIR.joinpath("spm.model").is_file(),
    reason="tokenizer not built: tokenizer/training/output/spm.model missing",
)


def build_source(
    corpus_dir,
    *,
    pack=True,
    batch_size=2,
    seed=1234,
    bucket_batches=True,
    pool_batches=pretrain.ul2.DEFAULT_POOL_BATCHES,
):
    """A `PretrainBatchSource` over real corpus files and the real tokenizer."""
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(max_source_length=256, max_target_length=256)
    objective = pretrain.ul2.UL2Objective(ul2_config, specials)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    corpus = pretrain.corpus_pkg.TokenizedCorpus(
        pretrain.corpus_pkg.discover_corpus_files(corpus_dir), sp, seed=seed
    )
    return pretrain.PretrainBatchSource(
        corpus,
        objective,
        specials,
        batch_size=batch_size,
        data_seed=seed,
        pack=pack,
        bucket_batches=bucket_batches,
        pool_batches=pool_batches,
    )


# -- packing actually happens ----------------------------------------------


def test_packed_batches_carry_segment_ids(mini_corpus_dir):
    source = build_source(mini_corpus_dir)
    batches = [next(source) for _ in range(8)]
    assert any("encoder_segment_ids" in batch for batch in batches), (
        "no batch was packed; the packing path is not being taken at all"
    )


def test_packing_produces_longer_examples_than_one_document_per_window(mini_corpus_dir):
    """
    The whole point, measured. Packed windows must carry materially more real
    content per example than the unpacked path, or the padding this change
    exists to remove is still there.
    """

    def mean_real_tokens(source):
        total = count = 0
        # Enough batches that the per-batch mode draw averages out: one batch
        # carries a single mode, and [S] batches are narrower by design.
        for _ in range(60):
            batch = next(source)
            for row in batch["encoder_attention_mask"]:
                total += sum(row)
                count += 1
        return total / count

    packed = mean_real_tokens(build_source(mini_corpus_dir, pack=True))
    unpacked = mean_real_tokens(build_source(mini_corpus_dir, pack=False))
    assert packed > unpacked * 1.5, f"packed={packed:.1f} vs unpacked={unpacked:.1f}"


def test_prefix_denoising_windows_are_never_packed(mini_corpus_dir):
    """
    [S] must stay one document per window — a packed [S] target could not say
    which prefix each continuation belongs to.
    """
    source = build_source(mini_corpus_dir, batch_size=4)
    for _ in range(15):
        batch = next(source)
        if "encoder_segment_ids" not in batch:
            continue
        for mode, segments, mask in zip(
            batch["modes"], batch["encoder_segment_ids"], batch["encoder_attention_mask"]
        ):
            if mode == "S":
                real = [s for s, m in zip(segments, mask) if m == 1]
                assert set(real) == {1}, "an [S] window held more than one document"


def test_every_packed_window_stays_inside_the_encoder_budget(mini_corpus_dir):
    source = build_source(mini_corpus_dir, batch_size=4)
    for _ in range(15):
        batch = next(source)
        for row in batch["encoder_attention_mask"]:
            assert sum(row) <= 256


# -- resumability ----------------------------------------------------------


def test_resuming_reproduces_the_batches_an_uninterrupted_run_would_make(
    mini_corpus_dir,
):
    """
    The property architecture.md §7.9 is about, now that a batch source also
    holds buffered documents: a resumed run must produce exactly the examples
    the uninterrupted run would have produced next — not merely valid ones.
    """
    straight = build_source(mini_corpus_dir)
    expected = [next(straight)["encoder_input_ids"] for _ in range(6)]

    interrupted = build_source(mini_corpus_dir)
    before = [next(interrupted)["encoder_input_ids"] for _ in range(3)]
    state = interrupted.state_dict()

    resumed = build_source(mini_corpus_dir)
    resumed.load_state_dict(state)
    after = [next(resumed)["encoder_input_ids"] for _ in range(3)]

    assert before + after == expected


def test_the_unpacked_buffer_is_part_of_the_saved_state(mini_corpus_dir):
    """
    Directly pins the failure mode: documents read from the corpus but not yet
    packed live only in the source's buffer. Left out of `state_dict`, they
    would be lost at every resume, which over a run that is interrupted often
    is real, silent data loss.
    """
    source = build_source(mini_corpus_dir)
    next(source)
    state = source.state_dict()
    assert "buffer" in state
    assert isinstance(state["buffer"], list)

    restored = build_source(mini_corpus_dir)
    restored.load_state_dict(state)
    assert restored._buffer == source._buffer


def test_a_checkpoint_written_before_packing_existed_still_loads(mini_corpus_dir):
    """Old checkpoints carry no `buffer` key; they must resume, not crash."""
    source = build_source(mini_corpus_dir)
    next(source)
    state = source.state_dict()
    del state["buffer"]

    restored = build_source(mini_corpus_dir)
    restored.load_state_dict(state)
    assert restored._buffer == []
    assert next(restored) is not None


def test_the_saved_state_is_json_serializable(mini_corpus_dir):
    import json

    source = build_source(mini_corpus_dir)
    next(source)
    assert json.loads(json.dumps(source.state_dict()))["buffer"] is not None


# -- evaluation is assembled the same way ----------------------------------


def test_frozen_eval_batches_are_packed_like_training(mini_corpus_dir):
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(max_source_length=256, max_target_length=256)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    batches = pretrain.build_frozen_eval_batches(
        pretrain.corpus_pkg.discover_corpus_files(mini_corpus_dir),
        sp,
        ul2_config,
        specials,
        seed=1,
        batch_size=4,
        max_examples=64,
        pack=True,
    )
    assert batches
    assert any("encoder_segment_ids" in batch for batch in batches)


def test_a_frozen_eval_set_still_verifies_its_own_digest(mini_corpus_dir):
    """
    Packed examples carry extra fields; the frozen set's self-check has to
    cover them, or an edited set would load without complaint.
    """
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(max_source_length=256, max_target_length=256)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    escape = pretrain.corpus_pkg.tokenizer_interop.escape_markers
    sequences = [
        sp.encode(escape(record), out_type=int)
        for record in pretrain.corpus_pkg.iter_records(
            pretrain.corpus_pkg.discover_corpus_files(mini_corpus_dir)
        )
    ][:32]

    frozen = pretrain.ul2.build_frozen_eval_set(
        sequences, ul2_config, specials, seed=3, pack=True
    )
    rebuilt = pretrain.ul2.build_frozen_eval_set(
        sequences, ul2_config, specials, seed=3, pack=True
    )
    assert frozen.digest == rebuilt.digest, "packed frozen sets are not reproducible"


def test_a_packed_frozen_set_survives_a_save_and_load(mini_corpus_dir, tmp_path):
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(max_source_length=256, max_target_length=256)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    escape = pretrain.corpus_pkg.tokenizer_interop.escape_markers
    sequences = [
        sp.encode(escape(record), out_type=int)
        for record in pretrain.corpus_pkg.iter_records(
            pretrain.corpus_pkg.discover_corpus_files(mini_corpus_dir)
        )
    ][:32]

    frozen = pretrain.ul2.build_frozen_eval_set(
        sequences, ul2_config, specials, seed=5, pack=True
    )
    path = frozen.save(tmp_path / "frozen.jsonl")
    reloaded = pretrain.ul2.FrozenEvalSet.load(path)

    assert reloaded.digest == frozen.digest
    assert reloaded.examples == frozen.examples
    # The segment ids specifically must survive the round trip, or a reloaded
    # set would evaluate with document isolation silently switched off.
    packed = [e for e in reloaded.examples if e.segment_ids]
    assert packed, "no packed example in the reloaded set"


# -- one bad document must not take a whole window with it ------------------


def test_a_short_document_does_not_discard_the_window_it_landed_in(mini_corpus_dir):
    """
    A window is corrupted as a unit, so an inadmissible document reaching the
    objective makes it reject *every* document packed alongside it. With up to
    64 documents per window that turns one unusable record into the silent
    loss of dozens of good ones.

    Admission therefore happens before a document joins a window. This pins
    that: a corpus salted with too-short records must still yield batches, and
    the good records must still come through.
    """

    class SaltedCorpus:
        """Yields a too-short document before every real one."""

        def __init__(self, inner):
            self.inner = inner
            self.turn = 0

        def __next__(self):
            self.turn += 1
            if self.turn % 2 == 1:
                return [5, 6, 7]  # below min_source_length
            return next(self.inner)

        def state_dict(self):
            return self.inner.state_dict()

        def load_state_dict(self, state):
            self.inner.load_state_dict(state)

    source = build_source(mini_corpus_dir)
    source.corpus = SaltedCorpus(source.corpus)

    # Several batches, because one batch carries a single mode and an [S]
    # batch over this fixture's short records is legitimately narrow — the
    # question is whether real content survives at all, not how wide one
    # arbitrary batch happens to be.
    widest = 0
    for _ in range(12):
        batch = next(source)
        assert batch["encoder_input_ids"], "every window was discarded"
        widest = max(widest, *(sum(row) for row in batch["encoder_attention_mask"]))

    # The salted records are 3 tokens each and are filtered out before packing,
    # so anything wider than that came from a real record that survived.
    assert widest > 10, (
        "only trivial content survived; good documents were discarded with the bad"
    )


def test_skipped_documents_are_counted_rather_than_vanishing(mini_corpus_dir):
    """Silent loss is the failure mode; telemetry is what makes it loud."""
    specials = pretrain.ul2.SpecialTokens.from_token_map(
        pretrain.DEFAULT_TOKENIZER_DIR / "token_map.json"
    )
    ul2_config = pretrain.ul2.UL2Config(max_source_length=256, max_target_length=256)
    telemetry = pretrain.ul2.MixtureTelemetry(ul2_config.mode_weights)
    objective = pretrain.ul2.UL2Objective(ul2_config, specials, telemetry)
    sp = pretrain.corpus_pkg.tokenizer_interop.load_processor(
        pretrain.DEFAULT_TOKENIZER_DIR / "spm.model"
    )
    corpus = pretrain.corpus_pkg.TokenizedCorpus(
        pretrain.corpus_pkg.discover_corpus_files(mini_corpus_dir), sp, seed=1234
    )

    class ShortFirst:
        def __init__(self, inner):
            self.inner = inner
            self.turn = 0

        def __next__(self):
            self.turn += 1
            return [5, 6, 7] if self.turn % 2 == 1 else next(self.inner)

    source = pretrain.PretrainBatchSource(
        ShortFirst(corpus), objective, specials, batch_size=2, data_seed=1234
    )
    next(source)
    assert telemetry.skipped_too_short > 0


# -- one mode per batch -----------------------------------------------------
#
# The three modes produce very different window lengths ([R]/[X] pack to the
# full budget, [S] never packs), and `pad_batch` pads every row to the longest
# row present. Drawing one mode per batch keeps a batch's rows on one length
# scale; mixing them pads every short row out to a packed row's width, and
# padding costs exactly as much compute as real content while teaching
# nothing.


def test_every_example_in_a_packed_batch_shares_one_mode(mini_corpus_dir):
    source = build_source(mini_corpus_dir, batch_size=8)
    for _ in range(20):
        modes = set(next(source)["modes"])
        assert len(modes) == 1, f"batch mixed modes {modes}; padding will be wasted"


def test_an_unpacked_run_still_samples_a_mode_per_example(mini_corpus_dir):
    """
    Unpacked examples are single documents that already share a length scale,
    so they keep the per-example sampling they had before packing existed.
    """
    source = build_source(mini_corpus_dir, pack=False, batch_size=8)
    seen = set()
    for _ in range(20):
        seen.update(next(source)["modes"])
    assert len(seen) > 1


def test_the_mode_mixture_still_matches_the_configured_weights(mini_corpus_dir):
    """
    Grouping changes *when* the dice are rolled, never the odds. If per-batch
    sampling drifted the realized mixture away from `mode_weights`, the model
    would train on a different objective blend than the configuration
    describes — and nothing else in the pipeline would notice.
    """
    source = build_source(mini_corpus_dir, batch_size=4)
    counts = {"R": 0, "X": 0, "S": 0}
    batches = 400
    for _ in range(batches):
        counts[next(source)["modes"][0]] += 1

    weights = pretrain.ul2.UL2Config().mode_weights
    for mode, expected in weights.items():
        realized = counts[mode] / batches
        # Generous band: 400 draws of a 3-way categorical is noisy, and the
        # point is to catch a systematic shift, not to re-test the sampler.
        assert abs(realized - expected) < 0.08, (
            f"[{mode}] realized {realized:.3f} vs configured {expected:.3f}"
        )


def test_batching_by_mode_wastes_less_of_each_batch_on_padding(mini_corpus_dir):
    """
    The measurable point of the change, pinned so a regression to per-example
    sampling shows up as a failure rather than as a quietly slower run.

    Padding is not free: those positions run through every matmul and every
    attention head exactly as real tokens do, and contribute nothing.
    """

    def efficiency(per_example_modes: bool):
        source = build_source(mini_corpus_dir, batch_size=8)
        if per_example_modes:
            objective, rng = source.objective, source.rng
            original = source._next_example
            source._batch_mode = lambda: "X"
            source._next_example = lambda _: original(objective.sampler.sample(rng))
        real = padded = 0
        for _ in range(60):
            batch = next(source)
            for row in batch["encoder_attention_mask"]:
                real += sum(row)
                padded += len(row)
        return real / padded

    by_batch = efficiency(per_example_modes=False)
    by_example = efficiency(per_example_modes=True)
    assert by_batch > by_example, (
        f"per-batch modes wasted more padding than per-example "
        f"({by_batch:.1%} vs {by_example:.1%})"
    )


def test_resuming_is_still_exact_with_per_batch_modes(mini_corpus_dir):
    """
    The mode is drawn from the same RNG the corruption uses, so the draw is
    part of the stream a resume has to reproduce. Re-checked here because the
    change moved where that draw happens.
    """
    straight = build_source(mini_corpus_dir)
    expected = [next(straight)["encoder_input_ids"] for _ in range(6)]

    interrupted = build_source(mini_corpus_dir)
    before = [next(interrupted)["encoder_input_ids"] for _ in range(3)]
    state = interrupted.state_dict()

    resumed = build_source(mini_corpus_dir)
    resumed.load_state_dict(state)
    after = [next(resumed)["encoder_input_ids"] for _ in range(3)]

    assert before + after == expected


# -- length bucketing, in the assembled pipeline ----------------------------


def test_bucketing_wastes_less_of_each_batch_on_padding(mini_corpus_dir):
    """
    The measurable point of length bucketing, in the real pipeline rather
    than on synthetic lengths (`UL2/test/unit/test_length_batching.py`).

    Pinned as a comparison against the same source with bucketing off, so the
    assertion stays meaningful if the corpus fixture or the window budget
    changes -- an absolute efficiency threshold would not.
    """

    def efficiency(bucket_batches: bool) -> float:
        source = build_source(
            mini_corpus_dir, batch_size=8, bucket_batches=bucket_batches, pool_batches=16
        )
        real = padded = 0
        for _ in range(40):
            batch = next(source)
            for row in batch["encoder_attention_mask"]:
                real += sum(row)
                padded += len(row)
        return real / padded

    bucketed = efficiency(True)
    plain = efficiency(False)
    assert bucketed > plain, (
        f"bucketing wasted more padding than not bucketing "
        f"({bucketed:.1%} vs {plain:.1%})"
    )


def test_rows_within_a_bucketed_batch_are_close_in_length(mini_corpus_dir):
    """The mechanism, not just its effect: batch-mates should be similar sizes."""
    source = build_source(mini_corpus_dir, batch_size=8, pool_batches=16)
    spreads = []
    for _ in range(30):
        widths = [sum(row) for row in next(source)["encoder_attention_mask"]]
        spreads.append(max(widths) - min(widths))

    plain = build_source(mini_corpus_dir, batch_size=8, bucket_batches=False)
    plain_spreads = []
    for _ in range(30):
        widths = [sum(row) for row in next(plain)["encoder_attention_mask"]]
        plain_spreads.append(max(widths) - min(widths))

    assert sum(spreads) / len(spreads) < sum(plain_spreads) / len(plain_spreads)


def test_bucketing_does_not_reorder_batches_into_length_order(mini_corpus_dir):
    """
    Sorting a pool and emitting it in order would walk the model short-to-long
    once per pool -- a periodic signal correlated with sequence length, which
    is exactly what shuffling exists to remove.
    """
    source = build_source(mini_corpus_dir, batch_size=4, pool_batches=8)
    # One mode only, so the widths are comparable to each other; [S] because
    # its natural length spread is the widest and makes ordering visible.
    widths = []
    for _ in range(60):
        batch = next(source)
        if batch["modes"][0] == "S":
            widths.append(len(batch["encoder_input_ids"][0]))

    assert len(widths) >= 8, f"too few [S] batches to judge ordering: {len(widths)}"
    assert widths != sorted(widths), (
        f"every [S] batch came out at least as wide as the last; the pool is "
        f"being emitted in sorted order: {widths}"
    )


def test_bucketing_still_gives_every_batch_one_mode(mini_corpus_dir):
    """Pools are per mode, so bucketing must not mix them back together."""
    source = build_source(mini_corpus_dir, batch_size=8)
    for _ in range(20):
        assert len(set(next(source)["modes"])) == 1


def test_bucketing_preserves_the_configured_mode_mixture(mini_corpus_dir):
    """
    Bucketing regroups examples; it must not reweight which modes appear.
    A per-mode pool that refilled greedily could easily skew this.
    """
    source = build_source(mini_corpus_dir, batch_size=4)
    counts = {"R": 0, "X": 0, "S": 0}
    for _ in range(300):
        counts[next(source)["modes"][0]] += 1

    weights = source.objective.config.mode_weights
    for mode, count in counts.items():
        assert abs(count / 300 - weights[mode]) < 0.08, f"[{mode}] {counts}"


def test_the_bucketing_pool_is_part_of_the_saved_state(mini_corpus_dir):
    """
    The same failure `buffer` guards against, one stage later: the pool holds
    examples already read from the corpus, so leaving it out of `state_dict`
    would lose them at every resume.
    """
    source = build_source(mini_corpus_dir, batch_size=4)
    next(source)
    state = source.state_dict()
    assert "batcher" in state
    assert source._batcher.pending_examples() > 0

    restored = build_source(mini_corpus_dir, batch_size=4)
    restored.load_state_dict(state)
    assert restored._batcher.pending_examples() == source._batcher.pending_examples()


def test_resuming_is_exact_with_bucketing_enabled(mini_corpus_dir):
    """
    §7.9's property, re-checked with a second buffer in the path: a resumed
    run must produce exactly the batches the uninterrupted run would have.
    """
    straight = build_source(mini_corpus_dir, batch_size=4)
    expected = [next(straight)["encoder_input_ids"] for _ in range(8)]

    interrupted = build_source(mini_corpus_dir, batch_size=4)
    before = [next(interrupted)["encoder_input_ids"] for _ in range(4)]
    state = interrupted.state_dict()

    resumed = build_source(mini_corpus_dir, batch_size=4)
    resumed.load_state_dict(state)
    after = [next(resumed)["encoder_input_ids"] for _ in range(4)]

    assert before + after == expected


def test_a_checkpoint_written_before_bucketing_existed_still_loads(mini_corpus_dir):
    """Old checkpoints carry no `batcher` key; they must resume, not crash."""
    source = build_source(mini_corpus_dir, batch_size=4)
    next(source)
    state = source.state_dict()
    del state["batcher"]

    restored = build_source(mini_corpus_dir, batch_size=4)
    restored.load_state_dict(state)
    assert next(restored) is not None


def test_the_bucketed_state_is_json_serializable(mini_corpus_dir):
    """`checkpoint.py` writes the data position into trainer_state.json."""
    import json

    source = build_source(mini_corpus_dir, batch_size=4)
    next(source)
    round_tripped = json.loads(json.dumps(source.state_dict()))

    restored = build_source(mini_corpus_dir, batch_size=4)
    restored.load_state_dict(round_tripped)
    assert next(restored) is not None


def test_bucketing_can_be_turned_off(mini_corpus_dir):
    """
    Plug-and-play in both directions: disabling it must restore the previous
    behaviour exactly, including carrying no bucketing state.
    """
    source = build_source(mini_corpus_dir, batch_size=4, bucket_batches=False)
    batch = next(source)
    assert len(batch["modes"]) == 4
    assert "batcher" not in source.state_dict()


def test_bucketed_batches_still_pass_the_model_signature(mini_corpus_dir):
    """Bucketing must not disturb the keys or shapes the trainer feeds the model."""
    source = build_source(mini_corpus_dir, batch_size=4)
    for _ in range(10):
        batch = next(source)
        width = len(batch["encoder_input_ids"][0])
        assert all(len(row) == width for row in batch["encoder_input_ids"])
        assert all(len(row) == width for row in batch["encoder_attention_mask"])
        if "encoder_segment_ids" in batch:
            assert all(len(row) == width for row in batch["encoder_segment_ids"])
        target = len(batch["labels"][0])
        assert all(len(row) == target for row in batch["labels"])
        assert all(len(row) == target for row in batch["decoder_input_ids"])
