"""
`BlendedCorpus` — the weighted source mixture the WSD cooldown reads from
(architecture_improvements.md item 2).

Three things are worth locking down here, in descending order of how badly a
regression would hurt:

  1. **It resumes exactly.** Same shape as `test_tokenized_corpus.py`'s
     headline test and `training/test/integration/test_resumability.py`'s: a
     stream split by a simulated crash must continue as an uninterrupted one
     would. The mixture makes this harder than a single stream, because the
     draw order *and* every child's file position have to come back.
  2. **It delivers the shares it was configured with.** A blend that quietly
     samples something else is the worst kind of bug here: the loss curve
     looks fine, and the only evidence is a downstream eval that came out
     lower than expected weeks later.
  3. **It refuses a mixture it cannot deliver.** Naming a source the corpus
     does not have, or one it has no weight for, is a configuration error and
     must not be renormalized around.
"""

from __future__ import annotations

import itertools
import json

import pytest

from src.blended_corpus import BlendedCorpus, build_blended_corpus
from src.errors import CorpusConfigError
from src.reader import discover_corpus_files, group_files_by_source, source_name
from src.tokenized_corpus import TokenizedCorpus

from conftest import write_jsonl


def make_corpus_tree(root, sources: dict[str, int]):
    """
    `pretrain_corpus/`'s real layout: one directory per source, sharded files
    inside it. Each record's text names its own source, so a drawn document
    can be attributed without guessing.
    """
    for name, count in sources.items():
        directory = root / name
        directory.mkdir(parents=True)
        write_jsonl(
            directory / f"{name}-w00-00000.jsonl",
            [{"text": f"{name} record {i}"} for i in range(count)],
        )
    return root


def streams(root, sp, names, seed=0):
    grouped = group_files_by_source(discover_corpus_files(root), root)
    return {name: TokenizedCorpus(grouped[name], sp, seed=seed) for name in names}


def decode(sp, token_ids) -> str:
    """`FakeProcessor` is one int per character, offset by 5 — invert it."""
    return "".join(chr((value - 5)) for value in token_ids)


# -- source naming ---------------------------------------------------------


def test_a_sources_name_is_its_directory(tmp_path):
    root = make_corpus_tree(tmp_path / "corpus", {"cli_tldr": 1})
    path = root / "cli_tldr" / "cli_tldr-w00-00000.jsonl"
    assert source_name(path, root) == "cli_tldr"


def test_a_file_in_the_root_is_its_own_source(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    write_jsonl(root / "loose.jsonl", [{"text": "x"}])
    assert source_name(root / "loose.jsonl", root) == "loose"


def test_a_symlink_is_named_by_where_it_sits_not_where_it_points(tmp_path):
    """
    `tokenizer_experiment/flattened_corpus/` is a farm of symlinks into the
    real corpus. Following them would report the *target's* source rather than
    the layout the caller handed over, which is a different answer to a
    different question — and one that would silently regroup a corpus the
    caller had deliberately flattened.
    """
    real = make_corpus_tree(tmp_path / "corpus", {"c4": 1})
    flat = tmp_path / "flat"
    flat.mkdir()
    link = flat / "c4_0mb.jsonl"
    link.symlink_to(real / "c4" / "c4-w00-00000.jsonl")
    assert source_name(link, flat) == "c4_0mb"


def test_grouping_partitions_every_file_exactly_once(tmp_path):
    root = make_corpus_tree(tmp_path / "corpus", {"c4": 2, "cli_tldr": 2})
    files = discover_corpus_files(root)
    grouped = group_files_by_source(files, root)
    assert sorted(grouped) == ["c4", "cli_tldr"]
    assert sum(len(paths) for paths in grouped.values()) == len(files)


# -- the mixture itself ----------------------------------------------------


def test_documents_come_from_the_sources_in_roughly_their_weights(
    tmp_path, fake_processor
):
    """
    Not an exact count — the draw is random. The tolerance is wide enough that
    it cannot fail by chance and narrow enough that a swapped or ignored
    weight cannot pass.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"big": 50, "small": 50})
    corpus = BlendedCorpus(
        streams(root, fake_processor, ["big", "small"]),
        {"big": 0.8, "small": 0.2},
        seed=7,
    )
    drawn = [decode(fake_processor, next(corpus)).split()[0] for _ in range(2000)]
    assert 0.75 < drawn.count("big") / len(drawn) < 0.85


def test_weights_are_relative_and_normalized(tmp_path, fake_processor):
    """
    `{a: 3, b: 1}` and `{a: 0.75, b: 0.25}` are the same mixture. Writing them
    to sum to 1.0 in the YAML is a readability choice, not a requirement, and
    a config that drifts to 0.999 must not silently become a different run.
    """
    corpus = BlendedCorpus(
        streams(make_corpus_tree(tmp_path / "corpus", {"a": 4, "b": 4}),
                fake_processor, ["a", "b"]),
        {"a": 3.0, "b": 1.0},
        seed=1,
    )
    assert corpus.weights == {"a": 0.75, "b": 0.25}


def test_a_source_smaller_than_its_share_wraps_rather_than_running_out(
    tmp_path, fake_processor
):
    """
    The case the cooldown is built around: cli_cheat_sheets is ~67,000 tokens
    and is deliberately given weight anyway. It must repeat, not stop — and
    `source_epochs` must say how often, because "how many times did we show
    the model the same file" is the number that decides whether an upweight
    was a good idea.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"tiny": 2, "large": 500})
    corpus = BlendedCorpus(
        streams(root, fake_processor, ["tiny", "large"]),
        {"tiny": 0.5, "large": 0.5},
        seed=3,
    )
    list(itertools.islice(corpus, 200))
    assert corpus.source_epochs()["tiny"] > 10
    assert corpus.source_epochs()["large"] == 0


def test_realized_shares_report_what_was_actually_drawn(tmp_path, fake_processor):
    root = make_corpus_tree(tmp_path / "corpus", {"a": 20, "b": 20})
    corpus = BlendedCorpus(
        streams(root, fake_processor, ["a", "b"]), {"a": 0.5, "b": 0.5}, seed=11
    )
    list(itertools.islice(corpus, 400))
    realized = corpus.realized_shares()
    assert sum(realized.values()) == pytest.approx(1.0)
    assert 0.4 < realized["a"] < 0.6


def test_an_unread_blend_reports_zero_shares_rather_than_dividing_by_zero(
    tmp_path, fake_processor
):
    corpus = BlendedCorpus(
        streams(make_corpus_tree(tmp_path / "corpus", {"a": 1}), fake_processor, ["a"]),
        {"a": 1.0},
    )
    assert corpus.realized_shares() == {"a": 0.0}


# -- resumption ------------------------------------------------------------


def test_a_resumed_blend_continues_an_uninterrupted_one_exactly(
    tmp_path, fake_processor
):
    """
    The headline test. Both the draw order and every child stream's position
    have to come back, so this compares a long uninterrupted run against one
    cut in half and restored into a freshly constructed object.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"a": 30, "b": 30, "c": 30})
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}

    uninterrupted = BlendedCorpus(
        streams(root, fake_processor, weights, seed=2), weights, seed=5
    )
    expected = [next(uninterrupted) for _ in range(120)]

    first = BlendedCorpus(
        streams(root, fake_processor, weights, seed=2), weights, seed=5
    )
    got = [next(first) for _ in range(60)]
    saved = json.loads(json.dumps(first.state_dict()))  # JSON round-trip, as §7.9 requires

    second = BlendedCorpus(
        streams(root, fake_processor, weights, seed=2), weights, seed=5
    )
    second.load_state_dict(saved)
    got += [next(second) for _ in range(60)]

    assert got == expected


def test_the_draw_counters_survive_a_resume(tmp_path, fake_processor):
    """
    Without them the epochs and realized shares reported at the end of a
    resumed run would describe only the last leg of it — a number that still
    looks plausible and is not the run's.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"a": 10, "b": 10})
    weights = {"a": 0.5, "b": 0.5}
    first = BlendedCorpus(streams(root, fake_processor, weights), weights, seed=4)
    list(itertools.islice(first, 50))

    second = BlendedCorpus(streams(root, fake_processor, weights), weights, seed=4)
    second.load_state_dict(first.state_dict())
    assert second.documents_drawn == first.documents_drawn


def test_resuming_into_a_different_set_of_sources_is_refused(tmp_path, fake_processor):
    """
    The blend changed, or the corpus did. Either way the saved positions do
    not describe this run, and continuing from them would silently re-read or
    skip whole sources.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"a": 5, "b": 5})
    saved = BlendedCorpus(
        streams(root, fake_processor, {"a": 1, "b": 1}), {"a": 1.0, "b": 1.0}
    ).state_dict()

    narrower = BlendedCorpus(
        streams(root, fake_processor, {"a": 1}), {"a": 1.0}
    )
    with pytest.raises(CorpusConfigError, match="changed since"):
        narrower.load_state_dict(saved)


# -- refusals --------------------------------------------------------------


def test_no_sources_at_all_is_refused(fake_processor):
    with pytest.raises(CorpusConfigError, match="at least one source"):
        BlendedCorpus({}, {})


def test_a_weight_for_a_source_that_is_not_there_is_refused(tmp_path, fake_processor):
    root = make_corpus_tree(tmp_path / "corpus", {"a": 3})
    with pytest.raises(CorpusConfigError, match="not present"):
        BlendedCorpus(
            streams(root, fake_processor, ["a"]), {"a": 0.5, "ghost": 0.5}
        )


def test_a_source_with_no_weight_is_refused(tmp_path, fake_processor):
    """
    An implicit zero is the one reading of a missing weight that cannot be
    told apart from a typo, so it is not offered.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"a": 3, "b": 3})
    with pytest.raises(CorpusConfigError, match="absent from the blend"):
        BlendedCorpus(streams(root, fake_processor, ["a", "b"]), {"a": 1.0})


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_a_non_positive_weight_is_refused(tmp_path, fake_processor, weight):
    root = make_corpus_tree(tmp_path / "corpus", {"a": 3})
    with pytest.raises(CorpusConfigError, match="must be positive"):
        BlendedCorpus(streams(root, fake_processor, ["a"]), {"a": weight})


# -- the builder -----------------------------------------------------------


def test_the_builder_takes_only_the_sources_the_blend_names(tmp_path, fake_processor):
    """
    A cooldown blend is a *subset* of the corpus by design — that is what
    upweighting means — so extra sources on disk are the normal case, not an
    error.
    """
    root = make_corpus_tree(tmp_path / "corpus", {"a": 3, "b": 3, "unused": 3})
    grouped = group_files_by_source(discover_corpus_files(root), root)
    corpus = build_blended_corpus(grouped, fake_processor, {"a": 0.5, "b": 0.5})
    assert sorted(corpus.sources) == ["a", "b"]


def test_the_builder_refuses_a_blend_naming_a_missing_source(tmp_path, fake_processor):
    root = make_corpus_tree(tmp_path / "corpus", {"a": 3})
    grouped = group_files_by_source(discover_corpus_files(root), root)
    with pytest.raises(CorpusConfigError, match="not found under the corpus root"):
        build_blended_corpus(grouped, fake_processor, {"a": 0.5, "ghost": 0.5})


def test_two_sources_do_not_shuffle_their_files_identically(tmp_path, fake_processor):
    """
    Per-source seeds are derived from the source name, not shared. A shared
    seed would give every source the same file permutation, which correlates
    the sources' contents with each other for no reason at all.
    """
    root = tmp_path / "corpus"
    for name in ("alpha", "beta"):
        directory = root / name
        directory.mkdir(parents=True)
        for shard in range(8):
            write_jsonl(
                directory / f"{name}-w{shard:02d}-00000.jsonl", [{"text": f"{name}{shard}"}]
            )
    grouped = group_files_by_source(discover_corpus_files(root), root)
    corpus = build_blended_corpus(grouped, fake_processor, {"alpha": 0.5, "beta": 0.5})
    assert corpus.sources["alpha"].state_dict()["order"] != (
        corpus.sources["beta"].state_dict()["order"]
    )


def test_the_per_source_seed_offset_does_not_depend_on_the_process(tmp_path, fake_processor):
    """
    Python salts `hash()` per process, so a `hash(name)`-derived seed would
    give a resumed run a different file order from the run it continues — a
    silent, seed-shaped bug of exactly the kind `TokenizedCorpus`'s saved
    `order` exists to prevent. Pinned by an explicit expected value.
    """
    from src.blended_corpus import _source_seed_offset

    assert _source_seed_offset("cli_tldr") == sum(ord(c) for c in "cli_tldr") * 31
