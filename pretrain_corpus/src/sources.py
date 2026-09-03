"""
Where the raw text comes from: one adapter per corpus in architecture.md §9.1.

A source's entire job is to yield raw strings. Everything that happens to
those strings — cleaning, PII masking, quality filtering, deduplication,
budgeting — belongs to `pipeline.py` and `writer.py` and is identical for all
seven sources. That split is what keeps the fidelity rules defined once
instead of five times with five chances to get them subtly wrong.

Two kinds of adapter
--------------------
**HuggingFace streaming** covers six of the seven. `streaming=True` means we
stop the moment the byte budget is met instead of downloading a terabyte to
keep a slice of it. Parallelism comes from sharding: each worker takes
`shard(num_shards, index)` of the stream, which splits by underlying file
where the format allows and therefore actually opens separate connections
rather than pulling one stream through several threads.

**Stack Exchange** is its own thing, because the data is not on the Hub: it is
362 per-site `.7z` archives of XML on archive.org, totalling 98GB compressed.
Download, unpack, parse, in a pipeline deep enough to deserve its own section
below.

Dataset availability, checked rather than assumed
-------------------------------------------------
Two of §9.1's seven need a note, both verified live against the Hub rather
than taken from the design doc:

* **SlimPajama.** `cerebras/SlimPajama-627B` now returns 401 without
  credentials. If `HF_TOKEN` is set the full 627B-token corpus is used; if it
  is not, this falls back to `DKYoon/SlimPajama-6B`, an ungated ~6B-token
  sample of the same data. The fallback caps that source at roughly 24GB, and
  the runner reports the cap instead of quietly under-filling the blend.
* **Permissive code.** `codeparrot/github-code-clean` on its main branch is a
  loading-script dataset, which current `datasets` refuses to execute — the
  same failure that forced `deepmind/pg19` to be swapped for a Parquet
  mirror. Its auto-converted Parquet branch works, so this reads
  `refs/convert/parquet` directly and selects only the permissively licensed
  configs (`all-mit`, `all-apache-2.0`, `all-bsd-3-clause`), which is exactly
  the "Apache/permissive" §9.1 asks for.
"""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .errors import ExtractionError, SourceConfigError
from .pipeline import PipelineConfig

# --------------------------------------------------------------------------
# Source specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One corpus: how to read it, how to treat it, and what it is for."""

    source_id: str
    kind: str  # "hf" | "stackexchange"
    role: str
    license: str
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    # HF-specific
    hf_dataset: str = ""
    hf_config: str | None = None
    hf_revision: str | None = None
    hf_data_files: tuple[str, ...] = ()
    text_field: str = "text"
    shuffle_buffer: int = 10_000
    # How many underlying files a worker's shuffle buffer draws from at once.
    #
    # This is a *memory* knob, not a quality one, and it is the single largest
    # consumer of RAM in a worker. `datasets` >= 5.0 fills the shuffle buffer
    # from `max_buffer_input_shards` files concurrently, holding a separate
    # open Parquet reader (and its download/decode buffers) for each. Measured
    # on this corpus, one reader costs ~2.4GB of resident memory, and the cost
    # is linear in the number of readers: the library default of 10 puts a
    # single FineWeb-Edu worker at ~24GB, which is what OOM-killed a 125GB box
    # running 20 workers. One reader holds the same worker at ~3GB.
    #
    # `buffer_size` is deliberately *not* the lever here: 1,000 and 10,000
    # both plateau around 24GB at 10 readers, because the memory is in the
    # readers rather than in the buffered rows. So the buffer stays large
    # (good mixing, nearly free) and the reader count comes down.
    #
    # What this costs: the buffer mixes rows within one file at a time instead
    # of across ten. File *order* is still shuffled -- `shuffle()` randomizes
    # the shard order independently of this -- so a budget-limited run still
    # takes a random slice of the dataset rather than a fixed prefix.
    shuffle_input_shards: int = 1
    needs_token: bool = False
    fallback: "SourceSpec | None" = None


def _hf(source_id: str, dataset: str, **kwargs) -> SourceSpec:
    return SourceSpec(source_id=source_id, kind="hf", hf_dataset=dataset, **kwargs)


# The permissively licensed Parquet configs of github-code-clean. Selected by
# license, not by language: "permissive code" in §9.1 is a licensing
# statement, and MIT/Apache-2.0/BSD-3 is what makes the resulting weights
# cleanly sellable.
CODE_PARQUET_CONFIGS = ("all-mit", "all-apache-2.0", "all-bsd-3-clause")
CODE_PARQUET_GLOB = (
    "hf://datasets/codeparrot/github-code-clean@refs/convert/parquet/{config}/partial-train/*.parquet"
)


SOURCES: dict[str, SourceSpec] = {
    "fineweb_edu": _hf(
        "fineweb_edu",
        "HuggingFaceFW/fineweb-edu",
        hf_config="sample-350BT",
        role="primary high-quality English web backbone",
        license="ODC-By 1.0",
    ),
    "c4": _hf(
        "c4",
        "allenai/c4",
        hf_config="en",
        role="broad web register diversity",
        license="ODC-By 1.0",
    ),
    "cosmopedia": _hf(
        "cosmopedia",
        "HuggingFaceTB/cosmopedia",
        hf_config="web_samples_v2",
        role="clean synthetic instructional prose; boosts fluency for small models",
        license="Apache-2.0",
    ),
    "gutenberg_pg19": _hf(
        "gutenberg_pg19",
        "emozilla/pg19",
        role="long-form literary English; balances short conversational text",
        license="Public domain (pre-1919 US works)",
        shuffle_buffer=2_000,
    ),
    "slimpajama": _hf(
        "slimpajama",
        "cerebras/SlimPajama-627B",
        role="extra deduplicated volume across mixed domains",
        license="permissive (RedPajama-derived)",
        needs_token=True,
        fallback=_hf(
            "slimpajama",
            "DKYoon/SlimPajama-6B",
            role="extra deduplicated volume (ungated ~6B-token sample)",
            license="permissive (RedPajama-derived)",
        ),
    ),
    "permissive_code": SourceSpec(
        source_id="permissive_code",
        kind="hf",
        hf_dataset="parquet",
        hf_data_files=tuple(CODE_PARQUET_GLOB.format(config=c) for c in CODE_PARQUET_CONFIGS),
        text_field="code",
        role="teaches code-span handling, for verbatim code preservation",
        license="MIT / Apache-2.0 / BSD-3-Clause (selected configs only)",
        # Code is not English prose. Every quality heuristic would reject it
        # for being exactly what it is meant to be, and an `@decorator` is
        # syntax rather than a mention of a person.
        pipeline=PipelineConfig(quality=False, mask_pii=False),
    ),
    "stackexchange": SourceSpec(
        source_id="stackexchange",
        kind="stackexchange",
        role="in-domain Q&A and technical prose; matches inference-time input",
        license="CC BY-SA 4.0",
        # Post bodies are HTML in the dump, and these are real users' posts —
        # the one source where PII masking is doing heavy lifting.
        pipeline=PipelineConfig(strip_markup=True, quality=True, mask_pii=True),
    ),
}


def resolve(source_id: str) -> SourceSpec:
    """Look up a source, substituting the ungated fallback when needed."""
    try:
        spec = SOURCES[source_id]
    except KeyError:
        raise SourceConfigError(
            f"unknown source {source_id!r}; known sources: {', '.join(sorted(SOURCES))}"
        ) from None
    if spec.needs_token and not _hf_token_present() and spec.fallback is not None:
        return spec.fallback
    return spec


def _hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


# --------------------------------------------------------------------------
# HuggingFace streaming
# --------------------------------------------------------------------------


def iter_hf_records(
    spec: SourceSpec, *, seed: int, shard_index: int = 0, num_shards: int = 1
) -> Iterator[str]:
    """
    Stream raw text from a Hub dataset, taking only this worker's shard.

    Shuffling uses a bounded buffer rather than a global shuffle — the only
    option when streaming without downloading the whole dataset — which is
    deterministic given the same seed and dataset revision.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        # Include str(exc): this branch fires for ANY ImportError, not only
        # "datasets isn't installed" -- a real install with a broken
        # transitive dependency (pyarrow/numpy ABI mismatch, a partial
        # install, a fork-safety issue surfacing only inside a worker
        # process) raises ImportError too, and would otherwise be
        # misreported as "pip install datasets" even though that's not the
        # actual fix needed.
        raise SourceConfigError(
            f"could not import the `datasets` package ({exc}). If it's "
            f"already installed, this is likely a broken/incompatible "
            f"install (pyarrow/numpy version mismatch is the most common "
            f"cause) rather than a missing package -- try `pip install "
            f"--force-reinstall datasets` or check `pip check`."
        ) from exc

    kwargs: dict = {"split": "train", "streaming": True}
    if spec.hf_data_files:
        kwargs["data_files"] = {"train": list(spec.hf_data_files)}
    elif spec.hf_config:
        kwargs["name"] = spec.hf_config
    if spec.hf_revision:
        kwargs["revision"] = spec.hf_revision

    try:
        dataset = load_dataset(spec.hf_dataset, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surfaced as a config error with context
        raise SourceConfigError(
            f"could not open {spec.hf_dataset} "
            f"(config={spec.hf_config}, revision={spec.hf_revision}): {exc}"
        ) from exc

    if num_shards > 1:
        dataset = _shard(dataset, num_shards, shard_index)
    if spec.shuffle_buffer:
        dataset = _shuffle(
            dataset,
            seed=seed + shard_index,
            buffer_size=spec.shuffle_buffer,
            input_shards=spec.shuffle_input_shards,
        )

    for row in dataset:
        text = row.get(spec.text_field)
        if text:
            yield text


def _shuffle(dataset, *, seed: int, buffer_size: int, input_shards: int):
    """
    Shuffle a streaming dataset, capping how many files feed the buffer.

    `max_buffer_input_shards` is a `datasets` >= 5.0 parameter; see
    `SourceSpec.shuffle_input_shards` for why capping it is what keeps a
    worker inside a few GB instead of a few tens of GB.

    The parameter is passed only when the installed `datasets` actually
    accepts it, checked by signature rather than by catching `TypeError`.
    `requirements.txt` allows `datasets>=2.14`, and a `try/except TypeError`
    around a call whose body is a generator-producing library function would
    also swallow a genuine `TypeError` raised *inside* it -- turning a real
    bug into a silent fallback to the 10-reader default, which is the exact
    failure this function exists to prevent.
    """
    import inspect

    kwargs: dict = {"seed": seed, "buffer_size": buffer_size}
    if "max_buffer_input_shards" in inspect.signature(dataset.shuffle).parameters:
        kwargs["max_buffer_input_shards"] = max(1, input_shards)
    return dataset.shuffle(**kwargs)


def _shard(dataset, num_shards: int, index: int):
    """
    Split by underlying file when the format allows, else stride.

    File-level sharding is what makes parallelism real: each worker opens its
    own files, so N workers move N times the bytes. The modulo fallback still
    divides the *work* correctly, it just does not divide the download — every
    worker would pull the whole stream and discard most of it — so it exists
    only so an unusual dataset degrades to "correct but slow" rather than
    crashing.
    """
    try:
        return dataset.shard(num_shards=num_shards, index=index)
    except (AttributeError, TypeError, ValueError):
        from itertools import islice

        return islice(dataset, index, None, num_shards)


# --------------------------------------------------------------------------
# Stack Exchange
# --------------------------------------------------------------------------

ARCHIVE_ITEM = "stackexchange_20251231"
ARCHIVE_METADATA_URL = f"https://archive.org/metadata/{ARCHIVE_ITEM}"
ARCHIVE_DOWNLOAD_BASE = f"https://archive.org/download/{ARCHIVE_ITEM}"

# Sites whose *medium* is not English. Excluded by name because a site-level
# decision is exact and free, where the per-document language filter is a
# statistical backstop. Two groups: the localized Stack Overflow instances,
# and the language-learning sites, which are majority target-language text.
NON_ENGLISH_SITES = frozenset(
    {
        "es.stackoverflow.com",
        "pt.stackoverflow.com",
        "ru.stackoverflow.com",
        "ja.stackoverflow.com",
        "rus.stackexchange.com",
        "chinese.stackexchange.com",
        "esperanto.stackexchange.com",
        "french.stackexchange.com",
        "german.stackexchange.com",
        "italian.stackexchange.com",
        "japanese.stackexchange.com",
        "korean.stackexchange.com",
        "latin.stackexchange.com",
        "portuguese.stackexchange.com",
        "russian.stackexchange.com",
        "spanish.stackexchange.com",
        "ukrainian.stackexchange.com",
    }
)

# Download order, least technical first, as requested. The ordering is a
# curriculum-shaped convenience rather than a correctness requirement: with a
# byte budget that stops mid-list, it decides *which* sites make the cut, so
# putting general-audience prose first means a partial download is still a
# broad English corpus rather than 40GB of C# questions.
#
# stackoverflow.com is pinned last on its own: at 69GB it is larger than the
# other 361 sites combined, and letting it run first would spend an entire
# budget before anything else was touched.
SITE_TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "everyday",
        (
            "cooking", "gardening", "diy", "outdoors", "bicycles", "boardgames",
            "crafts", "woodworking", "pets", "parenting", "travel", "fitness",
            "health", "martialarts", "coffee", "beer", "homebrew", "sports",
            "poker", "chess", "lifehacks", "frugalliving", "money", "personalfinance",
            "vegetarianism", "sustainability", "ham", "genealogy", "hermeneutics",
        ),
    ),
    (
        "culture",
        (
            "literature", "movies", "music", "scifi", "writers", "worldbuilding",
            "rpg", "gaming", "anime", "photo", "graphicdesign", "art", "history",
            "philosophy", "politics", "law", "skeptics", "judaism", "christianity",
            "islam", "buddhism", "hinduism", "mythology", "linguistics", "english",
            "ell", "puzzling", "cogsci", "psychology", "sound", "video", "musicfans",
        ),
    ),
    (
        "social",
        (
            "workplace", "academia", "interpersonal", "freelancing", "productivity",
            "expatriates", "ux", "pm", "sqa", "opensource", "law", "patents",
            "communitybuilding", "writingcritique",
        ),
    ),
    (
        "science",
        (
            "biology", "chemistry", "physics", "astronomy", "earthscience",
            "economics", "stats", "math", "matheducation", "medicalsciences",
            "bioinformatics", "quantitativefinance", "or", "hsm", "scicomp",
            "space", "aviation", "engineering", "mechanics", "robotics",
        ),
    ),
    (
        "technical",
        (
            "electronics", "3dprinting", "gis", "dsp", "crypto", "security",
            "networkengineering", "serverfault", "superuser", "askubuntu", "unix",
            "apple", "android", "webmasters", "wordpress", "drupal", "magento",
            "salesforce", "sharepoint", "dba", "datascience", "ai", "devops",
            "softwareengineering", "codereview", "codegolf", "tex", "emacs", "vi",
            "raspberrypi", "arduino", "elementaryos", "reverseengineering",
            "softwarerecs", "webapps", "gamedev", "ethereum", "bitcoin", "monero",
            "cardano", "stellar", "tezos", "iot", "sitecore", "umbraco", "craftcms",
            "expressionengine", "wordpress", "blender", "mathematica", "or",
        ),
    ),
)

LAST_SITE = "stackoverflow.com"


def _site_name(archive_name: str) -> str:
    """`stackexchange_20251231/cooking.stackexchange.com.7z` -> the domain."""
    return Path(archive_name).name[: -len(".7z")]


def _tier_of(site: str) -> int:
    """Which tier a site falls in; unmatched sites sort into the middle."""
    stem = site.split(".")[0]
    for index, (_name, keywords) in enumerate(SITE_TIERS):
        if stem in keywords:
            return index
    return len(SITE_TIERS) // 2


def list_site_archives(url: str = ARCHIVE_METADATA_URL) -> list[tuple[str, int]]:
    """`(archive_name, size_bytes)` for every `.7z` in the archive.org item."""
    with urllib.request.urlopen(url, timeout=120) as response:
        import json

        metadata = json.load(response)
    return [
        (f["name"], int(f.get("size", 0)))
        for f in metadata.get("files", [])
        if f["name"].endswith(".7z")
    ]


def order_sites(
    archives: list[tuple[str, int]],
    *,
    include_meta: bool = False,
    include_non_english: bool = False,
) -> list[tuple[str, int]]:
    """
    Filter and order the site list: non-technical first, stackoverflow last.

    Meta sites are excluded by default. They are half the site count but a
    sliver of the bytes, and their content is site administration — "why was
    my question closed" — which is repetitive, self-referential, and about the
    least useful English prose in the dump.

    Within a tier, smaller archives come first: early completions make
    progress visible and mean an interrupted run leaves whole finished sites
    rather than one half-parsed 4GB archive.
    """
    selected = []
    for name, size in archives:
        site = _site_name(name)
        if site == LAST_SITE:
            continue
        if not include_meta and (".meta." in site or site.startswith("meta.")):
            continue
        if not include_non_english and site in NON_ENGLISH_SITES:
            continue
        selected.append((name, size))

    selected.sort(key=lambda item: (_tier_of(_site_name(item[0])), item[1], item[0]))

    tail = [item for item in archives if _site_name(item[0]) == LAST_SITE]
    return selected + tail


def download_archive(archive_name: str, dest_dir: Path, *, timeout: int = 3600) -> Path:
    """
    Fetch one `.7z`, resuming-safe via a `.part` file and an atomic rename.

    A partially written archive that looks complete is the worst outcome here:
    7z would fail to open it, or worse, open it and yield a truncated site. So
    the final name only ever appears on a complete download.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(archive_name).name
    if dest.exists():
        return dest

    url = f"{ARCHIVE_DOWNLOAD_BASE}/{archive_name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001 - one site failing must not stop the run
        tmp.unlink(missing_ok=True)
        raise ExtractionError(f"download failed for {archive_name}: {exc}") from exc
    return dest


# Only these two files are unpacked. The rest of a Stack Exchange dump
# (Users, Badges, Votes, PostHistory, PostLinks, Tags) carries no prose worth
# training on and a good deal more personal data than we want to handle —
# the earlier cleanup deleted them for the same reason.
WANTED_XML = ("Posts.xml", "Comments.xml")


def extract_archive(archive_path: Path, dest_dir: Path) -> list[Path]:
    """Unpack just `Posts.xml` and `Comments.xml` from one site archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    binary = _seven_zip_binary()
    try:
        subprocess.run(
            [binary, "x", f"-o{dest_dir}", "-y", str(archive_path), *WANTED_XML],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ExtractionError(
            f"7z failed on {archive_path.name}: {exc.stderr.decode('utf-8', 'replace')[:400]}"
        ) from exc
    return [dest_dir / name for name in WANTED_XML if (dest_dir / name).is_file()]


def _seven_zip_binary() -> str:
    for candidate in ("7z", "7za", "7zr"):
        from shutil import which

        if which(candidate):
            return candidate
    raise SourceConfigError(
        "no 7z binary on PATH; the Stack Exchange dump ships as .7z archives. "
        "Install it with `apt-get install -y p7zip-full` (Debian/Ubuntu) or "
        "`brew install p7zip` (macOS)."
    )


def check_stackexchange_dependencies() -> None:
    """
    Verify `7z`, `lxml` and `html2text` are all available before any worker
    starts.

    Without this, a missing dependency is only ever discovered inside
    `iter_stackexchange_posts`/`extract_archive`, called separately for
    every one of up to 362 sites -- and both call sites are wrapped in a
    per-site `try/except Exception` (one bad site must not abort the whole
    worker), so a genuinely missing package doesn't fail loudly once, it
    fails identically 362 times, silently, recorded only as scattered
    `site_errors` entries nested in the run manifest. A worker can finish
    with zero output and no dependency error visible anywhere near the top
    of the run. Call this once, before spawning any worker, so a missing
    dependency is a single clear message instead of a few hundred silent
    per-site failures.
    """
    _seven_zip_binary()
    try:
        import lxml.etree  # noqa: F401
    except ImportError as exc:
        raise SourceConfigError(
            "the `lxml` package is required to parse Stack Exchange dump XML: "
            "pip install lxml"
        ) from exc
    try:
        import html2text  # noqa: F401
    except ImportError as exc:
        raise SourceConfigError(
            "the `html2text` package is required to convert Stack Exchange "
            "post bodies: pip install html2text"
        ) from exc


def check_hf_dependencies() -> None:
    """
    Verify `datasets` is importable in *this* process before any worker
    starts.

    Without this, a missing/broken `datasets` install is only ever
    discovered inside `iter_hf_records`, which every HF-source worker only
    calls after two levels of process creation (the outer per-source pool,
    then the inner per-worker pool). This is called from `run_hf_source`
    inside the *outer* worker but before the *inner* pool is created, so it
    still catches the failure one level earlier than it would otherwise
    surface, with one specific real-world case in mind: `python3
    download.py` resolving to a different interpreter than the one `pip
    install -r requirements.txt` targeted (common on machines with
    multiple Pythons on PATH -- conda, pyenv, Homebrew, system Python all
    coexisting). Checking here means that class of mismatch is caught as
    one clear message naming the exact interpreter in use, instead of a
    `ModuleNotFoundError` surfacing from inside a worker with no
    indication of *which* Python hit it.
    """
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        import sys

        raise SourceConfigError(
            f"could not import the `datasets` package under {sys.executable} "
            f"({exc}). This is the Python actually running download.py -- "
            f"if `pip install datasets` succeeded but this still fails, "
            f"you very likely installed it into a *different* Python "
            f"(check `which python3` / `which pip` agree, and that no "
            f"other Python earlier on PATH is shadowing the one you "
            f"activated). Fix: `{sys.executable} -m pip install -r "
            f"requirements.txt`, which installs into this exact interpreter "
            f"regardless of what `pip`/`python3` resolve to."
        ) from exc


def iter_stackexchange_posts(
    xml_path: Path, *, mask_name: Callable[[str], str] | None = None
) -> Iterator[str]:
    """
    Stream post/comment text out of one dump XML file.

    Two rules, both carried over from the earlier cleanup's decision log
    because both were learned the expensive way:

    **Never `html.unescape()` a `Body` before handing it to the markdown
    converter.** lxml has already decoded one XML layer, which leaves valid
    HTML in which a literal `&lt;target&gt;` the user typed is still escaped —
    exactly what the converter expects. Unescaping first turns it into a real
    -looking `<target>` tag, which the converter then deletes, silently
    destroying user content. That bug cost a full re-download once.

    **A fresh converter per file.** A shared `HTML2Text` instance was observed
    corrupting the last rows of a long run.

    Comments are plain markdown already and need `unescape` exactly once,
    which is the opposite rule and the reason the two are handled separately.
    """
    from lxml import etree

    is_posts = xml_path.name == "Posts.xml"
    converter = _html_converter() if is_posts else None
    processed = 0

    context = etree.iterparse(str(xml_path), events=("end",), tag="row", recover=True)
    for _event, element in context:
        if is_posts:
            raw = element.get("Body")
            text = converter.handle(raw) if raw else None
            processed += 1
            # Refresh periodically inside a huge file, per the note above.
            if processed % 20_000 == 0:
                converter = _html_converter()
        else:
            import html as html_module

            raw = element.get("Text")
            text = html_module.unescape(raw) if raw else None

        display_name = element.get("DisplayName") or element.get("OwnerDisplayName")
        if text and text.strip():
            if display_name and mask_name is not None:
                # The dump identifies this as a person's name structurally, so
                # replacing it carries no false-positive risk — unlike names
                # guessed out of running prose, which are left alone.
                text = text.replace(display_name, mask_name(display_name))
            yield text

        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]


def _html_converter():
    import html2text

    converter = html2text.HTML2Text()
    converter.body_width = 0  # never re-wrap: line breaks are content
    converter.ul_item_mark = "-"
    converter.unicode_snob = True  # keep real Unicode instead of ASCII-ising it
    converter.ignore_links = False  # URLs are the thing we most need preserved
    converter.ignore_images = False
    return converter
