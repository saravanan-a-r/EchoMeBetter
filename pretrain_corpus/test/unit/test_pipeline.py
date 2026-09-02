from __future__ import annotations

from src.dedup import build_deduplicator
from src.pii_masker import PIIMasker
from src.pii_registry import PIIRegistry
from src.pipeline import PipelineConfig, RecordPipeline
from src.quality import QualityFilter

GOOD_PROSE = (
    "The committee reviewed the quarterly roadmap and agreed that the next "
    "release should prioritize reliability over new features, since several "
    "customers had reported intermittent failures in the billing pipeline."
)


def _make_pipeline(tmp_path, config=None, *, with_masker=True):
    masker = None
    if with_masker:
        registry = PIIRegistry(tmp_path / "registry.json", salt=1)
        [reservation] = registry.reserve({"email": 100, "handle": 100, "phone": 100}, workers=1)
        masker = PIIMasker(
            reservation, legacy={"email": frozenset(), "phone": frozenset(), "handle": frozenset()}
        )
    config = config or PipelineConfig()
    return RecordPipeline(
        config,
        masker=masker,
        quality=QualityFilter(enabled=config.quality),
        deduplicator=build_deduplicator(1000),
    )


def test_good_record_survives_the_full_pipeline(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    out = pipeline.process(GOOD_PROSE)
    assert len(out) == 1
    assert pipeline.stats.chunks_emitted == 1


def test_empty_record_is_counted_and_produces_nothing(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    assert pipeline.process("") == []
    assert pipeline.stats.records_empty_after_clean == 1


def test_low_quality_chunk_is_dropped_and_counted(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    out = pipeline.process("### *** --- junk junk junk junk junk junk junk junk")
    assert out == []
    assert pipeline.stats.chunks_dropped_quality >= 1


def test_duplicate_chunk_is_dropped_on_second_pass(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    pipeline.process(GOOD_PROSE)
    out = pipeline.process(GOOD_PROSE)
    assert out == []
    assert pipeline.stats.chunks_dropped_duplicate == 1


def test_pii_is_masked_before_quality_and_dedup_see_the_chunk(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    text = GOOD_PROSE + " Contact john.doe@gmail.com for questions about this."
    [chunk] = pipeline.process(text)
    assert "john.doe@gmail.com" not in chunk


def test_quality_disabled_lets_code_like_text_through(tmp_path):
    pipeline = _make_pipeline(tmp_path, PipelineConfig(quality=False, mask_pii=False), with_masker=False)
    code = "def f(x):\n    return x+1\n" * 5
    out = pipeline.process(code)
    assert len(out) == 1


def test_mask_pii_disabled_leaves_email_untouched(tmp_path):
    pipeline = _make_pipeline(tmp_path, PipelineConfig(mask_pii=False), with_masker=False)
    text = GOOD_PROSE + " Contact john.doe@gmail.com for questions about this."
    [chunk] = pipeline.process(text)
    assert "john.doe@gmail.com" in chunk


def test_mask_name_delegates_to_masker_when_enabled(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    assert pipeline.mask_name("Jane Doe") != "Jane Doe"


def test_mask_name_is_a_noop_when_pii_masking_disabled(tmp_path):
    pipeline = _make_pipeline(tmp_path, PipelineConfig(mask_pii=False), with_masker=False)
    assert pipeline.mask_name("Jane Doe") == "Jane Doe"


def test_report_includes_pipeline_quality_and_pii_sections(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    pipeline.process(GOOD_PROSE)
    report = pipeline.report()
    assert "pipeline" in report
    assert "quality" in report
    assert "pii" in report


def test_report_omits_pii_section_when_no_masker(tmp_path):
    pipeline = _make_pipeline(tmp_path, PipelineConfig(mask_pii=False), with_masker=False)
    pipeline.process(GOOD_PROSE)
    report = pipeline.report()
    assert "pii" not in report
