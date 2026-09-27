"""
The configuration: the real YAML's safety invariants, strictness, and what a
resume is allowed to change.
"""

from __future__ import annotations

import math

import pytest

from conftest import REAL_CONFIG, config_document, make_config
from src.config import build_sft_config, load_sft_config
from src.errors import SFTConfigError


# -- the real adapter_config.yml ---------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_real_config_loads_for_both_devices(device):
    """A typo in the shipped YAML should fail here, not a day into a CPU run."""
    config = load_sft_config(REAL_CONFIG, device)
    assert config.device == device


def test_real_cpu_profile_keeps_away_from_pretraining():
    """
    The guarantees that let a CPU run sit beside pretraining. If someone
    edits the profile, this is the test that says what they are giving up.
    """
    resources = load_sft_config(REAL_CONFIG, "cpu").resources
    assert resources.cores == "0-21"
    assert resources.threads == 22
    assert resources.nice == 19
    assert resources.idle_io is True
    assert resources.precision == "fp32"


def test_real_gpu_profile_refuses_a_shared_gpu():
    assert load_sft_config(REAL_CONFIG, "gpu").resources.require_exclusive_gpu is True


def test_real_profiles_share_one_effective_batch_size():
    """
    Equal effective batches make the data order identical on both devices,
    which is what lets a CPU run be resumed on the GPU.
    """
    cpu = load_sft_config(REAL_CONFIG, "cpu").training
    gpu = load_sft_config(REAL_CONFIG, "gpu").training
    assert cpu.effective_batch_size == gpu.effective_batch_size


@pytest.mark.parametrize(
    "style, profile, data_bounds, acceptance",
    [
        ("concise", True, (0.2, 1.15), 1.0),
        ("professional", True, (0.5, 1.6), 1.6),
        ("friendly", False, (0.2, 3.0), 3.0),  # no styles.friendly: the shared, style-neutral values
    ],
)
def test_real_config_style_selects_token_and_profile(style, profile, data_bounds, acceptance):
    """
    `--style X` must reach the frame, the trainable embedding and X's own
    length bounds together -- concise's 1.15 cap on professional data would
    silently drop 8.6% of valid rewrites.
    """
    config = load_sft_config(REAL_CONFIG, "cpu", {"data": {"style": style}})
    assert config.data.style_token == f"<style:{style}>"
    assert config.data.expected_style == style
    assert config.adapter.trainable_tokens[0] == f"<style:{style}>"
    assert not [t for t in config.adapter.trainable_tokens if t.startswith("<style:") and t != f"<style:{style}>"]
    assert config.style_profile is profile
    assert (config.data.min_length_ratio, config.data.max_length_ratio) == data_bounds
    assert config.evaluation.acceptance.max_length_ratio == acceptance


def test_style_profile_sits_under_device_and_command_line_overrides(tmp_path):
    document = config_document(tmp_path)
    document["styles"] = {"concise": {"data": {"max_length_ratio": 1.3, "min_length_ratio": 0.3}}}
    document["devices"]["cpu"]["overrides"] = {"data": {"max_length_ratio": 1.4}}
    config = build_sft_config(document, "cpu", {"data": {"min_length_ratio": 0.4}})
    assert (config.data.min_length_ratio, config.data.max_length_ratio) == (0.4, 1.4)


def test_trainable_token_of_another_style_is_refused(tmp_path):
    document = config_document(tmp_path)
    document["adapter"]["trainable_tokens"] = ["<style:concise>", "<text_to_rewrite>"]
    with pytest.raises(SFTConfigError, match="selected style is <style:professional>"):
        build_sft_config(document, "cpu", {"data": {"style": "professional"}})


# -- strictness ----------------------------------------------------------------


def test_unknown_key_is_rejected(tmp_path):
    document = config_document(tmp_path)
    document["adapter"]["rnak"] = 64
    with pytest.raises(SFTConfigError, match="rnak"):
        build_sft_config(document, "cpu")


def test_device_override_of_a_nonexistent_key_is_rejected(tmp_path):
    document = config_document(tmp_path)
    document["devices"]["cpu"]["overrides"] = {"training": {"batchsize": 8}}
    with pytest.raises(SFTConfigError, match="batchsize"):
        build_sft_config(document, "cpu")


def test_device_override_applies_key_by_key(tmp_path):
    document = config_document(tmp_path)
    document["devices"]["gpu"]["overrides"] = {"training": {"batch_size": 8}}
    config = build_sft_config(document, "gpu")
    assert config.training.batch_size == 8
    assert config.training.gradient_accumulation_steps == 2  # untouched


def test_unknown_projection_in_a_target_group_is_rejected(tmp_path):
    document = config_document(tmp_path)
    document["adapter"]["targets"]["encoder_ffn"] = ["q_proj"]
    with pytest.raises(SFTConfigError, match="encoder_ffn"):
        build_sft_config(document, "cpu")


def test_output_inside_the_pretraining_run_is_refused(tmp_path):
    document = config_document(tmp_path)
    document["logging"]["logs_dir"] = "runs/pretrain/sft_logs"
    with pytest.raises(SFTConfigError, match="runs/pretrain"):
        build_sft_config(document, "cpu")


def test_cpu_profile_without_pinned_cores_is_refused(tmp_path):
    document = config_document(tmp_path)
    document["devices"]["cpu"]["resources"]["cores"] = None
    with pytest.raises(SFTConfigError, match="pin explicit cores"):
        build_sft_config(document, "cpu")


def test_ngram_blocking_with_beam_search_is_refused(tmp_path):
    """Beam search delegates to the model package, which has no n-gram blocking."""
    document = config_document(tmp_path)
    document["evaluation"]["decoding"].update(strategy="beam", no_repeat_ngram_size=3)
    with pytest.raises(SFTConfigError, match="no_repeat_ngram_size"):
        build_sft_config(document, "cpu")


# -- scaling and resume ----------------------------------------------------------


def test_rslora_scaling_is_alpha_over_sqrt_rank(tmp_path):
    config = make_config(tmp_path, adapter={"rank": 16, "alpha": 8.0, "use_rslora": True})
    assert config.adapter.scaling == pytest.approx(2.0)
    plain = make_config(tmp_path, adapter={"rank": 16, "alpha": 32.0, "use_rslora": False})
    assert plain.adapter.scaling == pytest.approx(2.0)
    assert make_config(tmp_path, adapter={"rank": 64, "alpha": 8.0}).adapter.scaling == pytest.approx(8 / math.sqrt(64))


def test_resume_signature_ignores_how_the_step_is_split_and_where_files_live(tmp_path):
    cpu = make_config(tmp_path, training={"batch_size": 4, "gradient_accumulation_steps": 2})
    gpu = make_config(
        tmp_path / "moved",
        training={"batch_size": 8, "gradient_accumulation_steps": 1, "max_steps": 99},
    )
    assert cpu.resume_signature() == gpu.resume_signature()


@pytest.mark.parametrize(
    "section, change",
    [
        ("adapter", {"rank": 8}),
        ("training", {"learning_rate": 1e-4}),
        ("training", {"batch_size": 8}),  # effective batch 16, not 8
        ("data", {"split_salt": "other"}),
        ("data", {"style": "professional"}),
    ],
)
def test_resume_signature_changes_with_what_shapes_the_run(tmp_path, section, change):
    assert make_config(tmp_path).resume_signature() != make_config(tmp_path, **{section: change}).resume_signature()
