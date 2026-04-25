"""Tests for workflow template hydration engine."""
import copy
import json
import os
import pytest

from handler.handler import TemplateEngine


@pytest.fixture
def t2v_mini_template():
    """Minimal T2V template with just the nodes the engine injects into."""
    return {
        "18": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "default prompt",
                "clip": ["12", 0],
            },
        },
        "17": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "default negative",
                "clip": ["12", 0],
            },
        },
        "11": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "add_noise": "enable",
                "noise_seed": 1000,
                "control_after_generate": "fixed",
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
                "model": ["8", 0],
                "positive": ["18", 0],
                "negative": ["17", 0],
                "latent_image": ["9", 0],
            },
        },
        "10": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 1000,
                "control_after_generate": "fixed",
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 10000,
                "return_with_leftover_noise": "disable",
                "model": ["5", 0],
                "positive": ["18", 0],
                "negative": ["17", 0],
                "latent_image": ["11", 0],
            },
        },
        "8": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": 5.0, "model": ["15", 0]},
        },
        "5": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": 5.0, "model": ["16", 0]},
        },
        "15": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["2", 0],
                "clip": ["12", 0],
                "lora_1": {
                    "on": True,
                    "lora": "Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors",
                    "strength": 1.0,
                },
                "lora_2": {
                    "on": True,
                    "lora": "wan/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors",
                    "strength": 0.8,
                },
                "lora_3": {
                    "on": True,
                    "lora": "wan/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
                    "strength": 1.0,
                },
            },
        },
        "16": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["3", 0],
                "clip": ["12", 0],
                "lora_1": {
                    "on": True,
                    "lora": "Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0.safetensors",
                    "strength": 1.0,
                },
                "lora_2": {
                    "on": True,
                    "lora": "wan/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors",
                    "strength": 0.8,
                },
                "lora_3": {
                    "on": True,
                    "lora": "wan/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
                    "strength": 1.0,
                },
            },
        },
        "9": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": 832,
                "height": 480,
                "length": 80,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "VIDEO/WAN22/0",
                "format": "video/h264-mp4",
                "images": ["20", 0],
            },
        },
        "19": {
            "class_type": "RIFE VFI",
            "inputs": {
                "ckpt_name": "rife49.pth",
                "clear_cache_after_n_frames": 100,
                "multiplier": 2,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
                "frames": ["20", 0],
            },
        },
    }


@pytest.fixture
def engine(tmp_path, t2v_mini_template):
    """Create a TemplateEngine with a temp template directory."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    with open(template_dir / "t2v-standard.json", "w") as f:
        json.dump(t2v_mini_template, f)
    return TemplateEngine(str(template_dir))


class TestTemplateLoading:
    def test_load_existing_template(self, engine):
        wf = engine.load_template("t2v-standard")
        assert "18" in wf
        assert wf["18"]["class_type"] == "CLIPTextEncode"

    def test_load_missing_template_raises(self, engine):
        with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
            engine.load_template("nonexistent")

    def test_list_templates(self, engine):
        templates = engine.list_templates()
        assert "t2v-standard" in templates


class TestPromptInjection:
    def test_set_positive_prompt(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_prompt(wf, "a beautiful sunset", pipeline="t2v")
        assert wf["18"]["inputs"]["text"] == "a beautiful sunset"

    def test_set_negative_prompt(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_negative_prompt(wf, "ugly, blurry", pipeline="t2v")
        assert wf["17"]["inputs"]["text"] == "ugly, blurry"


class TestSeedInjection:
    def test_set_seed_both_samplers(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_seed(wf, 42, pipeline="t2v")
        assert wf["11"]["inputs"]["noise_seed"] == 42
        assert wf["10"]["inputs"]["noise_seed"] == 42


class TestSamplerParams:
    def test_set_steps(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_steps(wf, 8, pipeline="t2v")
        assert wf["11"]["inputs"]["steps"] == 8
        assert wf["10"]["inputs"]["steps"] == 8
        assert wf["11"]["inputs"]["end_at_step"] == 4
        assert wf["10"]["inputs"]["start_at_step"] == 4

    def test_set_cfg_both(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_cfg(wf, 3.5, 3.5, pipeline="t2v")
        assert wf["11"]["inputs"]["cfg"] == 3.5
        assert wf["10"]["inputs"]["cfg"] == 3.5

    def test_set_cfg_asymmetric(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_cfg(wf, 3.0, 1.0, pipeline="t2v")
        assert wf["11"]["inputs"]["cfg"] == 3.0
        assert wf["10"]["inputs"]["cfg"] == 1.0

    def test_set_cfg_one_side(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_cfg(wf, 2.5, None, pipeline="t2v")
        assert wf["11"]["inputs"]["cfg"] == 2.5
        assert wf["10"]["inputs"]["cfg"] == 1.0  # untouched

    def test_set_shift_both(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_shift(wf, 8.0, 8.0, pipeline="t2v")
        assert wf["8"]["inputs"]["shift"] == 8.0
        assert wf["5"]["inputs"]["shift"] == 8.0

    def test_set_shift_asymmetric(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_shift(wf, 8.0, 5.0, pipeline="t2v")
        assert wf["8"]["inputs"]["shift"] == 8.0
        assert wf["5"]["inputs"]["shift"] == 5.0


class TestResolutionAndFrames:
    def test_set_resolution_t2v(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_resolution(wf, 640, 480, pipeline="t2v")
        assert wf["9"]["inputs"]["width"] == 640
        assert wf["9"]["inputs"]["height"] == 480

    def test_set_frame_count(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_frame_count(wf, 80, pipeline="t2v")
        assert wf["9"]["inputs"]["length"] == 80


class TestLoraInjection:
    def test_set_loras_separate_high_low(self, engine):
        wf = engine.load_template("t2v-standard")
        high = [
            {"name": "my-HIGH.safetensors", "strength": 0.9},
            {"name": "shared.safetensors", "strength": 0.5},
        ]
        low = [
            {"name": "my-LOW.safetensors", "strength": 0.9},
            {"name": "shared.safetensors", "strength": 0.5},
        ]
        engine.set_loras(wf, high, low, pipeline="t2v")

        high_loader = wf["15"]["inputs"]
        assert high_loader["lora_1"]["lora"] == "my-HIGH.safetensors"
        assert high_loader["lora_1"]["on"] is True
        assert high_loader["lora_2"]["lora"] == "shared.safetensors"

        low_loader = wf["16"]["inputs"]
        assert low_loader["lora_1"]["lora"] == "my-LOW.safetensors"
        assert low_loader["lora_1"]["on"] is True
        assert low_loader["lora_2"]["lora"] == "shared.safetensors"

    def test_set_loras_clears_extra_slots(self, engine):
        wf = engine.load_template("t2v-standard")
        high = [{"name": "single.safetensors", "strength": 1.0}]
        low = [{"name": "single.safetensors", "strength": 1.0}]
        engine.set_loras(wf, high, low, pipeline="t2v")

        high_loader = wf["15"]["inputs"]
        assert high_loader["lora_1"]["on"] is True
        assert high_loader["lora_2"]["on"] is False
        assert high_loader["lora_3"]["on"] is False


class TestVideoOutputParams:
    def test_set_fps(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_fps(wf, 30, pipeline="t2v")
        assert wf["7"]["inputs"]["frame_rate"] == 30

    def test_set_rife_multiplier(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_rife_multiplier(wf, 4, pipeline="t2v")
        assert wf["19"]["inputs"]["multiplier"] == 4


class TestFullHydration:
    def test_hydrate_standard_params(self, engine):
        params = {
            "prompt": "a cat in space",
            "resolution": {"width": 640, "height": 480},
            "duration": 3,
            "seed": 99,
        }
        wf = engine.hydrate("t2v-standard", params, pipeline="t2v")

        assert wf["18"]["inputs"]["text"] == "a cat in space"
        assert wf["9"]["inputs"]["width"] == 640
        assert wf["9"]["inputs"]["height"] == 480
        assert wf["9"]["inputs"]["length"] == 48  # 3 * 16
        assert wf["11"]["inputs"]["noise_seed"] == 99

    def test_hydrate_with_advanced_params(self, engine):
        params = {
            "prompt": "test",
            "steps": 8,
            "cfg": 2.0,
            "shift": 7.0,
            "fps": 30,
            "rife_multiplier": 4,
            "negative_prompt": "ugly",
        }
        wf = engine.hydrate("t2v-standard", params, pipeline="t2v")

        assert wf["11"]["inputs"]["steps"] == 8
        assert wf["11"]["inputs"]["cfg"] == 2.0
        assert wf["8"]["inputs"]["shift"] == 7.0
        assert wf["7"]["inputs"]["frame_rate"] == 30
        assert wf["19"]["inputs"]["multiplier"] == 4
        assert wf["17"]["inputs"]["text"] == "ugly"

    def test_hydrate_returns_deep_copy(self, engine):
        params = {"prompt": "changed"}
        wf = engine.hydrate("t2v-standard", params, pipeline="t2v")
        wf2 = engine.hydrate("t2v-standard", params, pipeline="t2v")
        wf["18"]["inputs"]["text"] = "mutated"
        assert wf2["18"]["inputs"]["text"] == "changed"


class TestModeBypass:
    def test_fast_mode_drops_rife_and_slowmo(self, engine):
        wf = engine.hydrate("t2v-standard", {"prompt": "x", "mode": "fast"}, pipeline="t2v")
        assert "19" not in wf  # RIFE
        assert "7" in wf       # normal VideoCombine kept

    def test_slow_mode_drops_normal_combine(self, engine):
        wf = engine.hydrate("t2v-standard", {"prompt": "x", "mode": "slow"}, pipeline="t2v")
        assert "7" not in wf   # normal VideoCombine
        assert "19" in wf      # RIFE kept

    def test_default_mode_keeps_all_nodes(self, engine):
        wf = engine.hydrate("t2v-standard", {"prompt": "x"}, pipeline="t2v")
        assert "7" in wf
        assert "19" in wf

    def test_invalid_mode_raises(self, engine):
        with pytest.raises(ValueError, match="mode must be 'fast' or 'slow'"):
            engine.hydrate("t2v-standard", {"prompt": "x", "mode": "bogus"}, pipeline="t2v")


class TestSampler:
    def test_set_sampler_both_ksamplers(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_sampler(wf, "res_multistep", "beta", pipeline="t2v")
        assert wf["11"]["inputs"]["sampler_name"] == "res_multistep"
        assert wf["11"]["inputs"]["scheduler"] == "beta"
        assert wf["10"]["inputs"]["sampler_name"] == "res_multistep"
        assert wf["10"]["inputs"]["scheduler"] == "beta"

    def test_set_sampler_name_only(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_sampler(wf, "dpmpp_2m_sde", None, pipeline="t2v")
        assert wf["11"]["inputs"]["sampler_name"] == "dpmpp_2m_sde"
        assert wf["11"]["inputs"]["scheduler"] == "simple"  # untouched


class TestStepsSplit:
    def test_even_split_default(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_steps(wf, 8, pipeline="t2v")
        assert wf["11"]["inputs"]["end_at_step"] == 4
        assert wf["10"]["inputs"]["start_at_step"] == 4

    def test_custom_split_ratio(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_steps(wf, 8, pipeline="t2v", split_ratio=0.75)
        assert wf["11"]["inputs"]["end_at_step"] == 6
        assert wf["10"]["inputs"]["start_at_step"] == 6

    def test_split_clamped_min(self, engine):
        wf = engine.load_template("t2v-standard")
        # split_ratio=0 → clamped to 1 so HIGH still runs at least 1 step
        engine.set_steps(wf, 4, pipeline="t2v", split_ratio=0.0)
        assert wf["11"]["inputs"]["end_at_step"] == 1
        assert wf["10"]["inputs"]["start_at_step"] == 1


class TestQualityPreset:
    def test_preset_applies_sampler_cfg_split(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "quality"},
                             pipeline="t2v")
        assert wf["11"]["inputs"]["sampler_name"] == "res_multistep"
        assert wf["11"]["inputs"]["scheduler"] == "beta"
        assert wf["11"]["inputs"]["cfg"] == 3.0
        assert wf["10"]["inputs"]["cfg"] == 1.0
        assert wf["11"]["inputs"]["steps"] == 6
        assert wf["8"]["inputs"]["shift"] == 8.0
        assert wf["5"]["inputs"]["shift"] == 5.0

    def test_fast_preset(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "fast"},
                             pipeline="t2v")
        assert wf["11"]["inputs"]["sampler_name"] == "euler"
        assert wf["11"]["inputs"]["cfg"] == 1.0
        assert wf["11"]["inputs"]["steps"] == 4

    def test_hero_preset(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "hero"},
                             pipeline="t2v")
        assert wf["11"]["inputs"]["steps"] == 12
        assert wf["11"]["inputs"]["cfg"] == 3.5
        assert wf["10"]["inputs"]["cfg"] == 1.5

    def test_explicit_override_beats_preset(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "quality", "cfg_high": 2.0},
                             pipeline="t2v")
        assert wf["11"]["inputs"]["cfg"] == 2.0   # override
        assert wf["10"]["inputs"]["cfg"] == 1.0   # preset wins where no override

    def test_invalid_preset_raises(self, engine):
        with pytest.raises(ValueError, match="quality_preset must be one of"):
            engine.hydrate("t2v-standard",
                            {"prompt": "x", "quality_preset": "turbo"},
                            pipeline="t2v")


class TestLightningRetune:
    def test_retune_matches_pattern(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "quality"},
                             pipeline="t2v")
        # quality preset sets HIGH Lightning to 0.7, LOW to 1.0
        high_loader = wf["15"]["inputs"]
        seko_slot = next(v for k, v in high_loader.items()
                         if isinstance(v, dict) and "Seko" in v.get("lora", ""))
        assert seko_slot["strength"] == 0.7

    def test_explicit_list_applies_override(self, engine):
        wf = engine.load_template("t2v-standard")
        high = [
            {"name": "Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors", "strength": 1.0},
            {"name": "content-lora.safetensors", "strength": 0.9},
        ]
        low = list(high)
        engine.set_loras(wf, high, low, pipeline="t2v",
                         lightning_strength_high=0.7)
        slot_lightning = wf["15"]["inputs"]["lora_1"]
        slot_content = wf["15"]["inputs"]["lora_2"]
        assert slot_lightning["strength"] == 0.7
        assert slot_content["strength"] == 0.9  # untouched


class TestStylePreset:
    def test_realistic_appends_tokens(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "a woman walking", "style_preset": "realistic"},
                             pipeline="t2v")
        text = wf["18"]["inputs"]["text"]
        assert text.startswith("a woman walking")
        assert "35mm film" in text
        assert "anatomically detailed" in text

    def test_invalid_style_raises(self, engine):
        with pytest.raises(ValueError, match="style_preset must be one of"):
            engine.hydrate("t2v-standard",
                            {"prompt": "x", "style_preset": "cartoon"},
                            pipeline="t2v")


class TestDefaultNegativePrompt:
    def test_preset_applies_default_negative(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "quality"},
                             pipeline="t2v")
        assert "malformed genitals" in wf["17"]["inputs"]["text"]

    def test_explicit_negative_beats_default(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "quality",
                              "negative_prompt": "my custom negative"},
                             pipeline="t2v")
        assert wf["17"]["inputs"]["text"] == "my custom negative"


class TestSLGInjection:
    def test_slg_not_added_by_default(self, engine):
        wf = engine.hydrate("t2v-standard", {"prompt": "x"}, pipeline="t2v")
        assert "_slg_high" not in wf
        assert "_slg_low" not in wf

    def test_quality_preset_does_not_inject_slg(self, engine):
        # Quality preset has slg_enabled=False because SkipLayerGuidanceWanVideo
        # currently requires TeaCacheKJ in the workflow (not present in our templates).
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "quality_preset": "quality"},
                             pipeline="t2v")
        assert "_slg_high" not in wf
        assert "_slg_low" not in wf

    def test_slg_explicit_true_still_injects(self, engine):
        # Caller can still force-enable SLG via the explicit param if they
        # know the workflow supports it. Keeps the injection path testable.
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "slg_enabled": True},
                             pipeline="t2v")
        assert wf["11"]["inputs"]["model"] == ["_slg_high", 0]
        assert wf["_slg_high"]["inputs"]["model"] == ["8", 0]
        assert wf["_slg_high"]["class_type"] == "SkipLayerGuidanceWanVideo"

    def test_slg_override_params(self, engine):
        wf = engine.hydrate("t2v-standard",
                             {"prompt": "x", "slg_enabled": True,
                              "slg": {"blocks": "4,5,6", "scale": 5.0}},
                             pipeline="t2v")
        assert wf["_slg_high"]["inputs"]["blocks"] == "4,5,6"
        assert wf["_slg_high"]["inputs"]["scale"] == 5.0


class TestLightningHelper:
    def test_is_lightning_detection(self):
        from handler.handler import _is_lightning
        assert _is_lightning("Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors")
        assert _is_lightning("lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank32_bf16.safetensors")
        assert _is_lightning("wan2.2_i2v_A14b_lightx2v_4step_1022_HIGH.safetensors")
        assert not _is_lightning("CumShot-High.safetensors")
        assert not _is_lightning("SECRET_SAUCE_WAN2.1_14B_fp8.safetensors")
