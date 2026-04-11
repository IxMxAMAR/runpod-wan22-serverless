"""Tests for workflow template hydration engine."""
import copy
import json
import os
import pytest

from handler.template_engine import TemplateEngine


@pytest.fixture
def t2v_mini_template():
    """Minimal T2V template with just the nodes the engine injects into."""
    return {
        "224": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "default prompt",
                "clip": ["214", 0],
            },
        },
        "258": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "default negative",
                "clip": ["214", 0],
            },
        },
        "262": {
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
                "model": ["264", 0],
                "positive": ["224", 0],
                "negative": ["258", 0],
                "latent_image": ["259", 0],
            },
        },
        "223": {
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
                "model": ["222", 0],
                "positive": ["224", 0],
                "negative": ["258", 0],
                "latent_image": ["262", 0],
            },
        },
        "264": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": 5.0, "model": ["471", 0]},
        },
        "222": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"shift": 5.0, "model": ["261", 0]},
        },
        "471": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["257", 0],
                "clip": ["214", 0],
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
        "261": {
            "class_type": "Power Lora Loader (rgthree)",
            "inputs": {
                "model": ["213", 0],
                "clip": ["214", 0],
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
        "259": {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": 832,
                "height": 480,
                "length": 80,
                "batch_size": 1,
            },
        },
        "256": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "VIDEO/WAN22/0",
                "format": "video/h264-mp4",
                "images": ["473", 0],
            },
        },
        "283": {
            "class_type": "RIFE VFI",
            "inputs": {
                "ckpt_name": "rife47.pth",
                "clear_cache_after_n_frames": 100,
                "multiplier": 2,
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
                "frames": ["473", 0],
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
        assert "224" in wf
        assert wf["224"]["class_type"] == "CLIPTextEncode"

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
        assert wf["224"]["inputs"]["text"] == "a beautiful sunset"

    def test_set_negative_prompt(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_negative_prompt(wf, "ugly, blurry", pipeline="t2v")
        assert wf["258"]["inputs"]["text"] == "ugly, blurry"


class TestSeedInjection:
    def test_set_seed_both_samplers(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_seed(wf, 42, pipeline="t2v")
        assert wf["262"]["inputs"]["noise_seed"] == 42
        assert wf["223"]["inputs"]["noise_seed"] == 42


class TestSamplerParams:
    def test_set_steps(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_steps(wf, 8, pipeline="t2v")
        assert wf["262"]["inputs"]["steps"] == 8
        assert wf["223"]["inputs"]["steps"] == 8
        assert wf["262"]["inputs"]["end_at_step"] == 4
        assert wf["223"]["inputs"]["start_at_step"] == 4

    def test_set_cfg(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_cfg(wf, 3.5, pipeline="t2v")
        assert wf["262"]["inputs"]["cfg"] == 3.5
        assert wf["223"]["inputs"]["cfg"] == 3.5

    def test_set_shift(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_shift(wf, 8.0, pipeline="t2v")
        assert wf["264"]["inputs"]["shift"] == 8.0
        assert wf["222"]["inputs"]["shift"] == 8.0


class TestResolutionAndFrames:
    def test_set_resolution_t2v(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_resolution(wf, 640, 480, pipeline="t2v")
        assert wf["259"]["inputs"]["width"] == 640
        assert wf["259"]["inputs"]["height"] == 480

    def test_set_frame_count(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_frame_count(wf, 80, pipeline="t2v")
        assert wf["259"]["inputs"]["length"] == 80


class TestLoraInjection:
    def test_set_loras(self, engine):
        wf = engine.load_template("t2v-standard")
        loras = [
            {"name": "my-lora.safetensors", "strength": 0.9},
            {"name": "wan/other.safetensors", "strength": 0.5},
        ]
        engine.set_loras(wf, loras, pipeline="t2v")

        high_loader = wf["471"]["inputs"]
        assert high_loader["lora_1"]["lora"] == "my-lora.safetensors"
        assert high_loader["lora_1"]["strength"] == 0.9
        assert high_loader["lora_1"]["on"] is True
        assert high_loader["lora_2"]["lora"] == "wan/other.safetensors"
        assert high_loader["lora_2"]["strength"] == 0.5

    def test_set_loras_clears_extra_slots(self, engine):
        wf = engine.load_template("t2v-standard")
        loras = [{"name": "single.safetensors", "strength": 1.0}]
        engine.set_loras(wf, loras, pipeline="t2v")

        high_loader = wf["471"]["inputs"]
        assert high_loader["lora_1"]["on"] is True
        assert high_loader["lora_2"]["on"] is False
        assert high_loader["lora_3"]["on"] is False


class TestVideoOutputParams:
    def test_set_fps(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_fps(wf, 30, pipeline="t2v")
        assert wf["256"]["inputs"]["frame_rate"] == 30

    def test_set_rife_multiplier(self, engine):
        wf = engine.load_template("t2v-standard")
        engine.set_rife_multiplier(wf, 4, pipeline="t2v")
        assert wf["283"]["inputs"]["multiplier"] == 4


class TestFullHydration:
    def test_hydrate_standard_params(self, engine):
        params = {
            "prompt": "a cat in space",
            "resolution": {"width": 640, "height": 480},
            "duration": 3,
            "seed": 99,
        }
        wf = engine.hydrate("t2v-standard", params, pipeline="t2v")

        assert wf["224"]["inputs"]["text"] == "a cat in space"
        assert wf["259"]["inputs"]["width"] == 640
        assert wf["259"]["inputs"]["height"] == 480
        assert wf["259"]["inputs"]["length"] == 48  # 3 * 16
        assert wf["262"]["inputs"]["noise_seed"] == 99

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

        assert wf["262"]["inputs"]["steps"] == 8
        assert wf["262"]["inputs"]["cfg"] == 2.0
        assert wf["264"]["inputs"]["shift"] == 7.0
        assert wf["256"]["inputs"]["frame_rate"] == 30
        assert wf["283"]["inputs"]["multiplier"] == 4
        assert wf["258"]["inputs"]["text"] == "ugly"

    def test_hydrate_returns_deep_copy(self, engine):
        params = {"prompt": "changed"}
        wf = engine.hydrate("t2v-standard", params, pipeline="t2v")
        wf2 = engine.hydrate("t2v-standard", params, pipeline="t2v")
        wf["224"]["inputs"]["text"] = "mutated"
        assert wf2["224"]["inputs"]["text"] == "changed"
