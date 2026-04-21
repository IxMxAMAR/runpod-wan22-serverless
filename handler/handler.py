"""RunPod serverless handler for WAN 2.2 video generation.

Self-contained — all logic inlined to avoid import path issues
when start.sh runs this from /handler.py.

Two modes:
  1. Template + Params: Simple API, handler builds workflow from template
  2. Full Workflow: Power user sends complete ComfyUI API-format JSON
"""
import base64
import copy
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path

import requests
import websocket

try:
    import runpod
except ImportError:
    runpod = None
    print("WARNING: runpod package not found — handler will not start in serverless mode")

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFY_PORT", "8188")
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", "/comfyui/templates")
VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
PIPELINE_TYPE = os.environ.get("PIPELINE_TYPE", "t2v")

DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480

FRAME_MULTIPLIERS = {"t2v": 16, "i2v": 15}

ASPECT_RATIOS = {
    "1:1": (512, 512),
    "4:3": (640, 480),
    "3:4": (480, 640),
    "16:9": (832, 480),
    "9:16": (480, 832),
    "3:2": (768, 512),
    "2:3": (512, 768),
    "21:9": (896, 384),
    "9:21": (384, 896),
}

NODE_IDS = {
    "t2v": {
        "positive_prompt": "18",
        "negative_prompt": "17",
        "ksampler_high": "11",
        "ksampler_low": "10",
        "lora_high": "15",
        "lora_low": "16",
        "shift_high": "8",
        "shift_low": "5",
        "latent": "9",
        "video_combine": "7",
        "video_combine_slowmo": "6",
        "rife": "19",
        "image_selector": "20",
        "save_image": "1",
    },
    "i2v": {
        "positive_prompt": "265",
        "negative_prompt": "273",
        "ksampler_high": "271",
        "ksampler_low": "272",
        "lora_high": "267",
        "lora_low": "268",
        "shift_high": "269",
        "shift_low": "220",
        "latent": "226",
        "video_combine": "276",
        "video_combine_slowmo": "288",
        "rife": "287",
        "save_image": "297",
        "load_image": "278",
    },
}

MAX_LORA_SLOTS = 10

# Nodes to drop per (pipeline, mode).
#   fast → no RIFE, no slow-mo combine (just normal h264)
#   slow → no normal combine (RIFE + slow-mo combine only)
# Dropped nodes are leaves in the graph, so removal never leaves dangling refs.
MODE_BYPASS = {
    ("t2v", "fast"): ["rife", "video_combine_slowmo"],
    ("t2v", "slow"): ["video_combine"],
    ("i2v", "fast"): ["rife", "video_combine_slowmo"],
    ("i2v", "slow"): ["video_combine"],
}

# Lightning / step-distillation LoRA name patterns (case-insensitive substring match).
# When a preset or user overrides `lightning_strength_*`, LoRAs whose filename
# matches any of these get the new strength; other LoRAs are left alone.
LIGHTNING_PATTERNS = (
    "lightning",
    "light_2",      # lightx2v_t2v / lightx2v_i2v
    "lightx2v",
    "4steps",
    "4-step",
    "cfg_step_distill",
    "seko",
)

# Quality presets tune sampler/scheduler, CFG, step split, Lightning strength,
# and SLG together. Individual params in the request override preset values.
# Pass `quality_preset: "fast" | "quality" | "hero"` (no preset = legacy behavior).
QUALITY_PRESETS = {
    "fast": {
        "steps": 4,
        "split_ratio": 0.5,
        "cfg_high": 1.0,
        "cfg_low": 1.0,
        "shift_high": 5.0,
        "shift_low": 5.0,
        "lightning_strength_high": 1.0,
        "lightning_strength_low": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "slg_enabled": False,
    },
    "quality": {
        "steps": 6,
        "split_ratio": 0.5,
        "cfg_high": 3.0,
        "cfg_low": 1.0,
        "shift_high": 8.0,
        "shift_low": 5.0,
        "lightning_strength_high": 0.7,
        "lightning_strength_low": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "beta",
        "slg_enabled": True,
    },
    "hero": {
        "steps": 12,
        "split_ratio": 0.5,
        "cfg_high": 3.5,
        "cfg_low": 1.5,
        "shift_high": 8.0,
        "shift_low": 5.0,
        "lightning_strength_high": 0.7,
        "lightning_strength_low": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "beta",
        "slg_enabled": True,
    },
}

# SLG defaults (applied when slg_enabled=True). Validated recipe: block 9,
# start 0.2, end 0.9, scale 3.0 — documented fix for morphing on WAN 2.2 14B.
SLG_DEFAULTS = {
    "blocks": "9",
    "start_percent": 0.2,
    "end_percent": 0.9,
    "scale": 3.0,
}

# Prompt style presets — appended to the user's prompt. Keep user's text as the
# load-bearing subject; presets just add camera/lighting/quality grammar.
STYLE_PRESETS = {
    "realistic": (
        "handheld close-up tracking shot, shallow depth of field, 35mm film, "
        "natural window light, golden hour, "
        "cinematic 35mm film grain, raw unfiltered, natural skin texture, anatomically detailed"
    ),
    "cinematic_film": (
        "slow dolly in, anamorphic lens, low angle, "
        "moody low-key lighting, rim light from behind, "
        "cinematic, film grain, shallow depth of field, bokeh, anatomically detailed"
    ),
    "pov_handheld": (
        "first person POV, handheld tracking, close-up, "
        "bedside lamp glow, warm tungsten, "
        "raw unfiltered, intimate, natural skin texture, anatomically detailed"
    ),
}

# Curated default negative prompt — biggest fix for fused-fingers / malformed
# anatomy at 4-step. Applied only if the user's request has no negative_prompt.
DEFAULT_NEGATIVE_PROMPT = (
    "bad anatomy, malformed genitals, fused fingers, extra fingers, missing fingers, "
    "extra limbs, deformed hands, mosaic censoring, pixelated censoring, black bar, "
    "morphing, warping, distortion, face deformation, jittering, flickering, "
    "motion blur, sudden changes, inconsistent lighting, background jitter, horizon bend, "
    "text, watermark, logo, subtitle, signature, username, "
    "blurry, low quality, low resolution, compression artifacts, "
    "cartoon, anime, 3d render, cgi, doll, plastic skin, waxy skin, airbrushed"
)


# ── Utility Functions ───────────────────────────────────────────────────────

def validate_loras(loras, volume_path=VOLUME_PATH):
    validated = []
    lora_base = os.path.join(volume_path, "models", "loras")
    for lora in loras:
        name = lora["name"]
        strength = lora.get("strength", 1.0)
        if not name.endswith(".safetensors"):
            name = f"{name}.safetensors"
        full_path = os.path.join(lora_base, name)
        if not os.path.isfile(full_path):
            raise ValueError(f"LoRA '{lora['name']}' not found at {full_path}")
        validated.append({"name": name, "strength": strength})
    return validated


def calculate_frames(duration, pipeline_type):
    if duration <= 0:
        raise ValueError("Duration must be positive")
    return int(duration * FRAME_MULTIPLIERS[pipeline_type])


def resolve_resolution(resolution):
    if not resolution:
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    if "width" in resolution and "height" in resolution:
        return (resolution["width"], resolution["height"])
    if "aspect_ratio" in resolution:
        return ASPECT_RATIOS.get(resolution["aspect_ratio"], (DEFAULT_WIDTH, DEFAULT_HEIGHT))
    return (DEFAULT_WIDTH, DEFAULT_HEIGHT)


def generate_seed(seed):
    if seed is not None:
        return seed
    return random.randint(0, 2**32 - 1)


def _is_lightning(lora_name):
    """Does this LoRA filename look like a step-distillation / Lightning LoRA?"""
    lower = lora_name.lower()
    return any(pat in lower for pat in LIGHTNING_PATTERNS)


# ── Template Engine ─────────────────────────────────────────────────────────

class TemplateEngine:
    def __init__(self, template_dir):
        self.template_dir = Path(template_dir)
        self._cache = {}

    def list_templates(self):
        return [f.stem for f in self.template_dir.glob("*.json")]

    def load_template(self, name):
        if name not in self._cache:
            path = self.template_dir / f"{name}.json"
            if not path.exists():
                available = ", ".join(self.list_templates())
                raise ValueError(f"Template '{name}' not found. Available: {available}")
            with open(path, encoding="utf-8") as f:
                self._cache[name] = json.load(f)
        return copy.deepcopy(self._cache[name])

    def hydrate(self, template_name, params, pipeline):
        wf = self.load_template(template_name)

        # Resolve quality preset first; explicit params always override preset values.
        preset_name = params.get("quality_preset")
        if preset_name is not None and preset_name not in QUALITY_PRESETS:
            raise ValueError(
                f"quality_preset must be one of {list(QUALITY_PRESETS)}, got {preset_name!r}"
            )
        preset = dict(QUALITY_PRESETS.get(preset_name, {}))

        def pick(key):
            """Param value if explicitly set, else preset, else None."""
            if key in params:
                return params[key]
            return preset.get(key)

        # Prompt + negative prompt (with style preset + default negative fallback).
        if "prompt" in params:
            prompt_text = params["prompt"]
            style = params.get("style_preset")
            if style is not None:
                if style not in STYLE_PRESETS:
                    raise ValueError(
                        f"style_preset must be one of {list(STYLE_PRESETS)}, got {style!r}"
                    )
                prompt_text = f"{prompt_text}, {STYLE_PRESETS[style]}"
            self.set_prompt(wf, prompt_text, pipeline)

        if "negative_prompt" in params:
            self.set_negative_prompt(wf, params["negative_prompt"], pipeline)
        elif preset_name is not None:
            # Presets imply the user wants our curated negative baseline.
            self.set_negative_prompt(wf, DEFAULT_NEGATIVE_PROMPT, pipeline)

        seed = params.get("seed")
        if seed is None:
            seed = generate_seed(None)
        self.set_seed(wf, seed, pipeline)

        width, height = resolve_resolution(params.get("resolution"))
        self.set_resolution(wf, width, height, pipeline)

        duration = params.get("duration", 5)
        frames = calculate_frames(duration, pipeline)
        self.set_frame_count(wf, frames, pipeline)

        # LoRA injection (accepts lightning_strength_* to retune Lightning slots).
        lightning_high = pick("lightning_strength_high")
        lightning_low = pick("lightning_strength_low")
        if "high_loras" in params or "low_loras" in params:
            high = params.get("high_loras", [])
            low = params.get("low_loras", [])
            self.set_loras(wf, high, low, pipeline, lightning_high, lightning_low)
        elif "loras" in params:
            high, low = [], []
            for lora in params["loras"]:
                name = lora["name"]
                s = lora.get("strength", 1.0)
                if "HIGH" in name:
                    high.append({"name": name, "strength": s})
                    low.append({"name": name.replace("HIGH", "LOW"), "strength": s})
                elif "LOW" not in name:
                    high.append({"name": name, "strength": s})
                    low.append({"name": name, "strength": s})
            self.set_loras(wf, high, low, pipeline, lightning_high, lightning_low)
        elif lightning_high is not None or lightning_low is not None:
            # No new LoRA list but preset wants Lightning retuned — patch in place.
            self.retune_lightning(wf, pipeline, lightning_high, lightning_low)

        # Step count + split ratio.
        steps = pick("steps")
        split_ratio = pick("split_ratio")
        if steps is not None:
            self.set_steps(wf, steps, pipeline, split_ratio)

        # CFG — accept `cfg` (both passes) or `cfg_high`/`cfg_low` individually.
        cfg_high = params.get("cfg_high", params.get("cfg", preset.get("cfg_high")))
        cfg_low = params.get("cfg_low", params.get("cfg", preset.get("cfg_low")))
        if cfg_high is not None or cfg_low is not None:
            self.set_cfg(wf, cfg_high, cfg_low, pipeline)

        # Shift — same pattern.
        shift_high = params.get("shift_high", params.get("shift", preset.get("shift_high")))
        shift_low = params.get("shift_low", params.get("shift", preset.get("shift_low")))
        if shift_high is not None or shift_low is not None:
            self.set_shift(wf, shift_high, shift_low, pipeline)

        # Sampler / scheduler.
        sampler_name = pick("sampler_name")
        scheduler = pick("scheduler")
        if sampler_name is not None or scheduler is not None:
            self.set_sampler(wf, sampler_name, scheduler, pipeline)

        if "fps" in params:
            self.set_fps(wf, params["fps"], pipeline)
        if "rife_multiplier" in params:
            self.set_rife_multiplier(wf, params["rife_multiplier"], pipeline)

        # SkipLayerGuidance — dynamic node injection when enabled.
        slg_enabled = pick("slg_enabled")
        if slg_enabled:
            slg_params = {**SLG_DEFAULTS, **params.get("slg", {})}
            self.inject_slg(wf, pipeline, slg_params)

        if "mode" in params:
            mode = params["mode"]
            if mode not in ("fast", "slow"):
                raise ValueError(f"mode must be 'fast' or 'slow', got {mode!r}")
            self.set_mode(wf, mode, pipeline)

        return wf

    def set_mode(self, wf, mode, pipeline):
        """Prune unused output nodes so ComfyUI skips RIFE / extra combine.

        'fast' keeps the normal h264 combine; drops RIFE + slow-mo combine.
        'slow' keeps RIFE + slow-mo combine; drops normal combine.
        """
        bypass_keys = MODE_BYPASS.get((pipeline, mode), [])
        ids = NODE_IDS[pipeline]
        for key in bypass_keys:
            node_id = ids.get(key)
            if node_id and node_id in wf:
                del wf[node_id]

    def set_prompt(self, wf, text, pipeline):
        wf[NODE_IDS[pipeline]["positive_prompt"]]["inputs"]["text"] = text

    def set_negative_prompt(self, wf, text, pipeline):
        wf[NODE_IDS[pipeline]["negative_prompt"]]["inputs"]["text"] = text

    def set_seed(self, wf, seed, pipeline):
        ids = NODE_IDS[pipeline]
        wf[ids["ksampler_high"]]["inputs"]["noise_seed"] = seed
        wf[ids["ksampler_low"]]["inputs"]["noise_seed"] = seed

    def set_steps(self, wf, steps, pipeline, split_ratio=None):
        """Set total step count and the HIGH/LOW split boundary.

        split_ratio ∈ (0, 1): fraction of steps the HIGH pass runs. Default 0.5.
        """
        if split_ratio is None:
            split_ratio = 0.5
        ids = NODE_IDS[pipeline]
        split = max(1, min(steps - 1, int(round(steps * split_ratio))))
        wf[ids["ksampler_high"]]["inputs"]["steps"] = steps
        wf[ids["ksampler_high"]]["inputs"]["end_at_step"] = split
        wf[ids["ksampler_low"]]["inputs"]["steps"] = steps
        wf[ids["ksampler_low"]]["inputs"]["start_at_step"] = split

    def set_cfg(self, wf, cfg_high, cfg_low, pipeline):
        """Set CFG per pass. Either arg may be None to leave that pass untouched."""
        ids = NODE_IDS[pipeline]
        if cfg_high is not None:
            wf[ids["ksampler_high"]]["inputs"]["cfg"] = cfg_high
        if cfg_low is not None:
            wf[ids["ksampler_low"]]["inputs"]["cfg"] = cfg_low

    def set_shift(self, wf, shift_high, shift_low, pipeline):
        """Set ModelSamplingSD3 shift per pass. Either arg may be None."""
        ids = NODE_IDS[pipeline]
        if shift_high is not None:
            wf[ids["shift_high"]]["inputs"]["shift"] = shift_high
        if shift_low is not None:
            wf[ids["shift_low"]]["inputs"]["shift"] = shift_low

    def set_sampler(self, wf, sampler_name, scheduler, pipeline):
        """Update sampler_name / scheduler on both KSamplerAdvanced nodes."""
        ids = NODE_IDS[pipeline]
        for k_id in (ids["ksampler_high"], ids["ksampler_low"]):
            if sampler_name is not None:
                wf[k_id]["inputs"]["sampler_name"] = sampler_name
            if scheduler is not None:
                wf[k_id]["inputs"]["scheduler"] = scheduler

    def set_resolution(self, wf, width, height, pipeline):
        node_id = NODE_IDS[pipeline]["latent"]
        wf[node_id]["inputs"]["width"] = width
        wf[node_id]["inputs"]["height"] = height

    def set_frame_count(self, wf, frames, pipeline):
        node_id = NODE_IDS[pipeline]["latent"]
        wf[node_id]["inputs"]["length"] = frames

    def set_loras(self, wf, high_loras, low_loras, pipeline,
                   lightning_strength_high=None, lightning_strength_low=None):
        """Inject LoRAs into HIGH and LOW Power Lora Loader nodes separately.

        When lightning_strength_* is provided, any LoRA whose name matches a
        Lightning pattern gets its strength replaced — lets a preset retune
        Lightning without the caller needing to know which slot it lives in.
        """
        ids = NODE_IDS[pipeline]
        overrides = {
            ids["lora_high"]: (high_loras, lightning_strength_high),
            ids["lora_low"]: (low_loras, lightning_strength_low),
        }
        for loader_id, (lora_list, lightning_override) in overrides.items():
            inputs = wf[loader_id]["inputs"]
            for i in range(1, MAX_LORA_SLOTS + 1):
                key = f"lora_{i}"
                if i <= len(lora_list):
                    entry = lora_list[i - 1]
                    strength = entry["strength"]
                    if lightning_override is not None and _is_lightning(entry["name"]):
                        strength = lightning_override
                    inputs[key] = {
                        "on": True,
                        "lora": entry["name"],
                        "strength": strength,
                    }
                elif key in inputs:
                    inputs[key]["on"] = False
                else:
                    break

    def retune_lightning(self, wf, pipeline, lightning_strength_high=None,
                          lightning_strength_low=None):
        """Rewrite Lightning-pattern LoRA strengths on the existing loaders."""
        ids = NODE_IDS[pipeline]
        pairs = [
            (ids["lora_high"], lightning_strength_high),
            (ids["lora_low"], lightning_strength_low),
        ]
        for loader_id, override in pairs:
            if override is None:
                continue
            inputs = wf[loader_id]["inputs"]
            for i in range(1, MAX_LORA_SLOTS + 1):
                key = f"lora_{i}"
                if key not in inputs:
                    break
                slot = inputs[key]
                if isinstance(slot, dict) and _is_lightning(slot.get("lora", "")):
                    slot["strength"] = override

    def inject_slg(self, wf, pipeline, slg_params):
        """Insert SkipLayerGuidanceWanVideo between ModelSamplingSD3 and each KSampler.

        Topology before:   ModelSamplingSD3 → KSamplerAdvanced (model input)
        Topology after:    ModelSamplingSD3 → SLG node → KSamplerAdvanced

        SLG nodes get deterministic IDs "_slg_high" / "_slg_low" so a second
        call is idempotent — re-hydration won't stack multiple SLG nodes.
        """
        ids = NODE_IDS[pipeline]
        for pass_name, k_key, m_key in (
            ("high", "ksampler_high", "shift_high"),
            ("low", "ksampler_low", "shift_low"),
        ):
            ksampler_id = ids[k_key]
            model_source = wf[ksampler_id]["inputs"]["model"]
            slg_id = f"_slg_{pass_name}"
            # Skip if already wired — idempotent.
            if isinstance(model_source, list) and model_source[0] == slg_id:
                continue
            wf[slg_id] = {
                "class_type": "SkipLayerGuidanceWanVideo",
                "inputs": {
                    "blocks": str(slg_params.get("blocks", SLG_DEFAULTS["blocks"])),
                    "start_percent": float(slg_params.get("start_percent", SLG_DEFAULTS["start_percent"])),
                    "end_percent": float(slg_params.get("end_percent", SLG_DEFAULTS["end_percent"])),
                    "scale": float(slg_params.get("scale", SLG_DEFAULTS["scale"])),
                    "model": model_source,
                },
            }
            wf[ksampler_id]["inputs"]["model"] = [slg_id, 0]

    def set_fps(self, wf, fps, pipeline):
        wf[NODE_IDS[pipeline]["video_combine"]]["inputs"]["frame_rate"] = fps

    def set_rife_multiplier(self, wf, multiplier, pipeline):
        wf[NODE_IDS[pipeline]["rife"]]["inputs"]["multiplier"] = multiplier


# ── Input Validation ────────────────────────────────────────────────────────

def validate_input(input_data):
    if not input_data:
        raise ValueError("Input must contain either 'template' or 'workflow'")

    if "workflow" in input_data:
        return "workflow", input_data

    if "template" in input_data:
        if "params" not in input_data:
            raise ValueError("'params' is required when using template mode")
        params = input_data["params"]
        if "prompt" not in params:
            raise ValueError("'prompt' is required in params")
        template_name = input_data["template"]
        if "i2v" in template_name and "input_image" not in params:
            raise ValueError("'input_image' is required for I2V templates")
        return "template", input_data

    raise ValueError("Input must contain either 'template' or 'workflow'")


# ── ComfyUI Communication ──────────────────────────────────────────────────

def wait_for_comfyui(max_retries=120, interval=1.0):
    for i in range(max_retries):
        try:
            resp = requests.get(f"{COMFY_BASE_URL}/system_stats", timeout=5)
            if resp.status_code == 200:
                logger.info("ComfyUI ready after %d attempts", i + 1)
                return True
        except (requests.ConnectionError, requests.Timeout, requests.RequestException):
            pass
        time.sleep(interval)
    raise RuntimeError(f"ComfyUI not ready after {max_retries} attempts")


def upload_image(image_name, image_data):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    resp = requests.post(
        f"{COMFY_BASE_URL}/upload/image",
        files={"image": (image_name, image_bytes, "image/png")},
        data={"overwrite": "true"},
    )
    resp.raise_for_status()
    return resp.json().get("name", image_name)


def queue_workflow(workflow, client_id):
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{COMFY_BASE_URL}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ComfyUI queue error: {data['error']}")
    if "node_errors" in data and data["node_errors"]:
        raise RuntimeError(f"ComfyUI node errors: {data['node_errors']}")
    return data["prompt_id"]


def monitor_execution(prompt_id, client_id):
    ws_url = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=600)
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                msg_type = data.get("type")
                if msg_type == "execution_error":
                    error_data = data.get("data", {})
                    raise RuntimeError(f"ComfyUI execution error: {json.dumps(error_data)}")
                if msg_type == "executing":
                    exec_data = data.get("data", {})
                    if exec_data.get("node") is None and exec_data.get("prompt_id") == prompt_id:
                        break
    finally:
        ws.close()


def collect_results(prompt_id, mode="slow"):
    resp = requests.get(f"{COMFY_BASE_URL}/history/{prompt_id}")
    resp.raise_for_status()
    history = resp.json()
    if prompt_id not in history:
        raise RuntimeError(f"No history found for prompt {prompt_id}")

    outputs = history[prompt_id].get("outputs", {})
    ids = NODE_IDS.get(PIPELINE_TYPE, {})

    def fetch_video(gif):
        video_resp = requests.get(
            f"{COMFY_BASE_URL}/view",
            params={
                "filename": gif["filename"],
                "subfolder": gif.get("subfolder", ""),
                "type": gif.get("type", "output"),
            },
        )
        video_resp.raise_for_status()
        return {
            "filename": gif["filename"],
            "type": "base64",
            "data": base64.b64encode(video_resp.content).decode(),
        }

    # Pick the expected combine node based on mode, with sensible fallback.
    preferred_key = "video_combine" if mode == "fast" else "video_combine_slowmo"
    preferred_id = ids.get(preferred_key)
    if preferred_id and preferred_id in outputs:
        gifs = outputs[preferred_id].get("gifs", [])
        if gifs:
            return {"videos": [fetch_video(gifs[0])]}

    # Fallback: first available video from any output node
    for node_id, node_output in outputs.items():
        for gif in node_output.get("gifs", []):
            return {"videos": [fetch_video(gif)]}

    return {"videos": []}


# ── Request Routing ─────────────────────────────────────────────────────────

engine = TemplateEngine(TEMPLATE_DIR)


def route_request(input_data):
    req_mode, data = validate_input(input_data)
    start_time = time.time()
    output_mode = None

    if req_mode == "template":
        template_name = data["template"]
        params = data["params"]
        output_mode = params.get("mode")

        if "high_loras" in params:
            params["high_loras"] = validate_loras(params["high_loras"], VOLUME_PATH)
        if "low_loras" in params:
            params["low_loras"] = validate_loras(params["low_loras"], VOLUME_PATH)
        if "loras" in params:
            params["loras"] = validate_loras(params["loras"], VOLUME_PATH)

        pipeline = "i2v" if "i2v" in template_name else "t2v"
        workflow = engine.hydrate(template_name, params, pipeline)

        if "input_image" in params:
            uploaded_name = upload_image("input.png", params["input_image"])
            load_image_id = NODE_IDS.get(pipeline, {}).get("load_image")
            if load_image_id and load_image_id in workflow:
                workflow[load_image_id]["inputs"]["image"] = uploaded_name

        seed = params.get("seed", 0)
    else:
        workflow = data["workflow"]
        seed = 0
        if "images" in data:
            for img in data["images"]:
                upload_image(img["name"], img["image"])

    client_id = str(uuid.uuid4())
    prompt_id = queue_workflow(workflow, client_id)
    monitor_execution(prompt_id, client_id)

    results = collect_results(prompt_id, mode=output_mode)
    generation_time = time.time() - start_time

    return {
        **results,
        "metadata": {
            "seed": seed,
            "generation_time_seconds": round(generation_time, 2),
        },
    }


# ── RunPod Handler ──────────────────────────────────────────────────────────

def handler(job):
    try:
        input_data = job.get("input", {})
        return route_request(input_data)
    except ValueError as e:
        return {"error": str(e), "refresh_worker": False}
    except RuntimeError as e:
        return {"error": str(e), "refresh_worker": True}
    except Exception as e:
        logger.exception("Unexpected error in handler")
        return {"error": f"Internal error: {str(e)}", "refresh_worker": True}


if runpod is not None:
    wait_for_comfyui()
    runpod.serverless.start({"handler": handler})
