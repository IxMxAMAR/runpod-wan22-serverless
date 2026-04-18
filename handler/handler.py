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
            with open(path) as f:
                self._cache[name] = json.load(f)
        return copy.deepcopy(self._cache[name])

    def hydrate(self, template_name, params, pipeline):
        wf = self.load_template(template_name)

        if "prompt" in params:
            self.set_prompt(wf, params["prompt"], pipeline)
        if "negative_prompt" in params:
            self.set_negative_prompt(wf, params["negative_prompt"], pipeline)

        seed = params.get("seed")
        if seed is None:
            seed = generate_seed(None)
        self.set_seed(wf, seed, pipeline)

        width, height = resolve_resolution(params.get("resolution"))
        self.set_resolution(wf, width, height, pipeline)

        duration = params.get("duration", 5)
        frames = calculate_frames(duration, pipeline)
        self.set_frame_count(wf, frames, pipeline)

        if "high_loras" in params or "low_loras" in params:
            high = params.get("high_loras", [])
            low = params.get("low_loras", [])
            self.set_loras(wf, high, low, pipeline)
        elif "loras" in params:
            # Backwards compat: single list, auto-split HIGH/LOW
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
            self.set_loras(wf, high, low, pipeline)
        if "steps" in params:
            self.set_steps(wf, params["steps"], pipeline)
        if "cfg" in params:
            self.set_cfg(wf, params["cfg"], pipeline)
        if "shift" in params:
            self.set_shift(wf, params["shift"], pipeline)
        if "fps" in params:
            self.set_fps(wf, params["fps"], pipeline)
        if "rife_multiplier" in params:
            self.set_rife_multiplier(wf, params["rife_multiplier"], pipeline)

        return wf

    def set_prompt(self, wf, text, pipeline):
        wf[NODE_IDS[pipeline]["positive_prompt"]]["inputs"]["text"] = text

    def set_negative_prompt(self, wf, text, pipeline):
        wf[NODE_IDS[pipeline]["negative_prompt"]]["inputs"]["text"] = text

    def set_seed(self, wf, seed, pipeline):
        ids = NODE_IDS[pipeline]
        wf[ids["ksampler_high"]]["inputs"]["noise_seed"] = seed
        wf[ids["ksampler_low"]]["inputs"]["noise_seed"] = seed

    def set_steps(self, wf, steps, pipeline):
        ids = NODE_IDS[pipeline]
        split = steps // 2
        wf[ids["ksampler_high"]]["inputs"]["steps"] = steps
        wf[ids["ksampler_high"]]["inputs"]["end_at_step"] = split
        wf[ids["ksampler_low"]]["inputs"]["steps"] = steps
        wf[ids["ksampler_low"]]["inputs"]["start_at_step"] = split

    def set_cfg(self, wf, cfg, pipeline):
        ids = NODE_IDS[pipeline]
        wf[ids["ksampler_high"]]["inputs"]["cfg"] = cfg
        wf[ids["ksampler_low"]]["inputs"]["cfg"] = cfg

    def set_shift(self, wf, shift, pipeline):
        ids = NODE_IDS[pipeline]
        wf[ids["shift_high"]]["inputs"]["shift"] = shift
        wf[ids["shift_low"]]["inputs"]["shift"] = shift

    def set_resolution(self, wf, width, height, pipeline):
        node_id = NODE_IDS[pipeline]["latent"]
        wf[node_id]["inputs"]["width"] = width
        wf[node_id]["inputs"]["height"] = height

    def set_frame_count(self, wf, frames, pipeline):
        node_id = NODE_IDS[pipeline]["latent"]
        wf[node_id]["inputs"]["length"] = frames

    def set_loras(self, wf, high_loras, low_loras, pipeline):
        """Inject LoRAs into HIGH and LOW Power Lora Loader nodes separately."""
        ids = NODE_IDS[pipeline]
        for loader_id, lora_list in [(ids["lora_high"], high_loras),
                                      (ids["lora_low"], low_loras)]:
            inputs = wf[loader_id]["inputs"]
            for i in range(1, MAX_LORA_SLOTS + 1):
                key = f"lora_{i}"
                if i <= len(lora_list):
                    inputs[key] = {
                        "on": True,
                        "lora": lora_list[i - 1]["name"],
                        "strength": lora_list[i - 1]["strength"],
                    }
                elif key in inputs:
                    inputs[key]["on"] = False
                else:
                    break

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


def collect_results(prompt_id):
    resp = requests.get(f"{COMFY_BASE_URL}/history/{prompt_id}")
    resp.raise_for_status()
    history = resp.json()
    if prompt_id not in history:
        raise RuntimeError(f"No history found for prompt {prompt_id}")

    outputs = history[prompt_id].get("outputs", {})

    # Prefer the slow-motion video node for the current pipeline
    slowmo_node_id = NODE_IDS.get(PIPELINE_TYPE, {}).get("video_combine_slowmo")

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

    # First pass: try to get the slow-mo video specifically
    if slowmo_node_id and slowmo_node_id in outputs:
        gifs = outputs[slowmo_node_id].get("gifs", [])
        if gifs:
            return {"videos": [fetch_video(gifs[0])]}

    # Fallback: return the first available video
    for node_id, node_output in outputs.items():
        for gif in node_output.get("gifs", []):
            return {"videos": [fetch_video(gif)]}

    return {"videos": []}


# ── Request Routing ─────────────────────────────────────────────────────────

engine = TemplateEngine(TEMPLATE_DIR)


def route_request(input_data):
    mode, data = validate_input(input_data)
    start_time = time.time()

    if mode == "template":
        template_name = data["template"]
        params = data["params"]

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

    results = collect_results(prompt_id)
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
