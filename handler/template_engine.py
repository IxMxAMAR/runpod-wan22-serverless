"""Workflow template hydration engine for WAN 2.2 serverless handler."""
import copy
import json
import os
from pathlib import Path

from handler.utils import calculate_frames, resolve_resolution, generate_seed

# Node ID mappings per pipeline type
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


class TemplateEngine:
    """Loads workflow templates and injects parameters."""

    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self._cache: dict[str, dict] = {}

    def list_templates(self) -> list[str]:
        return [f.stem for f in self.template_dir.glob("*.json")]

    def load_template(self, name: str) -> dict:
        if name not in self._cache:
            path = self.template_dir / f"{name}.json"
            if not path.exists():
                available = ", ".join(self.list_templates())
                raise ValueError(
                    f"Template '{name}' not found. Available: {available}"
                )
            with open(path) as f:
                self._cache[name] = json.load(f)
        return copy.deepcopy(self._cache[name])

    def hydrate(self, template_name: str, params: dict, pipeline: str) -> dict:
        wf = self.load_template(template_name)

        if "prompt" in params:
            self.set_prompt(wf, params["prompt"], pipeline)
        if "negative_prompt" in params:
            self.set_negative_prompt(wf, params["negative_prompt"], pipeline)

        seed = params.get("seed")
        if seed is None:
            seed = generate_seed(None)
        self.set_seed(wf, seed, pipeline)

        if "resolution" in params:
            width, height = resolve_resolution(params["resolution"])
        else:
            width, height = resolve_resolution(None)
        self.set_resolution(wf, width, height, pipeline)

        if "duration" in params:
            frames = calculate_frames(params["duration"], pipeline)
        else:
            frames = calculate_frames(5, pipeline)
        self.set_frame_count(wf, frames, pipeline)

        if "loras" in params:
            self.set_loras(wf, params["loras"], pipeline)
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

    def set_prompt(self, wf: dict, text: str, pipeline: str) -> None:
        node_id = NODE_IDS[pipeline]["positive_prompt"]
        wf[node_id]["inputs"]["text"] = text

    def set_negative_prompt(self, wf: dict, text: str, pipeline: str) -> None:
        node_id = NODE_IDS[pipeline]["negative_prompt"]
        wf[node_id]["inputs"]["text"] = text

    def set_seed(self, wf: dict, seed: int, pipeline: str) -> None:
        ids = NODE_IDS[pipeline]
        wf[ids["ksampler_high"]]["inputs"]["noise_seed"] = seed
        wf[ids["ksampler_low"]]["inputs"]["noise_seed"] = seed

    def set_steps(self, wf: dict, steps: int, pipeline: str) -> None:
        ids = NODE_IDS[pipeline]
        split = steps // 2
        wf[ids["ksampler_high"]]["inputs"]["steps"] = steps
        wf[ids["ksampler_high"]]["inputs"]["end_at_step"] = split
        wf[ids["ksampler_low"]]["inputs"]["steps"] = steps
        wf[ids["ksampler_low"]]["inputs"]["start_at_step"] = split

    def set_cfg(self, wf: dict, cfg: float, pipeline: str) -> None:
        ids = NODE_IDS[pipeline]
        wf[ids["ksampler_high"]]["inputs"]["cfg"] = cfg
        wf[ids["ksampler_low"]]["inputs"]["cfg"] = cfg

    def set_shift(self, wf: dict, shift: float, pipeline: str) -> None:
        ids = NODE_IDS[pipeline]
        wf[ids["shift_high"]]["inputs"]["shift"] = shift
        wf[ids["shift_low"]]["inputs"]["shift"] = shift

    def set_resolution(self, wf: dict, width: int, height: int, pipeline: str) -> None:
        node_id = NODE_IDS[pipeline]["latent"]
        wf[node_id]["inputs"]["width"] = width
        wf[node_id]["inputs"]["height"] = height

    def set_frame_count(self, wf: dict, frames: int, pipeline: str) -> None:
        node_id = NODE_IDS[pipeline]["latent"]
        wf[node_id]["inputs"]["length"] = frames

    def set_loras(self, wf: dict, loras: list[dict], pipeline: str) -> None:
        ids = NODE_IDS[pipeline]
        for loader_id in [ids["lora_high"], ids["lora_low"]]:
            inputs = wf[loader_id]["inputs"]
            for i in range(1, MAX_LORA_SLOTS + 1):
                key = f"lora_{i}"
                if key not in inputs:
                    if i <= len(loras):
                        inputs[key] = {}
                    else:
                        break
                if i <= len(loras):
                    lora = loras[i - 1]
                    inputs[key] = {
                        "on": True,
                        "lora": lora["name"],
                        "strength": lora["strength"],
                    }
                else:
                    inputs[key]["on"] = False

    def set_fps(self, wf: dict, fps: int, pipeline: str) -> None:
        node_id = NODE_IDS[pipeline]["video_combine"]
        wf[node_id]["inputs"]["frame_rate"] = fps

    def set_rife_multiplier(self, wf: dict, multiplier: int, pipeline: str) -> None:
        node_id = NODE_IDS[pipeline]["rife"]
        wf[node_id]["inputs"]["multiplier"] = multiplier
