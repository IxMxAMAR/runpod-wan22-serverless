"""RunPod serverless handler for WAN 2.2 video generation.

Two modes:
  1. Template + Params: Simple API, handler builds workflow from template
  2. Full Workflow: Power user sends complete ComfyUI API-format JSON

Environment Variables:
  COMFY_HOST: ComfyUI server host (default: 127.0.0.1)
  COMFY_PORT: ComfyUI server port (default: 8188)
  TEMPLATE_DIR: Path to workflow templates (default: /comfyui/templates)
  RUNPOD_VOLUME_PATH: Network volume mount (default: /runpod-volume)
  PIPELINE_TYPE: "t2v" or "i2v" (default: t2v)
"""
import base64
import json
import logging
import os
import time
import uuid

import requests
import websocket

try:
    import runpod
except ImportError:
    runpod = None  # For local testing

from handler.template_engine import TemplateEngine, NODE_IDS
from handler.utils import validate_loras, generate_seed

logger = logging.getLogger(__name__)

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFY_PORT", "8188")
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
TEMPLATE_DIR = os.environ.get("TEMPLATE_DIR", "/comfyui/templates")
VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
PIPELINE_TYPE = os.environ.get("PIPELINE_TYPE", "t2v")

engine = TemplateEngine(TEMPLATE_DIR)


def validate_input(input_data: dict) -> tuple[str, dict]:
    """Validate and classify the input as template or workflow mode.

    Returns:
        Tuple of (mode, validated_data) where mode is "template" or "workflow".

    Raises:
        ValueError: If input is invalid.
    """
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


def wait_for_comfyui(max_retries: int = 60, interval: float = 1.0) -> bool:
    """Wait for ComfyUI server to be ready."""
    for i in range(max_retries):
        try:
            resp = requests.get(f"{COMFY_BASE_URL}/system_stats", timeout=5)
            if resp.status_code == 200:
                logger.info("ComfyUI ready after %d attempts", i + 1)
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    raise RuntimeError(f"ComfyUI not ready after {max_retries} attempts")


def upload_image(image_name: str, image_data: str) -> str:
    """Upload a base64-encoded image to ComfyUI."""
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


def queue_workflow(workflow: dict, client_id: str) -> str:
    """Queue a workflow for execution in ComfyUI. Returns prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{COMFY_BASE_URL}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ComfyUI queue error: {data['error']}")
    if "node_errors" in data and data["node_errors"]:
        raise RuntimeError(f"ComfyUI node errors: {data['node_errors']}")
    return data["prompt_id"]


def monitor_execution(prompt_id: str, client_id: str) -> None:
    """Monitor workflow execution via WebSocket until complete."""
    ws_url = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url)
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                msg_type = data.get("type")
                if msg_type == "execution_error":
                    error_data = data.get("data", {})
                    raise RuntimeError(
                        f"ComfyUI execution error: {json.dumps(error_data)}"
                    )
                if msg_type == "executing":
                    exec_data = data.get("data", {})
                    if (
                        exec_data.get("node") is None
                        and exec_data.get("prompt_id") == prompt_id
                    ):
                        break
    finally:
        ws.close()


def collect_results(prompt_id: str) -> dict:
    """Collect output files from ComfyUI after execution."""
    resp = requests.get(f"{COMFY_BASE_URL}/history/{prompt_id}")
    resp.raise_for_status()
    history = resp.json()
    if prompt_id not in history:
        raise RuntimeError(f"No history found for prompt {prompt_id}")

    outputs = history[prompt_id].get("outputs", {})
    videos = []
    thumbnails = []

    for node_id, node_output in outputs.items():
        if "gifs" in node_output:
            for gif in node_output["gifs"]:
                filename = gif["filename"]
                subfolder = gif.get("subfolder", "")
                file_type = gif.get("type", "output")
                video_resp = requests.get(
                    f"{COMFY_BASE_URL}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": file_type},
                )
                video_resp.raise_for_status()
                video_b64 = base64.b64encode(video_resp.content).decode()
                videos.append({
                    "filename": filename,
                    "type": "base64",
                    "data": video_b64,
                })
        if "images" in node_output:
            for img in node_output["images"]:
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                file_type = img.get("type", "output")
                img_resp = requests.get(
                    f"{COMFY_BASE_URL}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": file_type},
                )
                img_resp.raise_for_status()
                img_b64 = base64.b64encode(img_resp.content).decode()
                thumbnails.append({
                    "filename": filename,
                    "type": "base64",
                    "data": img_b64,
                })

    return {"videos": videos, "thumbnails": thumbnails}


def route_request(input_data: dict) -> dict:
    """Route and process a request through either template or workflow mode."""
    mode, data = validate_input(input_data)
    start_time = time.time()

    if mode == "template":
        template_name = data["template"]
        params = data["params"]

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

    output = {
        **results,
        "metadata": {
            "seed": seed,
            "generation_time_seconds": round(generation_time, 2),
        },
    }
    return output


def handler(job: dict) -> dict:
    """RunPod serverless handler entry point."""
    try:
        input_data = job.get("input", {})
        result = route_request(input_data)
        return result
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
