"""Utility functions for the RunPod WAN 2.2 handler."""
import os
import random

# Network volume mount point
VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")

# Default resolution
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480

# Frame rate multipliers per pipeline type
FRAME_MULTIPLIERS = {
    "t2v": 16,  # T2V: seconds * 16
    "i2v": 15,  # I2V: seconds * 15
}

# Common aspect ratio presets (width, height at ~512px base)
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


def validate_loras(
    loras: list[dict], volume_path: str = VOLUME_PATH
) -> list[dict]:
    """Validate that all LoRA files exist on the network volume.

    Args:
        loras: List of {"name": str, "strength": float} dicts.
               Name can omit .safetensors extension.
               Name can include subdirectory (e.g., "wan/FusionX").
        volume_path: Root path of the network volume.

    Returns:
        List of validated LoRA dicts with .safetensors extension ensured.

    Raises:
        ValueError: If any LoRA file is not found.
    """
    validated = []
    lora_base = os.path.join(volume_path, "models", "loras")

    for lora in loras:
        name = lora["name"]
        strength = lora.get("strength", 1.0)

        # Ensure .safetensors extension
        if not name.endswith(".safetensors"):
            name = f"{name}.safetensors"

        # Check file exists
        full_path = os.path.join(lora_base, name)
        if not os.path.isfile(full_path):
            raise ValueError(
                f"LoRA '{lora['name']}' not found at {full_path}"
            )

        validated.append({"name": name, "strength": strength})

    return validated


def calculate_frames(duration: float, pipeline_type: str) -> int:
    """Convert duration in seconds to frame count.

    Args:
        duration: Video duration in seconds.
        pipeline_type: "t2v" or "i2v" — determines frame multiplier.

    Returns:
        Number of frames.

    Raises:
        ValueError: If duration is not positive.
    """
    if duration <= 0:
        raise ValueError("Duration must be positive")

    multiplier = FRAME_MULTIPLIERS[pipeline_type]
    return int(duration * multiplier)


def resolve_resolution(
    resolution: dict | None,
) -> tuple[int, int]:
    """Resolve resolution from explicit dimensions or aspect ratio.

    Args:
        resolution: Dict with either {"width", "height"} or {"aspect_ratio"}.
                    None or empty dict returns defaults.

    Returns:
        Tuple of (width, height).
    """
    if not resolution:
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    if "width" in resolution and "height" in resolution:
        return (resolution["width"], resolution["height"])

    if "aspect_ratio" in resolution:
        ratio = resolution["aspect_ratio"]
        if ratio in ASPECT_RATIOS:
            return ASPECT_RATIOS[ratio]
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    return (DEFAULT_WIDTH, DEFAULT_HEIGHT)


def generate_seed(seed: int | None) -> int:
    """Return the given seed or generate a random one.

    Args:
        seed: Explicit seed or None for random.

    Returns:
        Integer seed.
    """
    if seed is not None:
        return seed
    return random.randint(0, 2**32 - 1)
