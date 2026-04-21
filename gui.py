"""GUI for RunPod WAN 2.2 T2V/I2V Serverless Endpoints with LoRA management."""
import base64
import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import requests

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "gui_config.json"
SCENE_PRESETS_FILE = Path(__file__).parent / "scene_presets.json"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_scene_presets():
    """Read scene_presets.json. Returns [] if missing or malformed."""
    if not SCENE_PRESETS_FILE.exists():
        return []
    try:
        data = json.loads(SCENE_PRESETS_FILE.read_text(encoding="utf-8"))
        return data.get("presets", [])
    except (json.JSONDecodeError, OSError):
        return []


# Base LoRAs that should stay untouched by scene presets (they're general
# quality enhancers, not scene-specific content). Substring match, case-insensitive.
BASE_LORA_PATTERNS = (
    "secret_sauce",
    "fusionx",
    "cubeyai",
    "instagirl",
    "instamodel",
    "nsfw-bundle",
)


def is_base_lora(name):
    lower = name.lower()
    return any(p in lower for p in BASE_LORA_PATTERNS)


# Resolution presets — ordered by commonality for WAN 2.2 14B.
# Dims are divisible by 16 (latent constraint). 480p = fast, 720p = higher quality.
RESOLUTION_PRESETS = [
    ("16:9 landscape — 832×480 (fast default)", 832, 480),
    ("9:16 vertical — 480×832 (TikTok/Reels)", 480, 832),
    ("1:1 square — 512×512", 512, 512),
    ("1:1 square — 720×720 (HD)", 720, 720),
    ("4:3 — 640×480", 640, 480),
    ("3:4 — 480×640", 480, 640),
    ("3:2 — 768×512", 768, 512),
    ("2:3 — 512×768", 512, 768),
    ("16:9 HD — 1280×720 (slow)", 1280, 720),
    ("9:16 HD — 720×1280 (vertical HD, slow)", 720, 1280),
    ("21:9 ultrawide — 896×384", 896, 384),
    ("9:21 ultratall — 384×896", 384, 896),
]
RESOLUTION_CUSTOM = "Custom (edit W/H manually)"


DEFAULT_CONFIG = {
    "api_key": "",
    "t2v_endpoint": "",
    "i2v_endpoint": "",
    "lora_file": "",
    "mode": "t2v",
    "output_mode": "slow",
    "quality_preset": "",
    "style_preset": "",
    "slg_enabled": False,
    "prompt": "A golden retriever running through a sunlit meadow",
    "duration": 3,
    "seed": "42",
    "width": 832,
    "height": 480,
    "steps": "",
    "cfg": "",
    "shift": "",
    "fps": "",
    "rife": "",
    "default_strength": "1.0",
    # Per-mode LoRA state: {mode: {lora_name: {"enabled": bool, "strength": str}}}
    "lora_state": {"t2v": {}, "i2v": {}},
    # Last I2V image path
    "image_path": "",
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            loaded = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(loaded)
        # Ensure nested keys exist
        if "lora_state" not in cfg or not isinstance(cfg["lora_state"], dict):
            cfg["lora_state"] = {"t2v": {}, "i2v": {}}
        cfg["lora_state"].setdefault("t2v", {})
        cfg["lora_state"].setdefault("i2v", {})
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def load_lora_list(filepath):
    """Load LoRA names from a text file (one per line).
    Empty lines and lines starting with # are skipped.
    """
    loras = []
    if not filepath or not os.path.isfile(filepath):
        return loras
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                loras.append(name)
    return loras


# Patterns for deriving LOW name from HIGH name (longest match first)
HIGH_LOW_PATTERNS = [
    ("HIGHNOISE", "LOWNOISE"),
    ("highnoise", "lownoise"),
    ("HighNoise", "LowNoise"),
    ("high_noise", "low_noise"),
    ("HIGH", "LOW"),
    ("High", "Low"),
]


def derive_low_name(name):
    """Return the LOW counterpart of a HIGH LoRA name, or None if shared."""
    for hi, lo in HIGH_LOW_PATTERNS:
        if hi in name:
            return name.replace(hi, lo)
    return None  # Shared LoRA, no HIGH/LOW distinction


# LoRAs that should be enabled by default for a given pipeline.
# Match by substring — case-sensitive.
DEFAULT_ENABLED = {
    "t2v": [
        "SECRET_SAUCE",
        "Wan2.1_T2V_14B_FusionX",
        "Wan2.2-T2V-4steps-HIGH-rank64-Seko",
    ],
    "i2v": [
        "SECRET_SAUCE",
        "Wan2.1_I2V_14B_FusionX",
        "Wan2.2-I2V-HIGH-4steps-lora-rank64-Seko",
    ],
}


# Speed-LoRA detection — Lightning / step-distillation LoRAs that accelerate
# sampling. These get their own collapsible UI section so the main LoRA lists
# stay focused on content.
SPEED_PATTERNS = (
    "lightning",
    "lightx2v",
    "light_2",
    "4steps",
    "4-step",
    "_4step_",
    "cfg_step_distill",
    "seko",
)


def is_speed_lora(name):
    lower = name.lower()
    return any(p in lower for p in SPEED_PATTERNS)


# Quality preset → preferred Lightning LoRA per pipeline. On preset change the
# GUI auto-enables the matching Lightning and disables the others. The handler's
# lightning_strength_high/low retune (0.7 / 1.0 for quality+hero) runs server-side,
# so the GUI just manages which Lightning is active.
PRESET_LIGHTNING = {
    "fast": {
        "t2v": "Seko-V2.0",
        "i2v": "Seko-V1",
    },
    "quality": {
        "t2v": "lightx2v_4step_1217",
        "i2v": "lightx2v_4step_1022",
    },
    "hero": {
        "t2v": "lightx2v_4step_1217",
        "i2v": "lightx2v_4step_1022",
    },
}


def lora_for_pipeline(name, pipeline):
    """Return True if this LoRA should appear in the given pipeline's list.

    Rules:
      - Names with 't2v' (any case) → T2V only
      - Names with 'i2v' (any case) → I2V only
      - Names with neither → shown in both
    """
    lower = name.lower()
    has_t2v = "t2v" in lower
    has_i2v = "i2v" in lower
    if has_t2v and not has_i2v:
        return pipeline == "t2v"
    if has_i2v and not has_t2v:
        return pipeline == "i2v"
    return True  # No marker or both — universal


def is_default_enabled(name, pipeline):
    """Return True if this LoRA should be auto-enabled by default."""
    return any(pat in name for pat in DEFAULT_ENABLED.get(pipeline, []))


def truncate_display(name, max_len=60):
    """Truncate a LoRA name for compact display, preserving the meaningful end."""
    if len(name) <= max_len:
        return name
    # Show start + ... + end so HIGH/LOW suffix stays visible
    head = name[: max_len // 2 - 2]
    tail = name[-(max_len // 2 - 1):]
    return f"{head}…{tail}"


# ── API ─────────────────────────────────────────────────────────────────────

def send_request(endpoint_id, api_key, payload):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def poll_status(endpoint_id, api_key, job_id):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


# ── GUI ─────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("WAN 2.2 Serverless")
        self.root.geometry("750x900")
        self.root.minsize(500, 600)
        self.root.configure(bg="#1a1a2e")
        self.config = load_config()
        self.lora_vars = []  # List of (name, BooleanVar, StringVar_strength)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1a1a2e", foreground="#e0e0e0",
                        fieldbackground="#16213e", font=("Segoe UI", 10))
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("TButton", background="#0f3460", foreground="#e0e0e0",
                        padding=(12, 6))
        style.map("TButton", background=[("active", "#533483")])
        style.configure("TLabelframe", background="#1a1a2e", foreground="#a0a0c0")
        style.configure("TLabelframe.Label", background="#1a1a2e", foreground="#a0a0c0")
        style.configure("TCheckbutton", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("Accent.TButton", background="#533483", foreground="white",
                        padding=(16, 8), font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton", background=[("active", "#e94560")])
        style.configure("LoRA.TCheckbutton", background="#16213e", foreground="#e0e0e0")

        # Combobox readonly state needs explicit color mapping — clam theme's
        # defaults produce unreadable light-on-light text in readonly mode.
        style.configure("TCombobox", fieldbackground="#16213e", background="#0f3460",
                        foreground="#e0e0e0", arrowcolor="#a0a0c0")
        style.map("TCombobox",
                   fieldbackground=[("readonly", "#16213e"), ("disabled", "#16213e")],
                   foreground=[("readonly", "#e0e0e0"), ("disabled", "#888")],
                   selectbackground=[("readonly", "#0f3460")],
                   selectforeground=[("readonly", "#ffffff")],
                   arrowcolor=[("readonly", "#a0a0c0")])
        # The dropdown popup is a raw tk Listbox — theme it via option database
        self.root.option_add("*TCombobox*Listbox.Background", "#16213e")
        self.root.option_add("*TCombobox*Listbox.Foreground", "#e0e0e0")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#533483")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", "Segoe\\ UI 10")

        self._build_ui()

    def _build_ui(self):
        # Scrollable main frame
        self.canvas = tk.Canvas(self.root, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.main = ttk.Frame(self.canvas, padding=16)

        self.main.bind("<Configure>",
                       lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._canvas_win = self.canvas.create_window((0, 0), window=self.main, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Track canvas width so inner frame stretches on resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling. Skip if the event came from:
        #   - a combobox popup (path contains 'popdown' — ttk internal name)
        #   - a self-scrolling widget (Listbox, Text)
        #   - a separate Toplevel (defensive, catches any other popup)
        # Without these guards, bind_all fires on every wheel event, causing
        # the popup list and the outer canvas to scroll simultaneously.
        def _on_canvas_wheel(e):
            w = e.widget
            try:
                if "popdown" in str(w):
                    return
                if w.winfo_class() in ("Listbox", "Text"):
                    return
                if w.winfo_toplevel() is not self.root:
                    return
            except tk.TclError:
                pass
            self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_canvas_wheel)

        main = self.main

        # ── Settings ────────────────────────────────────────────────────
        settings = ttk.LabelFrame(main, text="Settings", padding=8)
        settings.pack(fill="x", pady=(0, 8))

        ttk.Label(settings, text="API Key:").grid(row=0, column=0, sticky="w", padx=4)
        self.api_key_var = tk.StringVar(value=self.config.get("api_key", ""))
        ttk.Entry(settings, textvariable=self.api_key_var, show="*", width=50).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=2)

        ttk.Label(settings, text="T2V Endpoint:").grid(row=1, column=0, sticky="w", padx=4)
        self.t2v_var = tk.StringVar(value=self.config.get("t2v_endpoint", ""))
        ttk.Entry(settings, textvariable=self.t2v_var, width=50).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=2)

        ttk.Label(settings, text="I2V Endpoint:").grid(row=2, column=0, sticky="w", padx=4)
        self.i2v_var = tk.StringVar(value=self.config.get("i2v_endpoint", ""))
        ttk.Entry(settings, textvariable=self.i2v_var, width=50).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=2)

        settings.columnconfigure(1, weight=1)

        ttk.Button(settings, text="Save Settings", command=self._save_settings).grid(
            row=3, column=2, sticky="e", padx=4, pady=4)

        # ── Mode ────────────────────────────────────────────────────────
        mode_frame = ttk.Frame(main)
        mode_frame.pack(fill="x", pady=(0, 8))

        self.mode_var = tk.StringVar(value=self.config.get("mode", "t2v"))
        ttk.Radiobutton(mode_frame, text="Text to Video (T2V)",
                        variable=self.mode_var, value="t2v",
                        command=self._toggle_mode).pack(side="left", padx=8)
        ttk.Radiobutton(mode_frame, text="Image to Video (I2V)",
                        variable=self.mode_var, value="i2v",
                        command=self._toggle_mode).pack(side="left", padx=8)

        ttk.Separator(mode_frame, orient="vertical").pack(side="left", fill="y", padx=12)
        self.output_mode_var = tk.StringVar(value=self.config.get("output_mode", "slow"))
        ttk.Radiobutton(mode_frame, text="Fast (no RIFE)",
                        variable=self.output_mode_var, value="fast").pack(side="left", padx=8)
        ttk.Radiobutton(mode_frame, text="Slow-mo (RIFE 2x)",
                        variable=self.output_mode_var, value="slow").pack(side="left", padx=8)

        # ── Presets ─────────────────────────────────────────────────────
        presets = ttk.LabelFrame(main, text="Presets", padding=8)
        presets.pack(fill="x", pady=(0, 8))

        ttk.Label(presets, text="Quality:").grid(row=0, column=0, sticky="w", padx=4)
        self.quality_preset_var = tk.StringVar(value=self.config.get("quality_preset", ""))
        self.quality_combo = ttk.Combobox(presets, textvariable=self.quality_preset_var,
                     values=["", "fast", "quality", "hero"],
                     state="readonly", width=12)
        self.quality_combo.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(presets, text="(sampler, CFG, steps, SLG, Lightning)",
                  foreground="#888").grid(row=0, column=2, sticky="w", padx=4)
        # Auto-manage Speed LoRAs when preset changes
        self.quality_preset_var.trace_add(
            "write", lambda *_: self._apply_preset_lightning())

        ttk.Label(presets, text="Style:").grid(row=1, column=0, sticky="w", padx=4)
        self.style_preset_var = tk.StringVar(value=self.config.get("style_preset", ""))
        self.style_combo = ttk.Combobox(presets, textvariable=self.style_preset_var,
                     values=["", "realistic", "cinematic_film", "pov_handheld"],
                     state="readonly", width=16)
        self.style_combo.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(presets, text="(appends camera/lighting tokens to prompt)",
                  foreground="#888").grid(row=1, column=2, sticky="w", padx=4)

        self.slg_enabled_var = tk.BooleanVar(value=self.config.get("slg_enabled", False))
        ttk.Checkbutton(presets, text="SkipLayerGuidance (anatomy fix)",
                        variable=self.slg_enabled_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(presets, text="(adds ~10% gen time)",
                  foreground="#888").grid(row=2, column=2, sticky="w", padx=4, pady=(4, 0))

        # ── Scene preset ────────────────────────────────────────────────
        self.scene_presets = load_scene_presets()
        ttk.Label(presets, text="Scene:").grid(row=3, column=0, sticky="w", padx=4, pady=(8, 0))
        self.scene_preset_var = tk.StringVar(value="")
        self.scene_preset_combo = ttk.Combobox(
            presets, textvariable=self.scene_preset_var,
            values=[""] + [self._scene_label(p) for p in self.scene_presets],
            state="readonly", width=40)
        self.scene_preset_combo.grid(row=3, column=1, columnspan=2, sticky="w",
                                      padx=4, pady=(8, 0))
        ttk.Label(presets, text="(picks content LoRAs + seeds the prompt)",
                  foreground="#888").grid(row=4, column=1, columnspan=2,
                                           sticky="w", padx=4)
        self.scene_preset_var.trace_add(
            "write", lambda *_: self._apply_scene_preset())

        # ── Prompt ──────────────────────────────────────────────────────
        prompt_frame = ttk.LabelFrame(main, text="Prompt", padding=8)
        prompt_frame.pack(fill="x", pady=(0, 8))

        self.prompt_text = tk.Text(prompt_frame, height=3, bg="#16213e", fg="#e0e0e0",
                                   insertbackground="#e0e0e0", font=("Segoe UI", 10),
                                   wrap="word", relief="flat", bd=2)
        self.prompt_text.pack(fill="x")
        self.prompt_text.insert("1.0", self.config.get("prompt", "A golden retriever running through a sunlit meadow"))

        # ── I2V Image ───────────────────────────────────────────────────
        self.image_frame = ttk.LabelFrame(main, text="Input Image (I2V)", padding=8)
        self.image_path_var = tk.StringVar(value=self.config.get("image_path", ""))
        self._thumb_photo = None  # prevent GC

        img_top = ttk.Frame(self.image_frame)
        img_top.pack(fill="x")
        ttk.Entry(img_top, textvariable=self.image_path_var, width=40).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(img_top, text="Browse", command=self._browse_image).pack(side="left", padx=(0, 4))
        ttk.Button(img_top, text="Clear", command=self._clear_image).pack(side="left")

        # Thumbnail preview (click to browse)
        self.thumb_label = tk.Label(self.image_frame, bg="#16213e", relief="sunken",
                                    text="Click to select image", fg="#666",
                                    font=("Segoe UI", 9), cursor="hand2",
                                    compound="center")
        self.thumb_label.pack(fill="x", pady=(8, 0), ipady=40)
        self.thumb_label.bind("<Button-1>", lambda e: self._browse_image())

        # ── LoRAs ───────────────────────────────────────────────────────
        lora_frame = ttk.LabelFrame(main, text="LoRAs", padding=8)
        lora_frame.pack(fill="x", pady=(0, 8))

        # File selector row
        file_row = ttk.Frame(lora_frame)
        file_row.pack(fill="x", pady=(0, 6))

        ttk.Label(file_row, text="List:").pack(side="left", padx=(0, 4))
        self.lora_file_var = tk.StringVar(value=self.config.get("lora_file", ""))
        ttk.Entry(file_row, textvariable=self.lora_file_var, width=35).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(file_row, text="Browse", command=self._browse_lora_file).pack(side="left", padx=(0, 4))
        ttk.Button(file_row, text="Load", command=self._load_loras).pack(side="left")

        # Two-column layout: HIGH | LOW
        columns = ttk.Frame(lora_frame)
        columns.pack(fill="x", pady=(0, 4))
        columns.columnconfigure(0, weight=1)
        columns.columnconfigure(1, weight=1)

        # HIGH column
        high_frame = ttk.LabelFrame(columns, text="HIGH Loader", padding=4)
        high_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.high_lora_container = ttk.Frame(high_frame)
        self.high_lora_container.pack(fill="x")

        # LOW column
        low_frame = ttk.LabelFrame(columns, text="LOW Loader", padding=4)
        low_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.low_lora_container = ttk.Frame(low_frame)
        self.low_lora_container.pack(fill="x")

        self.high_lora_vars = []  # [(name, BooleanVar, StringVar)]
        self.low_lora_vars = []

        # Buttons row
        btn_row = ttk.Frame(lora_frame)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="All", command=self._select_all_loras).pack(side="left", padx=2)
        ttk.Button(btn_row, text="None", command=self._deselect_all_loras).pack(side="left", padx=2)
        ttk.Label(btn_row, text="str:", foreground="#888").pack(side="right", padx=(4, 0))
        self.default_strength_var = tk.StringVar(value=self.config.get("default_strength", "1.0"))
        ttk.Entry(btn_row, textvariable=self.default_strength_var, width=5).pack(side="right")
        ttk.Label(btn_row, text="Default", foreground="#888").pack(side="right", padx=(0, 4))

        # ── Speed LoRAs (collapsible) ───────────────────────────────────
        # Lightning / step-distillation LoRAs live here so the main list stays
        # focused on content. Quality presets auto-manage which one is active.
        speed_frame = ttk.Frame(lora_frame)
        speed_frame.pack(fill="x", pady=(8, 0))

        self.speed_expanded = False
        self.speed_toggle_btn = ttk.Button(
            speed_frame,
            text="▸ Speed LoRAs (4-step acceleration — managed by Quality preset)",
            command=self._toggle_speed)
        self.speed_toggle_btn.pack(fill="x")

        self.speed_body = ttk.Frame(speed_frame)
        # packed lazily in _toggle_speed

        speed_cols = ttk.Frame(self.speed_body)
        speed_cols.pack(fill="x", pady=(4, 0))
        speed_cols.columnconfigure(0, weight=1)
        speed_cols.columnconfigure(1, weight=1)

        high_speed_frame = ttk.LabelFrame(speed_cols, text="HIGH Loader", padding=4)
        high_speed_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.high_speed_container = ttk.Frame(high_speed_frame)
        self.high_speed_container.pack(fill="x")

        low_speed_frame = ttk.LabelFrame(speed_cols, text="LOW Loader", padding=4)
        low_speed_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.low_speed_container = ttk.Frame(low_speed_frame)
        self.low_speed_container.pack(fill="x")

        ttk.Label(self.speed_body,
                  text="Picking a Quality preset auto-enables one Lightning + "
                       "tunes strengths (HIGH 0.7 / LOW 1.0 for quality+hero).",
                  foreground="#888", wraplength=680, font=("Segoe UI", 9)).pack(
            fill="x", pady=(4, 0))

        if self.lora_file_var.get():
            self._load_loras()

        # ── Parameters ──────────────────────────────────────────────────
        params = ttk.LabelFrame(main, text="Parameters", padding=8)
        params.pack(fill="x", pady=(0, 8))

        row = 0
        ttk.Label(params, text="Duration (sec):").grid(row=row, column=0, sticky="w", padx=4)
        self.duration_var = tk.IntVar(value=self.config.get("duration", 3))
        ttk.Spinbox(params, from_=1, to=30, textvariable=self.duration_var,
                     width=8).grid(row=row, column=1, sticky="w", padx=4)

        ttk.Label(params, text="Seed:").grid(row=row, column=2, sticky="w", padx=4)
        self.seed_var = tk.StringVar(value=self.config.get("seed", "42"))
        ttk.Entry(params, textvariable=self.seed_var, width=12).grid(
            row=row, column=3, sticky="w", padx=4)

        row = 1
        ttk.Label(params, text="Resolution:").grid(row=row, column=0, sticky="w", padx=4)
        self.resolution_var = tk.StringVar(value="")
        self.resolution_combo = ttk.Combobox(
            params, textvariable=self.resolution_var,
            values=[label for label, _, _ in RESOLUTION_PRESETS] + [RESOLUTION_CUSTOM],
            state="readonly", width=42)
        self.resolution_combo.grid(row=row, column=1, columnspan=3, sticky="w", padx=4)

        row = 2
        ttk.Label(params, text="Width:").grid(row=row, column=0, sticky="w", padx=4)
        self.width_var = tk.IntVar(value=self.config.get("width", 832))
        ttk.Entry(params, textvariable=self.width_var, width=8).grid(
            row=row, column=1, sticky="w", padx=4)

        ttk.Label(params, text="Height:").grid(row=row, column=2, sticky="w", padx=4)
        self.height_var = tk.IntVar(value=self.config.get("height", 480))
        ttk.Entry(params, textvariable=self.height_var, width=8).grid(
            row=row, column=3, sticky="w", padx=4)

        # Two-way sync: selecting a preset fills W/H; editing W/H flips to Custom
        self._resolution_syncing = False
        def _on_resolution_preset_change(*_):
            if self._resolution_syncing:
                return
            label = self.resolution_var.get()
            for preset_label, w, h in RESOLUTION_PRESETS:
                if preset_label == label:
                    self._resolution_syncing = True
                    self.width_var.set(w)
                    self.height_var.set(h)
                    self._resolution_syncing = False
                    return
        def _on_wh_manual_change(*_):
            if self._resolution_syncing:
                return
            try:
                w, h = int(self.width_var.get()), int(self.height_var.get())
            except (tk.TclError, ValueError):
                return
            match = next((lbl for lbl, pw, ph in RESOLUTION_PRESETS if pw == w and ph == h), None)
            self._resolution_syncing = True
            self.resolution_var.set(match if match else RESOLUTION_CUSTOM)
            self._resolution_syncing = False
        self.resolution_var.trace_add("write", _on_resolution_preset_change)
        self.width_var.trace_add("write", _on_wh_manual_change)
        self.height_var.trace_add("write", _on_wh_manual_change)
        # Seed dropdown from current width/height
        _on_wh_manual_change()

        # ── Advanced ────────────────────────────────────────────────────
        adv = ttk.LabelFrame(main, text="Advanced (optional)", padding=8)
        adv.pack(fill="x", pady=(0, 8))

        ttk.Label(adv, text="Steps:").grid(row=0, column=0, sticky="w", padx=4)
        self.steps_var = tk.StringVar(value=self.config.get("steps", ""))
        ttk.Entry(adv, textvariable=self.steps_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4)

        ttk.Label(adv, text="CFG:").grid(row=0, column=2, sticky="w", padx=4)
        self.cfg_var = tk.StringVar(value=self.config.get("cfg", ""))
        ttk.Entry(adv, textvariable=self.cfg_var, width=8).grid(
            row=0, column=3, sticky="w", padx=4)

        ttk.Label(adv, text="Shift:").grid(row=0, column=4, sticky="w", padx=4)
        self.shift_var = tk.StringVar(value=self.config.get("shift", ""))
        ttk.Entry(adv, textvariable=self.shift_var, width=8).grid(
            row=0, column=5, sticky="w", padx=4)

        ttk.Label(adv, text="FPS:").grid(row=1, column=0, sticky="w", padx=4)
        self.fps_var = tk.StringVar(value=self.config.get("fps", ""))
        ttk.Entry(adv, textvariable=self.fps_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4)

        ttk.Label(adv, text="RIFE mult:").grid(row=1, column=2, sticky="w", padx=4)
        self.rife_var = tk.StringVar(value=self.config.get("rife", ""))
        ttk.Entry(adv, textvariable=self.rife_var, width=8).grid(
            row=1, column=3, sticky="w", padx=4)

        # ── Generate Button ─────────────────────────────────────────────
        self.gen_btn = ttk.Button(main, text="Generate Video", style="Accent.TButton",
                                  command=self._generate)
        self.gen_btn.pack(pady=(4, 8))

        # ── Status ──────────────────────────────────────────────────────
        status_frame = ttk.LabelFrame(main, text="Status", padding=8)
        status_frame.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var,
                  font=("Segoe UI", 11)).pack(anchor="w")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 4))

        self.result_text = tk.Text(status_frame, height=6, bg="#16213e", fg="#a0e0a0",
                                    insertbackground="#e0e0e0", font=("Consolas", 9),
                                    wrap="word", relief="flat", bd=2, state="disabled")
        self.result_text.pack(fill="both", expand=True, pady=(4, 0))

        # Stop wheel events on text/combobox widgets from propagating to the
        # outer canvas scroll — otherwise scrolling inside a dropdown or text
        # area drags the whole page with it.
        def _consume_wheel(_event):
            return "break"
        for widget in (self.quality_combo, self.style_combo, self.scene_preset_combo,
                        self.resolution_combo, self.prompt_text, self.result_text):
            widget.bind("<MouseWheel>", _consume_wheel)

        self._toggle_mode()

    # ── LoRA Management ─────────────────────────────────────────────────

    def _browse_lora_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.lora_file_var.set(path)
            self._load_loras()

    def _load_loras(self):
        for container in (self.high_lora_container, self.low_lora_container,
                          self.high_speed_container, self.low_speed_container):
            for widget in container.winfo_children():
                widget.destroy()
        self.high_lora_vars.clear()
        self.low_lora_vars.clear()

        filepath = self.lora_file_var.get()
        names = load_lora_list(filepath)

        if not names:
            ttk.Label(self.high_lora_container, text="Load a file",
                      foreground="#666").pack(anchor="w")
            ttk.Label(self.low_lora_container, text="Load a file",
                      foreground="#666").pack(anchor="w")
            return

        default_str = self.default_strength_var.get()
        pipeline = self.mode_var.get()
        saved_state = self.config.get("lora_state", {}).get(pipeline, {})

        # Skip LOW-only lines whose HIGH counterpart is also in the list
        filtered = []
        for name in names:
            skip = False
            for hi, lo in HIGH_LOW_PATTERNS:
                if lo in name and hi not in name:
                    if name.replace(lo, hi) in names:
                        skip = True
                        break
            if not skip:
                filtered.append(name)

        # Filter by pipeline (T2V/I2V)
        visible = [n for n in filtered if lora_for_pipeline(n, pipeline)]

        speed_count = 0
        for name in visible:
            low_name = derive_low_name(name) or name
            # Prefer saved state; fall back to defaults
            if name in saved_state:
                auto_on = saved_state[name].get("enabled", False)
                strength_val = saved_state[name].get("strength", default_str)
            else:
                auto_on = is_default_enabled(name, pipeline)
                strength_val = default_str
            enabled = tk.BooleanVar(value=auto_on)
            strength = tk.StringVar(value=strength_val)

            # Route Lightning LoRAs to the collapsible Speed section
            if is_speed_lora(name):
                hi_container = self.high_speed_container
                lo_container = self.low_speed_container
                speed_count += 1
            else:
                hi_container = self.high_lora_container
                lo_container = self.low_lora_container

            self._add_lora_row(hi_container, self.high_lora_vars,
                               name, enabled, strength)
            self._add_lora_row(lo_container, self.low_lora_vars,
                               low_name, enabled, strength)

        content_count = len(visible) - speed_count
        self._log(f"Loaded {content_count} content + {speed_count} speed LoRAs for {pipeline.upper()}")

        # Re-apply preset Lightning selection after the fresh reload (saved state
        # may have been overridden by a new preset choice).
        self._apply_preset_lightning()

    def _add_lora_row(self, container, var_list, name, enabled, strength):
        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        display = truncate_display(name)
        ttk.Checkbutton(row, text=display, variable=enabled).pack(side="left")
        ttk.Entry(row, textvariable=strength, width=4).pack(side="right")
        var_list.append((name, enabled, strength))

    def _select_all_loras(self):
        # HIGH and LOW share vars now, so only iterate HIGH
        for _, enabled, _ in self.high_lora_vars:
            enabled.set(True)

    def _deselect_all_loras(self):
        for _, enabled, _ in self.high_lora_vars:
            enabled.set(False)

    def _toggle_speed(self):
        if self.speed_expanded:
            self.speed_body.pack_forget()
            self.speed_toggle_btn.configure(
                text="▸ Speed LoRAs (4-step acceleration — managed by Quality preset)")
        else:
            self.speed_body.pack(fill="x", pady=(4, 0))
            self.speed_toggle_btn.configure(
                text="▾ Speed LoRAs (4-step acceleration — managed by Quality preset)")
        self.speed_expanded = not self.speed_expanded

    def _apply_preset_lightning(self):
        """When a quality preset is selected, enable its preferred Lightning
        LoRA for the current pipeline and disable all other Lightning LoRAs.
        Strength values are retuned server-side by the handler preset."""
        if not hasattr(self, "quality_preset_var"):
            return  # called before preset UI is built
        preset = self.quality_preset_var.get().strip()
        if not preset or preset not in PRESET_LIGHTNING:
            return  # no preset selected → leave Lightning choices alone
        pipeline = self.mode_var.get()
        preferred = PRESET_LIGHTNING[preset].get(pipeline, "")
        if not preferred:
            return

        for name, enabled, _ in self.high_lora_vars:
            if not is_speed_lora(name):
                continue
            enabled.set(preferred in name)

    def _scene_label(self, preset):
        """Build the dropdown label for a scene preset."""
        pipe = preset.get("pipeline", "both").upper()
        return f"[{pipe}] {preset.get('category','?')} — {preset.get('name','?')}"

    def _scene_by_label(self, label):
        for p in self.scene_presets:
            if self._scene_label(p) == label:
                return p
        return None

    def _apply_scene_preset(self):
        """Tick matching content LoRAs, set strengths, seed the prompt box.

        Base LoRAs and Speed LoRAs are left alone — they're managed separately
        by the user / the Quality preset.
        """
        label = self.scene_preset_var.get().strip()
        if not label:
            return
        preset = self._scene_by_label(label)
        if not preset:
            return

        pipeline = self.mode_var.get()
        if preset.get("pipeline") not in ("both", pipeline):
            self._log(f"Scene '{preset.get('name')}' is {preset.get('pipeline').upper()}-only; "
                       f"switch mode first.")
            return

        # Apply LoRA selections
        targets = preset.get("loras", [])  # [{name_contains, strength}, ...]
        matched_any = False
        missing = []
        for name, enabled, strength in self.high_lora_vars:
            if is_speed_lora(name) or is_base_lora(name):
                continue  # don't touch speed or base LoRAs
            hit = None
            for t in targets:
                if t["name_contains"].lower() in name.lower():
                    hit = t
                    break
            if hit:
                enabled.set(True)
                strength.set(str(hit["strength"]))
                matched_any = True
            else:
                enabled.set(False)

        # Warn about any preset LoRAs we couldn't find on disk
        loaded_names_lower = [n.lower() for n, _, _ in self.high_lora_vars]
        for t in targets:
            needle = t["name_contains"].lower()
            if not any(needle in n for n in loaded_names_lower):
                missing.append(t["name_contains"])
        if missing:
            self._log(f"  ⚠ missing for scene '{preset.get('name')}': {', '.join(missing)}")

        # Seed the prompt (replaces whatever's there — user edits before submit)
        prompt_tpl = preset.get("prompt")
        if prompt_tpl:
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", prompt_tpl)

        self._log(f"Scene preset: {preset.get('name')} "
                   f"({sum(1 for t in targets)} LoRA{'s' if len(targets)!=1 else ''})")

    def _get_selected_loras(self):
        """Get selected LoRAs as {high_loras, low_loras} for the handler."""
        high = []
        for name, enabled, strength_var in self.high_lora_vars:
            if enabled.get():
                lora_name = name if name.endswith(".safetensors") else f"{name}.safetensors"
                try:
                    s = float(strength_var.get())
                except ValueError:
                    s = 1.0
                high.append({"name": lora_name, "strength": s})

        low = []
        for name, enabled, strength_var in self.low_lora_vars:
            if enabled.get():
                lora_name = name if name.endswith(".safetensors") else f"{name}.safetensors"
                try:
                    s = float(strength_var.get())
                except ValueError:
                    s = 1.0
                low.append({"name": lora_name, "strength": s})

        return high, low

    # ── Other UI ────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self._canvas_win, width=event.width)

    def _toggle_mode(self):
        if self.mode_var.get() == "i2v":
            self.image_frame.pack(fill="x", pady=(0, 8),
                                  after=self.prompt_text.master)
        else:
            self.image_frame.pack_forget()
        # Refresh LoRA list for the new pipeline (only if already loaded)
        if hasattr(self, "high_lora_container") and self.lora_file_var.get():
            self._load_loras()

    def _browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if path:
            self.image_path_var.set(path)
            self._show_thumbnail(path)

    def _clear_image(self):
        self.image_path_var.set("")
        self._thumb_photo = None
        self.thumb_label.configure(image="", text="Click to select image",
                                   fg="#666", font=("Segoe UI", 9),
                                   width=0, height=0)

    def _show_thumbnail(self, path):
        if not HAS_PIL or not os.path.isfile(path):
            return
        try:
            img = Image.open(path)
            img.thumbnail((400, 250), Image.LANCZOS)
            self._thumb_photo = ImageTk.PhotoImage(img)
            self.thumb_label.configure(image=self._thumb_photo, text="",
                                       width=img.width, height=img.height)
        except Exception as e:
            self.thumb_label.configure(image="", text=f"Preview unavailable: {e}",
                                       fg="#888", font=("Segoe UI", 9))

    def _save_settings(self):
        self.config["api_key"] = self.api_key_var.get()
        self.config["t2v_endpoint"] = self.t2v_var.get()
        self.config["i2v_endpoint"] = self.i2v_var.get()
        self.config["lora_file"] = self.lora_file_var.get()
        self.config["mode"] = self.mode_var.get()
        self.config["output_mode"] = self.output_mode_var.get()
        self.config["quality_preset"] = self.quality_preset_var.get()
        self.config["style_preset"] = self.style_preset_var.get()
        self.config["slg_enabled"] = self.slg_enabled_var.get()
        self.config["prompt"] = self.prompt_text.get("1.0", "end").strip()
        self.config["duration"] = self.duration_var.get()
        self.config["seed"] = self.seed_var.get()
        self.config["width"] = self.width_var.get()
        self.config["height"] = self.height_var.get()
        self.config["steps"] = self.steps_var.get()
        self.config["cfg"] = self.cfg_var.get()
        self.config["shift"] = self.shift_var.get()
        self.config["fps"] = self.fps_var.get()
        self.config["rife"] = self.rife_var.get()
        self.config["default_strength"] = self.default_strength_var.get()
        self.config["image_path"] = self.image_path_var.get()
        # Save current mode's LoRA state (enabled + strength per LoRA)
        mode = self.mode_var.get()
        lora_state = {}
        for name, enabled, strength in self.high_lora_vars:
            lora_state[name] = {
                "enabled": enabled.get(),
                "strength": strength.get(),
            }
        self.config["lora_state"][mode] = lora_state
        save_config(self.config)
        self._log("Settings saved.")

    def _log(self, msg):
        if not hasattr(self, "result_text"):
            return
        self.result_text.configure(state="normal")
        self.result_text.insert("end", msg + "\n")
        self.result_text.see("end")
        self.result_text.configure(state="disabled")

    def _build_payload(self):
        mode = self.mode_var.get()
        template = f"{mode}-standard"
        prompt = self.prompt_text.get("1.0", "end").strip()

        if not prompt:
            raise ValueError("Prompt is required")

        params = {
            "prompt": prompt,
            "duration": self.duration_var.get(),
            "mode": self.output_mode_var.get(),
            "resolution": {
                "width": self.width_var.get(),
                "height": self.height_var.get(),
            },
        }

        quality = self.quality_preset_var.get().strip()
        if quality:
            params["quality_preset"] = quality
        style = self.style_preset_var.get().strip()
        if style:
            params["style_preset"] = style
        # SLG checkbox overrides preset default only when user explicitly turned it on.
        if self.slg_enabled_var.get() and not quality:
            params["slg_enabled"] = True

        seed = self.seed_var.get().strip()
        if seed:
            params["seed"] = int(seed)

        # LoRAs (separate HIGH and LOW lists)
        high_loras, low_loras = self._get_selected_loras()
        if high_loras or low_loras:
            params["high_loras"] = high_loras
            params["low_loras"] = low_loras

        # Advanced params (only if filled)
        for key, var in [("steps", self.steps_var), ("cfg", self.cfg_var),
                         ("shift", self.shift_var), ("fps", self.fps_var),
                         ("rife_multiplier", self.rife_var)]:
            val = var.get().strip()
            if val:
                params[key] = float(val) if "." in val else int(val)

        # I2V: add input image
        if mode == "i2v":
            img_path = self.image_path_var.get()
            if not img_path or not os.path.isfile(img_path):
                raise ValueError("Input image is required for I2V")
            with open(img_path, "rb") as f:
                params["input_image"] = base64.b64encode(f.read()).decode()

        return {"input": {"template": template, "params": params}}

    def _generate(self):
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("Error", "API Key is required. Enter it in Settings.")
            return

        mode = self.mode_var.get()
        endpoint_id = self.t2v_var.get() if mode == "t2v" else self.i2v_var.get()
        if not endpoint_id:
            messagebox.showerror("Error", f"No {mode.upper()} endpoint ID configured.")
            return

        try:
            payload = self._build_payload()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.gen_btn.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Sending request...")
        self._log(f"--- New {mode.upper()} request ---")
        self._log(f"Prompt: {payload['input']['params']['prompt'][:80]}...")

        # Log selected LoRAs
        p = payload["input"]["params"]
        if p.get("high_loras") or p.get("low_loras"):
            h = [l["name"] for l in p.get("high_loras", [])]
            l = [l["name"] for l in p.get("low_loras", [])]
            self._log(f"HIGH: {', '.join(h) if h else 'none'}")
            self._log(f"LOW:  {', '.join(l) if l else 'none'}")
        else:
            self._log("LoRAs: template defaults")

        threading.Thread(target=self._run_job, args=(endpoint_id, api_key, payload),
                         daemon=True).start()

    def _run_job(self, endpoint_id, api_key, payload):
        try:
            result = send_request(endpoint_id, api_key, payload)
            job_id = result["id"]
            self.root.after(0, lambda: self._log(f"Job ID: {job_id}"))
            self.root.after(0, lambda: self.status_var.set(f"In queue... (job {job_id[:8]})"))

            while True:
                time.sleep(5)
                status = poll_status(endpoint_id, api_key, job_id)
                state = status.get("status", "UNKNOWN")
                self.root.after(0, lambda s=state: self.status_var.set(f"Status: {s}"))

                if state == "COMPLETED":
                    self.root.after(0, lambda: self._handle_result(status))
                    return
                elif state in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    error = status.get("output", {}).get("error", "Unknown error")
                    self.root.after(0, lambda e=error: self._handle_error(e))
                    return

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            if hasattr(e, "response") and e.response is not None:
                err_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            self.root.after(0, lambda m=err_msg: self._handle_error(m))

    def _handle_result(self, status):
        self.progress.stop()
        self.gen_btn.configure(state="normal")

        output = status.get("output", {})
        videos = output.get("videos", [])
        metadata = output.get("metadata", {})

        exec_time = status.get("executionTime", 0) / 1000
        delay = status.get("delayTime", 0) / 1000
        gen_time = metadata.get("generation_time_seconds", 0)

        self._log(f"Generation time: {gen_time:.1f}s")
        self._log(f"RunPod delay: {delay:.1f}s | Execution: {exec_time:.1f}s")

        if videos:
            video_data = base64.b64decode(videos[0]["data"])
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"wan22_{self.mode_var.get()}_{timestamp}.mp4"
            filepath = OUTPUT_DIR / filename

            with open(filepath, "wb") as f:
                f.write(video_data)

            size_mb = len(video_data) / (1024 * 1024)
            self._log(f"Saved: {filepath} ({size_mb:.1f} MB)")
            self.status_var.set(f"Done! Saved {filename}")

            try:
                os.startfile(str(filepath))
            except Exception:
                pass
        else:
            self.status_var.set("Done but no video returned")
            self._log("No video in response")

    def _handle_error(self, error):
        self.progress.stop()
        self.gen_btn.configure(state="normal")
        self.status_var.set("Failed")
        self._log(f"ERROR: {error}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
