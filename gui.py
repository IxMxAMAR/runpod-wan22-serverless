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
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        "api_key": "",
        "t2v_endpoint": "",
        "i2v_endpoint": "",
        "lora_file": "",
    }


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

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

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

        self.mode_var = tk.StringVar(value="t2v")
        ttk.Radiobutton(mode_frame, text="Text to Video (T2V)",
                        variable=self.mode_var, value="t2v",
                        command=self._toggle_mode).pack(side="left", padx=8)
        ttk.Radiobutton(mode_frame, text="Image to Video (I2V)",
                        variable=self.mode_var, value="i2v",
                        command=self._toggle_mode).pack(side="left", padx=8)

        # ── Prompt ──────────────────────────────────────────────────────
        prompt_frame = ttk.LabelFrame(main, text="Prompt", padding=8)
        prompt_frame.pack(fill="x", pady=(0, 8))

        self.prompt_text = tk.Text(prompt_frame, height=3, bg="#16213e", fg="#e0e0e0",
                                   insertbackground="#e0e0e0", font=("Segoe UI", 10),
                                   wrap="word", relief="flat", bd=2)
        self.prompt_text.pack(fill="x")
        self.prompt_text.insert("1.0", "A golden retriever running through a sunlit meadow")

        # ── I2V Image ───────────────────────────────────────────────────
        self.image_frame = ttk.LabelFrame(main, text="Input Image (I2V)", padding=8)
        self.image_path_var = tk.StringVar()
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
        self.default_strength_var = tk.StringVar(value="1.0")
        ttk.Entry(btn_row, textvariable=self.default_strength_var, width=5).pack(side="right")
        ttk.Label(btn_row, text="Default", foreground="#888").pack(side="right", padx=(0, 4))

        if self.lora_file_var.get():
            self._load_loras()

        # ── Parameters ──────────────────────────────────────────────────
        params = ttk.LabelFrame(main, text="Parameters", padding=8)
        params.pack(fill="x", pady=(0, 8))

        row = 0
        ttk.Label(params, text="Duration (sec):").grid(row=row, column=0, sticky="w", padx=4)
        self.duration_var = tk.IntVar(value=3)
        ttk.Spinbox(params, from_=1, to=30, textvariable=self.duration_var,
                     width=8).grid(row=row, column=1, sticky="w", padx=4)

        ttk.Label(params, text="Seed:").grid(row=row, column=2, sticky="w", padx=4)
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(params, textvariable=self.seed_var, width=12).grid(
            row=row, column=3, sticky="w", padx=4)

        row = 1
        ttk.Label(params, text="Width:").grid(row=row, column=0, sticky="w", padx=4)
        self.width_var = tk.IntVar(value=832)
        ttk.Entry(params, textvariable=self.width_var, width=8).grid(
            row=row, column=1, sticky="w", padx=4)

        ttk.Label(params, text="Height:").grid(row=row, column=2, sticky="w", padx=4)
        self.height_var = tk.IntVar(value=480)
        ttk.Entry(params, textvariable=self.height_var, width=8).grid(
            row=row, column=3, sticky="w", padx=4)

        # ── Advanced ────────────────────────────────────────────────────
        adv = ttk.LabelFrame(main, text="Advanced (optional)", padding=8)
        adv.pack(fill="x", pady=(0, 8))

        ttk.Label(adv, text="Steps:").grid(row=0, column=0, sticky="w", padx=4)
        self.steps_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.steps_var, width=8).grid(
            row=0, column=1, sticky="w", padx=4)

        ttk.Label(adv, text="CFG:").grid(row=0, column=2, sticky="w", padx=4)
        self.cfg_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.cfg_var, width=8).grid(
            row=0, column=3, sticky="w", padx=4)

        ttk.Label(adv, text="Shift:").grid(row=0, column=4, sticky="w", padx=4)
        self.shift_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.shift_var, width=8).grid(
            row=0, column=5, sticky="w", padx=4)

        ttk.Label(adv, text="FPS:").grid(row=1, column=0, sticky="w", padx=4)
        self.fps_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.fps_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4)

        ttk.Label(adv, text="RIFE mult:").grid(row=1, column=2, sticky="w", padx=4)
        self.rife_var = tk.StringVar(value="")
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

        self._toggle_mode()

    # ── LoRA Management ─────────────────────────────────────────────────

    def _browse_lora_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.lora_file_var.set(path)
            self._load_loras()

    def _load_loras(self):
        for widget in self.high_lora_container.winfo_children():
            widget.destroy()
        for widget in self.low_lora_container.winfo_children():
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

        # Skip lines that are LOW variants — they'll be auto-derived from HIGH
        low_patterns = [lo for _, lo in HIGH_LOW_PATTERNS]
        filtered = []
        for name in names:
            # Skip if this is a LOW variant whose HIGH counterpart is already in the list
            is_low = any(lo in name for lo in low_patterns) and not any(
                hi in name for hi, _ in HIGH_LOW_PATTERNS if hi != "High" or name.count("High") > 0
            )
            # Simpler: if line is pure LOW (no HIGH pattern), check if HIGH variant was listed
            skip = False
            for hi, lo in HIGH_LOW_PATTERNS:
                if lo in name and hi not in name:
                    # This is a LOW-only line. Check if HIGH counterpart is in names.
                    high_equiv = name.replace(lo, hi)
                    if high_equiv in names:
                        skip = True
                        break
            if not skip:
                filtered.append(name)

        for name in filtered:
            # HIGH column gets the name as-is
            self._add_lora_row(self.high_lora_container, self.high_lora_vars,
                               name, default_str)
            # LOW column gets derived name (or same name if shared)
            low_name = derive_low_name(name)
            if low_name is None:
                low_name = name  # Shared LoRA
            self._add_lora_row(self.low_lora_container, self.low_lora_vars,
                               low_name, default_str)

        self._log(f"Loaded {len(filtered)} LoRAs")

    def _add_lora_row(self, container, var_list, name, default_str):
        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        enabled = tk.BooleanVar(value=False)
        strength = tk.StringVar(value=default_str)
        # Truncate display name for compactness
        display = Path(name).stem if len(name) > 30 else name
        ttk.Checkbutton(row, text=display, variable=enabled).pack(side="left")
        ttk.Entry(row, textvariable=strength, width=4).pack(side="right")
        var_list.append((name, enabled, strength))

    def _select_all_loras(self):
        for _, enabled, _ in self.high_lora_vars + self.low_lora_vars:
            enabled.set(True)

    def _deselect_all_loras(self):
        for _, enabled, _ in self.high_lora_vars + self.low_lora_vars:
            enabled.set(False)

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
            "resolution": {
                "width": self.width_var.get(),
                "height": self.height_var.get(),
            },
        }

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
