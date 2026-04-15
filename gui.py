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

    The file should list HIGH variants only. Example:
        Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0
        wan/SECRET_SAUCE_WAN2.1_14B_fp8

    Lines containing 'HIGH' will auto-pair with a LOW counterpart.
    Lines without HIGH/LOW are shared LoRAs (used in both passes).
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
        canvas = tk.Canvas(self.root, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.main = ttk.Frame(canvas, padding=16)

        self.main.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.main, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

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
        ttk.Entry(self.image_frame, textvariable=self.image_path_var, width=50).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(self.image_frame, text="Browse", command=self._browse_image).pack(side="right")

        # ── LoRAs ───────────────────────────────────────────────────────
        lora_frame = ttk.LabelFrame(main, text="LoRAs", padding=8)
        lora_frame.pack(fill="x", pady=(0, 8))

        # File selector row
        file_row = ttk.Frame(lora_frame)
        file_row.pack(fill="x", pady=(0, 8))

        ttk.Label(file_row, text="LoRA list file:").pack(side="left", padx=(0, 4))
        self.lora_file_var = tk.StringVar(value=self.config.get("lora_file", ""))
        ttk.Entry(file_row, textvariable=self.lora_file_var, width=40).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(file_row, text="Browse", command=self._browse_lora_file).pack(side="left", padx=(0, 4))
        ttk.Button(file_row, text="Load", command=self._load_loras).pack(side="left")

        # LoRA checkboxes container
        self.lora_container = ttk.Frame(lora_frame)
        self.lora_container.pack(fill="x")

        # Default strength
        strength_row = ttk.Frame(lora_frame)
        strength_row.pack(fill="x", pady=(8, 0))
        ttk.Label(strength_row, text="Default strength:").pack(side="left", padx=(0, 4))
        self.default_strength_var = tk.StringVar(value="1.0")
        ttk.Entry(strength_row, textvariable=self.default_strength_var, width=6).pack(side="left")

        ttk.Button(strength_row, text="Select All", command=self._select_all_loras).pack(side="right", padx=4)
        ttk.Button(strength_row, text="Deselect All", command=self._deselect_all_loras).pack(side="right", padx=4)

        # Auto-load if file was saved
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
        # Clear existing
        for widget in self.lora_container.winfo_children():
            widget.destroy()
        self.lora_vars.clear()

        filepath = self.lora_file_var.get()
        names = load_lora_list(filepath)

        if not names:
            ttk.Label(self.lora_container, text="No LoRAs loaded. Select a text file.",
                      foreground="#888").pack(anchor="w")
            return

        for name in names:
            row = ttk.Frame(self.lora_container)
            row.pack(fill="x", pady=1)

            enabled = tk.BooleanVar(value=False)
            strength = tk.StringVar(value=self.default_strength_var.get())

            cb = ttk.Checkbutton(row, text=name, variable=enabled)
            cb.pack(side="left", padx=(0, 8))

            ttk.Label(row, text="str:", foreground="#888").pack(side="right", padx=(4, 0))
            ttk.Entry(row, textvariable=strength, width=5).pack(side="right")

            # Show if it's a HIGH/LOW pair or shared
            if "HIGH" in name:
                low_name = name.replace("HIGH", "LOW")
                ttk.Label(row, text=f"(+ {low_name})", foreground="#666",
                          font=("Segoe UI", 8)).pack(side="left", padx=4)

            self.lora_vars.append((name, enabled, strength))

        self._log(f"Loaded {len(names)} LoRAs from {Path(filepath).name}")

    def _select_all_loras(self):
        for _, enabled, _ in self.lora_vars:
            enabled.set(True)

    def _deselect_all_loras(self):
        for _, enabled, _ in self.lora_vars:
            enabled.set(False)

    def _get_selected_loras(self):
        """Get selected LoRAs as list of {name, strength} dicts.

        Names include .safetensors extension. The handler's set_loras
        will handle HIGH/LOW pairing automatically.
        """
        selected = []
        for name, enabled, strength_var in self.lora_vars:
            if enabled.get():
                lora_name = name
                if not lora_name.endswith(".safetensors"):
                    lora_name += ".safetensors"
                try:
                    strength = float(strength_var.get())
                except ValueError:
                    strength = 1.0
                selected.append({"name": lora_name, "strength": strength})
        return selected

    # ── Other UI ────────────────────────────────────────────────────────

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

    def _save_settings(self):
        self.config["api_key"] = self.api_key_var.get()
        self.config["t2v_endpoint"] = self.t2v_var.get()
        self.config["i2v_endpoint"] = self.i2v_var.get()
        self.config["lora_file"] = self.lora_file_var.get()
        save_config(self.config)
        self._log("Settings saved.")

    def _log(self, msg):
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

        # LoRAs
        loras = self._get_selected_loras()
        if loras:
            params["loras"] = loras

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
        loras = payload["input"]["params"].get("loras", [])
        if loras:
            self._log(f"LoRAs: {', '.join(l['name'] for l in loras)}")
        else:
            self._log("LoRAs: using template defaults")

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
