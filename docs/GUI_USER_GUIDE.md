# WAN 2.2 Serverless NSFW — Desktop GUI

A point-and-click tool that generates explicit AI videos by talking to a powerful GPU server running in the cloud. You don't need a fancy graphics card — your computer just sends a request, the server does the heavy lifting, and the finished video lands in this folder.

---

## Table of Contents

1. [What this actually does](#what-this-actually-does)
2. [What you need before you start](#what-you-need-before-you-start)
3. [First-time setup (5 minutes, one-time)](#first-time-setup)
4. [Your first video — the absolute basics](#your-first-video)
5. [The interface, section by section](#the-interface-section-by-section)
6. [Quality preset — what each option does](#quality-preset)
7. [Style preset — controlling the look](#style-preset)
8. [Scene preset — the magic auto-pilot](#scene-preset)
9. [LoRAs — the explicit content packs](#loras)
10. [Tips for better videos](#tips-for-better-videos)
11. [Troubleshooting](#troubleshooting)
12. [Recovering a job that vanished](#recovering-a-job)
13. [File reference](#file-reference)
14. [Glossary](#glossary)

---

## What this actually does

You type a prompt (or pick a preset that writes one for you), maybe drop in an image, hit Generate. About 3–10 minutes later you get a high-quality 3–5 second video.

The work happens on **RunPod**, a cloud service that rents out powerful GPUs by the second. This GUI is just the remote control.

**Two modes:**

| Mode | What it does | Needs |
|---|---|---|
| **T2V** (Text → Video) | Imagines the whole scene from just your written prompt | Just a prompt |
| **I2V** (Image → Video) | Takes your input image and animates it forward in time, following your prompt | An image + a prompt |

I2V is what most people want — give it a still photo of a character you like, describe what should happen next, get a video.

---

## What you need before you start

1. **A Windows PC** (this guide is Windows-focused; Mac/Linux work too but `.bat` files won't)
2. **Python 3.10 or newer** — free, official download from [python.org](https://python.org)
3. **A RunPod API key** — free account at [runpod.io](https://runpod.io), then create an API key. You only pay when you generate (about 5–15 cents per video).
4. **About 5 minutes** to do the one-time setup.

That's it. You don't need a graphics card. You don't need to know anything about AI. You don't need to install ComfyUI or anything else.

---

## First-time setup

Do this once. Never again.

### Step 1 — Install Python

1. Download Python from [python.org/downloads](https://python.org/downloads)
2. Run the installer.
3. **CRITICAL**: On the very first screen of the installer, **tick the checkbox that says "Add Python to PATH"** (it's at the bottom). If you miss this, nothing will work and you'll need to reinstall.
4. Click "Install Now". Wait ~2 minutes.

### Step 2 — Install the dependencies

This folder you're reading the README in has a file called `install_deps.bat`.

1. **Double-click it.**
2. A black window opens, downloads two small Python packages (`requests`, `Pillow`), and tells you when it's done.
3. Press any key to close the window.

You won't need to do this again unless you wipe Python.

### Step 3 — Launch the GUI

1. **Double-click `run.bat`.**
2. A purple-themed window opens. This is the GUI.

### Step 4 — Add your RunPod API key

Look at the top of the window — there's a **Settings** section.

1. Get an API key from [runpod.io/console/user/settings](https://runpod.io/console/user/settings) (click "+ API Key" → name it anything → copy the long string).
2. Paste that string into the **API Key** field.
3. The **T2V Endpoint** and **I2V Endpoint** fields are **already filled in** — don't touch them.
4. Click the **Save Settings** button on the right.

You'll see "Settings saved." in the Status box at the bottom.

### Step 5 — Load the LoRA list

LoRAs are content packs (more on these later). The list of available ones lives in `loras.txt` right next to this README.

1. In the **LoRAs** section, the **List** field already shows `loras.txt`.
2. Click the **Load** button next to it.
3. The HIGH and LOW columns fill up with checkboxes — these are all the available LoRAs.

**You're done with setup.** Future runs: just double-click `run.bat`, everything is remembered.

---

## Your first video

Let's do a quick test to make sure everything works.

1. Make sure mode is **T2V** (top left radio button).
2. Leave the default prompt as-is, or type something simple like "a golden retriever running through a meadow at sunset".
3. Leave **Duration** at 3 seconds, **Resolution** at the default (832×480), **Seed** at 42.
4. Click the big purple **Generate Video** button.
5. Watch the **Status** area at the bottom. You'll see:
   - "Sending request..."
   - "In queue..." (waiting for a GPU)
   - "Status: IN_PROGRESS" (it's generating)
   - "Done! Saved wan22_t2v_..."
6. Your video pops open in Windows' default player automatically.

First video takes longer because the server has to "wake up" (~1–2 minutes of cold start). Subsequent videos are faster.

The finished file is in the `output/` folder right next to this README.

**Total cost of that test: about 5 cents.**

---

## The interface, section by section

Top to bottom in the window:

### Settings
Your API key + endpoint IDs. Set once, forget. Click **Save Settings** any time you change something here.

### Mode bar
- **Text to Video (T2V)** vs **Image to Video (I2V)** — pick one.
- **Fast (no RIFE)** vs **Slow-mo (RIFE 2x)** — Slow-mo doubles the frames smoothly so motion looks buttery. Fast skips that step for quicker output.

### Presets
The convenience section. Three controls:

- **Quality** dropdown — overall quality level (see [Quality preset](#quality-preset) below)
- **Style** dropdown — controls camera/lighting/realism (see [Style preset](#style-preset))
- **SkipLayerGuidance checkbox** — leave this **OFF**. It's currently disabled because of a backend incompatibility. The label says "DISABLED — needs TeaCache".
- **Scene** dropdown — pre-built explicit scene recipes (see [Scene preset](#scene-preset))

### Prompt
Where you describe what you want. If you picked a Scene preset, this gets filled in automatically — just edit the `[BRACKETED]` parts.

### Input Image (I2V only)
Shows up when you're in I2V mode. Click the big preview area or hit **Browse** to load an image. The image's contents anchor what the video looks like.

### LoRAs
Two columns: **HIGH Loader** and **LOW Loader**. These are pass-specific content add-ons. See [LoRAs](#loras) for the full explanation.

Below them: **All** / **None** buttons to bulk-toggle, and a **Default str** field for default LoRA strength.

### Speed LoRAs (collapsible)
Click the "▸ Speed LoRAs" bar to expand. These are the "Lightning" acceleration LoRAs that make 4-step generation possible. **The Quality preset manages these automatically** — usually you don't need to touch this section.

### Parameters
- **Duration** — video length in seconds (1–30)
- **Seed** — same seed + same prompt + same LoRAs = same video. Useful for reproducing or slightly varying a good result. Type a number or leave blank for random.
- **Resolution dropdown** — pre-made aspect ratios (16:9 landscape, 9:16 vertical, square, etc.). Picking one fills in Width/Height automatically.
- **Width / Height** — manual override. The dropdown switches to "Custom" when you edit these.

### Advanced (optional)
Leave these blank unless you know what you're doing. They override the preset values.

- **Steps** — sampling steps. More = slower + slightly higher detail.
- **CFG** — how literally the model follows your prompt. Higher = more literal but stiffer.
- **Shift** — affects motion vs. detail balance.
- **FPS** — output framerate.
- **RIFE mult** — frame interpolation multiplier for slow-mo.

### Generate Video button
Big purple button. Click it. Wait.

### Status
Logs everything happening. Generated videos auto-open. If something fails, the error shows up here.

---

## Quality preset

Pick one of these and the GUI auto-configures sampler settings + auto-picks the right Lightning LoRA.

| Preset | What it does | Use when |
|---|---|---|
| *(blank)* | No automatic tuning. You're in manual mode. | Advanced users only |
| **fast** | 4 steps, simple sampler, original Seko Lightning. Fastest output. | Quick previews, testing prompts |
| **quality** *(recommended default)* | 6 steps, better sampler (`res_multistep/beta`), asymmetric CFG (3.0 HIGH / 1.0 LOW), newer lightx2v Lightning at reduced strength | Most generations |
| **hero** | 12 steps, max CFG, max effort | Important hero shots — slower but best quality |

**Pick `quality` for almost everything.** It's the sweet spot.

---

## Style preset

This is purely a prompt modifier. When you pick a style, the GUI appends camera/lighting/quality tokens to your prompt.

| Preset | Appends |
|---|---|
| *(blank)* | nothing |
| **realistic** | handheld close-up, 35mm film, natural window light, golden hour, raw skin texture |
| **cinematic_film** | slow dolly in, anamorphic lens, low angle, moody lighting, rim light, bokeh |
| **pov_handheld** | first person POV, handheld tracking, bedside lamp glow, intimate |

You can combine these with any scene preset. Style adds the "how it's shot" layer on top.

---

## Scene preset

This is the killer feature.

You pick a scene like "Doggystyle Forward-Bend Physics" or "Cowgirl Bounce + Anatomy", and the GUI:

1. Auto-selects the right content LoRAs (e.g., the Doggy LoRA + the Bouncy Physics LoRA + the Anatomy detail LoRA)
2. Sets their strengths according to research (so they stack cleanly without overcooking)
3. Pre-fills the prompt with a strong, scene-specific template using creator-verified trigger words

There are **42 presets** organized by category:

| Category | Examples |
|---|---|
| **POV Position** | POV Missionary (T2V + I2V), Doggystyle POV Aggressive |
| **Position** | Cowgirl Bounce, Doggystyle Forward-Bend, Reverse Cowgirl, MQ Doggy Diagonal, Piledriver, Cowgirl + Man Grabs Tits, Casting Couch Doggy |
| **Oral** | POV Big Cock Blowjob, Rough Deepthroat, Gentle Oral, Oral Classic, Titjob with Cumshot, Rough Tit Fuck POV |
| **Climax** | Oral → Facial Cumshot, Missionary Climax, Facial Cumshot, Cowgirl Creampie, Doggystyle Pull-Out, Cum In Mouth Swallow, Missionary Creampie |
| **Toys** | Dildo Ride Solo, Dildo Closeup Macro, Dildo Full-Body, Dildo + Fingering Solo |
| **Multi-partner** | Double Penetration (Behind), Double Penetration (Cowgirl-DP Front) |
| **Solo** | Solo Fingering (T2V/I2V), Instagirl Selfie Strip |
| **Foreplay** | Breast Play Self-Grab, Partner Breast Grab + Kiss, Passionate Kissing, Kiss → Strip → Breast Play |
| **Movement** | Twerk V3, Twerk Realism |
| **Expression** | Wink Tease |
| **Anatomy** | Pussy Macro, Pussy Play Solo |

### Pipeline tags

Each scene preset is tagged:
- **[BOTH]** — works in both T2V and I2V mode
- **[T2V]** — text-to-video only
- **[I2V]** — image-to-video only

If you pick an I2V-only preset while in T2V mode, the GUI will tell you to switch modes first.

### How to use scene presets

1. Pick **Mode** (T2V or I2V).
2. Pick **Quality** preset (use `quality`).
3. Pick a **Scene** preset.
4. The prompt fills in with placeholders like `[SUBJECT_DESCRIPTION]`. Edit those.
   - Example: `[SUBJECT_DESCRIPTION]` → "a curvy young woman with long dark hair and natural breasts"
5. (I2V) Load an input image.
6. Generate.

---

## LoRAs

This is the most important concept. Read this once and you'll understand 90% of how the tool works.

### What's a LoRA?

A LoRA is a small AI add-on file that teaches the base model to do something specific. Like a plugin. There are LoRAs for:
- **Positions** — cowgirl, doggy, missionary, piledriver, etc.
- **Actions** — cumshot, fingering, breast play, etc.
- **Anatomy** — detailed pussy rendering, etc.
- **Style** — Instagirl realism, etc.
- **Acceleration** — Lightning LoRAs that let you generate in 4 steps instead of 30+

You stack them together to get the scene you want.

### HIGH and LOW — what do they mean?

WAN 2.2 generates videos in two passes:
- **HIGH pass** — establishes the overall motion and composition (first half of generation)
- **LOW pass** — adds fine detail like skin texture, genital realism, face clarity (second half)

Most LoRAs come in HIGH+LOW pairs — you load one in each loader for full effect. The GUI shows them in two columns:

```
HIGH Loader              LOW Loader
☐ Doggystyle-HIGH        ☐ Doggystyle-LOW
☐ Cumshot-High           ☐ Cumshot-Low
☐ PussyLoRA-HighNoise    ☐ PussyLoRA-LowNoise
```

**Ticking HIGH automatically ticks LOW** — the GUI keeps them paired. The strength field also syncs.

### Three categories of LoRAs

1. **Base / Always-on** — quality enhancers like `SECRET_SAUCE` and `FusionX`. Stay enabled most of the time. Auto-enabled by default.

2. **Speed (Lightning)** — these live in the collapsible "Speed LoRAs" section below the main LoRA columns. The Quality preset auto-picks the right one. Don't stack multiple Lightnings — they conflict.

3. **Content** — the action / position / anatomy LoRAs in the main HIGH/LOW columns. You pick which ones to enable based on what scene you want. Scene presets pick these for you.

### Stacking rules (the GUI enforces these via presets)

- **Max 3 content LoRAs per pass.** More starts to degrade quality.
- **Combined strength ≤ 2.5 per pass.** Too much LoRA = oversaturated, weird output.
- **One Lightning at a time.** Either Seko OR lightx2v, never both.
- **HIGH-only and LOW-only LoRAs exist** — some creators only trained one side. Scene presets handle these correctly automatically.

### Adjusting strength

Each LoRA row has a number field on the right (0.0 to ~1.5). This is how strongly that LoRA affects the output.

- **1.0** = full effect (default)
- **0.7** = moderate
- **0.4** = subtle (good for "background" enhancement LoRAs like Instagirl)
- **0** = effectively off (just untick the box instead)

Scene presets set these intelligently. You can fine-tune after.

---

## Tips for better videos

### Use scene presets for explicit content
The 42 scene recipes have been researched — they use creator-verified trigger phrases the LoRAs were trained on. Writing prompts from scratch usually gets weaker results.

### For I2V: the input image dominates
The first second of the video looks a LOT like your input image. If you want a different setting (bedroom instead of beach), you need to either describe that strongly in the prompt OR re-render the input image first.

### Match aspect ratio
If your input image is portrait (taller than wide), pick a vertical resolution preset like `9:16 vertical — 480×832`. Mixing portrait input with landscape output produces letterboxing or weird crops.

### Don't go above 5 seconds
WAN 2.2 starts losing temporal coherence past ~5 seconds. Characters morph, anatomy drifts. Stick to 3–5 seconds. If you need longer, generate two clips and stitch them together in a video editor.

### Set seed = blank for variety, fixed for refinement
- Got something you like but want a tweak? Note the seed (it's logged in the Status box), then change one thing (prompt detail, LoRA strength) — same base composition with your edit applied.
- Want totally different output each time? Leave seed blank — it'll randomize.

### Use higher Quality preset for hero shots
`hero` preset costs ~2× the time of `quality`, but produces noticeably better detail for shots you'll actually use. Use `fast` for previews, `quality` for daily use, `hero` for keepers.

### Don't combine incompatible LoRAs
- **BigCockOral + Piledriver** = broken penises. Don't stack them.
- **Oral-Ins or CumShot + Lightning LoRAs** = degraded output per creator warning. Use these only with non-Lightning presets.
- The scene presets respect these rules.

### Style preset is free quality
Picking a Style preset adds professional camera/lighting language to your prompt. Use it.

---

## Troubleshooting

### "Python is not installed" when running `run.bat`
You missed the "Add Python to PATH" checkbox during install. Re-run the Python installer, tick it, finish install, then re-run `install_deps.bat` then `run.bat`.

### Generated video has weird/broken anatomy
- Lower your content LoRA strengths (try 0.7 instead of 1.0).
- Make sure you didn't stack too many LoRAs (max 3).
- Use the `quality` preset — it picks better defaults.
- For close-ups especially, add `PussyLoRA` at 0.4 — fixes most anatomy distortion.

### Generated video is slow-motion / sluggish
You probably have a Lightning LoRA enabled but also picked a high CFG. Either:
- Use the `quality` preset (it tunes this automatically)
- Or manually drop Lightning strength to 0.6–0.7 if you really want CFG above 1.0

### Video is just a blurry blob / static / nothing happens
- Your trigger phrase isn't in the prompt. Open the scene preset's prompt and make sure the trigger words are there verbatim (e.g., "p1ledriv3r", "twuuur", "Instagirl").
- Or the LoRA strength is too low. Bump it.

### "out of memory" error in Status
The server got assigned a smaller GPU than usual. Two options:
- Re-run the same job (next worker spin-up might land a bigger GPU).
- Drop resolution to 480p (`16:9 landscape — 832×480`).
- Shorten duration to 3 seconds.

### "ConnectionError: HTTPSConnectionPool ... WinError 10013"
Your Windows machine ran out of network sockets or Hyper-V/Docker reserved your port range. Fix:
1. Wait 30 seconds and try again (often transient).
2. If persistent, open Command Prompt **as administrator** and run:
   ```
   net stop winnat
   net start winnat
   ```

### My job seems to be running but the GUI lost connection
The job is still running on the server even if your local GUI errored. See [Recovering a job](#recovering-a-job).

### Generated video doesn't auto-play
Your default video player might not be set. Open the `output/` folder manually and double-click the video. Or right-click → Open with → choose a player (VLC works great).

---

## Recovering a job

If the GUI loses connection, errors out, or the worker gets stuck, your job ID is still in the Status log. You can fetch the finished video manually.

1. Find the **Job ID** in the Status log (looks like `bba5a760-21d3-4a8f-a527-296fbb9876c2-e1`).
2. Open PowerShell or Command Prompt in this folder.
3. Run:
   ```
   py fetch_job.py <job_id>
   ```
   Replace `<job_id>` with your actual ID. Example:
   ```
   py fetch_job.py bba5a760-21d3-4a8f-a527-296fbb9876c2-e1
   ```
4. The video saves to `output/`. Open it manually.

The script uses your saved API key, so credentials are handled.

If the job was a T2V job, add `t2v` at the end:
```
py fetch_job.py <job_id> t2v
```

---

## File reference

Everything in this folder explained:

| File / Folder | What it is | Do you edit it? |
|---|---|---|
| `README.md` | This guide | No |
| `gui.py` | The actual program (Python source) | No — read-only |
| `gui_config.json` | Your saved settings: API key, endpoints, last prompt, LoRA selections | Auto-managed by the GUI — don't hand-edit |
| `loras.txt` | List of all available LoRAs. Each line is a LoRA name. Lines starting with `#` are comments. | Optional — uncomment a line to make a LoRA appear in the GUI |
| `scene_presets.json` | The 42 scene recipes. Each has LoRAs + prompt template + caveats. | Optional — advanced users can edit/add presets here |
| `requirements.txt` | The two Python packages this needs | No |
| `install_deps.bat` | Run once after installing Python. Installs `requests` + `Pillow`. | No |
| `run.bat` | Daily launcher. Double-click to start the GUI. | No |
| `fetch_job.py` | Manual job-recovery script (see [Recovering a job](#recovering-a-job)) | No |
| `output/` | Generated videos land here. Auto-created on first run. | This is yours — organize/delete however |

---

## Glossary

**API key** — a long string of letters/numbers that proves you have permission to use the RunPod service. Treat it like a password. Don't share it.

**CFG** — "Classifier-Free Guidance". A number that controls how strictly the AI follows your prompt. Low CFG (~1) = AI is creative and adds its own ideas. High CFG (~5+) = AI tries to obey the prompt to the letter but output gets stiff.

**Cold start** — the first ~30–120 seconds before generation actually begins, while the cloud server "wakes up" and loads models. Subsequent generations on the same worker skip this.

**Endpoint** — the unique address of a specific server configuration. Your T2V and I2V endpoints point to two different server setups optimized for those modes.

**Frame interpolation (RIFE)** — a separate AI step that doubles or quadruples the framerate by inventing intermediate frames. Makes motion smooth. The "Slow-mo" mode uses this.

**HIGH / LOW** — the two passes of WAN 2.2 video generation. HIGH = motion/composition. LOW = detail/refinement.

**I2V** — Image-to-Video. Takes an image + a prompt, animates the scene forward.

**Job ID** — a unique identifier for each generation request. Like a tracking number. Shown in the Status log.

**Lightning LoRA** — a special LoRA that makes generation happen in 4 steps instead of 30+. Way faster.

**LoRA** — a small AI add-on file that teaches the base model to do something specific (a position, an action, an art style).

**Prompt** — the text description you type telling the AI what to generate.

**Sampler** — the AI algorithm that does the actual image-generating work. Different samplers produce different aesthetics.

**Seed** — a starting number for the AI's random generation. Same seed + same prompt = same video.

**Serverless** — RunPod scales the server up only when you make a request, scales it down when idle. You pay per second of actual use, not 24/7.

**Steps** — how many iterations the AI runs to refine the video. More = slower but slightly better.

**T2V** — Text-to-Video. Generates a video from just a prompt, no input image.

**Trigger word** — a specific phrase a LoRA's creator trained it on. You need to include it in your prompt for the LoRA to "fire". Example: `p1ledriv3r` for the Piledriver LoRA.

---

## Need help?

If something doesn't work and isn't covered above, give the person who sent you this folder the exact error message from the **Status** box at the bottom of the GUI. They can usually diagnose from that.

Happy generating.
