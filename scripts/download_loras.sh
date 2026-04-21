#!/bin/bash
# Download recommended NSFW LoRAs + newer Lightning distillations onto the
# RunPod network volume at /runpod-volume/models/loras.
#
# Run this ON YOUR RUNPOD POD (not locally) so the files land on the mounted
# network volume. Example:
#   1. SSH into a Pod with your volume attached
#   2. export CIVITAI_TOKEN=your_token_from_civitai.com/user/account
#   3. bash download_loras.sh
#
# Requires: wget, curl, python3, unzip (for ZIP-packaged LoRAs).

set -euo pipefail

: "${CIVITAI_TOKEN:?Set CIVITAI_TOKEN first. Get it at https://civitai.com/user/account → API Keys}"
LORA_DIR="${LORA_DIR:-/runpod-volume/models/loras}"
mkdir -p "$LORA_DIR"
cd "$LORA_DIR"

# ─── helpers ─────────────────────────────────────────────────────────────────

# Fetch latest version ID from a Civitai model page ID.
civitai_latest_version() {
    curl -sf "https://civitai.com/api/v1/models/$1" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["modelVersions"][0]["id"])'
}

# Download a specific Civitai version by version ID.
# Civitai redirects to the primary file for the version — we follow redirects
# and use -O to force the target name. ZIP detection runs after download.
dl_civitai_version() {
    local version_id=$1 target=$2
    if [[ -f "$target" ]]; then
        echo "⇢ skip $target (already exists)"
        return
    fi
    echo "⇢ $target (version $version_id)"
    wget -q --show-progress \
        -O "$target" \
        "https://civitai.com/api/download/models/${version_id}?token=${CIVITAI_TOKEN}"
    _maybe_unzip "$target"
}

# Download latest Civitai version for a model ID.
dl_civitai_latest() {
    local model_id=$1 target=$2
    local version_id
    version_id=$(civitai_latest_version "$model_id")
    dl_civitai_version "$version_id" "$target"
}

# If a downloaded "safetensors" file is actually a ZIP, rename and extract.
# Extracts all .safetensors from the ZIP into $LORA_DIR.
_maybe_unzip() {
    local path=$1
    # ZIPs start with magic bytes "PK\x03\x04"
    if head -c 4 "$path" 2>/dev/null | od -c | head -1 | grep -q 'P   K'; then
        local zipname="${path%.safetensors}.zip"
        mv "$path" "$zipname"
        echo "  ↳ $zipname is a ZIP — extracting .safetensors files"
        unzip -j -n -q "$zipname" '*.safetensors' -d "$LORA_DIR" || true
        # Keep the ZIP so reruns can skip re-download; delete if you want to save space.
    fi
}

# Public Hugging Face — no auth needed for these repos.
dl_hf() {
    local url=$1 target=$2
    if [[ -f "$target" ]]; then
        echo "⇢ skip $target (already exists)"
        return
    fi
    echo "⇢ $target"
    wget -q --show-progress -O "$target" "$url"
}

# ─── Action #8: newer Lightning distillation LoRAs (lightx2v, public) ────────
echo ""
echo "== Lightning LoRA upgrades (lightx2v org on Hugging Face) =="

dl_hf "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
      "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"

dl_hf "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
      "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"

dl_hf "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors" \
      "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors"

dl_hf "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors" \
      "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors"

# Kijai's older Seko re-uploads — keep as fallback
dl_hf "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors" \
      "Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors"

dl_hf "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors" \
      "Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors"

# ─── Action #6: NSFW LoRAs (Civitai, needs token) ────────────────────────────
echo ""
echo "== Civitai NSFW LoRAs =="
echo "(using latest version of each model ID — no hardcoded version IDs to rot)"

# -- I2V-focused (HIGH variants; LOW auto-derives in GUI where pattern matches)
dl_civitai_latest 1874153 "Oral-Insertion-WAN22-I2V-HIGH.safetensors"
dl_civitai_latest 2121078 "BigCockOral-Wan22-I2V-HIGH.safetensors"
dl_civitai_latest 1913617 "Wan22-BreastPlay-I2V-LOW.safetensors"
dl_civitai_latest 1433071 "Bouncy-Forward-Bends-Wan22-I2V-HIGH.safetensors"
dl_civitai_latest 1983608 "POV-Missionary-Insertion-Wan22-I2V-LOW.safetensors"
dl_civitai_latest 1443020 "SingularUnity-Twerk-Wan21-I2V-LOW.safetensors"
dl_civitai_latest 2557154 "Wan22-Piledriver-I2V.safetensors"
dl_civitai_latest 1897340 "Sigma-Face-Expression-Wan22-HIGH.safetensors"

# POV Missionary HIGH-noise — pinned to version 2098396 (confirmed in research)
dl_civitai_version 2098396 "POV-Missionary-Wan22-I2V-HIGH.safetensors"

# -- T2V-focused
dl_civitai_latest 1337157 "Wan-Cowgirl-T2V-HIGH.safetensors"
dl_civitai_latest 1885212 "NSFW-Bundle-SM-Wan22-T2V.safetensors"

# -- Universal (T2V + I2V)
# CubeyAI General NSFW — HIGH (v2073605) + LOW (v2083303) are separate versions.
# The older v2070122 is HIGH-only; v0.08a nightly split into two files.
dl_civitai_version 2073605 "CubeyAI-NSFW-Wan22-HIGH-v08a.safetensors"
dl_civitai_version 2083303 "CubeyAI-NSFW-Wan22-LOW-v08a.safetensors"

# Instagirl v2.5 ships as a 2.4 GB ZIP with workflow + multiple rank variants.
# _maybe_unzip will extract the .safetensors files into $LORA_DIR.
dl_civitai_latest 1822984 "Instagirl-Wan22-v2.5.safetensors"

# Instamodel v1.0 ships as a ZIP with workflow + single LoRA.
dl_civitai_latest 1850818 "Instamodel-1.0-Wan22.safetensors"

# ─── Optional — speed-mode A/B checkpoint ────────────────────────────────────
# Phr00t's AllInOne merged checkpoint — single file with Lightning + rCM + quality
# LoRAs baked in. Goes in diffusion_models, NOT loras. Uncomment to pull.
#
# DIFF_DIR=/runpod-volume/models/diffusion_models
# mkdir -p "$DIFF_DIR"
# dl_hf "https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne/resolve/main/WAN2.2-14B-Rapid-AllInOne-MEGA-v12-Q6_K.gguf" \
#       "$DIFF_DIR/WAN2.2-14B-Rapid-AllInOne-MEGA-v12-Q6_K.gguf"

echo ""
echo "✓ Done."
echo ""
ls -lh "$LORA_DIR" | tail -n +2
