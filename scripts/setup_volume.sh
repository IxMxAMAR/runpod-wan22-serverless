#!/bin/bash
# One-time setup script for RunPod network volume
# Run this on a temporary RunPod Pod with the volume attached at /runpod-volume/
#
# Usage: bash setup_volume.sh

set -euo pipefail

VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
LORA_DIR="$VOLUME_PATH/models/loras"
LORA_WAN_DIR="$LORA_DIR/wan"
RIFE_DIR="$VOLUME_PATH/models/rife"

echo "=== WAN 2.2 Network Volume Setup ==="
echo "Volume path: $VOLUME_PATH"

# Create directory structure
echo "Creating directories..."
mkdir -p "$LORA_DIR"
mkdir -p "$LORA_WAN_DIR"
mkdir -p "$RIFE_DIR"

# Download Lightning LoRAs (Seko V2.0)
echo ""
echo "=== Downloading T2V Lightning LoRAs ==="

if [ ! -f "$LORA_DIR/Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors" ]; then
    echo "Downloading Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors" \
        "https://huggingface.co/Seko0938/Wan2.2-T2V-4steps-lora-rank64/resolve/main/Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors"
else
    echo "Wan2.2-T2V-4steps-HIGH already exists, skipping"
fi

if [ ! -f "$LORA_DIR/Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0.safetensors" ]; then
    echo "Downloading Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0.safetensors" \
        "https://huggingface.co/Seko0938/Wan2.2-T2V-4steps-lora-rank64/resolve/main/Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0.safetensors"
else
    echo "Wan2.2-T2V-4steps-LOW already exists, skipping"
fi

echo ""
echo "=== Downloading I2V Lightning LoRAs ==="

if [ ! -f "$LORA_DIR/Wan2.2-I2V-HIGH-4steps-lora-rank64-Seko-V1.safetensors" ]; then
    echo "Downloading Wan2.2-I2V-HIGH-4steps..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-I2V-HIGH-4steps-lora-rank64-Seko-V1.safetensors" \
        "https://huggingface.co/Seko0938/Wan2.2-I2V-4steps-lora-rank64/resolve/main/Wan2.2-I2V-HIGH-4steps-lora-rank64-Seko-V1.safetensors"
else
    echo "Wan2.2-I2V-HIGH already exists, skipping"
fi

if [ ! -f "$LORA_DIR/Wan2.2-I2V-LOW-4steps-lora-rank64-Seko-V1.safetensors" ]; then
    echo "Downloading Wan2.2-I2V-LOW-4steps..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-I2V-LOW-4steps-lora-rank64-Seko-V1.safetensors" \
        "https://huggingface.co/Seko0938/Wan2.2-I2V-4steps-lora-rank64/resolve/main/Wan2.2-I2V-LOW-4steps-lora-rank64-Seko-V1.safetensors"
else
    echo "Wan2.2-I2V-LOW already exists, skipping"
fi

echo ""
echo "=== Downloading Quality LoRAs ==="

if [ ! -f "$LORA_WAN_DIR/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors" ]; then
    echo "Downloading SECRET_SAUCE..."
    echo "NOTE: Verify the HuggingFace URL for this model and update this script."
    echo "      This model may require manual download from CivitAI or HuggingFace."
    # wget -q --show-progress -O "$LORA_WAN_DIR/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors" \
    #     "REPLACE_WITH_ACTUAL_URL"
else
    echo "SECRET_SAUCE already exists, skipping"
fi

if [ ! -f "$LORA_WAN_DIR/Wan2.1_T2V_14B_FusionX_LoRA.safetensors" ]; then
    echo "Downloading FusionX T2V..."
    echo "NOTE: Verify the HuggingFace URL for this model and update this script."
    # wget -q --show-progress -O "$LORA_WAN_DIR/Wan2.1_T2V_14B_FusionX_LoRA.safetensors" \
    #     "REPLACE_WITH_ACTUAL_URL"
else
    echo "FusionX T2V already exists, skipping"
fi

if [ ! -f "$LORA_WAN_DIR/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" ]; then
    echo "Downloading FusionX I2V..."
    echo "NOTE: Verify the HuggingFace URL for this model and update this script."
    # wget -q --show-progress -O "$LORA_WAN_DIR/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" \
    #     "REPLACE_WITH_ACTUAL_URL"
else
    echo "FusionX I2V already exists, skipping"
fi

echo ""
echo "=== Downloading RIFE Model ==="

if [ ! -f "$RIFE_DIR/rife49.pth" ]; then
    echo "Downloading rife49.pth..."
    wget -q --show-progress -O "$RIFE_DIR/rife49.pth" \
        "https://huggingface.co/Fannovel16/RIFE/resolve/main/rife49.pth"
else
    echo "rife49.pth already exists, skipping"
fi

echo ""
echo "=== Volume Setup Complete ==="
echo ""
echo "Contents:"
find "$VOLUME_PATH/models" -type f -exec ls -lh {} \;
echo ""
echo "Total size:"
du -sh "$VOLUME_PATH/models/"
