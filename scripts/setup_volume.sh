#!/bin/bash
# One-time setup script for RunPod network volume
# Run this on a temporary RunPod Pod with the volume attached
#
# Usage:
#   export RUNPOD_VOLUME_PATH=/workspace  # or wherever your volume is mounted
#   bash setup_volume.sh

set -euo pipefail

VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/workspace}"
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

if [ ! -s "$LORA_DIR/Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors" ]; then
    echo "Downloading Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-T2V-4steps-HIGH-rank64-Seko-V2.0.safetensors" \
        "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors"
else
    echo "Wan2.2-T2V-4steps-HIGH already exists, skipping"
fi

if [ ! -s "$LORA_DIR/Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0.safetensors" ]; then
    echo "Downloading Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-T2V-4steps-LOW-rank64-Seko-V2.0.safetensors" \
        "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors"
else
    echo "Wan2.2-T2V-4steps-LOW already exists, skipping"
fi

echo ""
echo "=== Downloading I2V Lightning LoRAs ==="

if [ ! -s "$LORA_DIR/Wan2.2-I2V-HIGH-4steps-lora-rank64-Seko-V1.safetensors" ]; then
    echo "Downloading Wan2.2-I2V-HIGH-4steps..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-I2V-HIGH-4steps-lora-rank64-Seko-V1.safetensors" \
        "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
else
    echo "Wan2.2-I2V-HIGH already exists, skipping"
fi

if [ ! -s "$LORA_DIR/Wan2.2-I2V-LOW-4steps-lora-rank64-Seko-V1.safetensors" ]; then
    echo "Downloading Wan2.2-I2V-LOW-4steps..."
    wget -q --show-progress -O "$LORA_DIR/Wan2.2-I2V-LOW-4steps-lora-rank64-Seko-V1.safetensors" \
        "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"
else
    echo "Wan2.2-I2V-LOW already exists, skipping"
fi

echo ""
echo "=== Downloading Quality LoRAs ==="

if [ ! -s "$LORA_WAN_DIR/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors" ]; then
    echo "Downloading SECRET_SAUCE..."
    wget -q --show-progress -O "$LORA_WAN_DIR/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors" \
        "https://huggingface.co/comfyuistudio/wan2.1_loras/resolve/main/SECRET_SAUCE_WAN2.1_14B_fp8.safetensors"
else
    echo "SECRET_SAUCE already exists, skipping"
fi

if [ ! -s "$LORA_WAN_DIR/Wan2.1_T2V_14B_FusionX_LoRA.safetensors" ]; then
    echo "Downloading FusionX T2V..."
    wget -q --show-progress -O "$LORA_WAN_DIR/Wan2.1_T2V_14B_FusionX_LoRA.safetensors" \
        "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors"
else
    echo "FusionX T2V already exists, skipping"
fi

if [ ! -s "$LORA_WAN_DIR/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" ]; then
    echo "Downloading FusionX I2V..."
    wget -q --show-progress -O "$LORA_WAN_DIR/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" \
        "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
else
    echo "FusionX I2V already exists, skipping"
fi

echo ""
echo "=== Downloading RIFE Model ==="

if [ ! -s "$RIFE_DIR/rife49.pth" ]; then
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
