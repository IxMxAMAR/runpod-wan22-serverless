# RunPod Serverless WAN 2.2

Serverless video generation endpoints for WAN 2.2 14B on RunPod.

## Endpoints

- **wan22-t2v** — Text-to-Video (dual-pass HIGH/LOW, 4-step Lightning)
- **wan22-i2v** — Image-to-Video (dual-pass HIGH/LOW, 4-step Lightning)

## Quick Start

### API Call (Template Mode)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "template": "t2v-standard",
      "params": {
        "prompt": "A cat walking through a field of flowers",
        "duration": 5,
        "seed": 42
      }
    }
  }'
```

### API Call (Full Workflow Mode)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "workflow": { ...ComfyUI API-format JSON... }
    }
  }'
```

## Setup

See `docs/superpowers/specs/2026-04-11-runpod-wan22-serverless-design.md` for full architecture and deployment guide.

## Network Volume (LoRAs)

Add new LoRAs by uploading to `/runpod-volume/models/loras/` — no rebuild needed.
Reference by name: `"loras": [{"name": "my-lora", "strength": 1.0}]`
