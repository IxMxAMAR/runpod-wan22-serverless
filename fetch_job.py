"""Fetch a RunPod job result by job ID, save the video to ./output/.

Usage:
    python fetch_job.py <job_id>                    # uses i2v endpoint
    python fetch_job.py <job_id> t2v                # use t2v endpoint instead

Reads credentials from gui_config.json in the same folder.
"""
import base64
import json
import sys
import time
from pathlib import Path

import requests

if len(sys.argv) < 2:
    print("Usage: python fetch_job.py <job_id> [t2v|i2v]")
    sys.exit(1)

job_id = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "i2v"

here = Path(__file__).parent
cfg = json.load(open(here / "gui_config.json"))
endpoint = cfg[f"{which}_endpoint"]
api_key = cfg["api_key"]

url = f"https://api.runpod.ai/v2/{endpoint}/status/{job_id}"
print(f"Fetching {which.upper()} job {job_id}...")
r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
r.raise_for_status()
data = r.json()

print(f"status: {data.get('status')}")
print(f"executionTime: {data.get('executionTime', 0) / 1000:.1f}s")
print(f"delayTime:     {data.get('delayTime', 0) / 1000:.1f}s")

output = data.get("output") or {}
videos = output.get("videos", [])
err = data.get("error") or output.get("error")
if err:
    print("ERROR from server:", err)

if not videos:
    print("No videos in response. Job may still be running or failed.")
    sys.exit(0)

out_dir = here / "output"
out_dir.mkdir(exist_ok=True)
for i, v in enumerate(videos):
    raw = base64.b64decode(v["data"])
    name = v.get("filename") or f"recovered_{int(time.time())}_{i}.mp4"
    name = name.replace("/", "_")
    path = out_dir / name
    path.write_bytes(raw)
    print(f"saved: {path}  ({len(raw) / 1048576:.2f} MB)")
