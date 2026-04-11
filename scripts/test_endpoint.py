"""Smoke test for deployed RunPod serverless endpoints.

Usage:
    python scripts/test_endpoint.py --endpoint-id YOUR_ID --api-key YOUR_KEY --type t2v
    python scripts/test_endpoint.py --endpoint-id YOUR_ID --api-key YOUR_KEY --type i2v --image test.png
"""
import argparse
import base64
import json
import sys
import time

import requests


def test_t2v(endpoint_url: str, headers: dict) -> dict:
    payload = {
        "input": {
            "template": "t2v-standard",
            "params": {
                "prompt": "A golden retriever running through a sunlit meadow",
                "duration": 3,
                "seed": 42,
            },
        }
    }
    return send_request(endpoint_url, headers, payload)


def test_i2v(endpoint_url: str, headers: dict, image_path: str) -> dict:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "input": {
            "template": "i2v-standard",
            "params": {
                "prompt": "The scene slowly comes to life with gentle motion",
                "input_image": image_b64,
                "duration": 3,
                "seed": 42,
            },
        }
    }
    return send_request(endpoint_url, headers, payload)


def send_request(endpoint_url: str, headers: dict, payload: dict) -> dict:
    print(f"Sending request to {endpoint_url}/run ...")
    resp = requests.post(f"{endpoint_url}/run", headers=headers, json=payload)
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    print(f"Job queued: {job_id}")

    while True:
        status_resp = requests.get(
            f"{endpoint_url}/status/{job_id}", headers=headers
        )
        status_resp.raise_for_status()
        status = status_resp.json()

        state = status.get("status")
        print(f"  Status: {state}")

        if state == "COMPLETED":
            return status
        elif state in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"Job {state}: {json.dumps(status.get('output', {}), indent=2)}")
            sys.exit(1)

        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Smoke test RunPod endpoints")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--type", choices=["t2v", "i2v"], required=True)
    parser.add_argument("--image", help="Path to input image (required for i2v)")
    args = parser.parse_args()

    endpoint_url = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    if args.type == "t2v":
        result = test_t2v(endpoint_url, headers)
    else:
        if not args.image:
            print("ERROR: --image required for i2v test")
            sys.exit(1)
        result = test_i2v(endpoint_url, headers, args.image)

    output = result.get("output", {})
    videos = output.get("videos", [])
    thumbnails = output.get("thumbnails", [])
    metadata = output.get("metadata", {})

    print(f"\nSUCCESS!")
    print(f"  Videos: {len(videos)}")
    print(f"  Thumbnails: {len(thumbnails)}")
    print(f"  Generation time: {metadata.get('generation_time_seconds', 'N/A')}s")
    print(f"  Seed: {metadata.get('seed', 'N/A')}")

    if videos:
        video_data = base64.b64decode(videos[0]["data"])
        out_file = f"test_output_{args.type}.mp4"
        with open(out_file, "wb") as f:
            f.write(video_data)
        print(f"  Saved video to: {out_file}")

    delay = result.get("delayTime", 0)
    exec_time = result.get("executionTime", 0)
    print(f"\n  RunPod delay: {delay}ms")
    print(f"  RunPod execution: {exec_time}ms")


if __name__ == "__main__":
    main()
