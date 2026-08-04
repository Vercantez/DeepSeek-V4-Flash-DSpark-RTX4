#!/usr/bin/env python3
"""Bounded concurrent SSE load test for an authenticated OpenAI-compatible gateway."""

import concurrent.futures
import json
import math
import os
import statistics
import time
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "dynamic/deepseek-v4-auto")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "384"))
CONCURRENCY = [int(value) for value in os.environ.get("CONCURRENCY", "4,8,16,32").split(",")]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def request_one(index: int) -> dict:
    prompt = (
        "Write a detailed implementation guide for a concurrent job queue with retries, "
        "idempotency, backpressure, metrics, and graceful shutdown. Use complete paragraphs "
        f"and continue until the response limit. Request salt: {index}."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "qaml-gateway-load-test/1.0",
    }
    request = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    first_token = None
    usage = {}
    model = None
    finish_reason = None
    events = 0
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                model = event.get("model") or model
                usage = event.get("usage") or usage
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first_token = first_token or time.perf_counter()
                    events += 1
                finish_reason = choice.get("finish_reason") or finish_reason
        ended = time.perf_counter()
        output_tokens = int(usage.get("completion_tokens") or events)
        return {
            "ok": True,
            "model": model,
            "output_tokens": output_tokens,
            "ttfb": (first_token or ended) - started,
            "duration": ended - started,
            "decode_tps": output_tokens / max(ended - (first_token or started), 0.001),
            "finish_reason": finish_reason,
        }
    except Exception as error:
        return {"ok": False, "error": str(error), "duration": time.perf_counter() - started}


def run_phase(concurrency: int) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        rows = list(executor.map(request_one, range(concurrency)))
    wall = time.perf_counter() - started
    successful = [row for row in rows if row["ok"]]
    output_tokens = sum(row["output_tokens"] for row in successful)
    result = {
        "concurrency": concurrency,
        "success": f"{len(successful)}/{len(rows)}",
        "models": sorted({row["model"] for row in successful}),
        "output_tokens": output_tokens,
        "wall_seconds": round(wall, 3),
        "aggregate_output_tps": round(output_tokens / wall, 1),
        "errors": [row["error"] for row in rows if not row["ok"]],
    }
    if successful:
        result.update(
            {
                "ttfb_p50": round(statistics.median(row["ttfb"] for row in successful), 3),
                "ttfb_p95": round(percentile([row["ttfb"] for row in successful], 0.95), 3),
                "duration_p50": round(statistics.median(row["duration"] for row in successful), 3),
                "duration_p95": round(percentile([row["duration"] for row in successful], 0.95), 3),
                "per_stream_tps_p50": round(statistics.median(row["decode_tps"] for row in successful), 1),
            }
        )
    return result


def main() -> None:
    if not API_KEY:
        raise SystemExit("API_KEY is required")
    print(json.dumps({"warmup": request_one(-1)}, indent=2))
    for concurrency in CONCURRENCY:
        print(json.dumps(run_phase(concurrency), indent=2), flush=True)


if __name__ == "__main__":
    main()
