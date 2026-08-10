#!/usr/bin/env python3
"""Stress KV-cache recycling with deterministic concurrent requests and aborts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


CASES = (
    ("Reply with only the answer: 2 + 2", "4"),
    ("Reply with only the capital of France.", "paris"),
    ("Reply with exactly: alpha canary passed", "alpha canary passed"),
    ("Reply with only the answer: 7 * 8", "56"),
)


def request(
    base_url: str,
    model: str,
    wave: int,
    index: int,
    max_tokens: int,
    abort: bool,
) -> dict[str, object]:
    question, expected = CASES[(wave + index) % len(CASES)]
    # Deterministically vary prefill lengths while keeping the answer check stable.
    filler_words = (0, 16, 64, 256)[index % 4]
    prompt = (
        (("context " * filler_words) + "\nIgnore the context above.\n")
        if filler_words
        else ""
    ) + question
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    pieces: list[str] = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            content = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                pieces.append(content)
                if abort:
                    # Closing the response after first content exercises server abort
                    # cleanup and immediate reuse of the released cache blocks.
                    break
    elapsed = time.perf_counter() - started
    text = "".join(pieces).strip().lower()
    return {
        "abort": abort,
        "ok": abort or expected in text,
        "elapsed": elapsed,
        "expected": expected,
        "text": text[:160],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--abort-every", type=int, default=4)
    args = parser.parse_args()

    failures: list[dict[str, object]] = []
    survivor_latencies: list[float] = []
    total_aborts = 0
    for wave in range(args.waves):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = []
            for index in range(args.concurrency):
                abort = args.abort_every > 0 and index % args.abort_every == 0
                futures.append(
                    pool.submit(
                        request,
                        args.base_url,
                        args.model,
                        wave,
                        index,
                        args.max_tokens,
                        abort,
                    )
                )
            wave_results = []
            for index, future in enumerate(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - soak must capture all errors
                    result = {
                        "abort": False,
                        "ok": False,
                        "elapsed": 0.0,
                        "expected": "request success",
                        "text": f"{type(exc).__name__}: {exc}",
                    }
                wave_results.append(result)
                if result["abort"]:
                    total_aborts += 1
                else:
                    survivor_latencies.append(float(result["elapsed"]))
                if not result["ok"]:
                    failures.append({"wave": wave, "index": index, **result})
        survivors = sum(not bool(r["abort"]) for r in wave_results)
        exact = sum(bool(r["ok"]) and not bool(r["abort"]) for r in wave_results)
        print(
            json.dumps(
                {
                    "wave": wave + 1,
                    "survivors_exact": exact,
                    "survivors": survivors,
                    "aborts": len(wave_results) - survivors,
                    "failures_total": len(failures),
                }
            ),
            flush=True,
        )

    summary = {
        "ok": not failures,
        "waves": args.waves,
        "concurrency": args.concurrency,
        "survivors": len(survivor_latencies),
        "aborts": total_aborts,
        "p50_survivor_s": round(statistics.median(survivor_latencies), 3),
        "max_survivor_s": round(max(survivor_latencies), 3),
        "failures": failures[:20],
    }
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
