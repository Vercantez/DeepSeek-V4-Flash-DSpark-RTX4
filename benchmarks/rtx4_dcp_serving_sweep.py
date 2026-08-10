#!/usr/bin/env python3
"""Repeatable serving benchmark for RTX4 DeepSeek-V4 DCP profiles.

Run this once for each already-running DCP1/DCP2/DCP4 server profile.  The
script does not start, stop, or reconfigure the server.  It deliberately keeps
prefill and decode measurements separate and requires final usage data by
default, because one speculative-decoding SSE event may contain several output
tokens.

Example:

    python benchmarks/rtx4_dcp_serving_sweep.py \
      --dcp 2 --profile-label dcp2-eager-correctness \
      --concurrency 1,2,4,8,16,32,48,64 \
      --trials 3 --output-tokens 512 \
      --output benchmarks/results/raw/dcp2-eager.json

Use an isolated endpoint.  Unrelated traffic changes both the vLLM counter
deltas and the available GPU capacity during a trial.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request


SCHEMA_VERSION = 1
PROMETHEUS_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+"
    r"([-+]?(?:[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|Inf|NaN))"
    r"(?:\s+[0-9]+)?$"
)
PROMETHEUS_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)')

COUNTER_METRICS = {
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
}
GAUGE_METRICS = {
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
}
INFO_METRICS = {
    "vllm:cache_config_info",
    "vllm:lora_requests_info",
}
CONTENT_KEYS = ("content", "reasoning_content", "reasoning")
PROMPT_WORDS = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kepler",
    "lunar",
    "meadow",
    "nimbus",
    "onyx",
    "prairie",
    "quartz",
    "river",
    "syntax",
    "tundra",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
)
CORRECTNESS_CASES = (
    ("Reply with exactly: alpha cache canary passed", "alpha cache canary passed"),
    ("Reply with exactly: bravo rank canary passed", "bravo rank canary passed"),
    ("Reply with only the answer to 7 times 8.", "56"),
    ("Reply with only the capital of Japan.", "tokyo"),
    ("Reply with only the answer to 17 times 23.", "391"),
    (
        "Remember this phrase: NEBULA-QUARTZ-8815. Now reply with only the phrase.",
        "nebula-quartz-8815",
    ),
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


def describe(values: list[float], digits: int = 3) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "min": rounded(min(finite), digits),
        "p50": rounded(statistics.median(finite), digits),
        "p95": rounded(percentile(finite, 0.95), digits),
        "max": rounded(max(finite), digits),
    }


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = PROMETHEUS_LABEL.match(raw, position)
        if match is None:
            return {"_unparsed": raw}
        key, escaped = match.groups()
        try:
            labels[key] = json.loads(f'"{escaped}"')
        except json.JSONDecodeError:
            labels[key] = escaped
        position = match.end()
    return labels


def parse_prometheus(text: str) -> dict[str, list[dict[str, Any]]]:
    parsed: dict[str, list[dict[str, Any]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE.match(line)
        if match is None:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        parsed.setdefault(name, []).append(
            {"labels": parse_labels(raw_labels), "value": value}
        )
    return parsed


def aggregate_metrics(
    metrics: dict[str, list[dict[str, Any]]], names: set[str]
) -> dict[str, float]:
    return {
        name: sum(float(row["value"]) for row in metrics.get(name, ()))
        for name in sorted(names)
        if name in metrics
    }


def fetch_text(url: str, headers: dict[str, str], timeout: float = 10.0) -> str:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode(errors="replace")


def fetch_metrics(url: str, headers: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    if not url:
        return {}
    try:
        return parse_prometheus(fetch_text(url, headers))
    except (OSError, urllib.error.URLError, TimeoutError):
        return {}


def metric_delta(
    before: dict[str, list[dict[str, Any]]],
    after: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    before_aggregate = aggregate_metrics(before, COUNTER_METRICS)
    after_aggregate = aggregate_metrics(after, COUNTER_METRICS)
    return {
        name: round(after_aggregate[name] - before_aggregate.get(name, 0.0), 6)
        for name in sorted(after_aggregate)
    }


def default_metrics_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root + "/metrics"


def completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root += "/v1"
    return root + "/chat/completions"


def event_content(event: dict[str, Any]) -> str:
    choices = event.get("choices") or ()
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return "".join(str(delta.get(key) or "") for key in CONTENT_KEYS)


def iter_sse(response: Any):
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode(errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


@dataclass
class RequestResult:
    index: int
    ok: bool
    error: str | None
    http_status: int | None
    started: float
    first_content: float | None
    last_content: float | None
    ended: float
    prompt_tokens: int
    output_tokens: int
    usage_source: str
    content_events: int
    event_gaps_ms: list[float]
    finish_reason: str | None
    text: str

    def public(self, origin: float, capture_chars: int) -> dict[str, Any]:
        ttft_seconds = (
            self.first_content - self.started
            if self.first_content is not None
            else None
        )
        decode_seconds = (
            self.last_content - self.first_content
            if self.first_content is not None and self.last_content is not None
            else None
        )
        decode_tokens = max(0, self.output_tokens - 1)
        effective_itl_ms = (
            1000.0 * decode_seconds / decode_tokens
            if decode_seconds is not None and decode_seconds > 0 and decode_tokens > 0
            else None
        )
        return {
            "index": self.index,
            "ok": self.ok,
            "error": self.error,
            "http_status": self.http_status,
            "started_s": rounded(self.started - origin, 6),
            "ttft_s": rounded(ttft_seconds, 6),
            "prompt_tps_to_first": rounded(
                self.prompt_tokens / ttft_seconds
                if ttft_seconds is not None
                and ttft_seconds > 0
                and self.prompt_tokens > 0
                else None,
                3,
            ),
            "decode_window_s": rounded(decode_seconds, 6),
            "end_to_end_s": rounded(self.ended - self.started, 6),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "decode_tokens": decode_tokens,
            "effective_itl_ms": rounded(effective_itl_ms, 4),
            "output_tps_after_first": rounded(
                decode_tokens / decode_seconds
                if decode_seconds is not None and decode_seconds > 0
                else None,
                3,
            ),
            "usage_source": self.usage_source,
            "content_events": self.content_events,
            "sse_event_gap_ms": describe(self.event_gaps_ms, 3),
            "finish_reason": self.finish_reason,
            "text": self.text[:capture_chars],
        }


def make_prompt(
    index: int,
    concurrency: int,
    trial: int,
    words: int,
    prompt_template: str | None = None,
) -> str:
    nonce = f"dcp-{concurrency}-{trial}-{index}"
    if prompt_template is not None:
        return f"Unique request {nonce}.\n{prompt_template}"
    filler = " ".join(
        PROMPT_WORDS[(index * 7 + trial * 11 + offset) % len(PROMPT_WORDS)]
        for offset in range(words)
    )
    return (
        f"Unique request {nonce}. {filler}\n"
        "Write a long, precise implementation guide for a bounded concurrent "
        "job queue. Cover retries, idempotency, backpressure, observability, and "
        "graceful shutdown. Continue until the response limit."
    )


def do_stream_request(
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    index: int,
    prompt: str,
    max_tokens: int,
    timeout: float,
    start_barrier: threading.Barrier | None,
    ignore_eos: bool,
    require_usage: bool,
    allow_early_stop: bool,
) -> RequestResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    if start_barrier is not None:
        start_barrier.wait(timeout=60)
    started = time.perf_counter()
    first_content = None
    last_content = None
    content_times: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    http_status = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            for payload in iter_sse(response):
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or ()
                if choices and choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                content = event_content(event)
                if content:
                    now = time.perf_counter()
                    first_content = first_content or now
                    last_content = now
                    content_times.append(now)
                    pieces.append(content)
    except Exception as exc:  # noqa: BLE001 - benchmark records request failures
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    completion_value = usage.get("completion_tokens")
    prompt_value = usage.get("prompt_tokens")
    if completion_value is not None:
        output_tokens = int(completion_value)
        usage_source = "usage.completion_tokens"
    else:
        output_tokens = len(content_times)
        usage_source = "content_event_fallback"
    prompt_tokens = int(prompt_value or 0)
    if error is None and first_content is None:
        error = "response contained no output content"
    if error is None and require_usage and completion_value is None:
        error = "final usage.completion_tokens missing"
    if (
        error is None
        and ignore_eos
        and not allow_early_stop
        and output_tokens < max_tokens
    ):
        error = f"early stop: received {output_tokens} of {max_tokens} requested tokens"
    return RequestResult(
        index=index,
        ok=error is None and http_status == 200,
        error=error,
        http_status=http_status,
        started=started,
        first_content=first_content,
        last_content=last_content,
        ended=ended,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        usage_source=usage_source,
        content_events=len(content_times),
        event_gaps_ms=[
            1000.0 * (right - left)
            for left, right in zip(content_times, content_times[1:])
        ],
        finish_reason=finish_reason,
        text="".join(pieces),
    )


class TelemetrySampler:
    def __init__(
        self,
        *,
        metrics_url: str,
        headers: dict[str, str],
        interval: float,
        sample_gpus: bool,
    ) -> None:
        self.metrics_url = metrics_url
        self.headers = headers
        self.interval = interval
        self.sample_gpus = sample_gpus and shutil.which("nvidia-smi") is not None
        self.metrics_samples: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._origin = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._origin = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(12.0, self.interval * 2))

    def _sample_metrics(self, elapsed: float) -> None:
        metrics = fetch_metrics(self.metrics_url, self.headers)
        gauges = aggregate_metrics(metrics, GAUGE_METRICS)
        if gauges:
            self.metrics_samples.append({"elapsed_s": elapsed, **gauges})

    def _sample_gpus(self, elapsed: float) -> None:
        fields = (
            "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
            "power.draw,clocks.sm,temperature.gpu"
        )
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={fields}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in completed.stdout.splitlines():
                columns = [column.strip() for column in line.split(",")]
                if len(columns) != 8:
                    continue
                values: list[float | None] = []
                for raw_value in columns[1:]:
                    try:
                        values.append(float(raw_value))
                    except ValueError:
                        values.append(None)
                self.gpu_samples.append(
                    {
                        "elapsed_s": elapsed,
                        "gpu": int(columns[0]),
                        "gpu_util_pct": values[0],
                        "memory_util_pct": values[1],
                        "memory_used_mib": values[2],
                        "memory_total_mib": values[3],
                        "power_w": values[4],
                        "sm_clock_mhz": values[5],
                        "temperature_c": values[6],
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if not self.errors:
                self.errors.append(f"nvidia-smi sampling failed: {exc}")
            self.sample_gpus = False

    def _run(self) -> None:
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._origin
            if self.metrics_url:
                self._sample_metrics(elapsed)
            if self.sample_gpus:
                self._sample_gpus(elapsed)
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        gauge_summary: dict[str, Any] = {}
        for name in sorted(GAUGE_METRICS):
            values = [
                float(sample[name])
                for sample in self.metrics_samples
                if name in sample
            ]
            if values:
                gauge_summary[name] = describe(values, 4)
        gpu_summary: dict[str, Any] = {}
        for gpu in sorted({int(sample["gpu"]) for sample in self.gpu_samples}):
            rows = [sample for sample in self.gpu_samples if sample["gpu"] == gpu]
            gpu_summary[str(gpu)] = {
                field: describe(
                    [float(row[field]) for row in rows if row[field] is not None],
                    2,
                )
                for field in (
                    "gpu_util_pct",
                    "memory_util_pct",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_w",
                    "sm_clock_mhz",
                    "temperature_c",
                )
            }
        return {
            "metric_gauges": gauge_summary,
            "gpu": gpu_summary,
            "errors": self.errors,
        }


def run_batch(
    *,
    args: argparse.Namespace,
    headers: dict[str, str],
    concurrency: int,
    trial: int,
    max_tokens: int,
    collect_telemetry: bool,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)
    metrics_before = fetch_metrics(args.metrics_url, headers)
    sampler = TelemetrySampler(
        metrics_url=args.metrics_url if collect_telemetry else "",
        headers=headers,
        interval=args.sample_interval,
        sample_gpus=collect_telemetry and not args.no_gpu_sampling,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                do_stream_request,
                url=args.completions_url,
                headers=headers,
                model=args.model,
                index=index,
                prompt=make_prompt(
                    index,
                    concurrency,
                    trial,
                    args.prompt_words,
                    args.prompt_template,
                ),
                max_tokens=max_tokens,
                timeout=args.timeout,
                start_barrier=barrier,
                ignore_eos=True,
                require_usage=not args.allow_missing_usage,
                allow_early_stop=args.allow_early_stop,
            )
            for index in range(concurrency)
        ]
        sampler.start()
        batch_started = time.perf_counter()
        barrier.wait(timeout=60)
        rows = [future.result() for future in futures]
        batch_ended = time.perf_counter()
        sampler.stop()
    metrics_after = fetch_metrics(args.metrics_url, headers)
    successful = [row for row in rows if row.ok]
    first_times = [
        row.first_content for row in successful if row.first_content is not None
    ]
    last_times = [row.last_content for row in successful if row.last_content is not None]
    start_times = [row.started for row in successful]
    prompt_tokens = sum(row.prompt_tokens for row in successful)
    prefill_window = (
        max(first_times) - min(start_times) if first_times and start_times else None
    )
    decode_tokens = sum(max(0, row.output_tokens - 1) for row in successful)
    decode_window = (
        max(last_times) - min(first_times) if first_times and last_times else None
    )
    wall = batch_ended - batch_started
    public_rows = [row.public(batch_started, args.capture_chars) for row in rows]
    output_tokens = sum(row.output_tokens for row in successful)
    server_delta = metric_delta(metrics_before, metrics_after)
    generation_delta = server_delta.get("vllm:generation_tokens_total")
    accepted = server_delta.get("vllm:spec_decode_num_accepted_tokens_total")
    drafted = server_delta.get("vllm:spec_decode_num_draft_tokens_total")
    result = {
        "trial": trial,
        "concurrency": concurrency,
        "success": f"{len(successful)}/{len(rows)}",
        "output_tokens": output_tokens,
        "decode_tokens_after_first": decode_tokens,
        "wall_s": rounded(wall, 6),
        "global_prefill_to_first_window_s": rounded(prefill_window, 6),
        "global_decode_window_s": rounded(decode_window, 6),
        "aggregate_prompt_tps_to_first": rounded(
            prompt_tokens / prefill_window
            if prefill_window is not None
            and prefill_window > 0
            and prompt_tokens > 0
            else None,
            3,
        ),
        "aggregate_output_tps_end_to_end": rounded(output_tokens / wall, 3),
        "aggregate_output_tps_after_first": rounded(
            decode_tokens / decode_window
            if decode_tokens > 0 and decode_window is not None and decode_window > 0
            else None,
            3,
        ),
        "server_generation_tps_end_to_end": rounded(
            generation_delta / wall
            if generation_delta is not None and wall > 0
            else None,
            3,
        ),
        "ttft_s": describe(
            [
                row.first_content - row.started
                for row in successful
                if row.first_content is not None
            ],
            4,
        ),
        "per_stream_prompt_tps_to_first": describe(
            [
                float(row["prompt_tps_to_first"])
                for row in public_rows
                if row["ok"] and row["prompt_tps_to_first"] is not None
            ],
            3,
        ),
        "effective_itl_ms": describe(
            [
                float(row["effective_itl_ms"])
                for row in public_rows
                if row["ok"] and row["effective_itl_ms"] is not None
            ],
            4,
        ),
        "per_stream_output_tps_after_first": describe(
            [
                float(row["output_tps_after_first"])
                for row in public_rows
                if row["ok"] and row["output_tps_after_first"] is not None
            ],
            3,
        ),
        "sse_event_gap_ms": describe(
            [gap for row in successful for gap in row.event_gaps_ms], 3
        ),
        "prompt_tokens": describe(
            [float(row.prompt_tokens) for row in successful], 1
        ),
        "server_metric_delta": server_delta,
        "spec_decode_acceptance": rounded(
            accepted / drafted if accepted is not None and drafted else None,
            5,
        ),
        "cache_config_info": metrics_after.get("vllm:cache_config_info", []),
        "telemetry": sampler.summary(),
        "requests": public_rows,
    }
    return result


def canonical_answer(text: str) -> str:
    normalized = text.strip().lower().replace("`", "").replace('"', "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.rstrip(". !\n")


def run_correctness(
    *,
    args: argparse.Namespace,
    headers: dict[str, str],
    concurrency: int,
    wave: int,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        expected_values = []
        for index in range(concurrency):
            prompt, expected = CORRECTNESS_CASES[(wave + index) % len(CORRECTNESS_CASES)]
            filler_count = (0, 16, 64, 256)[index % 4]
            if filler_count:
                prompt = (
                    ("context " * filler_count)
                    + "\nIgnore the context above and follow this instruction:\n"
                    + prompt
                )
            expected_values.append(expected)
            futures.append(
                pool.submit(
                    do_stream_request,
                    url=args.completions_url,
                    headers=headers,
                    model=args.model,
                    index=index,
                    prompt=prompt,
                    max_tokens=32,
                    timeout=args.timeout,
                    start_barrier=barrier,
                    ignore_eos=False,
                    require_usage=False,
                    allow_early_stop=True,
                )
            )
        barrier.wait(timeout=60)
        rows = [future.result() for future in futures]
    checks = []
    for row, expected in zip(rows, expected_values):
        actual = canonical_answer(row.text)
        expected_normalized = canonical_answer(expected)
        exact = row.ok and actual == expected_normalized
        checks.append(
            {
                "index": row.index,
                "ok": row.ok,
                "exact": exact,
                "expected": expected_normalized,
                "actual": actual[: args.capture_chars],
                "error": row.error,
            }
        )
    return {
        "wave": wave,
        "concurrency": concurrency,
        "exact": f"{sum(check['exact'] for check in checks)}/{len(checks)}",
        "ok": all(check["exact"] for check in checks),
        "checks": checks,
    }


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "aggregate_prompt_tps_to_first",
        "aggregate_output_tps_end_to_end",
        "aggregate_output_tps_after_first",
        "server_generation_tps_end_to_end",
    )
    summary = {
        field: describe(
            [float(trial[field]) for trial in trials if trial[field] is not None], 3
        )
        for field in fields
    }
    summary["ttft_p50_s"] = describe(
        [
            float(trial["ttft_s"]["p50"])
            for trial in trials
            if trial["ttft_s"]["p50"] is not None
        ],
        4,
    )
    summary["effective_itl_p50_ms"] = describe(
        [
            float(trial["effective_itl_ms"]["p50"])
            for trial in trials
            if trial["effective_itl_ms"]["p50"] is not None
        ],
        4,
    )
    summary["all_requests_succeeded"] = all(
        trial["success"].split("/")[0] == trial["success"].split("/")[1]
        for trial in trials
    )
    return summary


def parse_concurrency(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        value = int(item.strip())
        if value <= 0:
            raise argparse.ArgumentTypeError("concurrency values must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one concurrency is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dcp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--backend", default="unspecified")
    parser.add_argument("--tensor-parallel-size", type=int, default=-1)
    parser.add_argument(
        "--expert-parallel", choices=("on", "off", "unspecified"), default="unspecified"
    )
    parser.add_argument("--dspark-tokens", type=int, default=-1)
    parser.add_argument("--cudagraph-mode", default="unspecified")
    parser.add_argument("--async-scheduling", default="unspecified")
    parser.add_argument(
        "--concurrency", type=parse_concurrency, default=parse_concurrency("1,2,4,8,16")
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--prompt-words", type=int, default=64)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--correctness-waves", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--allow-missing-usage", action="store_true")
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--capture-chars", type=int, default=512)
    parser.add_argument("--image-tag", default="unspecified")
    parser.add_argument("--runtime-commit", default="unspecified")
    parser.add_argument("--kv-dtype", default="unspecified")
    parser.add_argument("--dcp-interleave", type=int, default=-1)
    parser.add_argument("--max-num-seqs", type=int, default=-1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=-1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=-1.0)
    parser.add_argument("--output", type=Path)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "trials",
        "warmups",
        "warmup_tokens",
        "output_tokens",
        "prompt_words",
        "correctness_waves",
        "capture_chars",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.trials < 1 or args.output_tokens < 1:
        parser.error("--trials and --output-tokens must be at least 1")
    if args.warmups and args.warmup_tokens < 1:
        parser.error("--warmup-tokens must be at least 1 when warmups are enabled")
    if args.timeout <= 0 or args.sample_interval <= 0:
        parser.error("--timeout and --sample-interval must be positive")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        parser.error(f"--prompt-file does not exist: {args.prompt_file}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.completions_url = completions_url(args.base_url)
    args.metrics_url = args.metrics_url or default_metrics_url(args.base_url)
    args.prompt_template = (
        args.prompt_file.read_text() if args.prompt_file is not None else None
    )
    api_key = os.environ.get(args.api_key_env, "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": {
            "label": args.profile_label,
            "dcp": args.dcp,
            "backend": args.backend,
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel": args.expert_parallel,
            "dspark_tokens": args.dspark_tokens,
            "cudagraph_mode": args.cudagraph_mode,
            "async_scheduling": args.async_scheduling,
            "model": args.model,
            "base_url": args.base_url,
            "image_tag": args.image_tag,
            "runtime_commit": args.runtime_commit,
            "kv_dtype": args.kv_dtype,
            "dcp_interleave": args.dcp_interleave,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "settings": {
            "concurrency": args.concurrency,
            "trials": args.trials,
            "warmups": args.warmups,
            "warmup_tokens": args.warmup_tokens,
            "output_tokens": args.output_tokens,
            "measurement_mode": (
                "prefill_to_first_token" if args.output_tokens == 1 else "decode"
            ),
            "prompt_words": args.prompt_words,
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            "prompt_sha256": (
                hashlib.sha256(args.prompt_template.encode()).hexdigest()
                if args.prompt_template is not None
                else None
            ),
            "correctness_waves": args.correctness_waves,
            "sample_interval": args.sample_interval,
            "usage_required": not args.allow_missing_usage,
            "early_stop_allowed": args.allow_early_stop,
        },
        "metric_info": {},
        "levels": [],
    }
    initial_metrics = fetch_metrics(args.metrics_url, headers)
    result["metric_info"] = {
        name: initial_metrics[name]
        for name in sorted(INFO_METRICS)
        if name in initial_metrics
    }

    failed = False
    for concurrency in args.concurrency:
        level: dict[str, Any] = {
            "concurrency": concurrency,
            "correctness": [],
            "trials": [],
        }
        for wave in range(args.correctness_waves):
            correctness = run_correctness(
                args=args,
                headers=headers,
                concurrency=concurrency,
                wave=wave,
            )
            level["correctness"].append(correctness)
            failed = failed or not correctness["ok"]
            print(json.dumps({"correctness": correctness}), flush=True)

        for warmup in range(args.warmups):
            warmup_result = run_batch(
                args=args,
                headers=headers,
                concurrency=concurrency,
                trial=-(warmup + 1),
                max_tokens=args.warmup_tokens,
                collect_telemetry=False,
            )
            warmup_ok = warmup_result["success"] == f"{concurrency}/{concurrency}"
            print(
                json.dumps(
                    {
                        "warmup": warmup + 1,
                        "concurrency": concurrency,
                        "success": warmup_result["success"],
                    }
                ),
                flush=True,
            )
            if not warmup_ok:
                failed = True

        for trial in range(args.trials):
            trial_result = run_batch(
                args=args,
                headers=headers,
                concurrency=concurrency,
                trial=trial,
                max_tokens=args.output_tokens,
                collect_telemetry=True,
            )
            level["trials"].append(trial_result)
            failed = failed or trial_result["success"] != f"{concurrency}/{concurrency}"
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "concurrency": concurrency,
                        "success": trial_result["success"],
                        "ttft_p50_s": trial_result["ttft_s"]["p50"],
                        "aggregate_prompt_tps_to_first": trial_result[
                            "aggregate_prompt_tps_to_first"
                        ],
                        "effective_itl_p50_ms": trial_result["effective_itl_ms"]["p50"],
                        "aggregate_output_tps_after_first": trial_result[
                            "aggregate_output_tps_after_first"
                        ],
                    }
                ),
                flush=True,
            )
        level["summary"] = summarize_trials(level["trials"])
        result["levels"].append(level)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["ok"] = not failed
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps({"result_file": str(args.output), "ok": result["ok"]}))
    else:
        print(encoded, end="")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
