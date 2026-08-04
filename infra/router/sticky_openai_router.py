#!/usr/bin/env python3
"""Small sticky reverse proxy for vLLM's OpenAI and Anthropic APIs.

The router keeps requests for the same tenant/session on the same GPU host so
prefix/KV caches stay useful across multi-turn traffic. It can use static
ROUTER_BACKENDS or discover private IPs from an EC2 Auto Scaling Group.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
VLLM_PRIORITY_HEADER = "x-chiridion-vllm-priority"


class BackendState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backends: list[str] = []
        self._healthy: set[str] = set()
        self._rr = 0

    def replace(self, backends: Iterable[str], healthy: Iterable[str]) -> None:
        ordered = sorted(set(backends))
        healthy_set = set(healthy)
        with self._lock:
            self._backends = ordered
            self._healthy = healthy_set

    def snapshot(self) -> tuple[list[str], list[str]]:
        with self._lock:
            return list(self._backends), sorted(self._healthy)

    def choose(self, key: str | None) -> str | None:
        with self._lock:
            candidates = [b for b in self._backends if b in self._healthy]
            if not candidates:
                return None
            if key:
                return max(
                    candidates,
                    key=lambda b: hashlib.sha256(f"{key}\0{b}".encode()).digest(),
                )
            backend = candidates[self._rr % len(candidates)]
            self._rr += 1
            return backend

    def routes_to(self, key: str, backend: str) -> bool:
        with self._lock:
            candidates = [b for b in self._backends if b in self._healthy]
            if backend not in candidates:
                return False
            selected = max(
                candidates,
                key=lambda b: hashlib.sha256(f"{key}\0{b}".encode()).digest(),
            )
            return selected == backend


STATE = BackendState()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BACKEND_PORT = env_int("BACKEND_PORT", 8000)
DISCOVERY_INTERVAL = env_int("DISCOVERY_INTERVAL", 15)
HEALTH_TIMEOUT = env_int("HEALTH_TIMEOUT", 3)
REQUEST_TIMEOUT = env_int("REQUEST_TIMEOUT", 900)
MODEL_ID = os.environ.get("MODEL_ID", "deepseek-v4-flash-dspark")
MODEL_MAX_CONTEXT_TOKENS = env_int("MODEL_MAX_CONTEXT_TOKENS", 262_144)
MODEL_WORKING_CONTEXT_TOKENS = env_int("MODEL_WORKING_CONTEXT_TOKENS", 220_000)
MODEL_MAX_OUTPUT_TOKENS = env_int("MODEL_MAX_OUTPUT_TOKENS", 262_144)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
AWS_ASG_NAME = os.environ.get("AWS_ASG_NAME", "")
# Comma-separated `region:auto-scaling-group` targets. This lets one router
# keep rendezvous-hash affinity while discovering workers in multiple regions.
AWS_ASG_TARGETS = os.environ.get("AWS_ASG_TARGETS", "")
ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "")
DEFAULT_CHAT_TEMPLATE_THINKING = env_bool("DEFAULT_CHAT_TEMPLATE_THINKING", False)
DEFAULT_REQUEST_PARAMS = {
    key: value
    for key, value in {
        "temperature": env_optional_float("DEFAULT_REQUEST_TEMPERATURE"),
        "top_p": env_optional_float("DEFAULT_REQUEST_TOP_P"),
        "top_k": env_optional_int("DEFAULT_REQUEST_TOP_K"),
        "repetition_penalty": env_optional_float("DEFAULT_REQUEST_REPETITION_PENALTY"),
        "max_tokens": env_optional_int("DEFAULT_REQUEST_MAX_TOKENS"),
    }.items()
    if value is not None
}
# Undefined means transparent: preserve the caller's requested output budget
# and let vLLM enforce prompt + output against the hard context limit.
MAX_REQUEST_OUTPUT_TOKENS = env_optional_int("MAX_REQUEST_OUTPUT_TOKENS")
DEFAULT_VLLM_PRIORITY = env_int("DEFAULT_VLLM_PRIORITY", 100)
MIN_VLLM_PRIORITY = env_int("MIN_VLLM_PRIORITY", 0)
MAX_VLLM_PRIORITY = env_int("MAX_VLLM_PRIORITY", 1_000)

if MODEL_WORKING_CONTEXT_TOKENS > MODEL_MAX_CONTEXT_TOKENS:
    raise ValueError("MODEL_WORKING_CONTEXT_TOKENS cannot exceed MODEL_MAX_CONTEXT_TOKENS")
if MODEL_MAX_OUTPUT_TOKENS > MODEL_MAX_CONTEXT_TOKENS:
    raise ValueError("MODEL_MAX_OUTPUT_TOKENS cannot exceed MODEL_MAX_CONTEXT_TOKENS")
if MIN_VLLM_PRIORITY > MAX_VLLM_PRIORITY:
    raise ValueError("MIN_VLLM_PRIORITY cannot exceed MAX_VLLM_PRIORITY")


def normalize_backend(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    parsed = urllib.parse.urlparse(value)
    port = parsed.port or BACKEND_PORT
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def static_backends() -> list[str]:
    raw = os.environ.get("ROUTER_BACKENDS", "")
    return [b for b in (normalize_backend(v) for v in raw.split(",")) if b]


def asg_targets() -> list[tuple[str, str]]:
    if AWS_ASG_TARGETS:
        targets: list[tuple[str, str]] = []
        for value in AWS_ASG_TARGETS.split(","):
            region, separator, asg_name = value.strip().partition(":")
            if not separator or not region or not asg_name:
                raise ValueError(
                    "AWS_ASG_TARGETS entries must use region:auto-scaling-group"
                )
            targets.append((region, asg_name))
        return targets
    return [(AWS_REGION, AWS_ASG_NAME)] if AWS_ASG_NAME else []


def discover_asg_backends(region: str, asg_name: str) -> list[str]:
    if not asg_name:
        return []

    asg_cmd = [
        "aws",
        "autoscaling",
        "describe-auto-scaling-groups",
        "--region",
        region,
        "--auto-scaling-group-names",
        asg_name,
        "--query",
        "AutoScalingGroups[0].Instances[?LifecycleState=='InService'].InstanceId",
        "--output",
        "text",
    ]
    instance_ids = [
        value
        for value in subprocess.check_output(asg_cmd, text=True).split()
        if value and value != "None"
    ]
    if not instance_ids:
        return []

    ec2_cmd = [
        "aws",
        "ec2",
        "describe-instances",
        "--region",
        region,
        "--instance-ids",
        *instance_ids,
        "--query",
        "Reservations[].Instances[?State.Name=='running'].PrivateIpAddress",
        "--output",
        "text",
    ]
    ips = subprocess.check_output(ec2_cmd, text=True).split()
    return [normalize_backend(f"http://{ip}:{BACKEND_PORT}") for ip in ips]


def is_healthy(backend: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{backend}/v1/models", timeout=HEALTH_TIMEOUT
        ) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def discovery_loop() -> None:
    while True:
        backends = static_backends()
        for region, asg_name in asg_targets():
            try:
                backends.extend(discover_asg_backends(region, asg_name))
            except Exception as exc:
                print(
                    f"ASG discovery failed for {region}:{asg_name}: {exc}",
                    flush=True,
                )

        healthy = [backend for backend in sorted(set(backends)) if is_healthy(backend)]
        STATE.replace(backends, healthy)
        print(
            f"router backends={sorted(set(backends))} healthy={healthy}",
            flush=True,
        )
        time.sleep(DISCOVERY_INTERVAL)


def first_text(value) -> str | None:
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, list):
        return None
    for part in value:
        if isinstance(part, str) and part:
            return part
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            return text
        content = first_text(part.get("content"))
        if content:
            return content
    return None


def first_user_text(payload: dict) -> str | None:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = first_text(message.get("content"))
                if content:
                    return content

    response_input = payload.get("input")
    if isinstance(response_input, str) and response_input:
        return response_input
    if isinstance(response_input, list):
        for item in response_input:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = first_text(item.get("content"))
            if content:
                return content
    return None


def sticky_key(headers, body: bytes) -> str | None:
    for header in ("x-sticky-key", "x-session-id", "x-conversation-id", "x-user-id"):
        value = headers.get(header)
        if value:
            return value.strip()

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        payload = {}

    user = payload.get("user")
    if isinstance(user, str) and user:
        return f"user:{user}"

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_user = metadata.get("user_id")
        if isinstance(metadata_user, str) and metadata_user:
            return f"user:{metadata_user}"

    prompt_cache_key = payload.get("prompt_cache_key")
    if isinstance(prompt_cache_key, str) and prompt_cache_key:
        return f"prompt-cache:{prompt_cache_key}"

    # apply_request_defaults gives stored Responses requests a stable response
    # ID before routing. The next request carries that value as
    # previous_response_id, so both rendezvous-hash to the same vLLM process.
    for field in ("previous_response_id", "request_id"):
        response_id = payload.get(field)
        if isinstance(response_id, str) and response_id:
            return f"response:{response_id}"

    content = first_user_text(payload)
    if content:
        digest = hashlib.sha256(content[:4096].encode()).hexdigest()
        return f"prefix:{digest}"

    auth = headers.get("authorization")
    if auth:
        return f"auth:{hashlib.sha256(auth.encode()).hexdigest()}"
    return None


def vllm_priority(headers) -> int:
    """Read the trusted caller's priority and keep it in a bounded range."""
    raw_priority = headers.get(VLLM_PRIORITY_HEADER)
    if raw_priority is None:
        return DEFAULT_VLLM_PRIORITY
    try:
        priority = int(raw_priority)
    except ValueError:
        return DEFAULT_VLLM_PRIORITY
    return min(MAX_VLLM_PRIORITY, max(MIN_VLLM_PRIORITY, priority))


def apply_request_defaults(path: str, headers, body: bytes) -> bytes:
    request_path = urllib.parse.urlparse(path).path.rstrip("/")
    is_chat = request_path.endswith("/chat/completions")
    is_responses = request_path.endswith("/responses")
    is_messages = request_path.endswith("/messages")
    if not body or not (is_chat or is_responses or is_messages):
        return body

    content_type = headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return body

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body

    if not isinstance(payload, dict):
        return body

    if DEFAULT_CHAT_TEMPLATE_THINKING:
        chat_template_kwargs = payload.get("chat_template_kwargs")
        if chat_template_kwargs is None:
            chat_template_kwargs = {}
            payload["chat_template_kwargs"] = chat_template_kwargs
        if isinstance(chat_template_kwargs, dict):
            chat_template_kwargs.setdefault("thinking", True)

    # vLLM's three native APIs name the output budget differently and expose
    # slightly different sampling extensions. Only add fields accepted by the
    # selected schema so Anthropic clients retain Anthropic-compatible errors.
    max_tokens_key = "max_output_tokens" if is_responses else "max_tokens"
    for key, value in DEFAULT_REQUEST_PARAMS.items():
        target_key = max_tokens_key if key == "max_tokens" else key
        if key == "repetition_penalty" and not is_chat:
            continue
        payload.setdefault(target_key, value)
    if is_chat or is_responses:
        payload["priority"] = vllm_priority(headers)
    if is_responses and payload.get("store") is not False:
        payload.setdefault("request_id", f"resp_{uuid.uuid4().hex}")

    output_token_keys = (
        ("max_output_tokens",)
        if is_responses
        else ("max_tokens", "max_completion_tokens") if is_chat else ("max_tokens",)
    )
    for key in output_token_keys:
        value = payload.get(key)
        if (
            MAX_REQUEST_OUTPUT_TOKENS is not None
            and isinstance(value, int)
            and value > MAX_REQUEST_OUTPUT_TOKENS
        ):
            payload[key] = MAX_REQUEST_OUTPUT_TOKENS
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def align_response_id_affinity(path: str, headers, body: bytes, backend: str) -> bytes:
    """Make a stored response ID rendezvous-hash to its selected backend.

    This preserves previous_response_id routing even if the stateless router
    restarts between turns. Losing the GPU worker still loses vLLM's local
    response store, but restarting only the router does not.
    """
    request_path = urllib.parse.urlparse(path).path.rstrip("/")
    if not body or not request_path.endswith("/responses"):
        return body
    if "application/json" not in headers.get("content-type", "").lower():
        return body

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(payload, dict) or payload.get("store") is False:
        return body

    request_id = payload.get("request_id")
    if isinstance(request_id, str) and STATE.routes_to(f"response:{request_id}", backend):
        return body

    for _ in range(256):
        request_id = f"resp_{uuid.uuid4().hex}"
        if STATE.routes_to(f"response:{request_id}", backend):
            payload["request_id"] = request_id
            return json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return body


def normalize_content_parts_for_vllm(path: str, headers, body: bytes) -> bytes:
    """Translate OpenAI reasoning content parts to the vLLM-compatible shape."""
    if not body or "/chat/completions" not in path:
        return body
    if "application/json" not in headers.get("content-type", "").lower():
        return body

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    changed = False
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "reasoning":
                    part["type"] = "thinking"
                    changed = True
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") if changed else body


def router_capabilities() -> dict[str, int | str]:
    return {
        "model": MODEL_ID,
        "max_context_tokens": MODEL_MAX_CONTEXT_TOKENS,
        "working_context_tokens": MODEL_WORKING_CONTEXT_TOKENS,
        "max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
    }


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            _, healthy = STATE.snapshot()
            self.send_json(200 if healthy else 503, {"healthy": healthy})
            return
        if self.path == "/router/backends":
            backends, healthy = STATE.snapshot()
            self.send_json(200, {"backends": backends, "healthy": healthy})
            return
        if self.path == "/router/capabilities":
            self.send_json(200, router_capabilities())
            return
        self.proxy()

    def do_POST(self) -> None:
        self.proxy()

    def do_OPTIONS(self) -> None:
        self.proxy()

    def proxy(self) -> None:
        content_length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(content_length) if content_length else b""
        body = apply_request_defaults(self.path, self.headers, body)
        body = normalize_content_parts_for_vllm(self.path, self.headers, body)

        if ROUTER_API_KEY:
            expected = f"Bearer {ROUTER_API_KEY}"
            if self.headers.get("authorization") != expected:
                self.send_json(401, {"error": "unauthorized"})
                return

        backend = STATE.choose(sticky_key(self.headers, body))
        if not backend:
            self.send_json(503, {"error": "no healthy GPU backend"})
            return
        body = align_response_id_affinity(self.path, self.headers, body, backend)

        parsed = urllib.parse.urlparse(backend)
        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port or BACKEND_PORT, timeout=REQUEST_TIMEOUT
        )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
            and key.lower() not in {"host", "content-length"}
            and key.lower() != VLLM_PRIORITY_HEADER
        }
        headers["host"] = parsed.netloc
        if body:
            headers["content-length"] = str(len(body))
        headers["x-forwarded-for"] = self.client_address[0]
        headers["x-routed-backend"] = backend

        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status, response.reason)
            response_headers = response.getheaders()
            content_type = ""
            for key, value in response_headers:
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
                if key.lower() == "content-type":
                    content_type = value.lower()
            self.send_header("x-routed-backend", backend)
            self.send_header("connection", "close")
            self.end_headers()
            if content_type.startswith("text/event-stream"):
                while True:
                    line = response.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.send_json(502, {"error": f"backend proxy failed: {exc}"})
        finally:
            conn.close()


def main() -> None:
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    port = env_int("ROUTER_PORT", 8080)
    threading.Thread(target=discovery_loop, daemon=True).start()
    server = ThreadingHTTPServer((host, port), RouterHandler)
    print(f"sticky router listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
