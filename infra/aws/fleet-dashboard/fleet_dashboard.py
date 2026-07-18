#!/usr/bin/env python3
"""Read-only status page for the DeepSeek RTX Spot fleet.

Uses the VM's AWS CLI identity. No credentials are stored by this service.
"""

from __future__ import annotations

import argparse
import html
import json
import secrets
import subprocess
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FLEETS = (
    {"name": "Oregon", "region": "us-west-2", "group": "deepseek-rtx4-spot-asg-oregon"},
    {"name": "Ohio", "region": "us-east-2", "group": "deepseek-rtx4-spot-asg"},
    {"name": "Virginia", "region": "us-east-1", "group": "deepseek-rtx4-spot-asg-virginia"},
)
ROUTER_REGION = "us-east-2"
ROUTER_INSTANCE = "i-0211536fe1c0c464d"
REFRESH_SECONDS = 20
CSRF_TOKEN = secrets.token_urlsafe(32)
FLEETS_BY_NAME = {fleet["name"].lower(): fleet for fleet in FLEETS}


def aws(args: list[str], timeout: int = 20) -> str:
    result = subprocess.run(
        ["aws", *args], text=True, capture_output=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "AWS CLI failed")
    return result.stdout.strip()


def fleet_status(fleet: dict[str, str]) -> dict[str, Any]:
    try:
        raw = aws([
            "autoscaling", "describe-auto-scaling-groups", "--region", fleet["region"],
            "--auto-scaling-group-names", fleet["group"], "--output", "json",
        ])
        group = json.loads(raw)["AutoScalingGroups"][0]
        instances = [
            {
                "id": item["InstanceId"],
                "state": item["LifecycleState"],
                "health": item["HealthStatus"],
                "az": item["AvailabilityZone"],
            }
            for item in group["Instances"]
        ]
        return {
            **fleet,
            "desired": group["DesiredCapacity"],
            "min": group["MinSize"],
            "max": group["MaxSize"],
            "instances": instances,
            "error": None,
        }
    except Exception as error:  # Keep the rest of the fleet visible on a partial AWS failure.
        return {**fleet, "instances": [], "error": str(error)}


def router_backends() -> dict[str, Any]:
    command = "curl -fsS --max-time 5 http://127.0.0.1:8080/router/backends"
    try:
        command_id = aws([
            "ssm", "send-command", "--region", ROUTER_REGION,
            "--instance-ids", ROUTER_INSTANCE,
            "--document-name", "AWS-RunShellScript",
            "--parameters", json.dumps({"commands": [command]}),
            "--query", "Command.CommandId", "--output", "text",
        ])
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            raw = aws([
                "ssm", "get-command-invocation", "--region", ROUTER_REGION,
                "--command-id", command_id, "--instance-id", ROUTER_INSTANCE,
                "--output", "json",
            ])
            result = json.loads(raw)
            if result["Status"] == "Success":
                return {"data": json.loads(result["StandardOutputContent"]), "error": None}
            if result["Status"] in {"Failed", "Cancelled", "TimedOut"}:
                raise RuntimeError(result.get("StandardErrorContent") or result["Status"])
            time.sleep(1)
        raise RuntimeError("router health command timed out")
    except Exception as error:
        return {"data": None, "error": str(error)}


class State:
    lock = threading.Lock()
    data: dict[str, Any] | None = None
    updated_at = 0.0

    @classmethod
    def get(cls) -> dict[str, Any]:
        with cls.lock:
            if cls.data and time.monotonic() - cls.updated_at < REFRESH_SECONDS:
                return cls.data
            with ThreadPoolExecutor(max_workers=4) as pool:
                fleets = list(pool.map(fleet_status, FLEETS))
                router = router_backends()
            cls.data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "refresh_seconds": REFRESH_SECONDS,
                "fleets": fleets,
                "router": router,
            }
            cls.updated_at = time.monotonic()
            return cls.data

    @classmethod
    def update_capacity(cls, fleet_name: str, minimum: int, desired: int, maximum: int) -> dict[str, Any]:
        fleet = FLEETS_BY_NAME.get(fleet_name)
        if not fleet:
            raise ValueError("unknown fleet")
        if not 0 <= minimum <= desired <= maximum <= 2:
            raise ValueError("capacity must satisfy 0 <= min <= desired <= max <= 2")
        aws([
            "autoscaling", "update-auto-scaling-group", "--region", fleet["region"],
            "--auto-scaling-group-name", fleet["group"], "--min-size", str(minimum),
            "--desired-capacity", str(desired), "--max-size", str(maximum),
        ])
        with cls.lock:
            cls.data = None
            cls.updated_at = 0.0
        return cls.get()


def page(data: dict[str, Any]) -> str:
    cards = []
    for fleet in data["fleets"]:
        if fleet["error"]:
            detail = f'<p class="error">{html.escape(fleet["error"])}</p>'
        elif fleet["instances"]:
            rows = "".join(
                f"<li><code>{html.escape(item['id'])}</code> {html.escape(item['az'])} "
                f"<span class=\"ok\">{html.escape(item['state'])} / {html.escape(item['health'])}</span></li>"
                for item in fleet["instances"]
            )
            detail = f"<ul>{rows}</ul>"
        else:
            detail = '<p class="muted">No instances placed.</p>'
        actual = len(fleet["instances"])
        cards.append(f"""
          <section class="fleet">
            <div class="title"><h2>{html.escape(fleet['name'])}</h2><span class="count">{actual} active</span></div>
            <p class="capacity">Desired {fleet.get('desired', '-')} · Min {fleet.get('min', '-')} · Max {fleet.get('max', '-')}</p>
            {detail}
            <form class="capacity-form" data-fleet="{html.escape(fleet['name'].lower())}">
              <label>Min <input name="min" type="number" min="0" max="2" value="{fleet.get('min', 0)}"></label>
              <label>Desired <input name="desired" type="number" min="0" max="2" value="{fleet.get('desired', 0)}"></label>
              <label>Max <input name="max" type="number" min="0" max="2" value="{fleet.get('max', 2)}"></label>
              <button type="submit">Apply</button>
            </form>
          </section>""")
    router = data["router"]
    if router["error"]:
        router_html = f'<p class="error">{html.escape(router["error"])}</p>'
    else:
        healthy = router["data"].get("healthy", [])
        router_html = "".join(f"<li><code>{html.escape(url)}</code></li>" for url in healthy) or "<li>None</li>"
        router_html = f"<ul>{router_html}</ul>"
    updated = data["updated_at"].replace("+00:00", "Z")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}"><title>DeepSeek Fleet</title>
<style>
body{{margin:0;background:#10141b;color:#eef2f7;font:15px system-ui,sans-serif}}main{{max-width:920px;margin:42px auto;padding:0 24px}}h1{{margin:0;font-size:28px}}h2{{margin:0;font-size:18px}}.sub,.muted{{color:#9da8b5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:24px 0}}.fleet,.router{{border:1px solid #2b3544;border-radius:8px;padding:18px;background:#171d27}}.title{{display:flex;justify-content:space-between;gap:12px;align-items:center}}.count{{background:#233c31;color:#a9e6bf;padding:3px 8px;border-radius:4px;font-size:12px}}.capacity{{margin:10px 0 16px;color:#b9c5d2}}ul{{margin:0;padding-left:18px}}li{{margin:8px 0}}code{{color:#b8d6ff;word-break:break-all}}.ok{{color:#a9e6bf}}.error{{color:#ffaca5;white-space:pre-wrap}}footer{{font-size:13px;color:#9da8b5;margin-top:18px}}.capacity-form{{display:flex;flex-wrap:wrap;gap:8px;align-items:end;margin-top:18px;padding-top:14px;border-top:1px solid #2b3544}}label{{font-size:12px;color:#b9c5d2;display:grid;gap:4px}}input{{width:45px;background:#10141b;color:#eef2f7;border:1px solid #4a5768;border-radius:4px;padding:6px}}button{{background:#2e79d1;color:white;border:0;border-radius:4px;padding:7px 11px;cursor:pointer}}button:hover{{background:#3c8bea}}</style></head>
<body><main><h1>DeepSeek RTX Fleet</h1><p class="sub">AWS capacity and router status</p>
<div class="grid">{''.join(cards)}</div>
<section class="router"><h2>Router Healthy Backends</h2>{router_html}</section>
<footer>Updated {html.escape(updated)} · Refreshes every {REFRESH_SECONDS}s · <a href="/api/status" style="color:#b8d6ff">JSON</a></footer>
</main><script>
const csrf={json.dumps(CSRF_TOKEN)};
for (const form of document.querySelectorAll('.capacity-form')) {{
  form.addEventListener('submit', async (event) => {{
    event.preventDefault();
    const values=Object.fromEntries(new FormData(form));
    const fleet=form.dataset.fleet;
    if (!confirm(`Update ${{fleet}} to min ${{values.min}}, desired ${{values.desired}}, max ${{values.max}}?`)) return;
    const button=form.querySelector('button'); button.disabled=true; button.textContent='Applying...';
    try {{
      const response=await fetch(`/api/fleets/${{fleet}}/capacity`, {{method:'POST',headers:{{'content-type':'application/json','x-dashboard-csrf':csrf}},body:JSON.stringify(values)}});
      const result=await response.json();
      if (!response.ok) throw new Error(result.error || 'update failed');
      location.reload();
    }} catch (error) {{ alert(error.message); button.disabled=false; button.textContent='Apply'; }}
  }});
}}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/api/status"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = State.get()
        if self.path == "/api/status":
            body = json.dumps(data, indent=2).encode()
            content_type = "application/json"
        else:
            body = page(data).encode()
            content_type = "text/html; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "fleets"] or parts[3] != "capacity":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not secrets.compare_digest(self.headers.get("x-dashboard-csrf", ""), CSRF_TOKEN):
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "invalid CSRF token"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            data = State.update_capacity(
                parts[2], int(payload["min"]), int(payload["desired"]), int(payload["max"])
            )
        except (KeyError, TypeError, ValueError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        self.send_json(HTTPStatus.OK, data)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving fleet dashboard on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
