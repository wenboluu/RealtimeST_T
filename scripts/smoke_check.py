#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlparse

import websockets


def auth_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def get_json(url: str, timeout: float, api_key: str | None = None) -> dict:
    req = urllib.request.Request(url, headers=auth_headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return parsed._replace(scheme=scheme, path="/ws/transcribe", params="", query="", fragment="").geturl()


async def check_websocket(base_url: str, timeout: float, api_key: str | None = None) -> dict:
    target = ws_url(base_url)
    if api_key:
        sep = "&" if "?" in target else "?"
        target = f"{target}{sep}token={api_key}"
    async with websockets.connect(target, open_timeout=timeout, close_timeout=timeout) as ws:
        await ws.send(json.dumps({"type": "session.start", "language": "en"}))
        message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        event = json.loads(message)
        await ws.send(json.dumps({"type": "session.stop"}))
        if event.get("type") != "session.ready":
            raise RuntimeError(f"expected session.ready, got {event}")
        return event


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check a running Lab Realtime STT server.")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--skip-websocket", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("LAB_STT_API_KEY") or None)
    args = parser.parse_args()
    base = args.url.rstrip("/") + "/"

    try:
        health = get_json(urljoin(base, "api/health"), args.timeout, args.api_key)
        if not health.get("ok"):
            raise RuntimeError(f"health did not report ok: {health}")
        page = get_text(base, args.timeout)
        if "Lab Realtime STT" not in page:
            raise RuntimeError("index page did not contain expected title")
        ready = None if args.skip_websocket else asyncio.run(check_websocket(base, args.timeout, args.api_key))
    except (OSError, urllib.error.URLError, TimeoutError, RuntimeError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, "health": health, "session_ready": ready}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
