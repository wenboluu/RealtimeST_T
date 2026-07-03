#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def decode_to_pcm(path: Path) -> bytes:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stream an audio file to the Lab Realtime STT websocket.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:7860/ws/transcribe")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-seconds", type=float, default=None, help="Only stream the first N seconds of decoded audio.")
    parser.add_argument("--realtime", action="store_true", help="Sleep between chunks to emulate realtime capture.")
    args = parser.parse_args()

    raw = decode_to_pcm(args.audio)
    if args.max_seconds is not None:
        raw = raw[: int(args.max_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)]
    chunk_bytes = int(SAMPLE_RATE * BYTES_PER_SAMPLE * args.chunk_ms / 1000)
    started = time.perf_counter()
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "language": args.language}))

        async def receiver():
            async for message in ws:
                print(message, flush=True)

        recv_task = asyncio.create_task(receiver())
        for offset in range(0, len(raw), chunk_bytes):
            await ws.send(raw[offset : offset + chunk_bytes])
            if args.realtime:
                await asyncio.sleep(args.chunk_ms / 1000)
        await ws.send(json.dumps({"type": "session.stop"}))
        await asyncio.sleep(1.0)
        recv_task.cancel()
    elapsed = time.perf_counter() - started
    audio_seconds = len(raw) / BYTES_PER_SAMPLE / SAMPLE_RATE
    print(json.dumps({"audio_seconds": round(audio_seconds, 3), "elapsed_seconds": round(elapsed, 3)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
