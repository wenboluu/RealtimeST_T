from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .speaker import (
    PyannoteEmbeddingBackend,
    RollingAudioBuffer,
    SAMPLE_RATE,
    SpeakerCalibrator,
    SpeakerMatcher,
    SpeakerProfileStore,
    load_audio_bytes,
)

LOGGER = logging.getLogger("lab_realtime_stt")


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    device: str = "cuda"
    model: str = "small.en"
    realtime_model: str = "tiny.en"
    language: str = "en"
    compute_type: str = "float16"
    speaker_model: str = "pyannote/embedding"
    speaker_threshold: float = 0.3
    speaker_margin: float = 0.2
    speaker_window_seconds: float = 3.0
    speaker_min_voiced_seconds: float = 0.8
    speaker_calibrator_path: Path | None = Path("/data/wenbolu/checkpoints/lab-realtime-stt/calibration/librispeech_starter/speaker_calibrator.joblib")
    speaker_cohort_path: Path | None = Path("/data/wenbolu/checkpoints/lab-realtime-stt/cohorts/cohort_bank_librispeech_starter.npz")
    speaker_probability_threshold: float | None = None
    stable_after: int = 2
    profiles_dir: Path = Path("data/speaker_profiles")
    debug: bool = False


class RealtimeSession:
    def __init__(
        self,
        config: ServerConfig,
        matcher: SpeakerMatcher,
        loop: asyncio.AbstractEventLoop,
        events: asyncio.Queue,
        session_options: dict[str, Any] | None = None,
    ):
        self.config = config
        self.matcher = matcher
        self.loop = loop
        self.events = events
        self.session_id = uuid.uuid4().hex
        self.started_at = time.perf_counter()
        self.audio = RollingAudioBuffer(seconds=12.0, sample_rate=SAMPLE_RATE)
        self.closed = threading.Event()
        self.match_lock = threading.Lock()
        self.match_running = False
        self.last_match_at = 0.0
        self.last_match: dict[str, Any] | None = None
        self.speaker_history: deque[str] = deque(maxlen=3)
        self.speaker_miss_count = 0
        self.worker: threading.Thread | None = None
        self.recorder = self._build_recorder(session_options or {})
        self.worker = threading.Thread(target=self._final_loop, name=f"stt-final-{self.session_id[:8]}", daemon=True)
        self.worker.start()

    def _build_recorder(self, options: dict[str, Any]):
        try:
            from RealtimeSTT import AudioToTextRecorder
        except Exception as exc:  # pragma: no cover - covered by install smoke, not unit tests
            raise RuntimeError(
                "RealtimeSTT is not installed. Activate lab-realtime-stt or install requirements.txt."
            ) from exc

        language = str(options.get("language") or self.config.language or "en")
        return AudioToTextRecorder(
            use_microphone=False,
            model=self.config.model,
            realtime_model_type=self.config.realtime_model,
            language=language,
            device=self.config.device,
            compute_type=self.config.compute_type,
            enable_realtime_transcription=True,
            realtime_processing_pause=float(options.get("realtime_processing_pause") or 0.2),
            init_realtime_after_seconds=float(options.get("init_realtime_after_seconds") or 0.2),
            post_speech_silence_duration=float(options.get("post_speech_silence_duration") or 0.5),
            pre_recording_buffer_duration=float(options.get("pre_recording_buffer_duration") or 0.8),
            min_length_of_recording=float(options.get("min_length_of_recording") or 0.5),
            on_realtime_transcription_update=lambda text: self._emit_transcript("partial", text),
            on_realtime_transcription_stabilized=lambda text: self._emit_transcript("stable_partial", text),
            spinner=False,
            no_log_file=True,
            debug_mode=self.config.debug,
        )

    def _enqueue(self, event: dict[str, Any]) -> None:
        event.setdefault("session_id", self.session_id)
        event.setdefault("server_time", time.time())
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)

    def _latency(self) -> dict[str, Any]:
        audio_clock = self.started_at + self.audio.duration_seconds
        now = time.perf_counter()
        return {
            "audio_received_seconds": round(self.audio.duration_seconds, 3),
            "end_to_end_latency_seconds": round(max(0.0, now - audio_clock), 4),
        }

    def _speaker_payload(self) -> dict[str, Any]:
        if not self.last_match:
            return {
                "speaker": None,
                "speaker_state": "unknown",
                "speaker_score": None,
                "speaker_margin": None,
                "is_authorized_speaker": False,
            }
        return dict(self.last_match)

    def _emit_transcript(self, kind: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        payload = {
            "type": "transcript",
            "kind": kind,
            "text": text,
            **self._speaker_payload(),
            **self._latency(),
        }
        self._enqueue(payload)

    def _final_loop(self) -> None:
        while not self.closed.is_set():
            try:
                text = self.recorder.text()
            except Exception as exc:
                if not self.closed.is_set():
                    self._enqueue({"type": "error", "error": f"RealtimeSTT final loop failed: {exc}"})
                break
            if self.closed.is_set():
                break
            self._emit_transcript("final", text)

    def feed_audio(self, data: bytes) -> None:
        if self.closed.is_set():
            return
        self.audio.append_pcm16(data)
        self.recorder.feed_audio(data, original_sample_rate=SAMPLE_RATE)
        self._maybe_schedule_match()

    def _maybe_schedule_match(self) -> None:
        now = time.perf_counter()
        if now - self.last_match_at < 0.5:
            return
        recent = self.audio.recent(self.config.speaker_window_seconds)
        if recent.size < int(0.5 * SAMPLE_RATE):
            return
        with self.match_lock:
            if self.match_running:
                return
            self.match_running = True
            self.last_match_at = now
        threading.Thread(target=self._run_match, args=(recent,), name=f"speaker-match-{self.session_id[:8]}", daemon=True).start()

    def _run_match(self, audio) -> None:
        started = time.perf_counter()
        try:
            match = self.matcher.match_audio(audio)
            payload = asdict(match)
            speaker_id = payload.get("speaker_id")
            previous = dict(self.last_match) if self.last_match else None
            candidate_payload = {
                "speaker_probability": payload.get("probability"),
                "speaker_second_probability": payload.get("second_probability"),
                "speaker_probability_margin": payload.get("probability_margin"),
                "speaker_candidates": payload.get("candidates") or [],
            }
            if speaker_id:
                self.speaker_miss_count = 0
                self.speaker_history.append(speaker_id)
                count = Counter(self.speaker_history).most_common(1)[0]
                if count[0] == speaker_id and count[1] >= self.config.stable_after:
                    payload["state"] = "stable"
                self.last_match = {
                    "speaker": payload.get("name"),
                    "speaker_id": payload.get("speaker_id"),
                    "speaker_state": payload.get("state"),
                    "speaker_score": payload.get("score"),
                    "speaker_second_score": payload.get("second_score"),
                    "speaker_margin": payload.get("margin"),
                    "speaker_reason": payload.get("reason"),
                    **candidate_payload,
                    "is_authorized_speaker": bool(payload.get("speaker_id") and payload.get("state") == "stable"),
                }
            else:
                self.speaker_miss_count += 1
                confidence = payload.get("probability") if payload.get("probability") is not None else payload.get("score")
                hard_clear = confidence is not None and confidence < max(0.0, self.matcher.acceptance_threshold - 0.2)
                can_hold = previous and previous.get("speaker_id") and self.speaker_miss_count <= 2 and not hard_clear
                if can_hold:
                    self.last_match = {
                        **previous,
                        "speaker_score": payload.get("score"),
                        "speaker_second_score": payload.get("second_score"),
                        "speaker_margin": payload.get("margin"),
                        "speaker_reason": f"held_after_miss:{payload.get('reason')}",
                        **candidate_payload,
                    }
                else:
                    self.speaker_history.clear()
                    self.last_match = {
                        "speaker": None,
                        "speaker_id": None,
                        "speaker_state": payload.get("state"),
                        "speaker_score": payload.get("score"),
                        "speaker_second_score": payload.get("second_score"),
                        "speaker_margin": payload.get("margin"),
                        "speaker_reason": payload.get("reason"),
                        **candidate_payload,
                        "is_authorized_speaker": False,
                    }
            self._enqueue({
                "type": "speaker.match",
                **self.last_match,
                "speaker_match_seconds": round(time.perf_counter() - started, 4),
                **self._latency(),
            })
        finally:
            with self.match_lock:
                self.match_running = False

    def close(self) -> None:
        self.closed.set()
        try:
            self.recorder.shutdown()
        except Exception:
            pass


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="Lab Realtime STT", version="0.1.0")
    root = Path(__file__).resolve().parent
    static_dir = root / "static"
    profiles_dir = config.profiles_dir if config.profiles_dir.is_absolute() else Path.cwd() / config.profiles_dir
    embedder = PyannoteEmbeddingBackend(model_name=config.speaker_model, device=config.device)
    store = SpeakerProfileStore(profiles_dir, embedder=embedder)
    calibrator = None
    if config.speaker_calibrator_path is not None:
        calibrator_path = config.speaker_calibrator_path if config.speaker_calibrator_path.is_absolute() else Path.cwd() / config.speaker_calibrator_path
        cohort_path = None
        if config.speaker_cohort_path is not None:
            cohort_path = config.speaker_cohort_path if config.speaker_cohort_path.is_absolute() else Path.cwd() / config.speaker_cohort_path
        if calibrator_path.exists():
            try:
                calibrator = SpeakerCalibrator(
                    calibrator_path,
                    cohort_path=cohort_path if cohort_path and cohort_path.exists() else None,
                    threshold=config.speaker_probability_threshold,
                )
                LOGGER.info("Loaded speaker calibrator: %s", calibrator_path)
            except Exception as exc:
                LOGGER.warning("Could not load speaker calibrator %s: %s", calibrator_path, exc)
        else:
            LOGGER.info("Speaker calibrator not found: %s", calibrator_path)
    matcher = SpeakerMatcher(
        store,
        threshold=config.speaker_threshold,
        margin=config.speaker_margin,
        min_voiced_seconds=config.speaker_min_voiced_seconds,
        calibrator=calibrator,
    )

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.state.config = config
    app.state.store = store
    app.state.matcher = matcher
    app.state.calibrator = calibrator

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    async def health():
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        try:
            import RealtimeSTT  # noqa: F401
            realtime_stt = True
        except Exception:
            realtime_stt = False
        return {
            "ok": True,
            "realtime_stt_importable": realtime_stt,
            "cuda_available": cuda_available,
            "device": config.device,
            "model": config.model,
            "realtime_model": config.realtime_model,
            "speaker_model": config.speaker_model,
            "profiles": len(store.list_profiles()),
            "threshold": config.speaker_threshold,
            "margin": config.speaker_margin,
            "speaker_window_seconds": config.speaker_window_seconds,
            "speaker_min_voiced_seconds": config.speaker_min_voiced_seconds,
            "speaker_calibrator": bool(matcher.calibrator),
            "speaker_probability_threshold": matcher.calibrator.threshold if matcher.calibrator else None,
            "speaker_cohort_embeddings": int(matcher.calibrator.cohort_matrix.shape[0]) if matcher.calibrator is not None else 0,
        }

    @app.get("/api/speakers")
    async def list_speakers():
        return {"speakers": [asdict(profile) | {"embedding": None} for profile in store.list_profiles()]}

    @app.delete("/api/speakers/{speaker_id}")
    async def delete_speaker(speaker_id: str):
        deleted = store.delete_profile(speaker_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="speaker profile not found")
        return {"deleted": True, "speaker_id": speaker_id}

    @app.post("/api/speakers/enroll")
    async def enroll_speaker(name: str = Form(...), audio: UploadFile = File(...)):
        suffix = Path(audio.filename or "upload.webm").suffix or ".webm"
        data = await audio.read()
        try:
            decoded = load_audio_bytes(data, suffix=suffix)
            profile = store.enroll(name, decoded)
        except subprocess.CalledProcessError as exc:  # type: ignore[name-defined]
            raise HTTPException(status_code=400, detail=f"ffmpeg failed: {exc.stderr.decode(errors='ignore')}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        public = asdict(profile)
        public["embedding"] = None
        return {"profile": public}

    @app.post("/api/speakers/score")
    async def score_speaker(audio: UploadFile = File(...)):
        suffix = Path(audio.filename or "upload.webm").suffix or ".webm"
        data = await audio.read()
        try:
            decoded = load_audio_bytes(data, suffix=suffix)
            match = matcher.match_audio(decoded)
        except subprocess.CalledProcessError as exc:  # type: ignore[name-defined]
            raise HTTPException(status_code=400, detail=f"ffmpeg failed: {exc.stderr.decode(errors='ignore')}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"match": asdict(match), "audio_seconds": round(float(decoded.size) / SAMPLE_RATE, 3)}

    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        session: RealtimeSession | None = None

        async def sender():
            while True:
                event = await events.get()
                await websocket.send_text(json.dumps(event))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text_message = message.get("text")
                if text_message is not None:
                    data = json.loads(text_message)
                    if data.get("type") == "session.start":
                        if session is not None:
                            session.close()
                        try:
                            session = RealtimeSession(config, matcher, loop, events, data)
                        except Exception as exc:
                            await events.put({"type": "error", "error": str(exc)})
                            continue
                        await events.put({
                            "type": "session.ready",
                            "sample_rate": SAMPLE_RATE,
                            "speaker_threshold": config.speaker_threshold,
                            "speaker_margin": config.speaker_margin,
                            "profiles": len(store.list_profiles()),
                            "speaker_calibrator": bool(matcher.calibrator),
                            "speaker_probability_threshold": matcher.calibrator.threshold if matcher.calibrator else None,
                        })
                    elif data.get("type") == "session.stop":
                        break
                data_bytes = message.get("bytes")
                if data_bytes is not None and session is not None:
                    session.feed_audio(data_bytes)
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None:
                session.close()
            send_task.cancel()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealtimeSTT lab assistant server with enrolled speaker matching.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--realtime-model", default="tiny.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--speaker-model", default="pyannote/embedding")
    parser.add_argument("--speaker-threshold", type=float, default=0.3)
    parser.add_argument("--speaker-margin", type=float, default=0.2)
    parser.add_argument("--speaker-window-seconds", type=float, default=3.0)
    parser.add_argument("--speaker-min-voiced-seconds", type=float, default=0.8)
    parser.add_argument("--speaker-calibrator", default="/data/wenbolu/checkpoints/lab-realtime-stt/calibration/librispeech_starter/speaker_calibrator.joblib")
    parser.add_argument("--speaker-cohort-bank", default="/data/wenbolu/checkpoints/lab-realtime-stt/cohorts/cohort_bank_librispeech_starter.npz")
    parser.add_argument("--speaker-probability-threshold", type=float)
    parser.add_argument("--profiles-dir", default="data/speaker_profiles")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    config = ServerConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        model=args.model,
        realtime_model=args.realtime_model,
        language=args.language,
        compute_type=args.compute_type,
        speaker_model=args.speaker_model,
        speaker_threshold=args.speaker_threshold,
        speaker_margin=args.speaker_margin,
        speaker_window_seconds=args.speaker_window_seconds,
        speaker_min_voiced_seconds=args.speaker_min_voiced_seconds,
        speaker_calibrator_path=Path(args.speaker_calibrator) if args.speaker_calibrator else None,
        speaker_cohort_path=Path(args.speaker_cohort_bank) if args.speaker_cohort_bank else None,
        speaker_probability_threshold=args.speaker_probability_threshold,
        profiles_dir=Path(args.profiles_dir),
        debug=args.debug,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="debug" if config.debug else "info")


if __name__ == "__main__":
    main()
