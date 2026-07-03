from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .diarization import SpeakerTurnTracker
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

DEFAULT_CALIBRATOR_PATH = "artifacts/calibration/speaker_calibrator.joblib"
DEFAULT_COHORT_PATH = "artifacts/cohorts/cohort_bank.npz"


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


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
    speaker_margin: float = 0.1
    speaker_window_seconds: float = 3.0
    speaker_min_voiced_seconds: float = 0.8
    speaker_calibrator_path: Path | None = Path(DEFAULT_CALIBRATOR_PATH)
    speaker_cohort_path: Path | None = Path(DEFAULT_COHORT_PATH)
    speaker_probability_threshold: float | None = None
    enable_speaker_turns: bool = True
    speaker_turn_switch_after: int = 2
    speaker_turn_min_seconds: float = 0.8
    speaker_overlap_probability: float = 0.35
    speaker_overlap_margin: float = 0.25
    stable_after: int = 2
    profiles_dir: Path = Path("data/speaker_profiles")
    api_key: str | None = None
    max_upload_mb: float = 100.0
    max_upload_seconds: float = 180.0
    upload_decode_timeout: float = 45.0
    ws_max_frame_bytes: int = 256_000
    ws_max_session_seconds: float = 7200.0
    max_sessions: int = 4
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
        self.match_threads: list[threading.Thread] = []
        self.last_match_at = 0.0
        self.last_match: dict[str, Any] | None = None
        self.speaker_history: deque[str] = deque(maxlen=3)
        self.speaker_miss_count = 0
        self.turn_tracker = SpeakerTurnTracker(
            switch_after=self.config.speaker_turn_switch_after,
            min_turn_seconds=self.config.speaker_turn_min_seconds,
            overlap_probability=self.config.speaker_overlap_probability,
            overlap_margin=self.config.speaker_overlap_margin,
        ) if self.config.enable_speaker_turns else None
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
        if self.closed.is_set() or self.loop.is_closed():
            return
        event.setdefault("session_id", self.session_id)
        event.setdefault("server_time", time.time())
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.events.put_nowait, event)

    def _latency(self) -> dict[str, Any]:
        audio_clock = self.started_at + self.audio.duration_seconds
        now = time.perf_counter()
        return {
            "audio_received_seconds": round(self.audio.duration_seconds, 3),
            "end_to_end_latency_seconds": round(max(0.0, now - audio_clock), 4),
        }

    def _speaker_payload(self) -> dict[str, Any]:
        turn_payload = self.turn_tracker.current_payload() if self.turn_tracker else {}
        if not self.last_match:
            return {
                "speaker": None,
                "speaker_state": "unknown",
                "speaker_score": None,
                "speaker_margin": None,
                "is_authorized_speaker": False,
                **turn_payload,
            }
        return {**dict(self.last_match), **turn_payload}

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
        if self.audio.duration_seconds >= self.config.ws_max_session_seconds:
            raise ValueError("maximum websocket audio session duration exceeded")
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
        thread = threading.Thread(target=self._run_match, args=(recent,), name=f"speaker-match-{self.session_id[:8]}", daemon=True)
        thread.start()
        self.match_threads.append(thread)
        self.match_threads = [item for item in self.match_threads if item.is_alive()]

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
            turn_event = self.turn_tracker.update(self.last_match, self.audio.duration_seconds) if self.turn_tracker else {}
            if turn_event:
                self.last_match.update(turn_event)
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
        if self.worker and self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=1.0)
        for thread in list(self.match_threads):
            if thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=0.2)


def _path_for_cwd(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _check_profile_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-test-{uuid.uuid4().hex}"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def runtime_checks(config: ServerConfig, profiles_dir: Path, matcher: SpeakerMatcher) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "profiles_writable": _check_profile_dir(profiles_dir),
        "speaker_calibrator_loaded": bool(matcher.calibrator),
    }
    try:
        import RealtimeSTT  # noqa: F401
        checks["realtime_stt_importable"] = True
    except Exception as exc:
        checks["realtime_stt_importable"] = False
        checks["realtime_stt_error"] = exc.__class__.__name__
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        checks["torch_error"] = exc.__class__.__name__
    checks["cuda_available"] = cuda_available
    checks["device_available"] = config.device == "cpu" or not config.device.startswith("cuda") or cuda_available
    checks["auth_configured"] = bool(config.api_key)
    checks["network_exposed_without_auth"] = config.host not in {"127.0.0.1", "localhost"} and not config.api_key
    checks["ok"] = bool(checks["ffmpeg"] and checks["profiles_writable"] and checks["realtime_stt_importable"] and checks["device_available"])
    return checks


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    return authorization[len(prefix) :].strip() if authorization.startswith(prefix) else None


def _token_valid(config: ServerConfig, token: str | None) -> bool:
    return not config.api_key or bool(token) and token == config.api_key


async def read_limited_upload(upload: UploadFile, max_bytes: int) -> bytes:
    content_length = upload.headers.get("content-length") if upload.headers else None
    if content_length:
        with contextlib.suppress(ValueError):
            if int(content_length) > max_bytes:
                raise HTTPException(status_code=413, detail=f"upload exceeds {max_bytes} bytes")
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"upload exceeds {max_bytes} bytes")
    return data


def decode_upload(data: bytes, suffix: str, config: ServerConfig):
    return load_audio_bytes(
        data,
        suffix=suffix,
        timeout_seconds=config.upload_decode_timeout,
        max_duration_seconds=config.max_upload_seconds,
    )


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="Lab Realtime STT", version="0.1.0")
    root = Path(__file__).resolve().parent
    static_dir = root / "static"
    profiles_dir = _path_for_cwd(config.profiles_dir)
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
    app.state.active_sessions = 0
    app.state.session_lock = asyncio.Lock()

    async def require_auth(
        request: Request,
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        token = x_api_key or _extract_bearer(authorization)
        if not _token_valid(config, token):
            raise HTTPException(status_code=401, detail="valid API key required")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    async def health():
        checks = runtime_checks(config, store.directory, matcher)
        body = {
            "ok": checks["ok"],
            "checks": checks,
            "realtime_stt_importable": checks.get("realtime_stt_importable", False),
            "cuda_available": checks.get("cuda_available", False),
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
            "speaker_turns_enabled": config.enable_speaker_turns,
            "speaker_turn_switch_after": config.speaker_turn_switch_after,
            "speaker_turn_min_seconds": config.speaker_turn_min_seconds,
        }
        return JSONResponse(body, status_code=200 if checks["ok"] else 503)

    @app.get("/api/speakers")
    async def list_speakers(_auth: None = Depends(require_auth)):
        return {"speakers": [asdict(profile) | {"embedding": None} for profile in store.list_profiles()]}

    @app.delete("/api/speakers/{speaker_id}")
    async def delete_speaker(speaker_id: str, _auth: None = Depends(require_auth)):
        deleted = store.delete_profile(speaker_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="speaker profile not found")
        return {"deleted": True, "speaker_id": speaker_id}

    @app.post("/api/speakers/enroll")
    async def enroll_speaker(name: str = Form(...), audio: UploadFile = File(...), _auth: None = Depends(require_auth)):
        suffix = Path(audio.filename or "upload.webm").suffix or ".webm"
        data = await read_limited_upload(audio, int(config.max_upload_mb * 1024 * 1024))
        try:
            decoded = decode_upload(data, suffix, config)
            profile = store.enroll(name, decoded)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="ffmpeg decode timed out")
        except subprocess.CalledProcessError as exc:  # type: ignore[name-defined]
            raise HTTPException(status_code=400, detail=f"ffmpeg failed: {exc.stderr.decode(errors='ignore')}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        public = asdict(profile)
        public["embedding"] = None
        return {"profile": public}

    @app.post("/api/speakers/score")
    async def score_speaker(audio: UploadFile = File(...), _auth: None = Depends(require_auth)):
        suffix = Path(audio.filename or "upload.webm").suffix or ".webm"
        data = await read_limited_upload(audio, int(config.max_upload_mb * 1024 * 1024))
        try:
            decoded = decode_upload(data, suffix, config)
            match = matcher.match_audio(decoded)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="ffmpeg decode timed out")
        except subprocess.CalledProcessError as exc:  # type: ignore[name-defined]
            raise HTTPException(status_code=400, detail=f"ffmpeg failed: {exc.stderr.decode(errors='ignore')}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"match": asdict(match), "audio_seconds": round(float(decoded.size) / SAMPLE_RATE, 3)}

    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        token = websocket.query_params.get("token") or websocket.headers.get("x-api-key") or _extract_bearer(websocket.headers.get("authorization"))
        if not _token_valid(config, token):
            await websocket.close(code=1008, reason="valid API key required")
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        session: RealtimeSession | None = None
        session_counted = False

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
                    try:
                        data = json.loads(text_message)
                    except json.JSONDecodeError:
                        await events.put({"type": "error", "code": "bad_request", "error": "invalid JSON control message"})
                        await websocket.close(code=1003)
                        break
                    message_type = data.get("type")
                    if message_type == "session.start":
                        if session is not None:
                            session.close()
                            session = None
                            if session_counted:
                                async with app.state.session_lock:
                                    app.state.active_sessions = max(0, app.state.active_sessions - 1)
                                session_counted = False
                        async with app.state.session_lock:
                            if app.state.active_sessions >= config.max_sessions:
                                await events.put({"type": "error", "code": "busy", "error": "maximum active sessions reached"})
                                continue
                            app.state.active_sessions += 1
                            session_counted = True
                        try:
                            session = await loop.run_in_executor(None, lambda: RealtimeSession(config, matcher, loop, events, data))
                        except Exception as exc:
                            async with app.state.session_lock:
                                if session_counted:
                                    app.state.active_sessions = max(0, app.state.active_sessions - 1)
                                    session_counted = False
                            await events.put({"type": "error", "code": "session_start_failed", "error": str(exc)})
                            continue
                        await events.put({
                            "type": "session.ready",
                            "sample_rate": SAMPLE_RATE,
                            "speaker_threshold": config.speaker_threshold,
                            "speaker_margin": config.speaker_margin,
                            "profiles": len(store.list_profiles()),
                            "speaker_calibrator": bool(matcher.calibrator),
                            "speaker_probability_threshold": matcher.calibrator.threshold if matcher.calibrator else None,
                            "speaker_turns_enabled": config.enable_speaker_turns,
                        })
                    elif message_type == "session.stop":
                        break
                    else:
                        await events.put({"type": "error", "code": "bad_request", "error": f"unknown control message: {message_type}"})
                data_bytes = message.get("bytes")
                if data_bytes is not None:
                    if session is None:
                        await events.put({"type": "error", "code": "bad_request", "error": "audio received before session.start"})
                        continue
                    if len(data_bytes) > config.ws_max_frame_bytes:
                        await events.put({"type": "error", "code": "frame_too_large", "error": "audio frame too large"})
                        await websocket.close(code=1009)
                        break
                    try:
                        session.feed_audio(data_bytes)
                    except ValueError as exc:
                        await events.put({"type": "error", "code": "bad_audio", "error": str(exc)})
                    except Exception as exc:
                        await events.put({"type": "error", "code": "audio_feed_failed", "error": exc.__class__.__name__})
                        await websocket.close(code=1011)
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None:
                session.close()
            if session_counted:
                async with app.state.session_lock:
                    app.state.active_sessions = max(0, app.state.active_sessions - 1)
            send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
                await send_task

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealtimeSTT lab assistant server with enrolled speaker matching.")
    parser.add_argument("--host", default=_env_str("LAB_STT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("LAB_STT_PORT", 7860))
    parser.add_argument("--device", default=_env_str("LAB_STT_DEVICE", "cuda"))
    parser.add_argument("--model", default=_env_str("LAB_STT_MODEL", "small.en"))
    parser.add_argument("--realtime-model", default=_env_str("LAB_STT_REALTIME_MODEL", "tiny.en"))
    parser.add_argument("--language", default=_env_str("LAB_STT_LANGUAGE", "en"))
    parser.add_argument("--compute-type", default=_env_str("LAB_STT_COMPUTE_TYPE", "float16"))
    parser.add_argument("--speaker-model", default=_env_str("LAB_STT_SPEAKER_MODEL", "pyannote/embedding"))
    parser.add_argument("--speaker-threshold", type=float, default=_env_float("LAB_STT_SPEAKER_THRESHOLD", 0.3))
    parser.add_argument("--speaker-margin", type=float, default=_env_float("LAB_STT_SPEAKER_MARGIN", 0.1))
    parser.add_argument("--speaker-window-seconds", type=float, default=_env_float("LAB_STT_SPEAKER_WINDOW_SECONDS", 3.0))
    parser.add_argument("--speaker-min-voiced-seconds", type=float, default=_env_float("LAB_STT_SPEAKER_MIN_VOICED_SECONDS", 0.8))
    parser.add_argument("--speaker-calibrator", default=_env_str("LAB_STT_SPEAKER_CALIBRATOR", DEFAULT_CALIBRATOR_PATH))
    parser.add_argument("--speaker-cohort-bank", default=_env_str("LAB_STT_SPEAKER_COHORT_BANK", DEFAULT_COHORT_PATH))
    parser.add_argument("--speaker-probability-threshold", type=float, default=_env_optional_float("LAB_STT_SPEAKER_PROBABILITY_THRESHOLD"))
    parser.add_argument("--no-speaker-turns", action="store_true", help="Disable Stage 1 speaker-turn timeline tracking.")
    parser.add_argument("--speaker-turn-switch-after", type=int, default=2)
    parser.add_argument("--speaker-turn-min-seconds", type=float, default=0.8)
    parser.add_argument("--speaker-overlap-probability", type=float, default=0.35)
    parser.add_argument("--speaker-overlap-margin", type=float, default=0.25)
    parser.add_argument("--profiles-dir", default=_env_str("LAB_STT_PROFILES_DIR", "data/speaker_profiles"))
    parser.add_argument("--api-key", default=os.getenv("LAB_STT_API_KEY") or None, help="Optional API key required for speaker mutation APIs and websocket sessions.")
    parser.add_argument("--max-upload-mb", type=float, default=_env_float("LAB_STT_MAX_UPLOAD_MB", 100.0))
    parser.add_argument("--max-upload-seconds", type=float, default=_env_float("LAB_STT_MAX_UPLOAD_SECONDS", 180.0))
    parser.add_argument("--upload-decode-timeout", type=float, default=_env_float("LAB_STT_UPLOAD_DECODE_TIMEOUT", 45.0))
    parser.add_argument("--ws-max-frame-bytes", type=int, default=_env_int("LAB_STT_WS_MAX_FRAME_BYTES", 256000))
    parser.add_argument("--ws-max-session-seconds", type=float, default=_env_float("LAB_STT_WS_MAX_SESSION_SECONDS", 7200.0))
    parser.add_argument("--max-sessions", type=int, default=_env_int("LAB_STT_MAX_SESSIONS", 4))
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
        speaker_calibrator_path=_optional_path(args.speaker_calibrator),
        speaker_cohort_path=_optional_path(args.speaker_cohort_bank),
        speaker_probability_threshold=args.speaker_probability_threshold,
        enable_speaker_turns=not args.no_speaker_turns,
        speaker_turn_switch_after=args.speaker_turn_switch_after,
        speaker_turn_min_seconds=args.speaker_turn_min_seconds,
        speaker_overlap_probability=args.speaker_overlap_probability,
        speaker_overlap_margin=args.speaker_overlap_margin,
        profiles_dir=Path(args.profiles_dir),
        api_key=args.api_key,
        max_upload_mb=args.max_upload_mb,
        max_upload_seconds=args.max_upload_seconds,
        upload_decode_timeout=args.upload_decode_timeout,
        ws_max_frame_bytes=args.ws_max_frame_bytes,
        ws_max_session_seconds=args.ws_max_session_seconds,
        max_sessions=args.max_sessions,
        debug=args.debug,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="debug" if config.debug else "info")


if __name__ == "__main__":
    main()
