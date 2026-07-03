from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class SpeakerProfile:
    speaker_id: str
    name: str
    embedding: list[float]
    num_chunks: int
    voiced_seconds: float
    model: str
    created_at: float


@dataclass
class SpeakerMatch:
    speaker_id: str | None
    name: str | None
    score: float | None
    second_score: float | None
    margin: float | None
    state: str
    reason: str | None = None
    probability: float | None = None
    second_probability: float | None = None
    probability_margin: float | None = None
    candidates: list[dict[str, Any]] | None = None


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or f"speaker-{uuid.uuid4().hex[:8]}"


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("invalid embedding")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("zero embedding")
    return vector / norm


def load_audio_bytes(data: bytes, suffix: str = ".webm", sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode arbitrary audio bytes through ffmpeg into mono float32 at 16 kHz."""
    if not data:
        raise ValueError("empty audio upload")
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as src:
        src.write(data)
        src.flush()
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src.name,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if not raw:
        raise ValueError("ffmpeg decoded no audio")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def audio_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def voiced_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, rms_threshold: float = 0.006) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(0.03 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    if audio.size < frame:
        return 0.0
    voiced = 0
    total = 0
    for start in range(0, audio.size - frame + 1, hop):
        chunk = audio[start : start + frame]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        total += 1
        if rms >= rms_threshold and peak >= rms_threshold * 2:
            voiced += 1
    if total == 0:
        return 0.0
    return voiced * hop / sample_rate


def split_voiced_chunks(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    rms_threshold: float = 0.006,
    min_chunk_seconds: float = 1.5,
    max_chunk_seconds: float = 3.0,
) -> list[np.ndarray]:
    """Simple energy VAD splitter for enrollment audio.

    RealtimeSTT handles online VAD. This splitter is intentionally local and deterministic
    for building speaker profiles from uploaded/recorded enrollment clips.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []
    frame = max(1, int(0.03 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    voiced_frames: list[tuple[int, int]] = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        end = start + frame
        if end > audio.size:
            break
        chunk = audio[start:end]
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        if rms >= rms_threshold and peak >= rms_threshold * 2:
            voiced_frames.append((start, end))

    if not voiced_frames:
        return []

    merged: list[tuple[int, int]] = []
    gap = int(0.25 * sample_rate)
    cur_start, cur_end = voiced_frames[0]
    for start, end in voiced_frames[1:]:
        if start - cur_end <= gap:
            cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    min_samples = int(min_chunk_seconds * sample_rate)
    max_samples = int(max_chunk_seconds * sample_rate)
    chunks: list[np.ndarray] = []
    pad = int(0.15 * sample_rate)
    for start, end in merged:
        start = max(0, start - pad)
        end = min(audio.size, end + pad)
        if end - start < min_samples:
            continue
        segment = audio[start:end]
        for sub_start in range(0, segment.size, max_samples):
            sub = segment[sub_start : sub_start + max_samples]
            if sub.size >= min_samples:
                chunks.append(sub.copy())
    return chunks


class RollingAudioBuffer:
    def __init__(self, seconds: float = 12.0, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.max_samples = int(seconds * sample_rate)
        self.samples = np.zeros(0, dtype=np.float32)
        self.total_samples = 0

    def append_pcm16(self, data: bytes) -> None:
        chunk = pcm16_bytes_to_float32(data)
        if chunk.size == 0:
            return
        self.samples = np.concatenate([self.samples, chunk])
        self.total_samples += int(chunk.size)
        if self.samples.size > self.max_samples:
            self.samples = self.samples[-self.max_samples :]

    def recent(self, seconds: float) -> np.ndarray:
        count = int(seconds * self.sample_rate)
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        return self.samples[-count:].copy()

    @property
    def duration_seconds(self) -> float:
        return self.total_samples / self.sample_rate


class PyannoteEmbeddingBackend:
    def __init__(
        self,
        model_name: str = "pyannote/embedding",
        device: str = "cuda",
        token: str | None = None,
        cache_dir: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.token = token
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._inference = None

    def _load(self):
        if self._inference is not None:
            return self._inference
        import torch
        from pyannote.audio import Inference, Model

        token = self.token or os.getenv("HF_TOKEN") or True
        model = Model.from_pretrained(self.model_name, token=token, cache_dir=self.cache_dir)
        if model is None:
            raise RuntimeError(f"Could not load pyannote embedding model: {self.model_name}")
        device = torch.device(self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu")
        model.to(device)
        self._inference = Inference(model, window="whole", device=device, batch_size=self.batch_size)
        return self._inference

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        inference = self._load()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < int(0.5 * sample_rate):
            raise ValueError("not enough audio for speaker embedding")
        import torch

        file = {"waveform": torch.from_numpy(audio).float().unsqueeze(0), "sample_rate": sample_rate}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
            warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
            warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
            raw = inference(file)
        if isinstance(raw, tuple):
            raw = raw[0]
        data = raw.data if hasattr(raw, "data") else raw
        arr = np.asarray(data, dtype=np.float32)
        while arr.ndim > 1:
            arr = arr.mean(axis=0)
        return normalize_embedding(arr)


class SpeakerProfileStore:
    def __init__(self, directory: str | Path, embedder: PyannoteEmbeddingBackend):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder

    def list_profiles(self) -> list[SpeakerProfile]:
        profiles: list[SpeakerProfile] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                profiles.append(SpeakerProfile(**data))
            except Exception:
                continue
        return profiles

    def get_profile(self, speaker_id: str) -> SpeakerProfile | None:
        path = self.directory / f"{speaker_id}.json"
        if not path.exists():
            return None
        return SpeakerProfile(**json.loads(path.read_text()))

    def delete_profile(self, speaker_id: str) -> bool:
        path = self.directory / f"{speaker_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def enroll(
        self,
        name: str,
        audio: np.ndarray,
        rms_threshold: float = 0.006,
        min_total_voiced_seconds: float = 3.0,
    ) -> SpeakerProfile:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("speaker name is required")
        chunks = split_voiced_chunks(audio, rms_threshold=rms_threshold)
        total_voiced = sum(chunk.size for chunk in chunks) / SAMPLE_RATE
        if total_voiced < min_total_voiced_seconds:
            raise ValueError(
                f"not enough voiced enrollment audio: {total_voiced:.2f}s, need {min_total_voiced_seconds:.2f}s"
            )

        embeddings = []
        for chunk in chunks:
            try:
                embeddings.append(self.embedder.embed(chunk))
            except Exception:
                continue
        if not embeddings:
            raise ValueError("no speaker embeddings could be extracted")

        matrix = np.vstack(embeddings)
        centroid = normalize_embedding(matrix.mean(axis=0))
        if matrix.shape[0] >= 4:
            scores = matrix @ centroid
            cutoff = float(np.percentile(scores, 20))
            kept = matrix[scores >= cutoff]
            if kept.size:
                centroid = normalize_embedding(kept.mean(axis=0))
                matrix = kept

        base_id = slugify_name(clean_name)
        speaker_id = base_id
        i = 2
        while (self.directory / f"{speaker_id}.json").exists():
            speaker_id = f"{base_id}-{i}"
            i += 1

        profile = SpeakerProfile(
            speaker_id=speaker_id,
            name=clean_name,
            embedding=centroid.astype(float).tolist(),
            num_chunks=int(matrix.shape[0]),
            voiced_seconds=round(float(total_voiced), 3),
            model=self.embedder.model_name,
            created_at=time.time(),
        )
        tmp = self.directory / f"{speaker_id}.json.tmp"
        final = self.directory / f"{speaker_id}.json"
        tmp.write_text(json.dumps(asdict(profile), indent=2, sort_keys=True))
        tmp.replace(final)
        return profile


DEFAULT_CALIBRATOR_FEATURES = [
    "cosine",
    "z_norm_test",
    "z_norm_target",
    "s_norm",
    "test_cohort_mean",
    "test_cohort_std",
    "target_cohort_mean",
    "target_cohort_std",
    "duration_seconds",
    "voiced_seconds",
    "rms",
]


class SpeakerCalibrator:
    def __init__(
        self,
        model_path: str | Path,
        cohort_path: str | Path | None = None,
        threshold: float | None = None,
    ):
        import joblib

        self.model_path = Path(model_path)
        payload = joblib.load(self.model_path)
        self.model = payload.get("model", payload) if isinstance(payload, dict) else payload
        self.feature_names = list(payload.get("feature_names", DEFAULT_CALIBRATOR_FEATURES)) if isinstance(payload, dict) else DEFAULT_CALIBRATOR_FEATURES
        self.threshold = float(threshold if threshold is not None else self._threshold_from_payload(payload))
        resolved_cohort = cohort_path or (payload.get("cohort_output") if isinstance(payload, dict) else None)
        self.cohort_path = Path(resolved_cohort) if resolved_cohort else None
        self.cohort_matrix = self._load_cohort(self.cohort_path)

    @staticmethod
    def _threshold_from_payload(payload: Any) -> float:
        if isinstance(payload, dict):
            recommended = payload.get("recommended_thresholds") or {}
            for key in ("far_0.01", "eer", "far_0.05", "far_0.10"):
                value = recommended.get(key) if isinstance(recommended, dict) else None
                if isinstance(value, dict) and value.get("threshold") is not None:
                    return float(value["threshold"])
        return 0.5

    @staticmethod
    def _load_cohort(path: Path | None) -> np.ndarray:
        if path is None or not path.exists():
            return np.zeros((0, 0), dtype=np.float32)
        data = np.load(path, allow_pickle=True)
        matrix = np.asarray(data["embeddings"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms <= 1e-8] = 1.0
        return matrix / norms

    def cohort_stats(self, embedding: np.ndarray) -> tuple[float, float]:
        if self.cohort_matrix.size == 0:
            return 0.0, 1.0
        scores = self.cohort_matrix @ embedding
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return mean, max(std, 1e-4)

    def features(
        self,
        target_embedding: np.ndarray,
        test_embedding: np.ndarray,
        *,
        duration_seconds: float,
        voice_seconds: float,
        rms_value: float,
    ) -> np.ndarray:
        score = float(np.dot(target_embedding, test_embedding))
        test_mean, test_std = self.cohort_stats(test_embedding)
        target_mean, target_std = self.cohort_stats(target_embedding)
        z_norm_test = (score - test_mean) / test_std
        z_norm_target = (score - target_mean) / target_std
        s_norm = (score - 0.5 * (test_mean + target_mean)) / (0.5 * (test_std + target_std) + 1e-4)
        return np.asarray(
            [
                score,
                z_norm_test,
                z_norm_target,
                s_norm,
                test_mean,
                test_std,
                target_mean,
                target_std,
                duration_seconds,
                voice_seconds,
                rms_value,
            ],
            dtype=np.float32,
        )

    def probability(
        self,
        target_embedding: np.ndarray,
        test_embedding: np.ndarray,
        *,
        duration_seconds: float,
        voice_seconds: float,
        rms_value: float,
    ) -> float:
        features = self.features(
            target_embedding,
            test_embedding,
            duration_seconds=duration_seconds,
            voice_seconds=voice_seconds,
            rms_value=rms_value,
        )
        return float(self.model.predict_proba(features.reshape(1, -1))[0, 1])


class SpeakerMatcher:
    def __init__(
        self,
        store: SpeakerProfileStore,
        threshold: float = 0.5,
        margin: float = 0.08,
        min_voiced_seconds: float = 0.8,
        rms_threshold: float = 0.006,
        calibrator: SpeakerCalibrator | None = None,
    ):
        self.store = store
        self.threshold = threshold
        self.margin = margin
        self.min_voiced_seconds = min_voiced_seconds
        self.rms_threshold = rms_threshold
        self.calibrator = calibrator

    @property
    def acceptance_threshold(self) -> float:
        return self.calibrator.threshold if self.calibrator is not None else self.threshold

    def match_audio(self, audio: np.ndarray) -> SpeakerMatch:
        profiles = self.store.list_profiles()
        if not profiles:
            return SpeakerMatch(None, None, None, None, None, "unknown", "no_profiles", candidates=[])
        voice = voiced_seconds(audio, rms_threshold=self.rms_threshold)
        if voice < self.min_voiced_seconds:
            return SpeakerMatch(None, None, None, None, None, "unknown", "insufficient_voiced_audio", candidates=[])
        try:
            embedding = self.store.embedder.embed(audio)
        except Exception as exc:
            return SpeakerMatch(None, None, None, None, None, "unknown", f"embedding_error:{exc.__class__.__name__}", candidates=[])

        duration = float(np.asarray(audio).size) / SAMPLE_RATE
        rms_value = audio_rms(audio)
        candidates: list[dict[str, Any]] = []
        for profile in profiles:
            profile_embedding = normalize_embedding(np.asarray(profile.embedding, dtype=np.float32))
            score = float(np.dot(embedding, profile_embedding))
            probability = None
            if self.calibrator is not None:
                try:
                    probability = self.calibrator.probability(
                        profile_embedding,
                        embedding,
                        duration_seconds=duration,
                        voice_seconds=voice,
                        rms_value=rms_value,
                    )
                except Exception:
                    probability = None
            candidates.append(
                {
                    "speaker_id": profile.speaker_id,
                    "name": profile.name,
                    "score": round(score, 4),
                    "probability": round(probability, 4) if probability is not None and math.isfinite(probability) else None,
                    "num_chunks": profile.num_chunks,
                }
            )

        candidates.sort(
            key=lambda item: (
                float(item["probability"]) if item.get("probability") is not None else -1.0,
                float(item["score"]) if item.get("score") is not None else -1.0,
            ),
            reverse=True,
        )
        for index, candidate in enumerate(candidates, start=1):
            candidate["rank"] = index

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        best_score = float(best["score"])
        second_score = float(second["score"]) if second and second.get("score") is not None else -1.0
        cosine_margin = best_score - second_score
        best_probability = best.get("probability")
        second_probability = second.get("probability") if second else None
        probability_margin = None
        accepted = False
        reason = "below_threshold_or_margin"

        if self.calibrator is not None and best_probability is not None:
            second_probability_value = float(second_probability) if second_probability is not None else 0.0
            probability_margin = float(best_probability) - second_probability_value
            accepted = float(best_probability) >= self.calibrator.threshold and probability_margin >= self.margin
            reason = "below_probability_or_margin"
        else:
            accepted = best_score >= self.threshold and cosine_margin >= self.margin

        if accepted:
            return SpeakerMatch(
                best["speaker_id"],
                best["name"],
                round(best_score, 4),
                round(second_score, 4) if math.isfinite(second_score) else None,
                round(cosine_margin, 4),
                "tentative",
                None,
                round(float(best_probability), 4) if best_probability is not None else None,
                round(float(second_probability), 4) if second_probability is not None else None,
                round(float(probability_margin), 4) if probability_margin is not None else None,
                candidates,
            )
        return SpeakerMatch(
            None,
            None,
            round(best_score, 4),
            round(second_score, 4) if math.isfinite(second_score) else None,
            round(cosine_margin, 4),
            "unknown",
            reason,
            round(float(best_probability), 4) if best_probability is not None else None,
            round(float(second_probability), 4) if second_probability is not None else None,
            round(float(probability_margin), 4) if probability_margin is not None else None,
            candidates,
        )
