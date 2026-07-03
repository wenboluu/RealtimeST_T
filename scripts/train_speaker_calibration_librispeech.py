#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
import torchaudio
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lab_realtime_stt.speaker import (
    SAMPLE_RATE,
    PyannoteEmbeddingBackend,
    normalize_embedding,
    split_voiced_chunks,
    voiced_seconds,
)

FEATURE_NAMES = [
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

LIBRISPEECH_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]


@dataclass(frozen=True)
class AudioItem:
    speaker_id: str
    chapter_id: str
    utterance_id: str
    path: str
    seconds: float


@dataclass
class SpeakerAssets:
    speaker_id: str
    profile_embedding: np.ndarray
    eval_items: list[AudioItem]
    enrollment_items: list[AudioItem]


def to_mono_16k(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform.detach().cpu().numpy().astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(str(path))
    return to_mono_16k(waveform, sample_rate)


def ensure_dataset(root: Path, subset: str, download: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if download:
        torchaudio.datasets.LIBRISPEECH(str(root), url=subset, download=True)
    subset_dir = root / "LibriSpeech" / subset
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"LibriSpeech subset not found: {subset_dir}. Pass --download or point --dataset-root at an existing corpus."
        )
    return subset_dir


def build_index(root: Path, subset: str, download: bool) -> dict[str, list[AudioItem]]:
    subset_dir = ensure_dataset(root, subset, download)
    by_speaker: dict[str, list[AudioItem]] = {}
    for path in sorted(subset_dir.glob("*/*/*.flac")):
        speaker_id = path.parent.parent.name
        chapter_id = path.parent.name
        stem = path.stem
        utterance_id = stem.split("-")[-1]
        info = torchaudio.info(str(path))
        seconds = float(info.num_frames) / float(info.sample_rate)
        by_speaker.setdefault(speaker_id, []).append(
            AudioItem(
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                utterance_id=utterance_id,
                path=str(path),
                seconds=seconds,
            )
        )
    for items in by_speaker.values():
        items.sort(key=lambda item: (item.chapter_id, item.utterance_id))
    return by_speaker


def select_speakers(
    by_speaker: dict[str, list[AudioItem]],
    *,
    train_speakers: int,
    eval_speakers: int,
    cohort_speakers: int,
    min_utterances: int,
    enrollment_seconds: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    eligible = []
    for speaker_id, items in by_speaker.items():
        total = sum(item.seconds for item in items)
        if len(items) >= min_utterances and total >= enrollment_seconds + 2.0:
            eligible.append((speaker_id, total, len(items)))
    if len(eligible) < train_speakers + eval_speakers + cohort_speakers:
        raise RuntimeError(
            "Not enough eligible speakers: "
            f"need {train_speakers + eval_speakers + cohort_speakers}, found {len(eligible)}"
        )
    rng = random.Random(seed)
    eligible.sort(key=lambda row: row[0])
    rng.shuffle(eligible)
    train = [row[0] for row in eligible[:train_speakers]]
    eval_ids = [row[0] for row in eligible[train_speakers : train_speakers + eval_speakers]]
    cohort = [row[0] for row in eligible[train_speakers + eval_speakers : train_speakers + eval_speakers + cohort_speakers]]
    return train, eval_ids, cohort


def split_enrollment_eval(
    items: list[AudioItem],
    *,
    enrollment_seconds: float,
    eval_utterances: int,
) -> tuple[list[AudioItem], list[AudioItem]]:
    enrollment: list[AudioItem] = []
    total = 0.0
    for item in items:
        enrollment.append(item)
        total += item.seconds
        if total >= enrollment_seconds:
            break
    used = {(item.chapter_id, item.utterance_id) for item in enrollment}
    eval_items = [item for item in items if (item.chapter_id, item.utterance_id) not in used][:eval_utterances]
    return enrollment, eval_items


def concat_items(items: list[AudioItem], gap_seconds: float = 0.25) -> np.ndarray:
    silence = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for item in items:
        chunks.append(load_audio(item.path))
        chunks.append(silence)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def peak_normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak and max_abs > 1e-8:
        audio = audio * (peak / max_abs)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def add_noise(audio: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    audio_power = float(np.mean(np.square(audio))) + 1e-10
    noise_power = float(np.mean(np.square(noise))) + 1e-10
    target_noise_power = audio_power / (10.0 ** (snr_db / 10.0))
    noise = noise * math.sqrt(target_noise_power / noise_power)
    return peak_normalize(audio + noise)


def add_reverb(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ir_seconds = float(rng.uniform(0.12, 0.45))
    ir_len = max(8, int(ir_seconds * SAMPLE_RATE))
    times = np.arange(ir_len, dtype=np.float32) / SAMPLE_RATE
    decay = np.exp(-times / float(rng.uniform(0.045, 0.16))).astype(np.float32)
    ir = rng.normal(0.0, 1.0, size=ir_len).astype(np.float32) * decay
    ir[0] += 4.0
    for _ in range(int(rng.integers(2, 6))):
        delay = int(rng.uniform(0.015, ir_seconds) * SAMPLE_RATE)
        if 0 <= delay < ir_len:
            ir[delay] += float(rng.uniform(0.3, 1.2))
    ir = ir / (np.linalg.norm(ir) + 1e-8)
    wet = signal.fftconvolve(audio, ir, mode="full")[: audio.size].astype(np.float32)
    mix = float(rng.uniform(0.18, 0.45))
    return peak_normalize((1.0 - mix) * audio + mix * wet)


def bandlimit(audio: np.ndarray) -> np.ndarray:
    sos = signal.butter(6, [300.0, 3400.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfilt(sos, audio).astype(np.float32)
    return peak_normalize(filtered)


def random_gain(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain_db = float(rng.uniform(-10.0, 7.0))
    return peak_normalize(audio * (10.0 ** (gain_db / 20.0)))


def soft_clip(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = float(rng.uniform(1.6, 3.5))
    return peak_normalize(np.tanh(audio * gain) / np.tanh(gain))


def augment_audio(audio: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "clean":
        return peak_normalize(audio.copy())
    if kind == "noise":
        return add_noise(audio, rng, snr_db=float(rng.uniform(5.0, 20.0)))
    if kind == "reverb":
        return add_reverb(audio, rng)
    if kind == "bandpass":
        return bandlimit(audio)
    if kind == "gain":
        return random_gain(audio, rng)
    if kind == "clip":
        return soft_clip(audio, rng)
    if kind == "lab":
        out = add_reverb(audio, rng)
        out = add_noise(out, rng, snr_db=float(rng.uniform(8.0, 18.0)))
        if rng.random() < 0.5:
            out = bandlimit(out)
        return peak_normalize(out)
    raise ValueError(f"unknown augmentation: {kind}")


def embed_profile(embedder: PyannoteEmbeddingBackend, items: list[AudioItem], min_voiced_seconds: float) -> np.ndarray:
    audio = concat_items(items)
    chunks = split_voiced_chunks(audio, min_chunk_seconds=1.5, max_chunk_seconds=3.0)
    embeddings = []
    for chunk in chunks:
        if voiced_seconds(chunk) < min_voiced_seconds:
            continue
        try:
            embeddings.append(embedder.embed(chunk))
        except Exception:
            continue
    if not embeddings:
        raise RuntimeError("could not extract enrollment embeddings")
    matrix = np.vstack(embeddings)
    centroid = normalize_embedding(matrix.mean(axis=0))
    if matrix.shape[0] >= 4:
        scores = matrix @ centroid
        cutoff = float(np.percentile(scores, 20))
        kept = matrix[scores >= cutoff]
        if kept.size:
            centroid = normalize_embedding(kept.mean(axis=0))
    return centroid


def build_assets(
    embedder: PyannoteEmbeddingBackend,
    by_speaker: dict[str, list[AudioItem]],
    speaker_ids: list[str],
    *,
    enrollment_seconds: float,
    eval_utterances: int,
    min_enroll_chunk_voiced_seconds: float,
) -> list[SpeakerAssets]:
    assets = []
    for speaker_id in speaker_ids:
        enrollment_items, eval_items = split_enrollment_eval(
            by_speaker[speaker_id], enrollment_seconds=enrollment_seconds, eval_utterances=eval_utterances
        )
        if not eval_items:
            continue
        profile_embedding = embed_profile(embedder, enrollment_items, min_voiced_seconds=min_enroll_chunk_voiced_seconds)
        assets.append(
            SpeakerAssets(
                speaker_id=speaker_id,
                profile_embedding=profile_embedding,
                eval_items=eval_items,
                enrollment_items=enrollment_items,
            )
        )
    return assets


def build_cohort_bank(
    embedder: PyannoteEmbeddingBackend,
    by_speaker: dict[str, list[AudioItem]],
    cohort_ids: list[str],
    utterances_per_speaker: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = []
    meta = []
    for speaker_id in cohort_ids:
        for item in by_speaker[speaker_id][:utterances_per_speaker]:
            audio = load_audio(item.path)
            if audio.size < int(0.5 * SAMPLE_RATE):
                continue
            try:
                rows.append(embedder.embed(audio))
                meta.append(asdict(item))
            except Exception as exc:
                meta.append({**asdict(item), "error": exc.__class__.__name__})
    if not rows:
        raise RuntimeError("cohort bank is empty")
    return np.vstack(rows).astype(np.float32), meta


def cohort_stats(embedding: np.ndarray, cohort_matrix: np.ndarray) -> tuple[float, float]:
    if cohort_matrix.size == 0:
        return 0.0, 1.0
    scores = cohort_matrix @ embedding
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean, max(std, 1e-4)


def make_features(
    target_embedding: np.ndarray,
    test_embedding: np.ndarray,
    cohort_matrix: np.ndarray,
    duration_seconds: float,
    voice_seconds: float,
    audio_rms: float,
) -> np.ndarray:
    score = float(np.dot(target_embedding, test_embedding))
    test_mean, test_std = cohort_stats(test_embedding, cohort_matrix)
    target_mean, target_std = cohort_stats(target_embedding, cohort_matrix)
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
            audio_rms,
        ],
        dtype=np.float32,
    )


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def embed_eval_items(
    embedder: PyannoteEmbeddingBackend,
    assets: list[SpeakerAssets],
    augmentations: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        for item in asset.eval_items:
            clean = load_audio(item.path)
            for aug in augmentations:
                rng = np.random.default_rng(stable_seed(asset.speaker_id, item.chapter_id, item.utterance_id, aug, seed))
                audio = augment_audio(clean, aug, rng)
                if audio.size < int(0.5 * SAMPLE_RATE):
                    continue
                try:
                    embedding = embedder.embed(audio)
                except Exception:
                    continue
                rows.append(
                    {
                        "speaker_id": asset.speaker_id,
                        "item": asdict(item),
                        "augmentation": aug,
                        "embedding": embedding,
                        "duration_seconds": float(audio.size) / SAMPLE_RATE,
                        "voiced_seconds": voiced_seconds(audio),
                        "rms": rms(audio),
                    }
                )
    return rows


def build_trials(
    assets: list[SpeakerAssets],
    eval_rows: list[dict[str, Any]],
    cohort_matrix: np.ndarray,
    negative_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = random.Random(seed)
    profiles = {asset.speaker_id: asset.profile_embedding for asset in assets}
    speaker_ids = sorted(profiles)
    features = []
    labels = []
    metadata = []
    for row in eval_rows:
        speaker_id = row["speaker_id"]
        if speaker_id not in profiles:
            continue
        test_embedding = row["embedding"]
        common = {
            "source_speaker": speaker_id,
            "augmentation": row["augmentation"],
            "duration_seconds": round(float(row["duration_seconds"]), 3),
            "voiced_seconds": round(float(row["voiced_seconds"]), 3),
        }
        features.append(
            make_features(
                profiles[speaker_id],
                test_embedding,
                cohort_matrix,
                float(row["duration_seconds"]),
                float(row["voiced_seconds"]),
                float(row["rms"]),
            )
        )
        labels.append(1)
        metadata.append({**common, "target_speaker": speaker_id, "label": 1})

        negatives = [candidate for candidate in speaker_ids if candidate != speaker_id]
        rng.shuffle(negatives)
        for target_id in negatives[:negative_per_positive]:
            features.append(
                make_features(
                    profiles[target_id],
                    test_embedding,
                    cohort_matrix,
                    float(row["duration_seconds"]),
                    float(row["voiced_seconds"]),
                    float(row["rms"]),
                )
            )
            labels.append(0)
            metadata.append({**common, "target_speaker": target_id, "label": 0})
    if not features:
        raise RuntimeError("no trials were generated")
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.int64), metadata


def operating_points(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    thresholds = sorted(set(float(x) for x in probabilities), reverse=True)
    rows = []
    best_eer = None
    for threshold in thresholds:
        accept = probabilities >= threshold
        tp = int(np.sum((accept == 1) & (y_true == 1)))
        fp = int(np.sum((accept == 1) & (y_true == 0)))
        fn = positives - tp
        far = fp / negatives if negatives else 0.0
        frr = fn / positives if positives else 0.0
        tar = tp / positives if positives else 0.0
        rows.append({"threshold": threshold, "far": far, "frr": frr, "tar": tar})
        diff = abs(far - frr)
        if best_eer is None or diff < best_eer[0]:
            best_eer = (diff, threshold, 0.5 * (far + frr), far, frr)

    def best_for_far(limit: float) -> dict[str, Any] | None:
        valid = [row for row in rows if row["far"] <= limit]
        if not valid:
            return None
        best = max(valid, key=lambda row: (row["tar"], row["threshold"]))
        return {k: round(float(v), 6) for k, v in best.items()}

    return {
        "eer": {
            "threshold": round(float(best_eer[1]), 6),
            "eer": round(float(best_eer[2]), 6),
            "far": round(float(best_eer[3]), 6),
            "frr": round(float(best_eer[4]), 6),
        }
        if best_eer
        else None,
        "far_0.01": best_for_far(0.01),
        "far_0.05": best_for_far(0.05),
        "far_0.10": best_for_far(0.10),
    }


def evaluate_model(model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    probabilities = model.predict_proba(x)[:, 1]
    predicted = probabilities >= 0.5
    accuracy = float(np.mean(predicted == y))
    report = {
        "cases": int(y.size),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "accuracy_at_0.5": round(accuracy, 4),
        "probability_mean_positive": round(float(np.mean(probabilities[y == 1])), 4) if np.any(y == 1) else None,
        "probability_mean_negative": round(float(np.mean(probabilities[y == 0])), 4) if np.any(y == 0) else None,
        "operating_points": operating_points(y, probabilities),
    }
    if len(set(y.tolist())) == 2:
        report["roc_auc"] = round(float(roc_auc_score(y, probabilities)), 4)
        report["average_precision"] = round(float(average_precision_score(y, probabilities)), 4)
    else:
        report["roc_auc"] = None
        report["average_precision"] = None
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    cohort_output = Path(args.cohort_output)
    report_path = Path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    by_speaker = build_index(dataset_root, args.subset, args.download)
    train_ids, eval_ids, cohort_ids = select_speakers(
        by_speaker,
        train_speakers=args.train_speakers,
        eval_speakers=args.eval_speakers,
        cohort_speakers=args.cohort_speakers,
        min_utterances=args.min_utterances,
        enrollment_seconds=args.enrollment_seconds,
        seed=args.seed,
    )

    embedder = PyannoteEmbeddingBackend(model_name=args.speaker_model, device=args.device)
    cohort_matrix, cohort_meta = build_cohort_bank(embedder, by_speaker, cohort_ids, args.cohort_utterances_per_speaker)
    np.savez_compressed(
        cohort_output,
        embeddings=cohort_matrix.astype(np.float32),
        speaker_ids=np.asarray([row.get("speaker_id", "") for row in cohort_meta]),
        metadata=np.asarray([json.dumps(row, sort_keys=True) for row in cohort_meta]),
        feature_names=np.asarray(FEATURE_NAMES),
    )

    train_assets = build_assets(
        embedder,
        by_speaker,
        train_ids,
        enrollment_seconds=args.enrollment_seconds,
        eval_utterances=args.eval_utterances,
        min_enroll_chunk_voiced_seconds=args.min_enroll_chunk_voiced_seconds,
    )
    eval_assets = build_assets(
        embedder,
        by_speaker,
        eval_ids,
        enrollment_seconds=args.enrollment_seconds,
        eval_utterances=args.eval_utterances,
        min_enroll_chunk_voiced_seconds=args.min_enroll_chunk_voiced_seconds,
    )
    augmentations = [part.strip() for part in args.augmentations.split(",") if part.strip()]

    train_rows = embed_eval_items(embedder, train_assets, augmentations, args.seed)
    eval_rows = embed_eval_items(embedder, eval_assets, augmentations, args.seed + 1009)
    x_train, y_train, train_meta = build_trials(
        train_assets, train_rows, cohort_matrix, args.negative_per_positive, args.seed
    )
    x_eval, y_eval, eval_meta = build_trials(
        eval_assets, eval_rows, cohort_matrix, args.negative_per_positive, args.seed + 1
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )
    model.fit(x_train, y_train)
    train_summary = evaluate_model(model, x_train, y_train)
    eval_summary = evaluate_model(model, x_eval, y_eval)

    model_path = output_dir / "speaker_calibrator.joblib"
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "created_at": time.time(),
        "config": vars(args),
        "cohort_output": str(cohort_output),
        "recommended_thresholds": eval_summary["operating_points"],
    }
    joblib.dump(payload, model_path)

    report = {
        "created_at": time.time(),
        "dataset": {"name": "LibriSpeech", "subset": args.subset, "root": str(dataset_root)},
        "outputs": {"model": str(model_path), "cohort_bank": str(cohort_output)},
        "config": vars(args),
        "speakers": {"train": train_ids, "eval": eval_ids, "cohort": cohort_ids},
        "counts": {
            "cohort_embeddings": int(cohort_matrix.shape[0]),
            "train_assets": len(train_assets),
            "eval_assets": len(eval_assets),
            "train_eval_embeddings": len(train_rows),
            "eval_eval_embeddings": len(eval_rows),
            "train_trials": int(y_train.size),
            "eval_trials": int(y_eval.size),
        },
        "feature_names": FEATURE_NAMES,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "sample_train_trials": train_meta[:10],
        "sample_eval_trials": eval_meta[:10],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LibriSpeech speaker verification logistic calibrator with cohort normalization."
    )
    parser.add_argument("--dataset-root", default=os.getenv("LAB_STT_LIBRISPEECH_ROOT", "data/datasets/librispeech"))
    parser.add_argument("--subset", default="train-clean-100", choices=LIBRISPEECH_SUBSETS)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--speaker-model", default="pyannote/embedding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=os.getenv("LAB_STT_CALIBRATION_DIR", "artifacts/calibration"))
    parser.add_argument("--cohort-output", default=os.getenv("LAB_STT_COHORT_BANK", "artifacts/cohorts/cohort_bank.npz"))
    parser.add_argument("--report", default=os.getenv("LAB_STT_CALIBRATION_REPORT", "data/eval/librispeech_calibration_report.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-speakers", type=int, default=80)
    parser.add_argument("--eval-speakers", type=int, default=20)
    parser.add_argument("--cohort-speakers", type=int, default=100)
    parser.add_argument("--min-utterances", type=int, default=10)
    parser.add_argument("--enrollment-seconds", type=float, default=20.0)
    parser.add_argument("--eval-utterances", type=int, default=4)
    parser.add_argument("--cohort-utterances-per-speaker", type=int, default=2)
    parser.add_argument("--negative-per-positive", type=int, default=8)
    parser.add_argument("--min-enroll-chunk-voiced-seconds", type=float, default=0.8)
    parser.add_argument(
        "--augmentations",
        default="clean,noise,reverb,bandpass,lab",
        help="Comma-separated augmentations: clean,noise,reverb,bandpass,gain,clip,lab",
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*TorchCodec.*")
    warnings.filterwarnings("ignore", message=".*torchcodec.*")
    warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
    report = train(parse_args())
    print(json.dumps({"counts": report["counts"], "eval_summary": report["eval_summary"]}, indent=2, sort_keys=True))
