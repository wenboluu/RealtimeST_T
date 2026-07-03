#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np

from train_speaker_calibration_librispeech import augment_audio, build_index, concat_items, load_audio
from lab_realtime_stt.diarization import SpeakerTurnTracker
from lab_realtime_stt.speaker import (
    SAMPLE_RATE,
    PyannoteEmbeddingBackend,
    SpeakerCalibrator,
    SpeakerMatcher,
    SpeakerProfileStore,
)


def eligible_speakers(by_speaker: dict[str, list[Any]], min_utterances: int, enrollment_seconds: float) -> list[str]:
    rows = []
    for speaker_id, items in by_speaker.items():
        total = sum(float(item.seconds) for item in items)
        if len(items) >= min_utterances and total >= enrollment_seconds + 2.0:
            rows.append((speaker_id, total, len(items)))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return [row[0] for row in rows]


def split_enroll_eval(items: list[Any], enrollment_seconds: float, eval_utterances: int) -> tuple[list[Any], list[Any]]:
    enroll = []
    total = 0.0
    for item in items:
        enroll.append(item)
        total += float(item.seconds)
        if total >= enrollment_seconds:
            break
    used = {(item.chapter_id, item.utterance_id) for item in enroll}
    eval_items = [item for item in items if (item.chapter_id, item.utterance_id) not in used][:eval_utterances]
    return enroll, eval_items


def make_match_payload(match: Any) -> dict[str, Any]:
    data = asdict(match)
    return {
        "speaker": data.get("name"),
        "speaker_id": data.get("speaker_id"),
        "speaker_state": data.get("state"),
        "speaker_score": data.get("score"),
        "speaker_second_score": data.get("second_score"),
        "speaker_margin": data.get("margin"),
        "speaker_reason": data.get("reason"),
        "speaker_probability": data.get("probability"),
        "speaker_second_probability": data.get("second_probability"),
        "speaker_probability_margin": data.get("probability_margin"),
        "speaker_candidates": data.get("candidates") or [],
        "is_authorized_speaker": bool(data.get("speaker_id")),
    }


def label_at(segments: list[dict[str, Any]], timestamp: float) -> dict[str, Any] | None:
    for segment in segments:
        if segment["start"] <= timestamp < segment["end"]:
            return segment
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    latencies = np.asarray([row["match_latency_seconds"] for row in rows], dtype=np.float32)
    correct = sum(1 for row in rows if row["correct"])
    known_rows = [row for row in rows if row["ground_truth_known"]]
    unknown_rows = [row for row in rows if not row["ground_truth_known"]]
    by_aug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_aug[row["augmentation"]].append(row)
    return {
        "updates": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "known_accuracy": round(sum(1 for row in known_rows if row["correct"]) / len(known_rows), 4) if known_rows else None,
        "unknown_accuracy": round(sum(1 for row in unknown_rows if row["correct"]) / len(unknown_rows), 4) if unknown_rows else None,
        "match_latency_seconds": {
            "mean": round(float(np.mean(latencies)), 4),
            "p50": round(float(np.percentile(latencies, 50)), 4),
            "p95": round(float(np.percentile(latencies, 95)), 4),
            "max": round(float(np.max(latencies)), 4),
        },
        "by_augmentation": {
            aug: {
                "updates": len(items),
                "accuracy": round(sum(1 for item in items if item["correct"]) / len(items), 4),
            }
            for aug, items in sorted(by_aug.items())
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    by_speaker = build_index(Path(args.dataset_root), args.subset, args.download)
    selected = eligible_speakers(by_speaker, args.min_utterances, args.enrollment_seconds)
    needed = args.known_speakers + args.unknown_speakers
    if len(selected) < needed:
        raise RuntimeError(f"Need {needed} eligible speakers, found {len(selected)}")
    known_ids = selected[: args.known_speakers]
    unknown_ids = selected[args.known_speakers : needed]

    embedder = PyannoteEmbeddingBackend(model_name=args.speaker_model, device=args.device)
    calibrator = None
    if args.speaker_calibrator:
        calibrator_path = Path(args.speaker_calibrator)
        if calibrator_path.exists():
            cohort_path = Path(args.speaker_cohort_bank) if args.speaker_cohort_bank else None
            calibrator = SpeakerCalibrator(
                calibrator_path,
                cohort_path=cohort_path if cohort_path and cohort_path.exists() else None,
                threshold=args.speaker_probability_threshold,
            )

    rng = np.random.default_rng(args.seed)
    augmentations = [part.strip() for part in args.augmentations.split(",") if part.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stage1-diarization-") as profile_tmp:
        store = SpeakerProfileStore(profile_tmp, embedder=embedder)
        eval_items_by_speaker: dict[str, list[Any]] = {}
        profile_id_by_source: dict[str, str] = {}
        enrollments = []
        for speaker_id in known_ids:
            enroll_items, eval_items = split_enroll_eval(by_speaker[speaker_id], args.enrollment_seconds, args.eval_utterances)
            profile = store.enroll(f"speaker-{speaker_id}", concat_items(enroll_items), min_total_voiced_seconds=args.min_enroll_voiced_seconds)
            eval_items_by_speaker[speaker_id] = eval_items
            profile_id_by_source[speaker_id] = profile.speaker_id
            enrollments.append({"speaker_id": speaker_id, "profile_id": profile.speaker_id, "num_chunks": profile.num_chunks})
        for speaker_id in unknown_ids:
            _enroll_items, eval_items = split_enroll_eval(by_speaker[speaker_id], 0.0, args.eval_utterances)
            eval_items_by_speaker[speaker_id] = eval_items

        matcher = SpeakerMatcher(
            store,
            threshold=args.speaker_threshold,
            margin=args.speaker_margin,
            min_voiced_seconds=args.min_voiced_seconds,
            calibrator=calibrator,
        )
        tracker = SpeakerTurnTracker(
            switch_after=args.switch_after,
            min_turn_seconds=args.min_turn_seconds,
            overlap_probability=args.overlap_probability,
            overlap_margin=args.overlap_margin,
        )

        sequence = []
        speakers_in_order = known_ids + unknown_ids
        for round_index in range(args.rounds):
            for speaker_id in speakers_in_order:
                items = eval_items_by_speaker.get(speaker_id) or []
                if not items:
                    continue
                item = items[round_index % len(items)]
                audio = load_audio(item.path)
                augmentation = augmentations[(round_index + speakers_in_order.index(speaker_id)) % len(augmentations)]
                audio = augment_audio(audio, augmentation, rng)
                known = speaker_id in known_ids
                sequence.append(
                    {
                        "source_speaker_id": speaker_id,
                        "speaker_id": profile_id_by_source[speaker_id] if known else None,
                        "name": profile_id_by_source[speaker_id] if known else "Unknown",
                        "known": known,
                        "augmentation": augmentation,
                        "audio": audio,
                    }
                )

        segments = []
        audio_parts = []
        cursor = 0.0
        silence = np.zeros(int(args.silence_seconds * SAMPLE_RATE), dtype=np.float32)
        for item in sequence:
            audio = item["audio"]
            start = cursor
            end = cursor + float(audio.size) / SAMPLE_RATE
            segments.append({k: v for k, v in item.items() if k != "audio"} | {"start": start, "end": end})
            audio_parts.append(audio)
            cursor = end
            if args.silence_seconds > 0:
                audio_parts.append(silence)
                cursor += args.silence_seconds
        full_audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)

        rows = []
        hop_samples = max(1, int(args.hop_seconds * SAMPLE_RATE))
        window_samples = max(1, int(args.window_seconds * SAMPLE_RATE))
        for end_sample in range(hop_samples, full_audio.size + 1, hop_samples):
            timestamp = end_sample / SAMPLE_RATE
            gt = label_at(segments, timestamp)
            if gt is None:
                continue
            start_sample = max(0, end_sample - window_samples)
            window = full_audio[start_sample:end_sample]
            started = time.perf_counter()
            match = matcher.match_audio(window)
            match_latency = time.perf_counter() - started
            event = tracker.update(make_match_payload(match), timestamp)
            turn = event.get("speaker_turn") or {}
            predicted_id = turn.get("speaker_id")
            predicted_known = bool(turn.get("known"))
            correct = predicted_id == gt["speaker_id"] if gt["known"] else not predicted_known
            rows.append(
                {
                    "timestamp": round(timestamp, 3),
                    "ground_truth_source_speaker_id": gt.get("source_speaker_id"),
                    "ground_truth_speaker_id": gt["speaker_id"],
                    "ground_truth_known": gt["known"],
                    "predicted_speaker_id": predicted_id,
                    "predicted_speaker": turn.get("speaker"),
                    "predicted_known": predicted_known,
                    "probability": turn.get("probability"),
                    "score": turn.get("score"),
                    "augmentation": gt["augmentation"],
                    "correct": bool(correct),
                    "match_latency_seconds": round(float(match_latency), 5),
                    "speaker_turn_changed": bool(event.get("speaker_turn_changed")),
                }
            )

    report = {
        "created_at": time.time(),
        "dataset": {"name": "LibriSpeech", "subset": args.subset, "root": args.dataset_root},
        "config": vars(args),
        "speakers": {"known": known_ids, "unknown": unknown_ids},
        "enrollments": enrollments,
        "timeline": [{k: v for k, v in segment.items() if k != "audio"} for segment in segments],
        "summary": summarize(rows),
        "samples": rows[: min(200, len(rows))],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 known-speaker diarization-lite on LibriSpeech streams.")
    parser.add_argument("--dataset-root", default="/data/wenbolu/datasets/lab-realtime-stt/librispeech")
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", default="/data/wenbolu/outputs/lab-realtime-stt/reports/stage1_diarization_librispeech.json")
    parser.add_argument("--speaker-model", default="pyannote/embedding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker-calibrator", default="/data/wenbolu/checkpoints/lab-realtime-stt/calibration/librispeech_starter/speaker_calibrator.joblib")
    parser.add_argument("--speaker-cohort-bank", default="/data/wenbolu/checkpoints/lab-realtime-stt/cohorts/cohort_bank_librispeech_starter.npz")
    parser.add_argument("--speaker-probability-threshold", type=float)
    parser.add_argument("--speaker-threshold", type=float, default=0.3)
    parser.add_argument("--speaker-margin", type=float, default=0.2)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=0.75)
    parser.add_argument("--min-voiced-seconds", type=float, default=0.8)
    parser.add_argument("--switch-after", type=int, default=2)
    parser.add_argument("--min-turn-seconds", type=float, default=0.8)
    parser.add_argument("--overlap-probability", type=float, default=0.35)
    parser.add_argument("--overlap-margin", type=float, default=0.25)
    parser.add_argument("--known-speakers", type=int, default=2)
    parser.add_argument("--unknown-speakers", type=int, default=1)
    parser.add_argument("--min-utterances", type=int, default=8)
    parser.add_argument("--enrollment-seconds", type=float, default=12.0)
    parser.add_argument("--min-enroll-voiced-seconds", type=float, default=3.0)
    parser.add_argument("--eval-utterances", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--silence-seconds", type=float, default=0.25)
    parser.add_argument("--augmentations", default="clean,noise,reverb")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    report = evaluate(parse_args())
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
