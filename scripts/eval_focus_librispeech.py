#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import sys
import shutil
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import numpy as np
import torch
import torchaudio

from lab_realtime_stt.speaker import (
    SAMPLE_RATE,
    PyannoteEmbeddingBackend,
    SpeakerProfileStore,
    normalize_embedding,
    voiced_seconds,
)


def to_mono_16k(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform.detach().cpu().numpy().astype(np.float32)


def concat_until(items: list[dict[str, Any]], target_seconds: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks: list[np.ndarray] = []
    used: list[dict[str, Any]] = []
    silence = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    total = 0.0
    for item in items:
        audio = item["audio"]
        chunks.append(audio)
        chunks.append(silence)
        used.append({k: item[k] for k in ("speaker_id", "chapter_id", "utterance_id", "seconds")})
        total += float(audio.size) / SAMPLE_RATE
        if total >= target_seconds:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32), used
    return np.concatenate(chunks), used


def load_index(root: Path, url: str, download: bool) -> dict[str, list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(str(root), url=url, download=download)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx in range(len(dataset)):
        waveform, sample_rate, _text, speaker_id, chapter_id, utterance_id = dataset[idx]
        audio = to_mono_16k(waveform, sample_rate)
        seconds = float(audio.size) / SAMPLE_RATE
        by_speaker[str(speaker_id)].append(
            {
                "speaker_id": str(speaker_id),
                "chapter_id": str(chapter_id),
                "utterance_id": str(utterance_id),
                "audio": audio,
                "seconds": seconds,
            }
        )
    for items in by_speaker.values():
        items.sort(key=lambda x: (x["chapter_id"], x["utterance_id"]))
    return dict(by_speaker)


def select_speakers(
    by_speaker: dict[str, list[dict[str, Any]]],
    enroll_speakers: int,
    background_speakers: int,
    min_utterances: int,
    enrollment_seconds: float,
) -> tuple[list[str], list[str]]:
    eligible = []
    for speaker_id, items in by_speaker.items():
        total = sum(item["seconds"] for item in items)
        if len(items) >= min_utterances and total >= enrollment_seconds + 2.0:
            eligible.append((speaker_id, total, len(items)))
    eligible.sort(key=lambda row: (-row[1], row[0]))
    needed = enroll_speakers + background_speakers
    if len(eligible) < needed:
        raise RuntimeError(f"Need {needed} eligible speakers, found {len(eligible)}")
    enrolled = [row[0] for row in eligible[:enroll_speakers]]
    background = [row[0] for row in eligible[enroll_speakers:needed]]
    return enrolled, background


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    values: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    return values


def unique_sorted(values: list[float]) -> list[float]:
    return sorted(set(round(float(value), 6) for value in values))


def window_audio(audio: np.ndarray, window_seconds: float) -> np.ndarray:
    if window_seconds <= 0:
        return audio
    samples = int(round(window_seconds * SAMPLE_RATE))
    if samples <= 0 or audio.size <= samples:
        return audio
    return audio[-samples:].copy()


def profile_vectors(store: SpeakerProfileStore) -> list[tuple[Any, np.ndarray]]:
    rows = []
    for profile in store.list_profiles():
        rows.append((profile, normalize_embedding(np.asarray(profile.embedding, dtype=np.float32))))
    return rows


def score_audio(
    store: SpeakerProfileStore,
    profiles: list[tuple[Any, np.ndarray]],
    audio: np.ndarray,
    *,
    kind: str,
    source_speaker: str,
    expected_name: str | None,
    window_seconds: float,
) -> dict[str, Any]:
    seconds = float(audio.size) / SAMPLE_RATE
    voice = voiced_seconds(audio)
    result: dict[str, Any] = {
        "kind": kind,
        "source_speaker": source_speaker,
        "expected_name": expected_name,
        "seconds": round(seconds, 3),
        "voiced_seconds": round(float(voice), 3),
        "window_seconds": round(float(window_seconds), 3) if window_seconds > 0 else None,
        "best_speaker_id": None,
        "best_name": None,
        "score": None,
        "second_score": None,
        "margin": None,
        "score_reason": None,
    }
    if not profiles:
        result["score_reason"] = "no_profiles"
        return result
    try:
        embedding = store.embedder.embed(audio)
    except Exception as exc:
        result["score_reason"] = f"embedding_error:{exc.__class__.__name__}"
        return result

    scored = []
    for profile, profile_embedding in profiles:
        scored.append((float(np.dot(embedding, profile_embedding)), profile))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_profile = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    result.update(
        {
            "best_speaker_id": best_profile.speaker_id,
            "best_name": best_profile.name,
            "score": round(best_score, 4),
            "second_score": round(second_score, 4),
            "margin": round(best_score - second_score, 4),
        }
    )
    return result


def decide_case(raw: dict[str, Any], threshold: float, margin: float, min_voiced_seconds: float) -> dict[str, Any]:
    case = dict(raw)
    accepted = False
    reason = None
    if float(raw["voiced_seconds"]) < min_voiced_seconds:
        reason = "insufficient_voiced_audio"
    elif raw.get("score_reason") is not None:
        reason = raw["score_reason"]
    elif raw["score"] is not None and raw["margin"] is not None:
        accepted = float(raw["score"]) >= threshold and float(raw["margin"]) >= margin
        if not accepted:
            reason = "below_threshold_or_margin"

    case.update(
        {
            "speaker_id": raw["best_speaker_id"] if accepted else None,
            "name": raw["best_name"] if accepted else None,
            "state": "tentative" if accepted else "unknown",
            "reason": None if accepted else reason,
        }
    )
    if raw["kind"] == "positive":
        case["correct"] = accepted and raw["best_name"] == raw["expected_name"]
        case["false_speaker"] = accepted and raw["best_name"] != raw["expected_name"]
    else:
        case["false_accept"] = accepted
    return case


def summarize_cases(positive_cases: list[dict[str, Any]], background_cases: list[dict[str, Any]]) -> dict[str, Any]:
    positives = len(positive_cases)
    backgrounds = len(background_cases)
    positive_correct = sum(1 for case in positive_cases if case["correct"])
    positive_unknown = sum(1 for case in positive_cases if case["state"] == "unknown")
    positive_false_speaker = sum(1 for case in positive_cases if case.get("false_speaker"))
    background_false_accept = sum(1 for case in background_cases if case["false_accept"])
    return {
        "positive_cases": positives,
        "positive_top1_accuracy": round(positive_correct / positives, 4) if positives else None,
        "positive_unknown_rate": round(positive_unknown / positives, 4) if positives else None,
        "positive_false_speaker_rate": round(positive_false_speaker / positives, 4) if positives else None,
        "background_cases": backgrounds,
        "background_false_accept_rate": round(background_false_accept / backgrounds, 4) if backgrounds else None,
        "background_reject_rate": round(1.0 - background_false_accept / backgrounds, 4) if backgrounds else None,
    }


def evaluate_config(
    raw_positive_cases: list[dict[str, Any]],
    raw_background_cases: list[dict[str, Any]],
    *,
    threshold: float,
    margin: float,
    min_voiced_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    positive_cases = [decide_case(case, threshold, margin, min_voiced_seconds) for case in raw_positive_cases]
    background_cases = [decide_case(case, threshold, margin, min_voiced_seconds) for case in raw_background_cases]
    summary = summarize_cases(positive_cases, background_cases)
    return summary, positive_cases, background_cases


def recommendation_key(result: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    summary = result["summary"]
    return (
        float(summary["background_false_accept_rate"] if summary["background_false_accept_rate"] is not None else 1.0),
        float(summary["positive_false_speaker_rate"] if summary["positive_false_speaker_rate"] is not None else 1.0),
        -float(summary["positive_top1_accuracy"] if summary["positive_top1_accuracy"] is not None else 0.0),
        float(summary["positive_unknown_rate"] if summary["positive_unknown_rate"] is not None else 1.0),
        -float(result["config"]["threshold"]),
        -float(result["config"]["margin"]),
        float(result["config"]["window_seconds"] or 0.0),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    output = Path(args.output)
    profile_dir = Path(args.profiles_dir)
    if args.overwrite and profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    by_speaker = load_index(dataset_root, args.subset, args.download)
    enrolled_ids, background_ids = select_speakers(
        by_speaker,
        args.enroll_speakers,
        args.background_speakers,
        args.min_utterances,
        args.enrollment_seconds,
    )

    embedder = PyannoteEmbeddingBackend(model_name=args.speaker_model, device=args.device)
    store = SpeakerProfileStore(profile_dir, embedder=embedder)

    enrollments = []
    eval_items_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for speaker_id in enrolled_ids:
        items = by_speaker[speaker_id]
        enrollment_audio, used = concat_until(items, args.enrollment_seconds)
        used_keys = {(item["chapter_id"], item["utterance_id"]) for item in used}
        eval_items = [
            item
            for item in items
            if (item["chapter_id"], item["utterance_id"]) not in used_keys
        ][: args.eval_utterances]
        profile = store.enroll(f"speaker-{speaker_id}", enrollment_audio, min_total_voiced_seconds=args.min_enroll_voiced_seconds)
        enrollments.append(
            {
                "speaker_id": speaker_id,
                "profile_id": profile.speaker_id,
                "name": profile.name,
                "num_chunks": profile.num_chunks,
                "voiced_seconds": profile.voiced_seconds,
                "used_utterances": used,
            }
        )
        eval_items_by_speaker[speaker_id] = eval_items

    thresholds = unique_sorted([args.threshold] + (args.thresholds or []))
    margins = unique_sorted([args.margin] + (args.margins or []))
    window_seconds_values = unique_sorted([args.window_seconds] + (args.window_seconds_list or []))
    min_voiced_values = unique_sorted([args.min_match_voiced_seconds] + (args.min_match_voiced_seconds_list or []))
    primary_window_seconds = round(float(args.window_seconds), 6)

    vectors = profile_vectors(store)
    raw_cases_by_window: dict[float, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for window_seconds in window_seconds_values:
        raw_positive_cases = []
        for speaker_id, items in eval_items_by_speaker.items():
            expected_name = f"speaker-{speaker_id}"
            for item in items:
                audio = window_audio(item["audio"], window_seconds)
                raw_positive_cases.append(
                    score_audio(
                        store,
                        vectors,
                        audio,
                        kind="positive",
                        source_speaker=speaker_id,
                        expected_name=expected_name,
                        window_seconds=window_seconds,
                    )
                )

        raw_background_cases = []
        for speaker_id in background_ids:
            for item in by_speaker[speaker_id][: args.eval_utterances]:
                audio = window_audio(item["audio"], window_seconds)
                raw_background_cases.append(
                    score_audio(
                        store,
                        vectors,
                        audio,
                        kind="background",
                        source_speaker=speaker_id,
                        expected_name=None,
                        window_seconds=window_seconds,
                    )
                )
        raw_cases_by_window[window_seconds] = (raw_positive_cases, raw_background_cases)

    primary_raw_positive, primary_raw_background = raw_cases_by_window[primary_window_seconds]
    summary, positive_cases, background_cases = evaluate_config(
        primary_raw_positive,
        primary_raw_background,
        threshold=args.threshold,
        margin=args.margin,
        min_voiced_seconds=args.min_match_voiced_seconds,
    )

    sweep_results = []
    for window_seconds, threshold, margin, min_voiced_seconds in itertools.product(
        window_seconds_values, thresholds, margins, min_voiced_values
    ):
        raw_positive_cases, raw_background_cases = raw_cases_by_window[window_seconds]
        sweep_summary, _positive_cases, _background_cases = evaluate_config(
            raw_positive_cases,
            raw_background_cases,
            threshold=threshold,
            margin=margin,
            min_voiced_seconds=min_voiced_seconds,
        )
        sweep_results.append(
            {
                "config": {
                    "threshold": threshold,
                    "margin": margin,
                    "window_seconds": window_seconds if window_seconds > 0 else None,
                    "min_match_voiced_seconds": min_voiced_seconds,
                },
                "summary": sweep_summary,
            }
        )
    sweep_results.sort(key=recommendation_key)

    report = {
        "created_at": time.time(),
        "dataset": {"name": "LibriSpeech", "subset": args.subset, "root": str(dataset_root)},
        "config": {
            "speaker_model": args.speaker_model,
            "device": args.device,
            "threshold": args.threshold,
            "margin": args.margin,
            "window_seconds": args.window_seconds if args.window_seconds > 0 else None,
            "enrollment_seconds": args.enrollment_seconds,
            "min_enroll_voiced_seconds": args.min_enroll_voiced_seconds,
            "min_match_voiced_seconds": args.min_match_voiced_seconds,
            "eval_utterances": args.eval_utterances,
        },
        "speakers": {"enrolled": enrolled_ids, "background": background_ids},
        "enrollments": enrollments,
        "summary": summary,
        "sweeps": {
            "thresholds": thresholds,
            "margins": margins,
            "window_seconds": [value if value > 0 else None for value in window_seconds_values],
            "min_match_voiced_seconds": min_voiced_values,
            "recommended": sweep_results[0] if sweep_results else None,
            "results": sweep_results,
        },
        "positive_cases": positive_cases,
        "background_cases": background_cases,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate enrolled-speaker focus matching on a LibriSpeech subset.")
    parser.add_argument("--dataset-root", default="data/datasets")
    parser.add_argument("--subset", default="test-clean", choices=["dev-clean", "test-clean", "dev-other", "test-other"])
    parser.add_argument("--download", action="store_true", help="Download the LibriSpeech subset if it is not present.")
    parser.add_argument("--output", default="data/eval/focus_librispeech_report.json")
    parser.add_argument("--profiles-dir", default="data/eval/focus_profiles")
    parser.add_argument("--overwrite", action="store_true", help="Remove old eval profiles before enrolling selected speakers.")
    parser.add_argument("--speaker-model", default="pyannote/embedding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--thresholds", type=parse_float_list, help="Comma-separated threshold sweep values.")
    parser.add_argument("--margins", type=parse_float_list, help="Comma-separated margin sweep values.")
    parser.add_argument("--window-seconds", type=float, default=1.2, help="Tail window to score; use 0 for full utterances.")
    parser.add_argument("--window-seconds-list", type=parse_float_list, help="Comma-separated speaker-window sweep values; use 0 for full utterances.")
    parser.add_argument("--enroll-speakers", type=int, default=6)
    parser.add_argument("--background-speakers", type=int, default=4)
    parser.add_argument("--min-utterances", type=int, default=8)
    parser.add_argument("--enrollment-seconds", type=float, default=20.0)
    parser.add_argument("--min-enroll-voiced-seconds", type=float, default=3.0)
    parser.add_argument("--min-match-voiced-seconds", type=float, default=0.8)
    parser.add_argument(
        "--min-match-voiced-seconds-list",
        type=parse_float_list,
        help="Comma-separated min voiced seconds sweep values.",
    )
    parser.add_argument("--eval-utterances", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    report = evaluate(parse_args())
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
