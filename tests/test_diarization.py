from __future__ import annotations

from lab_realtime_stt.diarization import SpeakerTurnTracker


def payload(name=None, speaker_id=None, probability=None, state="stable"):
    candidates = []
    if speaker_id:
        candidates.append({"speaker_id": speaker_id, "name": name, "probability": probability, "score": 0.7})
    return {
        "speaker": name,
        "speaker_id": speaker_id,
        "speaker_state": state if speaker_id else "unknown",
        "speaker_probability": probability,
        "speaker_score": 0.7 if speaker_id else 0.1,
        "speaker_reason": None if speaker_id else "below_probability_or_margin",
        "speaker_candidates": candidates,
    }


def test_tracker_creates_known_and_unknown_turns_after_stable_switches():
    tracker = SpeakerTurnTracker(switch_after=2, min_turn_seconds=0.0)
    event = tracker.update(payload("Wenbo", "wenbo", 0.95), 1.0)
    assert event["speaker_turn_changed"] is True
    assert event["speaker_turn"]["speaker"] == "Wenbo"

    tracker.update(payload("Wenbo", "wenbo", 0.96), 1.5)
    event = tracker.update(payload(), 2.0)
    assert event["speaker_turn_changed"] is False
    event = tracker.update(payload(), 2.5)
    assert event["speaker_turn_changed"] is True
    assert event["speaker_turn"]["speaker"] == "Unknown"
    assert len(event["speaker_turns"]) == 2


def test_tracker_suppresses_single_window_speaker_blip():
    tracker = SpeakerTurnTracker(switch_after=2, min_turn_seconds=0.0)
    tracker.update(payload("Wenbo", "wenbo", 0.95), 1.0)
    event = tracker.update(payload("Tanming", "tanming", 0.93), 1.5)
    assert event["speaker_turn_changed"] is False
    event = tracker.update(payload("Wenbo", "wenbo", 0.94), 2.0)
    assert event["speaker_turn_changed"] is False
    assert event["speaker_turn"]["speaker"] == "Wenbo"
    assert len(event["speaker_turns"]) == 1


def test_tracker_marks_possible_overlap_from_second_probability():
    tracker = SpeakerTurnTracker(switch_after=1, min_turn_seconds=0.0, overlap_probability=0.35, overlap_margin=0.25)
    event = tracker.update(
        {
            "speaker": "Wenbo",
            "speaker_id": "wenbo",
            "speaker_state": "stable",
            "speaker_probability": 0.7,
            "speaker_score": 0.6,
            "speaker_candidates": [
                {"speaker_id": "wenbo", "name": "Wenbo", "probability": 0.7, "score": 0.6},
                {"speaker_id": "tanming", "name": "Tanming", "probability": 0.5, "score": 0.45},
            ],
        },
        1.0,
    )
    assert event["speaker_turn_overlap_possible"] is True
    assert event["speaker_turn"]["second_speaker"] == "Tanming"
