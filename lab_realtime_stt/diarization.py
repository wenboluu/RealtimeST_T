from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

UNKNOWN_SPEAKER_ID = "unknown"
UNKNOWN_SPEAKER_NAME = "Unknown"


@dataclass
class SpeakerTurn:
    turn_id: int
    start: float
    end: float
    speaker_id: str | None
    speaker: str
    known: bool
    state: str
    probability: float | None = None
    score: float | None = None
    reason: str | None = None
    overlap_possible: bool = False
    second_speaker: str | None = None
    second_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["start"] = round(float(row["start"]), 3)
        row["end"] = round(float(row["end"]), 3)
        return row


class SpeakerTurnTracker:
    """Stage-1 online diarization-lite for known speakers plus one Unknown class.

    The tracker consumes smoothed speaker-match payloads from the fast path. It does
    not try to discover multiple unknown identities; all non-enrolled speakers are
    grouped into a single Unknown turn.
    """

    def __init__(
        self,
        *,
        switch_after: int = 2,
        min_turn_seconds: float = 0.8,
        max_recent_turns: int = 12,
        overlap_probability: float = 0.35,
        overlap_margin: float = 0.25,
    ):
        self.switch_after = max(1, int(switch_after))
        self.min_turn_seconds = max(0.0, float(min_turn_seconds))
        self.max_recent_turns = max(1, int(max_recent_turns))
        self.overlap_probability = float(overlap_probability)
        self.overlap_margin = float(overlap_margin)
        self.turns: list[SpeakerTurn] = []
        self.current: SpeakerTurn | None = None
        self.pending_key: str | None = None
        self.pending_count = 0
        self.pending_since: float | None = None
        self.next_turn_id = 1

    def _candidate(self, payload: dict[str, Any]) -> dict[str, Any]:
        speaker_id = payload.get("speaker_id")
        known = bool(speaker_id)
        candidates = payload.get("speaker_candidates") or []
        top_probability = payload.get("speaker_probability")
        top_score = payload.get("speaker_score")
        second_speaker = None
        second_probability = payload.get("speaker_second_probability")
        if len(candidates) > 1:
            second_speaker = candidates[1].get("name") or candidates[1].get("speaker_id")
            if second_probability is None:
                second_probability = candidates[1].get("probability")
        elif len(candidates) == 1 and not known:
            second_speaker = candidates[0].get("name") or candidates[0].get("speaker_id")
            if second_probability is None:
                second_probability = candidates[0].get("probability")

        overlap_possible = False
        if top_probability is not None and second_probability is not None:
            top = float(top_probability)
            second = float(second_probability)
            overlap_possible = second >= self.overlap_probability and (top - second) <= self.overlap_margin

        return {
            "key": str(speaker_id or UNKNOWN_SPEAKER_ID),
            "speaker_id": speaker_id,
            "speaker": payload.get("speaker") or UNKNOWN_SPEAKER_NAME,
            "known": known,
            "state": payload.get("speaker_state") or "unknown",
            "probability": top_probability,
            "score": top_score,
            "reason": payload.get("speaker_reason"),
            "overlap_possible": overlap_possible,
            "second_speaker": second_speaker,
            "second_probability": second_probability,
        }

    @staticmethod
    def _same_turn(turn: SpeakerTurn, candidate: dict[str, Any]) -> bool:
        current_key = turn.speaker_id or UNKNOWN_SPEAKER_ID
        return current_key == candidate["key"]

    def _new_turn(self, candidate: dict[str, Any], start: float, end: float) -> SpeakerTurn:
        turn = SpeakerTurn(
            turn_id=self.next_turn_id,
            start=max(0.0, float(start)),
            end=max(float(start), float(end)),
            speaker_id=candidate["speaker_id"],
            speaker=candidate["speaker"],
            known=bool(candidate["known"]),
            state=candidate["state"],
            probability=candidate["probability"],
            score=candidate["score"],
            reason=candidate["reason"],
            overlap_possible=bool(candidate["overlap_possible"]),
            second_speaker=candidate["second_speaker"],
            second_probability=candidate["second_probability"],
        )
        self.next_turn_id += 1
        self.turns.append(turn)
        return turn

    def _refresh_turn(self, turn: SpeakerTurn, candidate: dict[str, Any], timestamp: float) -> None:
        turn.end = max(turn.end, float(timestamp))
        turn.state = candidate["state"]
        turn.probability = candidate["probability"]
        turn.score = candidate["score"]
        turn.reason = candidate["reason"]
        turn.overlap_possible = bool(candidate["overlap_possible"])
        turn.second_speaker = candidate["second_speaker"]
        turn.second_probability = candidate["second_probability"]

    def update(self, payload: dict[str, Any], timestamp: float) -> dict[str, Any]:
        timestamp = max(0.0, float(timestamp))
        candidate = self._candidate(payload)
        changed = False

        if self.current is None:
            self.current = self._new_turn(candidate, 0.0, timestamp)
            self.pending_key = None
            self.pending_count = 0
            self.pending_since = None
            changed = True
        elif self._same_turn(self.current, candidate):
            self._refresh_turn(self.current, candidate, timestamp)
            self.pending_key = None
            self.pending_count = 0
            self.pending_since = None
        else:
            if self.pending_key != candidate["key"]:
                self.pending_key = candidate["key"]
                self.pending_count = 1
                self.pending_since = timestamp
            else:
                self.pending_count += 1
            self.current.end = max(self.current.end, timestamp)
            can_switch = self.pending_count >= self.switch_after
            long_enough = (timestamp - self.current.start) >= self.min_turn_seconds
            if can_switch and long_enough:
                boundary = self.pending_since if self.pending_since is not None else timestamp
                boundary = min(max(float(boundary), self.current.start), timestamp)
                self.current.end = boundary
                self.current = self._new_turn(candidate, boundary, timestamp)
                self.pending_key = None
                self.pending_count = 0
                self.pending_since = None
                changed = True

        return self.to_event(changed=changed)

    def current_payload(self) -> dict[str, Any]:
        if self.current is None:
            return {
                "speaker_turn_id": None,
                "speaker_turn_start": None,
                "speaker_turn_end": None,
                "speaker_turn_speaker": None,
                "speaker_turn_known": False,
                "speaker_turn_overlap_possible": False,
            }
        return {
            "speaker_turn_id": self.current.turn_id,
            "speaker_turn_start": round(self.current.start, 3),
            "speaker_turn_end": round(self.current.end, 3),
            "speaker_turn_speaker": self.current.speaker,
            "speaker_turn_known": self.current.known,
            "speaker_turn_overlap_possible": self.current.overlap_possible,
        }

    def to_event(self, *, changed: bool = False) -> dict[str, Any]:
        recent = self.turns[-self.max_recent_turns :]
        return {
            **self.current_payload(),
            "speaker_turn_changed": bool(changed),
            "speaker_turn": self.current.to_dict() if self.current else None,
            "speaker_turns": [turn.to_dict() for turn in recent],
        }
