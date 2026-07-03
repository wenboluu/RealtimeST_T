from __future__ import annotations

import numpy as np

from lab_realtime_stt.speaker import (
    PyannoteEmbeddingBackend,
    SpeakerMatcher,
    SpeakerProfile,
    SpeakerProfileStore,
    normalize_embedding,
    split_voiced_chunks,
    voiced_seconds,
)


class FakeEmbedder(PyannoteEmbeddingBackend):
    def __init__(self, vector):
        super().__init__(model_name="fake", device="cpu")
        self.vector = normalize_embedding(np.asarray(vector, dtype=np.float32))

    def embed(self, audio, sample_rate=16000):
        return self.vector


def test_voiced_seconds_detects_synthetic_tone():
    t = np.arange(16000, dtype=np.float32) / 16000
    audio = 0.05 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    assert voiced_seconds(audio) > 0.8


def test_split_voiced_chunks_filters_silence():
    silence = np.zeros(16000, dtype=np.float32)
    tone = 0.05 * np.sin(2 * np.pi * 220 * np.arange(48000, dtype=np.float32) / 16000).astype(np.float32)
    chunks = split_voiced_chunks(np.concatenate([silence, tone, silence]))
    assert chunks
    assert all(chunk.size >= int(1.5 * 16000) for chunk in chunks)


def test_speaker_match_threshold_and_margin(tmp_path):
    store = SpeakerProfileStore(tmp_path, FakeEmbedder([1, 0, 0]))
    profile = SpeakerProfile(
        speaker_id="david",
        name="David",
        embedding=normalize_embedding(np.asarray([1, 0, 0], dtype=np.float32)).tolist(),
        num_chunks=1,
        voiced_seconds=3.0,
        model="fake",
        created_at=0.0,
    )
    (tmp_path / "david.json").write_text(__import__("json").dumps(profile.__dict__))
    matcher = SpeakerMatcher(store, threshold=0.5, margin=0.08, min_voiced_seconds=0.0)
    audio = np.ones(16000, dtype=np.float32) * 0.02
    match = matcher.match_audio(audio)
    assert match.speaker_id == "david"
    assert match.state == "tentative"
    assert match.score == 1.0
