# Lab Realtime STT

Realtime lab assistant speech prototype built on [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT). It provides a browser UI for low-latency transcription, manual speaker enrollment, known-speaker verification, and simple speaker-change transcript breaks.

Stage 1 speaker handling is deliberately conservative: enrolled voices can be labeled by name, every non-enrolled voice is grouped as `Unknown`, and overlapping speech is only flagged as possible overlap. It is not full source-separating diarization.

## Quick Start

```bash
mamba env create -f environment-lab-realtime-stt.yml
mamba activate lab-realtime-stt
cp .env.example .env
./scripts/run_server.sh
```

Open `http://127.0.0.1:7860`.

If you are running on a remote GPU server, forward the port from your laptop:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-server
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment details, API-key protection, CPU fallback, and smoke checks.

## Configuration

`scripts/run_server.sh` sources `.env` if it exists. Important settings:

```text
LAB_STT_HOST=127.0.0.1
LAB_STT_PORT=7860
LAB_STT_DEVICE=cuda
LAB_STT_MODEL=small.en
LAB_STT_REALTIME_MODEL=tiny.en
LAB_STT_COMPUTE_TYPE=float16
HF_TOKEN=
LAB_STT_API_KEY=
```

Speaker matching defaults:

```text
LAB_STT_SPEAKER_THRESHOLD=0.3
LAB_STT_SPEAKER_MARGIN=0.1
LAB_STT_SPEAKER_WINDOW_SECONDS=3.0
LAB_STT_SPEAKER_MIN_VOICED_SECONDS=0.8
```

Optional calibrated speaker verifier artifacts:

```text
artifacts/calibration/speaker_calibrator.joblib
artifacts/cohorts/cohort_bank.npz
```

If those files are absent, the server falls back to cosine speaker matching. A newly enrolled speaker does not require retraining; their profile embedding is compared against live audio through the configured matcher.

## Speaker Enrollment

See [PROFILE_RECORDING_SCRIPT.md](PROFILE_RECORDING_SCRIPT.md) for a 25-40 second script people can read while recording their voice profile.

Use the web UI to record or upload 20-30 seconds for each scientist. Profiles are local deployment data stored under:

```text
data/speaker_profiles/
```

Profiles are intentionally ignored by git.

## Run Checks

```bash
python -m pytest -q
python scripts/smoke_check.py --url http://127.0.0.1:7860
```

To stream a file into a running server:

```bash
python scripts/stream_file.py path/to/audio.wav --url ws://127.0.0.1:7860/ws/transcribe --realtime
```

## Evaluation

For a first objective check of enrolled-speaker matching, use LibriSpeech `test-clean`. It is not a perfect lab-noise benchmark, but it gives multiple known speakers with held-out utterances and background speakers.

```bash
python scripts/eval_focus_librispeech.py \
  --download \
  --subset test-clean \
  --enroll-speakers 6 \
  --background-speakers 4 \
  --eval-utterances 6 \
  --overwrite
```

Stage 1 streaming-style speaker-turn evaluation:

```bash
python scripts/eval_stage1_diarization_librispeech.py \
  --subset test-clean \
  --dataset-root /path/to/librispeech \
  --known-speakers 3 \
  --unknown-speakers 1 \
  --rounds 2 \
  --eval-utterances 2 \
  --enrollment-seconds 10 \
  --hop-seconds 0.75 \
  --window-seconds 3.0 \
  --augmentations clean,noise,reverb
```

The latency number in this evaluator is only the speaker-matching stage. Browser capture, WebSocket transport, and ASR decoding are measured live in the WebUI latency gauge.

## Calibration Training

Large speech datasets and calibration artifacts should live outside git. The LibriSpeech calibration script builds same-speaker and different-speaker verification trials from pyannote embeddings, applies clean/noise/reverb/bandpass/lab-style augmentation, trains a logistic regression calibrator, and saves a cohort bank for score normalization.

Smoke run:

```bash
python scripts/train_speaker_calibration_librispeech.py \
  --subset test-clean \
  --dataset-root /path/to/librispeech \
  --train-speakers 4 \
  --eval-speakers 2 \
  --cohort-speakers 4 \
  --min-utterances 6 \
  --enrollment-seconds 8 \
  --eval-utterances 2 \
  --cohort-utterances-per-speaker 1 \
  --negative-per-positive 2 \
  --augmentations clean,noise,reverb \
  --output-dir artifacts/calibration/smoke \
  --cohort-output artifacts/cohorts/cohort_bank_smoke.npz
```
