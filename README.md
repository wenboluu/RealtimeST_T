# Lab Realtime STT

Realtime lab assistant speech prototype built on [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) with manual speaker enrollment.

This replaces the earlier WhisperX realtime prototype for live use. The live path focuses on low-latency partial/final transcription plus known-speaker verification. It does not do word alignment or trigger-word enrollment.

Stage 1 speaker handling is intentionally simple: enrolled voices can be labeled by name, every non-enrolled voice is grouped as `Unknown`, and the UI breaks the transcript whenever the active speaker changes. This is diarization-lite rather than full overlap-separating diarization; overlapped speech is flagged as possible overlap, not separated into multiple transcripts.

## Setup

```bash
cd /home/wenbolu/projects/lab-realtime-stt
mamba env create -f environment-lab-realtime-stt.yml
mamba activate lab-realtime-stt
```

If the environment already exists, update it:

```bash
mamba env update -n lab-realtime-stt -f environment-lab-realtime-stt.yml
```

## Run

```bash
mamba activate lab-realtime-stt
python -m lab_realtime_stt.server \
  --host 127.0.0.1 \
  --port 7860 \
  --device cuda \
  --model small.en \
  --realtime-model tiny.en \
  --compute-type float16
```

From your laptop over SSH:

```bash
ssh -N -L 7860:127.0.0.1:7860 wenbolu@TRY-65691-3-gpu01
```

Open `http://127.0.0.1:7860` locally.

## Speaker Enrollment

See `PROFILE_RECORDING_SCRIPT.md` for a recommended 25-40 second script people can read while recording their voice profile.


Use the web UI to record or upload 20-30 seconds for each scientist. The server stores local profiles under:

```text
data/speaker_profiles/
```

Matching defaults:

```text
threshold = 0.3
margin = 0.1
speaker_window_seconds = 3.0
speaker_min_voiced_seconds = 0.8
stable_after = 2 matching decisions
```

The speaker label is only a verification/routing signal. If audio is noisy, too short, or likely overlapping, it should remain `unknown` rather than forcing a wrong speaker.

Speaker-turn tracking defaults:

```text
speaker_turn_switch_after = 2 stable speaker decisions
speaker_turn_min_seconds = 0.8
speaker_overlap_probability = 0.35
speaker_overlap_margin = 0.25
```

Use `--no-speaker-turns` to disable transcript breaks, or tune `--speaker-turn-switch-after`, `--speaker-turn-min-seconds`, `--speaker-overlap-probability`, and `--speaker-overlap-margin` for a specific room.

The live server automatically uses the LibriSpeech starter logistic calibrator when these files exist:

```text
/data/wenbolu/checkpoints/lab-realtime-stt/calibration/librispeech_starter/speaker_calibrator.joblib
/data/wenbolu/checkpoints/lab-realtime-stt/cohorts/cohort_bank_librispeech_starter.npz
```

With the calibrator enabled, the WebUI shows `P(same speaker)` for each enrolled profile. A newly enrolled speaker does not require retraining; their profile embedding is compared against live audio through the same generic same/different-speaker calibrator. Use `--speaker-calibrator`, `--speaker-cohort-bank`, and `--speaker-probability-threshold` to override this behavior.

## Stream a File for Testing

```bash
python scripts/stream_file.py data/audio_sample.mp3 --url ws://127.0.0.1:7860/ws/transcribe --realtime
```

Use this after copying the old sample into this project:

```bash
cp /home/wenbolu/projects/whisperX/data/audio_sample.mp3 /home/wenbolu/projects/lab-realtime-stt/data/audio_sample.mp3
```


## Speaker Focus Evaluation Dataset

For a first objective check of enrolled-speaker matching, use LibriSpeech `test-clean`. It is not a perfect lab-noise benchmark, but it gives multiple known speakers with held-out utterances and background speakers.

```bash
cd /home/wenbolu/projects/lab-realtime-stt
mamba activate lab-realtime-stt
python scripts/eval_focus_librispeech.py \
  --download \
  --subset test-clean \
  --enroll-speakers 6 \
  --background-speakers 4 \
  --eval-utterances 6 \
  --overwrite
```

The script saves a JSON report to:

```text
data/eval/focus_librispeech_report.json
```

By default, live speaker focus uses a 3.0 second rolling window with `speaker_threshold=0.3`, `speaker_margin=0.1`, and `speaker_min_voiced_seconds=0.8`, based on the LibriSpeech clean-speech sweep. Use `--window-seconds 0` in evaluation to score full utterances.

For a grid sweep, pass comma-separated values:

```bash
python scripts/eval_focus_librispeech.py \
  --download \
  --subset test-clean \
  --enroll-speakers 6 \
  --background-speakers 4 \
  --eval-utterances 6 \
  --thresholds 0.2,0.25,0.3,0.35,0.4,0.5 \
  --margins 0.06,0.08,0.1,0.15,0.2 \
  --window-seconds-list 1.2,1.6,2.0,3.0,0 \
  --min-match-voiced-seconds-list 0.4,0.6,0.8,1.0 \
  --overwrite
```

Metrics include positive top-1 accuracy, positive unknown rate, positive false-speaker rate, and background false-accept rate. Sweep summaries are written under `sweeps.results`, with the script's conservative zero-false-accept recommendation under `sweeps.recommended`.

## Stage 1 Speaker-Turn Evaluation

To test the live speaker-turn path without microphone or browser effects, run the LibriSpeech streaming-style evaluator. It enrolls a few known speakers, reserves at least one speaker as `Unknown`, streams synthetic conversations through the same speaker matcher and turn tracker, and reports both robustness and per-window matching latency.

```bash
python scripts/eval_stage1_diarization_librispeech.py \
  --subset test-clean \
  --dataset-root /data/wenbolu/datasets/lab-realtime-stt/librispeech \
  --known-speakers 3 \
  --unknown-speakers 1 \
  --rounds 2 \
  --eval-utterances 2 \
  --enrollment-seconds 10 \
  --hop-seconds 0.75 \
  --window-seconds 3.0 \
  --augmentations clean,noise,reverb \
  --output /data/wenbolu/outputs/lab-realtime-stt/reports/stage1_diarization_librispeech.json
```

The latency number here is only the speaker-matching stage. Browser capture, WebSocket transport, and ASR decoding are measured live in the WebUI latency gauge.

## Speaker Calibration Training

Large speech datasets and calibration artifacts live outside the repo:

```text
/data/wenbolu/datasets/lab-realtime-stt/
  librispeech/
  voxceleb/
  cnceleb/
  cohorts/

/data/wenbolu/checkpoints/lab-realtime-stt/
  calibration/
  cohorts/

/data/wenbolu/outputs/lab-realtime-stt/
  reports/
```

The LibriSpeech calibration script builds same-speaker and different-speaker verification trials from pyannote embeddings, applies clean/noise/reverb/bandpass/lab-style augmentation, trains a logistic regression calibrator, and saves a cohort bank for score normalization.

Smoke test on the already-copied `test-clean` subset:

```bash
python scripts/train_speaker_calibration_librispeech.py \
  --subset test-clean \
  --dataset-root /data/wenbolu/datasets/lab-realtime-stt/librispeech \
  --train-speakers 4 \
  --eval-speakers 2 \
  --cohort-speakers 4 \
  --min-utterances 6 \
  --enrollment-seconds 8 \
  --eval-utterances 2 \
  --cohort-utterances-per-speaker 1 \
  --negative-per-positive 2 \
  --augmentations clean,noise,reverb \
  --output-dir /data/wenbolu/checkpoints/lab-realtime-stt/calibration/smoke \
  --cohort-output /data/wenbolu/checkpoints/lab-realtime-stt/cohorts/cohort_bank_smoke.npz \
  --report /data/wenbolu/outputs/lab-realtime-stt/reports/librispeech_calibration_smoke.json
```

First larger LibriSpeech run:

```bash
python scripts/train_speaker_calibration_librispeech.py \
  --download \
  --subset train-clean-100 \
  --dataset-root /data/wenbolu/datasets/lab-realtime-stt/librispeech \
  --train-speakers 80 \
  --eval-speakers 20 \
  --cohort-speakers 100 \
  --augmentations clean,noise,reverb,bandpass,lab
```

Default outputs:

```text
/data/wenbolu/checkpoints/lab-realtime-stt/calibration/speaker_calibrator.joblib
/data/wenbolu/checkpoints/lab-realtime-stt/cohorts/cohort_bank.npz
/data/wenbolu/outputs/lab-realtime-stt/reports/librispeech_calibration_report.json
```

## Cleanup Old Prototype

After this project passes smoke tests, the old WhisperX realtime prototype can be removed from the WhisperX repo:

```text
scripts/lab_stt_web.py
scripts/benchmark_realtime.py
LAB_STT_WEB.md
environment-lab-stt.yml
```

Then remove the old env only after confirming no active work depends on it:

```bash
mamba env remove -n whisperx-lab
```
