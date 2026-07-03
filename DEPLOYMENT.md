# Deployment Guide

This project is a single FastAPI process that serves the browser UI, accepts microphone PCM over WebSocket, runs RealtimeSTT, and optionally verifies enrolled speakers with pyannote embeddings.

## 1. Clone and create the environment

```bash
git clone git@github.com:wenboluu/RealtimeST_T.git
cd RealtimeST_T
mamba env create -f environment-lab-realtime-stt.yml
mamba activate lab-realtime-stt
```

For an existing environment:

```bash
mamba env update -n lab-realtime-stt -f environment-lab-realtime-stt.yml
```

## 2. Configure local secrets and artifacts

```bash
cp .env.example .env
```

Set `HF_TOKEN` in `.env` if this machine has not already cached `pyannote/embedding`. The Hugging Face account must accept the model terms before first download.

If the app is reachable by anyone other than you, set `LAB_STT_API_KEY` in `.env`. Browser users can store the key locally before opening the UI:

```js
localStorage.setItem("labSttApiKey", "your-key")
```

Optional speaker-calibration artifacts are local deployment data and are not tracked by git:

```text
artifacts/calibration/speaker_calibrator.joblib
artifacts/cohorts/cohort_bank.npz
```

If these files are absent, the server still starts and falls back to cosine speaker matching.

## 3. Start the server

```bash
mamba activate lab-realtime-stt
./scripts/run_server.sh
```

The default bind is `127.0.0.1:7860`. CPU-only deployments can set `LAB_STT_DEVICE=cpu` and `LAB_STT_COMPUTE_TYPE=int8`; expect higher latency. For an SSH tunnel from a laptop:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-server
```

Open `http://127.0.0.1:7860` on the laptop.

## 4. Smoke check

In another shell:

```bash
mamba activate lab-realtime-stt
python scripts/smoke_check.py --url http://127.0.0.1:7860
```

This checks `/api/health`, the static UI, and a WebSocket `session.ready` handshake.

## 5. Production notes

- Keep `LAB_STT_HOST=127.0.0.1` when exposing the app through SSH tunneling or a reverse proxy.
- Only bind to `0.0.0.0` behind network controls and set `LAB_STT_API_KEY`; the UI can enroll and delete local speaker profiles.
- Keep `.env`, `data/speaker_profiles/*`, audio samples, and calibration artifacts out of git.
- The first request that needs RealtimeSTT or pyannote may download/cache models. Warm the server before a demo by opening the UI and starting a short test session.
- Browser microphone APIs require HTTPS unless the page is loaded from localhost.

## Useful commands

```bash
make test
make smoke
make clean
```
