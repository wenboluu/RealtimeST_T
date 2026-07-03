#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p data/speaker_profiles artifacts/calibration artifacts/cohorts logs

HOST="${LAB_STT_HOST:-127.0.0.1}"
PORT="${LAB_STT_PORT:-7860}"
DEVICE="${LAB_STT_DEVICE:-cuda}"
MODEL="${LAB_STT_MODEL:-small.en}"
REALTIME_MODEL="${LAB_STT_REALTIME_MODEL:-tiny.en}"
LANGUAGE="${LAB_STT_LANGUAGE:-en}"
COMPUTE_TYPE="${LAB_STT_COMPUTE_TYPE:-float16}"
SPEAKER_MODEL="${LAB_STT_SPEAKER_MODEL:-pyannote/embedding}"
SPEAKER_THRESHOLD="${LAB_STT_SPEAKER_THRESHOLD:-0.3}"
SPEAKER_MARGIN="${LAB_STT_SPEAKER_MARGIN:-0.1}"
SPEAKER_WINDOW_SECONDS="${LAB_STT_SPEAKER_WINDOW_SECONDS:-3.0}"
SPEAKER_MIN_VOICED_SECONDS="${LAB_STT_SPEAKER_MIN_VOICED_SECONDS:-0.8}"
SPEAKER_CALIBRATOR="${LAB_STT_SPEAKER_CALIBRATOR:-artifacts/calibration/speaker_calibrator.joblib}"
SPEAKER_COHORT_BANK="${LAB_STT_SPEAKER_COHORT_BANK:-artifacts/cohorts/cohort_bank.npz}"

args=(
  --host "$HOST"
  --port "$PORT"
  --device "$DEVICE"
  --model "$MODEL"
  --realtime-model "$REALTIME_MODEL"
  --language "$LANGUAGE"
  --compute-type "$COMPUTE_TYPE"
  --speaker-model "$SPEAKER_MODEL"
  --speaker-threshold "$SPEAKER_THRESHOLD"
  --speaker-margin "$SPEAKER_MARGIN"
  --speaker-window-seconds "$SPEAKER_WINDOW_SECONDS"
  --speaker-min-voiced-seconds "$SPEAKER_MIN_VOICED_SECONDS"
  --speaker-calibrator "$SPEAKER_CALIBRATOR"
  --speaker-cohort-bank "$SPEAKER_COHORT_BANK"
)

if [[ -n "${LAB_STT_SPEAKER_PROBABILITY_THRESHOLD:-}" ]]; then
  args+=(--speaker-probability-threshold "$LAB_STT_SPEAKER_PROBABILITY_THRESHOLD")
fi

exec python -m lab_realtime_stt.server "${args[@]}" "$@"
