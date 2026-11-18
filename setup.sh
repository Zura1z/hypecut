#!/usr/bin/env bash
# One-time setup for macOS and Linux. Creates .venv and installs everything HypeCut needs.
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "ffmpeg is missing. Install it first:"
  echo "  macOS:          brew install ffmpeg"
  echo "  Ubuntu/Debian:  sudo apt install ffmpeg"
  exit 1
fi
PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Python 3.10 or newer is required (found: $("$PY" --version 2>&1))."
  exit 1
fi

"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

# Face models (already in the repo; downloaded only if missing)
ZOO=https://github.com/opencv/opencv_zoo/raw/main/models
mkdir -p models
get() { [ -s "models/$2" ] || curl -fL --retry 3 -o "models/$2" "$ZOO/$1/$2"; }
get face_detection_yunet face_detection_yunet_2023mar.onnx
get facial_expression_recognition facial_expression_recognition_mobilefacenet_2022july.onnx

[ -f .env ] || cp .env.example .env
echo
echo "Ready. Try:  .venv/bin/python clipper.py \"<stream url or video file>\""
echo "For captions, put a Gemini key in .env and add --captions"
