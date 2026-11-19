#!/bin/sh
set -eu
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
if [ ! -f .venv/.requirements-installed ]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.requirements-installed
fi

if [ -x /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg ]; then
  PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH"
  export PATH
fi

echo "Short Captioner: http://127.0.0.1:${CAPTION_PORT:-8766}"
exec .venv/bin/python app.py
