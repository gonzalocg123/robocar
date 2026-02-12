#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "[robocar] If this is your first run on Raspberry Pi OS Bookworm, install once:"
echo "  sudo apt update"
echo "  sudo apt install -y python3-venv python3-picamera2 python3-opencv"
echo

# Recommended pin factory on Bookworm / Pi 5
export GPIOZERO_PIN_FACTORY="${GPIOZERO_PIN_FACTORY:-lgpio}"

# Create venv (with system-site-packages so apt-installed picamera2/opencv are visible)
if [ ! -d "venv" ]; then
  echo "[robocar] Creating virtual environment (venv)..."
  python3 -m venv --system-site-packages venv
fi

source venv/bin/activate

echo "[robocar] Upgrading pip..."
python -m pip install --upgrade pip

echo "[robocar] Installing Python dependencies..."
# Note: opencv + picamera2 are expected to come from apt on Bookworm.
pip install -r requirements.txt || true

echo "[robocar] Starting server on http://0.0.0.0:5000"
python3 robocar.py
