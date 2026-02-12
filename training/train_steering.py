#!/usr/bin/env python3
"""
Train a simple vision model to predict steering (-1..1) from images.

Expected dataset structure (created by robocar.py):
  data/
    run_YYYYmmdd_HHMMSS/
      images/
        img_*.jpg
      log.jsonl

Usage (on your PC with GPU if possible):
  python -m venv venv
  source venv/bin/activate   # Windows: venv\Scripts\activate
  pip install tensorflow opencv-python numpy

  python train_steering.py --data_root ../data --epochs 15

Outputs:
  models/model.h5
  models/model.tflite
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def pick_latest_run(data_root: str) -> str:
    runs = sorted(glob.glob(os.path.join(data_root, "run_*")))
    if not runs:
        raise SystemExit(f"No runs found under: {data_root}")
    return runs[-1]


def load_dataset(run_dir: str, img_w: int, img_h: int, min_abs_throttle: float = 0.05):
    log_path = os.path.join(run_dir, "log.jsonl")
    images_dir = os.path.join(run_dir, "images")

    if not os.path.exists(log_path):
        raise SystemExit(f"Missing log.jsonl: {log_path}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Missing images/: {images_dir}")

    xs, ys = [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            throttle = float(r.get("throttle", 0.0))
            steering = float(r.get("steering", 0.0))
            if abs(throttle) < min_abs_throttle:
                continue  # skip idle frames

            img_rel = r["image"]
            img_path = os.path.join(run_dir, img_rel)

            if cv2 is None:
                raise SystemExit("OpenCV is required for training script (pip install opencv-python).")

            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (img_w, img_h))
            xs.append(rgb)
            ys.append(steering)

    if not xs:
        raise SystemExit("No training samples loaded (maybe you didn't record enough frames?).")

    x = np.asarray(xs, dtype=np.float32) / 255.0
    y = np.asarray(ys, dtype=np.float32)
    return x, y


def build_model(h: int, w: int) -> keras.Model:
    inp = keras.Input(shape=(h, w, 3), name="image")
    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu")(inp)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(10, activation="relu")(x)
    out = layers.Dense(1, activation="tanh", name="steering")(x)  # [-1,1]
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def to_tflite(model: keras.Model, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data", help="Path to data/ (containing run_... folders)")
    ap.add_argument("--run_dir", default="", help="Specific run folder (overrides --data_root latest)")
    ap.add_argument("--img_w", type=int, default=160)
    ap.add_argument("--img_h", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min_abs_throttle", type=float, default=0.05)
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    run_dir = args.run_dir or pick_latest_run(args.data_root)
    print(f"Using run: {run_dir}")

    x, y = load_dataset(run_dir, args.img_w, args.img_h, args.min_abs_throttle)
    print(f"Loaded: {x.shape[0]} samples")

    # Shuffle/split
    n = x.shape[0]
    idx = np.random.permutation(n)
    x, y = x[idx], y[idx]
    split = int(n * 0.9)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = build_model(args.img_h, args.img_w)
    model.summary()

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = str(out_dir / "model.h5")
    tflite_path = str(out_dir / "model.tflite")

    model.save(h5_path)
    to_tflite(model, tflite_path)

    print("Saved:")
    print(" -", h5_path)
    print(" -", tflite_path)
    print("Copy model.tflite to your Pi under robocar/models/model.tflite")


if __name__ == "__main__":
    main()
