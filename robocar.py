import os
import io
import time
import json
import math
import threading
import datetime
import signal
from dataclasses import dataclass
from typing import Optional

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

# -----------------------------
# Optional dependencies / hardware backends
# -----------------------------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    cv2 = None  # type: ignore

try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None  # type: ignore

GPIO_AVAILABLE = True
_PIN_FACTORY = None
try:
    from gpiozero import PWMOutputDevice, DigitalOutputDevice  # type: ignore
    try:
        from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore
        _PIN_FACTORY = LGPIOFactory()
    except Exception:
        _PIN_FACTORY = None
except Exception:
    GPIO_AVAILABLE = False
    PWMOutputDevice = None  # type: ignore
    DigitalOutputDevice = None  # type: ignore


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


# -----------------------------
# Pins (BCM) - L298N
# -----------------------------
LEFT_MOTOR_IN1 = _env_int("LEFT_MOTOR_IN1", 17)
LEFT_MOTOR_IN2 = _env_int("LEFT_MOTOR_IN2", 27)
LEFT_MOTOR_ENA = _env_int("LEFT_MOTOR_ENA", 18)   # PWM

RIGHT_MOTOR_IN3 = _env_int("RIGHT_MOTOR_IN3", 22)
RIGHT_MOTOR_IN4 = _env_int("RIGHT_MOTOR_IN4", 23)
RIGHT_MOTOR_ENB = _env_int("RIGHT_MOTOR_ENB", 25)  # PWM

# Camera / streaming
STREAM_W = _env_int("STREAM_W", 640)
STREAM_H = _env_int("STREAM_H", 480)

# Recording (training dataset)
RECORD_W = _env_int("RECORD_W", 160)
RECORD_H = _env_int("RECORD_H", 120)
RECORD_HZ = _env_float("RECORD_HZ", 10.0)

# Safety
CONTROL_TIMEOUT_S = _env_float("CONTROL_TIMEOUT_S", 0.60)

# Deadzones
THROTTLE_DEADZONE = _env_float("THROTTLE_DEADZONE", 0.05)
STEERING_DEADZONE = _env_float("STEERING_DEADZONE", 0.01)

# "Feel" tuning (important)
STEER_WHEN_MOVING_REF = _env_float("STEER_WHEN_MOVING_REF", 0.20)

MAX_STEER = _env_float("MAX_STEER", 0.70)                # hard cap on steering effect
STEER_AT_LOW_SPEED = _env_float("STEER_AT_LOW_SPEED", 0.95)
STEER_AT_HIGH_SPEED = _env_float("STEER_AT_HIGH_SPEED", 0.80)

# Precision curves (Exponential)
# 1.0 = linear, 2.0 = quadratic (more precision at low stick), 3.0 = cubic
STEERING_EXPONENT = _env_float("STEERING_EXPONENT", 2.0)
THROTTLE_EXPONENT = _env_float("THROTTLE_EXPONENT", 2.0)

# Control polarity (1.0 = normal, -1.0 = inverted)
# User requested inversion, so defaulting to -1.0
THROTTLE_POLARITY = _env_float("THROTTLE_POLARITY", -1.0)
STEERING_POLARITY = _env_float("STEERING_POLARITY", 1.0)

# Pivot turn (optional)
ALLOW_PIVOT_TURN = os.getenv("ALLOW_PIVOT_TURN", "1") not in ("0", "false", "False")
PIVOT_STEER_THRESHOLD = _env_float("PIVOT_STEER_THRESHOLD", 0.20)
PIVOT_POWER = _env_float("PIVOT_POWER", 0.28)

# Motor stiction compensation (very important with L298N)
MIN_PWM = _env_float("MIN_PWM", 0.14)

# Autonomous
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.tflite"))
AUTO_THROTTLE = _env_float("AUTO_THROTTLE", 0.25)
AUTO_STEER_SMOOTH = _env_float("AUTO_STEER_SMOOTH", 0.30)

# Manual smoothing (EMA) — lower => smoother
MANUAL_STEER_SMOOTH = _env_float("MANUAL_STEER_SMOOTH", 0.30)  # Increased for snappier response
MANUAL_THROTTLE_SMOOTH = _env_float("MANUAL_THROTTLE_SMOOTH", 0.30)

# Data root
DATA_ROOT = os.getenv("DATA_DIR", "data")

# Debug
DEBUG_DRIVE = os.getenv("DEBUG_DRIVE", "0") in ("1", "true", "True")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _apply_deadzone(x: float, dz: float) -> float:
    return 0.0 if abs(x) < dz else x


def _apply_exponential(x: float, exp: float) -> float:
    return math.copysign(abs(x) ** exp, x)


def _apply_min_pwm(sp: float, min_pwm: float) -> float:
    """Overcome stiction without killing differential steering.
    Only bump when the commanded wheel speed is small-but-nontrivial.
    """
    if min_pwm <= 0:
        return sp
    if sp == 0.0:
        return 0.0
    a = abs(sp)
    # do NOT bump tiny corrections; only bump if wheel is meant to actually move
    if 0.0 < a < min_pwm:
        return (min_pwm if sp > 0 else -min_pwm)
    return sp


# -----------------------------
# Global state
# -----------------------------
state_lock = threading.Lock()
state = {
    "steering": 0.0,
    "throttle": 0.0,
    "mode": "manual",
    "recording": False,
    "speed_limit": _env_float("SPEED_LIMIT", 1.0),
    "run_id": None,
}
_last_control_ts = 0.0


# -----------------------------
# Motor controller
# -----------------------------
class MotorController:
    def __init__(self) -> None:
        self._enabled = False
        self._left_in1: Optional[DigitalOutputDevice] = None
        self._left_in2: Optional[DigitalOutputDevice] = None
        self._left_en: Optional[PWMOutputDevice] = None
        self._right_in3: Optional[DigitalOutputDevice] = None
        self._right_in4: Optional[DigitalOutputDevice] = None
        self._right_en: Optional[PWMOutputDevice] = None

        if not GPIO_AVAILABLE:
            print("GPIO not available -> simulation mode.")
            return

        try:
            kwargs = {}
            if _PIN_FACTORY is not None:
                kwargs["pin_factory"] = _PIN_FACTORY

            self._left_in1 = DigitalOutputDevice(LEFT_MOTOR_IN1, initial_value=False, **kwargs)
            self._left_in2 = DigitalOutputDevice(LEFT_MOTOR_IN2, initial_value=False, **kwargs)
            self._right_in3 = DigitalOutputDevice(RIGHT_MOTOR_IN3, initial_value=False, **kwargs)
            self._right_in4 = DigitalOutputDevice(RIGHT_MOTOR_IN4, initial_value=False, **kwargs)

            self._left_en = PWMOutputDevice(LEFT_MOTOR_ENA, initial_value=0.0, frequency=500, **kwargs)
            self._right_en = PWMOutputDevice(RIGHT_MOTOR_ENB, initial_value=0.0, frequency=500, **kwargs)

            self._force_stop_startup()
            self._enabled = True
        except Exception as e:
            print(f"MotorController init failed ({e}). Motors disabled.")
            self._enabled = False

    def _force_stop_startup(self) -> None:
        try:
            if self._left_en:
                self._left_en.value = 0.0
            if self._right_en:
                self._right_en.value = 0.0
            if self._left_in1:
                self._left_in1.off()
            if self._left_in2:
                self._left_in2.off()
            if self._right_in3:
                self._right_in3.off()
            if self._right_in4:
                self._right_in4.off()
        except Exception:
            pass

    def stop(self) -> None:
        if not self._enabled:
            return
        self.set_motors(0.0, 0.0)

    def set_motors(self, left_speed: float, right_speed: float) -> None:
        if not self._enabled:
            return

        left_speed = _clamp(left_speed, -1.0, 1.0)
        right_speed = _clamp(right_speed, -1.0, 1.0)

        def _apply(in_a: DigitalOutputDevice, in_b: DigitalOutputDevice, en: PWMOutputDevice, sp: float) -> None:
            sp = _clamp(sp, -1.0, 1.0)

            if sp == 0.0:
                en.value = 0.0
                in_a.off()
                in_b.off()
                return

            en.value = 0.0
            if sp > 0.0:
                in_a.on()
                in_b.off()
                en.value = sp
            else:
                in_a.off()
                in_b.on()
                en.value = abs(sp)

        _apply(self._left_in1, self._left_in2, self._left_en, left_speed)     # type: ignore
        _apply(self._right_in3, self._right_in4, self._right_en, right_speed) # type: ignore

    def drive(self, throttle: float, steering: float, speed_limit: float) -> None:
        # Polarity inversion (based on user request)
        throttle *= THROTTLE_POLARITY
        steering *= STEERING_POLARITY

        throttle = _apply_deadzone(_clamp(throttle, -1.0, 1.0), THROTTLE_DEADZONE)
        steering = _apply_deadzone(_clamp(steering, -1.0, 1.0), STEERING_DEADZONE)

        # Apply exponential curve for precision
        throttle = _apply_exponential(throttle, THROTTLE_EXPONENT)
        steering = _apply_exponential(steering, STEERING_EXPONENT)

        speed_limit = _clamp(speed_limit, 0.0, 1.0)

        # Gentle steering cap (key to avoid "brusco")
        steering = _clamp(steering, -MAX_STEER, MAX_STEER)

        # Pivot turn (optional)
        if throttle == 0.0:
            if ALLOW_PIVOT_TURN and abs(steering) >= PIVOT_STEER_THRESHOLD:
                # scale pivot power with steering magnitude
                p = _clamp(PIVOT_POWER * (abs(steering) / max(MAX_STEER, 1e-6)), 0.0, 1.0)
                left = p if steering > 0 else -p
                right = -left
                left = _apply_min_pwm(left * speed_limit, MIN_PWM)
                right = _apply_min_pwm(right * speed_limit, MIN_PWM)
                if DEBUG_DRIVE:
                    print(f"[pivot] steer={steering:.2f} -> L={left:.2f} R={right:.2f}")
                self.set_motors(left, right)
                return

            self.set_motors(0.0, 0.0)
            return

        # Progressive steering with speed (but still usable)
        moving = min(1.0, abs(throttle) / max(STEER_WHEN_MOVING_REF, 1e-6))
        scale = STEER_AT_LOW_SPEED + (STEER_AT_HIGH_SPEED - STEER_AT_LOW_SPEED) * moving
        steering = steering * _clamp(scale, 0.2, 1.0)

        left = throttle + steering
        right = throttle - steering

        m = max(abs(left), abs(right), 1e-9)
        if m > 1.0:
            left /= m
            right /= m

        left *= speed_limit
        right *= speed_limit

        # Stiction compensation (so the slow wheel still moves -> the car actually turns)
        left = _apply_min_pwm(left, MIN_PWM)
        right = _apply_min_pwm(right, MIN_PWM)

        if DEBUG_DRIVE:
            print(f"[drive] thr={throttle:.2f} steer={steering:.2f} -> L={left:.2f} R={right:.2f}")

        self.set_motors(left, right)


motors = MotorController()
try:
    motors.stop()
except Exception:
    pass


# -----------------------------
# Recorder
# -----------------------------
class DataRecorder:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.run_dir: Optional[str] = None
        self.images_dir: Optional[str] = None
        self.log_path: Optional[str] = None
        self._last_save_ts = 0.0
        os.makedirs(self.root_dir, exist_ok=True)

    def start(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{ts}"
        self.run_dir = os.path.join(self.root_dir, run_id)
        self.images_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "log.jsonl")
        self._last_save_ts = 0.0
        return run_id

    def stop(self) -> None:
        self.run_dir = None
        self.images_dir = None
        self.log_path = None
        self._last_save_ts = 0.0

    def can_save_now(self) -> bool:
        if RECORD_HZ <= 0:
            return False
        now = time.time()
        period = 1.0 / RECORD_HZ
        return (now - self._last_save_ts) >= period

    def mark_saved(self) -> None:
        self._last_save_ts = time.time()

    def save(self, frame_bgr, steering: float, throttle: float) -> None:
        if not self.images_dir or not self.log_path:
            return
        if not self.can_save_now():
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(self.images_dir, filename)

        if OPENCV_AVAILABLE and cv2 is not None:
            small = cv2.resize(frame_bgr, (RECORD_W, RECORD_H))
            cv2.imwrite(filepath, small)
        elif PIL_AVAILABLE:
            img = Image.fromarray(frame_bgr[..., ::-1])
            img = img.resize((RECORD_W, RECORD_H))
            img.save(filepath, format="JPEG", quality=90)
        else:
            return

        log_entry = {
            "timestamp": timestamp,
            "image": os.path.join("images", filename),
            "steering": float(steering),
            "throttle": float(throttle),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        self.mark_saved()


recorder = DataRecorder(DATA_ROOT)


# -----------------------------
# Autopilot (TFLite)
# -----------------------------
@dataclass
class Autopilot:
    model_path: str
    _interp: Optional[object] = None
    _input_index: Optional[int] = None
    _output_index: Optional[int] = None
    _ema_steer: float = 0.0

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"[autopilot] No model at {self.model_path} (autonomous inert).")
            return False

        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except Exception:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
            except Exception as e:
                print(f"[autopilot] No TFLite interpreter available: {e}")
                return False

        try:
            interp = Interpreter(model_path=self.model_path)
            interp.allocate_tensors()
            in_details = interp.get_input_details()
            out_details = interp.get_output_details()
            self._interp = interp
            self._input_index = int(in_details[0]["index"])
            self._output_index = int(out_details[0]["index"])
            print(f"[autopilot] Loaded model: {self.model_path}")
            return True
        except Exception as e:
            print(f"[autopilot] Failed to load model: {e}")
            return False

    def _preprocess(self, frame_bgr) -> Optional["object"]:
        if not OPENCV_AVAILABLE or cv2 is None:
            if not PIL_AVAILABLE:
                return None
            img = Image.fromarray(frame_bgr[..., ::-1])
            img = img.resize((RECORD_W, RECORD_H))
            import numpy as np
            arr = np.asarray(img).astype("float32") / 255.0
            return arr[None, ...]
        else:
            import numpy as np
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (RECORD_W, RECORD_H))
            arr = rgb.astype("float32") / 255.0
            return arr[None, ...]

    def predict_steering(self, frame_bgr) -> Optional[float]:
        if self._interp is None:
            if not self.load():
                return None

        x = self._preprocess(frame_bgr)
        if x is None:
            return None

        try:
            self._interp.set_tensor(self._input_index, x)  # type: ignore
            self._interp.invoke()  # type: ignore
            y = self._interp.get_tensor(self._output_index)  # type: ignore
            steer = float(y.reshape(-1)[0])
            steer = _clamp(steer, -1.0, 1.0)
            a = _clamp(AUTO_STEER_SMOOTH, 0.0, 1.0)
            self._ema_steer = (1 - a) * self._ema_steer + a * steer
            return self._ema_steer
        except Exception as e:
            print(f"[autopilot] inference error: {e}")
            return None


autopilot = Autopilot(MODEL_PATH)


# -----------------------------
# Camera
# -----------------------------
class Camera:
    def __init__(self) -> None:
        self.use_picamera2 = False
        self.picam2 = None
        self.video = None

        if PICAMERA2_AVAILABLE and Picamera2 is not None:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (STREAM_W, STREAM_H)}))
                self.picam2.start()
                self.use_picamera2 = True
                print("Camera: Using Picamera2")
            except Exception as e:
                print(f"Camera: Picamera2 init failed: {e}")
                self.picam2 = None
                self.use_picamera2 = False

        if not self.use_picamera2:
            if OPENCV_AVAILABLE and cv2 is not None:
                self.video = cv2.VideoCapture(0)
                if not self.video.isOpened():
                    print("Camera: Could not open video device 0")
            else:
                print("Camera: No Picamera2 and no OpenCV -> no video.")

    def _encode_jpeg(self, frame_bgr) -> Optional[bytes]:
        if OPENCV_AVAILABLE and cv2 is not None:
            ok, jpg = cv2.imencode(".jpg", frame_bgr)
            return jpg.tobytes() if ok else None
        if PIL_AVAILABLE:
            img = Image.fromarray(frame_bgr[..., ::-1])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        return None

    def get_frame(self) -> Optional[bytes]:
        frame = None

        if self.use_picamera2 and self.picam2 is not None:
            try:
                arr = self.picam2.capture_array()
                if OPENCV_AVAILABLE and cv2 is not None:
                    frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    frame = arr[..., ::-1]
            except Exception:
                return None
        elif self.video is not None and OPENCV_AVAILABLE and cv2 is not None:
            ok, f = self.video.read()
            if ok:
                frame = f

        if frame is None:
            return None

        with state_lock:
            mode = state["mode"]
            recording = state["recording"]
            steering = float(state["steering"])
            throttle = float(state["throttle"])
            speed_limit = float(state["speed_limit"])

        if recording:
            recorder.save(frame, steering=steering, throttle=throttle)

        if mode == "autonomous":
            pred = autopilot.predict_steering(frame)
            if pred is not None:
                with state_lock:
                    state["steering"] = float(pred)
                    state["throttle"] = float(AUTO_THROTTLE)
                    steering = float(state["steering"])
                    throttle = float(state["throttle"])
                    speed_limit = float(state["speed_limit"])
                motors.drive(throttle, steering, speed_limit)

        return self._encode_jpeg(frame)


camera = Camera()


def mjpeg_stream():
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        time.sleep(0.05)


# -----------------------------
# Watchdog (deadman)
# -----------------------------
_manual_steer_f = 0.0
_manual_throttle_f = 0.0


def _ema(prev: float, x: float, alpha: float) -> float:
    alpha = _clamp(alpha, 0.0, 1.0)
    return prev + (x - prev) * alpha


def _watchdog_loop() -> None:
    global _last_control_ts, _manual_steer_f, _manual_throttle_f

    try:
        motors.stop()
    except Exception:
        pass

    while True:
        time.sleep(0.05)

        with state_lock:
            mode = state["mode"]
            speed_limit = float(state["speed_limit"])
            throttle = float(state["throttle"])
            steering = float(state["steering"])
            last = float(_last_control_ts)

        if mode != "manual":
            continue

        if last <= 0.0 or (time.time() - last) > CONTROL_TIMEOUT_S:
            motors.stop()
            _manual_steer_f = 0.0
            _manual_throttle_f = 0.0
            with state_lock:
                state["throttle"] = 0.0
                state["steering"] = 0.0
            continue

        _manual_steer_f = _ema(_manual_steer_f, steering, MANUAL_STEER_SMOOTH)
        _manual_throttle_f = _ema(_manual_throttle_f, throttle, MANUAL_THROTTLE_SMOOTH)
        motors.drive(_manual_throttle_f, _manual_steer_f, speed_limit)


threading.Thread(target=_watchdog_loop, daemon=True).start()


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        return jsonify({**state, "model_path": MODEL_PATH})


@app.route("/control", methods=["POST"])
def control():
    global _last_control_ts
    data = request.get_json(force=True, silent=True) or {}

    steering = float(data.get("steering", 0.0))
    throttle = float(data.get("throttle", 0.0))
    steering = _clamp(steering, -1.0, 1.0)
    throttle = _clamp(throttle, -1.0, 1.0)

    with state_lock:
        if state["mode"] == "manual":
            state["steering"] = steering
            state["throttle"] = throttle
        _last_control_ts = time.time()

    return jsonify({"status": "ok"})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    global _last_control_ts, _manual_steer_f, _manual_throttle_f
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "manual")
    mode = "autonomous" if mode == "autonomous" else "manual"

    with state_lock:
        state["mode"] = mode
        if mode == "manual":
            state["steering"] = 0.0
            state["throttle"] = 0.0

    _manual_steer_f = 0.0
    _manual_throttle_f = 0.0
    _last_control_ts = 0.0

    motors.stop()
    return jsonify({"status": "ok", "mode": mode})


@app.route("/toggle_recording", methods=["POST"])
def toggle_recording():
    with state_lock:
        new_value = not bool(state["recording"])
        state["recording"] = new_value

    if new_value:
        run_id = recorder.start()
        with state_lock:
            state["run_id"] = run_id
        return jsonify({"status": "ok", "recording": True, "run_id": run_id})
    else:
        recorder.stop()
        with state_lock:
            state["run_id"] = None
        return jsonify({"status": "ok", "recording": False})


def _shutdown(*_args):
    print("Shutting down: stopping motors...")
    try:
        motors.stop()
    except Exception:
        pass
    os._exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


if __name__ == "__main__":
    _last_control_ts = 0.0
    try:
        motors.stop()
    except Exception:
        pass
    app.run(host="0.0.0.0", port=5000, threaded=True)
