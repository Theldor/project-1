import copy
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "camera": {
        "backend": "picamera2",
        "device_index": 0,
        "width": 320,
        "height": 240,
        "fps": 30,
        "flip_horizontal": False,
        "flip_vertical": False,
        "rotation": 0,
    },
    "pose": {
        "model_complexity": 0,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "metrics": {
        "visibility_threshold": 0.5,
        "smoothing": {
            "enabled": True,
            "type": "ema",
            "alpha": 0.2,
            "one_euro_min_cutoff": 1.2,
            "one_euro_beta": 0.02,
            "one_euro_d_cutoff": 1.0,
        },
        "hold_seconds": 0.5,
        "decay_seconds": 1.5,
    },
    "mapping": {
        "segments": 8,
        "neutral_angles": [90.0] * 8,
        "min_angles": [30.0] * 8,
        "max_angles": [150.0] * 8,
        "weights": [0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2],
        "upper_weights": [0.1, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4],
        "directions": [1.0] * 8,
        "upper_segment_start": 4,
        "gain": {"lean": 30.0, "neck": 25.0, "tilt": 10.0},
        "deadband_deg": 1.0,
        "max_deg_per_sec": 120.0,
    },
    "servo": {
        "enabled": True,
        "i2c_address": "0x40",
        "frequency_hz": 50,
        "channels": [0, 1, 2, 3, 4, 5, 6, 7],
        "pulse_min_us": 500,
        "pulse_max_us": 2500,
        "actuation_range": 180,
    },
    "stepper": {
        "enabled": False,
        "steps_per_rev": 200,
        "microstep": 1,
        "max_steps_per_sec": 800.0,
        "min_pulse_us": 2.0,
        "motors": [
            {
                "step_pin": 18,
                "dir_pin": 23,
                "enable_pin": 24,
                "enable_active_low": True,
                "angle_index": 0,
            }
        ],
    },
    "runtime": {
        "dry_run": True,
        "show_debug_view": False,
        "vision_hz": 10,
        "servo_hz": 20,
        "log_level": "INFO",
    },
    "calibration": {
        "baseline": {"lean_deg": 0.0, "neck_deg": 0.0, "tilt_deg": 0.0},
        "normalization": {"lean_deg": 25.0, "neck_deg": 25.0, "tilt_deg": 15.0},
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _default_weights(count):
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    mid = (count - 1) / 2.0
    span = max(mid, 1.0)
    weights = []
    for i in range(count):
        base = 1.0 - abs(i - mid) / span
        weights.append(0.2 + 0.8 * base)
    return weights


def _ensure_list(value, count, fill):
    if value is None:
        value = []
    value = list(value)
    if len(value) < count:
        value.extend([fill] * (count - len(value)))
    return value[:count]


def normalize_config(config):
    mapping = config["mapping"]
    segments = int(mapping.get("segments", 0))
    if segments <= 0:
        segments = 1
    mapping["segments"] = segments

    weights = mapping.get("weights") or _default_weights(segments)
    mapping["weights"] = _ensure_list(weights, segments, 0.5)

    upper_weights = mapping.get("upper_weights") or mapping["weights"]
    mapping["upper_weights"] = _ensure_list(upper_weights, segments, 0.5)

    mapping["neutral_angles"] = _ensure_list(mapping.get("neutral_angles"), segments, 90.0)
    mapping["min_angles"] = _ensure_list(mapping.get("min_angles"), segments, 30.0)
    mapping["max_angles"] = _ensure_list(mapping.get("max_angles"), segments, 150.0)
    mapping["directions"] = _ensure_list(mapping.get("directions"), segments, 1.0)

    upper_start = mapping.get("upper_segment_start")
    if upper_start is None:
        mapping["upper_segment_start"] = segments // 2
    else:
        mapping["upper_segment_start"] = max(0, min(int(upper_start), segments - 1))

    servo_cfg = config["servo"]
    channels = servo_cfg.get("channels")
    if not channels or len(channels) < segments:
        servo_cfg["channels"] = list(range(segments))
    else:
        servo_cfg["channels"] = list(channels[:segments])

    addr = servo_cfg.get("i2c_address", "0x40")
    if isinstance(addr, str):
        servo_cfg["i2c_address"] = int(addr, 0)

    stepper_cfg = config.get("stepper", {})
    stepper_cfg["steps_per_rev"] = max(1, int(stepper_cfg.get("steps_per_rev", 200)))
    stepper_cfg["microstep"] = max(1, int(stepper_cfg.get("microstep", 1)))
    stepper_cfg["max_steps_per_sec"] = float(stepper_cfg.get("max_steps_per_sec", 800.0))
    stepper_cfg["min_pulse_us"] = float(stepper_cfg.get("min_pulse_us", 2.0))
    motors = stepper_cfg.get("motors", [])
    if not isinstance(motors, list):
        motors = []
    stepper_cfg["motors"] = motors
    config["stepper"] = stepper_cfg

    runtime = config["runtime"]
    runtime["vision_hz"] = max(1, int(runtime.get("vision_hz", 10)))
    runtime["servo_hz"] = max(1, int(runtime.get("servo_hz", 20)))

    metrics_cfg = config["metrics"]
    smoothing = metrics_cfg.get("smoothing", {})
    smoothing["alpha"] = float(smoothing.get("alpha", 0.2))
    smoothing["one_euro_min_cutoff"] = float(smoothing.get("one_euro_min_cutoff", 1.2))
    smoothing["one_euro_beta"] = float(smoothing.get("one_euro_beta", 0.02))
    smoothing["one_euro_d_cutoff"] = float(smoothing.get("one_euro_d_cutoff", 1.0))


def load_config(path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            _deep_update(config, data)
    normalize_config(config)
    return config


def save_config(path, config):
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_config_path():
    return str(Path("config") / "config.json")
