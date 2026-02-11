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
        "enable_upper_body_fallback": True,
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
        "control_mode": "spine_blend",
        "segments": 8,
        "neutral_angles": [90.0] * 8,
        "min_angles": [30.0] * 8,
        "max_angles": [150.0] * 8,
        "weights": [0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2],
        "upper_weights": [0.1, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4],
        "directions": [1.0] * 8,
        "upper_segment_start": 4,
        "gain": {"lean": 30.0, "neck": 25.0, "tilt": 10.0},
        "hunch_push": {
            "source": "neck",
            "use_absolute": True,
            "activation_threshold": 0.1,
            "full_scale": 0.8,
            "max_push_deg": 30.0,
        },
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
    "software_feedback": {
        "enabled": False,
        "mode": "overlay",
        "allow_hardware_output": False,
        "source": "face_proximity",
        "use_absolute": False,
        "activation_threshold": 1.05,
        "full_scale": 1.25,
        "smoothing_alpha": 0.25,
        "face_proximity": {
            "visibility_threshold": 0.5,
            "baseline_samples": 20,
            "baseline_alpha": 0.2,
        },
        "overlay": {
            "max_opacity": 0.8,
            "update_interval_sec": 0.02,
            "topmost": True,
            "fullscreen": True,
            "disable_input": True,
        },
        "brightness": {
            "min_percent": 35.0,
            "max_percent": 100.0,
            "update_interval_sec": 0.3,
            "min_delta_percent": 2.0,
            "command_template": "",
        },
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
    mode = str(mapping.get("control_mode", "spine_blend")).lower()
    if mode not in ("spine_blend", "hunch_push"):
        mode = "spine_blend"
    mapping["control_mode"] = mode

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

    hunch_cfg = mapping.get("hunch_push", {})
    if not isinstance(hunch_cfg, dict):
        hunch_cfg = {}
    hunch_cfg["source"] = str(hunch_cfg.get("source", "neck")).lower()
    hunch_cfg["use_absolute"] = bool(hunch_cfg.get("use_absolute", True))
    hunch_cfg["activation_threshold"] = float(hunch_cfg.get("activation_threshold", 0.1))
    hunch_cfg["full_scale"] = float(hunch_cfg.get("full_scale", 0.8))
    hunch_cfg["max_push_deg"] = float(hunch_cfg.get("max_push_deg", 30.0))
    mapping["hunch_push"] = hunch_cfg

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

    feedback_cfg = config.get("software_feedback", {})
    if not isinstance(feedback_cfg, dict):
        feedback_cfg = {}
    feedback_cfg["enabled"] = bool(feedback_cfg.get("enabled", False))
    mode = str(feedback_cfg.get("mode", "overlay")).lower()
    if mode not in ("overlay", "brightness", "both"):
        mode = "overlay"
    feedback_cfg["mode"] = mode
    feedback_cfg["allow_hardware_output"] = bool(feedback_cfg.get("allow_hardware_output", False))
    feedback_cfg["source"] = str(feedback_cfg.get("source", "face_proximity")).lower()
    feedback_cfg["use_absolute"] = bool(feedback_cfg.get("use_absolute", False))
    feedback_cfg["activation_threshold"] = float(feedback_cfg.get("activation_threshold", 1.05))
    feedback_cfg["full_scale"] = float(feedback_cfg.get("full_scale", 1.25))
    feedback_cfg["smoothing_alpha"] = float(feedback_cfg.get("smoothing_alpha", 0.25))

    face_cfg = feedback_cfg.get("face_proximity", {})
    if not isinstance(face_cfg, dict):
        face_cfg = {}
    face_cfg["visibility_threshold"] = float(face_cfg.get("visibility_threshold", 0.5))
    face_cfg["baseline_samples"] = max(1, int(face_cfg.get("baseline_samples", 20)))
    face_cfg["baseline_alpha"] = float(face_cfg.get("baseline_alpha", 0.2))
    feedback_cfg["face_proximity"] = face_cfg

    overlay_cfg = feedback_cfg.get("overlay", {})
    if not isinstance(overlay_cfg, dict):
        overlay_cfg = {}
    overlay_cfg["max_opacity"] = float(overlay_cfg.get("max_opacity", 0.8))
    overlay_cfg["update_interval_sec"] = float(overlay_cfg.get("update_interval_sec", 0.02))
    overlay_cfg["topmost"] = bool(overlay_cfg.get("topmost", True))
    overlay_cfg["fullscreen"] = bool(overlay_cfg.get("fullscreen", True))
    overlay_cfg["disable_input"] = bool(overlay_cfg.get("disable_input", True))
    overlay_cfg["width"] = int(overlay_cfg.get("width", 1280))
    overlay_cfg["height"] = int(overlay_cfg.get("height", 720))
    overlay_cfg["x"] = int(overlay_cfg.get("x", 0))
    overlay_cfg["y"] = int(overlay_cfg.get("y", 0))
    feedback_cfg["overlay"] = overlay_cfg

    brightness_cfg = feedback_cfg.get("brightness", {})
    if not isinstance(brightness_cfg, dict):
        brightness_cfg = {}
    brightness_cfg["min_percent"] = float(brightness_cfg.get("min_percent", 35.0))
    brightness_cfg["max_percent"] = float(brightness_cfg.get("max_percent", 100.0))
    brightness_cfg["update_interval_sec"] = float(brightness_cfg.get("update_interval_sec", 0.3))
    brightness_cfg["min_delta_percent"] = float(brightness_cfg.get("min_delta_percent", 2.0))
    brightness_cfg["command_template"] = str(brightness_cfg.get("command_template", ""))
    feedback_cfg["brightness"] = brightness_cfg

    config["software_feedback"] = feedback_cfg

    runtime = config["runtime"]
    runtime["vision_hz"] = max(1, int(runtime.get("vision_hz", 10)))
    runtime["servo_hz"] = max(1, int(runtime.get("servo_hz", 20)))

    metrics_cfg = config["metrics"]
    metrics_cfg["enable_upper_body_fallback"] = bool(
        metrics_cfg.get("enable_upper_body_fallback", True)
    )
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
