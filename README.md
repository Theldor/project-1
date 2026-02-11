# Project 1 - Posture Feedback Runtime

This repo provides a posture feedback pipeline that can drive either hardware actuators (servos/steppers) or software feedback (ambient fullscreen overlay and/or screen brightness). It focuses on deterministic posture metrics (lean, neck angle, shoulder tilt) and smooth signal mapping without classification.

## Repo layout

```
config/
  config.json
src/spine/
  camera.py
  pose.py
  metrics.py
  filter.py
  mapping.py
  feedback.py
  servo.py
  main.py
  tools/
    servo_calibrate.py
    baseline_calibrate.py
```

## Setup (Pi OS Bookworm)

1. Enable I2C:
   - `sudo raspi-config` -> Interface Options -> I2C -> Enable
   - Reboot if prompted.
2. Install system packages:
   - `sudo apt update`
   - `sudo apt install -y python3-venv python3-picamera2 i2c-tools`
3. Create a venv and install Python deps:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -U pip`
   - `pip install -e .`
   - If you do not need a GUI window, you can replace `opencv-python` with `opencv-python-headless`.
   - On Homebrew Python (macOS), install Tk support for overlay mode: `brew install python-tk@3.11`

## Pose backend compatibility

- This project uses the legacy MediaPipe API: `mp.solutions.pose`.
- Recommended Python version is `3.11` on both macOS and Raspberry Pi.
- The dependency pins in `pyproject.toml` lock MediaPipe to versions that provide
  `mp.solutions.pose` for supported target platforms.
- If startup fails with an error like `module 'mediapipe' has no attribute 'solutions'`,
  reinstall MediaPipe with the pinned version for your platform.

## Running

Software feedback from the default config:
```
python -m spine.main --config config/config.json
```

Force software overlay mode (starts transparent and darkens as you move closer than baseline):
```
python -m spine.main --config config/config.json --software-feedback --feedback-mode overlay
```
Keep your neutral posture for the first ~2 seconds so the baseline can lock.

Dry run (no hardware or software feedback output):
```
python -m spine.main --config config/config.json --dry-run
```

Debug view (camera + landmarks, requires GUI):
```
python -m spine.main --config config/config.json --debug-view
```

## Calibration

Servo calibration (set neutral/min/max):
```
python -m spine.tools.servo_calibrate --config config/config.json
```

Baseline posture capture:
```
python -m spine.tools.baseline_calibrate --config config/config.json --seconds 3
```

## Wiring notes

- Power servos from an external 5-6V supply (do not power servos from the Pi).
- Connect the servo PSU ground to the Pi ground (common ground).
- PCA9685 SDA/SCL -> Pi SDA/SCL (I2C), VCC -> 3.3V, GND -> GND.

## Config highlights (`config/config.json`)

- `camera.backend`: `picamera2` or `opencv`
- `metrics.enable_upper_body_fallback`: keeps neck metric working when hips are out of frame (close camera framing)
- `mapping.control_mode`: `spine_blend` (original multi-metric mapping) or `hunch_push`
- `mapping.hunch_push`: hunch trigger settings (`activation_threshold`, `full_scale`, `max_push_deg`)
- `software_feedback.enabled`: enables software output loop
- `software_feedback.mode`: `overlay`, `brightness`, or `both`
- `software_feedback.allow_hardware_output`: keep hardware output active alongside software feedback
- `software_feedback.source`: signal source (`face_proximity` by default)
- `software_feedback.activation_threshold` / `full_scale`: face-distance ratios mapped to 0..1 feedback intensity
- `software_feedback.face_proximity.baseline_samples`: initial frames used to lock baseline face size
- `software_feedback.face_proximity.visibility_threshold`: minimum landmark visibility to trust face spacing
- `software_feedback.overlay.max_opacity`: maximum overlay darkness when user is much closer than baseline
- `software_feedback.brightness.min_percent` / `max_percent`: screen brightness range
- `software_feedback.brightness.command_template`: optional custom shell template with `{percent}` and `{scalar}`
- `software_feedback.overlay.disable_input`: enables click-through overlay behavior (enabled by default)
- `mapping.segments`: number of spine servos
- `mapping.weights` / `mapping.upper_weights`: curvature weighting
- `mapping.gain`: exaggeration values per metric
- `mapping.max_deg_per_sec`: rate limit to keep motion smooth
- `mapping.directions`: per-servo direction multipliers (use -1 to invert)
- `calibration.baseline`: captured upright posture offsets
- `calibration.normalization`: degree ranges mapped to +/-1.0

For two stepper arms that push a spring when the user is hunched:

- Set `mapping.control_mode` to `hunch_push`.
- Set `mapping.segments` to `2`.
- Set `stepper.motors` with two GPIO pin sets and matching `angle_index` values (`0`, `1`).
- Tune `mapping.directions` to `[1, -1]` or `[-1, 1]` so both arms push inward on hunch.

Brightness control backends are best-effort:

- macOS: `brightness` CLI if installed (`brew install brightness`)
- Linux: `brightnessctl` or `xrandr`
- Any OS: set `software_feedback.brightness.command_template` to your own command

For the ambient-overlay workflow, keep `mode=overlay` and `source=face_proximity`.

## Optional systemd service

Edit `deploy/spine.service` to match your Pi username and project path, then:
```
sudo cp deploy/spine.service /etc/systemd/system/spine.service
sudo systemctl daemon-reload
sudo systemctl enable spine.service
sudo systemctl start spine.service
```
