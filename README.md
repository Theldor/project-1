# Project 1 - Moving Spine (Posture Mapping)

This repo provides a minimal Raspberry Pi pipeline that maps a single-person pose to a multi-segment spine driven by servos. It focuses on deterministic posture metrics (lean, neck angle, shoulder tilt) and smooth servo mapping without classification.

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

## Pose backend compatibility

- This project uses the legacy MediaPipe API: `mp.solutions.pose`.
- Recommended Python version is `3.11` on both macOS and Raspberry Pi.
- The dependency pins in `pyproject.toml` lock MediaPipe to versions that provide
  `mp.solutions.pose` for supported target platforms.
- If startup fails with an error like `module 'mediapipe' has no attribute 'solutions'`,
  reinstall MediaPipe with the pinned version for your platform.

## Running

Dry run (no servos):
```
python -m spine.main --config config/config.json --dry-run
```

Debug view (requires GUI):
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
- `mapping.segments`: number of spine servos
- `mapping.weights` / `mapping.upper_weights`: curvature weighting
- `mapping.gain`: exaggeration values per metric
- `mapping.max_deg_per_sec`: rate limit to keep motion smooth
- `mapping.directions`: per-servo direction multipliers (use -1 to invert)
- `calibration.baseline`: captured upright posture offsets
- `calibration.normalization`: degree ranges mapped to +/-1.0

## Optional systemd service

Edit `deploy/spine.service` to match your Pi username and project path, then:
```
sudo cp deploy/spine.service /etc/systemd/system/spine.service
sudo systemctl daemon-reload
sudo systemctl enable spine.service
sudo systemctl start spine.service
```
