import argparse
import logging
import time

import cv2

from .camera import create_camera
from .config import default_config_path, load_config
from .filter import MetricSmoother
from .mapping import SpineMapper
from .metrics import NormalizedMetrics, compute_metrics, normalize_metrics
from .pose import PoseEstimator
from .servo import create_servo_controller


def parse_args():
    parser = argparse.ArgumentParser(description="Posture-to-spine mapping")
    parser.add_argument("--config", default=default_config_path(), help="Path to config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Disable servo output")
    parser.add_argument("--debug-view", action="store_true", help="Show debug overlay window")
    return parser.parse_args()


def _overlay(frame, fps, metrics, normalized):
    lines = [
        f"FPS: {fps:.1f}",
        f"Lean deg: {metrics.lean_deg:.1f}  norm: {normalized.lean:.2f}",
        f"Neck deg: {metrics.neck_deg:.1f}  norm: {normalized.neck:.2f}",
        f"Tilt deg: {metrics.tilt_deg:.1f}  norm: {normalized.tilt:.2f}",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        y += 18
    return frame


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        config["runtime"]["dry_run"] = True
    if args.debug_view:
        config["runtime"]["show_debug_view"] = True

    level_name = str(config["runtime"].get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    camera = create_camera(config["camera"])
    pose = PoseEstimator(config["pose"])
    smoother = MetricSmoother(config["metrics"])
    mapper = SpineMapper(config["mapping"])
    servo = create_servo_controller(config["servo"], config["runtime"]["dry_run"])

    vision_interval = 1.0 / config["runtime"]["vision_hz"]
    servo_interval = 1.0 / config["runtime"]["servo_hz"]
    last_vision = 0.0
    last_servo = 0.0
    last_frame = None
    last_metrics = NormalizedMetrics(0.0, 0.0, 0.0)
    last_filtered = None

    fps_counter = 0
    fps_timer = time.time()
    fps_value = 0.0
    last_log = 0.0

    try:
        while True:
            now = time.time()

            if now - last_vision >= vision_interval:
                last_vision = now
                frame = camera.read()
                if frame is None:
                    logging.warning("Camera frame read failed")
                    time.sleep(0.1)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = None
                try:
                    landmarks = pose.process(frame_rgb)
                except Exception as exc:
                    logging.error("Pose processing failed: %s", exc)

                raw_metrics = compute_metrics(
                    landmarks, config["metrics"].get("visibility_threshold", 0.5)
                )
                filtered = smoother.update(raw_metrics, now)
                last_filtered = filtered
                last_metrics = normalize_metrics(filtered, config["calibration"], config["mapping"])
                last_frame = frame

                fps_counter += 1
                if now - fps_timer >= 1.0:
                    fps_value = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                if config["runtime"]["dry_run"] and now - last_log >= 1.0:
                    last_log = now
                    logging.info(
                        "Metrics deg (lean/neck/tilt): %.1f %.1f %.1f",
                        filtered.lean_deg,
                        filtered.neck_deg,
                        filtered.tilt_deg,
                    )

            if now - last_servo >= servo_interval:
                last_servo = now
                angles = mapper.map_metrics(last_metrics, now)
                servo.set_angles(angles)

            if config["runtime"]["show_debug_view"] and last_frame is not None and last_filtered:
                overlay = _overlay(last_frame.copy(), fps_value, last_filtered, last_metrics)
                cv2.imshow("Spine Debug", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        servo.close()
        camera.close()
        pose.close()
        if config["runtime"]["show_debug_view"]:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
