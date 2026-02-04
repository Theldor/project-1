import argparse
import logging
import time

import cv2
import mediapipe as mp

from .camera import create_camera
from .config import default_config_path, load_config
from .filter import MetricSmoother
from .mapping import SpineMapper
from .metrics import NormalizedMetrics, compute_metrics, normalize_metrics
from .pose import PoseEstimator
from .servo import create_servo_controller
from .stepper import create_stepper_controller


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


def _draw_metric_nodes(frame, landmarks, visibility_threshold=0.5):
    if not landmarks:
        return frame
    height, width = frame.shape[:2]
    mp_pose = mp.solutions.pose.PoseLandmark
    left_shoulder = mp_pose.LEFT_SHOULDER.value
    right_shoulder = mp_pose.RIGHT_SHOULDER.value
    left_hip = mp_pose.LEFT_HIP.value
    right_hip = mp_pose.RIGHT_HIP.value
    left_ear = mp_pose.LEFT_EAR.value
    right_ear = mp_pose.RIGHT_EAR.value

    def vis(index):
        return getattr(landmarks[index], "visibility", 1.0)

    def draw(index, color, radius):
        if vis(index) < visibility_threshold:
            return
        lm = landmarks[index]
        cx = int(lm.x * width)
        cy = int(lm.y * height)
        cv2.circle(frame, (cx, cy), radius, color, -1)

    neck_side = left_ear if vis(left_ear) >= vis(right_ear) else right_ear
    neck_shoulder = left_shoulder if neck_side == left_ear else right_shoulder
    neck_hip = left_hip if neck_side == left_ear else right_hip

    base_color = (0, 165, 255)
    active_color = (0, 0, 255)

    for index in (left_shoulder, right_shoulder, left_hip, right_hip, left_ear, right_ear):
        draw(index, base_color, 4)
    for index in (neck_side, neck_shoulder, neck_hip):
        draw(index, active_color, 6)

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
    logging.info(
        "Pose backend: mp.solutions.pose (mediapipe=%s)",
        getattr(mp, "__version__", "unknown"),
    )

    camera = create_camera(config["camera"])
    pose = PoseEstimator(config["pose"])
    smoother = MetricSmoother(config["metrics"])
    mapper = SpineMapper(config["mapping"])
    use_stepper = bool(config.get("stepper", {}).get("enabled", False))
    if use_stepper:
        actuator = create_stepper_controller(config["stepper"], config["runtime"]["dry_run"])
    else:
        actuator = create_servo_controller(config["servo"], config["runtime"]["dry_run"])

    vision_interval = 1.0 / config["runtime"]["vision_hz"]
    servo_interval = 1.0 / config["runtime"]["servo_hz"]
    last_vision = 0.0
    last_servo = 0.0
    last_frame = None
    last_metrics = NormalizedMetrics(0.0, 0.0, 0.0)
    last_filtered = None
    last_landmarks = None

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
                last_landmarks = landmarks

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
                actuator.set_angles(angles)

            if config["runtime"]["show_debug_view"] and last_frame is not None and last_filtered:
                overlay = last_frame.copy()
                if last_landmarks:
                    visibility_threshold = config["metrics"].get("visibility_threshold", 0.5)
                    overlay = pose.draw(overlay, last_landmarks, visibility_threshold)
                    overlay = _draw_metric_nodes(overlay, last_landmarks, visibility_threshold)
                overlay = _overlay(overlay, fps_value, last_filtered, last_metrics)
                cv2.imshow("Spine Debug", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        actuator.close()
        camera.close()
        pose.close()
        if config["runtime"]["show_debug_view"]:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
