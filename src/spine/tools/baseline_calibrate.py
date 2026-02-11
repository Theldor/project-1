import argparse
import statistics
import time

import cv2

from spine.camera import create_camera
from spine.config import default_config_path, load_config, save_config
from spine.metrics import compute_metrics
from spine.pose import PoseEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Capture baseline posture metrics")
    parser.add_argument("--config", default=default_config_path(), help="Path to config JSON")
    parser.add_argument("--seconds", type=float, default=3.0, help="Duration to sample")
    parser.add_argument("--debug-view", action="store_true", help="Show live preview window")
    return parser.parse_args()


def _mean(values):
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def main():
    args = parse_args()
    config = load_config(args.config)
    camera = create_camera(config["camera"])
    pose = PoseEstimator(config["pose"])

    samples = {"lean_deg": [], "neck_deg": [], "tilt_deg": []}
    duration = max(1.0, args.seconds)
    interval = 1.0 / config["runtime"]["vision_hz"]
    start = time.time()

    try:
        while time.time() - start < duration:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = pose.process(frame_rgb)
            metrics = compute_metrics(
                landmarks,
                config["metrics"].get("visibility_threshold", 0.5),
                config["metrics"].get("enable_upper_body_fallback", True),
            )
            if metrics.lean_deg is not None:
                samples["lean_deg"].append(metrics.lean_deg)
            if metrics.neck_deg is not None:
                samples["neck_deg"].append(metrics.neck_deg)
            if metrics.tilt_deg is not None:
                samples["tilt_deg"].append(metrics.tilt_deg)

            if args.debug_view:
                cv2.imshow("Baseline Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(interval)
    finally:
        camera.close()
        pose.close()
        if args.debug_view:
            cv2.destroyAllWindows()

    baseline = {
        "lean_deg": _mean(samples["lean_deg"]),
        "neck_deg": _mean(samples["neck_deg"]),
        "tilt_deg": _mean(samples["tilt_deg"]),
    }

    config["calibration"]["baseline"] = baseline
    save_config(args.config, config)

    print("Baseline captured:")
    print(baseline)
    print(f"Saved to {args.config}")


if __name__ == "__main__":
    main()
