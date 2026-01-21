import argparse
import logging

from spine.config import default_config_path, load_config, save_config
from spine.servo import create_servo_controller


def parse_args():
    parser = argparse.ArgumentParser(description="Servo calibration helper")
    parser.add_argument("--config", default=default_config_path(), help="Path to config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Disable servo output")
    parser.add_argument("--step", type=float, default=2.0, help="Angle step for +/-")
    return parser.parse_args()


def _clamp(angle):
    return max(0.0, min(180.0, angle))


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        config["runtime"]["dry_run"] = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    channels = config["servo"]["channels"]
    segments = config["mapping"]["segments"]
    neutral = list(config["mapping"]["neutral_angles"])
    min_angles = list(config["mapping"]["min_angles"])
    max_angles = list(config["mapping"]["max_angles"])

    servo = create_servo_controller(config["servo"], config["runtime"]["dry_run"])
    angles = list(neutral)
    selected = 0
    step = args.step

    print("Servo calibration")
    print("Commands: ch <index>, +, -, step <deg>, neutral, min, max, show, save, quit")
    servo.set_angles(angles)

    try:
        while True:
            cmd = input(f"[segment {selected} angle {angles[selected]:.1f}]> ").strip().lower()
            if not cmd:
                continue
            if cmd in ("q", "quit", "exit"):
                break
            if cmd.startswith("ch"):
                parts = cmd.split()
                if len(parts) == 2 and parts[1].isdigit():
                    selected = max(0, min(segments - 1, int(parts[1])))
                else:
                    print("Usage: ch <index>")
                continue
            if cmd.startswith("step"):
                parts = cmd.split()
                if len(parts) == 2:
                    try:
                        step = float(parts[1])
                    except ValueError:
                        print("Invalid step")
                continue
            if cmd.startswith("+"):
                delta = step
                if len(cmd) > 1:
                    try:
                        delta = float(cmd[1:])
                    except ValueError:
                        pass
                angles[selected] = _clamp(angles[selected] + delta)
                servo.set_angles(angles)
                continue
            if cmd.startswith("-"):
                delta = step
                if len(cmd) > 1:
                    try:
                        delta = float(cmd[1:])
                    except ValueError:
                        pass
                angles[selected] = _clamp(angles[selected] - delta)
                servo.set_angles(angles)
                continue
            if cmd == "neutral":
                neutral[selected] = angles[selected]
                print(f"Set neutral[{selected}] = {neutral[selected]:.1f}")
                continue
            if cmd == "min":
                min_angles[selected] = angles[selected]
                print(f"Set min[{selected}] = {min_angles[selected]:.1f}")
                continue
            if cmd == "max":
                max_angles[selected] = angles[selected]
                print(f"Set max[{selected}] = {max_angles[selected]:.1f}")
                continue
            if cmd == "show":
                print("neutral:", [round(v, 1) for v in neutral])
                print("min:", [round(v, 1) for v in min_angles])
                print("max:", [round(v, 1) for v in max_angles])
                continue
            if cmd == "save":
                config["mapping"]["neutral_angles"] = neutral
                config["mapping"]["min_angles"] = min_angles
                config["mapping"]["max_angles"] = max_angles
                save_config(args.config, config)
                print(f"Saved to {args.config}")
                continue

            print("Unknown command")
    finally:
        servo.close()


if __name__ == "__main__":
    main()
