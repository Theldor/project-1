import logging
import platform
import shutil
import subprocess
import time

import mediapipe as mp


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class NoopFeedbackController:
    def set_level(self, _level):
        pass

    def poll(self):
        pass

    def close(self):
        pass


class CombinedFeedbackController:
    def __init__(self, controllers):
        self.controllers = list(controllers)

    def set_level(self, level):
        for controller in self.controllers:
            controller.set_level(level)

    def poll(self):
        for controller in self.controllers:
            controller.poll()

    def close(self):
        for controller in self.controllers:
            try:
                controller.close()
            except Exception:
                pass


class PostureSignalMapper:
    def __init__(self, config):
        self.source = str(config.get("source", "neck")).lower()
        self.use_absolute = bool(config.get("use_absolute", True))
        self.activation_threshold = float(config.get("activation_threshold", 0.1))
        self.full_scale = float(config.get("full_scale", 0.8))
        self.smoothing_alpha = _clamp(float(config.get("smoothing_alpha", 0.25)), 0.0, 1.0)
        self._smoothed = 0.0

        if self.full_scale <= self.activation_threshold:
            self.full_scale = self.activation_threshold + 1e-6

    def map_metrics(self, metrics):
        source_value = float(getattr(metrics, self.source, 0.0))
        return self.map_value(source_value)

    def map_value(self, source_value):
        source_value = float(source_value)
        if self.use_absolute:
            source_value = abs(source_value)
        else:
            source_value = max(0.0, source_value)

        if source_value <= self.activation_threshold:
            raw = 0.0
        else:
            raw = (source_value - self.activation_threshold) / (
                self.full_scale - self.activation_threshold
            )
            raw = _clamp(raw, 0.0, 1.0)

        if self.smoothing_alpha <= 0.0:
            self._smoothed = raw
        elif self.smoothing_alpha >= 1.0:
            self._smoothed = raw
        else:
            self._smoothed = self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * self._smoothed

        return self._smoothed


def _landmark_visibility(landmarks, index):
    return getattr(landmarks[index], "visibility", 1.0)


def _pair_distance(landmarks, index_a, index_b):
    a = landmarks[index_a]
    b = landmarks[index_b]
    dx = float(a.x) - float(b.x)
    dy = float(a.y) - float(b.y)
    return (dx * dx + dy * dy) ** 0.5


def compute_face_scale(landmarks, visibility_threshold=0.5):
    if not landmarks:
        return None

    pose = mp.solutions.pose.PoseLandmark
    pairs = (
        (pose.LEFT_EYE_INNER.value, pose.RIGHT_EYE_INNER.value),
        (pose.LEFT_EYE.value, pose.RIGHT_EYE.value),
        (pose.MOUTH_LEFT.value, pose.MOUTH_RIGHT.value),
        (pose.LEFT_EAR.value, pose.RIGHT_EAR.value),
    )

    distances = []
    for index_a, index_b in pairs:
        if (
            _landmark_visibility(landmarks, index_a) < visibility_threshold
            or _landmark_visibility(landmarks, index_b) < visibility_threshold
        ):
            continue
        distance = _pair_distance(landmarks, index_a, index_b)
        if distance > 1e-6:
            distances.append(distance)

    if not distances:
        return None

    distances.sort()
    return distances[len(distances) // 2]


class FaceProximitySignalMapper:
    def __init__(self, config):
        face_cfg = config.get("face_proximity", {})
        self.visibility_threshold = float(face_cfg.get("visibility_threshold", 0.5))
        self.baseline_samples = max(1, int(face_cfg.get("baseline_samples", 20)))
        self.baseline_alpha = _clamp(float(face_cfg.get("baseline_alpha", 0.2)), 0.0, 1.0)
        self.activation_ratio = float(config.get("activation_threshold", 1.05))
        self._baseline = None
        self._samples = 0

        self._mapper = PostureSignalMapper(
            {
                "source": "face_proximity",
                "use_absolute": False,
                "activation_threshold": float(config.get("activation_threshold", 1.05)),
                "full_scale": float(config.get("full_scale", 1.25)),
                "smoothing_alpha": float(config.get("smoothing_alpha", 0.25)),
            }
        )

    def map_landmarks(self, landmarks):
        scale = compute_face_scale(landmarks, self.visibility_threshold)
        if scale is None:
            return self._mapper.map_value(self.activation_ratio)

        if self._baseline is None:
            self._baseline = scale
            self._samples = 1
            return 0.0

        if self._samples < self.baseline_samples:
            self._baseline = (1.0 - self.baseline_alpha) * self._baseline + self.baseline_alpha * scale
            self._samples += 1
            return 0.0

        baseline = max(self._baseline, 1e-6)
        ratio = scale / baseline
        if ratio <= self.activation_ratio:
            self._baseline = (1.0 - self.baseline_alpha) * self._baseline + self.baseline_alpha * scale

        return self._mapper.map_value(ratio)


class OverlayFeedbackController:
    def __init__(self, config, dry_run=False):
        self.dry_run = dry_run
        self.max_opacity = _clamp(float(config.get("max_opacity", 0.8)), 0.0, 1.0)
        self.update_interval = max(0.01, float(config.get("update_interval_sec", 0.02)))
        self.topmost = bool(config.get("topmost", True))
        self.fullscreen = bool(config.get("fullscreen", True))
        self.disable_input = bool(config.get("disable_input", False))
        self.geometry = {
            "width": int(config.get("width", 1280)),
            "height": int(config.get("height", 720)),
            "x": int(config.get("x", 0)),
            "y": int(config.get("y", 0)),
        }

        self._target_level = 0.0
        self._last_alpha = None
        self._last_update = 0.0
        self._root = None
        self._tcl_error = None
        self._click_through_enabled = False

        if not self.dry_run:
            self._create_window()

    def _enable_macos_click_through(self, root):
        if platform.system().lower() != "darwin":
            return False

        try:
            from AppKit import NSApp
        except Exception as exc:
            logging.warning("Overlay click-through unavailable (AppKit import failed): %s", exc)
            return False

        try:
            root.update_idletasks()
            root.update()
            app = NSApp()
            if app is None:
                return False

            windows = list(app.orderedWindows() or [])
            if not windows:
                return False

            applied = False
            for ns_window in windows:
                try:
                    ns_window.setIgnoresMouseEvents_(True)
                    applied = True
                except Exception:
                    continue
            if not applied:
                return False

            return True
        except Exception as exc:
            logging.warning("Overlay click-through unavailable (Cocoa bridge failed): %s", exc)
            return False

    def _create_window(self):
        try:
            import tkinter as tk
        except Exception as exc:
            logging.warning("Overlay feedback unavailable (tkinter import failed): %s", exc)
            return

        self._tcl_error = tk.TclError

        try:
            root = tk.Tk()
            root.configure(bg="black")
            root.overrideredirect(True)
            if self.fullscreen:
                root.attributes("-fullscreen", True)
                width = int(root.winfo_screenwidth())
                height = int(root.winfo_screenheight())
                root.geometry(f"{width}x{height}+0+0")
            else:
                width = self.geometry["width"]
                height = self.geometry["height"]
                x = self.geometry["x"]
                y = self.geometry["y"]
                root.geometry(f"{width}x{height}+{x}+{y}")
            if self.topmost:
                root.attributes("-topmost", True)
            if self.disable_input:
                try:
                    root.attributes("-disabled", True)
                except Exception:
                    pass
            root.attributes("-alpha", 0.0)
            root.update_idletasks()
            if self.disable_input:
                self._click_through_enabled = self._enable_macos_click_through(root)
                if self._click_through_enabled:
                    logging.info("Overlay click-through enabled on macOS")
                elif platform.system().lower() == "darwin":
                    logging.warning(
                        "Overlay click-through requested but could not be enabled on macOS."
                    )
            self._root = root
        except Exception as exc:
            logging.warning("Overlay feedback unavailable (window init failed): %s", exc)
            self._root = None

    def set_level(self, level):
        self._target_level = _clamp(float(level), 0.0, 1.0)

    def poll(self):
        if self.dry_run or self._root is None:
            return

        now = time.time()
        if now - self._last_update < self.update_interval:
            return
        self._last_update = now

        alpha = self._target_level * self.max_opacity
        if self._last_alpha is None or abs(alpha - self._last_alpha) >= 0.01:
            try:
                self._root.attributes("-alpha", alpha)
                self._last_alpha = alpha
            except Exception:
                pass

        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception as exc:
            if self._tcl_error and isinstance(exc, self._tcl_error):
                self._root = None
            else:
                logging.debug("Overlay feedback update failed: %s", exc)

    def close(self):
        if self._root is None:
            return
        try:
            self._root.destroy()
        except Exception:
            pass
        self._root = None


class BrightnessFeedbackController:
    def __init__(self, config, dry_run=False):
        self.dry_run = dry_run
        min_percent = float(config.get("min_percent", 35.0))
        max_percent = float(config.get("max_percent", 100.0))
        self.min_percent = _clamp(min_percent, 1.0, 100.0)
        self.max_percent = _clamp(max_percent, self.min_percent, 100.0)
        self.update_interval = max(0.05, float(config.get("update_interval_sec", 0.3)))
        self.min_delta_percent = max(0.0, float(config.get("min_delta_percent", 2.0)))
        self.command_template = str(config.get("command_template", "")).strip()

        self.backend = None
        self._xrandr_output = None
        self._last_percent = None
        self._last_update = 0.0

        if not self.dry_run:
            self._detect_backend()

    def _detect_backend(self):
        if self.command_template:
            self.backend = "template"
            return

        system = platform.system().lower()

        if system == "darwin" and shutil.which("brightness"):
            self.backend = "brightness"
            return

        if system == "linux" and shutil.which("brightnessctl"):
            self.backend = "brightnessctl"
            return

        if system == "linux" and shutil.which("xrandr"):
            output = self._detect_xrandr_primary_output()
            if output:
                self.backend = "xrandr"
                self._xrandr_output = output
                return

        logging.warning(
            "Brightness feedback unavailable; configure software_feedback.brightness.command_template "
            "or install a supported backend (brightness/brightnessctl/xrandr)."
        )

    def _detect_xrandr_primary_output(self):
        try:
            result = subprocess.run(
                ["xrandr", "--query"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        primary_output = None
        fallback_output = None
        for line in result.stdout.splitlines():
            if " connected" not in line:
                continue
            name = line.split()[0]
            if fallback_output is None:
                fallback_output = name
            if " connected primary " in line:
                primary_output = name
                break

        return primary_output or fallback_output

    def _run_command(self, command):
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        except Exception as exc:
            logging.debug("Brightness command failed (%s): %s", command, exc)
            return False

    def _apply_percent(self, percent):
        if self.backend == "template":
            scalar = percent / 100.0
            command = self.command_template.format(percent=int(round(percent)), scalar=f"{scalar:.3f}")
            return self._run_command(command if isinstance(command, list) else ["sh", "-lc", command])

        if self.backend == "brightness":
            scalar = percent / 100.0
            return self._run_command(["brightness", f"{scalar:.3f}"])

        if self.backend == "brightnessctl":
            return self._run_command(["brightnessctl", "set", f"{int(round(percent))}%"])

        if self.backend == "xrandr" and self._xrandr_output:
            scalar = max(0.10, percent / 100.0)
            return self._run_command(
                ["xrandr", "--output", self._xrandr_output, "--brightness", f"{scalar:.3f}"]
            )

        return False

    def set_level(self, level):
        level = _clamp(float(level), 0.0, 1.0)
        target_percent = self.max_percent - level * (self.max_percent - self.min_percent)

        now = time.time()
        if self._last_percent is not None:
            if abs(target_percent - self._last_percent) < self.min_delta_percent:
                return
            if now - self._last_update < self.update_interval:
                return

        if self.dry_run:
            logging.debug("Dry-run brightness target: %.1f%%", target_percent)
            self._last_percent = target_percent
            self._last_update = now
            return

        if not self.backend:
            return

        if self._apply_percent(target_percent):
            self._last_percent = target_percent
            self._last_update = now

    def poll(self):
        pass

    def close(self):
        pass


def create_feedback_controller(config, dry_run=False):
    if not config or not bool(config.get("enabled", False)):
        return NoopFeedbackController()

    mode = str(config.get("mode", "overlay")).lower()
    controllers = []

    if mode in ("overlay", "both"):
        controllers.append(OverlayFeedbackController(config.get("overlay", {}), dry_run=dry_run))
    if mode in ("brightness", "both"):
        controllers.append(BrightnessFeedbackController(config.get("brightness", {}), dry_run=dry_run))

    if not controllers:
        return NoopFeedbackController()
    if len(controllers) == 1:
        return controllers[0]
    return CombinedFeedbackController(controllers)
