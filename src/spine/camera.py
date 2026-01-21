import logging
import time

import cv2


class CameraBase:
    def read(self):
        raise NotImplementedError

    def close(self):
        pass


class Picamera2Camera(CameraBase):
    def __init__(self, config):
        from picamera2 import Picamera2

        self.config = config
        self.picam2 = Picamera2()
        size = (int(config["width"]), int(config["height"]))
        video_config = self.picam2.create_video_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.picam2.configure(video_config)
        self.picam2.start()

        fps = int(config.get("fps", 30))
        if fps > 0:
            frame_time = int(1_000_000 / fps)
            try:
                self.picam2.set_controls({"FrameDurationLimits": (frame_time, frame_time)})
            except Exception:
                pass

        time.sleep(0.1)

    def read(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return apply_transforms(frame, self.config)

    def close(self):
        try:
            self.picam2.stop()
        except Exception:
            pass


class OpenCVCamera(CameraBase):
    def __init__(self, config):
        self.config = config
        index = int(config.get("device_index", 0))
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV camera {index} failed to open")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))
        fps = int(config.get("fps", 30))
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return apply_transforms(frame, self.config)

    def close(self):
        self.cap.release()


def apply_transforms(frame, config):
    if frame is None:
        return None
    if config.get("flip_horizontal"):
        frame = cv2.flip(frame, 1)
    if config.get("flip_vertical"):
        frame = cv2.flip(frame, 0)
    rotation = int(config.get("rotation", 0))
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def create_camera(config):
    backend = config.get("backend", "picamera2").lower()
    if backend in ("picamera2", "auto"):
        try:
            return Picamera2Camera(config)
        except Exception as exc:
            logging.warning("Picamera2 unavailable, falling back to OpenCV: %s", exc)
            if backend == "picamera2":
                raise
    return OpenCVCamera(config)
