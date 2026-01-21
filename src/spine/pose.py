import cv2
import mediapipe as mp


class PoseEstimator:
    def __init__(self, config):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(config.get("model_complexity", 0)),
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
        )

    def process(self, frame_rgb):
        result = self.pose.process(frame_rgb)
        if result.pose_landmarks:
            return result.pose_landmarks.landmark
        return None

    def close(self):
        self.pose.close()

    def draw(self, frame_bgr, landmarks, visibility_threshold=0.5):
        if not landmarks:
            return frame_bgr
        height, width = frame_bgr.shape[:2]
        for landmark in landmarks:
            if getattr(landmark, "visibility", 1.0) < visibility_threshold:
                continue
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)
            cv2.circle(frame_bgr, (cx, cy), 2, (0, 255, 0), -1)
        return frame_bgr
