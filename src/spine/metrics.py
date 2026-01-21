import math
from dataclasses import dataclass

import mediapipe as mp


@dataclass
class MetricValues:
    lean_deg: float | None
    neck_deg: float | None
    tilt_deg: float | None


@dataclass
class NormalizedMetrics:
    lean: float
    neck: float
    tilt: float


def _angle_between(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    dot = v1x * v2x + v1y * v2y
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _visibility(landmark):
    return getattr(landmark, "visibility", 1.0)


def _mean_point(landmarks, index_a, index_b):
    a = landmarks[index_a]
    b = landmarks[index_b]
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _choose_side(landmarks, left_index, right_index):
    left_vis = _visibility(landmarks[left_index])
    right_vis = _visibility(landmarks[right_index])
    return left_index if left_vis >= right_vis else right_index


def compute_metrics(landmarks, visibility_threshold=0.5):
    if not landmarks:
        return MetricValues(None, None, None)

    mp_pose = mp.solutions.pose.PoseLandmark

    left_shoulder = mp_pose.LEFT_SHOULDER.value
    right_shoulder = mp_pose.RIGHT_SHOULDER.value
    left_hip = mp_pose.LEFT_HIP.value
    right_hip = mp_pose.RIGHT_HIP.value
    left_ear = mp_pose.LEFT_EAR.value
    right_ear = mp_pose.RIGHT_EAR.value

    shoulder_vis = min(_visibility(landmarks[left_shoulder]), _visibility(landmarks[right_shoulder]))
    hip_vis = min(_visibility(landmarks[left_hip]), _visibility(landmarks[right_hip]))

    lean_deg = None
    if shoulder_vis >= visibility_threshold and hip_vis >= visibility_threshold:
        mid_shoulder = _mean_point(landmarks, left_shoulder, right_shoulder)
        mid_hip = _mean_point(landmarks, left_hip, right_hip)
        torso_vec = (mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1])
        angle = _angle_between(torso_vec, (0.0, -1.0))
        lean_sign = 1.0 if torso_vec[0] >= 0 else -1.0
        lean_deg = lean_sign * angle

    neck_deg = None
    neck_side = _choose_side(landmarks, left_ear, right_ear)
    shoulder_side = left_shoulder if neck_side == left_ear else right_shoulder
    hip_side = left_hip if neck_side == left_ear else right_hip
    neck_vis = min(
        _visibility(landmarks[neck_side]),
        _visibility(landmarks[shoulder_side]),
        _visibility(landmarks[hip_side]),
    )
    if neck_vis >= visibility_threshold:
        ear = landmarks[neck_side]
        shoulder = landmarks[shoulder_side]
        hip = landmarks[hip_side]
        v1 = (ear.x - shoulder.x, ear.y - shoulder.y)
        v2 = (hip.x - shoulder.x, hip.y - shoulder.y)
        angle = _angle_between(v1, v2)
        forward = 180.0 - angle
        sign = 1.0 if v1[0] >= 0 else -1.0
        neck_deg = sign * forward

    tilt_deg = None
    if shoulder_vis >= visibility_threshold:
        left = landmarks[left_shoulder]
        right = landmarks[right_shoulder]
        dx = right.x - left.x
        dy = right.y - left.y
        tilt_deg = math.degrees(math.atan2(dy, dx))

    return MetricValues(lean_deg=lean_deg, neck_deg=neck_deg, tilt_deg=tilt_deg)


def normalize_metrics(values, calibration, mapping):
    baseline = calibration.get("baseline", {})
    normalization = calibration.get("normalization", {})
    deadband = float(mapping.get("deadband_deg", 0.0))

    lean = _normalize_one(values.lean_deg, baseline.get("lean_deg", 0.0), normalization.get("lean_deg", 1.0), deadband)
    neck = _normalize_one(values.neck_deg, baseline.get("neck_deg", 0.0), normalization.get("neck_deg", 1.0), deadband)
    tilt = _normalize_one(values.tilt_deg, baseline.get("tilt_deg", 0.0), normalization.get("tilt_deg", 1.0), deadband)

    return NormalizedMetrics(lean=lean, neck=neck, tilt=tilt)


def _normalize_one(value, baseline, scale, deadband):
    if value is None:
        return 0.0
    centered = value - baseline
    if abs(centered) < deadband:
        return 0.0
    scale = float(scale) if scale else 1.0
    if scale == 0:
        return 0.0
    norm = centered / scale
    return max(-1.0, min(1.0, norm))
