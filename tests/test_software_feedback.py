import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import mediapipe as mp

# Keep tests runnable without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spine.feedback import FaceProximitySignalMapper, PostureSignalMapper, compute_face_scale
from spine.metrics import NormalizedMetrics, compute_metrics


def _landmarks():
    total = len(mp.solutions.pose.PoseLandmark)
    return [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(total)]


def _face_landmarks(scale):
    landmarks = _landmarks()
    pose = mp.solutions.pose.PoseLandmark
    center_x = 0.5
    center_y = 0.45

    landmarks[pose.LEFT_EYE_INNER.value] = SimpleNamespace(
        x=center_x - scale / 2.0, y=center_y, visibility=1.0
    )
    landmarks[pose.RIGHT_EYE_INNER.value] = SimpleNamespace(
        x=center_x + scale / 2.0, y=center_y, visibility=1.0
    )
    landmarks[pose.LEFT_EYE.value] = SimpleNamespace(
        x=center_x - scale * 0.6, y=center_y - 0.01, visibility=1.0
    )
    landmarks[pose.RIGHT_EYE.value] = SimpleNamespace(
        x=center_x + scale * 0.6, y=center_y - 0.01, visibility=1.0
    )
    landmarks[pose.MOUTH_LEFT.value] = SimpleNamespace(
        x=center_x - scale * 0.45, y=center_y + 0.08, visibility=1.0
    )
    landmarks[pose.MOUTH_RIGHT.value] = SimpleNamespace(
        x=center_x + scale * 0.45, y=center_y + 0.08, visibility=1.0
    )
    landmarks[pose.LEFT_EAR.value] = SimpleNamespace(
        x=center_x - scale, y=center_y - 0.02, visibility=1.0
    )
    landmarks[pose.RIGHT_EAR.value] = SimpleNamespace(
        x=center_x + scale, y=center_y - 0.02, visibility=1.0
    )
    return landmarks


class SoftwareFeedbackTests(unittest.TestCase):
    def test_signal_mapper_scales_between_thresholds(self):
        mapper = PostureSignalMapper(
            {
                "source": "neck",
                "use_absolute": True,
                "activation_threshold": 0.2,
                "full_scale": 0.8,
                "smoothing_alpha": 1.0,
            }
        )

        self.assertAlmostEqual(mapper.map_metrics(NormalizedMetrics(0.0, 0.1, 0.0)), 0.0)
        self.assertAlmostEqual(mapper.map_metrics(NormalizedMetrics(0.0, 0.5, 0.0)), 0.5)
        self.assertAlmostEqual(mapper.map_metrics(NormalizedMetrics(0.0, 0.8, 0.0)), 1.0)

    def test_signal_mapper_can_ignore_negative_direction(self):
        mapper = PostureSignalMapper(
            {
                "source": "neck",
                "use_absolute": False,
                "activation_threshold": 0.2,
                "full_scale": 0.8,
                "smoothing_alpha": 1.0,
            }
        )

        self.assertAlmostEqual(mapper.map_metrics(NormalizedMetrics(0.0, -0.9, 0.0)), 0.0)
        self.assertAlmostEqual(mapper.map_metrics(NormalizedMetrics(0.0, 0.8, 0.0)), 1.0)

    def test_neck_metric_falls_back_to_upper_body_when_hips_missing(self):
        landmarks = _landmarks()
        pose = mp.solutions.pose.PoseLandmark

        left_shoulder = pose.LEFT_SHOULDER.value
        right_shoulder = pose.RIGHT_SHOULDER.value
        left_ear = pose.LEFT_EAR.value
        right_ear = pose.RIGHT_EAR.value

        landmarks[left_shoulder] = SimpleNamespace(x=0.45, y=0.58, visibility=1.0)
        landmarks[right_shoulder] = SimpleNamespace(x=0.55, y=0.58, visibility=1.0)
        landmarks[left_ear] = SimpleNamespace(x=0.42, y=0.40, visibility=0.1)
        landmarks[right_ear] = SimpleNamespace(x=0.67, y=0.40, visibility=1.0)

        metrics = compute_metrics(landmarks, visibility_threshold=0.5, allow_upper_body_neck_fallback=True)

        self.assertIsNone(metrics.lean_deg)
        self.assertIsNotNone(metrics.neck_deg)
        self.assertGreater(metrics.neck_deg, 0.0)

    def test_neck_metric_requires_hips_when_fallback_disabled(self):
        landmarks = _landmarks()
        pose = mp.solutions.pose.PoseLandmark

        left_shoulder = pose.LEFT_SHOULDER.value
        right_shoulder = pose.RIGHT_SHOULDER.value
        right_ear = pose.RIGHT_EAR.value

        landmarks[left_shoulder] = SimpleNamespace(x=0.45, y=0.58, visibility=1.0)
        landmarks[right_shoulder] = SimpleNamespace(x=0.55, y=0.58, visibility=1.0)
        landmarks[right_ear] = SimpleNamespace(x=0.67, y=0.40, visibility=1.0)

        metrics = compute_metrics(landmarks, visibility_threshold=0.5, allow_upper_body_neck_fallback=False)

        self.assertIsNone(metrics.neck_deg)

    def test_face_scale_increases_when_face_gets_closer(self):
        far_scale = compute_face_scale(_face_landmarks(0.08), visibility_threshold=0.5)
        near_scale = compute_face_scale(_face_landmarks(0.16), visibility_threshold=0.5)
        self.assertIsNotNone(far_scale)
        self.assertIsNotNone(near_scale)
        self.assertGreater(near_scale, far_scale)

    def test_face_proximity_mapper_stays_clear_until_baseline_collected(self):
        mapper = FaceProximitySignalMapper(
            {
                "activation_threshold": 1.1,
                "full_scale": 1.3,
                "smoothing_alpha": 1.0,
                "face_proximity": {
                    "visibility_threshold": 0.5,
                    "baseline_samples": 3,
                    "baseline_alpha": 0.2,
                },
            }
        )
        self.assertEqual(mapper.map_landmarks(_face_landmarks(0.10)), 0.0)
        self.assertEqual(mapper.map_landmarks(_face_landmarks(0.10)), 0.0)
        self.assertEqual(mapper.map_landmarks(_face_landmarks(0.10)), 0.0)
        self.assertGreater(mapper.map_landmarks(_face_landmarks(0.13)), 0.9)


if __name__ == "__main__":
    unittest.main()
