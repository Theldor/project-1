import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Keep tests runnable without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spine.pose import PoseEstimator


class _FakePose:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass


class PoseBackendTests(unittest.TestCase):
    def test_pose_estimator_uses_solutions_pose(self):
        fake_mp = SimpleNamespace(
            solutions=SimpleNamespace(pose=SimpleNamespace(Pose=_FakePose))
        )
        config = {
            "model_complexity": 1,
            "min_detection_confidence": 0.6,
            "min_tracking_confidence": 0.7,
        }

        with patch("spine.pose.mp", fake_mp):
            estimator = PoseEstimator(config)

        self.assertIs(estimator.mp_pose, fake_mp.solutions.pose)
        self.assertIsInstance(estimator.pose, _FakePose)
        self.assertEqual(estimator.pose.kwargs["model_complexity"], 1)
        self.assertEqual(estimator.pose.kwargs["min_detection_confidence"], 0.6)
        self.assertEqual(estimator.pose.kwargs["min_tracking_confidence"], 0.7)

    def test_pose_estimator_fails_with_actionable_error_when_solutions_missing(self):
        fake_mp = SimpleNamespace(__version__="0.10.32")

        with patch("spine.pose.mp", fake_mp):
            with self.assertRaises(RuntimeError) as ctx:
                PoseEstimator({})

        message = str(ctx.exception)
        self.assertIn("mp.solutions.pose", message)
        self.assertIn("version='0.10.32'", message)
        self.assertIn("pip install --upgrade", message)
        self.assertIn("mediapipe==", message)


if __name__ == "__main__":
    unittest.main()
