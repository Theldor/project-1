import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spine.mapping import SpineMapper
from spine.metrics import NormalizedMetrics


class HunchPushMappingTests(unittest.TestCase):
    def test_hunch_push_relaxes_when_upright(self):
        mapper = SpineMapper(
            {
                "control_mode": "hunch_push",
                "segments": 2,
                "neutral_angles": [0.0, 0.0],
                "min_angles": [-35.0, -35.0],
                "max_angles": [35.0, 35.0],
                "weights": [1.0, 1.0],
                "directions": [1.0, -1.0],
                "upper_weights": [1.0, 1.0],
                "upper_segment_start": 1,
                "max_deg_per_sec": 10000.0,
                "hunch_push": {
                    "source": "neck",
                    "use_absolute": True,
                    "activation_threshold": 0.1,
                    "full_scale": 0.8,
                    "max_push_deg": 30.0,
                },
            }
        )
        metrics = NormalizedMetrics(lean=0.0, neck=0.0, tilt=0.0)
        angles = mapper.map_metrics(metrics, now=1.0)
        self.assertEqual(angles, [0.0, 0.0])

    def test_hunch_push_drives_two_arms_inward_when_hunched(self):
        mapper = SpineMapper(
            {
                "control_mode": "hunch_push",
                "segments": 2,
                "neutral_angles": [0.0, 0.0],
                "min_angles": [-35.0, -35.0],
                "max_angles": [35.0, 35.0],
                "weights": [1.0, 1.0],
                "directions": [1.0, -1.0],
                "upper_weights": [1.0, 1.0],
                "upper_segment_start": 1,
                "max_deg_per_sec": 10000.0,
                "hunch_push": {
                    "source": "neck",
                    "use_absolute": True,
                    "activation_threshold": 0.1,
                    "full_scale": 0.8,
                    "max_push_deg": 30.0,
                },
            }
        )
        mapper.map_metrics(NormalizedMetrics(0.0, 0.0, 0.0), now=1.0)
        angles = mapper.map_metrics(NormalizedMetrics(0.0, 0.8, 0.0), now=2.0)
        self.assertEqual(angles, [30.0, -30.0])

    def test_hunch_push_uses_absolute_neck_signal(self):
        mapper = SpineMapper(
            {
                "control_mode": "hunch_push",
                "segments": 2,
                "neutral_angles": [0.0, 0.0],
                "min_angles": [-35.0, -35.0],
                "max_angles": [35.0, 35.0],
                "weights": [1.0, 1.0],
                "directions": [1.0, -1.0],
                "upper_weights": [1.0, 1.0],
                "upper_segment_start": 1,
                "max_deg_per_sec": 10000.0,
                "hunch_push": {
                    "source": "neck",
                    "use_absolute": True,
                    "activation_threshold": 0.1,
                    "full_scale": 0.8,
                    "max_push_deg": 30.0,
                },
            }
        )
        mapper.map_metrics(NormalizedMetrics(0.0, 0.0, 0.0), now=1.0)
        angles_positive = mapper.map_metrics(NormalizedMetrics(0.0, 0.6, 0.0), now=2.0)
        angles_negative = mapper.map_metrics(NormalizedMetrics(0.0, -0.6, 0.0), now=3.0)
        self.assertEqual(angles_positive, angles_negative)


if __name__ == "__main__":
    unittest.main()
