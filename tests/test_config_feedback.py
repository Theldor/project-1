import copy
import sys
import unittest
from pathlib import Path

# Keep tests runnable without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spine.config import DEFAULT_CONFIG, normalize_config


class ConfigFeedbackTests(unittest.TestCase):
    def test_overlay_disable_input_defaults_true(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["software_feedback"]["overlay"].pop("disable_input", None)

        normalize_config(config)

        self.assertTrue(config["software_feedback"]["overlay"]["disable_input"])

    def test_overlay_disable_input_can_be_disabled_explicitly(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["software_feedback"]["overlay"]["disable_input"] = False

        normalize_config(config)

        self.assertFalse(config["software_feedback"]["overlay"]["disable_input"])


if __name__ == "__main__":
    unittest.main()
