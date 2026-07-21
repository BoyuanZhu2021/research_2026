from __future__ import annotations

import ast
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("h1_inprocess_curriculum_controller.py")


class ControllerSourceTest(unittest.TestCase):
    def test_controller_is_instance_bounded_and_hash_recovers(self):
        source = PATH.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("AUTHORIZED_INSTANCE", source)
        self.assertIn("download_and_verify", source)
        self.assertIn("controller log recovery hash mismatch", source)
        self.assertIn("REMOTE_SERVICE_MANAGER", source)
        self.assertIn('"orphan-stop"', source)
        self.assertIn("VICTIM_SERVICE_START_LAUNCHED", source)
        self.assertIn("REMOTE_TARGETED_NONE_CONFIG", source)
        self.assertIn('"--targeted-none"', source)
        self.assertIn("REMOTE_TARGETED_LEGACY_CONFIG", source)
        self.assertIn('"--targeted-legacy"', source)
        self.assertNotIn("fa85409945-b6dee8ab", source)
        self.assertNotIn("V100", source)


if __name__ == "__main__":
    unittest.main()
