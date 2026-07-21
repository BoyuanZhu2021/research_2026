from __future__ import annotations

import copy
import unittest

from h1_inprocess_curriculum_analyze import analyze


def panel(arm: str, seed: int, successes: int, phi: float = 0.0) -> dict:
    by_goal = {}
    remaining = successes
    for index in range(12):
        count = min(4, remaining)
        remaining -= count
        by_goal[f"goal-{index}"] = {
            "asr": count / 4,
            "mean_max_phi": max(phi, count / 4),
        }
    return {
        "arm": arm, "seed": seed, "run_dir": "x",
        "payload_sha256": (arm[0] + str(seed)).ljust(64, "a"),
        "by_goal": by_goal,
    }


class AnalyzeTest(unittest.TestCase):
    def test_seed0_stop(self):
        value = analyze([
            panel("base", 0, 8), panel("dense", 0, 7), panel("sparse", 0, 8),
        ], bootstrap_samples=100)
        self.assertEqual(value["verdict"], "SEED0_STOP")

    def test_two_seed_supported(self):
        panels = []
        for seed in (0, 1):
            panels.extend([
                panel("base", seed, 4), panel("dense", seed, 20), panel("sparse", seed, 8),
            ])
        value = analyze(panels, bootstrap_samples=1_000)
        self.assertEqual(value["verdict"], "PRELIMINARY_SUPPORTED")

    def test_incomplete_panel_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze([panel("base", 0, 1), panel("dense", 0, 2)], bootstrap_samples=10)

    def test_targeted_preliminary_support_requires_dense_update(self):
        panels = [
            panel("base", 0, 2), panel("dense", 0, 5), panel("sparse", 0, 3),
        ]
        for value in panels:
            value["curriculum_variant"] = "gate-partial-targeted-v1"
            value["optimizer_updates"] = 1 if value["arm"] == "dense" else 0
        result = analyze(panels, bootstrap_samples=100)
        self.assertEqual(
            result["verdict"],
            "PRELIMINARY_H1_SUPPORTED_IN_GATE_PARTIAL_SUBSET",
        )


if __name__ == "__main__":
    unittest.main()
