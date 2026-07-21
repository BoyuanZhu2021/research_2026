from __future__ import annotations

import copy
import unittest
from pathlib import Path

from .inprocess_curriculum_protocol import (
    build_balanced_schedule,
    build_run_config,
    load_config,
    validate_seal,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "h1_partial_reachable_curriculum_v1.json"
TARGETED_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "h1_gate_partial_curriculum_v1.json"
TARGETED_NONE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "h1_gate_partial_none_curriculum_v1.json"
)
TARGETED_LEGACY_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "h1_gate_partial_legacy_curriculum_v1.json"
)


class CurriculumProtocolTest(unittest.TestCase):
    def test_config_and_balanced_paired_schedule(self):
        config = load_config(CONFIG)
        for seed in (0, 1):
            schedule = build_balanced_schedule(seed=seed)
            self.assertEqual(schedule, build_balanced_schedule(seed=seed))
            self.assertEqual(len(schedule), 12)
            self.assertEqual(
                sorted(index for row in schedule for index in row),
                sorted(list(range(12)) * 4),
            )
        self.assertTrue(set(config["data"]["training_goal_ids"]).isdisjoint(
            config["data"]["heldout_goal_ids"]
        ))

    def test_run_config_is_sealed(self):
        config = load_config(CONFIG)
        digest = "a" * 64
        value = build_run_config(
            config=config, arm="dense", seed=0, tag="x", gpu_uuid="GPU-x",
            deployment_tree_sha256=digest,
            service_manifest_payload_sha256=digest,
            gate_selection_payload_sha256=digest,
            initial_lora_sha256=digest,
        )
        self.assertEqual(validate_seal(value), value)
        broken = copy.deepcopy(value)
        broken["seed"] = 1
        with self.assertRaisesRegex(ValueError, "seal"):
            validate_seal(broken)

    def test_targeted_config_and_eight_step_schedule(self):
        config = load_config(TARGETED_CONFIG)
        schedule = build_balanced_schedule(seed=0, n_goals=8)
        self.assertEqual(len(schedule), 8)
        self.assertEqual(
            sorted(index for row in schedule for index in row),
            sorted(list(range(8)) * 4),
        )
        self.assertEqual(config["training"]["steps_per_arm"], 8)
        self.assertEqual(len(config["data"]["heldout_goal_ids"]), 12)
        self.assertEqual(config["data"]["source_split"], "calibration")

    def test_none_tier_targeted_config_reuses_the_frozen_panel(self):
        light = load_config(TARGETED_CONFIG)
        none = load_config(TARGETED_NONE_CONFIG)
        self.assertEqual(none["curriculum_variant"], "gate-partial-none-targeted-v1")
        self.assertEqual(none["interaction"]["defense_tier"], "none")
        self.assertEqual(
            none["data"]["training_goal_ids"], light["data"]["training_goal_ids"]
        )
        self.assertEqual(
            none["data"]["heldout_goal_ids"], light["data"]["heldout_goal_ids"]
        )

    def test_legacy_protocol_targeted_config_reuses_the_frozen_panel(self):
        light = load_config(TARGETED_CONFIG)
        legacy = load_config(TARGETED_LEGACY_CONFIG)
        self.assertEqual(
            legacy["curriculum_variant"], "gate-partial-legacy-targeted-v1"
        )
        self.assertEqual(legacy["interaction"]["defense_tier"], "light")
        self.assertEqual(
            legacy["interaction"]["victim_decision_protocol_id"],
            "h1-victim-one-decision-step-bound-observation-ref-v3",
        )
        self.assertIs(
            legacy["interaction"]["strict_declared_action_arguments"], True
        )
        self.assertEqual(legacy["interaction"]["final_answer_max_length"], 512)
        self.assertEqual(legacy["interaction"]["action_string_max_length"], 512)
        self.assertEqual(
            legacy["data"]["training_goal_ids"], light["data"]["training_goal_ids"]
        )
        self.assertEqual(
            legacy["data"]["heldout_goal_ids"], light["data"]["heldout_goal_ids"]
        )


if __name__ == "__main__":
    unittest.main()
