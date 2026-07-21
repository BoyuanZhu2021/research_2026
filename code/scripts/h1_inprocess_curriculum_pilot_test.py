from __future__ import annotations

import ast
import unittest
from pathlib import Path


WORKER = Path(__file__).with_name("h1_inprocess_curriculum_pilot.py")
TRAINER = Path(__file__).with_name("h1_mt_grpo_train_h20.py")


class CurriculumWorkerSourceTest(unittest.TestCase):
    def test_worker_is_gpu_guarded_and_keeps_raw_victim_ledgers(self):
        source = WORKER.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("AUTHORIZED_INSTANCE", source)
        self.assertIn("LocalVllmVictimClient", source)
        self.assertIn('"train_victim_ledger.jsonl"', source)
        self.assertIn('"eval_victim_ledger.jsonl"', source)
        self.assertIn("require_complete=True", source)
        self.assertNotIn('load_goals("final_ood"', source)
        self.assertIn('"n_goals": config["training"]["goals_per_step"]', source)
        self.assertIn('config["training"]["steps_per_arm"]', source)
        self.assertIn('domain.load_goals(source_split', source)
        self.assertIn("exact_outer_inject_unwrapped", source)
        self.assertIn("raw_model_text", source)
        self.assertIn("raw_text_sha256", source)
        self.assertIn("durably recorded before parsing", source)
        self.assertIn("paired panel deployment tree differs", source)
        self.assertIn("paired panel curriculum config differs", source)
        self.assertIn('config["interaction"]["defense_tier"]', source)
        self.assertIn('config["interaction"][\n        "victim_decision_protocol_id"', source)
        self.assertNotIn('defense_tier="light"', source)
        self.assertNotIn("LOCAL_COMPACT_PROTOCOL_ID", source)
        self.assertIn("strict_declared_action_arguments=strict_declared_action_arguments", source)
        self.assertIn("final_answer_max_length=final_answer_max_length", source)
        self.assertIn("action_string_max_length=action_string_max_length", source)
        self.assertIn("paired panel final-answer bound differs", source)
        self.assertIn("paired panel action-string bound differs", source)
        self.assertIn('deployment_tree_sha256=deployment["deployed_tree_sha256"]', source)

    def test_shape_override_is_bounded_by_formal_ceiling(self):
        source = TRAINER.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("training shape override exceeds frozen ceiling", source)
        self.assertIn('set(shape_override) != set(shape)', source)


if __name__ == "__main__":
    unittest.main()
