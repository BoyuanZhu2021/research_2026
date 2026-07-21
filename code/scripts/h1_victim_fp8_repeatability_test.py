"""Pure-CPU regression tests for the active FP8-only victim proof."""
from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1_victim_fp8_repeatability as fp8  # noqa: E402
import h1_victim_quant_spotcheck as base  # noqa: E402
from h1_victim_quant_spotcheck_test import _add_valid_first_retrieve, _artifact  # noqa: E402


class FP8RepeatabilityTests(unittest.TestCase):
    def _write_proof(self, root: Path, *, drift: bool = False) -> tuple[Path, Path, dict]:
        fp8_a = _artifact("fp8", lifecycle_offset=0)
        fp8_b = _artifact("fp8", lifecycle_offset=100)
        if drift:
            fp8_b = _add_valid_first_retrieve(fp8_b)
        for name, document in (("fp8_a.json", fp8_a), ("fp8_b.json", fp8_b)):
            (root / name).write_text(
                json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
            )
        repeatability = fp8.compare_fp8_documents(fp8_a, fp8_b)
        repeatability_path = root / "repeatability.json"
        repeatability_path.write_text(
            json.dumps(repeatability, indent=2, sort_keys=True), encoding="utf-8"
        )
        cycle = base._seal({
            "schema_version": 1,
            "kind": fp8.CYCLE_KIND,
            "completed_at": "2026-07-18T00:00:00+00:00",
            "status": "FP8_RESTORED" if drift else "FP8_REPEATABILITY_VERIFIED",
            "comparison_exit_code": 1 if drift else 0,
            "primary_error": None,
            "fp8_manifest_verified": True,
            "fp8_api_verified": True,
            "fp8_process_verified": True,
            "repeatability_payload_sha256": repeatability["payload_sha256"],
            "fp8_a_payload_sha256": repeatability["fp8_a_payload_sha256"],
            "fp8_b_payload_sha256": repeatability["fp8_b_payload_sha256"],
            "restored_fp8_runtime": base.runtime_reference(fp8_b["victim"]["manifest"]),
        })
        cycle_path = root / "cycle_status.json"
        cycle_path.write_text(json.dumps(cycle, indent=2, sort_keys=True), encoding="utf-8")
        return repeatability_path, cycle_path, repeatability

    def test_two_distinct_fp8_lifecycles_are_repeatable(self):
        result = fp8.compare_fp8_documents(
            _artifact("fp8"), _artifact("fp8", lifecycle_offset=100)
        )
        self.assertEqual(result["status"], "FP8_REPEATABLE")
        self.assertFalse(result["material"])
        self.assertNotEqual(result["fp8_a_process"], result["fp8_b_process"])

    def test_same_process_lifecycle_is_rejected(self):
        artifact = _artifact("fp8")
        with self.assertRaisesRegex(base.SpotcheckError, "same process lifecycle"):
            fp8.compare_fp8_documents(artifact, copy.deepcopy(artifact))

    def test_any_ordered_decision_or_phi_change_is_material(self):
        result = fp8.compare_fp8_documents(
            _artifact("fp8"),
            _add_valid_first_retrieve(_artifact("fp8", lifecycle_offset=100)),
        )
        self.assertEqual(result["status"], "RUNTIME_DRIFT")
        self.assertTrue(result["material"])
        self.assertIn("ordered_decisions", result["differences"][0]["differences"])

    def test_loader_recomputes_sources_and_binds_active_b_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            repeatability_path, cycle_path, repeatability = self._write_proof(Path(temporary))
            proof = fp8.load_clean_fp8_repeatability(
                repeatability_path, cycle_path, expected_deployment_tree="a" * 64
            )
            self.assertEqual(proof["status"], "FP8_REPEATABLE")
            self.assertEqual(
                proof["repeatability_payload_sha256"], repeatability["payload_sha256"]
            )
            self.assertEqual(
                proof["runtime_bundle"]["restored_fp8_runtime"]["process"]["pid"],
                repeatability["fp8_b_process"]["pid"],
            )
            self.assertEqual(
                [item[0] for item in fp8.fresh_fp8_artifact_registry(proof)],
                ["repeatability", "cycle_status", "fp8_a", "fp8_b"],
            )

    def test_loader_rejects_drift_wrong_tree_tamper_and_old_quant_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            repeatability_path, cycle_path, _ = self._write_proof(Path(temporary), drift=True)
            with self.assertRaises(base.SpotcheckError):
                fp8.load_clean_fp8_repeatability(
                    repeatability_path, cycle_path, expected_deployment_tree="a" * 64
                )
        with tempfile.TemporaryDirectory() as temporary:
            repeatability_path, cycle_path, _ = self._write_proof(Path(temporary))
            with self.assertRaises(base.SpotcheckError):
                fp8.load_clean_fp8_repeatability(
                    repeatability_path, cycle_path, expected_deployment_tree="b" * 64
                )
            document = base._load_json(repeatability_path)
            document["status"] = "RUNTIME_DRIFT"
            repeatability_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(base.SpotcheckError):
                fp8.load_clean_fp8_repeatability(
                    repeatability_path, cycle_path, expected_deployment_tree="a" * 64
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = base.compare_documents(_artifact("fp8"), _artifact("bf16"))
            old_path = root / "repeatability.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            (root / "cycle_status.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(base.SpotcheckError):
                fp8.load_clean_fp8_repeatability(
                    old_path, root / "cycle_status.json", expected_deployment_tree="a" * 64
                )

    def test_loader_rejects_runtime_not_bound_to_fp8_b(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repeatability_path, cycle_path, _ = self._write_proof(root)
            cycle = base._load_json(cycle_path)
            fp8_a = base._load_json(root / "fp8_a.json")
            cycle["restored_fp8_runtime"] = base.runtime_reference(
                fp8_a["victim"]["manifest"]
            )
            cycle_path.write_text(
                json.dumps(base._seal(cycle), indent=2, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(base.SpotcheckError, "not the measured B"):
                fp8.load_clean_fp8_repeatability(
                    repeatability_path, cycle_path, expected_deployment_tree="a" * 64
                )

    def test_cycle_dry_run_explicitly_excludes_bf16(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = fp8.main(["cycle", "--out-dir", "unused"])
        self.assertEqual(code, 0)
        payload = json.loads(stream.getvalue())
        self.assertFalse(payload["bf16_started"])
        self.assertTrue(all("BF16" not in step for step in payload["plan"]))


if __name__ == "__main__":
    unittest.main()
