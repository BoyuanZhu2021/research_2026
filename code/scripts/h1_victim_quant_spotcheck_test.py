"""Pure-CPU regression tests for the victim quantization spot-check safety contract."""
from __future__ import annotations

import copy
import errno
import io
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1_victim_quant_spotcheck as spot  # noqa: E402
from src.h20_serving_identity import (  # noqa: E402
    validate_fresh_h20_campaign_runtime_proof as validate_portable_fresh_proof,
)


def _service_manifest(quantization: str, *, pid: int = 123, start_ticks: int = 456) -> dict:
    gpu = {
        "index": 0,
        "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "name": "NVIDIA H20",
        "memory_total_mib": 97871,
    }
    command = spot._serve_command(quantization)
    process = {
        "pid": pid,
        "start_time_ticks": start_ticks,
        "cmdline": command,
        "cmdline_sha256": spot._sha256_json(command),
        "environ_sha256": "e" * 64,
        "environment": spot.h20_expected_environment(quantization, gpu["uuid"]),
        "gpu_uuid": gpu["uuid"],
    }
    return spot.build_service_manifest(
        quantization=quantization,
        gpu=gpu,
        process=process,
        service={
            "endpoint": spot.VICTIM_URL,
            "model_ids": [spot.VICTIM_SERVED_NAME],
            "observed_at": "2026-07-16T00:00:00+00:00",
        },
        sealed_at="2026-07-16T00:00:00+00:00",
    )


def _artifact(quantization: str, *, lifecycle_offset: int = 0) -> dict:
    domain, selected = spot._load_frozen_cases()
    records = []
    for goal, injection in selected:
        response = spot.json.dumps({
            "kind": "final", "answer": "I will not follow that instruction.",
        }, sort_keys=True)
        base_messages = domain.build_victim_messages(
            goal, domain.initial_scratchpad(goal, injection)
        )
        contract = spot.build_request_contract(
            base_messages, spot.goal_tool_schemas(goal)
        )
        transcript = "Final Answer: I will not follow that instruction."
        oracle = domain.score(goal, [transcript], [])
        records.append({
            "goal_id": goal.id,
            "status": "valid",
            "input": {
                "injection": injection,
                "user_instruction": goal.meta["User Instruction"],
                "user_tool": goal.meta["User Tool"],
                "target_tools": list(goal.meta["target_tools"]),
                "intended_recipient": goal.meta["intended_recipient"],
                "canary": goal.meta["canary"],
                "tool_schemas": goal.meta["tool_schemas"],
            },
            "steps": [{
                "step": 0,
                "messages": contract["messages"],
                "structured_schema_sha256": contract["schema_sha256"],
                "response": response,
                "parsed_calls": [],
                "stop_reason": "no_action",
            }],
            "victim_transcript": transcript,
            "calls": [],
            "ordered_decisions": [],
            "oracle": oracle,
            "max_phi": float(oracle["phi"]),
            "success": bool(oracle["security"]),
        })
    manifest = _service_manifest(
        quantization,
        pid=(123 if quantization == "fp8" else 124) + lifecycle_offset,
        start_ticks=(456 if quantization == "fp8" else 457) + lifecycle_offset,
    )
    process = manifest["process"]
    document = {
        "schema_version": spot.SCHEMA_VERSION,
        "kind": spot.KIND,
        "created_at": "2026-07-16T00:00:00+00:00",
        "status": "valid",
        "quantization": quantization,
        "victim": {
            "hf_model": spot.VICTIM_HF_MODEL,
            "revision": spot.VICTIM_REVISION,
            "served_model": spot.VICTIM_SERVED_NAME,
            "url": spot.VICTIM_URL,
            "manifest": manifest,
            "process_pid": process["pid"],
            "process_start_time_ticks": process["start_time_ticks"],
            "process_command": spot._serve_command(quantization),
            "process_cmdline_sha256": process["cmdline_sha256"],
            "process_environ_sha256": process["environ_sha256"],
            "gpu_uuid": process["gpu_uuid"],
        },
        "deployment": {
            "deployed_tree_sha256": "a" * 64,
            "injecagent_commit": spot.INJECAGENT_COMMIT,
        },
        "dataset": {
            "split_manifest_id": "injecagent-ds-base-split-v1",
            "split_manifest_sha256": "c" * 64,
            "dataset_sha256": "d" * 64,
            "split": "calibration",
        },
        "oracle_version": spot.ORACLE_VERSION,
        "case_set": spot._expected_case_set(),
        "config": dict(spot.QUANT_CONFIG),
        "cases": records,
        "error": None,
    }
    return spot._seal(document)


def _add_valid_first_retrieve(document: dict) -> dict:
    changed = copy.deepcopy(document)
    domain, selected = spot._load_frozen_cases()
    goal = selected[0][0]
    tool = goal.meta["target_tools"][0]
    raw_args = "{}"
    decision = spot._decision(tool, raw_args)
    injection = selected[0][1]
    first_response = spot.json.dumps({
        "kind": "action", "tool": tool, "arguments": {},
    }, sort_keys=True)
    scratchpad = domain.initial_scratchpad(goal, injection)
    first_contract = spot.build_request_contract(
        domain.build_victim_messages(goal, scratchpad), spot.goal_tool_schemas(goal)
    )
    observation = domain.mock_observation(goal, tool, raw_args)
    scratchpad += (
        f"Action: {tool}\nAction Input: {raw_args}\nObservation: {observation}\n"
    )
    second_response = spot.json.dumps({
        "kind": "final", "answer": "I will not forward it.",
    }, sort_keys=True)
    second_contract = spot.build_request_contract(
        domain.build_victim_messages(goal, scratchpad),
        spot.goal_tool_schemas(goal),
        current_step=1,
    )
    transcript = (
        f"Action: {tool}\nAction Input: {raw_args}\n"
        "Final Answer: I will not forward it."
    )
    oracle = domain.score(goal, [transcript], [(tool, raw_args)])
    record = changed["cases"][0]
    record["steps"] = [
        {
            "step": 0,
            "messages": first_contract["messages"],
            "structured_schema_sha256": first_contract["schema_sha256"],
            "response": first_response,
            "parsed_calls": [{
                "tool": tool,
                "raw_args": raw_args,
                "decision": decision,
            }],
            "executed_call": {
                "tool": tool,
                "raw_args": raw_args,
                "decision": decision,
            },
            "observation": observation,
        },
        {
            "step": 1,
            "messages": second_contract["messages"],
            "structured_schema_sha256": second_contract["schema_sha256"],
            "response": second_response,
            "parsed_calls": [],
            "stop_reason": "no_action",
        },
    ]
    record["victim_transcript"] = transcript
    record["calls"] = [{"tool": tool, "raw_args": raw_args, "decision": decision}]
    record["ordered_decisions"] = [decision]
    record["oracle"] = oracle
    record["max_phi"] = float(oracle["phi"])
    record["success"] = bool(oracle["security"])
    return spot._seal(changed)


class QuantSpotcheckTests(unittest.TestCase):
    def _write_gate_artifacts(
        self, root: Path, *, material: bool = False, lifecycle_offset: int = 0,
    ) -> tuple[Path, Path, dict]:
        fp8 = _artifact("fp8", lifecycle_offset=lifecycle_offset)
        bf16 = _artifact("bf16", lifecycle_offset=lifecycle_offset)
        bf16 = _add_valid_first_retrieve(bf16) if material else bf16
        (root / "fp8.json").write_text(
            spot.json.dumps(fp8, indent=2, sort_keys=True), encoding="utf-8"
        )
        (root / "bf16.json").write_text(
            spot.json.dumps(bf16, indent=2, sort_keys=True), encoding="utf-8"
        )
        comparison = spot.compare_documents(fp8, bf16)
        comparison_path = root / "comparison.json"
        comparison_path.write_text(
            spot.json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )
        cycle = spot._seal({
            "schema_version": spot.SCHEMA_VERSION,
            "kind": "h1_victim_quant_cycle_status",
            "completed_at": "2026-07-16T00:01:00+00:00",
            "status": "FP8_RESTORED",
            "comparison_exit_code": 1 if material else 0,
            "primary_error": None,
            "fp8_manifest_verified": True,
            "fp8_api_verified": True,
            "fp8_process_verified": True,
            "comparison_payload_sha256": comparison["payload_sha256"],
            "fp8_payload_sha256": comparison["fp8_payload_sha256"],
            "bf16_payload_sha256": comparison["bf16_payload_sha256"],
            "restored_fp8_runtime": spot.runtime_reference(fp8["victim"]["manifest"]),
        })
        cycle_path = root / "cycle_status.json"
        cycle_path.write_text(spot.json.dumps(cycle, indent=2, sort_keys=True), encoding="utf-8")
        return comparison_path, cycle_path, comparison

    def test_frozen_cases_are_exact_calibration_inputs(self):
        _domain, selected = spot._load_frozen_cases()
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            [goal.id for goal, _ in selected],
            [goal_id for goal_id, _ in spot.FROZEN_CASES],
        )
        contexts = [goal.meta["User Tool"] for goal, _ in selected]
        self.assertEqual(contexts.count("AmazonGetProductDetails"), 2)
        self.assertEqual(contexts.count("GmailSearchEmails"), 2)
        self.assertEqual(contexts.count("WebBrowserNavigateTo"), 2)

    def test_process_identity_is_exact_and_quantization_aware(self):
        base = spot._serve_command("bf16")
        self.assertTrue(spot._process_identity_matches(base + ["--quantization", "fp8"], "fp8"))
        self.assertTrue(spot._process_identity_matches(base, "bf16"))
        self.assertFalse(spot._process_identity_matches(base, "fp8"))
        self.assertFalse(
            spot._process_identity_matches(
                [token if token != spot.VICTIM_REVISION else "wrong" for token in base], "bf16"
            )
        )
        self.assertFalse(
            spot._process_identity_matches(
                [token if token != str(spot.PORT) else "8001" for token in base], "bf16"
            )
        )
        self.assertFalse(
            spot._process_identity_matches(
                ["/tmp/vllm" if Path(token) == spot.VLLM_BIN else token for token in base], "bf16"
            )
        )
        self.assertFalse(
            spot._process_identity_matches(
                [token if token != "8192" else "4096" for token in base], "bf16"
            )
        )

    def test_process_selection_rejects_mismatch_and_duplicates(self):
        exact = {
            "pid": 10,
            "tokens": spot._serve_command("fp8"),
        }
        self.assertEqual(spot._select_exact_process([exact], "fp8")["pid"], 10)
        with self.assertRaises(spot.SpotcheckError):
            spot._select_exact_process([], "fp8")
        with self.assertRaises(spot.SpotcheckError):
            spot._select_exact_process([exact, {**exact, "pid": 11}], "fp8")
        with self.assertRaises(spot.SpotcheckError):
            spot._select_exact_process([exact], "bf16")

    def test_vllm_toolchain_requires_exact_executables_and_versions(self):
        with patch.object(Path, "is_file", return_value=False), patch.object(
            spot.os, "access", return_value=False
        ):
            with self.assertRaisesRegex(spot.SpotcheckError, "ninja executable missing"):
                spot._require_vllm_toolchain()

        cases = (
            ("metadata", ["1.13.1"], "metadata version mismatch"),
            (
                "binary",
                [spot.VLLM_NINJA_METADATA_VERSION, "1.12.0"],
                "binary version mismatch",
            ),
            (
                "nvcc",
                [
                    spot.VLLM_NINJA_METADATA_VERSION,
                    spot.VLLM_NINJA_BINARY_VERSION,
                    "Cuda compilation tools, release 12.7, V12.7.0",
                ],
                "nvcc release mismatch",
            ),
        )
        for label, outputs, message in cases:
            with self.subTest(label=label), patch.object(
                Path, "is_file", return_value=True
            ), patch.object(
                spot.os, "access", return_value=True
            ), patch.object(
                spot, "_run_toolchain_probe", side_effect=outputs
            ):
                with self.assertRaisesRegex(spot.SpotcheckError, message):
                    spot._require_vllm_toolchain()

        outputs = [
            spot.VLLM_NINJA_METADATA_VERSION,
            spot.VLLM_NINJA_BINARY_VERSION,
            f"Cuda compilation tools, release {spot.CUDA_NVCC_RELEASE}, V12.8.0",
            "c++ exact version",
        ]
        with patch.object(Path, "is_file", return_value=True), patch.object(
            spot.os, "access", return_value=True
        ), patch.object(
            spot, "_run_toolchain_probe", side_effect=outputs
        ):
            result = spot._require_vllm_toolchain()
        self.assertEqual(result["path"], spot.VLLM_TOOLCHAIN_PATH)
        self.assertEqual(result["cuda_home"], spot.CUDA_HOME)

    def test_start_sets_toolchain_and_probes_before_manifest(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        exact = {"pid": 99, "tokens": spot._serve_command("fp8")}
        launched = MagicMock(pid=99)
        launched.poll.return_value = None
        events = []
        with patch.object(Path, "is_file", return_value=True), patch.object(
            spot, "_require_h20_inventory", return_value={
                "index": 0,
                "uuid": expected["gpu_uuid"],
                "name": "NVIDIA H20",
                "memory_total_mib": 97871,
            }
        ), patch.object(spot, "_require_vllm_version"), patch.object(
            spot, "_require_vllm_toolchain"
        ) as toolchain, patch.object(
            spot, "_port_vllm_processes",
            side_effect=[[], [exact], [exact], [exact]],
        ), patch.object(
            spot, "_port_is_open", return_value=False
        ), patch.object(
            Path, "open", mock_open()
        ), patch.object(
            spot.subprocess, "Popen", return_value=launched
        ) as popen, patch.object(
            spot.time, "monotonic", side_effect=[0.0, 1.0, 2.0]
        ), patch.object(
            spot, "_api_models", return_value={spot.VICTIM_SERVED_NAME}
        ), patch.object(
            spot, "_startup_inference_probe",
            side_effect=lambda **_kwargs: events.append("probe") or {
                "completion_tokens": 1,
            },
        ) as probe, patch.object(
            spot, "_process_observation", return_value=expected
        ), patch.object(
            spot, "_write_current_manifest",
            side_effect=lambda *_args: events.append("manifest") or {},
        ) as writer, patch.object(spot, "_verify_current"):
            spot._start_exact_server("fp8", 600)
        toolchain.assert_called_once_with()
        probe.assert_called_once_with(timeout=300.0)
        writer.assert_called_once()
        self.assertEqual(events, ["probe", "manifest"])
        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(child_env["PATH"], spot.VLLM_TOOLCHAIN_PATH)
        self.assertEqual(child_env["CUDA_HOME"], spot.CUDA_HOME)

    def test_startup_inference_probe_requires_real_one_token_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = spot.json.dumps({
            "model": spot.VICTIM_SERVED_NAME,
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"completion_tokens": 1},
        }).encode("utf-8")
        with patch.object(spot.request, "urlopen", return_value=response) as urlopen:
            result = spot._startup_inference_probe(timeout=300)
        self.assertEqual(result["completion_tokens"], 1)
        request_body = spot.json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(request_body["max_tokens"], 1)
        self.assertEqual(request_body["temperature"], 0.0)
        self.assertFalse(request_body["stream"])

        http500 = spot.error.HTTPError(
            f"{spot.VICTIM_URL}/chat/completions", 500,
            "Internal Server Error", {}, io.BytesIO(b"EngineCore died"),
        )
        with patch.object(spot.request, "urlopen", side_effect=http500):
            with self.assertRaisesRegex(spot.SpotcheckError, "HTTP 500.*EngineCore died"):
                spot._startup_inference_probe(timeout=300)

    def test_behavior_probe_uses_structured_output_and_rejects_residual_marker(self):
        messages = [{
            "role": "user",
            "content": "Prior transcript:\nObservation: prompt-owned tool result",
        }]
        response = MagicMock()
        response.__enter__.return_value.read.return_value = spot.json.dumps({
            "choices": [{"message": {"content": "Thought: ok\nAction: GmailSearchEmails"}}],
        }).encode("utf-8")
        structured_outputs = {"json": {"type": "object"}}
        with patch.object(spot.request, "urlopen", return_value=response) as urlopen:
            self.assertEqual(
                spot._victim_chat(messages, structured_outputs),
                "Thought: ok\nAction: GmailSearchEmails",
            )
        request_body = spot.json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(request_body["messages"], messages)
        self.assertEqual(request_body["stop"], ["\nObservation:"])
        self.assertIs(request_body["include_stop_str_in_output"], False)
        self.assertEqual(request_body["structured_outputs"], structured_outputs)

        for content, rejected in (
            ("Action: GmailSearchEmails\nObservation: leaked", True),
            ("Action: GmailSearchEmails Observation: inline", False),
            (r"Action: GmailSearchEmails\nObservation: escaped", False),
        ):
            response = MagicMock()
            response.__enter__.return_value.read.return_value = spot.json.dumps({
                "choices": [{"message": {"content": content}}],
            }).encode("utf-8")
            with self.subTest(content=content), patch.object(
                spot.request, "urlopen", return_value=response
            ):
                if rejected:
                    with self.assertRaisesRegex(spot.SpotcheckError, "stop marker"):
                        spot._victim_chat(messages, structured_outputs)
                else:
                    self.assertEqual(
                        spot._victim_chat(messages, structured_outputs), content
                    )

    def test_probe_error_or_process_death_writes_no_manifest_and_cleans(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        exact = {"pid": 99, "tokens": spot._serve_command("fp8")}
        for label, probe_error, poll_results, message in (
            ("http500", spot.SpotcheckError("startup inference HTTP 500"), [None], "HTTP 500"),
            ("process-death", None, [None, 1], "died after startup inference"),
        ):
            launched = MagicMock(pid=99)
            launched.poll.side_effect = poll_results
            with self.subTest(label=label), patch.object(
                Path, "is_file", return_value=True
            ), patch.object(
                spot, "_require_h20_inventory", return_value={
                    "index": 0,
                    "uuid": expected["gpu_uuid"],
                    "name": "NVIDIA H20",
                    "memory_total_mib": 97871,
                }
            ), patch.object(spot, "_require_vllm_version"), patch.object(
                spot, "_require_vllm_toolchain"
            ), patch.object(
                spot, "_port_vllm_processes",
                side_effect=[[], [exact], [exact]],
            ), patch.object(
                spot, "_port_is_open", return_value=False
            ), patch.object(Path, "open", mock_open()), patch.object(
                spot.subprocess, "Popen", return_value=launched
            ), patch.object(
                spot.time, "monotonic", side_effect=[0.0, 1.0, 2.0]
            ), patch.object(
                spot, "_api_models", return_value={spot.VICTIM_SERVED_NAME}
            ), patch.object(
                spot, "_startup_inference_probe",
                side_effect=probe_error,
            ), patch.object(
                spot, "_cleanup_failed_launch"
            ) as cleanup, patch.object(
                spot, "_write_current_manifest"
            ) as writer:
                with self.assertRaisesRegex(spot.SpotcheckError, message):
                    spot._start_exact_server("fp8", 600)
            writer.assert_not_called()
            cleanup.assert_called_once_with(launched, "fp8")

    def test_failed_launch_cleanup_stops_only_exact_launched_lifecycle(self):
        exact = {"pid": 99, "tokens": spot._serve_command("fp8")}
        launched = MagicMock(pid=99)
        launched.poll.return_value = None
        with patch.object(
            spot, "_port_vllm_processes", side_effect=[[exact], []]
        ), patch.object(
            spot, "_stop_exact_server"
        ) as stop, patch.object(
            spot, "_port_is_open", return_value=False
        ), patch.object(
            spot, "VICTIM_MANIFEST", Path("missing-startup-manifest")
        ):
            spot._cleanup_failed_launch(launched, "fp8")
        stop.assert_called_once_with("fp8", require_manifest=False)

    def test_compare_identical_tool_behavior_is_clean(self):
        result = spot.compare_documents(_artifact("fp8"), _artifact("bf16"))
        self.assertEqual(result["status"], "NO_MATERIAL_DRIFT")
        self.assertFalse(result["material"])
        self.assertEqual(result["n_differences"], 0)

    def test_any_ordered_action_or_phi_change_is_material(self):
        fp8 = _artifact("fp8")
        bf16 = _add_valid_first_retrieve(_artifact("bf16"))
        result = spot.compare_documents(fp8, bf16)
        self.assertTrue(result["material"])
        self.assertEqual(result["n_differences"], 1)
        changed = result["differences"][0]["differences"]
        self.assertIn("ordered_decisions", changed)
        self.assertIn("max_phi", changed)

    def test_formal_gate_loader_requires_clean_comparison_and_restored_fp8(self):
        with tempfile.TemporaryDirectory() as temporary:
            comparison_path, cycle_path, comparison = self._write_gate_artifacts(
                Path(temporary)
            )
            proof = spot.load_clean_quant_comparison(
                comparison_path,
                cycle_path,
                expected_deployment_tree="a" * 64,
            )
            self.assertEqual(proof["status"], "NO_MATERIAL_DRIFT")
            self.assertEqual(proof["comparison_payload_sha256"], comparison["payload_sha256"])

            cycle = spot._load_json(cycle_path)
            cycle["fp8_process_verified"] = False
            cycle_path.write_text(
                spot.json.dumps(spot._seal(cycle), indent=2, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaises(spot.SpotcheckError):
                spot.load_clean_quant_comparison(
                    comparison_path, cycle_path, expected_deployment_tree="a" * 64
                )

    def test_formal_gate_loader_rejects_drift_tamper_and_other_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison_path, cycle_path, _ = self._write_gate_artifacts(root, material=True)
            with self.assertRaises(spot.SpotcheckError):
                spot.load_clean_quant_comparison(
                    comparison_path, cycle_path, expected_deployment_tree="a" * 64
                )

        with tempfile.TemporaryDirectory() as temporary:
            comparison_path, cycle_path, _ = self._write_gate_artifacts(Path(temporary))
            with self.assertRaises(spot.SpotcheckError):
                spot.load_clean_quant_comparison(
                    comparison_path, cycle_path, expected_deployment_tree="e" * 64
                )
            comparison = spot._load_json(comparison_path)
            comparison["status"] = "MATERIAL_DRIFT"  # do not reseal: tamper must be detected first
            comparison_path.write_text(
                spot.json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaises(spot.SpotcheckError):
                spot.load_clean_quant_comparison(
                    comparison_path, cycle_path, expected_deployment_tree="a" * 64
                )

    def test_tamper_or_wrong_quantization_is_invalid_not_a_behavior_sample(self):
        tampered = _artifact("fp8")
        tampered["cases"][0]["max_phi"] = 0.5
        with self.assertRaises(spot.SpotcheckError):
            spot.compare_documents(tampered, _artifact("bf16"))
        wrong_label = _artifact("bf16")
        with self.assertRaises(spot.SpotcheckError):
            spot._validate_artifact(wrong_label, "fp8")

        trace_tamper = _artifact("fp8")
        trace_tamper["cases"][0]["victim_transcript"] = "different from step response"
        trace_tamper = spot._seal(trace_tamper)  # structural trace checks, not only the outer seal
        with self.assertRaises(spot.SpotcheckError):
            spot._validate_artifact(trace_tamper, "fp8")

    def test_signal_rechecks_full_identity_and_start_ticks_before_pidfd_signal(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        wrong_command = copy.deepcopy(expected)
        wrong_command["cmdline"][wrong_command["cmdline"].index(
            spot.VICTIM_REVISION
        )] = "wrong-revision"
        reused_pid = copy.deepcopy(expected)
        reused_pid["start_time_ticks"] = 999
        for label, observed in (("command", wrong_command), ("pid-reuse", reused_pid)):
            with self.subTest(label=label), patch.object(
                spot, "_process_observation", return_value=observed
            ), patch.object(
                spot.signal, "pidfd_send_signal", create=True
            ) as sender:
                with self.assertRaises(spot.SpotcheckError):
                    spot._signal_exact(7, expected, "fp8", signal.SIGTERM)
            sender.assert_not_called()

        with patch.object(
            spot, "_process_observation", return_value=expected
        ), patch.object(
            spot.signal, "pidfd_send_signal", create=True
        ) as sender:
            self.assertTrue(spot._signal_exact(7, expected, "fp8", signal.SIGTERM))
        sender.assert_called_once_with(7, signal.SIGTERM, None, 0)

    def test_pidfd_signal_fallback_uses_only_x86_64_linux_syscall(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        syscall = MagicMock(return_value=0)
        libc = MagicMock()
        libc.syscall = syscall
        with patch.object(
            spot, "_process_observation", return_value=expected
        ), patch.object(
            spot.signal, "pidfd_send_signal", None, create=True
        ), patch.object(
            spot.sys, "platform", "linux"
        ), patch.object(
            spot.platform, "machine", return_value="x86_64"
        ), patch.object(
            spot.ctypes, "CDLL", return_value=libc
        ) as loader:
            self.assertTrue(spot._signal_exact(7, expected, "fp8", signal.SIGTERM))
        loader.assert_called_once_with("libc.so.6", use_errno=True)
        syscall.assert_called_once()
        arguments = syscall.call_args.args
        self.assertEqual([argument.value for argument in arguments], [
            spot._PIDFD_SEND_SIGNAL_SYSCALL_X86_64,
            7,
            int(signal.SIGTERM),
            None,
            0,
        ])

    def test_pidfd_signal_fallback_rejects_platform_arch_and_syscall_error(self):
        for label, runtime_platform, architecture, message in (
            ("platform", "win32", "x86_64", "requires Linux"),
            ("architecture", "linux", "aarch64", "only Linux x86_64"),
        ):
            with self.subTest(label=label), patch.object(
                spot.signal, "pidfd_send_signal", None, create=True
            ), patch.object(
                spot.sys, "platform", runtime_platform
            ), patch.object(
                spot.platform, "machine", return_value=architecture
            ), patch.object(spot.ctypes, "CDLL") as loader:
                with self.assertRaisesRegex(spot.SpotcheckError, message):
                    spot._send_pidfd_signal(7, signal.SIGTERM)
            loader.assert_not_called()

        syscall = MagicMock(return_value=-1)
        libc = MagicMock()
        libc.syscall = syscall
        with patch.object(
            spot.signal, "pidfd_send_signal", None, create=True
        ), patch.object(
            spot.sys, "platform", "linux"
        ), patch.object(
            spot.platform, "machine", return_value="x86_64"
        ), patch.object(
            spot.ctypes, "CDLL", return_value=libc
        ), patch.object(
            spot.ctypes, "get_errno", return_value=errno.EPERM
        ):
            with self.assertRaisesRegex(PermissionError, "Operation not permitted"):
                spot._send_pidfd_signal(7, signal.SIGTERM)

    def test_pid_reuse_blocks_syscall_fallback_before_libc_load(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        reused = copy.deepcopy(expected)
        reused["start_time_ticks"] = 999
        with patch.object(
            spot, "_process_observation", return_value=reused
        ), patch.object(
            spot.signal, "pidfd_send_signal", None, create=True
        ), patch.object(
            spot.sys, "platform", "linux"
        ), patch.object(
            spot.platform, "machine", return_value="x86_64"
        ), patch.object(spot.ctypes, "CDLL") as loader:
            with self.assertRaisesRegex(spot.SpotcheckError, "changed full identity"):
                spot._signal_exact(7, expected, "fp8", signal.SIGTERM)
        loader.assert_not_called()

    def test_pidfd_open_accepts_verified_syscall_fallback(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        libc = MagicMock()
        libc.syscall = MagicMock(return_value=17)
        with patch.object(
            spot, "_process_observation", side_effect=[expected, expected]
        ), patch.object(
            spot.os, "pidfd_open", None, create=True
        ), patch.object(
            spot.signal, "pidfd_send_signal", MagicMock(), create=True
        ), patch.object(
            spot.sys, "platform", "linux"
        ), patch.object(
            spot.platform, "machine", return_value="x86_64"
        ), patch.object(
            spot.ctypes, "CDLL", return_value=libc
        ) as loader:
            self.assertEqual(spot._open_exact_pidfd(expected, "fp8"), 17)
        loader.assert_called_once_with("libc.so.6", use_errno=True)
        arguments = libc.syscall.call_args.args
        self.assertEqual([argument.value for argument in arguments], [
            spot._PIDFD_OPEN_SYSCALL_X86_64, 99, 0,
        ])

    def test_pidfd_open_fallback_rejects_platform_arch_and_syscall_error(self):
        for label, runtime_platform, architecture, message in (
            ("platform", "win32", "x86_64", "requires Linux"),
            ("architecture", "linux", "aarch64", "only Linux x86_64"),
        ):
            with self.subTest(label=label), patch.object(
                spot.os, "pidfd_open", None, create=True
            ), patch.object(
                spot.sys, "platform", runtime_platform
            ), patch.object(
                spot.platform, "machine", return_value=architecture
            ), patch.object(spot.ctypes, "CDLL") as loader:
                with self.assertRaisesRegex(spot.SpotcheckError, message):
                    spot._open_pidfd(99, 0)
            loader.assert_not_called()

        syscall = MagicMock(return_value=-1)
        libc = MagicMock()
        libc.syscall = syscall
        with patch.object(
            spot.os, "pidfd_open", None, create=True
        ), patch.object(
            spot.sys, "platform", "linux"
        ), patch.object(
            spot.platform, "machine", return_value="x86_64"
        ), patch.object(
            spot.ctypes, "CDLL", return_value=libc
        ), patch.object(
            spot.ctypes, "get_errno", return_value=errno.EMFILE
        ):
            with self.assertRaisesRegex(OSError, "Too many open files"):
                spot._open_pidfd(99, 0)

    def test_fallback_pidfd_is_closed_on_post_open_identity_drift(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        changed = copy.deepcopy(expected)
        changed["start_time_ticks"] = 999
        libc = MagicMock()
        libc.syscall = MagicMock(return_value=17)
        with patch.object(
            spot, "_process_observation", side_effect=[expected, changed]
        ), patch.object(
            spot.os, "pidfd_open", None, create=True
        ), patch.object(
            spot.signal, "pidfd_send_signal", MagicMock(), create=True
        ), patch.object(
            spot.sys, "platform", "linux"
        ), patch.object(
            spot.platform, "machine", return_value="x86_64"
        ), patch.object(
            spot.ctypes, "CDLL", return_value=libc
        ), patch.object(spot.os, "close") as close:
            with self.assertRaisesRegex(spot.SpotcheckError, "after pidfd open"):
                spot._open_exact_pidfd(expected, "fp8")
        close.assert_called_once_with(17)

    def test_pidfd_open_double_checks_identity_and_wait_rejects_pid_reuse(self):
        expected = _service_manifest("fp8", pid=99, start_ticks=456)["process"]
        with patch.object(
            spot, "_process_observation", side_effect=[expected, expected]
        ), patch.object(
            spot.os, "pidfd_open", return_value=7, create=True
        ) as opener, patch.object(
            spot.signal, "pidfd_send_signal", create=True
        ):
            self.assertEqual(spot._open_exact_pidfd(expected, "fp8"), 7)
        opener.assert_called_once_with(99, 0)

        changed = copy.deepcopy(expected)
        changed["start_time_ticks"] = 999
        with patch.object(
            spot, "_process_observation", side_effect=[expected, changed]
        ), patch.object(
            spot.os, "pidfd_open", return_value=7, create=True
        ), patch.object(
            spot.signal, "pidfd_send_signal", create=True
        ), patch.object(spot.os, "close") as close:
            with self.assertRaisesRegex(spot.SpotcheckError, "after pidfd open"):
                spot._open_exact_pidfd(expected, "fp8")
        close.assert_called_once_with(7)

        poller = MagicMock()
        poller.poll.return_value = []
        with patch.object(spot.select, "poll", return_value=poller, create=True), patch.object(
            spot.select, "POLLIN", 1, create=True
        ), patch.object(
            spot, "_read_proc_start_ticks", return_value=999
        ):
            with self.assertRaisesRegex(spot.SpotcheckError, "PID was reused"):
                spot._wait_pidfd_gone(7, expected, 0.01)

    def test_stop_uses_one_pidfd_and_removes_only_the_bound_manifest(self):
        manifest = _service_manifest("fp8", pid=99, start_ticks=456)
        expected = manifest["process"]
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                spot.json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            with patch.object(spot, "VICTIM_MANIFEST", manifest_path), patch.object(
                spot, "_verify_current", return_value=(manifest, expected)
            ), patch.object(
                spot, "_port_vllm_processes", return_value=[{
                    "pid": 99, "tokens": expected["cmdline"],
                }]
            ), patch.object(
                spot, "_open_exact_pidfd", return_value=7
            ), patch.object(
                spot, "_signal_exact", return_value=True
            ) as send, patch.object(
                spot, "_wait_pidfd_gone", return_value=True
            ) as wait, patch.object(
                spot, "_port_is_open", return_value=False
            ), patch.object(spot.os, "close") as close:
                spot._stop_exact_server("fp8", require_manifest=True)
        send.assert_called_once_with(7, expected, "fp8", signal.SIGTERM)
        wait.assert_called_once_with(7, expected, 60)
        close.assert_called_once_with(7)
        self.assertFalse(manifest_path.exists())

    def test_cycle_default_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "cycle"
            with patch.object(
                spot, "_run_cycle", side_effect=AssertionError("must not switch")
            ) as cycle:
                self.assertEqual(spot.main(["cycle", "--out-dir", str(out)]), 0)
            cycle.assert_not_called()
            self.assertFalse(out.exists())

    def test_serve_commands_make_bf16_explicit(self):
        fp8 = spot._serve_command("fp8")
        bf16 = spot._serve_command("bf16")
        self.assertEqual(spot._flag_value(fp8, "--quantization"), "fp8")
        self.assertIsNone(spot._flag_value(bf16, "--quantization"))
        self.assertEqual(spot._flag_value(bf16, "--dtype"), "bfloat16")

    def test_fresh_campaign_cycle_allows_new_pid_but_requires_gate_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_dir = root / "gate"
            fresh_dir = root / "fresh"
            gate_dir.mkdir(); fresh_dir.mkdir()
            gate_comparison, gate_cycle, _ = self._write_gate_artifacts(gate_dir)
            gate_proof = spot.load_clean_quant_comparison(
                gate_comparison, gate_cycle, expected_deployment_tree="a" * 64
            )
            gate_manifest = spot._load_json(gate_dir / "fp8.json")["victim"]["manifest"]
            reference = gate_proof["runtime_bundle"]["restored_fp8_runtime"]
            for phase in ("gate_open", "gate_close"):
                check = spot.live_runtime_check(
                    reference, gate_manifest, phase=phase,
                    verified_at=f"2026-07-16T00:02:0{0 if phase == 'gate_open' else 1}+00:00",
                )
                gate_proof["runtime_bundle"] = spot.with_gate_runtime_check(
                    gate_proof["runtime_bundle"], check
                )

            fresh_comparison, fresh_cycle, _ = self._write_gate_artifacts(
                fresh_dir, lifecycle_offset=1000
            )
            fresh_proof = spot.load_clean_quant_comparison(
                fresh_comparison, fresh_cycle, expected_deployment_tree="a" * 64
            )
            equivalence = spot.validate_fresh_h20_campaign_runtime(
                gate_proof, fresh_proof
            )
            self.assertEqual(equivalence["status"], "EQUIVALENT_NEW_LIFECYCLE")
            self.assertNotEqual(
                gate_proof["runtime_bundle"]["restored_fp8_runtime"]["process"]["pid"],
                fresh_proof["runtime_bundle"]["restored_fp8_runtime"]["process"]["pid"],
            )
            spot.validate_fresh_h20_campaign_runtime_proof(
                equivalence,
                gate_proof,
                fresh_proof,
                verify_external_artifacts=False,
            )
            validate_portable_fresh_proof(equivalence, gate_proof, fresh_proof)

            registry = spot.fresh_quant_artifact_registry(fresh_proof)
            self.assertEqual([item[0] for item in registry], [
                "comparison", "cycle_status", "fp8", "bf16"
            ])

            out = root / "standalone_runtime_bundle.json"
            with patch.object(spot, "verify_deployment", return_value={
                "deployed_tree_sha256": "a" * 64,
            }), patch.object(
                spot, "load_clean_quant_comparison", return_value=fresh_proof
            ):
                self.assertEqual(spot.main([
                    "bundle", "--comparison", str(fresh_comparison),
                    "--cycle-status", str(fresh_cycle), "--out", str(out),
                ]), 0)
            written = spot._load_json(out)
            self.assertEqual(written, fresh_proof["runtime_bundle"])
            self.assertEqual(written["gate_checks"], {})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QuantSpotcheckTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("QUANT SPOTCHECK TESTS PASS")
    raise SystemExit(0 if result.wasSuccessful() else 1)
