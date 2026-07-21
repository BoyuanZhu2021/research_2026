"""CPU cross-contract/tamper tests for the formal single-H20 victim runtime."""
from __future__ import annotations

import ast
import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))

try:
    import paramiko as _paramiko  # noqa: F401
except ModuleNotFoundError:
    sys.modules["paramiko"] = types.ModuleType("paramiko")

import h1_serve_victim_h20 as serve  # noqa: E402
import h1_tooluse_gate1_local as gate  # noqa: E402
import h1_victim_quant_spotcheck as quant  # noqa: E402
from src import h20_gate_runtime as gate_runtime  # noqa: E402
from src.h20_eval_artifacts import EVALUATION_PROTOCOL  # noqa: E402
from src.h20_serving_identity import (  # noqa: E402
    build_formal_runtime_bundle,
    build_service_manifest,
    canonical_provenance_summary,
    expected_cmdline,
    expected_environment,
    live_runtime_check,
    runtime_reference,
    seal,
    validate_h20_formal_runtime_bundle,
    validate_h20_runtime_protocol_equivalence,
    validate_runtime_reference,
    validate_service_manifest,
    with_gate_runtime_check,
)
from src.h20_training_protocol import ORACLE_AND_INTERACTION  # noqa: E402
from src.runtime_profile import LEGACY_H20_PROFILE_ID, canonical_sha256  # noqa: E402
from src.training_protocol import VICTIM_REQUEST  # noqa: E402
from src.victim_decision_protocol import (  # noqa: E402
    STRUCTURED_OUTPUT_CONFIG_JSON,
    build_request_contract,
    parse_victim_decision,
)


GPU = {
    "index": 0,
    "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "name": "NVIDIA H20",
    "memory_total_mib": 97871,
}


def _manifest(
    quantization: str = "fp8", *, pid: int = 123, start_ticks: int = 456,
) -> dict:
    command = expected_cmdline(quantization)
    process = {
        "pid": pid,
        "start_time_ticks": start_ticks,
        "cmdline": command,
        "cmdline_sha256": canonical_sha256(command),
        "environ_sha256": "e" * 64,
        "environment": expected_environment(quantization, GPU["uuid"]),
        "gpu_uuid": GPU["uuid"],
    }
    return build_service_manifest(
        quantization=quantization,
        gpu=GPU,
        process=process,
        service={
            "endpoint": "http://127.0.0.1:8000/v1",
            "model_ids": ["qwen3.5-9b"],
            "observed_at": "2026-07-17T00:00:00+00:00",
        },
        sealed_at="2026-07-17T00:00:00+00:00",
    )


def _complete_bundle(manifest: dict | None = None) -> dict:
    manifest = manifest or _manifest()
    reference = runtime_reference(manifest)
    bundle = build_formal_runtime_bundle(
        quant_cycle_status_payload_sha256="c" * 64,
        restored_fp8_runtime=reference,
        sealed_at="2026-07-17T00:01:00+00:00",
    )
    for phase in ("gate_open", "gate_close"):
        check = live_runtime_check(
            reference,
            manifest,
            phase=phase,
            verified_at=f"2026-07-17T00:02:0{0 if phase == 'gate_open' else 1}+00:00",
        )
        bundle = with_gate_runtime_check(bundle, check)
    return bundle


class H20RuntimeContractTests(unittest.TestCase):
    def test_all_formal_paths_share_one_canonical_react_stop_contract(self):
        stop_expected = {
            "stop": ["\nObservation:"],
            "include_stop_str_in_output": False,
        }
        self.assertEqual(
            {
                "stop": list(gate.VICTIM_REACT_STOP),
                "include_stop_str_in_output": gate.VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
            },
            stop_expected,
        )
        self.assertEqual(quant.QUANT_CONFIG["stop"], stop_expected["stop"])
        self.assertIs(
            quant.QUANT_CONFIG["include_stop_str_in_output"],
            stop_expected["include_stop_str_in_output"],
        )
        self.assertEqual(
            {key: VICTIM_REQUEST[key] for key in stop_expected}, stop_expected
        )
        self.assertEqual(
            {
                "stop": ORACLE_AND_INTERACTION["victim_generation_stop"],
                "include_stop_str_in_output": ORACLE_AND_INTERACTION[
                    "victim_include_stop_str_in_output"
                ],
            },
            stop_expected,
        )
        self.assertEqual(EVALUATION_PROTOCOL["victim_stop"], stop_expected["stop"])
        self.assertIs(
            EVALUATION_PROTOCOL["victim_include_stop_str_in_output"],
            stop_expected["include_stop_str_in_output"],
        )
        protocol = gate.VICTIM_DECISION_PROTOCOL
        self.assertEqual(quant.QUANT_CONFIG["victim_output_protocol"], protocol)
        self.assertEqual(VICTIM_REQUEST["victim_output_protocol"], protocol)
        self.assertEqual(
            ORACLE_AND_INTERACTION["victim_output_protocol"], protocol
        )
        self.assertEqual(EVALUATION_PROTOCOL["victim_output_protocol"], protocol)

    def test_gate_powered_runtime_has_no_trainer_or_v100_startup_dependency(self):
        gate_tree = ast.parse(
            (HERE / "h1_tooluse_gate1_local.py").read_text(encoding="utf-8")
        )
        runtime_source = (CODE / "src" / "h20_gate_runtime.py").read_text(
            encoding="utf-8"
        )
        runtime_tree = ast.parse(runtime_source)

        imported = set()
        for tree in (gate_tree, runtime_tree):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
        self.assertNotIn("h1_mt_grpo_train", imported)
        self.assertNotIn("h1_mt_grpo_train_h20", imported)
        self.assertNotIn("prepare_worker_before_torch", runtime_source)
        self.assertNotIn("validate_torch_runtime", runtime_source)
        self.assertNotIn("torchrun", runtime_source.lower())
        real_class = next(
            node for node in gate_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RealLocalRuntime"
        )
        self.assertEqual(
            [base.id for base in real_class.bases if isinstance(base, ast.Name)],
            ["H20GateRuntime"],
        )

    def test_gate_h20_deployment_and_quantization_contracts_are_exact(self):
        self.assertEqual(
            gate.FORMAL_QUANT_DEPLOYMENT_REQUIRED,
            gate_runtime.H20_GATE_DEPLOYMENT_REQUIRED,
        )
        self.assertIn(
            "code/src/h20_gate_runtime.py", gate.FORMAL_QUANT_DEPLOYMENT_REQUIRED
        )
        self.assertEqual(gate_runtime.ATTACKER_RUNTIME_IDENTITY, {
            "model": gate_runtime.ATTACKER_MODEL,
            "revision": gate_runtime.ATTACKER_REVISION,
            "quantization": "bitsandbytes-nf4",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "output_protocol": gate_runtime.ATTACKER_OUTPUT_PROTOCOL,
        })
        runtime_source = (CODE / "src" / "h20_gate_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("local_files_only=True", runtime_source)
        self.assertIn("is_loaded_in_4bit", runtime_source)

    def test_gate_environment_requires_exact_uuid_and_rejects_torchrun_state(self):
        manifest = _manifest()
        selectors = {
            gate_runtime.H20_GPU_UUID_ENV: GPU["uuid"],
            "CUDA_VISIBLE_DEVICES": GPU["uuid"],
        }
        with patch.dict(os.environ, selectors, clear=True):
            checked = gate_runtime._exact_or_set_environment(manifest)
            self.assertEqual(checked["CUDA_VISIBLE_DEVICES"], GPU["uuid"])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(
                os.environ["H1_RUNTIME_PROFILE_ID"], LEGACY_H20_PROFILE_ID
            )

        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RuntimeError, "explicit exact GPU selectors"
        ):
            gate_runtime._exact_or_set_environment(manifest)

        with patch.dict(os.environ, {**selectors, "WORLD_SIZE": "2"}, clear=True), \
                self.assertRaisesRegex(RuntimeError, "distributed/V100"):
            gate_runtime._exact_or_set_environment(manifest)

    def test_gate_cuda_contract_requires_one_matching_bf16_h20(self):
        manifest = _manifest()
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = 1
        cuda.get_device_properties.return_value = types.SimpleNamespace(
            name=GPU["name"], total_memory=97356 * 1024 * 1024
        )
        cuda.get_device_capability.return_value = (9, 0)
        cuda.is_bf16_supported.return_value = True
        torch_module = types.SimpleNamespace(
            cuda=cuda,
            __version__="2.8.0+cu128",
            version=types.SimpleNamespace(cuda="12.8"),
        )
        with patch.dict(os.environ, {
            gate_runtime.H20_GPU_UUID_ENV: GPU["uuid"],
            "CUDA_VISIBLE_DEVICES": GPU["uuid"],
        }, clear=True):
            checked = gate_runtime.validate_single_h20_cuda(torch_module, manifest)
        self.assertEqual(checked["gpu_uuid"], GPU["uuid"])
        self.assertEqual(checked["compute_capability"], [9, 0])
        self.assertTrue(checked["bf16_supported"])
        self.assertEqual(checked["nvidia_smi_memory_total_mib"], 97871)
        self.assertEqual(checked["torch_memory_total_mib"], 97356)
        self.assertEqual(checked["memory_total_delta_mib"], 515)
        cuda.set_device.assert_called_once_with(0)

        cuda.is_bf16_supported.return_value = False
        with patch.dict(os.environ, {
            gate_runtime.H20_GPU_UUID_ENV: GPU["uuid"],
            "CUDA_VISIBLE_DEVICES": GPU["uuid"],
        }, clear=True), self.assertRaisesRegex(RuntimeError, "BF16"):
            gate_runtime.validate_single_h20_cuda(torch_module, manifest)

        cuda.is_bf16_supported.return_value = True
        for label, name, torch_total, needle in (
            ("negative-delta", GPU["name"], 97872, "delta.*outside"),
            ("over-1024-delta", GPU["name"], 96846, "delta.*outside"),
            ("name-mismatch", "NVIDIA H20 SXM", 97356, "name differs"),
        ):
            cuda.get_device_properties.return_value = types.SimpleNamespace(
                name=name, total_memory=torch_total * 1024 * 1024
            )
            with self.subTest(label=label), patch.dict(os.environ, {
                gate_runtime.H20_GPU_UUID_ENV: GPU["uuid"],
                "CUDA_VISIBLE_DEVICES": GPU["uuid"],
            }, clear=True), self.assertRaisesRegex(RuntimeError, needle):
                gate_runtime.validate_single_h20_cuda(torch_module, manifest)

    def test_gate_attacker_seed_and_victim_request_call_contracts(self):
        self.assertEqual(gate.ATTACKER_MAX_NEW, 256)
        self.assertEqual(gate.VICTIM_MAX_TOKENS, 1024)
        goal_id = "BankManagerSearchPayee__GmailSearchEmails"
        expected_seed = gate_runtime.attacker_call_seed(goal_id, 3)
        self.assertEqual(gate.attacker_call_seed(goal_id, 3), expected_seed)
        self.assertNotEqual(expected_seed, gate_runtime.attacker_call_seed(goal_id, 4))

        messages = [{"role": "user", "content": "test"}]
        structured_outputs = {"json": {"type": "object"}}
        payload = gate_runtime.victim_request_payload(
            messages,
            max_tokens=gate.VICTIM_MAX_TOKENS,
            temperature=gate.VICTIM_TEMPERATURE,
            enable_thinking=False,
            seed=0,
            structured_outputs=structured_outputs,
        )
        self.assertEqual(payload, {
            "model": gate_runtime.VICTIM_H20_SERVED_NAME,
            "messages": messages,
            "max_tokens": gate.VICTIM_MAX_TOKENS,
            "temperature": gate.VICTIM_TEMPERATURE,
            "stop": ["\nObservation:"],
            "include_stop_str_in_output": False,
            "structured_outputs": structured_outputs,
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": 0,
        })

        powered = gate_runtime.H20GateRuntime.__new__(gate_runtime.H20GateRuntime)
        powered._torch = MagicMock()
        powered._torch.random.fork_rng.return_value.__enter__.return_value = None
        powered._generate = MagicMock(return_value=[{"text": "x"}])
        result = powered.attacker(messages, {"goal_id": goal_id, "turn": 3}, None)
        self.assertEqual(result, "x")
        powered._torch.random.fork_rng.assert_called_once_with(devices=[0])
        powered._torch.manual_seed.assert_called_once_with(expected_seed)
        powered._torch.cuda.manual_seed_all.assert_called_once_with(expected_seed)
        powered._generate.assert_called_once_with([messages])

    def test_gate_victim_react_stop_payload_is_exact_and_immutable(self):
        self.assertIsInstance(gate.VICTIM_REACT_STOP, tuple)
        self.assertEqual(
            gate.VICTIM_REACT_STOP,
            ("\nObservation:",),
        )
        kwargs = {
            "max_tokens": gate.VICTIM_MAX_TOKENS,
            "temperature": gate.VICTIM_TEMPERATURE,
            "enable_thinking": False,
            "seed": 0,
            "structured_outputs": {"json": {"type": "object"}},
        }
        messages = [{"role": "user", "content": "test"}]
        first = gate_runtime.victim_request_payload(messages, **kwargs)
        second = gate_runtime.victim_request_payload(messages, **kwargs)
        self.assertEqual(first["stop"], ["\nObservation:"])
        self.assertEqual(second["stop"], ["\nObservation:"])
        self.assertIs(first["include_stop_str_in_output"], False)
        self.assertEqual(first["structured_outputs"], kwargs["structured_outputs"])
        self.assertIsNot(first["stop"], second["stop"])

        first["stop"].append("mutated")
        self.assertEqual(second["stop"], ["\nObservation:"])
        self.assertEqual(
            gate.VICTIM_REACT_STOP,
            ("\nObservation:",),
        )

    def test_gate_victim_structured_schema_permits_exactly_one_decision(self):
        tools = {
            "AmazonViewSavedAddresses": {
                "name": "AmazonViewSavedAddresses", "parameters": [],
            },
            "GmailSendEmail": {
                "name": "GmailSendEmail",
                "parameters": [
                    {"name": "to", "type": "string", "required": True},
                    {"name": "subject", "type": "string", "required": True},
                    {"name": "body", "type": "string", "required": True},
                ],
            },
        }
        contract = build_request_contract(
            [{"role": "user", "content": "test"}], tools
        )
        self.assertEqual(
            len(contract["structured_outputs"]["json"]["oneOf"]), 3
        )
        action = parse_victim_decision(
            json.dumps({
                "kind": "action", "tool": "AmazonViewSavedAddresses", "arguments": {},
            }),
            tools,
        )
        self.assertEqual((action.tool, action.arguments), ("AmazonViewSavedAddresses", "{}"))
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            parse_victim_decision(
                "Action: AmazonViewSavedAddresses\nAction Input: {}", tools
            )

    def test_direct_launcher_help_is_self_contained_and_cpu_only(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / "h1_serve_victim_h20.py"), "--help"],
            cwd=str(CODE.parent),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("{start,status,stop}", completed.stdout)
        self.assertNotIn("paramiko", completed.stderr.lower())
        self.assertNotIn("ModuleNotFoundError", completed.stderr)

    def test_h20_preflight_is_exact_offline_vllm_only_and_scoped(self):
        self.assertNotIn("tfserve", serve.H20_PREFLIGHT.lower())
        self.assertNotIn("h1_tooluse_dl.log", serve.H20_PREFLIGHT)
        self.assertIn(serve.ATTACKER_MODEL, serve.H20_ATTACKER_PREFLIGHT)
        self.assertIn(serve.ATTACKER_REVISION, serve.H20_ATTACKER_PREFLIGHT)
        self.assertIn(serve.VICTIM_HF, serve.H20_VICTIM_PREFLIGHT)
        self.assertIn(serve.VICTIM_REVISION, serve.H20_VICTIM_PREFLIGHT)
        self.assertIn(serve.H20_VLLM_VERSION, serve.H20_VLLM_PREFLIGHT)
        self.assertIn("H20_VLLM_TOOLCHAIN_OK", serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn(serve.H20_VLLM_NINJA, serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn(serve.H20_VLLM_NINJA_METADATA_VERSION, serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn(serve.H20_VLLM_NINJA_BINARY_VERSION, serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn(serve.H20_CUDA_NVCC, serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn(serve.H20_CUDA_NVCC_RELEASE, serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn(serve.H20_SYSTEM_CXX, serve.H20_TOOLCHAIN_PREFLIGHT)
        self.assertIn("p.name==r", serve.H20_PREFLIGHT)
        self.assertIn("local_files_only=True", serve.H20_PREFLIGHT)
        for command in (
            serve.H20_CUDA_CHECK,
            serve.H20_VLLM_PREFLIGHT,
            serve.H20_TOOLCHAIN_PREFLIGHT,
            serve.H20_ATTACKER_PREFLIGHT,
            serve.H20_VICTIM_PREFLIGHT,
        ):
            self.assertIn(
                "env -u OMP_NUM_THREADS OMP_NUM_THREADS=16",
                command,
            )
        complete = "\n".join(f"{marker} detail" for marker in serve.H20_PREFLIGHT_MARKERS)
        self.assertEqual(serve._missing_preflight_markers(complete), ())
        spoofed = complete.replace(
            "VICTIM_REVISION_OK detail", "prefix VICTIM_REVISION_OK detail"
        )
        self.assertEqual(
            serve._missing_preflight_markers(spoofed), ("VICTIM_REVISION_OK",)
        )

    def test_h20_start_uses_own_preflight_and_scoped_manager_launch(self):
        fake_rm = MagicMock()
        fake_cli = fake_rm.connect.return_value
        markers = "\n".join(
            f"{marker} exact" for marker in serve.H20_PREFLIGHT_MARKERS
        )
        fake_rm.run.side_effect = [
            (0, "0, NVIDIA H20", ""),
            (0, "NVIDIA H20", ""),
            (0, markers, ""),
            (0, '{"status":"running"}', ""),
        ]
        with patch.object(
            serve, "_load_remote_stack", return_value=(fake_rm, serve.H20_PREFLIGHT)
        ):
            self.assertEqual(serve.main(serve.LEGACY_H20_PROFILE_ID, "start"), 0)

        commands = [call.args[1] for call in fake_rm.run.call_args_list]
        self.assertEqual(commands[1], serve.H20_CUDA_CHECK)
        self.assertEqual(commands[2], serve.H20_PREFLIGHT)
        self.assertIn(
            "env -u OMP_NUM_THREADS OMP_NUM_THREADS=16",
            commands[3],
        )
        self.assertIn(serve.REMOTE_MANAGER, commands[3])
        self.assertNotIn("tfserve", "\n".join(commands).lower())
        fake_cli.close.assert_called_once_with()

    def test_h20_start_missing_exact_marker_never_calls_manager(self):
        fake_rm = MagicMock()
        fake_cli = fake_rm.connect.return_value
        markers = "\n".join(
            f"{marker} exact"
            for marker in serve.H20_PREFLIGHT_MARKERS
            if marker != "H20_VLLM_STACK_OK"
        )
        fake_rm.run.side_effect = [
            (0, "0, NVIDIA H20", ""),
            (0, "NVIDIA H20", ""),
            (0, markers, ""),
        ]
        with patch.object(
            serve, "_load_remote_stack", return_value=(fake_rm, serve.H20_PREFLIGHT)
        ):
            self.assertEqual(serve.main(serve.LEGACY_H20_PROFILE_ID, "start"), 1)
        self.assertEqual(fake_rm.run.call_count, 3)
        fake_cli.close.assert_called_once_with()

    def test_fp8_and_bf16_share_one_exact_schema_and_canonical_command(self):
        for quantization in ("fp8", "bf16"):
            manifest = _manifest(quantization)
            checked = validate_service_manifest(
                manifest, expected_quantization=quantization
            )
            self.assertEqual(checked["process"]["cmdline"], expected_cmdline(quantization))
            self.assertEqual(
                checked["process"]["environment"],
                expected_environment(quantization, GPU["uuid"]),
            )
        self.assertIn("--quantization", expected_cmdline("fp8"))
        self.assertNotIn("--quantization", expected_cmdline("bf16"))
        self.assertEqual(
            expected_cmdline("fp8")[
                expected_cmdline("fp8").index("--structured-outputs-config") + 1
            ],
            STRUCTURED_OUTPUT_CONFIG_JSON,
        )
        self.assertEqual(shlex.split(serve.SERVE), expected_cmdline("fp8"))

    def test_manifest_tamper_pid_start_cmd_env_gpu_and_api_fail_closed(self):
        mutations = (
            ("pid-unsealed", False, lambda value: value["process"].__setitem__("pid", 999)),
            ("start-unsealed", False, lambda value: value["process"].__setitem__(
                "start_time_ticks", 999
            )),
            ("cmd", True, lambda value: value["process"]["cmdline"].__setitem__(-1, "wrong")),
            ("env", True, lambda value: value["process"]["environment"].__setitem__(
                "HF_HUB_OFFLINE", "0"
            )),
            ("gpu", True, lambda value: value["gpu"].__setitem__(
                "uuid", "GPU-bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee"
            )),
            ("api", True, lambda value: value["service"].__setitem__(
                "model_ids", ["wrong"]
            )),
        )
        for label, reseal, mutate in mutations:
            tampered = copy.deepcopy(_manifest())
            mutate(tampered)
            if reseal:
                tampered = seal(tampered)
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_service_manifest(tampered, expected_quantization="fp8")

    def test_runtime_reference_rejects_pid_reuse_and_valid_new_server(self):
        original = _manifest(pid=123, start_ticks=456)
        reference = runtime_reference(original)
        validate_runtime_reference(reference, original)
        with self.assertRaises(ValueError):
            validate_runtime_reference(reference, _manifest(pid=123, start_ticks=999))
        with self.assertRaises(ValueError):
            validate_runtime_reference(reference, _manifest(pid=124, start_ticks=1000))

        fresh_manifest = _manifest(pid=124, start_ticks=1000)
        fresh_bundle = build_formal_runtime_bundle(
            quant_cycle_status_payload_sha256="d" * 64,
            restored_fp8_runtime=runtime_reference(fresh_manifest),
            sealed_at="2026-07-17T01:00:00+00:00",
        )
        equivalence = validate_h20_runtime_protocol_equivalence(
            _complete_bundle(original), fresh_bundle
        )
        self.assertEqual(equivalence["status"], "EQUIVALENT_NEW_LIFECYCLE")

        reused_process_bundle = build_formal_runtime_bundle(
            quant_cycle_status_payload_sha256="f" * 64,
            restored_fp8_runtime=reference,
            sealed_at="2026-07-17T01:01:00+00:00",
        )
        with self.assertRaisesRegex(ValueError, "reused.*process lifecycle"):
            validate_h20_runtime_protocol_equivalence(
                _complete_bundle(original), reused_process_bundle
            )

        reused_cycle_bundle = build_formal_runtime_bundle(
            quant_cycle_status_payload_sha256="c" * 64,
            restored_fp8_runtime=runtime_reference(fresh_manifest),
            sealed_at="2026-07-17T01:02:00+00:00",
        )
        with self.assertRaisesRegex(ValueError, "reused.*quant cycle status"):
            validate_h20_runtime_protocol_equivalence(
                _complete_bundle(original), reused_cycle_bundle
            )

    def test_bundle_requires_both_gate_checks_and_all_bind_same_runtime(self):
        manifest = _manifest()
        complete = _complete_bundle(manifest)
        checked = validate_h20_formal_runtime_bundle(
            complete, manifest, require_gate_checks=True
        )
        summary = canonical_provenance_summary(checked)
        self.assertEqual(summary["gpu_uuid"], GPU["uuid"])
        self.assertEqual(set(summary["gate_check_payload_sha256"]), {
            "gate_open", "gate_close"
        })

        missing = copy.deepcopy(complete)
        missing["gate_checks"].pop("gate_close")
        missing = seal(missing)
        with self.assertRaises(ValueError):
            validate_h20_formal_runtime_bundle(missing, require_gate_checks=True)

        tampered = copy.deepcopy(complete)
        tampered["gate_checks"]["gate_close"]["process"]["start_time_ticks"] += 1
        tampered["gate_checks"]["gate_close"] = seal(
            tampered["gate_checks"]["gate_close"]
        )
        tampered = seal(tampered)
        with self.assertRaises(ValueError):
            validate_h20_formal_runtime_bundle(tampered, require_gate_checks=True)

    def test_public_live_adapter_rejects_a_different_restored_process(self):
        bundle = _complete_bundle(_manifest())
        with patch.object(quant, "_verify_current", return_value=(
            _manifest(pid=999, start_ticks=1000), {}
        )):
            with self.assertRaises(quant.SpotcheckError):
                quant.validate_live_h20_runtime_bundle(bundle, phase="eval_open")

    def test_service_manager_status_start_stop_use_exact_manifest(self):
        manifest = _manifest()
        process = manifest["process"]
        with patch.object(quant, "_verify_current", return_value=(manifest, process)):
            status = quant.manage_fp8_service("status")
        self.assertEqual((status["pid"], status["start_time_ticks"]), (123, 456))

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            quant, "VICTIM_MANIFEST", Path(temporary) / "manifest.json"
        ), patch.object(quant, "_port_vllm_processes", return_value=[]), patch.object(
            quant, "_port_is_open", return_value=False
        ), patch.object(quant, "_start_exact_server") as start, patch.object(
            quant, "_verify_current", return_value=(manifest, process)
        ):
            started = quant.manage_fp8_service("start")
        start.assert_called_once_with("fp8", 600)
        self.assertFalse(started["already_running"])

        with patch.object(quant, "_verify_current", return_value=(manifest, process)), \
                patch.object(quant, "_stop_exact_server") as stop, \
                patch.object(quant, "_port_vllm_processes", return_value=[]), \
                patch.object(quant, "_port_is_open", return_value=False), \
                patch.object(quant, "VICTIM_MANIFEST", Path("missing-manifest")):
            stopped = quant.manage_fp8_service("stop")
        stop.assert_called_once_with("fp8", require_manifest=True)
        self.assertEqual(stopped["status"], "stopped")

    def test_h20_direct_cli_routes_all_actions(self):
        for action in ("start", "status", "stop"):
            with self.subTest(action=action), patch.object(
                sys, "argv", ["h1_serve_victim_h20.py", "--profile", LEGACY_H20_PROFILE_ID, action]
            ), patch.object(serve, "main", return_value=0) as main:
                self.assertEqual(serve._cli(), 0)
                main.assert_called_once_with(LEGACY_H20_PROFILE_ID, action)

    def test_gate_wrapper_always_rechecks_on_success_and_exception(self):
        argv = ["--fp8-repeatability", "repeatability.json", "--fp8-cycle-status", "cycle.json"]
        with patch.object(gate, "verify_fp8_cycle_runtime_live") as verify, patch.object(
            gate, "_main", return_value=0
        ):
            self.assertEqual(gate.main(argv), 0)
        self.assertEqual(
            [call.kwargs["phase"] for call in verify.call_args_list],
            ["gate_preflight", "gate_postflight"],
        )

        with patch.object(gate, "verify_fp8_cycle_runtime_live") as verify, patch.object(
            gate, "_main", side_effect=RuntimeError("science core failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "science core failed"):
                gate.main(argv)
        self.assertEqual(
            [call.kwargs["phase"] for call in verify.call_args_list],
            ["gate_preflight", "gate_exception_close"],
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(H20RuntimeContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("H20 VICTIM RUNTIME CONTRACT TESTS PASS")
    raise SystemExit(0 if result.wasSuccessful() else 1)
