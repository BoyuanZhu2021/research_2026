"""CPU/tamper goldens for the independent single-H20 QLoRA contract."""
from __future__ import annotations

import copy
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.h20_training_artifacts import (
    build_artifact_manifest,
    build_checkpoint_manifest,
    load_and_validate_h20_adapter,
    sha256_file,
    tree_sha256,
    validate_artifact_manifest,
)
from src.h20_training_protocol import (
    ATTACKER_COMPUTE_DTYPE,
    DATA_IDENTITY,
    FIXED_TRAINING,
    FORMAL_TRAINING_RUN_REGISTRY,
    FORMAL_TRAINING_PROTOCOL_SHA256,
    LORA_CONFIG,
    MAX_REMAINING_TRAINING_CAMPAIGN_GPU_HOURS,
    MAX_SINGLE_RUN_ARTIFACT_BYTES,
    MODEL_IDENTITY,
    ORACLE_AND_INTERACTION,
    QLORA_CONFIG,
    RUN_CONFIG_KIND,
    RUN_CONFIG_SCHEMA_VERSION,
    SINGLE_H20_EXECUTION,
    build_benchmark_manifest,
    build_benchmark_result,
    build_budget_authorization,
    build_goal_schedule,
    canonical_sha256,
    construction_seed_record,
    formal_training_protocol,
    generation_call_seed,
    h20_policy_load_spec,
    load_and_validate_formal_training_inputs,
    seal_payload,
    validate_benchmark_result,
    validate_formal_training_values,
    validate_budget_authorization,
    validate_paired_initial_lora,
    validate_run_config,
)
from src.runtime_profile import H20_RUNTIME_PROFILE_SHA256, LEGACY_H20_PROFILE_ID
from src.h20_serving_identity import (
    ENDPOINT,
    FORMAL_RUNTIME_BUNDLE_KIND,
    LIVE_CHECK_KIND,
    MANIFEST_PATH,
    MODEL_IDENTITY as H20_VICTIM_MODEL_IDENTITY,
    RUNTIME_REFERENCE_KIND,
    SCHEMA_VERSION as H20_RUNTIME_SCHEMA_VERSION,
    backend_identity,
    seal as runtime_seal,
)


def _identity(value: dict) -> dict:
    return {"canonical_sha256": canonical_sha256(value)}


def _runtime_reference() -> dict:
    return runtime_seal({
        "schema_version": H20_RUNTIME_SCHEMA_VERSION,
        "kind": RUNTIME_REFERENCE_KIND,
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "service_manifest_path": MANIFEST_PATH,
        "service_manifest_payload_sha256": "a" * 64,
        "model": H20_VICTIM_MODEL_IDENTITY,
        "backend": backend_identity("fp8"),
        "endpoint": ENDPOINT,
        "gpu_uuid": "GPU-test-h20",
        "process": {
            "pid": 101,
            "start_time_ticks": 9001,
            "cmdline_sha256": "b" * 64,
            "environ_sha256": "c" * 64,
        },
        "sealed_at": "2026-07-17T00:00:00Z",
    })


def _live_check(reference: dict, phase: str) -> dict:
    return runtime_seal({
        "schema_version": H20_RUNTIME_SCHEMA_VERSION,
        "kind": LIVE_CHECK_KIND,
        "phase": phase,
        "verified_at": "2026-07-17T00:00:01Z",
        "runtime_reference_payload_sha256": reference["payload_sha256"],
        "service_manifest_payload_sha256": reference["service_manifest_payload_sha256"],
        "gpu_uuid": reference["gpu_uuid"],
        "process": reference["process"],
    })


def _runtime_bundle(reference: dict) -> dict:
    return runtime_seal({
        "schema_version": H20_RUNTIME_SCHEMA_VERSION,
        "kind": FORMAL_RUNTIME_BUNDLE_KIND,
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "quant_cycle_status_payload_sha256": "d" * 64,
        "restored_fp8_runtime": reference,
        "gate_checks": {
            "gate_open": _live_check(reference, "gate_open"),
            "gate_close": _live_check(reference, "gate_close"),
        },
        "sealed_at": "2026-07-17T00:00:02Z",
    })


def _run_config(*, arm: str = "dense", seed: int = 0) -> dict:
    schedule = build_goal_schedule(seed=seed, steps=60, n_goals=8, n_train=322)
    gate = {"schema_version": 1, "kind": "tooluse_gate1_frozen", "verdict": "PASS", "passed": True}
    reference = _runtime_reference()
    runtime = _runtime_bundle(reference)
    deployment = {
        "schema_version": 1,
        "injecagent_commit": DATA_IDENTITY["injecagent_commit"],
        "deployed_tree_sha256": "b" * 64,
    }
    gate_identity = _identity(gate)
    runtime_identity = _identity(runtime)
    deployment_identity = _identity(deployment)
    benchmark = build_benchmark_manifest(
        gate1_identity=gate_identity,
        runtime_identity=runtime_identity,
        deployment_identity=deployment_identity,
        step_seconds=30.0,
        benchmark_artifact_bytes=1024,
        serialized_adapter_bytes=2048,
    )
    benchmark_result = build_benchmark_result(
        benchmark,
        runtime_bundle=runtime,
        runtime_open_check=_live_check(reference, "benchmark_open"),
        runtime_close_check=_live_check(reference, "benchmark_close"),
    )
    return seal_payload({
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "kind": RUN_CONFIG_KIND,
        "canonical_training_run": True,
        "run_kind": "formal",
        "tag": f"h20-{arm}-s{seed}",
        "arm": arm,
        "seed": seed,
        **FIXED_TRAINING,
        "smoke": False,
        "benchmark": False,
        "runtime_profile": LEGACY_H20_PROFILE_ID,
        "runtime_profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "execution": SINGLE_H20_EXECUTION,
        "models": MODEL_IDENTITY,
        "qlora": QLORA_CONFIG,
        "lora": LORA_CONFIG,
        "data": DATA_IDENTITY,
        "oracle_and_interaction": ORACLE_AND_INTERACTION,
        "training_protocol": formal_training_protocol(),
        "training_protocol_sha256": FORMAL_TRAINING_PROTOCOL_SHA256,
        "global_goal_schedule": schedule,
        "global_goal_schedule_sha256": canonical_sha256(schedule),
        "gate1": gate,
        "gate1_identity": gate_identity,
        "runtime": runtime,
        "runtime_identity": runtime_identity,
        "runtime_open_check": _live_check(reference, "train_open"),
        "deployment": deployment,
        "deployment_identity": deployment_identity,
        "benchmark_manifest": benchmark,
        "benchmark_manifest_path": "/fixture/benchmark_manifest.json",
        "benchmark_identity": {
            "path": "/fixture/benchmark_manifest.json",
            "file_sha256": "8" * 64,
            "canonical_sha256": canonical_sha256(benchmark),
        },
        "benchmark_result": benchmark_result,
        "benchmark_result_path": "/fixture/benchmark_result.json",
        "benchmark_result_identity": {
            "path": "/fixture/benchmark_result.json",
            "file_sha256": "9" * 64,
            "canonical_sha256": canonical_sha256(benchmark_result),
        },
        "budget_authorization": None,
        "budget_authorization_identity": None,
        "budget_authorization_path": None,
        "construction_seeds": construction_seed_record(seed),
        "initial_lora_sha256": "c" * 64,
    })


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, dict, dict]:
    run = root / "runs" / "h20-dense-s0"
    run.mkdir(parents=True)
    config = _run_config()
    _write_json(run / "run_config.json", config)
    (run / "adapter_step30").mkdir()
    (run / "adapter_step30" / "adapter_model.safetensors").write_bytes(b"step30")
    checkpoint = build_checkpoint_manifest(
        step=30,
        adapter_tree_sha256=tree_sha256(run / "adapter_step30"),
        lora_sha256="d" * 64,
        run_config_file_sha256=sha256_file(run / "run_config.json"),
        run_config=config,
    )
    checkpoint_path = run / "adapter_step30.manifest.json"
    _write_json(checkpoint_path, checkpoint)
    checkpoint_ref = {
        "step": 30,
        "adapter_path": "adapter_step30",
        "adapter_tree_sha256": checkpoint["adapter_tree_sha256"],
        "lora_sha256": checkpoint["lora_sha256"],
        "manifest_path": checkpoint_path.name,
        "manifest_file_sha256": sha256_file(checkpoint_path),
        "manifest_payload_sha256": checkpoint["payload_sha256"],
    }
    (run / "adapter").mkdir()
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"final")
    progress_rows = []
    for step in range(1, 61):
        lora_sha256 = "e" * 64 if step == 60 else "d" * 64 if step == 30 else "a" * 64
        progress_rows.append({
            "step": step,
            "tag": config["tag"],
            "arm": config["arm"],
            "seed": config["seed"],
            "global_B": 8 * 8,
            "goal_schedule_sha256": config["global_goal_schedule_sha256"],
            "lora_sha256": lora_sha256,
        })
    (run / "progress.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress_rows),
        encoding="utf-8",
    )
    rollout_rows = (
        {
            "step": step,
            "global_group_slot": slot,
            "trajectory_index": trajectory,
            "arm": config["arm"],
            "seed": config["seed"],
        }
        for step in range(1, 61)
        for slot in range(8)
        for trajectory in range(8)
    )
    (run / "rollouts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rollout_rows),
        encoding="utf-8",
    )
    artifact = build_artifact_manifest(
        run_config=config,
        run_config_file_sha256=sha256_file(run / "run_config.json"),
        adapter_tree_sha256=tree_sha256(run / "adapter"),
        final_lora_sha256="e" * 64,
        progress_sha256=sha256_file(run / "progress.jsonl"),
        progress_rows=60,
        rollouts_sha256=sha256_file(run / "rollouts.jsonl"),
        rollout_rows=60 * 8 * 8,
        checkpoints=[checkpoint_ref],
        runtime_close_check=_live_check(
            config["runtime"]["restored_fp8_runtime"], "train_close"
        ),
    )
    _write_json(run / "artifact_manifest.json", artifact)
    return run, config, artifact


class H20TrainingProtocolTest(unittest.TestCase):
    def test_frozen_shape_quantization_and_schedule(self) -> None:
        config = _run_config()
        self.assertEqual(validate_run_config(config), config)
        self.assertEqual(ATTACKER_COMPUTE_DTYPE, "bfloat16")
        self.assertEqual(SINGLE_H20_EXECUTION["world_size"], 1)
        self.assertEqual(
            (FIXED_TRAINING["steps"], FIXED_TRAINING["n_goals"], FIXED_TRAINING["G"]),
            (60, 8, 8),
        )
        self.assertEqual((FIXED_TRAINING["T"], FIXED_TRAINING["max_calls"]), (5, 3))
        self.assertTrue(QLORA_CONFIG["load_in_4bit"])
        self.assertEqual(QLORA_CONFIG["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(QLORA_CONFIG["bnb_4bit_use_double_quant"])
        self.assertTrue(
            QLORA_CONFIG["prepare_model_for_kbit_training"]["before_get_peft_model"]
        )
        self.assertEqual((LORA_CONFIG["r"], LORA_CONFIG["lora_alpha"]), (32, 64))
        self.assertEqual(
            FORMAL_TRAINING_RUN_REGISTRY,
            (
                ("dense", 0), ("sparse", 0),
                ("dense", 1), ("sparse", 1),
                ("dense", 2), ("sparse", 2),
            ),
        )
        self.assertEqual(MAX_REMAINING_TRAINING_CAMPAIGN_GPU_HOURS, 12.0)
        self.assertEqual(MAX_SINGLE_RUN_ARTIFACT_BYTES, 5 * 1024**3)
        self.assertEqual(
            config["benchmark_manifest"]["measurement"][
                "projected_training_campaign_gpu_hours"
            ],
            3.0,
        )
        self.assertEqual(len(config["global_goal_schedule"]), 60)
        self.assertEqual(len(config["global_goal_schedule"][0]), 8)
        self.assertEqual(
            generation_call_seed(seed=1, step=2, turn=3),
            generation_call_seed(seed=1, step=2, turn=3),
        )
        self.assertEqual(h20_policy_load_spec(adapter=True)["quantization"], QLORA_CONFIG)
        self.assertEqual(
            {construction_seed_record(seed)["seed"] for seed in (0, 1, 2)},
            {0, 1, 2},
        )

    def test_formal_values_and_v100_schema_fail_closed(self) -> None:
        values = {**FIXED_TRAINING, "arm": "sparse", "seed": 2,
                  "smoke": False, "benchmark": False}
        validate_formal_training_values(values)
        for key, value in (("steps", 59), ("n_goals", 4), ("G", 4), ("T", 4),
                           ("max_calls", 2), ("seed", 3)):
            tampered = dict(values)
            tampered[key] = value
            with self.assertRaises(ValueError):
                validate_formal_training_values(tampered)
        v100 = _run_config()
        v100["runtime_profile"] = "dual-v100-sxm2-32gb-colocated-ddp-v1"
        v100 = seal_payload(v100)
        with self.assertRaisesRegex(ValueError, "runtime_profile"):
            validate_run_config(v100)

    def test_run_config_seal_gate_runtime_and_deployment_tamper(self) -> None:
        for mutate, pattern in (
            (lambda value: value["gate1"].update(verdict="MARGINAL", passed=False), "gate1"),
            (lambda value: value["runtime"].update(profile_id="other"), "runtime"),
            (lambda value: value["deployment"].update(deployed_tree_sha256="bad"), "deployment"),
            (lambda value: value["models"]["attacker"].update(revision="main"), "models"),
        ):
            value = _run_config()
            mutate(value)
            value = seal_payload(value)
            with self.assertRaisesRegex(ValueError, pattern):
                validate_run_config(value)

    def test_above_threshold_requires_sealed_arm_seed_authorization(self) -> None:
        config = _run_config()
        boundary = build_benchmark_manifest(
            gate1_identity=config["gate1_identity"],
            runtime_identity=config["runtime_identity"],
            deployment_identity=config["deployment_identity"],
            step_seconds=120.0,
            benchmark_artifact_bytes=1024,
            serialized_adapter_bytes=2048,
        )
        self.assertEqual(boundary["status"], "PASS")
        self.assertEqual(
            boundary["measurement"]["projected_single_run_gpu_hours"], 2.0
        )
        self.assertEqual(
            boundary["measurement"]["projected_training_campaign_gpu_hours"], 12.0
        )
        high = build_benchmark_manifest(
            gate1_identity=config["gate1_identity"],
            runtime_identity=config["runtime_identity"],
            deployment_identity=config["deployment_identity"],
            step_seconds=121.0,
            benchmark_artifact_bytes=1024,
            serialized_adapter_bytes=2048,
        )
        self.assertEqual(high["status"], "BUDGET_REVIEW_REQUIRED")
        high_disk = build_benchmark_manifest(
            gate1_identity=config["gate1_identity"],
            runtime_identity=config["runtime_identity"],
            deployment_identity=config["deployment_identity"],
            step_seconds=1.0,
            benchmark_artifact_bytes=5 * 1024**3,
            serialized_adapter_bytes=2048,
        )
        self.assertEqual(high_disk["status"], "BUDGET_REVIEW_REQUIRED")
        with self.assertRaisesRegex(ValueError, "lacks sealed"):
            validate_budget_authorization(None, high, arm="dense", seed=0)
        authorization = build_budget_authorization(
            high,
            authorized_by="PI @fixture", authorized_at="2026-07-17T00:00:00Z",
            approval_reference="Discussion.md fixture explicit approval",
        )
        self.assertEqual(
            validate_budget_authorization(authorization, high, arm="dense", seed=0),
            authorization,
        )
        self.assertEqual(
            validate_budget_authorization(authorization, high, arm="sparse", seed=2),
            authorization,
        )
        self.assertEqual(authorization["formal_runs"], [
            {"arm": arm, "seed": seed}
            for arm, seed in FORMAL_TRAINING_RUN_REGISTRY
        ])
        tampered = copy.deepcopy(authorization)
        tampered["formal_runs"][0]["seed"] = 2
        tampered = seal_payload(tampered)
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_budget_authorization(tampered, high, arm="dense", seed=0)
        with self.assertRaisesRegex(ValueError, "must not carry"):
            validate_budget_authorization(
                authorization, config["benchmark_manifest"], arm="dense", seed=0
            )
        with self.assertRaisesRegex(ValueError, "explicit 'PI @"):
            build_budget_authorization(
                high,
                authorized_by="Agent @fixture",
                authorized_at="2026-07-17T00:00:00Z",
                approval_reference="not sufficient",
            )
        with self.assertRaisesRegex(ValueError, "approval_reference"):
            build_budget_authorization(
                high,
                authorized_by="PI @fixture",
                authorized_at="2026-07-17T00:00:00Z",
                approval_reference="",
            )

    def test_benchmark_manifest_immutable_and_result_binds_runtime_checks(self) -> None:
        config = _run_config()
        manifest = config["benchmark_manifest"]
        before = copy.deepcopy(manifest)
        reference = config["runtime"]["restored_fp8_runtime"]
        result = build_benchmark_result(
            manifest,
            runtime_bundle=config["runtime"],
            runtime_open_check=_live_check(reference, "benchmark_open"),
            runtime_close_check=_live_check(reference, "benchmark_close"),
        )
        self.assertEqual(manifest, before)
        self.assertIsNot(result["benchmark_manifest"], manifest)

        tampered = copy.deepcopy(result)
        tampered["runtime_close_check"] = _live_check(reference, "benchmark_open")
        tampered = seal_payload(tampered)
        with self.assertRaisesRegex(ValueError, "phase"):
            validate_benchmark_result(
                tampered,
                runtime_bundle=config["runtime"],
                gate1_identity=config["gate1_identity"],
                runtime_identity=config["runtime_identity"],
                deployment_identity=config["deployment_identity"],
            )

    def test_formal_input_files_are_revalidated_together(self) -> None:
        config = _run_config()
        with tempfile.TemporaryDirectory(prefix="h20-benchmark-inputs-") as temp:
            root = Path(temp)
            manifest_path = root / "benchmark_manifest.json"
            result_path = root / "benchmark_result.json"
            _write_json(manifest_path, config["benchmark_manifest"])
            _write_json(result_path, config["benchmark_result"])
            checked = load_and_validate_formal_training_inputs(
                benchmark_manifest_path=manifest_path,
                benchmark_result_path=result_path,
                budget_authorization_path=None,
                runtime_bundle=config["runtime"],
                gate1_identity=config["gate1_identity"],
                runtime_identity=config["runtime_identity"],
                deployment_identity=config["deployment_identity"],
                arm="dense",
                seed=0,
            )
            self.assertEqual(checked["benchmark_manifest"], config["benchmark_manifest"])
            self.assertEqual(checked["benchmark_result"], config["benchmark_result"])
            self.assertEqual(checked["benchmark_identity"]["path"], str(manifest_path.resolve()))

            other_manifest = build_benchmark_manifest(
                gate1_identity=config["gate1_identity"],
                runtime_identity=config["runtime_identity"],
                deployment_identity=config["deployment_identity"],
                step_seconds=31.0,
                benchmark_artifact_bytes=1024,
                serialized_adapter_bytes=2048,
            )
            mismatched_result = build_benchmark_result(
                other_manifest,
                runtime_bundle=config["runtime"],
                runtime_open_check=_live_check(
                    config["runtime"]["restored_fp8_runtime"], "benchmark_open"
                ),
                runtime_close_check=_live_check(
                    config["runtime"]["restored_fp8_runtime"], "benchmark_close"
                ),
            )
            _write_json(root / "mismatched_result.json", mismatched_result)
            with self.assertRaisesRegex(ValueError, "supplied manifest unchanged"):
                load_and_validate_formal_training_inputs(
                    benchmark_manifest_path=manifest_path,
                    benchmark_result_path=root / "mismatched_result.json",
                    budget_authorization_path=None,
                    runtime_bundle=config["runtime"],
                    gate1_identity=config["gate1_identity"],
                    runtime_identity=config["runtime_identity"],
                    deployment_identity=config["deployment_identity"],
                    arm="dense",
                    seed=0,
                )

    def test_same_seed_dense_sparse_initial_lora_pair(self) -> None:
        dense = _run_config(arm="dense", seed=1)
        sparse = _run_config(arm="sparse", seed=1)
        self.assertEqual(dense["global_goal_schedule"], sparse["global_goal_schedule"])
        self.assertEqual(
            dense["global_goal_schedule_sha256"],
            sparse["global_goal_schedule_sha256"],
        )
        self.assertEqual(validate_paired_initial_lora(dense, sparse), "c" * 64)
        sparse["initial_lora_sha256"] = "d" * 64
        sparse = seal_payload(sparse)
        with self.assertRaisesRegex(ValueError, "initial LoRA hashes differ"):
            validate_paired_initial_lora(dense, sparse)

    def test_trainer_source_orders_seed_kbit_prepare_and_peft(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "h1_mt_grpo_train_h20.py"
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)

        def calls(function_name: str) -> list[str]:
            function = next(
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            )
            names = []
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if isinstance(target, ast.Name):
                    names.append((node.lineno, target.id))
                elif isinstance(target, ast.Attribute):
                    names.append((node.lineno, target.attr))
            return [name for _line, name in sorted(names)]

        seed_calls = calls("seed_before_model_construction")
        self.assertLess(seed_calls.index("seed"), seed_calls.index("manual_seed"))
        self.assertIn("manual_seed_all", seed_calls)
        construction_calls = calls("construct_qlora_model")
        self.assertLess(
            construction_calls.index("from_pretrained"),
            construction_calls.index("prepare_model_for_kbit_training"),
        )
        self.assertLess(
            construction_calls.index("prepare_model_for_kbit_training"),
            construction_calls.index("get_peft_model"),
        )
        execute_calls = calls("_execute")
        self.assertLess(
            execute_calls.index("seed_before_model_construction"),
            execute_calls.index("construct_qlora_model"),
        )
        self.assertIn('parser.add_argument("--benchmark-result")', source)
        self.assertIn('parser.add_argument("--budget-authorization")', source)
        self.assertNotIn("allow-over-budget", source)
        self.assertIn("loaded attacker quantization differs from NF4 double-quant/BF16", source)
        self.assertIn("k-bit preparation did not enable gradient checkpointing", source)
        self.assertIn("time.monotonic()", source)
        self.assertNotIn("time.time()", source)

    def test_cli_help_and_negative_paths_do_not_import_powered_stack(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "h1_mt_grpo_train_h20.py"

        def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, str(script), *arguments],
                cwd=Path(__file__).parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

        helped = invoke("--help")
        self.assertEqual(helped.returncode, 0, helped.stderr)
        self.assertIn("--benchmark-result", helped.stdout)
        self.assertNotIn("ModuleNotFoundError", helped.stderr)

        missing = invoke()
        self.assertEqual(missing.returncode, 2)
        self.assertIn("the following arguments are required", missing.stderr)
        self.assertNotIn("ModuleNotFoundError", missing.stderr)

        common = (
            "--profile", LEGACY_H20_PROFILE_ID,
            "--arm", "dense", "--seed", "0", "--gate1-spec", "fixture.json",
        )
        mixed = invoke(
            *common, "--benchmark", "--benchmark-manifest", "fixture.json"
        )
        self.assertEqual(mixed.returncode, 2)
        self.assertIn(
            "--benchmark rejects formal benchmark/budget evidence arguments",
            mixed.stderr,
        )

        incomplete = invoke(*common, "--benchmark-manifest", "fixture.json")
        self.assertEqual(incomplete.returncode, 2)
        self.assertIn(
            "formal training requires both --benchmark-manifest and --benchmark-result",
            incomplete.stderr,
        )

    def test_artifact_roundtrip_and_campaign_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h20-artifact-") as temp:
            run, config, artifact = _fixture(Path(temp))
            self.assertEqual(
                validate_artifact_manifest(artifact, run_dir=run, run_config=config), artifact
            )
            provenance = load_and_validate_h20_adapter(
                run / "adapter", expected_arm="dense", expected_seed=0,
                expected_gate_bundle=config["gate1"],
                expected_runtime_reference=config["runtime"],
                expected_deployment=config["deployment"],
            )
            self.assertEqual(provenance["adapter_tree_sha256"], artifact["adapter"]["tree_sha256"])
            with self.assertRaisesRegex(ValueError, "arm/seed"):
                load_and_validate_h20_adapter(
                    run / "adapter", expected_arm="sparse", expected_seed=0
                )
            with self.assertRaisesRegex(ValueError, "Gate provenance"):
                load_and_validate_h20_adapter(
                    run / "adapter", expected_arm="dense", expected_seed=0,
                    expected_gate_bundle={"verdict": "PASS", "different": True},
                )

    def test_adapter_checkpoint_and_manifest_byte_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            (run / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "adapter tree hash"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            checkpoint_path = run / "adapter_step30.manifest.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["lora_sha256"] = "f" * 64
            _write_json(checkpoint_path, checkpoint)
            artifact["checkpoints"][0]["manifest_file_sha256"] = sha256_file(checkpoint_path)
            artifact = seal_payload(artifact)
            with self.assertRaisesRegex(ValueError, "payload seal"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            (run / "adapter_step30" / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint adapter tree hash"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            with (run / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            artifact["progress"]["file_sha256"] = sha256_file(run / "progress.jsonl")
            artifact = seal_payload(artifact)
            with self.assertRaisesRegex(ValueError, "progress file row count"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            progress = [
                json.loads(line)
                for line in (run / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            progress[-1]["lora_sha256"] = "f" * 64
            (run / "progress.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress),
                encoding="utf-8",
            )
            artifact["progress"]["file_sha256"] = sha256_file(run / "progress.jsonl")
            artifact = seal_payload(artifact)
            with self.assertRaisesRegex(ValueError, "final adapter LoRA hash"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            progress = [
                json.loads(line)
                for line in (run / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            progress[29]["lora_sha256"] = "f" * 64
            (run / "progress.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress),
                encoding="utf-8",
            )
            artifact["progress"]["file_sha256"] = sha256_file(run / "progress.jsonl")
            artifact = seal_payload(artifact)
            with self.assertRaisesRegex(ValueError, "checkpoint 30 LoRA hash"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            rollouts = (run / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
            rollouts[0], rollouts[1] = rollouts[1], rollouts[0]
            (run / "rollouts.jsonl").write_text(
                "\n".join(rollouts) + "\n", encoding="utf-8"
            )
            artifact["rollouts"]["file_sha256"] = sha256_file(run / "rollouts.jsonl")
            artifact = seal_payload(artifact)
            with self.assertRaisesRegex(ValueError, "exact formal step/group/trajectory order"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)

        with tempfile.TemporaryDirectory(prefix="h20-tamper-") as temp:
            run, config, artifact = _fixture(Path(temp))
            (run / "unsealed.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "formal run inventory mismatch"):
                validate_artifact_manifest(artifact, run_dir=run, run_config=config)


if __name__ == "__main__":
    unittest.main()
