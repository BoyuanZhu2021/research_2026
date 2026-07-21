"""Run one audited calibration or final-OOD panel with the proven in-process profile."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))

import h1_inprocess_curriculum_pilot as pilot  # noqa: E402
import h1_mt_grpo_train_h20 as core  # noqa: E402
from src.deployment_identity import verify_deployment  # noqa: E402
from src.domains.tooluse_injection import ToolUseInjectionDomain  # noqa: E402
from src.h20_training_artifacts import tree_sha256 as canonical_adapter_tree_sha256  # noqa: E402
from src.generation_runtime import ReservedTagDecoderGuard  # noqa: E402
from src.inprocess_curriculum_protocol import (  # noqa: E402
    AUTHORIZED_INSTANCE, file_sha256, seal_payload, validate_seal,
)
from src.local_vllm_victim import (  # noqa: E402
    FINAL_C0_TRANSPORT_ID, FINAL_C0_TRANSPORT_POLICY_SHA256,
    LOCAL_VICTIM_MODEL, LOCAL_VICTIM_PROVIDER, LocalVllmVictimClient,
    load_local_vllm_ledger,
)
from src.model_pins import ATTACKER_MODEL, ATTACKER_REVISION, INJECAGENT_COMMIT  # noqa: E402
from src.qwen35_fast_kernels import require_qwen35_fast_kernels  # noqa: E402
from src.tooluse_gate1_spec import VICTIM_MAX_TOKENS  # noqa: E402


PROFILE_ID = "h1-gate-partial-confirmatory-final-c0-transport-v1"
PROFILE_CONFIG = CODE / "configs" / "h1_confirmatory_final_c0_transport_v1.json"
PHASES = {"learning_report": ("calibration", 69), "final_ood": ("final_ood", 153)}
GRID = (("base", None), ("dense", 0), ("sparse", 0), ("dense", 1),
        ("sparse", 1), ("dense", 2), ("sparse", 2))
FINAL_AUTH_KIND = "h1_gate_partial_final_ood_authorization"
REQUIRED_DEPLOYMENT_PATHS = pilot.REQUIRED_DEPLOYMENT_PATHS + (
    "code/scripts/h1_inprocess_confirmatory_eval.py",
    "code/scripts/h1_inprocess_confirmatory_analyze.py",
    "code/configs/h1_confirmatory_final_c0_transport_v1.json",
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tree_sha256(root: Path) -> str:
    """Reuse the exact tree serialization that sealed the training adapter."""
    return canonical_adapter_tree_sha256(root)


def _load_profile_config() -> dict:
    value = json.loads(PROFILE_CONFIG.read_text(encoding="utf-8"))
    guard = value.get("decoder_guard") or {}
    transport = value.get("victim_transport") or {}
    campaign = value.get("campaign_policy") or {}
    if (
        value.get("schema_version") != 1
        or value.get("profile_id") != PROFILE_ID
        or guard.get("guard_id") != "h1-content-only-reserved-tag-dfa-v1"
        or guard.get("post_generation_repair") is not False
        or transport.get("transport_id") != FINAL_C0_TRANSPORT_ID
        or transport.get("policy_payload_sha256")
        != FINAL_C0_TRANSPORT_POLICY_SHA256
        or transport.get("strict_parse_first") is not True
        or transport.get("action_decisions_fail_closed") is not True
        or transport.get("http_retry") is not False
        or campaign.get("post_exposure_confirmation") is not True
        or campaign.get("fresh_campaign_required") is not True
        or campaign.get("old_and_new_panel_mixing_forbidden") is not True
        or campaign.get("registered_grid")
        != [panel_key(arm, seed) for arm, seed in GRID]
    ):
        raise ValueError("constrained-eval profile config is invalid")
    return value


def _load_adapter_provenance(adapter: Path, arm: str, seed: int) -> dict:
    adapter = adapter.resolve()
    if adapter.name != "adapter_final":
        raise ValueError("confirmatory adapter path must end in adapter_final")
    run_dir = adapter.parent
    config = validate_seal(json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")))
    result = validate_seal(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
    manifest = validate_seal(json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8")))
    if (config.get("arm"), config.get("seed")) != (arm, seed):
        raise ValueError("adapter arm/seed mismatch")
    if result.get("run_config_payload_sha256") != config.get("payload_sha256"):
        raise ValueError("adapter result/config binding mismatch")
    if manifest.get("result_payload_sha256") != result.get("payload_sha256"):
        raise ValueError("adapter manifest/result binding mismatch")
    for relative, digest in (manifest.get("files") or {}).items():
        if file_sha256(run_dir / relative) != digest:
            raise ValueError(f"adapter source artifact drift: {relative}")
    expected_runtime = {
        "defense_tier": "light",
        "victim_decision_protocol_id": "h1-victim-one-decision-step-bound-observation-ref-v3",
        "strict_declared_action_arguments": True,
        "final_answer_max_length": 512,
        "action_string_max_length": 512,
    }
    for key, expected in expected_runtime.items():
        if config.get(key) != expected:
            raise ValueError(f"adapter runtime mismatch: {key}")
    tree = _tree_sha256(adapter)
    if tree != result.get("adapter_tree_sha256"):
        raise ValueError("adapter tree hash mismatch")
    return {
        "kind": "h1_gate_partial_confirmatory_adapter",
        "arm": arm,
        "training_seed": seed,
        "source_tag": result["tag"],
        "source_result_payload_sha256": result["payload_sha256"],
        "source_run_config_payload_sha256": config["payload_sha256"],
        "adapter_tree_sha256": tree,
        "final_lora_sha256": result["final_lora_sha256"],
    }


def _base_provenance() -> dict:
    return {
        "kind": "h1_gate_partial_confirmatory_base_best_of_4",
        "model": ATTACKER_MODEL,
        "revision": ATTACKER_REVISION,
        "construction_seed": 0,
        "best_of_k": 4,
    }


def _validate_final_inputs(args: argparse.Namespace, provenance: dict) -> tuple[dict, dict]:
    report = validate_seal(json.loads(Path(args.learning_report).read_text(encoding="utf-8")))
    authorization = validate_seal(json.loads(Path(args.final_authorization).read_text(encoding="utf-8")))
    if (
        report.get("kind") != "h1_gate_partial_learning_report"
        or report.get("profile_id") != PROFILE_ID
        or report.get("complete") is not True
        or report.get("decision_bearing") is not False
        or report.get("post_exposure_confirmation") is not True
        or report.get("victim_final_c0_transport_id")
        != FINAL_C0_TRANSPORT_ID
        or report.get("victim_final_c0_transport_policy_sha256")
        != FINAL_C0_TRANSPORT_POLICY_SHA256
    ):
        raise ValueError("final OOD requires a complete non-decision-bearing learning report")
    if (
        authorization.get("kind") != FINAL_AUTH_KIND
        or authorization.get("instance_id") != AUTHORIZED_INSTANCE
        or authorization.get("profile_id") != PROFILE_ID
        or authorization.get("post_exposure_confirmation") is not True
        or authorization.get("victim_final_c0_transport_id")
        != FINAL_C0_TRANSPORT_ID
        or authorization.get("victim_final_c0_transport_policy_sha256")
        != FINAL_C0_TRANSPORT_POLICY_SHA256
        or authorization.get("learning_report_payload_sha256") != report.get("payload_sha256")
        or authorization.get("learning_report_file_sha256") != file_sha256(args.learning_report)
        or authorization.get("final_campaign_id") != args.campaign_id
    ):
        raise ValueError("final-OOD authorization binding mismatch")
    key = panel_key(args.arm, args.seed)
    if (authorization.get("policy_registry") or {}).get(key) != provenance:
        raise ValueError("final-OOD panel policy is not the authorized policy")
    return report, authorization


def _load_adapter_into_model(model, adapter: Path, expected_lora_sha256: str) -> None:
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    weights = load_peft_weights(str(adapter), device="cpu")
    set_peft_model_state_dict(model, weights, adapter_name="default")
    if core.lora_parameter_sha256(model) != expected_lora_sha256:
        raise RuntimeError("loaded adapter LoRA hash differs from sealed training result")


def panel_key(arm: str, seed: int | None) -> str:
    return "base-k4" if arm == "base" else f"{arm}-s{seed}"


def _evaluate_chunked(*, model, tokenizer, decoder_guard: ReservedTagDecoderGuard,
                      domain, goals: list, args: argparse.Namespace,
                      run_dir: Path) -> tuple[list[dict], dict, dict]:
    training_seed = 0 if args.arm == "base" else args.seed
    generator = pilot._pilot_compatible_generator(core.make_gen_batch_fn(
        model, tokenizer, training_seed=training_seed, cached_eval_mode=True,
        max_new_tokens=256, logits_processor_factory=decoder_guard.bind,
    ), ledger_path=run_dir / "raw_attacker_ledger.jsonl")
    client = LocalVllmVictimClient(
        run_dir / "raw_victim_ledger.jsonl",
        final_answer_max_length=512,
        action_string_max_length=512,
        strict_declared_action_arguments=True,
        final_c0_transport_id=FINAL_C0_TRANSPORT_ID,
    )
    victim_batch = core.make_victim_batch_fn(
        domain, core.FIXED_TRAINING["workers"], client, max_calls=3,
        max_tokens=VICTIM_MAX_TOKENS, temperature=0.0,
        provider=LOCAL_VICTIM_PROVIDER, model=LOCAL_VICTIM_MODEL,
        decision_protocol_id="h1-victim-one-decision-step-bound-observation-ref-v3",
    )
    eval_seeds = range(4) if args.arm == "base" else (args.seed,)
    rows: list[dict] = []
    generation_before = float(generator.metrics["generation_seconds"])
    started = time.monotonic()
    for eval_seed in eval_seeds:
        for offset in range(0, len(goals), args.chunk_size):
            chunk = goals[offset:offset + args.chunk_size]
            generator.set_step(100_000 + int(eval_seed) * 1_000 + offset)
            results = core.rollout_batch(domain, chunk, generator, victim_batch, T=5, tau=1.0)
            if len(results) != len(chunk):
                raise RuntimeError("confirmatory evaluation denominator mismatch")
            for local_index, (goal, result) in enumerate(zip(chunk, results, strict=True)):
                row = pilot._portable_rollout(
                    result, goal_id=goal.id,
                    trajectory_index=int(eval_seed) if args.arm == "base" else 0,
                )
                row.update({
                    "phase": args.phase,
                    "campaign_id": args.campaign_id,
                    "arm": args.arm,
                    "training_seed": args.seed,
                    "eval_seed": int(eval_seed),
                    "goal_index": offset + local_index,
                })
                rows.append(row)
    elapsed = time.monotonic() - started
    victim_ledger = load_local_vllm_ledger(
        run_dir / "raw_victim_ledger.jsonl",
        require_complete=True,
        expected_final_c0_transport_id=FINAL_C0_TRANSPORT_ID,
    )
    expected = len(goals) * (4 if args.arm == "base" else 1)
    if len(rows) != expected or any(not math.isfinite(float(row["max_phi"])) for row in rows):
        raise RuntimeError("confirmatory panel is incomplete or non-finite")
    summary = {
        "n_goals": len(goals),
        "n_rows": len(rows),
        "success_count": sum(int(row["success"]) for row in rows),
        "asr": sum(int(row["success"]) for row in rows) / len(rows),
        "mean_max_phi": sum(float(row["max_phi"]) for row in rows) / len(rows),
        "attacker_generation_seconds": (
            float(generator.metrics["generation_seconds"]) - generation_before
        ),
        "wall_seconds": elapsed,
        "decoder_guard_metrics": decoder_guard.metrics(),
    }
    return rows, summary, victim_ledger


def execute(args: argparse.Namespace) -> Path:
    core._load_powered_stack()
    core._validate_environment_overrides()
    hardware = pilot._hardware_identity()
    run_root = Path(args.run_root).resolve()
    deployment = verify_deployment(run_root, required_paths=REQUIRED_DEPLOYMENT_PATHS)
    if deployment.get("injecagent_commit") != INJECAGENT_COMMIT:
        raise RuntimeError("confirmatory deployment InjecAgent identity mismatch")
    service_manifest = pilot._validate_live_service()
    profile_config = _load_profile_config()
    provenance = (
        _base_provenance() if args.arm == "base"
        else _load_adapter_provenance(Path(args.adapter), args.arm, args.seed)
    )
    learning_report = final_auth = None
    if args.phase == "final_ood":
        learning_report, final_auth = _validate_final_inputs(args, provenance)

    # This is deliberately after every final-authorization check above.
    split, expected_count = PHASES[args.phase]
    domain = ToolUseInjectionDomain(attack="ds", defense_tier="light")
    goals = domain.load_goals(split, seed=0, n=expected_count)
    if len(goals) != expected_count:
        raise RuntimeError("confirmatory split denominator mismatch")

    run_dir = run_root / "confirmatory_eval" / args.campaign_id / panel_key(args.arm, args.seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    core.snapshot_download(ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True)
    construction_seed = 0 if args.arm == "base" else args.seed
    core.seed_before_model_construction(construction_seed)
    tokenizer, model = core.construct_qlora_model()
    kernel_status = require_qwen35_fast_kernels(model)
    decoder_guard = ReservedTagDecoderGuard(
        tokenizer, tokenizer_revision=ATTACKER_REVISION,
    )
    decoder_guard_identity = decoder_guard.identity()
    initial_lora_sha256 = core.lora_parameter_sha256(model)
    if args.arm != "base":
        _load_adapter_into_model(model, Path(args.adapter), provenance["final_lora_sha256"])

    run_config = seal_payload({
        "schema_version": 1,
        "kind": "h1_gate_partial_confirmatory_eval_run",
        "profile_id": PROFILE_ID,
        "profile_config_file_sha256": file_sha256(PROFILE_CONFIG),
        "profile_config": profile_config,
        "phase": args.phase,
        "decision_bearing": args.phase == "final_ood",
        "post_exposure_confirmation": True,
        "campaign_id": args.campaign_id,
        "panel": panel_key(args.arm, args.seed),
        "arm": args.arm,
        "training_seed": args.seed,
        "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": hardware["gpu_uuid"],
        "deployment_tree_sha256": deployment["deployed_tree_sha256"],
        "service_manifest_payload_sha256": service_manifest["payload_sha256"],
        "goal_ids_sha256": hashlib.sha256("\n".join(goal.id for goal in goals).encode()).hexdigest(),
        "policy_provenance": provenance,
        "initial_lora_sha256": initial_lora_sha256,
        "learning_report_payload_sha256": None if learning_report is None else learning_report["payload_sha256"],
        "final_authorization_payload_sha256": None if final_auth is None else final_auth["payload_sha256"],
        "victim_decision_protocol_id": "h1-victim-one-decision-step-bound-observation-ref-v3",
        "strict_declared_action_arguments": True,
        "final_answer_max_length": 512,
        "action_string_max_length": 512,
        "victim_final_c0_transport_id": FINAL_C0_TRANSPORT_ID,
        "victim_final_c0_transport_policy_sha256": (
            FINAL_C0_TRANSPORT_POLICY_SHA256
        ),
        "hardware": hardware,
        "kernel_status": kernel_status,
        "attacker_decoder_guard": decoder_guard_identity,
        "created_at": _now(),
    })
    _atomic_json(run_dir / "run_config.json", run_config)
    rows, metrics, victim_ledger = _evaluate_chunked(
        model=model, tokenizer=tokenizer, decoder_guard=decoder_guard,
        domain=domain, goals=goals,
        args=args, run_dir=run_dir,
    )
    _atomic_jsonl(run_dir / "rows.jsonl", rows)
    result = seal_payload({
        "schema_version": 1,
        "kind": "h1_gate_partial_confirmatory_eval_result",
        "profile_id": PROFILE_ID,
        "phase": args.phase,
        "decision_bearing": args.phase == "final_ood",
        "post_exposure_confirmation": True,
        "campaign_id": args.campaign_id,
        "panel": panel_key(args.arm, args.seed),
        "arm": args.arm,
        "training_seed": args.seed,
        "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": hardware["gpu_uuid"],
        "run_config_payload_sha256": run_config["payload_sha256"],
        "policy_provenance": provenance,
        "attacker_decoder_guard": decoder_guard_identity,
        "metrics": metrics,
        "victim_ledger": victim_ledger,
        "rows_file_sha256": file_sha256(run_dir / "rows.jsonl"),
        "raw_attacker_ledger_file_sha256": file_sha256(run_dir / "raw_attacker_ledger.jsonl"),
        "raw_victim_ledger_file_sha256": file_sha256(run_dir / "raw_victim_ledger.jsonl"),
        "completed_at": _now(),
    })
    _atomic_json(run_dir / "result.json", result)
    files = {
        path.relative_to(run_dir).as_posix(): file_sha256(path)
        for path in sorted(run_dir.rglob("*")) if path.is_file()
    }
    manifest = seal_payload({
        "schema_version": 1,
        "kind": "h1_gate_partial_confirmatory_eval_manifest",
        "profile_id": PROFILE_ID,
        "phase": args.phase,
        "panel": panel_key(args.arm, args.seed),
        "result_payload_sha256": result["payload_sha256"],
        "files": files,
    })
    _atomic_json(run_dir / "artifact_manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "run_dir": str(run_dir), **metrics}, sort_keys=True))
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=tuple(PHASES))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--arm", required=True, choices=("base", "dense", "sparse"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--adapter")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--learning-report")
    parser.add_argument("--final-authorization")
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.arm == "base":
        if args.seed is not None or args.adapter is not None:
            parser.error("base forbids --seed and --adapter")
    elif args.seed is None or args.adapter is None:
        parser.error("trained arms require --seed and --adapter")
    final_inputs = (args.learning_report, args.final_authorization)
    if args.phase == "final_ood" and any(value is None for value in final_inputs):
        parser.error("final_ood requires learning report and explicit authorization")
    if args.phase != "final_ood" and any(value is not None for value in final_inputs):
        parser.error("learning_report phase forbids final inputs")
    if args.chunk_size < 1 or args.chunk_size > 64:
        parser.error("chunk size must be in [1,64]")
    return args


if __name__ == "__main__":
    execute(build_parser())
