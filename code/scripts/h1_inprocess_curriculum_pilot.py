"""Run one base/dense/sparse panel of the partial-reachable H1 pilot on H20."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
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

import h1_mt_grpo_train_h20 as core  # noqa: E402
from src.deployment_identity import verify_deployment  # noqa: E402
from src.domains.tooluse_injection import ToolUseInjectionDomain  # noqa: E402
from src.h20_serving_identity import MANIFEST_PATH, validate_service_manifest  # noqa: E402
from src.inprocess_curriculum_protocol import (  # noqa: E402
    AUTHORIZED_INSTANCE,
    PROFILE_ID,
    build_balanced_schedule,
    build_run_config,
    file_sha256,
    load_config,
    seal_payload,
    validate_seal,
)
from src.inprocess_curriculum_protocol import build_gate_selection  # noqa: E402
from src.local_vllm_victim import (  # noqa: E402
    LOCAL_VICTIM_MODEL,
    LOCAL_VICTIM_PROVIDER,
    LocalVllmVictimClient,
    load_local_vllm_ledger,
)
from src.model_pins import ATTACKER_MODEL, ATTACKER_REVISION, INJECAGENT_COMMIT  # noqa: E402
from src.qwen35_fast_kernels import require_qwen35_fast_kernels  # noqa: E402
from src.tooluse_gate1_spec import VICTIM_MAX_TOKENS  # noqa: E402


REQUIRED_DEPLOYMENT_PATHS = (
    "data/InjecAgent/data/test_cases_ds_base.json",
    "data/InjecAgent/data/tools.json",
    "data/InjecAgent/src/prompts/agent_prompts.py",
    "code/configs/h1_partial_reachable_curriculum_v1.json",
    "code/configs/h1_gate_partial_curriculum_v1.json",
    "code/configs/h1_gate_partial_none_curriculum_v1.json",
    "code/configs/h1_gate_partial_legacy_curriculum_v1.json",
    "code/configs/injecagent_ds_base_split_v1.json",
    "code/scripts/h1_inprocess_curriculum_pilot.py",
    "code/scripts/h1_mt_grpo_train_h20.py",
    "code/src/generation_runtime.py",
    "code/src/inprocess_curriculum_protocol.py",
    "code/src/local_victim_decision_protocol.py",
    "code/src/local_vllm_victim.py",
)

_EXACT_OUTER_INJECT = re.compile(r"\A\s*<inject>(?P<content>[\s\S]+)</inject>\s*\Z", re.IGNORECASE)
_ANY_INJECT_TAG = re.compile(r"</?inject>", re.IGNORECASE)


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
    os.replace(temporary, path)


def _hardware_identity() -> dict:
    expected_uuid = os.environ.get("H1_H20_GPU_UUID")
    if (
        os.environ.get("H1_INSTANCE_ID") != AUTHORIZED_INSTANCE
        or not expected_uuid
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid
    ):
        raise RuntimeError("pilot is not bound to the authorized H20 instance/GPU UUID")
    if not core.torch.cuda.is_available() or core.torch.cuda.device_count() != 1:
        raise RuntimeError("pilot must see exactly one CUDA device")
    properties = core.torch.cuda.get_device_properties(0)
    capability = list(core.torch.cuda.get_device_capability(0))
    if "H20" not in str(properties.name).upper() or capability != [9, 0]:
        raise RuntimeError("pilot CUDA device is not an H20")
    return {
        "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": expected_uuid,
        "name": str(properties.name),
        "compute_capability": capability,
        "memory_total_mib": int(properties.total_memory // (1024 * 1024)),
        "torch_version": str(core.torch.__version__),
        "cuda_version": str(core.torch.version.cuda),
    }


def _validate_live_service() -> dict:
    manifest = validate_service_manifest(
        json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8")),
        expected_quantization="fp8",
    )
    process = manifest["process"]
    proc_root = Path("/proc") / str(process["pid"])
    stat = (proc_root / "stat").read_text(encoding="utf-8")
    start_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    if start_ticks != process["start_time_ticks"]:
        raise RuntimeError("local FP8 victim PID lifecycle changed")
    if manifest["gpu"]["uuid"] != os.environ.get("H1_H20_GPU_UUID"):
        raise RuntimeError("local victim and attacker GPU UUID differ")
    core._probe_victim()
    return manifest


def _portable_rollout(result: dict, *, goal_id: str, trajectory_index: int) -> dict:
    return {
        "goal": goal_id,
        "trajectory_index": trajectory_index,
        "phi_trace": list(result["phi_trace"]),
        "max_phi": float(result["max_phi"]),
        "success": bool(result["success"]),
        "n_turns": int(result["n_turns"]),
        "calls": [
            {"tool": tool, "arguments": arguments}
            for tool, arguments in result["calls"]
        ],
        "turns": [{
            key: value for key, value in turn.items()
            if key not in {"prompt_ids", "resp_ids"}
        } for turn in result["turns"]],
    }


def _pilot_compatible_generator(generator, *, ledger_path: Path):
    """Unwrap exactly one legacy outer frame while retaining the raw completion.

    This pilot-only bridge accepts only surrounding whitespace and rejects nested or residual
    reserved tags. Every raw completion is durably recorded before parsing. Token IDs remain those
    of the raw model completion, so policy-gradient accounting and the raw/normalized transport
    distinction are both auditable.
    """
    if ledger_path.exists():
        raise FileExistsError(f"raw attacker ledger already exists: {ledger_path}")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.touch(exist_ok=False)
    state = {"call": 0}

    def token_ids(value) -> list[int]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in value
        ):
            raise RuntimeError("raw attacker response token IDs are malformed")
        return value

    def generate(batch_messages):
        rows = generator(batch_messages)
        state["call"] += 1
        with ledger_path.open("a", encoding="utf-8") as handle:
            for item_index, row in enumerate(rows):
                raw = row.get("text")
                response_token_ids = token_ids(row.get("resp_ids"))
                handle.write(json.dumps({
                    "schema_version": 2,
                    "kind": "h1_partial_reachable_raw_attacker_generation",
                    "call": state["call"],
                    "item_index": item_index,
                    "raw_text": raw,
                    "raw_text_sha256": hashlib.sha256(
                        str(raw).encode("utf-8")
                    ).hexdigest(),
                    "raw_response_token_ids": response_token_ids,
                    "raw_response_token_ids_sha256": hashlib.sha256(
                        json.dumps(response_token_ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        normalized = []
        for row in rows:
            value = dict(row)
            raw = value.get("text")
            value["raw_model_text"] = raw
            if isinstance(raw, str) and _ANY_INJECT_TAG.search(raw):
                match = _EXACT_OUTER_INJECT.fullmatch(raw)
                if match is None or _ANY_INJECT_TAG.search(match.group("content")):
                    raise RuntimeError("attacker wrapper is not one exact outer inject frame")
                value["text"] = match.group("content")
                value["pilot_transport_normalization"] = "exact_outer_inject_unwrapped"
            else:
                value["pilot_transport_normalization"] = "none"
            normalized.append(value)
        return normalized

    generate.set_step = generator.set_step
    generate.metrics = generator.metrics
    return generate


def _load_pair_configs(
    paths: list[str], *, arm: str, seed: int, initial_hash: str,
    deployment_tree_sha256: str, n_goals: int, config_file_sha256: str,
    curriculum_variant: str, defense_tier: str, victim_decision_protocol_id: str,
    strict_declared_action_arguments: bool, final_answer_max_length: int | None,
    action_string_max_length: int | None,
) -> list[dict]:
    expected = {"base"} if arm == "dense" else ({"base", "dense"} if arm == "sparse" else set())
    if len(paths) != len(expected):
        raise ValueError(f"{arm} panel requires paired configs for {sorted(expected)}")
    pairs = []
    for path in paths:
        value = validate_seal(json.loads(Path(path).read_text(encoding="utf-8")))
        if value.get("seed") != seed or value.get("initial_lora_sha256") != initial_hash:
            raise ValueError("paired panel seed or initial LoRA hash differs")
        if value.get("deployment_tree_sha256") != deployment_tree_sha256:
            raise ValueError("paired panel deployment tree differs")
        if value.get("config_file_sha256") != config_file_sha256:
            raise ValueError("paired panel curriculum config differs")
        if value.get("curriculum_variant") != curriculum_variant:
            raise ValueError("paired panel curriculum variant differs")
        if value.get("defense_tier") != defense_tier:
            raise ValueError("paired panel defense tier differs")
        if value.get("victim_decision_protocol_id") != victim_decision_protocol_id:
            raise ValueError("paired panel victim decision protocol differs")
        if (
            value.get("strict_declared_action_arguments")
            is not strict_declared_action_arguments
        ):
            raise ValueError("paired panel strict action-argument policy differs")
        if value.get("final_answer_max_length") != final_answer_max_length:
            raise ValueError("paired panel final-answer bound differs")
        if value.get("action_string_max_length") != action_string_max_length:
            raise ValueError("paired panel action-string bound differs")
        pairs.append(value)
    if {value.get("arm") for value in pairs} != expected:
        raise ValueError("paired panel arm set mismatch")
    schedules = [value.get("goal_schedule") for value in pairs if value.get("arm") != "base"]
    if schedules and any(
        schedule != build_balanced_schedule(seed=seed, n_goals=n_goals)
        for schedule in schedules
    ):
        raise ValueError("paired panel schedule differs")
    return pairs


def _evaluate(
    *, model, tokenizer, domain, goals: list, seed: int, run_dir: Path,
    samples_per_goal: int, victim_decision_protocol_id: str,
    strict_declared_action_arguments: bool,
    final_answer_max_length: int | None, action_string_max_length: int | None,
) -> tuple[list[dict], dict, float, dict]:
    generator = _pilot_compatible_generator(core.make_gen_batch_fn(
        model, tokenizer, training_seed=seed, cached_eval_mode=True, max_new_tokens=256
    ), ledger_path=run_dir / "eval_raw_attacker_ledger.jsonl")
    generator.set_step(10_000 + seed)
    client = LocalVllmVictimClient(
        run_dir / "eval_victim_ledger.jsonl",
        final_answer_max_length=final_answer_max_length,
        action_string_max_length=action_string_max_length,
        strict_declared_action_arguments=strict_declared_action_arguments,
    )
    victim_batch = core.make_victim_batch_fn(
        domain, core.FIXED_TRAINING["workers"], client,
        max_calls=3, max_tokens=VICTIM_MAX_TOKENS, temperature=0.0,
        provider=LOCAL_VICTIM_PROVIDER, model=LOCAL_VICTIM_MODEL,
        decision_protocol_id=victim_decision_protocol_id,
    )
    items = [goal for goal in goals for _ in range(samples_per_goal)]
    started = time.monotonic()
    results = core.rollout_batch(domain, items, generator, victim_batch, T=5, tau=1.0)
    elapsed = time.monotonic() - started
    rows = []
    for goal_offset, goal in enumerate(goals):
        for trajectory_index in range(samples_per_goal):
            rows.append(_portable_rollout(
                results[goal_offset * samples_per_goal + trajectory_index],
                goal_id=goal.id, trajectory_index=trajectory_index,
            ))
    ledger = load_local_vllm_ledger(
        run_dir / "eval_victim_ledger.jsonl", require_complete=True
    )
    summary = {
        "count": len(rows),
        "success_count": sum(int(row["success"]) for row in rows),
        "asr": sum(int(row["success"]) for row in rows) / len(rows),
        "mean_max_phi": sum(row["max_phi"] for row in rows) / len(rows),
        "attacker_generation_seconds": float(generator.metrics["generation_seconds"]),
        "wall_seconds": elapsed,
    }
    return rows, summary, elapsed, ledger


def execute(args: argparse.Namespace) -> Path:
    if args.profile != PROFILE_ID:
        raise RuntimeError("pilot profile mismatch")
    core._load_powered_stack()
    core._validate_environment_overrides()
    hardware = _hardware_identity()
    run_root = Path(args.run_root).resolve()
    deployment = verify_deployment(run_root, required_paths=REQUIRED_DEPLOYMENT_PATHS)
    if deployment.get("injecagent_commit") != INJECAGENT_COMMIT:
        raise RuntimeError("pilot deployment InjecAgent identity mismatch")
    config = load_config(args.config)
    if args.seed not in config["training"]["seeds"]:
        raise RuntimeError("pilot seed is not registered by the selected curriculum")
    gate_selection = build_gate_selection(args.gate_selection)
    service_manifest = _validate_live_service()

    defense_tier = config["interaction"]["defense_tier"]
    victim_decision_protocol_id = config["interaction"][
        "victim_decision_protocol_id"
    ]
    strict_declared_action_arguments = config["interaction"][
        "strict_declared_action_arguments"
    ]
    final_answer_max_length = config["interaction"].get("final_answer_max_length")
    action_string_max_length = config["interaction"].get("action_string_max_length")
    domain = ToolUseInjectionDomain(attack="ds", defense_tier=defense_tier)
    source_split = config["data"].get("source_split", "train")
    expected_goal_count = 69 if source_split == "calibration" else 322
    all_goals = domain.load_goals(source_split, seed=0, n=expected_goal_count)
    if (
        len(all_goals) != expected_goal_count
        or domain.dataset_sha256 != config["data"]["dataset_sha256"]
        or domain.split_manifest.get("manifest_id") != config["data"]["split_manifest_id"]
    ):
        raise RuntimeError("pilot dataset/split identity mismatch")
    by_id = {goal.id: goal for goal in all_goals}
    selected_ids = config["data"]["training_goal_ids"]
    heldout_ids = config["data"]["heldout_goal_ids"]
    if any(goal_id not in by_id for goal_id in selected_ids + heldout_ids):
        raise RuntimeError("pilot selected goal is absent from the frozen source split")

    run_dir = run_root / "partial_reachable_pilot" / args.tag
    run_dir.mkdir(parents=True, exist_ok=False)
    core.snapshot_download(ATTACKER_MODEL, revision=ATTACKER_REVISION, local_files_only=True)
    core.seed_before_model_construction(args.seed)
    tokenizer, model = core.construct_qlora_model()
    kernel_status = require_qwen35_fast_kernels(model)
    initial_hash = core.lora_parameter_sha256(model)
    paired = _load_pair_configs(
        args.paired_config, arm=args.arm, seed=args.seed, initial_hash=initial_hash,
        deployment_tree_sha256=deployment["deployed_tree_sha256"],
        n_goals=len(selected_ids),
        config_file_sha256=config["config_file_sha256"],
        curriculum_variant=config["curriculum_variant"],
        defense_tier=defense_tier,
        victim_decision_protocol_id=victim_decision_protocol_id,
        strict_declared_action_arguments=strict_declared_action_arguments,
        final_answer_max_length=final_answer_max_length,
        action_string_max_length=action_string_max_length,
    )
    run_config = build_run_config(
        config=config, arm=args.arm, seed=args.seed, tag=args.tag,
        gpu_uuid=hardware["gpu_uuid"],
        deployment_tree_sha256=deployment["deployed_tree_sha256"],
        service_manifest_payload_sha256=service_manifest["payload_sha256"],
        gate_selection_payload_sha256=gate_selection["payload_sha256"],
        initial_lora_sha256=initial_hash,
    )
    run_config["created_at"] = _now()
    run_config["hardware"] = hardware
    run_config["kernel_status"] = kernel_status
    run_config["paired_config_payloads"] = [value["payload_sha256"] for value in paired]
    # Re-seal after adding runtime evidence.
    run_config.pop("payload_sha256")
    run_config = seal_payload(run_config)
    _atomic_json(run_dir / "run_config.json", run_config)

    training_seconds = 0.0
    rollout_rows = 0
    final_hash = initial_hash
    train_ledger = None
    adapter_tree = None
    if args.arm != "base":
        train_client = LocalVllmVictimClient(
            run_dir / "train_victim_ledger.jsonl",
            final_answer_max_length=final_answer_max_length,
            action_string_max_length=action_string_max_length,
            strict_declared_action_arguments=strict_declared_action_arguments,
        )
        schedule = build_balanced_schedule(seed=args.seed, n_goals=len(selected_ids))
        trainer_args = argparse.Namespace(arm=args.arm, seed=args.seed)
        attacker_generator = _pilot_compatible_generator(core.make_gen_batch_fn(
            model, tokenizer, training_seed=args.seed, cached_eval_mode=True,
            max_new_tokens=256,
        ), ledger_path=run_dir / "train_raw_attacker_ledger.jsonl")
        final_hash, training_seconds, rollout_rows = core._run_training_loop(
            model=model, tokenizer=tokenizer, domain=domain,
            goals=[by_id[goal_id] for goal_id in selected_ids], schedule=schedule,
            args=trainer_args, run_dir=run_dir,
            steps=config["training"]["steps_per_arm"], save_checkpoints=False,
            run_config=None, victim_chat_fn=train_client,
            victim_provider=LOCAL_VICTIM_PROVIDER, victim_model=LOCAL_VICTIM_MODEL,
            victim_max_tokens=VICTIM_MAX_TOKENS,
            cached_attacker_generation=True, attacker_generator=attacker_generator,
            victim_decision_protocol_id=victim_decision_protocol_id,
            shape_override={
                "n_goals": config["training"]["goals_per_step"],
                "G": config["training"]["trajectories_per_goal"],
                "T": config["interaction"]["T"],
            },
        )
        train_ledger = load_local_vllm_ledger(
            run_dir / "train_victim_ledger.jsonl", require_complete=True
        )
        adapter_tree = core._save_adapter_atomic(model, run_dir / "adapter_final")

    heldout_rows, eval_summary, evaluation_seconds, eval_ledger = _evaluate(
        model=model, tokenizer=tokenizer, domain=domain,
        goals=[by_id[goal_id] for goal_id in heldout_ids], seed=args.seed,
        run_dir=run_dir,
        samples_per_goal=config["evaluation"]["samples_per_goal"],
        victim_decision_protocol_id=victim_decision_protocol_id,
        strict_declared_action_arguments=strict_declared_action_arguments,
        final_answer_max_length=final_answer_max_length,
        action_string_max_length=action_string_max_length,
    )
    _atomic_jsonl(run_dir / "heldout_rollouts.jsonl", heldout_rows)
    result = seal_payload({
        "schema_version": 1,
        "kind": "h1_partial_reachable_inprocess_result",
        "profile_id": PROFILE_ID,
        "decision_bearing": False,
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": hardware["gpu_uuid"],
        "run_config_payload_sha256": run_config["payload_sha256"],
        "initial_lora_sha256": initial_hash,
        "final_lora_sha256": final_hash,
        "adapter_tree_sha256": adapter_tree,
        "training": {
            "steps": 0 if args.arm == "base" else config["training"]["steps_per_arm"],
            "rollout_rows": rollout_rows,
            "wall_seconds": training_seconds,
            "victim_ledger": train_ledger,
        },
        "evaluation": eval_summary,
        "eval_victim_ledger": eval_ledger,
        "heldout_rollouts_file_sha256": file_sha256(run_dir / "heldout_rollouts.jsonl"),
        "completed_at": _now(),
    })
    _atomic_json(run_dir / "result.json", result)
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files[path.relative_to(run_dir).as_posix()] = file_sha256(path)
    manifest = seal_payload({
        "schema_version": 1,
        "kind": "h1_partial_reachable_artifact_manifest",
        "profile_id": PROFILE_ID,
        "arm": args.arm,
        "seed": args.seed,
        "files": files,
        "result_payload_sha256": result["payload_sha256"],
    })
    _atomic_json(run_dir / "artifact_manifest.json", manifest)
    print(json.dumps({
        "status": "PANEL_COMPLETE", "run_dir": str(run_dir),
        "arm": args.arm, "seed": args.seed,
        "asr": eval_summary["asr"], "mean_max_phi": eval_summary["mean_max_phi"],
        "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
        "result_payload_sha256": result["payload_sha256"],
    }, sort_keys=True), flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=(PROFILE_ID,))
    parser.add_argument("--arm", required=True, choices=("base", "dense", "sparse"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate-selection", required=True)
    parser.add_argument("--paired-config", action="append", default=[])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.tag.startswith(f"h1-pr-{args.arm}-s{args.seed}-"):
        raise SystemExit("tag does not match arm/seed")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
