"""Canonical two-worker GRPO/QLoRA protocol for formal tool-use H1 training."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from .runtime_profile import (
    RUNTIME_PROFILE_ID,
    RUNTIME_PROFILE_SHA256,
    WORKER_VICTIM_PORTS,
    WORLD_SIZE,
    runtime_profile,
    validate_runtime_manifest,
    validate_software_versions,
)
from .tooluse_gate1_spec import (
    VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
    VICTIM_REACT_STOP,
)
from .victim_decision_protocol import VICTIM_DECISION_PROTOCOL


TRAINING_PROTOCOL_SCHEMA_VERSION = 2
TRAINING_PROTOCOL_ID = "h1-tooluse-formal-grpo-v100-ddp-v2"
FORMAL_TRAINING_SEEDS = (0, 1, 2)

DISTRIBUTED_PROTOCOL = {
    "launcher": "torchrun",
    "world_size": WORLD_SIZE,
    "backend": "nccl",
    "global_n_goals": 8,
    "local_n_goals": 4,
    "trajectories_per_goal": 8,
    "shard_unit": "complete_goal_group",
    "slot_owner": "global_goal_slot_mod_world_size",
    "logical_device_index": 0,
    "victim_ports_by_local_rank": list(WORKER_VICTIM_PORTS),
    "gradient_sync": "manual_lora_grad_sum_once_per_step",
    "loss_normalization": "global_nonzero_example_count",
    "missing_lora_grad": "materialize_zero_before_all_reduce",
    "base_grad_policy": "must_remain_none",
    "artifact_writer": "rank0_only_with_per_rank_rollout_shards",
    "goal_schedule": "sha256-counter-seed-step-slot-no-arm-v1",
    "generation_seed": "sha256-counter-seed-step-rank-turn-no-arm-v1",
    "single_rank_failover": False,
    "victim_workers_scope": "per_rank",
    "throughput_status": "requires_full_shape_benchmark_then_protocol_freeze",
}

VICTIM_REQUEST = {
    "stop": list(VICTIM_REACT_STOP),
    "include_stop_str_in_output": VICTIM_INCLUDE_STOP_STR_IN_OUTPUT,
    "victim_output_protocol": VICTIM_DECISION_PROTOCOL,
}

_FORMAL_TRAINING_PROTOCOL = {
    "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
    "protocol_id": TRAINING_PROTOCOL_ID,
    "fixed_training": {
        "domain": "tooluse",
        "smoke": False,
        "benchmark": False,
        "steps": 60,
        "n_goals": 8,
        "G": 8,
        "T": 5,
        "tau": 1.0,
        "K": 5,
        "max_calls": 3,
        "n_train": 322,
        "goal_seed": 0,
        "lr": 3e-6,
        "beta_kl": 0.02,
        "max_new": 160,
        "attacker_temperature": 1.0,
        "gen_chunk": 16,
        "workers": 16,
        "victim_temperature": 0.0,
        "ckpt_every": 30,
    },
    "allowed_variation": {
        "arm": ["dense", "sparse"],
        "seed": list(FORMAL_TRAINING_SEEDS),
        "metadata_or_path": ["tag", "run_root"],
    },
    "distributed": DISTRIBUTED_PROTOCOL,
    "victim_request": VICTIM_REQUEST,
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "optimizer": {
        "class": "torch.optim.AdamW",
        "kwargs": {
            "lr": 3e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
        },
    },
    "gradient_clipping": {
        "function": "torch.nn.utils.clip_grad_norm_",
        "max_norm": 1.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False,
        "foreach": None,
    },
    "runtime_profile": runtime_profile(),
    "runtime_profile_sha256": RUNTIME_PROFILE_SHA256,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sealed_payload_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("payload_sha256", None)
    return canonical_sha256(payload)


FORMAL_TRAINING_PROTOCOL_SHA256 = canonical_sha256(_FORMAL_TRAINING_PROTOCOL)


def formal_training_protocol() -> dict:
    return deepcopy(_FORMAL_TRAINING_PROTOCOL)


def build_global_goal_schedule(*, seed: int, steps: int, n_goals: int, n_train: int) -> list[list[int]]:
    """Counter-based schedule independent of arm, rank, process timing, and RNG libraries."""
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (steps, n_goals, n_train)):
        raise ValueError("steps, n_goals, and n_train must be positive integers")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    schedule: list[list[int]] = []
    for step in range(1, steps + 1):
        row = []
        for slot in range(n_goals):
            payload = f"h1-goal-schedule-v1|{seed}|{step}|{slot}".encode("utf-8")
            row.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_train)
        schedule.append(row)
    return schedule


def goal_schedule_sha256(schedule: list[list[int]]) -> str:
    return canonical_sha256(schedule)


def generation_call_seed(*, seed: int, step: int, rank: int, turn: int) -> int:
    """Stable per-call CUDA seed.  The arm is deliberately absent."""
    payload = f"h1-generation-v1|{seed}|{step}|{rank}|{turn}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def local_goal_slots(global_n_goals: int, rank: int, world_size: int = WORLD_SIZE) -> list[int]:
    if world_size != WORLD_SIZE or rank not in range(world_size):
        raise ValueError("formal training requires ranks 0/1 in world_size=2")
    slots = [slot for slot in range(global_n_goals) if slot % world_size == rank]
    if len(slots) * world_size != global_n_goals:
        raise ValueError("global goal count must divide evenly across workers")
    return slots


def _same_typed_value(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def validate_formal_training_values(config: Mapping[str, Any]) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("formal training config must be a mapping")
    expected_fixed = _FORMAL_TRAINING_PROTOCOL["fixed_training"]
    mismatched = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_fixed.items()
        if not _same_typed_value(config.get(key), expected)
    }
    if config.get("arm") not in ("dense", "sparse"):
        mismatched["arm"] = config.get("arm")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in FORMAL_TRAINING_SEEDS:
        mismatched["seed"] = seed
    if mismatched:
        raise ValueError(f"formal training protocol mismatch: {mismatched}")
    return formal_training_protocol()


def validate_formal_training_record(run_config: Mapping[str, Any]) -> dict:
    protocol = validate_formal_training_values(run_config)
    mismatched: dict[str, Any] = {}
    if run_config.get("training_protocol") != protocol:
        mismatched["training_protocol"] = "not canonical"
    if run_config.get("training_protocol_sha256") != FORMAL_TRAINING_PROTOCOL_SHA256:
        mismatched["training_protocol_sha256"] = run_config.get("training_protocol_sha256")
    if run_config.get("runtime_profile") != RUNTIME_PROFILE_ID:
        mismatched["runtime_profile"] = run_config.get("runtime_profile")
    if run_config.get("runtime_profile_sha256") != RUNTIME_PROFILE_SHA256:
        mismatched["runtime_profile_sha256"] = run_config.get("runtime_profile_sha256")
    if run_config.get("distributed") != DISTRIBUTED_PROTOCOL:
        mismatched["distributed"] = run_config.get("distributed")
    if run_config.get("run_kind") != "formal" or run_config.get("canonical_training_run") is not True:
        mismatched["run_kind"] = {
            "run_kind": run_config.get("run_kind"),
            "canonical_training_run": run_config.get("canonical_training_run"),
        }
    gate = run_config.get("gate1_spec")
    gate_artifacts = run_config.get("gate1_artifacts")
    try:
        if not isinstance(gate, Mapping) or not isinstance(gate_artifacts, Mapping):
            raise ValueError("Gate specification/artifact identity is missing")
        if set(gate_artifacts) != {"frozen_spec", "merge_manifest", "restart_proof"}:
            raise ValueError("Gate artifact set is not exact")
        frozen = gate_artifacts["frozen_spec"]
        merge = gate_artifacts["merge_manifest"]
        proof = gate_artifacts["restart_proof"]
        distributed_gate = gate.get("distributed_execution") or {}
        proof_reference = gate.get("runtime_proof") or {}
        hashes = (
            frozen.get("file_sha256"), frozen.get("canonical_sha256"),
            merge.get("file_sha256"), merge.get("payload_sha256"),
            proof.get("file_sha256"), proof.get("payload_sha256"),
        )
        if not all(isinstance(value, str) and len(value) == 64
                   and all(character in "0123456789abcdef" for character in value)
                   for value in hashes):
            raise ValueError("Gate artifact hashes are malformed")
        if (frozen["canonical_sha256"] != canonical_sha256(gate)
                or merge["file_sha256"]
                != distributed_gate.get("merge_manifest_file_sha256")
                or merge["payload_sha256"]
                != distributed_gate.get("merge_manifest_payload_sha256")
                or proof["file_sha256"] != proof_reference.get("file_sha256")
                or proof["payload_sha256"] != proof_reference.get("payload_sha256")):
            raise ValueError("Gate raw/merge/proof identity is inconsistent")
    except (KeyError, TypeError, ValueError) as exc:
        mismatched["gate1_artifacts"] = str(exc)
    manifests = run_config.get("runtime_manifests")
    if not isinstance(manifests, list) or len(manifests) != WORLD_SIZE:
        mismatched["runtime_manifests"] = "requires exactly two worker manifests"
    else:
        try:
            checked = [validate_runtime_manifest(item, require_role="worker") for item in manifests]
            if [item["rank"] for item in checked] != [0, 1]:
                raise ValueError("worker manifests must be ordered ranks [0,1]")
            if checked[0]["inventory"] != checked[1]["inventory"]:
                raise ValueError("worker manifests disagree on physical inventory")
        except ValueError as exc:
            mismatched["runtime_manifests"] = str(exc)
    victim_manifests = run_config.get("victim_manifests")
    victim_pair = run_config.get("victim_pair_manifest")
    if (not isinstance(victim_manifests, list) or len(victim_manifests) != WORLD_SIZE
            or not isinstance(victim_pair, Mapping)):
        mismatched["victim_pair_manifest"] = "requires two replicas and one aggregate pair"
    else:
        try:
            if victim_pair.get("payload_sha256") != sealed_payload_sha256(victim_pair):
                raise ValueError("aggregate pair payload seal mismatch")
            if run_config.get("victim_pair_manifest_sha256") != victim_pair["payload_sha256"]:
                raise ValueError("run config does not bind the aggregate pair payload hash")
            refs = victim_pair.get("replicas")
            if not isinstance(refs, list) or len(refs) != WORLD_SIZE:
                raise ValueError("aggregate pair must reference exactly two replicas")
            if [ref.get("rank") for ref in refs] != [0, 1]:
                raise ValueError("aggregate pair replica refs must be ordered ranks [0,1]")
            for rank, (ref, replica) in enumerate(zip(refs, victim_manifests)):
                if not isinstance(replica, Mapping):
                    raise ValueError(f"victim replica {rank} is not a mapping")
                if replica.get("payload_sha256") != sealed_payload_sha256(replica):
                    raise ValueError(f"victim replica {rank} payload seal mismatch")
                if (replica.get("rank") != rank
                        or ref.get("rank") != rank
                        or ref.get("port") != WORKER_VICTIM_PORTS[rank]
                        or ref.get("path") != replica.get("path")
                        or ref.get("payload_sha256") != replica.get("payload_sha256")):
                    raise ValueError(f"aggregate pair ref does not bind victim replica {rank}")
                checked_runtime = validate_runtime_manifest(
                    replica.get("runtime_manifest"), require_role="worker"
                )
                if checked_runtime["rank"] != rank:
                    raise ValueError(f"victim replica {rank} runtime rank mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            mismatched["victim_pair_manifest"] = str(exc)
    expected_schedule = build_global_goal_schedule(
        seed=run_config.get("seed"),
        steps=run_config.get("steps"),
        n_goals=run_config.get("n_goals"),
        n_train=run_config.get("n_train"),
    )
    if run_config.get("global_goal_schedule") != expected_schedule:
        mismatched["global_goal_schedule"] = "not the canonical arm-independent schedule"
    if run_config.get("global_goal_schedule_sha256") != goal_schedule_sha256(expected_schedule):
        mismatched["global_goal_schedule_sha256"] = run_config.get("global_goal_schedule_sha256")
    try:
        validate_software_versions(run_config.get("software_versions"))
    except ValueError as exc:
        mismatched["software_versions"] = str(exc)
    if mismatched:
        raise ValueError(f"formal training protocol record mismatch: {mismatched}")
    return protocol
