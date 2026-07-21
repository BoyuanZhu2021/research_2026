#!/usr/bin/env python3
"""Fail-closed offline audit for one recovered single-H20 formal Gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
for entry in (CODE, CODE / "src", CODE / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import h1_deploy_mt as deploy  # noqa: E402
import h1_tooluse_gate1_local as gate  # noqa: E402
from src.domains.tooluse_injection import ToolUseInjectionDomain  # noqa: E402
from src.tooluse_gate1_spec import (  # noqa: E402
    LEGACY_H20_PROFILE_ID,
    load_frozen_gate1,
    sha256_file,
)
from src.victim_decision_protocol import canonical_sha256  # noqa: E402


TIERS = ("none", "light", "moderate")
EXPECTED_FILES = {
    "run_manifest.json",
    "gate1_summary.json",
    "frozen_gate1.json",
    *{
        f"tier-{tier}/{name}"
        for tier in TIERS
        for name in (
            "tier_manifest.json",
            "tier_summary.json",
            "episodes.jsonl",
            "llm_calls.jsonl",
        )
    },
}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line in {path.name}:{line_number}")
        value = json.loads(line, object_pairs_hook=_strict_object)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row in {path.name}:{line_number}")
        rows.append(value)
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_calls(run: dict, records: list[dict], calls: list[dict]) -> Counter:
    expected_attacker = {}
    expected_victim = {}
    for record in records:
        goal_id = record["goal_id"]
        episode_index = record["episode_index"]
        for turn in record["turns"]:
            attacker_key = (goal_id, turn["turn"])
            _require(attacker_key not in expected_attacker, "duplicate attacker trace key")
            expected_attacker[attacker_key] = (episode_index, turn)
            for step in turn["victim_steps"]:
                victim_key = (goal_id, turn["turn"], step["step"])
                _require(victim_key not in expected_victim, "duplicate victim trace key")
                expected_victim[victim_key] = (episode_index, step)

    observed_attacker = set()
    observed_victim = set()
    statuses = Counter()
    for call in calls:
        role = call.get("role")
        status = call.get("status")
        statuses[(role, status)] += 1
        _require(status == "ok", "LLM ledger contains a failed call")
        context = call.get("context")
        request = call.get("request")
        _require(isinstance(context, dict) and isinstance(request, dict), "malformed LLM ledger row")
        if role == "attacker":
            key = (context.get("goal_id"), context.get("turn"))
            _require(key in expected_attacker and key not in observed_attacker,
                     "attacker ledger key mismatch")
            episode_index, turn = expected_attacker[key]
            _require(context.get("episode_index") == episode_index,
                     "attacker episode identity mismatch")
            _require(request.get("messages") == turn["attacker_prompt_messages"],
                     "attacker prompt mismatch")
            _require(call.get("output") == turn["attacker_output"],
                     "attacker output mismatch")
            _require(request.get("output_protocol") == run["config"]["attacker_output_protocol"],
                     "attacker output protocol identity mismatch")
            observed_attacker.add(key)
        elif role == "victim":
            key = (context.get("goal_id"), context.get("turn"), context.get("step"))
            _require(key in expected_victim and key not in observed_victim,
                     "victim ledger key mismatch")
            episode_index, step = expected_victim[key]
            _require(context.get("episode_index") == episode_index,
                     "victim episode identity mismatch")
            _require(request.get("messages") == step["messages"], "victim prompt mismatch")
            _require(call.get("output") == step["output"], "victim raw output mismatch")
            _require(request.get("structured_schema_sha256") == step["structured_schema_sha256"],
                     "victim schema hash/episode mismatch")
            structured = request.get("structured_outputs")
            _require(
                isinstance(structured, dict)
                and canonical_sha256(structured.get("json")) == step["structured_schema_sha256"],
                "victim structured schema bytes/hash mismatch",
            )
            _require(request.get("output_protocol") == run["config"]["victim_output_protocol"],
                     "victim output protocol identity mismatch")
            observed_victim.add(key)
        else:
            raise ValueError(f"unexpected LLM role: {role!r}")

    _require(observed_attacker == set(expected_attacker), "attacker ledger is incomplete")
    _require(observed_victim == set(expected_victim), "victim ledger is incomplete")
    return statuses


def audit(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    recovery_root = (ROOT / "artifacts" / "h20-formal-gates").resolve()
    _require(run_dir.parent.parent == recovery_root,
             "run must be under one artifacts/h20-formal-gates controller directory")
    _require(run_dir.parent.name.startswith("h1-formal-gate-"), "unexpected controller id")
    _require(run_dir.name.startswith("tooluse-gate1-formal-"), "unexpected formal run id")
    files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*") if path.is_file()
    }
    _require(files == EXPECTED_FILES, f"artifact file set mismatch: {sorted(files)}")

    run = _load_json(run_dir / "run_manifest.json")
    gate_summary = _load_json(run_dir / "gate1_summary.json")
    frozen_path = run_dir / "frozen_gate1.json"
    frozen = _load_json(frozen_path)
    run_id = run_dir.name
    _require(run.get("run_id") == run_id, "run id mismatch")
    _require((run.get("mode"), run.get("status"), run.get("verdict"))
             == ("formal", "completed", "PASS"), "run terminal state mismatch")
    _require(run.get("siliconflow_fallback") is False, "fallback was not disabled")
    _require(isinstance(run.get("h20_runtime_provenance"), dict),
             "formal runtime provenance missing")

    _plan, local_deployment = deploy._prepare_local_release()
    _require(run.get("deployment_manifest") == local_deployment,
             "recovered run does not bind the current registered deployment bytes")

    split_path = CODE / "configs" / "injecagent_ds_base_split_v1.json"
    manifest, goal_ids, goals_by_tier = gate._load_calibration(split_path)
    active_goal_ids = run["calibration"]["active_goal_ids"]
    _require(active_goal_ids == goal_ids and len(goal_ids) == 69,
             "formal calibration goal order/denominator drift")
    _require(run["calibration"]["formal_count"] == 69, "formal denominator drift")
    _require(run["split_manifest"]["file_sha256"] == sha256_file(split_path),
             "split hash drift")
    _require(run["split_manifest"]["manifest_id"] == manifest["manifest_id"],
             "split id drift")

    summaries = []
    role_statuses = Counter()
    for tier in TIERS:
        tier_dir = run_dir / f"tier-{tier}"
        tier_manifest = _load_json(tier_dir / "tier_manifest.json")
        tier_summary = _load_json(tier_dir / "tier_summary.json")
        records = _load_jsonl(tier_dir / "episodes.jsonl")
        calls = _load_jsonl(tier_dir / "llm_calls.jsonl")
        _require(tier_manifest.get("run_id") == run_id, "tier/run identity mismatch")
        _require(tier_manifest.get("goal_ids") == goal_ids, "tier goal order mismatch")
        _require(len(records) == len(goal_ids) == 69, "episode denominator mismatch")
        goals = {goal.id: goal for goal in goals_by_tier[tier]}
        for index, (goal_id, record) in enumerate(zip(goal_ids, records)):
            _require(record.get("episode_index") == index, "episode index mismatch")
            _require(record.get("goal_id") == goal_id, "episode goal order mismatch")
            domain = ToolUseInjectionDomain(attack=gate.ATTACK, defense_tier=tier)
            gate.recompute_episode_record(domain, goals[goal_id], tier, record)
        recomputed = json.loads(json.dumps(gate.summarize_tier(tier, records, expected=69)))
        _require(recomputed == tier_summary, "tier summary differs after recomputation")
        summaries.append(recomputed)
        role_statuses.update(_verify_calls(run, records, calls))

    _require(gate_summary.get("tiers") == summaries,
             "Gate summary tiers differ after recomputation")
    chosen = gate.select_tier(summaries)
    _require(chosen is not None and chosen["passes"], "recomputed Gate is not PASS")
    _require(
        gate_summary.get("verdict") == "PASS"
        and gate_summary.get("chosen_tier") == chosen["tier"]
        and gate_summary.get("decision_bearing") is True
        and gate_summary.get("frozen_spec_written") is True,
        "Gate summary verdict/selection mismatch",
    )
    validated_frozen = load_frozen_gate1(
        frozen_path,
        require_pass=True,
        code_root=CODE,
        verify_external_artifacts=False,
        expected_profile_id=LEGACY_H20_PROFILE_ID,
    )
    _require(validated_frozen == frozen, "frozen Gate changed during validation")
    _require(frozen.get("chosen_tier") == chosen["tier"], "frozen chosen tier mismatch")

    controller_root = run_dir.parent
    hashes = {
        path.relative_to(controller_root).as_posix(): sha256_file(path)
        for path in sorted(controller_root.rglob("*")) if path.is_file()
    }
    registry_payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return {
        "status": "OFFLINE_FORMAL_GATE_AUDIT_OK",
        "controller_id": controller_root.name,
        "run_id": run_id,
        "deployment_tree_sha256": local_deployment["deployed_tree_sha256"],
        "episodes_recomputed": 69 * len(TIERS),
        "invalid_episodes": sum(summary["n_invalid"] for summary in summaries),
        "chosen_tier": chosen["tier"],
        "tiers": summaries,
        "llm_call_statuses": {
            f"{role}:{status}": count
            for (role, status), count in sorted(role_statuses.items())
        },
        "frozen_gate_file_sha256": sha256_file(frozen_path),
        "frozen_gate_canonical_sha256": canonical_sha256(frozen),
        "recovered_file_count": len(hashes),
        "hash_registry_sha256": hashlib.sha256(registry_payload.encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
