"""Fail-closed artifacts for the formal single-H20 H1 evaluation campaign.

This module is deliberately CPU-only.  Runtime, Gate, deployment, and training
owners validate their own records; this module binds those validated records to
one immutable evaluation identity and to an exact seven-run registry.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .h20_serving_identity import (
    validate_fresh_h20_campaign_runtime_proof as validate_fresh_runtime_envelope,
    validate_h20_formal_runtime_bundle,
    validate_live_runtime_check,
    validate_runtime_reference,
)
from .h20_training_protocol import (
    DATA_IDENTITY,
    MODEL_IDENTITY,
    ORACLE_AND_INTERACTION,
    canonical_sha256,
    sealed_payload_sha256,
)
from .model_pins import INJECAGENT_COMMIT
from .runtime_profile import H20_RUNTIME_PROFILE_SHA256, LEGACY_H20_PROFILE_ID


SCHEMA_VERSION = 1
ARTIFACT_KIND = "h1_h20_tooluse_eval_artifact"
CAMPAIGN_KIND = "h1_h20_tooluse_eval_campaign"
FINAL_OOD_AUTHORIZATION_KIND = "h1_h20_final_ood_explicit_authorization"
LEARNING_REPORT_KIND = "h1_h20_learning_gate_review_report"
LEARNING_GATE_PHASE = "learning_gate"
FINAL_OOD_PHASE = "final_ood"
PHASE_SPLIT = {LEARNING_GATE_PHASE: "calibration", FINAL_OOD_PHASE: "final_ood"}
DOMAIN_SPLIT = {LEARNING_GATE_PHASE: "calibration", FINAL_OOD_PHASE: "ood"}
PHASE_GOALS = {LEARNING_GATE_PHASE: 69, FINAL_OOD_PHASE: 153}
BASE_K = 4
TRAINING_SEEDS = (0, 1, 2)
ARMS = ("base", "dense", "sparse")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

EVALUATION_PROTOCOL = {
    "domain": "tooluse",
    "interaction": "multi_turn_fresh_attempts",
    "T": 5,
    "tau": 1.0,
    "max_calls": 3,
    "max_new": 256,
    "attacker_temperature": 0.7,
    "victim_temperature": 0.0,
    "victim_max_tokens": ORACLE_AND_INTERACTION["victim_max_tokens"],
    "victim_stop": list(ORACLE_AND_INTERACTION["victim_generation_stop"]),
    "victim_include_stop_str_in_output": ORACLE_AND_INTERACTION[
        "victim_include_stop_str_in_output"
    ],
    "victim_output_protocol": copy.deepcopy(
        ORACLE_AND_INTERACTION["victim_output_protocol"]
    ),
    "attacker_output_protocol": copy.deepcopy(
        ORACLE_AND_INTERACTION["attacker_output_protocol"]
    ),
    "gen_chunk": 16,
    "workers": 16,
    "run_seed": 0,
    "goal_seed": 0,
    "base_k": BASE_K,
    "training_seeds": list(TRAINING_SEEDS),
    "eval_seed_panel": {
        "base": [0, 1, 2, 3],
        "trained": "eval_seed_equals_training_seed",
        "rng_key": ["phase", "eval_seed", "turn"],
        "arm_in_rng_key": False,
        "published_row_order": "goal_major_then_seed_index",
    },
    "victim_url": "http://127.0.0.1:8000/v1",
    "victim_response_model_must_match": MODEL_IDENTITY["victim"]["served_model"],
    "siliconflow_fallback": False,
    "v100_artifacts_accepted": False,
}

_IDENTITY_KEYS = {
    "campaign_id", "phase", "split", "domain_split", "n_goals", "goal_ids",
    "goal_ids_sha256", "decision_bearing", "profile_id", "profile_sha256",
    "gate_bundle", "gate_bundle_payload_sha256", "eval_quantization_check",
    "eval_runtime_proof", "eval_runtime_bundle", "eval_runtime_reference",
    "deployment", "deployment_sha256", "models", "data",
    "oracle_and_interaction", "evaluation", "learning_campaign_manifest",
    "learning_report", "final_ood_authorization", "created_at",
}
_CAMPAIGN_KEYS = {
    "schema_version", "kind", "status", "identity", "identity_sha256",
    "provenance_files", "artifacts", "payload_sha256",
}
_PROVENANCE_FILE_KEYS = {
    "label", "identity_pointer", "source_path", "file", "file_sha256",
}
_SNAPSHOT_KEYS = {"root", "files", "tree_sha256"}
_SNAPSHOT_FILE_KEYS = {"file", "sha256"}
_REGISTRY_KEYS = {
    "arm", "training_seed", "seeds", "tag", "n_goals", "n_rows",
    "summary_file", "summary_sha256", "rows_file", "rows_sha256",
    "artifact_manifest_file", "artifact_manifest_file_sha256",
    "artifact_manifest_payload_sha256", "adapter_sha256", "adapter_tree_sha256",
    "adapter_provenance_sha256", "training_snapshot",
}
_ARTIFACT_KEYS = {
    "schema_version", "kind", "status", "profile_id", "profile_sha256", "phase",
    "split", "campaign_identity_sha256", "arm", "tag", "training_seed",
    "adapter_sha256", "adapter_tree_sha256", "adapter_provenance",
    "adapter_provenance_sha256", "goal_count", "goal_ids_sha256", "seeds",
    "row_count", "rows_sha256", "runtime_reference", "runtime_open_check",
    "runtime_close_check", "runtime_reference_sha256", "payload_sha256",
}
_AUTHORIZATION_KEYS = {
    "schema_version", "kind", "status", "scope", "final_campaign_id",
    "learning_campaign_id", "learning_campaign_identity_sha256",
    "learning_campaign_manifest_sha256", "learning_metrics_decision_bearing",
    "learning_report_payload_sha256", "learning_report_sha256",
    "criterion", "approval_reference", "authorized_by", "authorized_at",
    "payload_sha256",
}
_LEARNING_REPORT_KEYS = {
    "schema_version", "kind", "status", "campaign_id",
    "campaign_identity_sha256", "campaign_manifest_sha256", "phase",
    "n_goals", "decision_bearing", "purpose", "implicit_pass_threshold",
    "h1_verdict", "arm_metrics", "analyzed_at", "payload_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid H20 eval artifact: {message}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _safe_relative_file(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value and "\\" not in value,
             f"unsafe {label}")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
             f"unsafe {label}")
    return value


def seal(document: Mapping[str, Any]) -> dict:
    result = copy.deepcopy(dict(document))
    result.pop("payload_sha256", None)
    result["payload_sha256"] = canonical_sha256(result)
    return result


def expected_grid(phase: str) -> tuple[tuple[str, int | None, int], ...]:
    """The only allowed arm/training-seed/evaluation-seed grid."""
    _require(phase in PHASE_SPLIT, f"unknown phase {phase!r}")
    return (
        ("base", None, BASE_K),
        *(("dense", seed, 1) for seed in TRAINING_SEEDS),
        *(("sparse", seed, 1) for seed in TRAINING_SEEDS),
    )


def canonical_tag(arm: str, training_seed: int | None) -> str:
    if arm == "base" and training_seed is None:
        return "base"
    _require(arm in {"dense", "sparse"} and training_seed in TRAINING_SEEDS,
             "cannot construct tag for invalid arm/training seed")
    return f"{arm}-s{training_seed}"


def expected_row_keys(goal_ids: Sequence[str], seeds: int) -> list[tuple[int, str, int]]:
    return [
        (goal_index, goal_id, seed_index)
        for goal_index, goal_id in enumerate(goal_ids)
        for seed_index in range(seeds)
    ]


def snapshot_tree_sha256(root: str, files: Sequence[Mapping[str, Any]]) -> str:
    """Hash a portable file registry using the training tree-hash construction."""
    root_prefix = PurePosixPath(root)
    records: list[tuple[str, str]] = []
    for record in files:
        _require(isinstance(record, Mapping) and set(record) == _SNAPSHOT_FILE_KEYS,
                 "snapshot file registry shape mismatch")
        name = _safe_relative_file(record.get("file"), "snapshot file")
        path = PurePosixPath(name)
        try:
            relative = path.relative_to(root_prefix).as_posix()
        except ValueError as exc:
            raise ValueError("invalid H20 eval artifact: snapshot file escapes root") from exc
        _require(relative and relative != ".", "snapshot file cannot equal root")
        digest = record.get("sha256")
        _require(_is_sha256(digest), "snapshot file hash is malformed")
        records.append((relative, digest))
    _require(records and records == sorted(records) and len(records) == len(set(records)),
             "snapshot files must be non-empty, unique, and sorted")
    import hashlib
    digest = hashlib.sha256()
    for relative, file_hash in records:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(file_hash.encode("ascii") + b"\n")
    return digest.hexdigest()


def validate_training_snapshot(snapshot: Any, *, tag: str) -> dict:
    _require(isinstance(snapshot, Mapping), "trained arm lacks portable training snapshot")
    result = copy.deepcopy(dict(snapshot))
    _require(set(result) == _SNAPSHOT_KEYS, "training snapshot field set mismatch")
    root = _safe_relative_file(result.get("root"), "training snapshot root")
    _require(root == f"provenance/training/{tag}", "training snapshot root is not canonical")
    files = result.get("files")
    _require(isinstance(files, list), "training snapshot file registry missing")
    _require(result.get("tree_sha256") == snapshot_tree_sha256(root, files),
             "training snapshot registry tree hash mismatch")
    return result


def validate_runtime_reference_shape(reference: Any) -> dict:
    _require(isinstance(reference, Mapping), "runtime reference must be an object")
    result = validate_runtime_reference(reference)
    _require(result.get("profile_id") == LEGACY_H20_PROFILE_ID,
             "runtime reference is not the single-H20 profile")
    _require(result.get("profile_sha256") == H20_RUNTIME_PROFILE_SHA256,
             "runtime profile hash mismatch")
    return result


def _validate_gate_bundle(bundle: Any) -> dict:
    # Imported lazily so --help and CPU artifact utilities do not import a GPU runner.
    from .tooluse_gate1_spec import validate_h20_gate_artifact_bundle
    checked = validate_h20_gate_artifact_bundle(
        bundle, verify_external_artifacts=False
    )
    _require(isinstance(checked, Mapping) and _is_sha256(checked.get("payload_sha256")),
             "Gate bundle is not a sealed portable H20 artifact")
    identity = checked.get("gate_identity") or {}
    frozen = checked.get("frozen_gate") or {}
    _require(identity.get("verdict") == "PASS"
             and frozen.get("verdict") == "PASS"
             and frozen.get("passed") is True,
             "Gate bundle is not jointly verdict=PASS and passed=true")
    return copy.deepcopy(dict(checked))


def validate_fresh_eval_runtime_proof(
    proof: Any, *, gate_bundle: Mapping[str, Any],
    fresh_quantization_check: Mapping[str, Any],
) -> dict:
    """Validate the sealed new-lifecycle proof without reopening raw files.

    The evaluator first calls the runtime owner's external validator, which reopens
    all four fresh FP8 repeatability-cycle files. Campaign/analyzer validation then uses this
    pure binding; the analyzer separately re-hashes the portable copies.
    """
    gate = _validate_gate_bundle(gate_bundle)
    gate_check = gate["frozen_gate"]["quantization_check"]
    result = validate_fresh_runtime_envelope(
        proof, gate_check, fresh_quantization_check
    )
    fresh_runtime = result["fresh_runtime_bundle"]
    _require(not fresh_runtime["gate_checks"],
             "fresh eval runtime bundle must precede eval and have no Gate checks")
    old_process = gate["runtime_bundle"]["restored_fp8_runtime"]["process"]
    new_process = fresh_runtime["restored_fp8_runtime"]["process"]
    _require((old_process["pid"], old_process["start_time_ticks"])
             != (new_process["pid"], new_process["start_time_ticks"]),
             "fresh eval proof reused the immutable Gate process lifecycle")
    _require(result.get("deployment") == fresh_quantization_check.get("deployment")
             and result.get("case_set") == fresh_quantization_check.get("case_set")
             and result.get("config") == fresh_quantization_check.get("config")
             and result.get("oracle_version") == fresh_quantization_check.get("oracle_version"),
             "fresh eval runtime proof invariant summary mismatch")
    behavior = result.get("behavior_signature")
    _require(isinstance(behavior, Mapping) and _is_sha256(behavior.get("sha256")),
             "fresh eval behavior signature is malformed")
    return result


def validate_campaign_identity(identity: Any) -> dict:
    _require(isinstance(identity, Mapping), "campaign identity missing")
    result = copy.deepcopy(dict(identity))
    _require(set(result) == _IDENTITY_KEYS, "campaign identity field set mismatch")
    campaign_id = result.get("campaign_id")
    _require(isinstance(campaign_id, str) and SAFE_ID.fullmatch(campaign_id) is not None,
             "campaign ID is unsafe")
    phase = result.get("phase")
    _require(phase in PHASE_SPLIT and result.get("split") == PHASE_SPLIT[phase]
             and result.get("domain_split") == DOMAIN_SPLIT[phase]
             and result.get("n_goals") == PHASE_GOALS[phase],
             "campaign phase/split/denominator mismatch")
    goals = result.get("goal_ids")
    _require(isinstance(goals, list) and len(goals) == PHASE_GOALS[phase]
             and len(set(goals)) == len(goals)
             and all(isinstance(item, str) and item for item in goals)
             and result.get("goal_ids_sha256") == canonical_sha256(goals),
             "campaign goal manifest/order mismatch")
    _require(result.get("decision_bearing") is (phase == FINAL_OOD_PHASE),
             "campaign decision-bearing flag mismatch")
    _require(result.get("profile_id") == LEGACY_H20_PROFILE_ID
             and result.get("profile_sha256") == H20_RUNTIME_PROFILE_SHA256,
             "campaign is not the single-H20 profile")

    gate = _validate_gate_bundle(result.get("gate_bundle"))
    _require(result.get("gate_bundle_payload_sha256") == gate["payload_sha256"],
             "campaign Gate bundle hash mismatch")
    runtime_proof = validate_fresh_eval_runtime_proof(
        result.get("eval_runtime_proof"), gate_bundle=gate,
        fresh_quantization_check=result.get("eval_quantization_check"),
    )
    runtime = validate_h20_formal_runtime_bundle(
        result.get("eval_runtime_bundle"), require_gate_checks=False
    )
    reference = validate_runtime_reference_shape(result.get("eval_runtime_reference"))
    _require(runtime == runtime_proof["fresh_runtime_bundle"]
             and reference == runtime["restored_fp8_runtime"],
             "campaign eval runtime differs from fresh lifecycle proof")

    deployment = result.get("deployment")
    _require(isinstance(deployment, Mapping)
             and deployment.get("injecagent_commit") == INJECAGENT_COMMIT
             and deployment.get("injecagent_clean") is True
             and _is_sha256(deployment.get("deployed_tree_sha256"))
             and result.get("deployment_sha256") == canonical_sha256(deployment),
             "campaign deployment identity mismatch")
    proof_deployment = runtime_proof.get("deployment") or {}
    _require(proof_deployment.get("deployed_tree_sha256")
             == deployment.get("deployed_tree_sha256")
             and proof_deployment.get("injecagent_commit")
             == deployment.get("injecagent_commit"),
             "fresh eval runtime proof deployment differs from campaign tree")
    _require(result.get("models") == MODEL_IDENTITY
             and result.get("data") == DATA_IDENTITY
             and result.get("oracle_and_interaction") == ORACLE_AND_INTERACTION
             and result.get("evaluation") == EVALUATION_PROTOCOL,
             "campaign frozen model/data/oracle/evaluation protocol mismatch")
    _require(isinstance(result.get("created_at"), str) and result["created_at"],
             "campaign creation timestamp missing")

    if phase == LEARNING_GATE_PHASE:
        _require(result.get("learning_campaign_manifest") is None
                 and result.get("learning_report") is None
                 and result.get("final_ood_authorization") is None,
                 "learning campaign cannot carry final-OOD authorization")
    else:
        learning = validate_campaign_registry(result.get("learning_campaign_manifest"))
        learning_identity = learning["identity"]
        _require(learning_identity.get("gate_bundle_payload_sha256")
                 == gate["payload_sha256"]
                 and learning_identity.get("deployment_sha256")
                 == result.get("deployment_sha256"),
                 "final campaign Gate/deployment differs from learning campaign")
        report = validate_learning_report(result.get("learning_report"), learning)
        authorization = validate_final_ood_authorization(
            result.get("final_ood_authorization"), learning, report,
            final_campaign_id=campaign_id,
        )
        _require(authorization["final_campaign_id"] == campaign_id,
                 "authorization does not unlock this exact final campaign")
    return result


def build_campaign_identity(
    *, campaign_id: str, phase: str, goal_ids: Sequence[str], gate_bundle: Mapping[str, Any],
    eval_quantization_check: Mapping[str, Any], eval_runtime_proof: Mapping[str, Any],
    deployment: Mapping[str, Any], created_at: str,
    learning_campaign_manifest: Mapping[str, Any] | None = None,
    learning_report: Mapping[str, Any] | None = None,
    final_ood_authorization: Mapping[str, Any] | None = None,
) -> dict:
    gate = _validate_gate_bundle(gate_bundle)
    runtime_proof = validate_fresh_eval_runtime_proof(
        eval_runtime_proof, gate_bundle=gate,
        fresh_quantization_check=eval_quantization_check,
    )
    runtime = runtime_proof["fresh_runtime_bundle"]
    identity = {
        "campaign_id": campaign_id,
        "phase": phase,
        "split": PHASE_SPLIT.get(phase),
        "domain_split": DOMAIN_SPLIT.get(phase),
        "n_goals": len(goal_ids),
        "goal_ids": list(goal_ids),
        "goal_ids_sha256": canonical_sha256(list(goal_ids)),
        "decision_bearing": phase == FINAL_OOD_PHASE,
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "gate_bundle": gate,
        "gate_bundle_payload_sha256": gate["payload_sha256"],
        "eval_quantization_check": copy.deepcopy(dict(eval_quantization_check)),
        "eval_runtime_proof": runtime_proof,
        "eval_runtime_bundle": copy.deepcopy(runtime),
        "eval_runtime_reference": copy.deepcopy(runtime["restored_fp8_runtime"]),
        "deployment": copy.deepcopy(dict(deployment)),
        "deployment_sha256": canonical_sha256(deployment),
        "models": copy.deepcopy(MODEL_IDENTITY),
        "data": copy.deepcopy(DATA_IDENTITY),
        "oracle_and_interaction": copy.deepcopy(ORACLE_AND_INTERACTION),
        "evaluation": copy.deepcopy(EVALUATION_PROTOCOL),
        "learning_campaign_manifest": (
            copy.deepcopy(dict(learning_campaign_manifest))
            if learning_campaign_manifest is not None else None
        ),
        "learning_report": (
            copy.deepcopy(dict(learning_report))
            if learning_report is not None else None
        ),
        "final_ood_authorization": (
            copy.deepcopy(dict(final_ood_authorization))
            if final_ood_authorization is not None else None
        ),
        "created_at": created_at,
    }
    return validate_campaign_identity(identity)


def validate_provenance_files(files: Any) -> list[dict]:
    _require(isinstance(files, list), "campaign provenance file registry missing")
    result: list[dict] = []
    seen: set[str] = set()
    for raw in files:
        _require(isinstance(raw, Mapping), "provenance file entry is not an object")
        item = copy.deepcopy(dict(raw))
        _require(set(item) == _PROVENANCE_FILE_KEYS,
                 "provenance file entry field set mismatch")
        _require(isinstance(item.get("label"), str) and item["label"],
                 "provenance file label missing")
        _require(item.get("identity_pointer") is None
                 or (isinstance(item["identity_pointer"], str)
                     and item["identity_pointer"].startswith("/")),
                 "provenance identity pointer malformed")
        _require(isinstance(item.get("source_path"), str) and item["source_path"],
                 "provenance source path missing")
        name = _safe_relative_file(item.get("file"), "provenance file")
        _require(name.startswith("provenance/"), "provenance file is outside provenance/")
        _require(name not in seen, "duplicate registered provenance file")
        seen.add(name)
        _require(_is_sha256(item.get("file_sha256")), "provenance file hash malformed")
        result.append(item)
    _require(result == sorted(result, key=lambda item: item["file"]),
             "provenance file registry must be sorted")
    return result


def validate_registry_entry(entry: Any, *, phase: str) -> dict:
    _require(isinstance(entry, Mapping), "campaign registry entry is not an object")
    item = copy.deepcopy(dict(entry))
    _require(set(item) == _REGISTRY_KEYS, "campaign registry entry field set mismatch")
    arm, training_seed, seeds = (
        item.get("arm"), item.get("training_seed"), item.get("seeds")
    )
    _require((arm, training_seed, seeds) in expected_grid(phase),
             "registry arm/training/eval seed tuple is invalid")
    tag = canonical_tag(arm, training_seed)
    _require(item.get("tag") == tag, "registry tag is not canonical")
    _require(item.get("n_goals") == PHASE_GOALS[phase]
             and item.get("n_rows") == PHASE_GOALS[phase] * seeds,
             "registry denominator mismatch")
    for field in ("summary_file", "rows_file", "artifact_manifest_file"):
        _safe_relative_file(item.get(field), field)
    for field in (
        "summary_sha256", "rows_sha256", "artifact_manifest_file_sha256",
        "artifact_manifest_payload_sha256", "adapter_provenance_sha256",
    ):
        _require(_is_sha256(item.get(field)), f"malformed registry hash {field}")
    if arm == "base":
        _require(item.get("adapter_sha256") == "base"
                 and item.get("adapter_tree_sha256") == "base"
                 and item.get("training_snapshot") is None,
                 "base registry must use base sentinels and no training snapshot")
    else:
        _require(_is_sha256(item.get("adapter_sha256"))
                 and _is_sha256(item.get("adapter_tree_sha256")),
                 "trained adapter hashes are malformed")
        validate_training_snapshot(item.get("training_snapshot"), tag=tag)
    return item


def validate_campaign_manifest(manifest: Any, *, require_complete: bool = False) -> dict:
    _require(isinstance(manifest, Mapping), "campaign manifest must be an object")
    result = copy.deepcopy(dict(manifest))
    _require(set(result) == _CAMPAIGN_KEYS, "campaign manifest field set mismatch")
    _require(result.get("payload_sha256") == sealed_payload_sha256(result),
             "campaign payload seal mismatch")
    _require(result.get("schema_version") == SCHEMA_VERSION
             and result.get("kind") == CAMPAIGN_KIND
             and result.get("status") in {"collecting", "complete"},
             "campaign schema/kind/status mismatch")
    identity = validate_campaign_identity(result.get("identity"))
    _require(result.get("identity_sha256") == canonical_sha256(identity),
             "campaign identity seal mismatch")
    validate_provenance_files(result.get("provenance_files"))
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, list), "campaign artifact registry missing")
    checked = [validate_registry_entry(item, phase=identity["phase"]) for item in artifacts]
    actual = [(item["arm"], item["training_seed"], item["seeds"]) for item in checked]
    grid = list(expected_grid(identity["phase"]))
    _require(actual == grid[:len(actual)],
             "campaign registry is not the canonical arm/seed prefix")
    files = [
        item[field]
        for item in checked
        for field in ("summary_file", "rows_file", "artifact_manifest_file")
    ]
    _require(len(files) == len(set(files)), "duplicate registered evaluation file")
    complete = len(checked) == len(grid)
    _require(result["status"] == ("complete" if complete else "collecting"),
             "campaign completion status disagrees with registry")
    if require_complete:
        _require(complete, "campaign registry is incomplete")
    return result


def validate_campaign_registry(manifest: Any) -> dict:
    return validate_campaign_manifest(manifest, require_complete=True)


def build_campaign_manifest(
    identity: Mapping[str, Any], provenance_files: Sequence[Mapping[str, Any]],
) -> dict:
    checked_identity = validate_campaign_identity(identity)
    checked_files = validate_provenance_files(list(provenance_files))
    return validate_campaign_manifest(seal({
        "schema_version": SCHEMA_VERSION,
        "kind": CAMPAIGN_KIND,
        "status": "collecting",
        "identity": checked_identity,
        "identity_sha256": canonical_sha256(checked_identity),
        "provenance_files": checked_files,
        "artifacts": [],
    }))


def register_campaign_artifact(manifest: Mapping[str, Any], entry: Mapping[str, Any]) -> dict:
    current = validate_campaign_manifest(manifest)
    _require(current["status"] == "collecting", "campaign is already complete")
    checked = validate_registry_entry(entry, phase=current["identity"]["phase"])
    grid = list(expected_grid(current["identity"]["phase"]))
    position = len(current["artifacts"])
    _require((checked["arm"], checked["training_seed"], checked["seeds"])
             == grid[position], "artifact is not the next canonical campaign run")
    current["artifacts"].append(checked)
    if len(current["artifacts"]) == len(grid):
        current["status"] = "complete"
    return validate_campaign_manifest(seal(current))


def validate_eval_artifact(
    document: Any, *, campaign_identity_sha256: str, phase: str, arm: str,
    training_seed: int | None, tag: str, adapter_sha256: str,
    adapter_tree_sha256: str, adapter_provenance: Mapping[str, Any],
    goal_ids: Sequence[str], seeds: int, rows: Sequence[Mapping[str, Any]],
) -> dict:
    _require(isinstance(document, Mapping), "artifact must be an object")
    artifact = copy.deepcopy(dict(document))
    _require(set(artifact) == _ARTIFACT_KEYS, "artifact field set mismatch")
    _require(artifact.get("payload_sha256") == sealed_payload_sha256(artifact),
             "artifact payload seal mismatch")
    _require(artifact.get("schema_version") == SCHEMA_VERSION
             and artifact.get("kind") == ARTIFACT_KIND
             and artifact.get("status") == "complete",
             "artifact schema/kind/status mismatch")
    _require(artifact.get("profile_id") == LEGACY_H20_PROFILE_ID
             and artifact.get("profile_sha256") == H20_RUNTIME_PROFILE_SHA256,
             "artifact runtime profile mismatch")
    _require(phase in PHASE_SPLIT and artifact.get("phase") == phase
             and artifact.get("split") == PHASE_SPLIT[phase],
             "artifact phase/split mismatch")
    _require(_is_sha256(campaign_identity_sha256)
             and artifact.get("campaign_identity_sha256") == campaign_identity_sha256,
             "campaign identity hash mismatch")
    _require((arm, training_seed, seeds) in expected_grid(phase)
             and artifact.get("arm") == arm
             and artifact.get("training_seed") == training_seed
             and artifact.get("seeds") == seeds
             and tag == canonical_tag(arm, training_seed)
             and artifact.get("tag") == tag,
             "artifact arm/tag/seed grid mismatch")
    provenance = copy.deepcopy(dict(adapter_provenance))
    _require(artifact.get("adapter_sha256") == adapter_sha256
             and artifact.get("adapter_tree_sha256") == adapter_tree_sha256
             and artifact.get("adapter_provenance") == provenance
             and artifact.get("adapter_provenance_sha256") == canonical_sha256(provenance),
             "adapter identity/provenance mismatch")
    if arm == "base":
        _require(adapter_sha256 == "base" and adapter_tree_sha256 == "base",
                 "base artifact must use base sentinels")
    else:
        _require(_is_sha256(adapter_sha256) and _is_sha256(adapter_tree_sha256),
                 "trained adapter hashes are malformed")

    ordered_goal_ids = list(goal_ids)
    _require(len(ordered_goal_ids) == PHASE_GOALS[phase]
             and len(set(ordered_goal_ids)) == len(ordered_goal_ids),
             "goal denominator/order source is invalid")
    expected_keys = expected_row_keys(ordered_goal_ids, seeds)
    actual_keys = [
        (row.get("goal_index"), row.get("goal"), row.get("seed_idx"))
        for row in rows if isinstance(row, Mapping)
    ]
    _require(len(actual_keys) == len(rows) and actual_keys == expected_keys,
             "rows do not cover the exact manifest-order goal/seed grid")
    _require(artifact.get("goal_count") == len(ordered_goal_ids)
             and artifact.get("goal_ids_sha256") == canonical_sha256(ordered_goal_ids)
             and artifact.get("row_count") == len(rows)
             and artifact.get("rows_sha256") == canonical_sha256(rows),
             "artifact denominator or rows hash mismatch")
    runtime_reference = validate_runtime_reference_shape(artifact.get("runtime_reference"))
    validate_live_runtime_check(
        artifact.get("runtime_open_check"), runtime_reference, expected_phase="eval_open"
    )
    validate_live_runtime_check(
        artifact.get("runtime_close_check"), runtime_reference, expected_phase="eval_close"
    )
    _require(artifact.get("runtime_reference_sha256") == canonical_sha256(runtime_reference),
             "runtime reference digest mismatch")
    return artifact


def build_eval_artifact(
    *, campaign_identity_sha256: str, phase: str, arm: str,
    training_seed: int | None, tag: str, adapter_sha256: str,
    adapter_tree_sha256: str, adapter_provenance: Mapping[str, Any],
    goal_ids: Sequence[str], seeds: int, rows: Sequence[Mapping[str, Any]],
    runtime_reference: Mapping[str, Any], runtime_open_check: Mapping[str, Any],
    runtime_close_check: Mapping[str, Any],
) -> dict:
    provenance = copy.deepcopy(dict(adapter_provenance))
    document = seal({
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "status": "complete",
        "profile_id": LEGACY_H20_PROFILE_ID,
        "profile_sha256": H20_RUNTIME_PROFILE_SHA256,
        "phase": phase,
        "split": PHASE_SPLIT.get(phase),
        "campaign_identity_sha256": campaign_identity_sha256,
        "arm": arm,
        "tag": tag,
        "training_seed": training_seed,
        "adapter_sha256": adapter_sha256,
        "adapter_tree_sha256": adapter_tree_sha256,
        "adapter_provenance": provenance,
        "adapter_provenance_sha256": canonical_sha256(provenance),
        "goal_count": len(goal_ids),
        "goal_ids_sha256": canonical_sha256(list(goal_ids)),
        "seeds": seeds,
        "row_count": len(rows),
        "rows_sha256": canonical_sha256(rows),
        "runtime_reference": copy.deepcopy(dict(runtime_reference)),
        "runtime_open_check": copy.deepcopy(dict(runtime_open_check)),
        "runtime_close_check": copy.deepcopy(dict(runtime_close_check)),
        "runtime_reference_sha256": canonical_sha256(runtime_reference),
    })
    return validate_eval_artifact(
        document, campaign_identity_sha256=campaign_identity_sha256, phase=phase,
        arm=arm, training_seed=training_seed, tag=tag,
        adapter_sha256=adapter_sha256, adapter_tree_sha256=adapter_tree_sha256,
        adapter_provenance=provenance, goal_ids=goal_ids, seeds=seeds, rows=rows,
    )


def validate_learning_report(
    document: Any, learning_campaign_manifest: Mapping[str, Any],
) -> dict:
    _require(isinstance(document, Mapping), "learning report must be an object")
    report = copy.deepcopy(dict(document))
    _require(set(report) == _LEARNING_REPORT_KEYS,
             "learning report field set mismatch")
    _require(report.get("payload_sha256") == sealed_payload_sha256(report),
             "learning report payload seal mismatch")
    learning = validate_campaign_registry(learning_campaign_manifest)
    identity = learning["identity"]
    _require(report.get("schema_version") == SCHEMA_VERSION
             and report.get("kind") == LEARNING_REPORT_KIND
             and report.get("status") == "complete"
             and report.get("phase") == LEARNING_GATE_PHASE
             and report.get("n_goals") == PHASE_GOALS[LEARNING_GATE_PHASE],
             "learning report schema/phase/denominator mismatch")
    _require(report.get("campaign_id") == identity["campaign_id"]
             and report.get("campaign_identity_sha256") == learning["identity_sha256"]
             and report.get("campaign_manifest_sha256") == canonical_sha256(learning),
             "learning report campaign binding mismatch")
    _require(report.get("decision_bearing") is False
             and report.get("implicit_pass_threshold") is None
             and report.get("h1_verdict") is None
             and report.get("purpose")
             == "PI_review_only_before_explicit_final_ood_authorization",
             "learning report must remain non-decision-bearing without a threshold")
    metrics = report.get("arm_metrics")
    _require(isinstance(metrics, Mapping) and set(metrics) == set(ARMS),
             "learning report arm metrics are incomplete")
    expected_rows = {"base": 69 * BASE_K, "dense": 69 * 3, "sparse": 69 * 3}
    for arm, rows in expected_rows.items():
        values = metrics.get(arm)
        _require(isinstance(values, Mapping) and values.get("rows") == rows,
                 f"learning report row denominator mismatch for {arm}")
    _require(isinstance(report.get("analyzed_at"), str) and report["analyzed_at"],
             "learning report analyzed_at is missing")
    return report


def build_final_ood_authorization(
    learning_campaign_manifest: Mapping[str, Any], learning_report: Mapping[str, Any],
    *, final_campaign_id: str,
    approval_reference: str, authorized_by: str, authorized_at: str,
) -> dict:
    """Create a post-calibration PI authorization; no threshold is inferred."""
    learning = validate_campaign_registry(learning_campaign_manifest)
    report = validate_learning_report(learning_report, learning)
    identity = learning["identity"]
    _require(identity["phase"] == LEARNING_GATE_PHASE
             and identity["decision_bearing"] is False,
             "authorization source is not the completed non-decision-bearing 69-goal campaign")
    _require(isinstance(final_campaign_id, str)
             and SAFE_ID.fullmatch(final_campaign_id) is not None,
             "final campaign ID is unsafe")
    _require(isinstance(authorized_by, str) and authorized_by.strip(),
             "authorized_by is missing")
    _require(isinstance(authorized_at, str) and authorized_at.strip(),
             "authorized_at is missing")
    _require(isinstance(approval_reference, str) and approval_reference.strip(),
             "approval_reference is missing")
    return seal({
        "schema_version": SCHEMA_VERSION,
        "kind": FINAL_OOD_AUTHORIZATION_KIND,
        "status": "authorized",
        "scope": "unlock_exact_153_final_ood_campaign_once",
        "final_campaign_id": final_campaign_id,
        "learning_campaign_id": identity["campaign_id"],
        "learning_campaign_identity_sha256": learning["identity_sha256"],
        "learning_campaign_manifest_sha256": canonical_sha256(learning),
        "learning_metrics_decision_bearing": False,
        "learning_report_payload_sha256": report["payload_sha256"],
        "learning_report_sha256": canonical_sha256(report),
        "criterion": (
            "explicit_PI_directive_after_complete_learning_report;"
            "no_implicit_threshold"
        ),
        "approval_reference": approval_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_at": authorized_at.strip(),
    })


def validate_final_ood_authorization(
    document: Any, learning_campaign_manifest: Mapping[str, Any],
    learning_report: Mapping[str, Any], *,
    final_campaign_id: str,
) -> dict:
    _require(isinstance(document, Mapping), "final-OOD authorization must be an object")
    authorization = copy.deepcopy(dict(document))
    _require(set(authorization) == _AUTHORIZATION_KEYS,
             "final-OOD authorization field set mismatch")
    _require(authorization.get("payload_sha256") == sealed_payload_sha256(authorization),
             "final-OOD authorization payload seal mismatch")
    learning = validate_campaign_registry(learning_campaign_manifest)
    report = validate_learning_report(learning_report, learning)
    expected = build_final_ood_authorization(
        learning, report, final_campaign_id=final_campaign_id,
        approval_reference=str(authorization.get("approval_reference", "")),
        authorized_by=str(authorization.get("authorized_by", "")),
        authorized_at=str(authorization.get("authorized_at", "")),
    )
    _require(authorization == expected,
             "final-OOD authorization differs from the completed learning campaign or target")
    return authorization
