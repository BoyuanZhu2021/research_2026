"""Sealed artifact contract for the single-H20 formal QLoRA trainer.

All helpers are CPU-only.  Evaluation consumes this module rather than guessing
adapter identity from a directory name or accepting dual-V100 artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .h20_training_protocol import (
    FORMAL_TRAINING_PROTOCOL_SHA256,
    RUN_CONFIG_KIND,
    canonical_sha256,
    is_sha256,
    sealed_payload_sha256,
    validate_run_config,
)
from .runtime_profile import H20_RUNTIME_PROFILE_SHA256, LEGACY_H20_PROFILE_ID
from .h20_serving_identity import validate_live_runtime_check


SCHEMA_VERSION = 1
CHECKPOINT_KIND = "h1_h20_qlora_adapter_checkpoint"
ARTIFACT_KIND = "h1_h20_formal_training_artifacts"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: str | Path) -> str:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"adapter tree is unavailable: {root_path}")
    entries = sorted(root_path.rglob("*"), key=lambda path: path.relative_to(root_path).as_posix())
    if any(path.is_symlink() for path in entries):
        raise ValueError(f"adapter tree may not contain symlinks: {root_path}")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError(f"adapter tree is empty: {root_path}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_jsonl_objects(path: Path, label: str) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{label} contains a blank row at line {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{label} row {line_number} must contain a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label} {path}: {exc}") from exc
    return rows


def _validate_run_inventory(root: Path, checkpoint_steps: Sequence[int]) -> None:
    expected = {
        "run_config.json", "progress.jsonl", "rollouts.jsonl", "adapter",
        *{f"adapter_step{step}" for step in checkpoint_steps},
        *{f"adapter_step{step}.manifest.json" for step in checkpoint_steps},
    }
    actual = {path.name for path in root.iterdir()}
    # The final manifest cannot hash itself.  It is absent during pre-write
    # validation and present whenever a completed run is consumed.
    if "artifact_manifest.json" in actual:
        if (root / "artifact_manifest.json").is_symlink():
            raise ValueError("formal artifact manifest may not be a symlink")
        actual.remove("artifact_manifest.json")
    if actual != expected:
        raise ValueError(
            "formal run inventory mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for name in expected:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"formal run inventory may not contain symlinks: {name}")
        should_be_directory = name == "adapter" or name.startswith("adapter_step") and "." not in name
        if should_be_directory and not path.is_dir():
            raise ValueError(f"formal run inventory entry must be a directory: {name}")
        if not should_be_directory and not path.is_file():
            raise ValueError(f"formal run inventory entry must be a regular file: {name}")


def _exact_relative_path(value: Any, expected: str, label: str) -> str:
    if value != expected or Path(str(value)).is_absolute() or ".." in Path(str(value)).parts:
        raise ValueError(f"{label} path must be exactly {expected!r}")
    return expected


def _provenance_from_run_config(run_config: Mapping[str, Any]) -> dict:
    return {
        "runtime_profile": run_config["runtime_profile"],
        "runtime_profile_sha256": run_config["runtime_profile_sha256"],
        "training_protocol_sha256": run_config["training_protocol_sha256"],
        "gate1_identity": copy.deepcopy(run_config["gate1_identity"]),
        "runtime_identity": copy.deepcopy(run_config["runtime_identity"]),
        "deployment_identity": copy.deepcopy(run_config["deployment_identity"]),
        "benchmark_identity": copy.deepcopy(run_config["benchmark_identity"]),
        "benchmark_result_identity": copy.deepcopy(
            run_config["benchmark_result_identity"]
        ),
        "budget_authorization_identity": copy.deepcopy(
            run_config["budget_authorization_identity"]
        ),
        "models": copy.deepcopy(run_config["models"]),
        "data": copy.deepcopy(run_config["data"]),
        "oracle_and_interaction": copy.deepcopy(run_config["oracle_and_interaction"]),
    }


def build_checkpoint_manifest(
    *,
    step: int,
    adapter_tree_sha256: str,
    lora_sha256: str,
    run_config_file_sha256: str,
    run_config: Mapping[str, Any],
) -> dict:
    checked = validate_run_config(run_config)
    expected_steps = list(
        range(checked["ckpt_every"], checked["steps"], checked["ckpt_every"])
    )
    if (not isinstance(step, int) or isinstance(step, bool)
            or step not in expected_steps):
        raise ValueError("checkpoint step must exactly match the formal checkpoint cadence")
    for label, value in (
        ("adapter tree", adapter_tree_sha256),
        ("LoRA parameters", lora_sha256),
        ("run config file", run_config_file_sha256),
    ):
        if not is_sha256(value):
            raise ValueError(f"checkpoint {label} SHA-256 is malformed")
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "canonical_training_run": True,
        "tag": checked["tag"],
        "arm": checked["arm"],
        "seed": checked["seed"],
        "step": step,
        "adapter_path": f"adapter_step{step}",
        "adapter_tree_sha256": adapter_tree_sha256,
        "lora_sha256": lora_sha256,
        "run_config_path": "run_config.json",
        "run_config_file_sha256": run_config_file_sha256,
        "run_config_payload_sha256": checked["payload_sha256"],
        "provenance": _provenance_from_run_config(checked),
    }
    document["payload_sha256"] = canonical_sha256(document)
    return document


def validate_checkpoint_manifest(
    document: Mapping[str, Any],
    *,
    run_dir: str | Path,
    run_config: Mapping[str, Any],
) -> dict:
    if not isinstance(document, Mapping):
        raise ValueError("checkpoint manifest must be an object")
    checked = copy.deepcopy(dict(document))
    config = validate_run_config(run_config)
    if checked.get("schema_version") != SCHEMA_VERSION or checked.get("kind") != CHECKPOINT_KIND:
        raise ValueError("checkpoint manifest schema/kind mismatch")
    if checked.get("payload_sha256") != sealed_payload_sha256(checked):
        raise ValueError("checkpoint manifest payload seal mismatch")
    step = checked.get("step")
    if not isinstance(step, int) or isinstance(step, bool):
        raise ValueError("checkpoint manifest step is malformed")
    expected = build_checkpoint_manifest(
        step=step,
        adapter_tree_sha256=checked.get("adapter_tree_sha256"),
        lora_sha256=checked.get("lora_sha256"),
        run_config_file_sha256=checked.get("run_config_file_sha256"),
        run_config=config,
    )
    if checked != expected:
        raise ValueError("checkpoint manifest differs from the canonical H20 identity")
    root = Path(run_dir).resolve()
    adapter_name = _exact_relative_path(
        checked["adapter_path"], f"adapter_step{step}", "checkpoint adapter"
    )
    _exact_relative_path(checked["run_config_path"], "run_config.json", "checkpoint run config")
    if tree_sha256(root / adapter_name) != checked["adapter_tree_sha256"]:
        raise ValueError("checkpoint adapter tree hash mismatch")
    if sha256_file(root / "run_config.json") != checked["run_config_file_sha256"]:
        raise ValueError("checkpoint run config file hash mismatch")
    return checked


def build_artifact_manifest(
    *,
    run_config: Mapping[str, Any],
    run_config_file_sha256: str,
    adapter_tree_sha256: str,
    final_lora_sha256: str,
    progress_sha256: str,
    progress_rows: int,
    rollouts_sha256: str,
    rollout_rows: int,
    checkpoints: Sequence[Mapping[str, Any]],
    runtime_close_check: Mapping[str, Any],
) -> dict:
    checked = validate_run_config(run_config)
    for label, value in (
        ("run config file", run_config_file_sha256),
        ("adapter tree", adapter_tree_sha256),
        ("final LoRA", final_lora_sha256),
        ("progress", progress_sha256),
        ("rollouts", rollouts_sha256),
    ):
        if not is_sha256(value):
            raise ValueError(f"artifact {label} SHA-256 is malformed")
    expected_rows = checked["steps"] * checked["n_goals"] * checked["G"]
    if progress_rows != checked["steps"]:
        raise ValueError(
            f"progress row count must be {checked['steps']}, got {progress_rows}"
        )
    if rollout_rows != expected_rows:
        raise ValueError(f"rollout row count must be {expected_rows}, got {rollout_rows}")
    checkpoint_refs = [copy.deepcopy(dict(item)) for item in checkpoints]
    expected_steps = list(range(checked["ckpt_every"], checked["steps"], checked["ckpt_every"]))
    if [item.get("step") for item in checkpoint_refs] != expected_steps:
        raise ValueError("checkpoint registry does not exactly match the formal cadence")
    close_check = validate_live_runtime_check(
        runtime_close_check,
        checked["runtime"]["restored_fp8_runtime"],
        expected_phase="train_close",
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "canonical_training_run": True,
        "tag": checked["tag"],
        "arm": checked["arm"],
        "seed": checked["seed"],
        "adapter": {
            "path": "adapter",
            "tree_sha256": adapter_tree_sha256,
            "lora_sha256": final_lora_sha256,
        },
        "run_config": {
            "path": "run_config.json",
            "file_sha256": run_config_file_sha256,
            "payload_sha256": checked["payload_sha256"],
        },
        "progress": {
            "path": "progress.jsonl",
            "file_sha256": progress_sha256,
            "rows": progress_rows,
        },
        "rollouts": {
            "path": "rollouts.jsonl",
            "file_sha256": rollouts_sha256,
            "rows": rollout_rows,
        },
        "checkpoints": checkpoint_refs,
        "runtime_close_check": close_check,
        "provenance": _provenance_from_run_config(checked),
    }
    document["payload_sha256"] = canonical_sha256(document)
    return document


def validate_artifact_manifest(
    document: Mapping[str, Any],
    *,
    run_dir: str | Path,
    run_config: Mapping[str, Any],
) -> dict:
    if not isinstance(document, Mapping):
        raise ValueError("artifact manifest must be an object")
    checked = copy.deepcopy(dict(document))
    config = validate_run_config(run_config)
    if checked.get("schema_version") != SCHEMA_VERSION or checked.get("kind") != ARTIFACT_KIND:
        raise ValueError("artifact manifest schema/kind mismatch (V100 artifacts are forbidden)")
    if checked.get("payload_sha256") != sealed_payload_sha256(checked):
        raise ValueError("artifact manifest payload seal mismatch")
    root = Path(run_dir).resolve()
    adapter = checked.get("adapter") or {}
    run_ref = checked.get("run_config") or {}
    progress = checked.get("progress") or {}
    rollouts = checked.get("rollouts") or {}
    _exact_relative_path(adapter.get("path"), "adapter", "final adapter")
    _exact_relative_path(run_ref.get("path"), "run_config.json", "run config")
    _exact_relative_path(progress.get("path"), "progress.jsonl", "progress")
    _exact_relative_path(rollouts.get("path"), "rollouts.jsonl", "rollouts")
    if tree_sha256(root / "adapter") != adapter.get("tree_sha256"):
        raise ValueError("final adapter tree hash mismatch")
    if sha256_file(root / "run_config.json") != run_ref.get("file_sha256"):
        raise ValueError("final run config file hash mismatch")
    if run_ref.get("payload_sha256") != config["payload_sha256"]:
        raise ValueError("final run config payload binding mismatch")
    if sha256_file(root / "progress.jsonl") != progress.get("file_sha256"):
        raise ValueError("progress file hash mismatch")
    progress_rows = _load_jsonl_objects(root / "progress.jsonl", "progress")
    if len(progress_rows) != progress.get("rows"):
        raise ValueError("progress file row count mismatch")
    expected_progress_identity = {
        "tag": config["tag"],
        "arm": config["arm"],
        "seed": config["seed"],
        "global_B": config["n_goals"] * config["G"],
        "goal_schedule_sha256": config["global_goal_schedule_sha256"],
    }
    for expected_step, row in enumerate(progress_rows, start=1):
        if row.get("step") != expected_step:
            raise ValueError("progress steps are not the exact ordered formal cadence")
        if any(row.get(key) != value for key, value in expected_progress_identity.items()):
            raise ValueError(f"progress row {expected_step} identity mismatch")
        if not is_sha256(row.get("lora_sha256")):
            raise ValueError(f"progress row {expected_step} LoRA hash is malformed")
    if not progress_rows or progress_rows[-1]["lora_sha256"] != adapter.get("lora_sha256"):
        raise ValueError("final adapter LoRA hash differs from the final progress row")
    if sha256_file(root / "rollouts.jsonl") != rollouts.get("file_sha256"):
        raise ValueError("rollout file hash mismatch")
    rollout_rows = _load_jsonl_objects(root / "rollouts.jsonl", "rollouts")
    if len(rollout_rows) != rollouts.get("rows"):
        raise ValueError("rollout file row count mismatch")
    expected_rollout_keys = [
        (step, slot, trajectory)
        for step in range(1, config["steps"] + 1)
        for slot in range(config["n_goals"])
        for trajectory in range(config["G"])
    ]
    actual_rollout_keys = [
        (
            row.get("step"), row.get("global_group_slot"),
            row.get("trajectory_index"),
        )
        for row in rollout_rows
    ]
    if actual_rollout_keys != expected_rollout_keys:
        raise ValueError("rollout rows do not follow the exact formal step/group/trajectory order")
    if any(
        row.get("arm") != config["arm"] or row.get("seed") != config["seed"]
        for row in rollout_rows
    ):
        raise ValueError("rollout arm/seed identity mismatch")
    checkpoint_refs = checked.get("checkpoints")
    if not isinstance(checkpoint_refs, list):
        raise ValueError("checkpoint registry must be a list")
    expected_steps = list(range(config["ckpt_every"], config["steps"], config["ckpt_every"]))
    _validate_run_inventory(root, expected_steps)
    if [item.get("step") for item in checkpoint_refs] != expected_steps:
        raise ValueError("checkpoint registry cadence mismatch")
    for reference in checkpoint_refs:
        step = reference["step"]
        expected_path = f"adapter_step{step}.manifest.json"
        _exact_relative_path(reference.get("manifest_path"), expected_path, "checkpoint manifest")
        manifest_path = root / expected_path
        if sha256_file(manifest_path) != reference.get("manifest_file_sha256"):
            raise ValueError(f"checkpoint {step} manifest file hash mismatch")
        manifest = _load_object(manifest_path, f"checkpoint {step} manifest")
        validated = validate_checkpoint_manifest(manifest, run_dir=root, run_config=config)
        if validated["lora_sha256"] != progress_rows[step - 1]["lora_sha256"]:
            raise ValueError(f"checkpoint {step} LoRA hash differs from its progress row")
        if reference != {
            "step": step,
            "adapter_path": validated["adapter_path"],
            "adapter_tree_sha256": validated["adapter_tree_sha256"],
            "lora_sha256": validated["lora_sha256"],
            "manifest_path": expected_path,
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": validated["payload_sha256"],
        }:
            raise ValueError(f"checkpoint {step} registry binding mismatch")
    try:
        runtime_close_check = validate_live_runtime_check(
            checked.get("runtime_close_check"),
            config["runtime"]["restored_fp8_runtime"],
            expected_phase="train_close",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"final H20 runtime close check mismatch: {exc}") from exc
    expected = build_artifact_manifest(
        run_config=config,
        run_config_file_sha256=run_ref.get("file_sha256"),
        adapter_tree_sha256=adapter.get("tree_sha256"),
        final_lora_sha256=adapter.get("lora_sha256"),
        progress_sha256=progress.get("file_sha256"),
        progress_rows=progress.get("rows"),
        rollouts_sha256=rollouts.get("file_sha256"),
        rollout_rows=rollouts.get("rows"),
        checkpoints=checkpoint_refs,
        runtime_close_check=runtime_close_check,
    )
    if checked != expected:
        raise ValueError("artifact manifest differs from canonical H20 training identity")
    return checked


def load_and_validate_h20_adapter(
    adapter_dir: str | Path,
    *,
    expected_arm: str,
    expected_seed: int,
    expected_gate_bundle: Mapping[str, Any] | None = None,
    expected_runtime_reference: Mapping[str, Any] | None = None,
    expected_deployment: Mapping[str, Any] | None = None,
) -> dict:
    """Load a final H20 adapter and return its canonical evaluation provenance."""
    adapter_path = Path(adapter_dir).resolve()
    if adapter_path.name != "adapter" or not adapter_path.is_dir():
        raise ValueError("formal H20 adapter path must end in the final 'adapter' directory")
    run_dir = adapter_path.parent
    config = validate_run_config(_load_object(run_dir / "run_config.json", "run config"))
    artifact = validate_artifact_manifest(
        _load_object(run_dir / "artifact_manifest.json", "artifact manifest"),
        run_dir=run_dir,
        run_config=config,
    )
    if config["kind"] != RUN_CONFIG_KIND or config["runtime_profile"] != LEGACY_H20_PROFILE_ID:
        raise ValueError("adapter is not from the explicit single-H20 training schema")
    if config["runtime_profile_sha256"] != H20_RUNTIME_PROFILE_SHA256:
        raise ValueError("adapter H20 runtime profile hash mismatch")
    if config["training_protocol_sha256"] != FORMAL_TRAINING_PROTOCOL_SHA256:
        raise ValueError("adapter training protocol hash mismatch")
    if (config["arm"], config["seed"]) != (expected_arm, expected_seed):
        raise ValueError("adapter arm/seed differs from the evaluation registry")
    for label, expected, actual in (
        ("Gate", expected_gate_bundle, config["gate1"]),
        ("runtime", expected_runtime_reference, config["runtime"]),
        ("deployment", expected_deployment, config["deployment"]),
    ):
        if expected is not None and canonical_sha256(expected) != canonical_sha256(actual):
            raise ValueError(f"adapter {label} provenance differs from the campaign")
    return {
        "adapter_path": str(adapter_path),
        "adapter_tree_sha256": artifact["adapter"]["tree_sha256"],
        "final_lora_sha256": artifact["adapter"]["lora_sha256"],
        "artifact_manifest": artifact,
        "artifact_manifest_file_sha256": sha256_file(run_dir / "artifact_manifest.json"),
        "artifact_manifest_payload_sha256": artifact["payload_sha256"],
        "run_config": config,
        "run_config_file_sha256": sha256_file(run_dir / "run_config.json"),
        "run_config_payload_sha256": config["payload_sha256"],
        "arm": config["arm"],
        "seed": config["seed"],
    }
