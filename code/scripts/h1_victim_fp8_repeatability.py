#!/usr/bin/env python3
"""Fail-closed FP8-only lifecycle repeatability proof for the pinned H1 victim.

The active H20 experiment uses exactly one victim precision: FP8.  One proof
collects the six frozen cases from an exact FP8 process (A), replaces that
process with a second exact FP8 lifecycle (B), and requires identical ordered
tool decisions, Phi, and success.  BF16 is never started by this entrypoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
for entry in (CODE, CODE / "src", HERE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import h1_victim_quant_spotcheck as base  # noqa: E402
from src.deployment_identity import verify_deployment  # noqa: E402
from src.domains.tooluse_oracle import ORACLE_VERSION  # noqa: E402
from src.h20_serving_identity import (  # noqa: E402
    build_formal_runtime_bundle,
    fresh_fp8_artifact_registry as pure_fresh_fp8_artifact_registry,
    live_runtime_check,
    runtime_reference,
    validate_fresh_h20_campaign_runtime_proof as validate_fresh_runtime_envelope,
    validate_h20_formal_runtime_bundle,
    validate_h20_runtime_protocol_equivalence,
    validate_runtime_reference,
)
from src.model_pins import VICTIM_HF_MODEL, VICTIM_REVISION, VICTIM_SERVED_NAME  # noqa: E402


SCHEMA_VERSION = 1
REPEATABILITY_KIND = "h1_victim_fp8_lifecycle_repeatability"
CYCLE_KIND = "h1_victim_fp8_repeatability_cycle_status"
DEPLOYMENT_REQUIRED = (*base.DEPLOYMENT_REQUIRED, "code/scripts/h1_victim_fp8_repeatability.py")


def _process_key(document: dict) -> tuple[int, int]:
    victim = document.get("victim") or {}
    pid = victim.get("process_pid")
    ticks = victim.get("process_start_time_ticks")
    if not isinstance(pid, int) or not isinstance(ticks, int):
        raise base.SpotcheckError("FP8 source lacks a canonical process lifecycle")
    return pid, ticks


def compare_fp8_documents(fp8_a: dict, fp8_b: dict) -> dict:
    """Recompute both sources and compare only pre-registered material fields."""
    a_cases = base._validate_artifact(fp8_a, "fp8")
    b_cases = base._validate_artifact(fp8_b, "fp8")
    for field in ("dataset", "oracle_version", "case_set", "config", "deployment"):
        if fp8_a.get(field) != fp8_b.get(field):
            raise base.SpotcheckError(f"FP8 A/B artifacts differ in identity field {field!r}")
    for field in ("hf_model", "revision", "served_model", "url", "quantization"):
        if (fp8_a.get("victim") or {}).get(field) != (fp8_b.get("victim") or {}).get(field):
            raise base.SpotcheckError(f"FP8 A/B victim identity differs in {field!r}")
    process_a = _process_key(fp8_a)
    process_b = _process_key(fp8_b)
    if process_a == process_b:
        raise base.SpotcheckError("FP8 A/B reused the same process lifecycle")

    differences: list[dict[str, Any]] = []
    for left, right in zip(a_cases, b_cases, strict=True):
        goal_id = left["goal_id"]
        if right.get("goal_id") != goal_id:
            raise base.SpotcheckError("FP8 A/B case ordering differs")
        changed: dict[str, Any] = {}
        for field in ("ordered_decisions", "max_phi", "success"):
            if left.get(field) != right.get(field):
                changed[field] = {"fp8_a": left.get(field), "fp8_b": right.get(field)}
        if changed:
            differences.append({"goal_id": goal_id, "differences": changed})

    return base._seal({
        "schema_version": SCHEMA_VERSION,
        "kind": REPEATABILITY_KIND,
        "created_at": base._now(),
        "status": "RUNTIME_DRIFT" if differences else "FP8_REPEATABLE",
        "material": bool(differences),
        "rule": (
            "any per-case difference in ordered canonical tool+argument decisions, max_phi, "
            "or success across two distinct FP8 process lifecycles is material"
        ),
        "case_set": base._expected_case_set(),
        "victim": {
            "hf_model": VICTIM_HF_MODEL,
            "revision": VICTIM_REVISION,
            "served_model": VICTIM_SERVED_NAME,
            "quantization": "fp8",
        },
        "deployment": fp8_a["deployment"],
        "config": fp8_a["config"],
        "fp8_a_payload_sha256": fp8_a["payload_sha256"],
        "fp8_b_payload_sha256": fp8_b["payload_sha256"],
        "fp8_a_process": {"pid": process_a[0], "start_time_ticks": process_a[1]},
        "fp8_b_process": {"pid": process_b[0], "start_time_ticks": process_b[1]},
        "n_cases": len(a_cases),
        "n_differences": len(differences),
        "differences": differences,
    })


def load_clean_fp8_repeatability(
    repeatability_path: str | Path,
    cycle_status_path: str | Path,
    *,
    expected_deployment_tree: str,
) -> dict:
    """Load and recompute the only proof accepted by the active FP8-only Gate."""
    repeatability_file = Path(repeatability_path).resolve()
    cycle_file = Path(cycle_status_path).resolve()
    if repeatability_file.parent != cycle_file.parent:
        raise base.SpotcheckError("repeatability and cycle status must share one unique directory")
    fp8_a_file = repeatability_file.parent / "fp8_a.json"
    fp8_b_file = repeatability_file.parent / "fp8_b.json"

    repeatability = base._load_json(repeatability_file)
    base._verify_seal(repeatability)
    if (repeatability.get("schema_version") != SCHEMA_VERSION
            or repeatability.get("kind") != REPEATABILITY_KIND):
        raise base.SpotcheckError("unsupported FP8 repeatability schema/kind")
    if (repeatability.get("status") != "FP8_REPEATABLE"
            or repeatability.get("material") is not False
            or repeatability.get("n_differences") != 0
            or repeatability.get("differences") != []):
        raise base.SpotcheckError("FP8 repeatability did not establish zero runtime drift")
    if repeatability.get("case_set") != base._expected_case_set():
        raise base.SpotcheckError("FP8 repeatability case set mismatch")
    expected_victim = {
        "hf_model": VICTIM_HF_MODEL,
        "revision": VICTIM_REVISION,
        "served_model": VICTIM_SERVED_NAME,
        "quantization": "fp8",
    }
    if repeatability.get("victim") != expected_victim:
        raise base.SpotcheckError("FP8 repeatability victim identity mismatch")
    deployment = repeatability.get("deployment") or {}
    if deployment.get("deployed_tree_sha256") != expected_deployment_tree:
        raise base.SpotcheckError("FP8 repeatability differs from the verified deployment tree")

    fp8_a = base._load_json(fp8_a_file)
    fp8_b = base._load_json(fp8_b_file)
    recomputed = compare_fp8_documents(fp8_a, fp8_b)
    compared_fields = (
        "status", "material", "rule", "case_set", "victim", "deployment", "config",
        "fp8_a_payload_sha256", "fp8_b_payload_sha256", "fp8_a_process",
        "fp8_b_process", "n_cases", "n_differences", "differences",
    )
    for field in compared_fields:
        if recomputed.get(field) != repeatability.get(field):
            raise base.SpotcheckError(f"FP8 repeatability differs from sources in {field!r}")

    cycle = base._load_json(cycle_file)
    base._verify_seal(cycle)
    required_cycle = {
        "schema_version": SCHEMA_VERSION,
        "kind": CYCLE_KIND,
        "status": "FP8_REPEATABILITY_VERIFIED",
        "comparison_exit_code": 0,
        "primary_error": None,
        "fp8_manifest_verified": True,
        "fp8_api_verified": True,
        "fp8_process_verified": True,
        "repeatability_payload_sha256": repeatability["payload_sha256"],
        "fp8_a_payload_sha256": repeatability["fp8_a_payload_sha256"],
        "fp8_b_payload_sha256": repeatability["fp8_b_payload_sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": cycle.get(key)}
        for key, value in required_cycle.items() if cycle.get(key) != value
    }
    if mismatches:
        raise base.SpotcheckError(f"FP8 repeatability cycle proof mismatch: {mismatches}")
    try:
        restored_runtime = validate_runtime_reference(cycle.get("restored_fp8_runtime"))
        b_process = repeatability["fp8_b_process"]
        if any(restored_runtime["process"].get(key) != b_process[key]
               for key in ("pid", "start_time_ticks")):
            raise base.SpotcheckError("active FP8 runtime is not the measured B lifecycle")
        runtime_bundle = build_formal_runtime_bundle(
            # Compatibility field: this hash binds the active lifecycle status proof.  The
            # FP8-only status document proves that no cross-precision process was launched.
            quant_cycle_status_payload_sha256=cycle["payload_sha256"],
            restored_fp8_runtime=restored_runtime,
            sealed_at=cycle["completed_at"],
        )
    except (KeyError, ValueError) as exc:
        raise base.SpotcheckError(f"FP8 repeatability runtime identity invalid: {exc}") from exc

    return {
        "repeatability_path": str(repeatability_file),
        "repeatability_file_sha256": base._sha256_file(repeatability_file),
        "repeatability_payload_sha256": repeatability["payload_sha256"],
        "cycle_status_path": str(cycle_file),
        "cycle_status_file_sha256": base._sha256_file(cycle_file),
        "cycle_status_payload_sha256": cycle["payload_sha256"],
        "fp8_a_artifact_path": str(fp8_a_file),
        "fp8_a_artifact_file_sha256": base._sha256_file(fp8_a_file),
        "fp8_a_payload_sha256": repeatability["fp8_a_payload_sha256"],
        "fp8_b_artifact_path": str(fp8_b_file),
        "fp8_b_artifact_file_sha256": base._sha256_file(fp8_b_file),
        "fp8_b_payload_sha256": repeatability["fp8_b_payload_sha256"],
        "case_set": repeatability["case_set"],
        "victim": repeatability["victim"],
        "deployment": repeatability["deployment"],
        "config": repeatability["config"],
        "oracle_version": ORACLE_VERSION,
        "status": repeatability["status"],
        "material": repeatability["material"],
        "cycle_status": cycle["status"],
        "fp8_restored": True,
        "source_status": {"fp8_a": fp8_a["status"], "fp8_b": fp8_b["status"]},
        "repeatability_bindings": {
            "fp8_a_payload_sha256": repeatability["fp8_a_payload_sha256"],
            "fp8_b_payload_sha256": repeatability["fp8_b_payload_sha256"],
        },
        "cycle_bindings": {
            "repeatability_payload_sha256": cycle["repeatability_payload_sha256"],
            "fp8_a_payload_sha256": cycle["fp8_a_payload_sha256"],
            "fp8_b_payload_sha256": cycle["fp8_b_payload_sha256"],
        },
        "runtime_bundle": runtime_bundle,
    }


def verify_fp8_cycle_runtime_live(cycle_status_path: str | Path, *, phase: str) -> dict:
    cycle = base._load_json(Path(cycle_status_path).resolve())
    base._verify_seal(cycle)
    if (cycle.get("schema_version") != SCHEMA_VERSION or cycle.get("kind") != CYCLE_KIND
            or cycle.get("status") != "FP8_REPEATABILITY_VERIFIED"):
        raise base.SpotcheckError("cycle status is not an FP8 repeatability runtime proof")
    try:
        reference = validate_runtime_reference(cycle.get("restored_fp8_runtime"))
        manifest, _process = base._verify_current("fp8")
        return live_runtime_check(reference, manifest, phase=phase, verified_at=base._now())
    except ValueError as exc:
        raise base.SpotcheckError(f"live FP8 runtime differs from repeatability proof: {exc}") from exc


def fresh_fp8_artifact_registry(proof: dict) -> list[tuple[str, str, str]]:
    try:
        return pure_fresh_fp8_artifact_registry(proof)
    except ValueError as exc:
        raise base.SpotcheckError(str(exc)) from exc


def _behavior_signature(proof: dict, paths: dict[str, str | Path] | None = None) -> dict:
    by_lifecycle: dict[str, list[dict]] = {}
    for lifecycle in ("fp8_a", "fp8_b"):
        path = Path((paths or {}).get(
            lifecycle, proof.get(f"{lifecycle}_artifact_path", "")
        )).resolve()
        expected_hash = proof.get(f"{lifecycle}_artifact_file_sha256")
        if not path.is_file() or base._sha256_file(path) != expected_hash:
            raise base.SpotcheckError(f"{lifecycle} artifact file/hash mismatch")
        cases = base._validate_artifact(base._load_json(path), "fp8")
        by_lifecycle[lifecycle] = [{
            "goal_id": case["goal_id"],
            "ordered_decisions": case["ordered_decisions"],
            "max_phi": case["max_phi"],
            "success": case["success"],
        } for case in cases]
    return {
        "case_set": proof["case_set"],
        "by_lifecycle": by_lifecycle,
        "sha256": base._sha256_json(by_lifecycle),
    }


def validate_fresh_h20_campaign_runtime(gate_check: dict, fresh_check: dict) -> dict:
    try:
        equivalence = validate_h20_runtime_protocol_equivalence(
            gate_check.get("runtime_bundle"), fresh_check.get("runtime_bundle")
        )
    except ValueError as exc:
        raise base.SpotcheckError(f"fresh H20 runtime protocol mismatch: {exc}") from exc
    invariants = (
        "case_set", "victim", "deployment", "config", "oracle_version",
        "status", "material", "source_status",
    )
    mismatches = {field: {"gate": gate_check.get(field), "fresh": fresh_check.get(field)}
                  for field in invariants if gate_check.get(field) != fresh_check.get(field)}
    if mismatches:
        raise base.SpotcheckError(f"fresh FP8 protocol differs from Gate: {mismatches}")
    gate_behavior = _behavior_signature(gate_check)
    fresh_behavior = _behavior_signature(fresh_check)
    if gate_behavior != fresh_behavior:
        raise base.SpotcheckError("fresh FP8 behavior differs from immutable Gate lifecycle")
    proof = base._seal({
        "schema_version": SCHEMA_VERSION,
        "kind": "h1_h20_fresh_campaign_runtime",
        "status": "EQUIVALENT_NEW_LIFECYCLE",
        "immutable_gate_quantization_check_sha256": base._sha256_json(gate_check),
        "fresh_quantization_check_sha256": base._sha256_json(fresh_check),
        "runtime_equivalence": equivalence,
        "behavior_signature": gate_behavior,
        "fresh_runtime_bundle": fresh_check["runtime_bundle"],
        "deployment": fresh_check["deployment"],
        "case_set": fresh_check["case_set"],
        "config": fresh_check["config"],
        "oracle_version": fresh_check["oracle_version"],
    })
    return validate_fresh_h20_campaign_runtime_proof(
        proof, gate_check, fresh_check, verify_external_artifacts=True
    )


def validate_fresh_h20_campaign_runtime_proof(
    proof: dict, gate_check: dict, fresh_check: dict, *,
    verify_external_artifacts: bool = False,
    gate_artifact_paths: dict[str, str | Path] | None = None,
    fresh_artifact_paths: dict[str, str | Path] | None = None,
) -> dict:
    try:
        checked = validate_fresh_runtime_envelope(proof, gate_check, fresh_check)
    except ValueError as exc:
        raise base.SpotcheckError(f"fresh H20 runtime proof envelope invalid: {exc}") from exc
    if verify_external_artifacts:
        for check, overrides, label in (
            (gate_check, gate_artifact_paths, "gate"),
            (fresh_check, fresh_artifact_paths, "fresh"),
        ):
            for artifact_label, original, digest in fresh_fp8_artifact_registry(check):
                path = Path((overrides or {}).get(artifact_label, original)).resolve()
                if not path.is_file() or base._sha256_file(path) != digest:
                    raise base.SpotcheckError(f"{label} {artifact_label} artifact file/hash mismatch")
        gate_behavior = _behavior_signature(gate_check, gate_artifact_paths)
        fresh_behavior = _behavior_signature(fresh_check, fresh_artifact_paths)
        if checked["behavior_signature"] != gate_behavior \
                or checked["behavior_signature"] != fresh_behavior:
            raise base.SpotcheckError("fresh FP8 campaign behavior differs from raw artifacts")
    return checked


def verify_fp8_gate_runtime(check: dict, *, phase: str) -> dict:
    return base.verify_quant_gate_runtime(check, phase=phase)


def _ensure_fp8(timeout: int) -> dict:
    try:
        manifest, _process = base._verify_current("fp8")
        return manifest
    except Exception:
        return base._restore_fp8(timeout)


def _run_cycle(out_dir: Path, timeout: int) -> int:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise base.SpotcheckError(f"refusing to reuse cycle directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    fp8_a_path = out_dir / "fp8_a.json"
    fp8_b_path = out_dir / "fp8_b.json"
    repeatability_path = out_dir / "repeatability.json"
    primary_error: Exception | None = None
    exit_code = 2
    fp8_a: dict | None = None
    fp8_b: dict | None = None
    repeatability: dict | None = None
    restored_manifest: dict | None = None
    try:
        base._verify_current("fp8")
        fp8_a = base.collect_document("fp8")
        base._write_json_new(fp8_a_path, fp8_a)
        if fp8_a["status"] != "valid":
            raise base.SpotcheckError("FP8-A collection invalid; refusing lifecycle replacement")
        base._stop_exact_server("fp8", require_manifest=True)
        base._start_exact_server("fp8", timeout)
        fp8_b = base.collect_document("fp8")
        base._write_json_new(fp8_b_path, fp8_b)
        if fp8_b["status"] != "valid":
            raise base.SpotcheckError("FP8-B collection invalid")
        repeatability = compare_fp8_documents(fp8_a, fp8_b)
        base._write_json_new(repeatability_path, repeatability)
        exit_code = 1 if repeatability["material"] else 0
    except Exception as exc:
        primary_error = exc
    finally:
        try:
            restored_manifest = _ensure_fp8(timeout)
        except Exception as restore_exc:
            failure = base._seal({
                "schema_version": SCHEMA_VERSION,
                "kind": CYCLE_KIND,
                "completed_at": base._now(),
                "status": "RESTORE_FAILED",
                "primary_error": str(primary_error) if primary_error else None,
                "restore_error": str(restore_exc),
            })
            try:
                base._write_json_new(out_dir / "cycle_status.json", failure)
            except Exception:
                pass
            raise base.SpotcheckError(
                f"FP8 RESTORE FAILED: {restore_exc}; primary error: {primary_error}"
            ) from restore_exc
    status = base._seal({
        "schema_version": SCHEMA_VERSION,
        "kind": CYCLE_KIND,
        "completed_at": base._now(),
        "status": "FP8_REPEATABILITY_VERIFIED" if exit_code == 0 else "FP8_RESTORED",
        "comparison_exit_code": exit_code,
        "primary_error": str(primary_error) if primary_error else None,
        "fp8_manifest_verified": True,
        "fp8_api_verified": True,
        "fp8_process_verified": True,
        "repeatability_payload_sha256": (
            repeatability.get("payload_sha256") if repeatability is not None else None
        ),
        "fp8_a_payload_sha256": fp8_a.get("payload_sha256") if fp8_a else None,
        "fp8_b_payload_sha256": fp8_b.get("payload_sha256") if fp8_b else None,
        "restored_fp8_runtime": runtime_reference(restored_manifest) if restored_manifest else None,
    })
    base._write_json_new(out_dir / "cycle_status.json", status)
    if primary_error is not None:
        raise base.SpotcheckError(str(primary_error)) from primary_error
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--fp8-a", type=Path, required=True)
    compare.add_argument("--fp8-b", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--repeatability", type=Path, required=True)
    bundle.add_argument("--cycle-status", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    cycle = sub.add_parser("cycle")
    cycle.add_argument("--out-dir", type=Path, required=True)
    cycle.add_argument("--execute-restart", action="store_true")
    cycle.add_argument("--serve-timeout", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compare":
            result = compare_fp8_documents(base._load_json(args.fp8_a), base._load_json(args.fp8_b))
            base._write_json_new(args.out, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1 if result["material"] else 0
        if args.command == "bundle":
            deployment = verify_deployment(ROOT, required_paths=DEPLOYMENT_REQUIRED)
            check = load_clean_fp8_repeatability(
                args.repeatability, args.cycle_status,
                expected_deployment_tree=deployment["deployed_tree_sha256"],
            )
            runtime = validate_h20_formal_runtime_bundle(
                check["runtime_bundle"], require_gate_checks=False
            )
            base._write_json_new(args.out, runtime)
            print(json.dumps({
                "status": "FP8_RUNTIME_BUNDLE_READY",
                "artifact": str(args.out.resolve()),
                "runtime_bundle_payload_sha256": runtime["payload_sha256"],
            }, indent=2, sort_keys=True))
            return 0
        if args.serve_timeout < 60:
            raise base.SpotcheckError("--serve-timeout must be at least 60 seconds")
        if not args.execute_restart:
            print(json.dumps({
                "status": "DRY_RUN_NO_CHANGES",
                "plan": [
                    "verify and collect exact FP8 lifecycle A",
                    "stop only exact FP8 lifecycle A",
                    "start and collect exact FP8 lifecycle B",
                    "compare decisions/Phi/success and leave verified FP8 B running",
                ],
                "bf16_started": False,
                "out_dir_if_executed": str(args.out_dir.resolve()),
            }, indent=2, sort_keys=True))
            return 0
        return _run_cycle(args.out_dir, args.serve_timeout)
    except (base.SpotcheckError, RuntimeError, ValueError, OSError) as exc:
        print(f"FP8_REPEATABILITY_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
