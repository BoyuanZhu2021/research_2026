"""Audit confirmatory panels, authorize final OOD, and emit the registered H1 verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))

from h1_inprocess_confirmatory_eval import (  # noqa: E402
    FINAL_AUTH_KIND, GRID, PHASES, PROFILE_ID, panel_key,
)
from src.domains.tooluse_injection import ToolUseInjectionDomain  # noqa: E402
from src.inprocess_curriculum_protocol import (  # noqa: E402
    AUTHORIZED_INSTANCE, file_sha256, seal_payload, validate_seal,
)
from src.local_vllm_victim import (  # noqa: E402
    FINAL_C0_TRANSPORT_ID, FINAL_C0_TRANSPORT_POLICY_SHA256,
    load_local_vllm_ledger,
)


PI_AUTHORIZATION = (
    "批准 h1-victim-final-c0-canonicalization-v1，我立即实现回归测试、部署新 profile，"
    "完整重跑七组 learning，创建新 final 授权并完成 153 条 post-exposure OOD。"
)
BOOTSTRAP_SAMPLES = 20_000
ALPHA = 0.05


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_exclusive(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_panel(path: str | Path, *, expected_phase: str) -> dict:
    root = Path(path)
    config = validate_seal(json.loads((root / "run_config.json").read_text(encoding="utf-8")))
    result = validate_seal(json.loads((root / "result.json").read_text(encoding="utf-8")))
    manifest = validate_seal(json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8")))
    if config.get("profile_id") != PROFILE_ID or result.get("profile_id") != PROFILE_ID:
        raise ValueError("confirmatory panel profile mismatch")
    if (
        config.get("post_exposure_confirmation") is not True
        or result.get("post_exposure_confirmation") is not True
        or config.get("victim_final_c0_transport_id") != FINAL_C0_TRANSPORT_ID
        or config.get("victim_final_c0_transport_policy_sha256")
        != FINAL_C0_TRANSPORT_POLICY_SHA256
    ):
        raise ValueError("confirmatory final C0 transport identity mismatch")
    if config.get("phase") != expected_phase or result.get("phase") != expected_phase:
        raise ValueError("confirmatory panel phase mismatch")
    if result.get("run_config_payload_sha256") != config.get("payload_sha256"):
        raise ValueError("confirmatory result/config binding mismatch")
    if result.get("attacker_decoder_guard") != config.get("attacker_decoder_guard"):
        raise ValueError("confirmatory decoder-guard identity mismatch")
    if manifest.get("result_payload_sha256") != result.get("payload_sha256"):
        raise ValueError("confirmatory manifest/result binding mismatch")
    for relative, digest in (manifest.get("files") or {}).items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f"confirmatory panel artifact drift: {relative}")
    rows = _load_jsonl(root / "rows.jsonl")
    if file_sha256(root / "rows.jsonl") != result.get("rows_file_sha256"):
        raise ValueError("confirmatory rows hash mismatch")
    raw_attacker = _load_jsonl(root / "raw_attacker_ledger.jsonl")
    if len(raw_attacker) != sum(int(row.get("n_turns", 0)) for row in rows):
        raise ValueError("raw attacker ledger denominator mismatch")
    if any(
        "raw_text" not in item
        or "raw_text_sha256" not in item
        or "raw_response_token_ids" not in item
        or "raw_response_token_ids_sha256" not in item
        for item in raw_attacker
    ):
        raise ValueError("raw attacker response is missing")
    for item in raw_attacker:
        if hashlib.sha256(str(item["raw_text"]).encode("utf-8")).hexdigest() != item["raw_text_sha256"]:
            raise ValueError("raw attacker response text drifted")
        token_ids = item["raw_response_token_ids"]
        digest = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if digest != item["raw_response_token_ids_sha256"]:
            raise ValueError("raw attacker response token IDs drifted")
    victim = load_local_vllm_ledger(
        root / "raw_victim_ledger.jsonl",
        require_complete=True,
        expected_final_c0_transport_id=FINAL_C0_TRANSPORT_ID,
    )
    if victim != result.get("victim_ledger"):
        raise ValueError("raw victim ledger summary mismatch")
    return {"root": str(root), "config": config, "result": result, "rows": rows}


def _replay_panels(panels: list[dict], *, phase: str) -> None:
    split, n_goals = PHASES[phase]
    domain = ToolUseInjectionDomain(attack="ds", defense_tier="light")
    goals = domain.load_goals(split, seed=0, n=n_goals)
    by_id = {goal.id: goal for goal in goals}
    for panel in panels:
        expected_rows = n_goals * (4 if panel["result"]["arm"] == "base" else 1)
        if len(panel["rows"]) != expected_rows:
            raise ValueError("confirmatory panel row denominator mismatch")
        grouped = defaultdict(list)
        for row in panel["rows"]:
            goal = by_id.get(row.get("goal"))
            if goal is None:
                raise ValueError("confirmatory row goal is outside registered split")
            prev_phi = 0.0
            replay_trace = []
            for turn in row.get("turns") or []:
                calls = [tuple(item) for item in (turn.get("attempt_calls") or [])]
                score = domain.score(goal, [turn.get("victim_reply", "")], calls)
                attempt_phi = float(score["phi"])
                phi = max(prev_phi, attempt_phi)
                if not math.isclose(float(turn.get("attempt_phi")), attempt_phi, abs_tol=1e-12):
                    raise ValueError("offline Oracle attempt-Phi mismatch")
                if not math.isclose(float(turn.get("phi")), phi, abs_tol=1e-12):
                    raise ValueError("offline Oracle running-Phi mismatch")
                replay_trace.append(phi)
                prev_phi = phi
            max_phi = max(replay_trace) if replay_trace else 0.0
            if replay_trace != row.get("phi_trace"):
                raise ValueError("offline Oracle Phi trace mismatch")
            if not math.isclose(float(row.get("max_phi")), max_phi, abs_tol=1e-12):
                raise ValueError("offline Oracle max-Phi mismatch")
            if bool(row.get("success")) != (max_phi == 1.0):
                raise ValueError("offline Oracle success mismatch")
            grouped[row["goal"]].append(row)
        expected_k = 4 if panel["result"]["arm"] == "base" else 1
        if len(grouped) != n_goals or any(len(rows) != expected_k for rows in grouped.values()):
            raise ValueError("confirmatory goal/K shape mismatch")


def _validate_grid(panels: list[dict], *, phase: str) -> dict[str, dict]:
    by_panel = {panel["result"]["panel"]: panel for panel in panels}
    expected = {panel_key(arm, seed) for arm, seed in GRID}
    if set(by_panel) != expected or len(by_panel) != len(panels):
        raise ValueError("confirmatory panel grid is incomplete or duplicated")
    campaigns = {panel["result"]["campaign_id"] for panel in panels}
    deployments = {panel["config"]["deployment_tree_sha256"] for panel in panels}
    services = {panel["config"]["service_manifest_payload_sha256"] for panel in panels}
    gpu_uuids = {panel["config"]["gpu_uuid"] for panel in panels}
    profile_configs = {panel["config"]["profile_config_file_sha256"] for panel in panels}
    guard_payloads = {
        panel["config"]["attacker_decoder_guard"]["payload_sha256"]
        for panel in panels
    }
    victim_transports = {
        (
            panel["config"]["victim_final_c0_transport_id"],
            panel["config"]["victim_final_c0_transport_policy_sha256"],
        )
        for panel in panels
    }
    if (
        len(campaigns) != 1
        or len(deployments) != 1
        or len(services) != 1
        or len(gpu_uuids) != 1
        or len(profile_configs) != 1
        or len(guard_payloads) != 1
        or victim_transports
        != {(FINAL_C0_TRANSPORT_ID, FINAL_C0_TRANSPORT_POLICY_SHA256)}
    ):
        raise ValueError("confirmatory campaign runtime identity drift")
    if any(panel["result"]["phase"] != phase for panel in panels):
        raise ValueError("confirmatory grid phase drift")
    policy_registry = {
        key: by_panel[key]["result"]["policy_provenance"] for key in sorted(by_panel)
    }
    for arm in ("dense", "sparse"):
        for seed in (0, 1, 2):
            provenance = policy_registry[f"{arm}-s{seed}"]
            if (provenance.get("arm"), provenance.get("training_seed")) != (arm, seed):
                raise ValueError("confirmatory policy registry arm/seed mismatch")
    return policy_registry


def build_learning_report(panel_dirs: list[str]) -> dict:
    panels = [load_panel(path, expected_phase="learning_report") for path in panel_dirs]
    policy_registry = _validate_grid(panels, phase="learning_report")
    _replay_panels(panels, phase="learning_report")
    metrics = {panel["result"]["panel"]: panel["result"]["metrics"] for panel in panels}
    return seal_payload({
        "schema_version": 1,
        "kind": "h1_gate_partial_learning_report",
        "profile_id": PROFILE_ID,
        "complete": True,
        "decision_bearing": False,
        "post_exposure_confirmation": True,
        "final_ood_read": False,
        "campaign_id": panels[0]["result"]["campaign_id"],
        "instance_id": AUTHORIZED_INSTANCE,
        "panel_count": 7,
        "policy_registry": policy_registry,
        "profile_config_file_sha256": panels[0]["config"]["profile_config_file_sha256"],
        "attacker_decoder_guard": panels[0]["config"]["attacker_decoder_guard"],
        "victim_final_c0_transport_id": FINAL_C0_TRANSPORT_ID,
        "victim_final_c0_transport_policy_sha256": (
            FINAL_C0_TRANSPORT_POLICY_SHA256
        ),
        "metrics": metrics,
        "offline_oracle_replay": "PASS",
        "scope": "calibration integrity only; no tuning, seed selection, or formal H1 decision",
    })


def build_final_authorization(learning_report_path: str | Path, *, final_campaign_id: str) -> dict:
    path = Path(learning_report_path)
    report = validate_seal(json.loads(path.read_text(encoding="utf-8")))
    if (
        report.get("kind") != "h1_gate_partial_learning_report"
        or report.get("profile_id") != PROFILE_ID
        or report.get("complete") is not True
        or report.get("decision_bearing") is not False
        or report.get("final_ood_read") is not False
        or report.get("post_exposure_confirmation") is not True
        or report.get("victim_final_c0_transport_id")
        != FINAL_C0_TRANSPORT_ID
        or report.get("victim_final_c0_transport_policy_sha256")
        != FINAL_C0_TRANSPORT_POLICY_SHA256
    ):
        raise ValueError("authorization requires a complete learning report")
    return seal_payload({
        "schema_version": 1,
        "kind": FINAL_AUTH_KIND,
        "profile_id": PROFILE_ID,
        "instance_id": AUTHORIZED_INSTANCE,
        "scope": "unlock the exact 153-goal final-OOD campaign once",
        "post_exposure_confirmation": True,
        "predecessor_final_campaign": "h1-confirm-final-20260721T090531Z",
        "predecessor_status": "engineering-invalid-after-first-read",
        "victim_final_c0_transport_id": FINAL_C0_TRANSPORT_ID,
        "victim_final_c0_transport_policy_sha256": (
            FINAL_C0_TRANSPORT_POLICY_SHA256
        ),
        "pi_authorization": PI_AUTHORIZATION,
        "final_campaign_id": final_campaign_id,
        "learning_report_file_sha256": file_sha256(path),
        "learning_report_payload_sha256": report["payload_sha256"],
        "policy_registry": report["policy_registry"],
        "expected_grid": [panel_key(arm, seed) for arm, seed in GRID],
        "decision_rule": {
            "alpha": ALPHA,
            "holm_family": ["dense_minus_sparse", "dense_minus_baseK", "sparse_minus_baseK"],
            "support": "all point estimates positive, 95% bootstrap intervals exclude zero, and Holm-adjusted p<0.05",
        },
    })


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def _goal_values(panels: dict[str, dict]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    base_rows = defaultdict(list)
    for row in panels["base-k4"]["rows"]:
        base_rows[row["goal"]].append(row)
    for goal, rows in base_rows.items():
        values[goal]["baseK"] = max(int(row["success"]) for row in rows)
        values[goal]["baseK_phi"] = max(float(row["max_phi"]) for row in rows)
    for arm in ("dense", "sparse"):
        rows_by_goal = defaultdict(list)
        for seed in (0, 1, 2):
            for row in panels[f"{arm}-s{seed}"]["rows"]:
                rows_by_goal[row["goal"]].append(row)
        for goal, rows in rows_by_goal.items():
            if len(rows) != 3:
                raise ValueError("trained arm does not have three seed rows per goal")
            values[goal][arm] = sum(int(row["success"]) for row in rows) / 3
            values[goal][f"{arm}_phi"] = sum(float(row["max_phi"]) for row in rows) / 3
    return dict(values)


def _contrast(values: dict[str, dict[str, float]], left: str, right: str,
              *, rng_seed: int) -> tuple[dict, float]:
    goals = sorted(values)
    diffs = [values[goal][left] - values[goal][right] for goal in goals]
    point = sum(diffs) / len(diffs)
    rng = random.Random(rng_seed)
    boots = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    lower, upper = _quantile(boots, 0.025), _quantile(boots, 0.975)
    p_two_sided = min(1.0, 2 * min(
        sum(value <= 0 for value in boots) / len(boots),
        sum(value >= 0 for value in boots) / len(boots),
    ))
    return ({"point": point, "ci95": [lower, upper], "bootstrap_samples": BOOTSTRAP_SAMPLES}, p_two_sided)


def analyze_final(panel_dirs: list[str], authorization_path: str | Path) -> dict:
    authorization = validate_seal(json.loads(Path(authorization_path).read_text(encoding="utf-8")))
    if (
        authorization.get("kind") != FINAL_AUTH_KIND
        or authorization.get("pi_authorization") != PI_AUTHORIZATION
        or authorization.get("post_exposure_confirmation") is not True
        or authorization.get("victim_final_c0_transport_id")
        != FINAL_C0_TRANSPORT_ID
        or authorization.get("victim_final_c0_transport_policy_sha256")
        != FINAL_C0_TRANSPORT_POLICY_SHA256
    ):
        raise ValueError("final analysis authorization mismatch")
    loaded = [load_panel(path, expected_phase="final_ood") for path in panel_dirs]
    policy_registry = _validate_grid(loaded, phase="final_ood")
    if policy_registry != authorization.get("policy_registry"):
        raise ValueError("final policy registry differs from authorization")
    if {panel["result"]["campaign_id"] for panel in loaded} != {authorization["final_campaign_id"]}:
        raise ValueError("final campaign ID differs from authorization")
    _replay_panels(loaded, phase="final_ood")
    panels = {panel["result"]["panel"]: panel for panel in loaded}
    values = _goal_values(panels)
    specs = {
        "dense_minus_sparse": ("dense", "sparse"),
        "dense_minus_baseK": ("dense", "baseK"),
        "sparse_minus_baseK": ("sparse", "baseK"),
    }
    contrasts, pvalues = {}, {}
    for index, (name, (left, right)) in enumerate(specs.items()):
        contrasts[name], pvalues[name] = _contrast(values, left, right, rng_seed=20260720 + index)
    adjusted = _holm(pvalues)
    for name in contrasts:
        contrasts[name]["p_two_sided"] = pvalues[name]
        contrasts[name]["p_holm"] = adjusted[name]
    supported = all(
        contrasts[name]["point"] > 0
        and contrasts[name]["ci95"][0] > 0
        and contrasts[name]["p_holm"] < ALPHA
        for name in contrasts
    )
    means = {
        key: sum(value[key] for value in values.values()) / len(values)
        for key in ("baseK", "dense", "sparse", "baseK_phi", "dense_phi", "sparse_phi")
    }
    return seal_payload({
        "schema_version": 1,
        "kind": "h1_gate_partial_final_ood_holm_verdict",
        "profile_id": PROFILE_ID,
        "decision_bearing": True,
        "final_ood_read": True,
        "post_exposure_confirmation": True,
        "victim_final_c0_transport_id": FINAL_C0_TRANSPORT_ID,
        "victim_final_c0_transport_policy_sha256": (
            FINAL_C0_TRANSPORT_POLICY_SHA256
        ),
        "n_goals": 153,
        "seeds_per_trained_arm": 3,
        "base_k": 4,
        "authorization_payload_sha256": authorization["payload_sha256"],
        "offline_oracle_replay": "PASS",
        "alpha": ALPHA,
        "contrasts": contrasts,
        "means": means,
        "h1_supported": supported,
        "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "scope_limit": (
            "post-exposure confirmation at ds/base,m=2; the first untouched read was "
            "consumed by engineering-invalid campaign h1-confirm-final-20260721T090531Z; "
            "no extrapolation to m>=3"
        ),
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("learning", "authorize-final", "final"))
    parser.add_argument("--panel-dir", action="append", default=[])
    parser.add_argument("--learning-report")
    parser.add_argument("--final-campaign-id")
    parser.add_argument("--final-authorization")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.mode in {"learning", "final"} and len(args.panel_dir) != 7:
        parser.error("learning/final analysis requires exactly seven panel directories")
    if args.mode == "learning":
        value = build_learning_report(args.panel_dir)
    elif args.mode == "authorize-final":
        if not args.learning_report or not args.final_campaign_id or args.panel_dir:
            parser.error("authorize-final requires learning report and final campaign ID only")
        value = build_final_authorization(args.learning_report, final_campaign_id=args.final_campaign_id)
    else:
        if not args.final_authorization:
            parser.error("final analysis requires explicit final authorization")
        value = analyze_final(args.panel_dir, args.final_authorization)
    _write_exclusive(Path(args.output), value)
    print(json.dumps({
        "kind": value["kind"], "payload_sha256": value["payload_sha256"],
        "verdict": value.get("verdict"), "h1_supported": value.get("h1_supported"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
