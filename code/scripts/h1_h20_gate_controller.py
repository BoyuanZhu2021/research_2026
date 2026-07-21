#!/usr/bin/env python3
"""SSH controller for the canonical single-H20 69x3 formal Gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from h1_h20_smoke_controller import _download_tree, _local_hashes, _remote_hashes  # noqa: E402
import h1_victim_fp8_repeatability as fp8  # noqa: E402

REMOTE_ROOT = "/root/autodl-tmp/h1mt"
REMOTE_GATE_ROOT = f"{REMOTE_ROOT}/formal_gate_runs"
LOCAL_GATE_ROOT = ROOT / "artifacts" / "h20-formal-gates"
LOCAL_FP8_ROOT = ROOT / "artifacts" / "h20-fp8-cycles"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_id(parser: argparse.ArgumentParser, controller_id: str) -> str:
    if (not controller_id.startswith("h1-formal-gate-")
            or posixpath.basename(controller_id) != controller_id):
        parser.error("controller id must be one exact h1-formal-gate run id")
    return controller_id


def _validate_cycle_id(parser: argparse.ArgumentParser, cycle_id: str) -> str:
    if (not cycle_id.startswith("h1-fp8-cycle-")
            or posixpath.basename(cycle_id) != cycle_id):
        parser.error("--fp8-cycle-id must be one exact h1-fp8-cycle run id")
    return cycle_id


def _load_cycle_binding(parser: argparse.ArgumentParser, cycle_id: str) -> dict:
    cycle_id = _validate_cycle_id(parser, cycle_id)
    cycle_dir = (LOCAL_FP8_ROOT / cycle_id).resolve()
    if cycle_dir.parent != LOCAL_FP8_ROOT.resolve() or not cycle_dir.is_dir():
        parser.error(f"recovered FP8 cycle does not exist: {cycle_dir}")
    try:
        repeatability = fp8.base._load_json(cycle_dir / "repeatability.json")
        deployment_tree = repeatability["deployment"]["deployed_tree_sha256"]
        proof = fp8.load_clean_fp8_repeatability(
            cycle_dir / "repeatability.json",
            cycle_dir / "cycle_status.json",
            expected_deployment_tree=deployment_tree,
        )
        gpu_uuid = proof["runtime_bundle"]["restored_fp8_runtime"]["gpu_uuid"]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(f"recovered FP8 cycle is not a clean Gate prerequisite: {exc}")
    if re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid or "") is None:
        parser.error("recovered FP8 cycle has an invalid GPU UUID")
    return {
        "cycle_id": cycle_id,
        "cycle_dir": cycle_dir,
        "gpu_uuid": gpu_uuid,
        "deployment_tree_sha256": deployment_tree,
        "runtime_bundle_payload_sha256": proof["runtime_bundle"]["payload_sha256"],
    }


def gate_command(controller_id: str, cycle_id: str, gpu_uuid: str) -> str:
    env = (
        "env -u OMP_NUM_THREADS OMP_NUM_THREADS=16 "
        f"H1_H20_GPU_UUID={gpu_uuid} CUDA_VISIBLE_DEVICES={gpu_uuid} "
        "RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1"
    )
    script = f"{REMOTE_ROOT}/code/scripts/h1_tooluse_gate1_local.py"
    out_root = f"{REMOTE_GATE_ROOT}/{controller_id}"
    cycle_root = f"{REMOTE_ROOT}/fp8_cycles/{cycle_id}"
    return (
        f"cd {REMOTE_ROOT} && {env} /root/miniconda3/bin/python -u {script} "
        f"--out-root {out_root} "
        f"--fp8-repeatability {cycle_root}/repeatability.json "
        f"--fp8-cycle-status {cycle_root}/cycle_status.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--status-log")
    action.add_argument("--recover")
    action.add_argument("--verify-recovered")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--fp8-cycle-id")
    args = parser.parse_args()
    if args.pid is not None and args.status_log is None:
        parser.error("--pid requires --status-log")
    if args.execute and args.fp8_cycle_id is None:
        parser.error("--execute requires --fp8-cycle-id")
    if any((args.status_log, args.recover, args.verify_recovered)) and args.fp8_cycle_id is not None:
        parser.error("--fp8-cycle-id is accepted only with --execute or plan mode")

    import remote  # pylint: disable=import-outside-toplevel

    if args.status_log:
        normalized = posixpath.normpath(args.status_log)
        prefix = f"{REMOTE_ROOT}/formal_gate_h1-formal-gate-"
        if not normalized.startswith(prefix) or not normalized.endswith(".log"):
            parser.error("--status-log must name one controller-created formal Gate log")
        client = remote.connect()
        try:
            rc, out, err = remote.run(client, f"tail -n 30 {shlex.quote(normalized)}", timeout=60)
            print(out, end="")
            if err:
                print(err, file=sys.stderr, end="")
            if args.pid is not None:
                _rc, process, process_err = remote.run(
                    client, f"ps -p {args.pid} -o pid=,stat=,etime=,cmd=", timeout=30
                )
                print("PROCESS " + (process.strip() or "EXITED"))
                if process_err:
                    print(process_err, file=sys.stderr, end="")
            return rc
        finally:
            client.close()

    selected = args.recover or args.verify_recovered
    if selected:
        controller_id = _validate_id(parser, selected)
        remote_dir = f"{REMOTE_GATE_ROOT}/{controller_id}"
        local_dir = LOCAL_GATE_ROOT / controller_id
        client = remote.connect()
        try:
            sftp = client.open_sftp()
            try:
                if args.recover:
                    if local_dir.exists() and any(local_dir.iterdir()):
                        parser.error(f"recovery destination is non-empty: {local_dir}")
                    count = _download_tree(sftp, remote_dir, local_dir)
                    print(f"RECOVERED files={count} remote={remote_dir} local={local_dir}")
                    return 0
                if not local_dir.is_dir():
                    parser.error(f"recovered Gate root does not exist: {local_dir}")
                remote_hashes = _remote_hashes(sftp, remote_dir)
            finally:
                sftp.close()
        finally:
            client.close()
        local_hashes = _local_hashes(local_dir)
        if remote_hashes != local_hashes:
            raise RuntimeError("recovered formal Gate differs from remote source")
        payload = json.dumps(local_hashes, sort_keys=True, separators=(",", ":"))
        print(json.dumps({
            "status": "RECOVERY_VERIFIED", "controller_id": controller_id,
            "file_count": len(local_hashes),
            "hash_registry_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }, indent=2, sort_keys=True))
        return 0

    if not args.execute:
        placeholder = "h1-formal-gate-<UTC>"
        if args.fp8_cycle_id is None:
            print(json.dumps({
                "mode": "plan", "fp8_cycle_id": "<required with --execute>",
                "calibration_goals": 69, "tiers": ["none", "light", "moderate"],
            }, indent=2, sort_keys=True))
            return 0
        binding = _load_cycle_binding(parser, args.fp8_cycle_id)
        print(json.dumps({
            "mode": "plan", "gpu_uuid": binding["gpu_uuid"],
            "remote_command": gate_command(
                placeholder, binding["cycle_id"], binding["gpu_uuid"]
            ),
            "calibration_goals": 69, "tiers": ["none", "light", "moderate"],
            "fp8_cycle_id": binding["cycle_id"],
            "deployment_tree_sha256": binding["deployment_tree_sha256"],
            "runtime_bundle_payload_sha256": binding["runtime_bundle_payload_sha256"],
        }, indent=2, sort_keys=True))
        return 0

    binding = _load_cycle_binding(parser, args.fp8_cycle_id)
    controller_id = f"h1-formal-gate-{_stamp()}"
    logfile = f"{REMOTE_ROOT}/formal_gate_{controller_id}.log"
    client = remote.connect()
    try:
        sftp = client.open_sftp()
        try:
            remote_cycle_dir = f"{REMOTE_ROOT}/fp8_cycles/{binding['cycle_id']}"
            remote_hashes = _remote_hashes(sftp, remote_cycle_dir)
        finally:
            sftp.close()
        if remote_hashes != _local_hashes(binding["cycle_dir"]):
            raise RuntimeError("formal Gate prerequisite differs between local recovery and remote")
        rc, out, err = remote.run(client, f"mkdir -p {shlex.quote(REMOTE_GATE_ROOT)}", timeout=30)
        if rc != 0:
            raise RuntimeError(f"cannot create formal Gate root: {err or out}")
        remote.run_bg(
            client,
            gate_command(controller_id, binding["cycle_id"], binding["gpu_uuid"]),
            logfile,
        )
    finally:
        client.close()
    print(
        f"FORMAL_GATE_LAUNCHED controller_id={controller_id} "
        f"fp8_cycle_id={binding['cycle_id']} gpu_uuid={binding['gpu_uuid']} log={logfile}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
