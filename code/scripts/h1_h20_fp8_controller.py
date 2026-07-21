#!/usr/bin/env python3
"""SSH controller for one canonical FP8-A -> FP8-B repeatability cycle."""
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

REMOTE_ROOT = "/root/autodl-tmp/h1mt"
REMOTE_CYCLE_ROOT = f"{REMOTE_ROOT}/fp8_cycles"
LOCAL_RECOVERY_ROOT = ROOT / "artifacts" / "h20-fp8-cycles"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_id(parser: argparse.ArgumentParser, cycle_id: str) -> str:
    if not cycle_id.startswith("h1-fp8-cycle-") or posixpath.basename(cycle_id) != cycle_id:
        parser.error("cycle id must be one exact h1-fp8-cycle run id")
    return cycle_id


def _validate_gpu_uuid(parser: argparse.ArgumentParser, gpu_uuid: str) -> str:
    if re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid or "") is None:
        parser.error("--gpu-uuid must be one exact NVIDIA GPU UUID")
    return gpu_uuid


def cycle_command(cycle_id: str, gpu_uuid: str) -> str:
    env = (
        "env -u OMP_NUM_THREADS OMP_NUM_THREADS=16 "
        f"H1_H20_GPU_UUID={gpu_uuid} CUDA_VISIBLE_DEVICES={gpu_uuid} "
        "RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1"
    )
    script = f"{REMOTE_ROOT}/code/scripts/h1_victim_fp8_repeatability.py"
    out_dir = f"{REMOTE_CYCLE_ROOT}/{cycle_id}"
    return (
        f"{env} /root/miniconda3/bin/python -u {script} cycle "
        f"--out-dir {out_dir} --execute-restart --serve-timeout 900"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--status-log")
    action.add_argument("--recover-cycle")
    action.add_argument("--verify-recovered")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.pid is not None and args.status_log is None:
        parser.error("--pid requires --status-log")
    if args.execute and args.gpu_uuid is None:
        parser.error("--execute requires --gpu-uuid")
    if not args.execute and args.gpu_uuid is not None:
        parser.error("--gpu-uuid is accepted only with --execute")

    import remote  # pylint: disable=import-outside-toplevel

    if args.status_log:
        normalized = posixpath.normpath(args.status_log)
        prefix = f"{REMOTE_ROOT}/fp8_cycle_h1-fp8-cycle-"
        if not normalized.startswith(prefix) or not normalized.endswith(".log"):
            parser.error("--status-log must name one controller-created FP8 log")
        client = remote.connect()
        try:
            rc, out, err = remote.run(client, f"tail -n 260 {shlex.quote(normalized)}", timeout=60)
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

    selected = args.recover_cycle or args.verify_recovered
    if selected:
        cycle_id = _validate_id(parser, selected)
        remote_dir = f"{REMOTE_CYCLE_ROOT}/{cycle_id}"
        local_dir = LOCAL_RECOVERY_ROOT / cycle_id
        client = remote.connect()
        try:
            sftp = client.open_sftp()
            try:
                if args.recover_cycle:
                    if local_dir.exists() and any(local_dir.iterdir()):
                        parser.error(f"recovery destination is non-empty: {local_dir}")
                    count = _download_tree(sftp, remote_dir, local_dir)
                    print(f"RECOVERED files={count} remote={remote_dir} local={local_dir}")
                    return 0
                if not local_dir.is_dir():
                    parser.error(f"recovered cycle does not exist: {local_dir}")
                remote_hashes = _remote_hashes(sftp, remote_dir)
            finally:
                sftp.close()
        finally:
            client.close()
        local_hashes = _local_hashes(local_dir)
        if remote_hashes != local_hashes:
            raise RuntimeError("recovered FP8 cycle differs from remote source")
        payload = json.dumps(local_hashes, sort_keys=True, separators=(",", ":"))
        print(json.dumps({
            "status": "RECOVERY_VERIFIED", "cycle_id": cycle_id,
            "file_count": len(local_hashes), "file_hashes": local_hashes,
            "hash_registry_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }, indent=2, sort_keys=True))
        return 0

    if not args.execute:
        placeholder = "h1-fp8-cycle-<UTC>"
        print(json.dumps({
            "mode": "plan", "gpu_uuid": "<required with --execute>",
            "remote_command": cycle_command(placeholder, "GPU-<exact-uuid>"),
            "switches": ["collect FP8-A", "restart exact FP8", "collect FP8-B", "compare"],
            "bf16_started": False,
        }, indent=2, sort_keys=True))
        return 0

    cycle_id = f"h1-fp8-cycle-{_stamp()}"
    gpu_uuid = _validate_gpu_uuid(parser, args.gpu_uuid)
    logfile = f"{REMOTE_ROOT}/fp8_cycle_{cycle_id}.log"
    client = remote.connect()
    try:
        rc, out, err = remote.run(client, f"mkdir -p {shlex.quote(REMOTE_CYCLE_ROOT)}", timeout=30)
        if rc != 0:
            raise RuntimeError(f"cannot create FP8 cycle root: {err or out}")
        remote.run_bg(client, cycle_command(cycle_id, gpu_uuid), logfile)
    finally:
        client.close()
    print(f"FP8_CYCLE_LAUNCHED cycle_id={cycle_id} log={logfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
