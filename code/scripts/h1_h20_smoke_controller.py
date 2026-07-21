#!/usr/bin/env python3
"""Explicit local controller for the single authorized H20 Gate smoke.

The default is plan-only.  ``--execute`` is required before this script opens
SSH or mutates remote state.  Keeping the command here avoids shell-dependent
quoting and makes the exact H20 launch contract auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import shlex
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

H20_GPU_UUID = "GPU-5467966a-0fdf-bf65-f715-629e45792480"
REMOTE_ROOT = "/root/autodl-tmp/h1mt"
REMOTE_OUT_ROOT = f"{REMOTE_ROOT}/gate1_smoke"
LOCAL_RECOVERY_ROOT = ROOT / "artifacts" / "h20-smokes"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def smoke_command() -> str:
    env = (
        "env -u OMP_NUM_THREADS OMP_NUM_THREADS=16 "
        f"H1_H20_GPU_UUID={H20_GPU_UUID} "
        f"CUDA_VISIBLE_DEVICES={H20_GPU_UUID} "
        "RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1"
    )
    script = f"{REMOTE_ROOT}/code/scripts/h1_tooluse_gate1_local.py"
    return (
        f"{env} /root/miniconda3/bin/python -u {script} "
        f"--smoke --out-root {REMOTE_OUT_ROOT}"
    )


def build_plan(logfile: str) -> dict:
    return {
        "mode": "execute" if logfile else "plan",
        "remote_command": smoke_command(),
        "remote_log": logfile or f"{REMOTE_ROOT}/gate1_smoke_structured_<UTC>.log",
        "remote_out_root": REMOTE_OUT_ROOT,
        "gpu_uuid": H20_GPU_UUID,
        "world_size": 1,
    }


def _download_tree(sftp, remote_dir: str, local_dir: Path) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for attr in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, attr.filename)
        local_path = local_dir / attr.filename
        if stat.S_ISDIR(attr.st_mode):
            count += _download_tree(sftp, remote_path, local_path)
        else:
            # OpenSSH limits concurrent SFTP read requests.  Paramiko's unlimited
            # default can stall large artifact recovery once that server-side
            # window is exhausted, so match OpenSSH's documented request cap.
            sftp.get(
                remote_path,
                str(local_path),
                max_concurrent_prefetch_requests=64,
            )
            count += 1
    return count


def _remote_hashes(sftp, remote_dir: str, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for attr in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, attr.filename)
        relative = posixpath.join(prefix, attr.filename) if prefix else attr.filename
        if stat.S_ISDIR(attr.st_mode):
            result.update(_remote_hashes(sftp, remote_path, relative))
        else:
            digest = hashlib.sha256()
            with sftp.file(remote_path, "rb") as handle:
                handle.prefetch(
                    file_size=attr.st_size,
                    max_concurrent_requests=64,
                )
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            result[relative] = digest.hexdigest()
    return result


def _local_hashes(local_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in local_dir.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        result[path.relative_to(local_dir).as_posix()] = digest.hexdigest()
    return result


def _validate_run_id(parser: argparse.ArgumentParser, run_id: str) -> str:
    if (
        not run_id.startswith("tooluse-gate1-smoke-")
        or posixpath.basename(run_id) != run_id
    ):
        parser.error("run id must be one exact tooluse-gate1-smoke run id")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or explicitly launch the authorized H20 structured-output smoke."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--execute",
        action="store_true",
        help="connect to H20 and launch the detached smoke (default: plan only)",
    )
    action.add_argument(
        "--status-log",
        metavar="REMOTE_LOG",
        help="read one controller-created smoke log without changing remote state",
    )
    action.add_argument(
        "--recover-run",
        metavar="RUN_ID",
        help="download one exact smoke run into artifacts/h20-smokes",
    )
    action.add_argument(
        "--verify-recovered",
        metavar="RUN_ID",
        help="compare every recovered byte with the exact remote smoke run",
    )
    parser.add_argument("--pid", type=int, help="optional detached PID to inspect with --status-log")
    args = parser.parse_args()

    if args.pid is not None and args.status_log is None:
        parser.error("--pid requires --status-log")

    if args.recover_run is not None:
        run_id = _validate_run_id(parser, args.recover_run)
        local_dir = LOCAL_RECOVERY_ROOT / run_id
        if local_dir.exists() and any(local_dir.iterdir()):
            parser.error(f"recovery destination already exists and is non-empty: {local_dir}")
        remote_dir = f"{REMOTE_OUT_ROOT}/{run_id}"
        import remote  # pylint: disable=import-outside-toplevel

        client = remote.connect()
        try:
            sftp = client.open_sftp()
            try:
                count = _download_tree(sftp, remote_dir, local_dir)
            finally:
                sftp.close()
        finally:
            client.close()
        print(f"RECOVERED files={count} remote={remote_dir} local={local_dir}")
        return 0

    if args.verify_recovered is not None:
        run_id = _validate_run_id(parser, args.verify_recovered)
        local_dir = LOCAL_RECOVERY_ROOT / run_id
        if not local_dir.is_dir():
            parser.error(f"recovered run does not exist: {local_dir}")
        remote_dir = f"{REMOTE_OUT_ROOT}/{run_id}"
        import remote  # pylint: disable=import-outside-toplevel

        client = remote.connect()
        try:
            sftp = client.open_sftp()
            try:
                remote_hashes = _remote_hashes(sftp, remote_dir)
            finally:
                sftp.close()
        finally:
            client.close()
        local_hashes = _local_hashes(local_dir)
        if remote_hashes != local_hashes:
            remote_only = sorted(set(remote_hashes) - set(local_hashes))
            local_only = sorted(set(local_hashes) - set(remote_hashes))
            mismatched = sorted(
                key for key in set(remote_hashes) & set(local_hashes)
                if remote_hashes[key] != local_hashes[key]
            )
            raise RuntimeError(
                f"recovery mismatch remote_only={remote_only} local_only={local_only} "
                f"hash_mismatch={mismatched}"
            )
        payload = json.dumps(local_hashes, sort_keys=True, separators=(",", ":"))
        print(json.dumps({
            "status": "RECOVERY_VERIFIED",
            "run_id": run_id,
            "file_count": len(local_hashes),
            "file_hashes": local_hashes,
            "hash_registry_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }, indent=2, sort_keys=True))
        return 0

    if args.status_log is not None:
        expected_prefix = f"{REMOTE_ROOT}/gate1_smoke_structured_"
        normalized = posixpath.normpath(args.status_log)
        if not normalized.startswith(expected_prefix) or not normalized.endswith(".log"):
            parser.error("--status-log must name a controller-created H20 smoke log")
        import remote  # pylint: disable=import-outside-toplevel

        client = remote.connect()
        try:
            rc, out, err = remote.run(
                client, f"tail -n 240 {shlex.quote(normalized)}", timeout=60
            )
            print(out, end="")
            if err:
                print(err, file=sys.stderr, end="")
            if rc != 0:
                return rc
            if args.pid is not None:
                rc, out, err = remote.run(
                    client,
                    f"ps -p {args.pid} -o pid=,stat=,etime=,cmd=",
                    timeout=30,
                )
                print("PROCESS " + (out.strip() or "EXITED"))
                if err:
                    print(err, file=sys.stderr, end="")
                if rc not in (0, 1):
                    return rc
        finally:
            client.close()
        return 0

    logfile = f"{REMOTE_ROOT}/gate1_smoke_structured_{_stamp()}.log" if args.execute else ""
    plan = build_plan(logfile)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute:
        return 0

    import remote  # pylint: disable=import-outside-toplevel

    client = remote.connect()
    try:
        remote.run_bg(client, smoke_command(), logfile)
    finally:
        client.close()
    print(f"SMOKE_LAUNCHED log={logfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
