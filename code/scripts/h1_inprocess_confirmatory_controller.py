"""SSH launch/status/recovery controller for the in-process H1 confirmation panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "code", ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from h1_h20_controller_common import download_and_verify, service_preflight  # noqa: E402
from h1_inprocess_confirmatory_eval import panel_key  # noqa: E402
from src.inprocess_curriculum_protocol import AUTHORIZED_INSTANCE  # noqa: E402


REMOTE_ROOT = "/root/autodl-tmp/h1mt"
BASE_PYTHON = "/root/miniconda3/bin/python"
LOCAL_ROOT = ROOT / "artifacts" / "h20-confirmatory"
SAFE_CAMPAIGN = re.compile(r"h1-confirm-(?:learning|final)-[0-9]{8}T[0-9]{6}Z")


def _validate_campaign(value: str) -> str:
    if SAFE_CAMPAIGN.fullmatch(value) is None:
        raise ValueError("unsafe confirmatory campaign ID")
    return value


def _adapter_path(tag: str) -> str:
    if re.fullmatch(r"h1-pr-(?:dense|sparse)-s[012]-[0-9]{8}T[0-9]{6}Z", tag) is None:
        raise ValueError("invalid source adapter tag")
    return f"{REMOTE_ROOT}/partial_reachable_pilot/{tag}/adapter_final"


def _upload_control_file(client, local_path: str, remote_path: str) -> str:
    local = Path(local_path).resolve()
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    directory = remote_path.rsplit("/", 1)[0]
    _stdin, stdout, stderr = client.exec_command(
        f"mkdir -p {shlex.quote(directory)}", timeout=30
    )
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(stderr.read().decode(errors="replace"))
    sftp = client.open_sftp()
    try:
        try:
            sftp.stat(remote_path)
        except OSError:
            temporary = remote_path + ".tmp"
            sftp.put(str(local), temporary)
            sftp.rename(temporary, remote_path)
        with sftp.file(remote_path, "rb") as handle:
            raw = handle.read()
            if isinstance(raw, str):
                raw = raw.encode()
    finally:
        sftp.close()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("uploaded control file hash mismatch")
    return digest


def launch(client, args: argparse.Namespace) -> dict:
    campaign = _validate_campaign(args.campaign_id)
    preflight = service_preflight(client)
    panel = panel_key(args.arm, args.seed)
    command = [
        BASE_PYTHON, "-u", "code/scripts/h1_inprocess_confirmatory_eval.py",
        "--phase", args.phase, "--campaign-id", campaign,
        "--arm", args.arm, "--run-root", REMOTE_ROOT,
        "--chunk-size", str(args.chunk_size),
    ]
    if args.arm != "base":
        command.extend(["--seed", str(args.seed), "--adapter", _adapter_path(args.adapter_tag)])
    uploaded = {}
    if args.phase == "final_ood":
        control_root = f"{REMOTE_ROOT}/confirmatory_control/{campaign}"
        learning_remote = f"{control_root}/learning_report.json"
        auth_remote = f"{control_root}/final_authorization.json"
        uploaded["learning_report_sha256"] = _upload_control_file(
            client, args.learning_report, learning_remote
        )
        uploaded["final_authorization_sha256"] = _upload_control_file(
            client, args.final_authorization, auth_remote
        )
        command.extend([
            "--learning-report", learning_remote,
            "--final-authorization", auth_remote,
        ])
    environment = " ".join((
        f"H1_INSTANCE_ID={AUTHORIZED_INSTANCE}",
        f"H1_H20_GPU_UUID={preflight['gpu_uuid']}",
        f"CUDA_VISIBLE_DEVICES={preflight['gpu_uuid']}",
        "RANK=0", "LOCAL_RANK=0", "WORLD_SIZE=1", "LOCAL_WORLD_SIZE=1",
    ))
    shell = f"cd {shlex.quote(REMOTE_ROOT)} && {environment} {shlex.join(command)}"
    logfile = f"{REMOTE_ROOT}/confirmatory_{campaign}_{panel}.log"
    detached = f"nohup bash -lc {shlex.quote(shell)} > {shlex.quote(logfile)} 2>&1 & echo $!"
    _stdin, stdout, stderr = client.exec_command(detached, timeout=30)
    pid_text = stdout.read().decode(errors="replace").strip()
    error = stderr.read().decode(errors="replace").strip()
    if not pid_text.isdigit():
        raise RuntimeError(f"confirmatory panel did not return PID: {error or pid_text}")
    return {
        "status": "CONFIRMATORY_PANEL_LAUNCHED",
        "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": preflight["gpu_uuid"],
        "campaign_id": campaign,
        "phase": args.phase,
        "panel": panel,
        "pid": int(pid_text),
        "log": logfile,
        "uploaded": uploaded,
    }


def status(client, *, campaign_id: str, arm: str, seed: int | None, pid: int | None) -> int:
    import remote
    campaign = _validate_campaign(campaign_id)
    panel = panel_key(arm, seed)
    logfile = f"{REMOTE_ROOT}/confirmatory_{campaign}_{panel}.log"
    rc, out, err = remote.run(client, f"tail -n 80 {shlex.quote(logfile)}", timeout=60)
    print(out, end="")
    if err:
        print(err, file=sys.stderr, end="")
    if pid is not None:
        _rc, process, _err = remote.run(
            client, f"ps -p {pid} -o pid=,stat=,etime=,pcpu=,pmem=,cmd=", timeout=30
        )
        print("PROCESS " + (process.strip() or "EXITED"))
    _rc, gpu, _err = remote.run(
        client,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
        timeout=30,
    )
    print("GPU_UTIL_MEMORY " + gpu.strip())
    return rc


def recover(client, *, campaign_id: str, arm: str, seed: int | None) -> dict:
    campaign = _validate_campaign(campaign_id)
    panel = panel_key(arm, seed)
    remote_dir = f"{REMOTE_ROOT}/confirmatory_eval/{campaign}/{panel}"
    local_dir = LOCAL_ROOT / campaign / panel
    recovery = download_and_verify(client, remote_dir, local_dir)
    remote_log = f"{REMOTE_ROOT}/confirmatory_{campaign}_{panel}.log"
    local_log = LOCAL_ROOT / campaign / f"{panel}.controller.log"
    sftp = client.open_sftp()
    try:
        local_log.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_log, str(local_log))
    finally:
        sftp.close()
    return {
        "status": "CONFIRMATORY_RECOVERY_VERIFIED",
        "instance_id": AUTHORIZED_INSTANCE,
        "campaign_id": campaign,
        "panel": panel,
        **recovery,
        "controller_log": str(local_log),
        "controller_log_sha256": hashlib.sha256(local_log.read_bytes()).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--launch", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--recover", action="store_true")
    parser.add_argument("--phase", choices=("learning_report", "final_ood"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--arm", required=True, choices=("base", "dense", "sparse"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--adapter-tag")
    parser.add_argument("--learning-report")
    parser.add_argument("--final-authorization")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.arm == "base":
        if args.seed is not None or args.adapter_tag is not None:
            parser.error("base forbids seed/adapter tag")
    elif args.seed is None or args.adapter_tag is None:
        parser.error("trained panel requires seed and adapter tag")
    if args.launch and args.phase is None:
        parser.error("launch requires phase")
    if args.launch and args.phase == "final_ood" and (
        args.learning_report is None or args.final_authorization is None
    ):
        parser.error("final launch requires learning report and authorization")
    return args


def main() -> int:
    args = build_parser()
    import remote
    client = remote.connect()
    try:
        transport = client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH transport is inactive")
        transport.set_keepalive(30)
        if args.launch:
            print(json.dumps(launch(client, args), indent=2, sort_keys=True))
            return 0
        if args.status:
            return status(
                client, campaign_id=args.campaign_id, arm=args.arm,
                seed=args.seed, pid=args.pid,
            )
        print(json.dumps(recover(
            client, campaign_id=args.campaign_id, arm=args.arm, seed=args.seed,
        ), indent=2, sort_keys=True))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
