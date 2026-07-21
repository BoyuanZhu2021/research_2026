"""SSH launch/status/recovery controller for the partial-reachable H1 pilot."""
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
for entry in (ROOT / "code", ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from h1_h20_controller_common import (  # noqa: E402
    GATE_SELECTION,
    download_and_verify,
    service_preflight,
)
from src.inprocess_curriculum_protocol import AUTHORIZED_INSTANCE, PROFILE_ID  # noqa: E402


REMOTE_ROOT = "/root/autodl-tmp/h1mt"
REMOTE_RUN_ROOT = f"{REMOTE_ROOT}/partial_reachable_pilot"
REMOTE_CONFIG = f"{REMOTE_ROOT}/code/configs/h1_partial_reachable_curriculum_v1.json"
REMOTE_TARGETED_CONFIG = f"{REMOTE_ROOT}/code/configs/h1_gate_partial_curriculum_v1.json"
REMOTE_TARGETED_NONE_CONFIG = (
    f"{REMOTE_ROOT}/code/configs/h1_gate_partial_none_curriculum_v1.json"
)
REMOTE_TARGETED_LEGACY_CONFIG = (
    f"{REMOTE_ROOT}/code/configs/h1_gate_partial_legacy_curriculum_v1.json"
)
REMOTE_CONFIRMATORY_CONFIG = (
    f"{REMOTE_ROOT}/code/configs/h1_gate_partial_legacy_confirmatory_v1.json"
)
LOCAL_ROOT = ROOT / "artifacts" / "h20-partial-reachable-pilot"
BASE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_SERVICE_MANAGER = f"{REMOTE_ROOT}/code/scripts/h1_victim_quant_spotcheck.py"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_tag(value: str) -> str:
    if not re.fullmatch(r"h1-pr-(?:base|dense|sparse)-s[012]-[0-9]{8}T[0-9]{6}Z", value):
        raise ValueError("invalid partial-reachable panel tag")
    return value


def _panel_command(
    *, tag: str, arm: str, seed: int, gpu_uuid: str, paired_tags: list[str],
    config_path: str = REMOTE_CONFIG,
) -> str:
    command = [
        BASE_PYTHON, "-u", "code/scripts/h1_inprocess_curriculum_pilot.py",
        "--profile", PROFILE_ID, "--arm", arm, "--seed", str(seed),
        "--tag", tag, "--run-root", REMOTE_ROOT,
        "--config", config_path, "--gate-selection", GATE_SELECTION,
    ]
    for paired_tag in paired_tags:
        command.extend([
            "--paired-config", f"{REMOTE_RUN_ROOT}/{_validate_tag(paired_tag)}/run_config.json"
        ])
    environment = " ".join((
        f"H1_INSTANCE_ID={AUTHORIZED_INSTANCE}",
        f"H1_H20_GPU_UUID={gpu_uuid}",
        f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
        "RANK=0", "LOCAL_RANK=0", "WORLD_SIZE=1", "LOCAL_WORLD_SIZE=1",
    ))
    return f"cd {shlex.quote(REMOTE_ROOT)} && {environment} {shlex.join(command)}"


def launch(
    client, *, arm: str, seed: int, paired_tags: list[str], targeted: bool = False,
    targeted_none: bool = False, targeted_legacy: bool = False,
    confirmatory: bool = False,
) -> dict:
    if sum((targeted, targeted_none, targeted_legacy, confirmatory)) > 1:
        raise ValueError("select at most one targeted curriculum")
    expected = 0 if arm == "base" else (1 if arm == "dense" else 2)
    if len(paired_tags) != expected:
        raise ValueError(f"{arm} requires exactly {expected} paired tags")
    preflight = service_preflight(client)
    tag = f"h1-pr-{arm}-s{seed}-{_stamp()}"
    logfile = f"{REMOTE_ROOT}/partial_reachable_{tag}.log"
    config_path = (
        REMOTE_CONFIRMATORY_CONFIG if confirmatory else (
        REMOTE_TARGETED_LEGACY_CONFIG if targeted_legacy else (
            REMOTE_TARGETED_NONE_CONFIG
            if targeted_none else (REMOTE_TARGETED_CONFIG if targeted else REMOTE_CONFIG)
        ))
    )
    command = _panel_command(
        tag=tag, arm=arm, seed=seed, gpu_uuid=preflight["gpu_uuid"],
        paired_tags=paired_tags,
        config_path=config_path,
    )
    detached = f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(logfile)} 2>&1 & echo $!"
    _stdin, stdout, stderr = client.exec_command(detached, timeout=30)
    pid_text = stdout.read().decode(errors="replace").strip()
    error_text = stderr.read().decode(errors="replace").strip()
    if not pid_text.isdigit():
        raise RuntimeError(f"panel did not return a PID: {error_text or pid_text}")
    return {
        "status": "PANEL_LAUNCHED", "instance_id": AUTHORIZED_INSTANCE,
        "gpu_uuid": preflight["gpu_uuid"], "arm": arm, "seed": seed,
        "tag": tag, "pid": int(pid_text), "log": logfile,
        "paired_tags": paired_tags, "targeted": targeted,
        "targeted_none": targeted_none,
        "targeted_legacy": targeted_legacy,
        "confirmatory": confirmatory,
    }


def status(client, *, tag: str, pid: int | None) -> int:
    import remote
    tag = _validate_tag(tag)
    logfile = f"{REMOTE_ROOT}/partial_reachable_{tag}.log"
    rc, out, err = remote.run(client, f"tail -n 80 {shlex.quote(logfile)}", timeout=60)
    print(out, end="")
    if err:
        print(err, file=sys.stderr, end="")
    if pid is not None:
        _rc, process, process_err = remote.run(
            client, f"ps -p {pid} -o pid=,stat=,etime=,pcpu=,pmem=,cmd=", timeout=30
        )
        print("PROCESS " + (process.strip() or "EXITED"))
        if process_err:
            print(process_err, file=sys.stderr, end="")
    run_dir = f"{REMOTE_RUN_ROOT}/{tag}"
    _rc, files, _err = remote.run(
        client,
        f"find {shlex.quote(run_dir)} -maxdepth 2 -type f -printf '%P %s bytes\\n' 2>/dev/null | sort || true",
        timeout=30,
    )
    print("ARTIFACTS\n" + (files.strip() or "not-created"))
    _rc, gpu, gpu_err = remote.run(
        client,
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
        timeout=30,
    )
    print("GPU_UTIL_MEMORY " + gpu.strip())
    if gpu_err:
        print(gpu_err, file=sys.stderr, end="")
    return rc


def recover(client, *, tag: str) -> dict:
    tag = _validate_tag(tag)
    remote_dir = f"{REMOTE_RUN_ROOT}/{tag}"
    local_dir = LOCAL_ROOT / tag
    recovery = download_and_verify(client, remote_dir, local_dir)
    remote_log = f"{REMOTE_ROOT}/partial_reachable_{tag}.log"
    local_log = LOCAL_ROOT / f"{tag}.controller.log"
    if local_log.exists():
        raise FileExistsError(f"controller log destination exists: {local_log}")
    local_log.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote_log, str(local_log))
        with sftp.file(remote_log, "rb") as handle:
            raw = handle.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
    finally:
        sftp.close()
    digest = hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(local_log.read_bytes()).hexdigest() != digest:
        raise RuntimeError("panel controller log recovery hash mismatch")
    return {
        "status": "PANEL_RECOVERY_VERIFIED", "instance_id": AUTHORIZED_INSTANCE,
        "tag": tag, **recovery, "controller_log": str(local_log),
        "controller_log_sha256": digest,
    }


def manage_service(client, *, action: str) -> dict:
    if action not in {"start", "status", "stop", "diagnose", "orphan-stop"}:
        raise ValueError("unsupported victim service action")
    import remote
    if action == "diagnose":
        diagnostics = {}
        commands = {
            "gpu": (
                "nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu "
                "--format=csv,noheader,nounits"
            ),
            "port_8000": "ss -ltnp 'sport = :8000' || true",
            "victim_log_tail": "tail -n 120 /root/autodl-tmp/h1_victim_fp8_spotcheck.log || true",
        }
        for label, diagnostic_command in commands.items():
            _rc, out, err = remote.run(client, diagnostic_command, timeout=60)
            diagnostics[label] = {"stdout": out, "stderr": err}
        return {
            "status": "VICTIM_SERVICE_DIAGNOSTICS",
            "instance_id": AUTHORIZED_INSTANCE,
            "action": action,
            "diagnostics": diagnostics,
        }
    if action == "orphan-stop":
        recovery_code = (
            "import json,sys; from pathlib import Path; "
            "sys.path[:0]=['/root/autodl-tmp/h1mt/code/scripts',"
            "'/root/autodl-tmp/h1mt/code','/root/autodl-tmp/h1mt']; "
            "import h1_victim_quant_spotcheck as s; "
            "assert not s.VICTIM_MANIFEST.exists(), 'manifest exists; use normal stop'; "
            "ps=s._port_vllm_processes(); selected=s._select_exact_process(ps,'fp8'); "
            "pid=int(selected['pid']); s._stop_exact_server('fp8',require_manifest=False); "
            "assert not s._port_vllm_processes() and not s._port_is_open(); "
            "print(json.dumps({'status':'EXACT_ORPHAN_STOPPED','pid':pid}))"
        )
        rc, out, err = remote.run(
            client, shlex.join([BASE_PYTHON, "-c", recovery_code]), timeout=120
        )
        if rc != 0:
            raise RuntimeError(f"exact orphan stop failed: {err or out}")
        return {
            "status": "VICTIM_SERVICE_ACTION_COMPLETE",
            "instance_id": AUTHORIZED_INSTANCE,
            "action": action,
            "service": json.loads(out),
        }
    command = shlex.join([
        BASE_PYTHON, REMOTE_SERVICE_MANAGER, "service", action,
        "--serve-timeout", "600",
    ])
    if action == "start":
        logfile = "/root/autodl-tmp/h1_victim_service_start_controller.log"
        detached = (
            f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(logfile)} 2>&1 "
            "& echo $!"
        )
        _stdin, stdout, stderr = client.exec_command(detached, timeout=30)
        pid_text = stdout.read().decode(errors="replace").strip()
        error_text = stderr.read().decode(errors="replace").strip()
        if not pid_text.isdigit():
            raise RuntimeError(
                f"victim service start did not return a PID: {error_text or pid_text}"
            )
        return {
            "status": "VICTIM_SERVICE_START_LAUNCHED",
            "instance_id": AUTHORIZED_INSTANCE,
            "action": action,
            "controller_pid": int(pid_text),
            "controller_log": logfile,
        }
    rc, out, err = remote.run(client, command, timeout=660)
    if rc != 0:
        raise RuntimeError(f"victim service {action} failed: {err or out}")
    try:
        payload = json.loads(out)
    except json.JSONDecodeError as exc:
        raise RuntimeError("victim service response is not JSON") from exc
    return {
        "status": "VICTIM_SERVICE_ACTION_COMPLETE",
        "instance_id": AUTHORIZED_INSTANCE,
        "action": action,
        "service": payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--launch", action="store_true")
    action.add_argument("--status-tag")
    action.add_argument("--recover-tag")
    action.add_argument(
        "--service-action",
        choices=("start", "status", "stop", "diagnose", "orphan-stop"),
    )
    parser.add_argument("--arm", choices=("base", "dense", "sparse"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--paired-tag", action="append", default=[])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--targeted-none", action="store_true")
    parser.add_argument("--targeted-legacy", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.launch, args.status_tag, args.recover_tag, args.service_action)):
        print(json.dumps({
            "mode": "plan", "instance_id": AUTHORIZED_INSTANCE,
            "sequence": ["base-s0", "dense-s0", "sparse-s0", "analyze", "conditional seed1"],
        }, indent=2))
        return 0
    if args.launch and (args.arm is None or args.seed is None):
        raise SystemExit("--launch requires --arm and --seed")
    if args.pid is not None and args.status_tag is None:
        raise SystemExit("--pid requires --status-tag")
    import remote
    client = remote.connect()
    try:
        transport = client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH transport is not active after connect")
        transport.set_keepalive(30)
        if args.launch:
            print(json.dumps(launch(
                client, arm=args.arm, seed=args.seed, paired_tags=args.paired_tag,
                targeted=args.targeted, targeted_none=args.targeted_none,
                targeted_legacy=args.targeted_legacy,
                confirmatory=args.confirmatory,
            ), indent=2, sort_keys=True))
            return 0
        if args.status_tag:
            return status(client, tag=args.status_tag, pid=args.pid)
        if args.service_action:
            print(json.dumps(
                manage_service(client, action=args.service_action),
                indent=2, sort_keys=True,
            ))
            return 0
        print(json.dumps(recover(client, tag=args.recover_tag), indent=2, sort_keys=True))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
