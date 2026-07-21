"""Reusable SSH runner for the H20 (AutoDL). Reads creds from .env fresh every connect
(host/port/password may rotate on container restart). NEVER logs the password.

Commands are wrapped in `bash -lc` so conda/python/PATH resolve (AutoDL's non-login shell
does not expose them). `run_in_background` starts a detached process via nohup so an SSH drop
does not kill it.
"""
from __future__ import annotations

import re
import shlex

try:
    from .providers import load_env
except ImportError:
    from providers import load_env

import paramiko


def _creds() -> dict:
    env = load_env()
    m = re.search(r"-p\s+(\d+)\s+(\S+)@(\S+)", env["REMOTE_HOST"])
    if not m:
        raise RuntimeError("cannot parse REMOTE_HOST (expected 'ssh -p PORT user@host')")
    return {"host": m.group(3), "port": int(m.group(1)), "user": m.group(2), "pw": env["REMOTE_PASSWORD"]}


def connect(timeout: int = 30, retries: int = 5) -> paramiko.SSHClient:
    """Connect with retries — the AutoDL gateway drops connections intermittently."""
    import time as _t
    last = None
    for a in range(retries):
        try:
            c = _creds()
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(c["host"], port=c["port"], username=c["user"], password=c["pw"],
                        timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
            return cli
        except Exception as e:  # noqa: BLE001
            last = e
            _t.sleep(5)
    raise last


def run(cli: paramiko.SSHClient, cmd: str, timeout: int = 300, login: bool = True) -> tuple[int, str, str]:
    """Run `cmd` (in a login bash by default). Returns (rc, stdout, stderr)."""
    full = f"bash -lc {shlex.quote(cmd)}" if login else cmd
    _i, o, e = cli.exec_command(full, timeout=timeout)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    rc = o.channel.recv_exit_status()
    return rc, out, err


def run_bg(cli: paramiko.SSHClient, cmd: str, logfile: str, *, append: bool = False) -> None:
    """Start `cmd` detached (nohup) writing to `logfile`; returns immediately (survives SSH drop)."""
    redirect = ">>" if append else ">"
    detached = (
        f"nohup bash -lc {shlex.quote(cmd)} {redirect} {shlex.quote(logfile)} "
        "2>&1 & echo started_pid=$!"
    )
    _i, o, _e = cli.exec_command(detached, timeout=30)
    print(o.read().decode(errors="replace").strip())


def put_text(cli: paramiko.SSHClient, text: str, remote_path: str) -> None:
    sftp = cli.open_sftp()
    try:
        with sftp.file(remote_path, "w") as f:
            f.write(text)
    finally:
        sftp.close()


def put_file(cli: paramiko.SSHClient, local_path: str, remote_path: str) -> None:
    """SFTP a local file to the remote (parent dirs must exist)."""
    sftp = cli.open_sftp()
    try:
        sftp.put(local_path, remote_path)
    finally:
        sftp.close()


PROBE = r"""
echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
echo "=== cuda ==="; nvcc --version 2>/dev/null | tail -1 || echo no-nvcc
echo "=== python/conda ==="; which python python3 conda 2>/dev/null; python --version 2>&1
echo "=== conda envs ==="; conda env list 2>/dev/null || echo no-conda
echo "=== torch ==="; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" 2>&1 | tail -1
echo "=== vllm ==="; /root/miniconda3/envs/vllm/bin/python -c "import vllm;print('vllm',vllm.__version__)" 2>&1 | tail -1
echo "=== trl ==="; python -c "import trl;print('trl',trl.__version__)" 2>&1 | tail -1
echo "=== disk ==="; df -h /root/autodl-tmp | tail -1
echo "=== hf_home ==="; du -sh /root/autodl-tmp/hf_home 2>/dev/null; ls /root/autodl-tmp/hf_home 2>/dev/null | head
echo "=== turbo ==="; source /etc/network_turbo 2>/dev/null && echo turbo_ok || echo no-turbo
"""


def main():
    cli = connect()
    print("connected.")
    rc, out, err = run(cli, PROBE, timeout=120)
    print(out)
    if err.strip():
        print("STDERR:", err[:500])
    cli.close()


if __name__ == "__main__":
    main()
