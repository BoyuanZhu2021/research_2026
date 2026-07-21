"""Shared local-controller primitives for the active single-H20 experiment path."""
from __future__ import annotations

import hashlib
import json
import posixpath
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "code", ROOT / "code" / "src", ROOT / "code" / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from h1_h20_smoke_controller import _download_tree, _local_hashes  # noqa: E402
from src.qwen35_fast_kernels import QWEN35_FAST_KERNEL_PINS  # noqa: E402


ACTIVE_INSTANCE_ID = "20d84f9474-d7816b14"
REMOTE_ROOT = "/root/autodl-tmp/h1mt"
BASE_PYTHON = "/root/miniconda3/bin/python"
GATE_SELECTION = (
    f"{REMOTE_ROOT}/formal_gate_runs/h1-formal-gate-20260719T070505Z/"
    "tooluse-gate1-formal-20260719T070543.477430Z-7f8e25d6/frozen_gate1.json"
)
GATE_SELECTION_SHA256 = "47d06a72ba9787494155c66c6985d3724e8102cced0da66f47d0064d98accf71"


def _kernel_status_command() -> str:
    return (
        f"cd {shlex.quote(REMOTE_ROOT)} && PYTHONPATH=code {BASE_PYTHON} -c "
        + shlex.quote(
            "import json; "
            "from src.qwen35_fast_kernels import qwen35_fast_kernel_status; "
            "print(json.dumps(qwen35_fast_kernel_status(),sort_keys=True))"
        )
    )


def _parse_kernel_status(output: str) -> dict:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("kernel status did not return one JSON line")
    value = json.loads(lines[0])
    if value.get("required_distribution_pins") != QWEN35_FAST_KERNEL_PINS:
        raise RuntimeError("remote fast-kernel pin contract drift")
    return value


def _remote_hashes(client, remote_dir: str) -> dict[str, str]:
    import remote

    command = (
        f"cd {shlex.quote(remote_dir)} && "
        "find . -type f -print0 | sort -z | xargs -0 sha256sum"
    )
    rc, out, err = remote.run(client, command, timeout=600)
    if rc != 0:
        raise RuntimeError(f"remote artifact hashing failed: {err or out}")
    result: dict[str, str] = {}
    for line in out.splitlines():
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise RuntimeError("malformed remote hash registry row") from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError("malformed remote artifact SHA-256")
        if not relative.startswith("./"):
            raise RuntimeError("remote hash path is not relative")
        relative = posixpath.normpath(relative[2:])
        if relative in ("", ".") or relative.startswith("../") or relative in result:
            raise RuntimeError("invalid or duplicate remote hash path")
        result[relative] = digest
    if not result:
        raise RuntimeError("remote artifact hash registry is empty")
    return dict(sorted(result.items()))


def download_and_verify(client, remote_dir: str, local_dir: Path) -> dict:
    """Recover one immutable remote artifact tree and compare every file hash."""
    if local_dir.exists() and any(local_dir.iterdir()):
        raise RuntimeError(f"recovery destination is non-empty: {local_dir}")
    sftp = client.open_sftp()
    try:
        count = _download_tree(sftp, remote_dir, local_dir)
    finally:
        sftp.close()
    remote_hashes = _remote_hashes(client, remote_dir)
    local_hashes = _local_hashes(local_dir)
    if remote_hashes != local_hashes:
        raise RuntimeError("recovered bytes differ from the remote source")
    payload = json.dumps(local_hashes, sort_keys=True, separators=(",", ":"))
    return {
        "files": count,
        "hash_registry_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "remote_dir": remote_dir,
        "local_dir": str(local_dir),
    }


def service_preflight(client) -> dict:
    """Verify the exact active H20, victim process, fast kernels and Gate selection."""
    import remote

    rc, gpu_out, gpu_err = remote.run(
        client,
        "nvidia-smi --query-gpu=name,uuid,memory.total,memory.used "
        "--format=csv,noheader,nounits",
        timeout=60,
    )
    if rc != 0:
        raise RuntimeError(f"nvidia-smi failed: {gpu_err or gpu_out}")
    rows = [line.strip() for line in gpu_out.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("probe requires exactly one H20 row")
    name, gpu_uuid, total, used = [part.strip() for part in rows[0].split(",")]
    if (
        "H20" not in name.upper()
        or not gpu_uuid.startswith("GPU-")
        or int(total) < 90_000
        or int(used) < 1
    ):
        raise RuntimeError("local victim service preflight GPU identity/state mismatch")
    rc, manifest_out, manifest_err = remote.run(
        client,
        f"cd {shlex.quote(REMOTE_ROOT)} && PYTHONPATH=code {BASE_PYTHON} -c "
        + shlex.quote(
            "import json; from pathlib import Path; "
            "from src.h20_serving_identity import MANIFEST_PATH,validate_service_manifest; "
            "m=validate_service_manifest(json.loads(Path(MANIFEST_PATH).read_text()),"
            "expected_quantization='fp8'); print(json.dumps({'gpu_uuid':m['gpu']['uuid'],"
            "'pid':m['process']['pid'],'start_time_ticks':m['process']['start_time_ticks'],"
            "'payload_sha256':m['payload_sha256']},sort_keys=True))"
        ),
        timeout=60,
    )
    if rc != 0:
        raise RuntimeError(f"local FP8 service manifest invalid: {manifest_err or manifest_out}")
    service = json.loads(manifest_out.strip())
    if service.get("gpu_uuid") != gpu_uuid:
        raise RuntimeError("local victim service/GPU UUID mismatch")
    kernel_rc, kernel_out, kernel_err = remote.run(
        client, _kernel_status_command(), timeout=60
    )
    if kernel_rc != 0:
        raise RuntimeError(f"fast-kernel status failed: {kernel_err or kernel_out}")
    kernels = _parse_kernel_status(kernel_out)
    if not kernels.get("distribution_pins_match"):
        raise RuntimeError("fast-kernel packages are not exact before probe")
    rc, gate_out, gate_err = remote.run(
        client, f"sha256sum {shlex.quote(GATE_SELECTION)}", timeout=30
    )
    if rc != 0 or gate_out.split(maxsplit=1)[0] != GATE_SELECTION_SHA256:
        raise RuntimeError(f"defense selection hash mismatch: {gate_err or gate_out}")
    return {
        "instance_id": ACTIVE_INSTANCE_ID,
        "gpu_name": name,
        "gpu_uuid": gpu_uuid,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "service": service,
        "kernel_status": kernels,
    }

