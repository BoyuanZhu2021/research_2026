"""Explicit H20/vLLM victim start/status/stop controller for the frozen profile."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE.parent))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "src"))
from src.h20_serving_identity import (  # noqa: E402
    CUDA_HOME as H20_CUDA_HOME,
    CUDA_NVCC as H20_CUDA_NVCC,
    CUDA_NVCC_RELEASE as H20_CUDA_NVCC_RELEASE,
    SYSTEM_CXX as H20_SYSTEM_CXX,
    VLLM_NINJA as H20_VLLM_NINJA,
    VLLM_NINJA_BINARY_VERSION as H20_VLLM_NINJA_BINARY_VERSION,
    VLLM_NINJA_METADATA_VERSION as H20_VLLM_NINJA_METADATA_VERSION,
    VLLM_PYTHON as H20_VLLM_PYTHON,
    VLLM_TOOLCHAIN_PATH as H20_VLLM_TOOLCHAIN_PATH,
    VLLM_VERSION as H20_VLLM_VERSION,
    expected_cmdline,
)
from model_pins import (  # noqa: E402
    ATTACKER_MODEL,
    ATTACKER_REVISION,
    REMOTE_HF_HOME,
    VICTIM_H20_SERVED_NAME,
    VICTIM_HF_MODEL,
    VICTIM_REVISION as PINNED_VICTIM_REVISION,
)
from runtime_profile import (  # noqa: E402
    LEGACY_H20_PROFILE_ID,
)

TMP = "/root/autodl-tmp"
VICTIM_HF = VICTIM_HF_MODEL
VICTIM_REVISION = PINNED_VICTIM_REVISION
VICTIM_NAME = VICTIM_H20_SERVED_NAME
VICTIM_QUANT = "fp8"
VICTIM_MANIFEST = f"{TMP}/h1_victim_manifest.json"
SERVE = shlex.join(expected_cmdline("fp8"))
REMOTE_ROOT = f"{TMP}/h1mt"
REMOTE_MANAGER = f"{REMOTE_ROOT}/code/scripts/h1_victim_quant_spotcheck.py"
H20_BASE_PYTHON = "/root/miniconda3/bin/python"
H20_OMP_NUM_THREADS = "16"


def _scoped_command(argv: list[str], *, offline_hf: bool = False) -> str:
    """Build one process-scoped command with a known-good OpenMP value.

    AutoDL login shells can expose an invalid inherited ``OMP_NUM_THREADS``.  Unset and replace it
    only for the probe/manager process instead of mutating the remote shell or host configuration.
    """
    environment = [
        "env", "-u", "OMP_NUM_THREADS", f"OMP_NUM_THREADS={H20_OMP_NUM_THREADS}",
    ]
    if offline_hf:
        environment.extend((
            f"HF_HOME={REMOTE_HF_HOME}",
            "HF_HUB_DISABLE_XET=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
        ))
    return shlex.join([*environment, *argv])


def _model_preflight_source(role: str, model: str, revision: str) -> str:
    """Return a CPU-only exact-snapshot cache probe for one frozen model role."""
    return (
        "import json; from pathlib import Path; "
        "from huggingface_hub import snapshot_download; "
        "from safetensors import safe_open; "
        "from transformers import AutoConfig,AutoTokenizer; "
        f"m={model!r}; r={revision!r}; "
        "p=Path(snapshot_download(m,revision=r,local_files_only=True)).resolve(); "
        "assert p.name==r,(str(p),r); "
        "idx=json.loads((p/'model.safetensors.index.json').read_text()); "
        "fs=sorted(set(idx['weight_map'].values())); assert fs,'no weight shards'; "
        "bad=[f for f in fs if not (p/f).is_file() or (p/f).stat().st_size<1024]; "
        "assert not bad,bad; "
        "heads=[len(safe_open(str(p/f),framework='pt',device='cpu').keys()) for f in fs]; "
        "assert all(heads),heads; "
        "c=AutoConfig.from_pretrained(str(p),local_files_only=True); "
        "t=AutoTokenizer.from_pretrained(str(p),local_files_only=True); "
        f"print('{role}_REVISION_OK',m,r); "
        f"print('{role}_WEIGHTS_OK',len(fs),sum((p/f).stat().st_size for f in fs),sum(heads)); "
        f"print('{role}_CACHE_OK',c.model_type,len(t))"
    )


H20_CUDA_CHECK = _scoped_command([
    H20_VLLM_PYTHON,
    "-c",
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
])
H20_VLLM_PREFLIGHT = _scoped_command([
    H20_VLLM_PYTHON,
    "-c",
    (
        "from importlib.metadata import version as v; import vllm; "
        f"expected={H20_VLLM_VERSION!r}; actual=v('vllm'); "
        "assert actual==expected,(actual,expected); "
        "assert vllm.__version__==expected,(vllm.__version__,expected); "
        "print('H20_VLLM_STACK_OK',actual)"
    ),
])
H20_TOOLCHAIN_PREFLIGHT = _scoped_command([
    H20_VLLM_PYTHON,
    "-c",
    (
        "import os,re,subprocess; from pathlib import Path; "
        "from importlib.metadata import version as v; "
        f"ninja={H20_VLLM_NINJA!r}; nvcc={H20_CUDA_NVCC!r}; "
        f"cxx={H20_SYSTEM_CXX!r}; expected_meta={H20_VLLM_NINJA_METADATA_VERSION!r}; "
        f"expected_ninja={H20_VLLM_NINJA_BINARY_VERSION!r}; "
        f"expected_cuda={H20_CUDA_NVCC_RELEASE!r}; "
        "assert v('ninja')==expected_meta,(v('ninja'),expected_meta); "
        "assert all(Path(p).is_file() and os.access(p,os.X_OK) "
        "for p in (ninja,nvcc,cxx)); "
        "n=subprocess.run([ninja,'--version'],capture_output=True,text=True,timeout=30); "
        "assert n.returncode==0 and n.stdout.strip()==expected_ninja," 
        "(n.returncode,n.stdout,n.stderr); "
        "c=subprocess.run([nvcc,'--version'],capture_output=True,text=True,timeout=30); "
        "rels=re.findall(r'\\brelease\\s+([0-9]+\\.[0-9]+)\\b',c.stdout); "
        "assert c.returncode==0 and rels==[expected_cuda],(c.returncode,rels,c.stderr); "
        "x=subprocess.run([cxx,'--version'],capture_output=True,text=True,timeout=30); "
        "assert x.returncode==0 and x.stdout.strip(),(x.returncode,x.stderr); "
        f"assert {H20_CUDA_HOME!r}=='/usr/local/cuda'; "
        f"assert {H20_VLLM_TOOLCHAIN_PATH!r}.split(':')[0]=="
        f"{H20_VLLM_NINJA.rsplit('/', 1)[0]!r}; "
        "print('H20_VLLM_TOOLCHAIN_OK',expected_meta,expected_ninja,expected_cuda)"
    ),
])
H20_ATTACKER_PREFLIGHT = _scoped_command([
    H20_BASE_PYTHON,
    "-c",
    _model_preflight_source("ATTACKER", ATTACKER_MODEL, ATTACKER_REVISION),
], offline_hf=True)
H20_VICTIM_PREFLIGHT = _scoped_command([
    H20_VLLM_PYTHON,
    "-c",
    _model_preflight_source("VICTIM", VICTIM_HF, VICTIM_REVISION),
], offline_hf=True)
H20_PREFLIGHT = " && ".join((
    H20_VLLM_PREFLIGHT,
    H20_TOOLCHAIN_PREFLIGHT,
    H20_ATTACKER_PREFLIGHT,
    H20_VICTIM_PREFLIGHT,
))
H20_PREFLIGHT_MARKERS = (
    "H20_VLLM_STACK_OK",
    "H20_VLLM_TOOLCHAIN_OK",
    "ATTACKER_REVISION_OK",
    "ATTACKER_WEIGHTS_OK",
    "ATTACKER_CACHE_OK",
    "VICTIM_REVISION_OK",
    "VICTIM_WEIGHTS_OK",
    "VICTIM_CACHE_OK",
)


def _load_remote_stack():
    """Import SSH-only dependencies after argparse has accepted a real action.

    Deployment safety gates execute ``--help`` in a minimal CPU environment.  Importing
    ``remote`` at module load time pulled in Paramiko before argparse could exit, making a
    read-only help probe depend on the controller's optional SSH packages.
    """
    import remote as remote_module  # pylint: disable=import-outside-toplevel

    return remote_module, H20_PREFLIGHT


def _missing_preflight_markers(output: str) -> tuple[str, ...]:
    observed = {
        line.split(maxsplit=1)[0]
        for line in output.splitlines()
        if line.strip()
    }
    return tuple(marker for marker in H20_PREFLIGHT_MARKERS if marker not in observed)


def _parse_exact_h20_inventory(output: str) -> dict | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    fields = [field.strip() for field in lines[0].split(",", 3)]
    if len(fields) != 4 or fields[0] != "0" or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", fields[1]):
        return None
    if not re.search(r"(?:^|\s)H20(?:\s|$)", fields[2], flags=re.IGNORECASE):
        return None
    try:
        memory_total_mib = int(fields[3])
    except ValueError:
        return None
    if memory_total_mib < 90_000:
        return None
    return {
        "index": 0, "uuid": fields[1], "name": fields[2],
        "memory_total_mib": memory_total_mib,
    }


def _is_h20_line(line: str) -> bool:
    """Compatibility helper retained for CPU regression tests."""
    return bool(re.search(r"^\s*0\s*,.*\bH20\b", line, flags=re.IGNORECASE))


def _api_has_exact_model(payload: str) -> bool:
    try:
        ids = {item.get("id") for item in json.loads(payload).get("data", [])}
    except (AttributeError, TypeError, ValueError):
        return False
    return ids == {VICTIM_NAME}


def _process_matches(command: str) -> bool:
    def line_matches(line: str) -> bool:
        try:
            tokens = shlex.split(line)
        except ValueError:
            return False
        launch = any(
            Path(tokens[i]).name == "vllm"
            and tokens[i + 1:i + 3] == ["serve", VICTIM_HF]
            for i in range(max(0, len(tokens) - 2))
        )
        if not launch:
            return False
        expected_flags = {
            "--revision": VICTIM_REVISION,
            "--port": "8000",
            "--served-model-name": VICTIM_NAME,
        }
        for flag, expected in expected_flags.items():
            try:
                index = tokens.index(flag)
            except ValueError:
                return False
            if index + 1 >= len(tokens) or tokens[index + 1] != expected:
                return False
        if VICTIM_QUANT:
            try:
                index = tokens.index("--quantization")
            except ValueError:
                return False
            return index + 1 < len(tokens) and tokens[index + 1] == VICTIM_QUANT
        return "--quantization" not in tokens

    return any(line_matches(line) for line in command.splitlines())


def main(profile_id: str, action: str = "start") -> int:
    if profile_id != LEGACY_H20_PROFILE_ID:
        print(f"PROFILE_MISMATCH: H20 launcher requires {LEGACY_H20_PROFILE_ID}")
        return 2
    if action not in {"start", "status", "stop"}:
        print(f"ACTION_MISMATCH: unsupported H20 action {action!r}")
        return 2
    RM, preflight = _load_remote_stack()
    cli = RM.connect()
    try:
        if action == "start":
            # Cheap controller-side guards preserve no-card safety.  The deployed manager then
            # performs the authoritative UUID/memory/process/environment/API validation.
            rc, gpu_output, _ = RM.run(
                cli,
                "nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1",
                timeout=30,
            )
            if rc != 0 or not _is_h20_line(gpu_output.strip()):
                print(f"GPU_NOT_READY: {gpu_output.strip()[:160] or 'nvidia-smi failed'}")
                return 1
            rc, cuda_name, cuda_err = RM.run(
                cli,
                H20_CUDA_CHECK,
                timeout=60,
            )
            if rc != 0 or not re.search(r"\bH20\b", cuda_name, flags=re.IGNORECASE):
                detail = cuda_err.strip() or cuda_name.strip() or "vLLM torch CUDA unavailable"
                print(f"CUDA_NOT_READY: {detail[:200]}")
                return 1
            rc, cached, cache_err = RM.run(cli, preflight, timeout=300)
            missing = _missing_preflight_markers(cached)
            if rc != 0 or missing:
                detail = cache_err.strip()[:500] or cached.strip()[:500]
                if missing:
                    detail = f"missing markers={','.join(missing)}; {detail}"
                print(f"CACHE_NOT_READY: {detail}")
                return 1
            print("CACHE_PREFLIGHT_OK")

        manager = _scoped_command([
            H20_BASE_PYTHON,
            REMOTE_MANAGER,
            "service",
            action,
        ])
        command = (
            f"cd {shlex.quote(REMOTE_ROOT)} && "
            f"{manager}"
        )
        rc, out, err = RM.run(cli, command, timeout=900 if action == "start" else 180)
        if out.strip():
            print(out.strip())
        if rc != 0:
            print(f"H20_SERVICE_{action.upper()}_FAILED: {(err or out).strip()[:800]}")
            return 1
        return 0
    finally:
        cli.close()


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=(LEGACY_H20_PROFILE_ID,))
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    return main(args.profile, args.action)


if __name__ == "__main__":
    raise SystemExit(_cli())
