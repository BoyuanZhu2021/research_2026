"""No-card-safe provisioning for the tool-use H1 run.

The AutoDL container may be running without an attached GPU.  This script deliberately keeps all
preparation GPU-free: inspect disk/cache state, optionally remove the two superseded model caches,
verify the already-installed stacks, and download the new 4B attacker + 9B victim in a detached job.
The active ``--h20-only`` path reuses the exact H20 vLLM/toolchain/cache contract and never creates
the retired V100 Transformers Serve environment.  Starting a serving backend or loading quantized
weights remains a separate GPU-attached step.

Safe status (default):
  python code/scripts/h1_provision.py

Prepare/resume the active H20 downloads:
  python code/scripts/h1_provision.py --prepare --h20-only
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import h1_serve_victim_h20 as H20_SERVE  # noqa: E402
from model_pins import (  # noqa: E402
    ATTACKER_MODEL,
    ATTACKER_REVISION,
    REMOTE_HF_HOME,
    VICTIM_HF_MODEL,
    VICTIM_REVISION,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TMP = "/root/autodl-tmp"
HF_HOME = REMOTE_HF_HOME
HUB = f"{HF_HOME}/hub"
TFSERVE_VERSION = "5.13.1"
TFSERVE_ENV = f"{TMP}/envs/tfserve"
TFSERVE_PYTHON = f"{TFSERVE_ENV}/bin/python"
TFSERVE_BIN = f"{TFSERVE_ENV}/bin/transformers"
ATTACKER_HF = ATTACKER_MODEL
VICTIM_HF = VICTIM_HF_MODEL
OLD_CACHES = (
    f"{HUB}/models--Qwen--Qwen3-8B",
    f"{HUB}/models--Qwen--Qwen3.6-27B-FP8",
)

STATUS = rf"""
echo "## remote time"
date -Is
echo "## gpu (no-card is expected during preparation)"
nvidia-smi -L 2>&1 || true
echo "## disk"
df -h {TMP} | tail -1
echo "## model caches"
du -sh {HUB}/models--Qwen--Qwen3-8B 2>/dev/null || true
du -sh {HUB}/models--Qwen--Qwen3.6-27B-FP8 2>/dev/null || true
du -sh {HUB}/models--Qwen--Qwen3.5-4B 2>/dev/null || true
du -sh {HUB}/models--Qwen--Qwen3.5-9B 2>/dev/null || true
echo "## incomplete weight files (bytes, timestamp, path)"
find {HUB}/models--Qwen--Qwen3.5-4B {HUB}/models--Qwen--Qwen3.5-9B \
  -type f -name '*.incomplete' -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -nr | head -12 || true
echo "## stack"
python -c "from importlib.metadata import version as v; print('base',*[v(x) for x in ('torch','transformers','trl','peft','bitsandbytes','accelerate')])" 2>&1
/root/miniconda3/envs/vllm/bin/python -c "from importlib.metadata import version as v; print('vllm',*[v(x) for x in ('torch','transformers','vllm')])" 2>&1
if [ -x {TFSERVE_PYTHON} ]; then
  {TFSERVE_PYTHON} -c "from importlib.metadata import version as v; print('tfserve',*[v(x) for x in ('torch','transformers','fastapi','uvicorn')])" 2>&1
else
  echo "tfserve not prepared: {TFSERVE_ENV}"
fi
echo "## download log"
tail -5 {TMP}/h1_tooluse_dl.log 2>/dev/null || true
echo
echo "## download process"
pgrep -af "(hf|huggingface-cli) download (Qwen/Qwen3.5-4B|Qwen/Qwen3.5-9B)" || true
"""

HF_ENV = (f"source /etc/network_turbo >/dev/null 2>&1 || true; "
          f"export HF_HOME={HF_HOME} HF_HUB_DISABLE_XET=1")
DOWNLOAD = (f"{HF_ENV}; "
            f"if command -v hf >/dev/null 2>&1; then "
            f"hf download {ATTACKER_HF} --revision {ATTACKER_REVISION} && "
            f"hf download {VICTIM_HF} --revision {VICTIM_REVISION} && "
            f"echo H1_TOOLUSE_DOWNLOAD_DONE; "
            f"else huggingface-cli download {ATTACKER_HF} --revision {ATTACKER_REVISION} && "
            f"huggingface-cli download {VICTIM_HF} --revision {VICTIM_REVISION} && "
            f"echo H1_TOOLUSE_DOWNLOAD_DONE; fi")
DOWNLOAD_LOCK = f"{TMP}/h1_tooluse_download.lock"
LOCKED_DOWNLOAD = (
    f"flock -n -E 75 {DOWNLOAD_LOCK} bash -lc "
    + shlex.quote(DOWNLOAD)
    + " || { rc=$?; [ \"$rc\" -eq 75 ] && echo H1_TOOLUSE_DOWNLOAD_ALREADY_RUNNING; exit \"$rc\"; }"
)

PREPARE_TFSERVE = (
    f"set -eu; mkdir -p {TMP}/envs; "
    f"if [ ! -x {TFSERVE_PYTHON} ]; then "
    f"python -m venv --system-site-packages {TFSERVE_ENV}; fi; "
    f"grep -q '^include-system-site-packages = true$' {TFSERVE_ENV}/pyvenv.cfg; "
    f"source /etc/network_turbo >/dev/null 2>&1 || true; "
    f"{TFSERVE_PYTHON} -m pip install --upgrade "
    + shlex.quote(f"transformers[serving]=={TFSERVE_VERSION}")
)

VERIFY_TFSERVE = (
    f"{TFSERVE_PYTHON} -c \"from importlib.metadata import version as v; "
    f"assert v('transformers')=='{TFSERVE_VERSION}',v('transformers'); "
    "import torch,fastapi,uvicorn,pydantic; "
    "assert torch.__version__.startswith('2.8.'),torch.__version__; "
    "assert str(torch.version.cuda).startswith('12.8'),torch.version.cuda; "
    "print('TFSERVE_STACK_OK',torch.__version__,torch.version.cuda,v('transformers'),"
    "v('fastapi'),v('uvicorn'))\""
)

VERIFY_STACKS = (
    "python -c \"import torch,transformers,trl,peft,bitsandbytes,accelerate; "
    "print('TRAIN_STACK_OK',torch.__version__,transformers.__version__,trl.__version__)\" && "
    "/root/miniconda3/envs/vllm/bin/python -c \"import torch,transformers,vllm; "
    "print('VLLM_STACK_OK',torch.__version__,transformers.__version__,vllm.__version__)\" && "
    f"{VERIFY_TFSERVE}"
)

# Reuse the active serving contract instead of duplicating or weakening its exact vLLM, JIT
# toolchain, revision, weight-header, config, and tokenizer checks.  The base training stack check
# is included because the same cache will later load the 4B NF4 attacker.
VERIFY_H20_STACKS = (
    "python -c \"import torch,transformers,trl,peft,bitsandbytes,accelerate; "
    "print('TRAIN_STACK_OK',torch.__version__,transformers.__version__,trl.__version__)\" && "
    f"{H20_SERVE.H20_VLLM_PREFLIGHT} && {H20_SERVE.H20_TOOLCHAIN_PREFLIGHT}"
)
H20_PREFLIGHT = H20_SERVE.H20_PREFLIGHT
H20_PREFLIGHT_MARKERS = H20_SERVE.H20_PREFLIGHT_MARKERS

PREFLIGHT = (
    f"grep -q H1_TOOLUSE_DOWNLOAD_DONE {TMP}/h1_tooluse_dl.log || "
    f"{{ echo DOWNLOAD_NOT_COMPLETE; exit 3; }}; "
    f"{HF_ENV}; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
    f"python -c \"import json; from pathlib import Path; from huggingface_hub import snapshot_download; "
    "from safetensors import safe_open; "
    f"m='{ATTACKER_HF}'; r='{ATTACKER_REVISION}'; "
    "p=Path(snapshot_download(m,revision=r,local_files_only=True)); "
    "idx=json.loads((p/'model.safetensors.index.json').read_text()); "
    "fs=sorted(set(idx['weight_map'].values())); bad=[f for f in fs if not (p/f).is_file() or (p/f).stat().st_size<1024]; "
    "assert not bad,bad; heads=[len(safe_open(str(p/f),framework='pt',device='cpu').keys()) for f in fs]; "
    "assert all(heads),heads; print('ATTACKER_WEIGHTS_OK',len(fs),sum((p/f).stat().st_size for f in fs),sum(heads))\" && "
    f"python -c \"from transformers import AutoConfig,AutoTokenizer; m='{ATTACKER_HF}'; "
    f"r='{ATTACKER_REVISION}'; c=AutoConfig.from_pretrained(m,revision=r,local_files_only=True); "
    "t=AutoTokenizer.from_pretrained(m,revision=r,local_files_only=True); "
    "print('ATTACKER_CACHE_OK',c.model_type,len(t))\" && "
    f"{TFSERVE_PYTHON} -c \"import json; from pathlib import Path; "
    f"from huggingface_hub import snapshot_download; from safetensors import safe_open; m='{VICTIM_HF}'; "
    f"r='{VICTIM_REVISION}'; p=Path(snapshot_download(m,revision=r,local_files_only=True)); "
    "idx=json.loads((p/'model.safetensors.index.json').read_text()); "
    "fs=sorted(set(idx['weight_map'].values())); bad=[f for f in fs if not (p/f).is_file() or (p/f).stat().st_size<1024]; "
    "assert not bad,bad; heads=[len(safe_open(str(p/f),framework='pt',device='cpu').keys()) for f in fs]; "
    "assert all(heads),heads; print('VICTIM_WEIGHTS_OK',len(fs),sum((p/f).stat().st_size for f in fs),sum(heads))\" && "
    f"{TFSERVE_PYTHON} -c \"from transformers import AutoConfig,AutoTokenizer; "
    f"m='{VICTIM_HF}'; r='{VICTIM_REVISION}'; "
    "c=AutoConfig.from_pretrained(m,revision=r,local_files_only=True); "
    "t=AutoTokenizer.from_pretrained(m,revision=r,local_files_only=True); "
    "print('VICTIM_CACHE_OK',c.model_type,len(t))\""
)


def _delete_old(cli, remote_module) -> None:
    """Delete only the two statically-declared old model cache directories."""
    quoted = " ".join(OLD_CACHES)
    cmd = (f"set -eu; root={HUB}; "
           f"for d in {quoted}; do "
           "[ ! -e \"$d\" ] && continue; r=$(readlink -f \"$d\"); "
           "case \"$r\" in "
           f"{OLD_CACHES[0]}|{OLD_CACHES[1]}) ;; "
           "*) echo REFUSE_UNEXPECTED_PATH:$r; exit 2 ;; esac; "
           "rm -rf -- \"$r\"; echo DELETED:$r; done; "
           f"df -h {TMP} | tail -1")
    rc, out, err = remote_module.run(cli, cmd, timeout=600)
    print(out.rstrip())
    if rc:
        raise RuntimeError(f"old-cache deletion failed rc={rc}: {err[:500]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true",
                    help="verify stacks and launch/resume the detached 4B+9B download")
    ap.add_argument("--delete-old-models", action="store_true",
                    help="with --prepare, remove only the superseded 8B and 27B caches")
    ap.add_argument("--preflight", action="store_true",
                    help="verify every weight shard and offline-load configs/tokenizers without a GPU")
    ap.add_argument(
        "--h20-only", action="store_true",
        help="use the active H20 vLLM/toolchain/cache contract and never prepare V100 tfserve",
    )
    args = ap.parse_args()
    if args.delete_old_models and not args.prepare:
        ap.error("--delete-old-models requires --prepare")
    if args.prepare and args.preflight:
        ap.error("run --preflight after the detached download has completed")
    if args.h20_only and not (args.prepare or args.preflight):
        ap.error("--h20-only requires --prepare or --preflight")

    # Keep --help/import CPU-only on the remote deployment gate, where the base training Python
    # intentionally does not carry the controller-side Paramiko dependency.
    import remote as RM

    cli = RM.connect()
    try:
        print("connected.")
        if args.preflight:
            selected_preflight = H20_PREFLIGHT if args.h20_only else PREFLIGHT
            required = H20_PREFLIGHT_MARKERS if args.h20_only else (
                "ATTACKER_WEIGHTS_OK",
                "ATTACKER_CACHE_OK",
                "VICTIM_WEIGHTS_OK",
                "VICTIM_CACHE_OK",
            )
            rc, out, err = RM.run(cli, selected_preflight, timeout=300)
            print(out.rstrip() or err.rstrip()[:1000])
            if rc or any(marker not in out for marker in required):
                raise RuntimeError(f"offline model preflight failed rc={rc}: {err[:1000]}")
            return 0

        if not args.prepare:
            _rc, out, err = RM.run(cli, STATUS, timeout=180)
            print(out.rstrip())
            if err.strip():
                print("STDERR:", err[:500])
            return 0

        if args.delete_old_models:
            _delete_old(cli, RM)

        if not args.h20_only:
            rc, out, err = RM.run(cli, PREPARE_TFSERVE, timeout=1200)
            print(out.rstrip() or err.rstrip()[:500])
            if rc:
                raise RuntimeError("isolated Transformers Serve environment preparation failed")

        stack_check = VERIFY_H20_STACKS if args.h20_only else VERIFY_STACKS
        rc, out, err = RM.run(cli, stack_check, timeout=180)
        print(out.rstrip() or err.rstrip()[:500])
        if rc:
            label = "H20-only" if args.h20_only else "train/vLLM/Transformers-Serve"
            raise RuntimeError(f"remote {label} stack verification failed")

        rc, state, err = RM.run(
            cli,
            f"if grep -q H1_TOOLUSE_DOWNLOAD_DONE {TMP}/h1_tooluse_dl.log 2>/dev/null; then "
            "echo COMPLETE; "
            "elif pgrep -af '[d]ownload Qwen/Qwen3.5' >/dev/null; then echo RUNNING; "
            "else echo START; fi",
            timeout=30,
        )
        if rc:
            raise RuntimeError(f"could not inspect download state: {err[:500]}")
        state = state.strip()
        if state == "COMPLETE":
            selected_preflight = H20_PREFLIGHT if args.h20_only else PREFLIGHT
            markers = H20_PREFLIGHT_MARKERS if args.h20_only else (
                "ATTACKER_WEIGHTS_OK", "ATTACKER_CACHE_OK",
                "VICTIM_WEIGHTS_OK", "VICTIM_CACHE_OK",
            )
            rc, verified, verify_err = RM.run(cli, selected_preflight, timeout=300)
            if rc or any(marker not in verified for marker in markers):
                print("stale completion marker or incomplete cache; resuming downloads")
                if verify_err.strip():
                    print(verify_err.rstrip()[:500])
                state = "START"
            else:
                print("download completion marker and both caches verified")
        if state == "START":
            RM.run_bg(cli, LOCKED_DOWNLOAD, f"{TMP}/h1_tooluse_dl.log", append=True)
            print(f"download launched: {ATTACKER_HF} + {VICTIM_HF}")
        else:
            print(f"download not relaunched: state={state}")
        print(f"monitor with: python code/scripts/h1_provision.py  (log: {TMP}/h1_tooluse_dl.log)")
        return 0
    finally:
        cli.close()


if __name__ == "__main__":
    raise SystemExit(main())
