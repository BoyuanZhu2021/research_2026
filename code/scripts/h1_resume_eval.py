"""Resume the H1 verdict eval after the GPU returns (see HANDOFF.md). Run from the LOCAL repo; it
drives the H20 over SSH. Idempotent-ish: safe to re-run.

Steps:
  1. connect to the H20 (retry the flaky gateway); verify the GPU is back (else stop).
  2. ensure the vLLM 27B victim is serving (serve + poll readiness if not) — WITHOUT HF_HUB_OFFLINE.
  3. deploy the latest code (h1_deploy_mt) and re-pull the FULL sparse adapter to the local backup.
  4. launch the eval chain detached: OOD (base best-of-K + dense + sparse, same n=150 goals) +
     in-domain learning gate. Writes ood_eval/ and indomain_eval/ on the H20.
  5. print the monitor + analyze commands.

After the chain finishes (~1 h): `python code/scripts/h1_mt_powered_analyze.py --pull` for the
paired dense-sparse OOD ASR + Holm verdict; then record an EXP + power off the H20.

  python code/scripts/h1_resume_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import remote as RM  # noqa: E402

TMP = "/root/autodl-tmp"
SERVE = (f"source /etc/network_turbo; export HF_HOME={TMP}/hf_home HF_HUB_DISABLE_XET=1 "
         f"OMP_NUM_THREADS=16 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 "
         f"VLLM_ATTENTION_BACKEND=FLASH_ATTN; "
         f"/root/miniconda3/envs/vllm/bin/vllm serve Qwen/Qwen3.6-27B-FP8 --served-model-name qwen3.6-27b "
         f"--host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.45 --max-model-len 8192 "
         f"--gdn-prefill-backend triton --max-num-seqs 128 --enable-auto-tool-choice --tool-call-parser qwen3_xml")

ENV = ("OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 HF_HOME=/root/autodl-tmp/hf_home HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 VICTIM_URL=http://127.0.0.1:8000/v1 VICTIM_MODEL=qwen3.6-27b")
E = "python -u code/scripts/h1_mt_ood_eval.py --T 4 --workers 48"
IND = f"--out {TMP}/h1mt/indomain_eval"
EVAL_CHAIN = (
    f"cd {TMP}/h1mt && export {ENV}; "
    f"{E} --adapter base --split ood --n 150 --seeds 4 --tag base && "
    f"{E} --adapter runs/mt-dense-s0/adapter  --split ood --n 150 --seeds 3 --tag dense-s0 && "
    f"{E} --adapter runs/mt-sparse-s0/adapter --split ood --n 150 --seeds 3 --tag sparse-s0 && "
    f"{E} --adapter base --split indomain --n 48 --seeds 2 {IND} --tag base && "
    f"{E} --adapter runs/mt-dense-s0/adapter  --split indomain --n 48 --seeds 2 {IND} --tag dense-s0 && "
    f"{E} --adapter runs/mt-sparse-s0/adapter --split indomain --n 48 --seeds 2 {IND} --tag sparse-s0 && "
    f"echo ALL_EVAL_DONE")


def connect():
    for a in range(8):
        try:
            return RM.connect(timeout=25, retries=1)
        except Exception as e:  # noqa: BLE001
            print(f"  connect retry {a}: {type(e).__name__}"); time.sleep(12)
    return None


def main() -> int:
    cli = connect()
    if cli is None:
        print("H20 UNREACHABLE — try again later."); return 1
    print("connected.")

    rc, out, _ = RM.run(cli, "nvidia-smi -L 2>&1 | head -1", timeout=30)
    if "No devices" in out or "GPU 0" not in out and "NVIDIA" not in out:
        print(f"GPU NOT ready ({out.strip()[:60]}) — AutoDL still hasn't re-attached it. Stop.")
        cli.close(); return 1
    print(f"GPU up: {out.strip()}")

    # victim serving?
    rc, out, _ = RM.run(cli, "curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 30", timeout=20)
    if "qwen3.6-27b" not in out:
        print("victim not serving -> launching (cached load, ~4 min)...")
        RM.run_bg(cli, SERVE, f"{TMP}/h1_victim.log")
        for i in range(12):
            time.sleep(50)
            rc, out, _ = RM.run(cli, "curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 30", timeout=20)
            if "qwen3.6-27b" in out:
                print(f"  victim ready after ~{(i+1)*50}s"); break
            print(f"  [{(i+1)*50}s] not ready")
        else:
            print("victim did not come up — check /root/autodl-tmp/h1_victim.log"); cli.close(); return 1
    else:
        print("victim already serving.")

    # deploy latest code + re-pull the full sparse adapter for the local backup
    cli.close()
    print("deploying latest code (h1_deploy_mt)...")
    import subprocess
    subprocess.run([sys.executable, str(CODE / "scripts" / "h1_deploy_mt.py")], check=False)
    cli = connect()
    if cli is None:
        print("reconnect failed; re-run."); return 1
    try:
        sftp = cli.open_sftp()
        (CODE / "runs" / "h1mt_adapters" / "sparse").mkdir(parents=True, exist_ok=True)
        sftp.get(f"{TMP}/h1mt/runs/mt-sparse-s0/adapter/adapter_model.safetensors",
                 str(CODE / "runs" / "h1mt_adapters" / "sparse" / "adapter_model.safetensors"))
        sftp.close(); print("re-pulled full sparse adapter to local backup.")
    except Exception as e:  # noqa: BLE001
        print(f"  sparse re-pull skipped ({type(e).__name__}); H20 copy is authoritative.")

    RM.run(cli, f"rm -rf {TMP}/h1mt/ood_eval {TMP}/h1mt/indomain_eval; echo cleaned", timeout=20)
    RM.run_bg(cli, EVAL_CHAIN, f"{TMP}/h1mt/eval.log")
    cli.close()
    print("\n>>> eval chain launched (OOD n=150 x3 arms + in-domain gate; ~1 h).")
    print("Monitor:  grep -c ALL_EVAL_DONE /root/autodl-tmp/h1mt/eval.log   (via remote.run)")
    print("Verdict:  python code/scripts/h1_mt_powered_analyze.py --pull")
    print("Then: record EXP + power off (shutdown -h now); confirm stopped in AutoDL console.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
