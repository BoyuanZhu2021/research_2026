"""Start the local vLLM victim server (Qwen3.6-27B-FP8) on the H20, detached, and wait until ready.

Serves on 127.0.0.1:8000 (OpenAI-compatible), capped to ~45% VRAM so the GRPO trainer has room.
Run after provisioning (model downloaded + vllm env ready). Idempotent-ish: if already serving,
the readiness poll just succeeds.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import remote as RM  # noqa: E402

TMP = "/root/autodl-tmp"
# The AutoDL container has no CUDA compiler (no nvcc/ninja), so every FlashInfer JIT path fails.
# Disable them and use vLLM's precompiled kernels: FLASH_ATTN attention, native sampler, cutlass FP8,
# triton GDN prefill. This is the reliable config for this box.
SERVE = (f"source /etc/network_turbo; export HF_HOME={TMP}/hf_home OMP_NUM_THREADS=16 "
         f"VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 "
         f"VLLM_ATTENTION_BACKEND=FLASH_ATTN; "
         f"/root/miniconda3/envs/vllm/bin/vllm serve Qwen/Qwen3.6-27B-FP8 "
         f"--served-model-name qwen3.6-27b --host 127.0.0.1 --port 8000 "
         f"--gpu-memory-utilization 0.45 --max-model-len 8192 --gdn-prefill-backend triton "
         f"--max-num-seqs 128 "  # Qwen3.6 hybrid: cap decode seqs to <= Mamba cache blocks (192 @ 0.45 util)
         f"--enable-auto-tool-choice --tool-call-parser qwen3_xml")  # Qwen3.6 emits <function=..><parameter=..> XML


def main():
    cli = RM.connect()
    # already up?
    rc, out, _ = RM.run(cli, "curl -s http://127.0.0.1:8000/v1/models || true", timeout=20)
    if "qwen3.6-27b" in out:
        print("victim already serving.")
        cli.close()
        return
    RM.run_bg(cli, SERVE, f"{TMP}/h1_victim.log")
    print("victim server launching (h1_victim.log); polling readiness (weights load ~1-3 min)...")
    for i in range(40):  # ~40 x 15s = 10 min
        time.sleep(15)
        rc, out, _ = RM.run(cli, "curl -s http://127.0.0.1:8000/v1/models || true", timeout=20)
        if "qwen3.6-27b" in out:
            print(f"VICTIM_READY after ~{(i+1)*15}s")
            cli.close()
            return
        rc, tail, _ = RM.run(cli, f"tail -1 {TMP}/h1_victim.log 2>/dev/null", timeout=20)
        print(f"  [{(i+1)*15}s] not ready: {tail.strip()[:120]}")
    print("VICTIM_TIMEOUT — check h1_victim.log")
    cli.close()


if __name__ == "__main__":
    main()
