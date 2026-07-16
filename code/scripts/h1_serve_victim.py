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

import os
TMP = "/root/autodl-tmp"
# Tool-use reframe (2026-07-15): lighter Qwen3.5-9B victim. Standard transformer (NOT the 3.6 hybrid
# GDN/Mamba), so the GDN/Mamba/max-num-seqs constraints drop. The victim emits ReAct text (Thought/
# Action/Action Input) parsed by OUR parse_react_calls, so vLLM native tool-calling is NOT needed.
# FlashInfer disabled (no nvcc/ninja on the box). Quant: fp8 online (fast, near-lossless on H20 for a 9B)
# — env-overridable to bf16 (drop --quantization) or a pre-quantized repo if fp8 online misbehaves.
VICTIM_HF = os.environ.get("VICTIM_HF_MODEL", "Qwen/Qwen3.5-9B")
VICTIM_NAME = os.environ.get("VICTIM_SERVED_NAME", "qwen3.5-9b")
VICTIM_QUANT = os.environ.get("VICTIM_QUANT", "fp8")   # "fp8" | "" (bf16) | "awq" | ...
_qflag = f"--quantization {VICTIM_QUANT} " if VICTIM_QUANT else ""
SERVE = (f"source /etc/network_turbo; export HF_HOME={TMP}/hf_home HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=16 "
         f"VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 "
         f"VLLM_ATTENTION_BACKEND=FLASH_ATTN; "
         f"/root/miniconda3/envs/vllm/bin/vllm serve {VICTIM_HF} "
         f"--served-model-name {VICTIM_NAME} --host 127.0.0.1 --port 8000 "
         f"--gpu-memory-utilization 0.55 --max-model-len 8192 {_qflag}--max-num-seqs 256")


def main():
    cli = RM.connect()
    # already up?
    rc, out, _ = RM.run(cli, "curl -s http://127.0.0.1:8000/v1/models || true", timeout=20)
    if VICTIM_NAME in out:
        print("victim already serving.")
        cli.close()
        return
    RM.run_bg(cli, SERVE, f"{TMP}/h1_victim.log")
    print("victim server launching (h1_victim.log); polling readiness (weights load ~1-3 min)...")
    for i in range(40):  # ~40 x 15s = 10 min
        time.sleep(15)
        rc, out, _ = RM.run(cli, "curl -s http://127.0.0.1:8000/v1/models || true", timeout=20)
        if VICTIM_NAME in out:
            print(f"VICTIM_READY after ~{(i+1)*15}s")
            cli.close()
            return
        rc, tail, _ = RM.run(cli, f"tail -1 {TMP}/h1_victim.log 2>/dev/null", timeout=20)
        print(f"  [{(i+1)*15}s] not ready: {tail.strip()[:120]}")
    print("VICTIM_TIMEOUT — check h1_victim.log")
    cli.close()


if __name__ == "__main__":
    main()
