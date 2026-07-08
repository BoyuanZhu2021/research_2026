"""Provision the H20 for H1 Phase-A (run once GPU is attached). Kicks off 3 background jobs:
  1. model downloads  -> h1_dl.log        (Qwen3.6-27B-FP8 victim + Qwen3-8B attacker)
  2. training stack    -> h1_pip_train.log (trl/peft/bitsandbytes/accelerate/datasets in base env)
  3. vllm env          -> h1_vllm.log      (separate conda env, decoupled from trl's deps)

Architecture: victim served by vLLM (separate env) on localhost:8000; attacker trained by TRL in
base env (torch 2.8 already present); rollouts hit the LOCAL victim. Decoupling avoids vllm<->trl
dependency conflicts. Poll the logs with h1_remote_status.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import remote as RM  # noqa: E402

TMP = "/root/autodl-tmp"
HFENV = f"source /etc/network_turbo; export HF_HOME={TMP}/hf_home HF_HUB_ENABLE_HF_TRANSFER=1"

DL = (f"{HFENV}; "
      f"huggingface-cli download Qwen/Qwen3.6-27B-FP8 && "
      f"huggingface-cli download Qwen/Qwen3-8B && echo H1_DOWNLOAD_DONE")

TRAIN_STACK = ("python -m pip install -q -U trl peft bitsandbytes accelerate datasets && "
               "python -c \"import trl,peft,bitsandbytes,accelerate,transformers as t;"
               "print('H1_TRAIN_STACK_OK trl',trl.__version__,'tf',t.__version__)\"")

VLLM_ENV = ("conda create -y -n vllm python=3.12 && "
            "/root/miniconda3/envs/vllm/bin/pip install -q -U vllm && "
            "/root/miniconda3/envs/vllm/bin/python -c \"import vllm;print('H1_VLLM_OK',vllm.__version__)\"")


def main():
    cli = RM.connect()
    print("connected.")
    # hf CLI + fast transfer first (needed by the download job)
    rc, out, err = RM.run(cli, "python -m pip install -q -U 'huggingface_hub[cli]' hf_transfer && "
                               "huggingface-cli version", timeout=300)
    print("hf cli:", out.strip() or err.strip()[:200])
    # kick off the three long jobs, detached
    RM.run_bg(cli, DL, f"{TMP}/h1_dl.log")
    RM.run_bg(cli, TRAIN_STACK, f"{TMP}/h1_pip_train.log")
    RM.run_bg(cli, VLLM_ENV, f"{TMP}/h1_vllm.log")
    print("launched: downloads -> h1_dl.log | train stack -> h1_pip_train.log | vllm env -> h1_vllm.log")
    cli.close()


if __name__ == "__main__":
    main()
