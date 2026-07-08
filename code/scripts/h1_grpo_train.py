"""H1 Phase-A GRPO trainer (runs ON the H20). One invocation = one cell (regime x arm).

Attacker: Qwen3-8B, 4-bit NF4 + LoRA, TRL GRPOTrainer. For each sampled payload, we run the FULL
AgentDojo rollout against the LOCAL vLLM victim (Qwen3.6-27B-FP8 @ localhost:8000) and score with
reward.py (sparse=security / dense=potential Phi). Every attacker payload + reward and every victim
call are logged.

  python h1_grpo_train.py --regime single --arm sparse --steps 15 --seed 0 --tag smoke
  python h1_grpo_train.py --regime multi  --arm dense  --steps 200 --seed 0 --tag M-dense-s0

Cells: S-sparse (single/sparse), M-dense (multi/dense), M-sparse (multi/sparse).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # our modules sit next to this script on the remote

import goalpool as GP  # noqa: E402
import reward as R  # noqa: E402
from agentdojo_client import build_pipeline, make_client  # noqa: E402
from h1_rollout import ATTACK_SYS, attacker_messages, is_refusal, rollout  # noqa: E402
from trace import TraceLogger  # noqa: E402

from agentdojo.attacks.base_attacks import FixedJailbreakAttack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

ATTACKER_MODEL = "Qwen/Qwen3-8B"
VICTIM_URL = "http://127.0.0.1:8000/v1"
VICTIM_MODEL = "qwen3.6-27b"

# globals wired in main() and read by the reward function
G = {}


def reward_fn(completions, **kwargs):
    """GRPO reward: run each sampled payload through a victim rollout, return sparse|dense reward."""
    specs = kwargs["spec"]
    payloads = [(c[0]["content"] if isinstance(c, list) else c) or "" for c in completions]

    def score(payload, spec_json):
        spec = json.loads(spec_json)
        task = GP.build_task(spec)
        res = rollout(G["suite"], G["carrier"], task, {c: payload for c in G["cands"]}, G["victim"])
        rwd = res["dense"] if G["arm"] == "dense" else res["sparse"]
        G["trace"].log_record({"kind": "train_rollout", "arm": G["arm"], "regime": spec["regime"],
                               "depth": spec["depth"], "reward": rwd, "secured": res["secured"],
                               "n_calls": res["n_calls"], "refusal": is_refusal(payload),
                               "err": res["err"], "payload": payload[:400]})
        return float(rwd)

    with ThreadPoolExecutor(max_workers=G["workers"]) as ex:
        return list(ex.map(lambda a: score(*a), zip(payloads, specs)))


class ProgressCallback(TrainerCallback):
    """Stream per-log-step metrics to progress.jsonl for the live dashboard."""
    def __init__(self, path, arm, regime, seed):
        self.path, self.meta = path, {"arm": arm, "regime": regime, "seed": seed}

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        rec = {"step": state.global_step, **self.meta,
               **{k: round(v, 5) for k, v in logs.items() if isinstance(v, (int, float))}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def build_dataset(regime: str, tokenizer, n_smoke: int | None, min_depth: int = 1):
    rows = [json.loads(x) for x in (HERE / "goals_train.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["regime"] == regime and r["depth"] >= min_depth]
    if n_smoke:
        rows = rows[:n_smoke]
    data = []
    for r in rows:
        goal = GP.build_task(r).GOAL
        data.append({"prompt": attacker_messages(goal, r["suite"]), "spec": json.dumps(r)})
    return Dataset.from_list(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["single", "multi"], required=True)
    ap.add_argument("--arm", choices=["sparse", "dense"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-gen", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--min-depth", type=int, default=1, help="train only on goals with depth >= this")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"{args.regime}-{args.arm}-s{args.seed}"
    run_root = Path("/root/autodl-tmp/h1")
    run_dir = run_root / "runs" / tag
    trace = TraceLogger(run_dir)
    victim = build_pipeline(make_client(trace, VICTIM_URL, "EMPTY", role="victim", vllm_local=True),
                            VICTIM_MODEL, max_iters=15)

    suite = get_suites("v1")["banking"]
    helper = FixedJailbreakAttack("{goal}", suite, victim)
    carrier = next(ut for ut in suite.user_tasks.values()
                   if _injectable(helper, ut))
    cands = list(helper.get_injection_candidates(carrier))
    G.update(arm=args.arm, suite=suite, carrier=carrier, cands=cands, victim=victim,
             trace=trace, workers=args.workers)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    ds = build_dataset(args.regime, tok, n_smoke=8 if args.smoke else None, min_depth=args.min_depth)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = GRPOConfig(
        output_dir=str(run_dir / "ckpt"),
        use_vllm=False, num_generations=args.num_gen,
        per_device_train_batch_size=args.num_gen, gradient_accumulation_steps=4,
        num_iterations=1, max_completion_length=512,
        temperature=1.0, beta=0.0, scale_rewards=False, loss_type="dr_grpo",
        learning_rate=1e-5, max_steps=args.steps, logging_steps=1,
        save_steps=50, save_total_limit=3,  # intermediate LoRA checkpoints (crash resilience)
        bf16=True, report_to="none", seed=args.seed,
        chat_template_kwargs={"enable_thinking": False},  # Qwen3: no CoT in attacker payloads
    )
    model = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    trainer = GRPOTrainer(
        model=model, args=cfg, peft_config=lora, reward_funcs=reward_fn, train_dataset=ds, processing_class=tok,
    )
    trainer.add_callback(ProgressCallback(str(run_root / "progress.jsonl"), args.arm, args.regime, args.seed))
    print(f"[grpo] cell={tag} regime={args.regime} arm={args.arm} steps={args.steps} "
          f"n_train={len(ds)} G={args.num_gen} carrier={carrier.ID} vectors={cands}")
    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    trace.close()
    print(f"[grpo] DONE cell={tag} -> {run_dir}")


def _injectable(helper, ut) -> bool:
    try:
        return bool(helper.get_injection_candidates(ut))
    except Exception:
        return False


if __name__ == "__main__":
    main()
