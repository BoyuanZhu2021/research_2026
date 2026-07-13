"""Full-trajectory multi-turn GRPO trainer (runs ON the H20). method.md §2 (PI-approved 2026-07-13).

The attacker policy (Qwen3-8B, 4-bit QLoRA) generates EVERY turn on-policy; the frozen victim
(Qwen3.6-27B) is served LOCALLY by vLLM (no API). Per goal we roll out G trajectories, score per-turn
Phi with the rule oracle, form per-turn potential rewards (dense = ΔΦ_t / sparse = terminal) and
return-to-go group-relative advantages (mt_grpo), then do one REINFORCE-with-baseline update weighting
each attacker turn's tokens by its advantage. One invocation = one arm.

  VICTIM_URL=http://127.0.0.1:8000/v1 python h1_mt_grpo_train.py --arm dense --steps 25 --seed 0

Custom loop (not TRL GRPO — that is single-turn). On-policy single update per rollout, so plain
REINFORCE (no importance ratio) is correct.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))       # so `import src...` works
sys.path.insert(0, str(HERE.parent / "src"))

from src.domains.extraction_multifield import MultiFieldExtractionDomain  # noqa: E402
from src.mt_grpo import frac_zero_gradient, group_advantages, per_turn_rewards  # noqa: E402
from src.mt_rollout import rollout_trajectory  # noqa: E402

ATTACKER_MODEL = "Qwen/Qwen3-8B"
VICTIM_URL = os.environ.get("VICTIM_URL", "http://127.0.0.1:8000/v1")
VICTIM_MODEL = os.environ.get("VICTIM_MODEL", "qwen3.6-27b")


# ---------------------------------------------------------------- victim (local vLLM, no API)
def victim_chat(system: str, user_conversation: list[dict], *, max_tokens=512, temperature=0.7,
                retries=4) -> str:
    messages = [{"role": "system", "content": system}, *user_conversation]
    body = json.dumps({"model": VICTIM_MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    last = ""
    for a in range(retries):
        try:
            req = request.Request(f"{VICTIM_URL}/chat/completions", data=body, method="POST",
                                  headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"})
            with request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
        except (error.URLError, error.HTTPError, KeyError, TimeoutError, ValueError) as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"; time.sleep(1.5 * (a + 1))
    return f"[victim_error: {last}]"


def make_victim_fn(dom):
    import re
    fin = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)

    def victim_fn(goal, conversation):
        reply = victim_chat(dom.build_victim_system(goal), conversation)
        m = fin.search(reply)
        return (m.group(1).strip() if m else reply.strip()), []
    return victim_fn


# ---------------------------------------------------------------- policy generation + loss
def make_gen_fn(model, tok, records, max_new_tokens, temperature):
    def gen_fn(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=1.0, pad_token_id=tok.pad_token_id)
        resp = out[0, ids.shape[1]:]
        records.append((ids[0].detach().to("cpu"), resp.detach().to("cpu")))
        return tok.decode(resp, skip_special_tokens=True)
    return gen_fn


def turn_logprob(model, prompt_ids, resp_ids):
    """sum log pi(resp | prompt) under the current policy (grad-enabled)."""
    seq = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(model.device)
    logits = model(seq).logits[0, :-1]                     # predict token t+1 from t
    logp = F.log_softmax(logits.float(), dim=-1)
    start = prompt_ids.shape[0] - 1
    tgt = resp_ids.to(model.device)
    sel = logp[start:start + resp_ids.shape[0]].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return sel.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dense", "sparse"], required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-goals", type=int, default=2, help="goals sampled per step")
    ap.add_argument("--G", type=int, default=6, help="trajectories per goal")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--defense", default="light")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--run-root", default="/root/autodl-tmp/h1mt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.smoke:
        args.steps, args.n_goals, args.G, args.n_train = 2, 1, 3, 6

    tag = args.tag or f"mt-{args.arm}-s{args.seed}"
    run_dir = Path(args.run_root) / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = open(Path(args.run_root) / "progress.jsonl", "a", encoding="utf-8")
    roll_f = open(run_dir / "rollouts.jsonl", "a", encoding="utf-8")

    dom = MultiFieldExtractionDomain(K=args.K, tau=args.tau, defense_tier=args.defense)
    goals = dom.load_goals("indomain", seed=0, n=args.n_train)
    victim_fn = make_victim_fn(dom)

    tok = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(ATTACKER_MODEL, quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16, device_map={"": 0})
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)
    print(f"[mt-grpo] arm={args.arm} steps={args.steps} n_goals={args.n_goals} G={args.G} T={args.T} "
          f"tau={args.tau} victim={VICTIM_MODEL}@{VICTIM_URL}")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        gidx = torch.randint(0, len(goals), (args.n_goals,), generator=rng).tolist()
        examples = []                       # (prompt_ids, resp_ids, advantage)
        step_maxphi, step_succ, all_rw = [], 0, []
        for gi in gidx:
            goal = goals[gi]
            traj_turns, traj_rewards = [], []
            for i in range(args.G):
                recs: list = []
                gen_fn = make_gen_fn(model, tok, recs, args.max_new, temperature=1.0)
                r = rollout_trajectory(dom, goal, gen_fn, victim_fn, T=args.T, tau=args.tau)
                traj_turns.append(recs)                       # per-turn (prompt_ids, resp_ids)
                traj_rewards.append(per_turn_rewards(r["phi_trace"], args.tau, args.arm))
                step_maxphi.append(r["max_phi"]); step_succ += int(r["success"])
                roll_f.write(json.dumps({"step": step, "arm": args.arm, "goal": goal.id,
                    "phi_trace": r["phi_trace"], "success": r["success"], "n_turns": r["n_turns"]}) + "\n")
            all_rw += traj_rewards
            adv = group_advantages(traj_rewards)              # A[i][t]
            for i in range(len(traj_turns)):
                for t in range(len(traj_turns[i])):
                    a = adv[i][t] if t < len(adv[i]) else 0.0
                    if abs(a) < 1e-9:
                        continue
                    p_ids, r_ids = traj_turns[i][t]
                    if r_ids.shape[0] > 0:
                        examples.append((p_ids, r_ids, a))
        roll_f.flush()

        # REINFORCE update: grad-accumulate over turn examples, one optimizer step
        opt.zero_grad()
        loss_val = 0.0
        for p_ids, r_ids, a in examples:
            lp = turn_logprob(model, p_ids, r_ids)
            loss = -(a / max(1, len(examples))) * lp
            loss.backward()
            loss_val += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        rec = {"step": step, "arm": args.arm, "seed": args.seed, "loss": round(loss_val, 5),
               "grad_norm": round(float(gnorm), 4), "mean_max_phi": round(sum(step_maxphi) / len(step_maxphi), 4),
               "success_rate": round(step_succ / len(step_maxphi), 4), "n_examples": len(examples),
               "frac_zero_grad": round(frac_zero_gradient(all_rw), 4), "step_time": round(time.time() - t0, 1)}
        prog.write(json.dumps(rec) + "\n"); prog.flush()
        print(f"  step {step}/{args.steps} loss={rec['loss']:.4f} meanΦ={rec['mean_max_phi']:.3f} "
              f"succ={rec['success_rate']:.2f} nex={rec['n_examples']} fzg={rec['frac_zero_grad']:.2f} "
              f"{rec['step_time']:.0f}s")

    model.save_pretrained(str(run_dir / "adapter"))
    prog.close(); roll_f.close()
    print(f"[mt-grpo] DONE {tag} -> {run_dir}")


if __name__ == "__main__":
    main()
