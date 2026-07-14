"""Full-trajectory multi-turn GRPO trainer (runs ON the H20). method.md §2/§3 (PI-approved 2026-07-13).

The attacker policy (Qwen3-8B, 4-bit QLoRA) generates EVERY turn on-policy; the frozen victim
(Qwen3.6-27B) is served LOCALLY by vLLM (no API). Per step we roll out B = n_goals x G trajectories
IN LOCKSTEP (`rollout_batch`): each turn batches the active trajectories' attacker generation into one
padded `model.generate` and fires the victim replies concurrently. We score per-turn Phi, form per-turn
potential rewards (dense = ΔΦ_t / sparse = terminal), return-to-go group-relative advantages (mt_grpo),
and do one REINFORCE-with-baseline update weighting each attacker turn's tokens by its advantage, with a
KL-to-reference penalty (reference = adapter-disabled base, a la GRPO) for stability. One invocation = one arm.

  VICTIM_URL=http://127.0.0.1:8000/v1 python h1_mt_grpo_train.py --arm dense --steps 60 --seed 0

Custom loop (not TRL GRPO — that is single-turn). On-policy single update per rollout, so plain
REINFORCE (no importance ratio) + KL(pi||pi_ref) is correct.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from src.mt_rollout import rollout_batch  # noqa: E402

ATTACKER_MODEL = "Qwen/Qwen3-8B"
VICTIM_URL = os.environ.get("VICTIM_URL", "http://127.0.0.1:8000/v1")
VICTIM_MODEL = os.environ.get("VICTIM_MODEL", "qwen3.6-27b")
_FIN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)


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
                txt = json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
            m = _FIN.search(txt)
            return m.group(1).strip() if m else txt.strip()
        except (error.URLError, error.HTTPError, KeyError, TimeoutError, ValueError) as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"; time.sleep(1.5 * (a + 1))
    return f"[victim_error: {last}]"


def make_victim_batch_fn(dom, workers):
    def one(item):
        goal, conversation = item
        return victim_chat(dom.build_victim_system(goal), conversation), []

    def victim_batch_fn(items):          # items = list of (goal, conversation) -> concurrent
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, items))
    return victim_batch_fn


# ---------------------------------------------------------------- batched policy generation
def _trim_resp(resp, pad_id):
    mask = resp != pad_id
    if not bool(mask.any()):
        return resp[:0]
    last = int(mask.nonzero()[-1]) + 1
    return resp[:last]


def make_gen_batch_fn(model, tok, max_new, temperature, dev, chunk):
    """gen_batch_fn(batch_messages) -> [{"text","prompt_ids","resp_ids"}]. Left-pads within chunks of
    size `chunk` (bounds generate memory), one batched generate per chunk; prompt_ids are UNPADDED."""
    pad = tok.pad_token_id

    def tokenize(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=False)
        return (enc if isinstance(enc, torch.Tensor) else enc["input_ids"])[0]

    def gen_batch_fn(batch_messages):
        prompt_ids_list = [tokenize(m) for m in batch_messages]
        out_results = [None] * len(prompt_ids_list)
        for c0 in range(0, len(prompt_ids_list), chunk):
            sub = prompt_ids_list[c0:c0 + chunk]
            mlen = max(x.shape[0] for x in sub)
            inp = torch.full((len(sub), mlen), pad, dtype=torch.long)
            attn = torch.zeros((len(sub), mlen), dtype=torch.long)
            for k, ids in enumerate(sub):
                L = ids.shape[0]
                inp[k, mlen - L:] = ids                    # LEFT pad
                attn[k, mlen - L:] = 1
            with torch.no_grad():
                out = model.generate(inp.to(dev), attention_mask=attn.to(dev), max_new_tokens=max_new,
                                     do_sample=True, temperature=temperature, top_p=1.0, pad_token_id=pad)
            for k, ids in enumerate(sub):
                resp = _trim_resp(out[k, mlen:].detach().to("cpu"), pad)
                out_results[c0 + k] = {"text": tok.decode(resp, skip_special_tokens=True),
                                       "prompt_ids": ids.detach().to("cpu"), "resp_ids": resp}
        return out_results
    return gen_batch_fn


def token_logps(model, prompt_ids, resp_ids, dev):
    """per-token log pi(resp | prompt) under the current policy (grad follows caller's context)."""
    seq = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(dev)
    logits = model(seq).logits[0, :-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    start = prompt_ids.shape[0] - 1
    tgt = resp_ids.to(dev)
    return logp[start:start + resp_ids.shape[0]].gather(1, tgt.unsqueeze(1)).squeeze(1)   # [resp_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dense", "sparse"], required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-goals", type=int, default=8, help="goals sampled per step")
    ap.add_argument("--G", type=int, default=8, help="trajectories per goal")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--defense", default="light")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--gen-chunk", type=int, default=16, help="generate sub-batch size (bounds GPU mem)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent victim calls")
    ap.add_argument("--ckpt-every", type=int, default=30)
    ap.add_argument("--run-root", default="/root/autodl-tmp/h1mt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.smoke:
        args.steps, args.n_goals, args.G, args.n_train = 2, 2, 4, 8

    tag = args.tag or f"mt-{args.arm}-s{args.seed}"
    run_dir = Path(args.run_root) / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    prog = open(Path(args.run_root) / "progress.jsonl", "a", encoding="utf-8")
    roll_f = open(run_dir / "rollouts.jsonl", "a", encoding="utf-8")

    dom = MultiFieldExtractionDomain(K=args.K, tau=args.tau, defense_tier=args.defense)
    goals = dom.load_goals("indomain", seed=0, n=args.n_train)
    victim_batch_fn = make_victim_batch_fn(dom, args.workers)

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
    dev = next(model.parameters()).device
    gen_batch_fn = make_gen_batch_fn(model, tok, args.max_new, 1.0, dev, args.gen_chunk)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)
    print(f"[mt-grpo] arm={args.arm} steps={args.steps} n_goals={args.n_goals} G={args.G} "
          f"(B={args.n_goals*args.G}/step) T={args.T} lr={args.lr} beta_kl={args.beta_kl} "
          f"victim={VICTIM_MODEL}@{VICTIM_URL}")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        gidx = torch.randint(0, len(goals), (args.n_goals,), generator=rng).tolist()
        items, groups = [], []                       # items: goals (G each); groups: (goal, [item idxs])
        for gi in gidx:
            idxs = list(range(len(items), len(items) + args.G))
            items += [goals[gi]] * args.G
            groups.append((goals[gi], idxs))

        results = rollout_batch(dom, items, gen_batch_fn, victim_batch_fn, T=args.T, tau=args.tau)

        examples, step_maxphi, step_succ, all_rw = [], [], 0, []
        for goal, idxs in groups:
            traj_rewards = [per_turn_rewards(results[i]["phi_trace"], args.tau, args.arm) for i in idxs]
            all_rw += traj_rewards
            adv = group_advantages(traj_rewards)
            for li, i in enumerate(idxs):
                r = results[i]
                step_maxphi.append(r["max_phi"]); step_succ += int(r["success"])
                roll_f.write(json.dumps({"step": step, "arm": args.arm, "goal": goal.id,
                    "phi_trace": r["phi_trace"], "success": r["success"], "n_turns": r["n_turns"]}) + "\n")
                for t, turn in enumerate(r["turns"]):
                    a = adv[li][t] if t < len(adv[li]) else 0.0
                    if abs(a) < 1e-9:
                        continue
                    p_ids, r_ids = turn.get("prompt_ids"), turn.get("resp_ids")
                    if p_ids is not None and r_ids is not None and r_ids.shape[0] > 0:
                        examples.append((p_ids, r_ids, a))
        roll_f.flush()

        # REINFORCE-with-baseline + KL(pi||pi_ref); ref = adapter-disabled base. Grad-accumulate, one step.
        opt.zero_grad()
        N = max(1, len(examples))
        pg_val = kl_val = 0.0
        for p_ids, r_ids, a in examples:
            lp = token_logps(model, p_ids, r_ids, dev)                    # [L], grad
            with torch.no_grad(), model.disable_adapter():
                lp_ref = token_logps(model, p_ids, r_ids, dev)            # [L], reference (base)
            ratio = lp_ref - lp
            kl = (ratio.exp() - ratio - 1.0).sum()                       # k3 estimator, >= 0
            loss = (-(a) * lp.sum() + args.beta_kl * kl) / N
            loss.backward()
            pg_val += float((-(a) * lp.sum()).item()) / N
            kl_val += float((args.beta_kl * kl).item()) / N
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        B = len(items)
        rec = {"step": step, "arm": args.arm, "seed": args.seed, "pg_loss": round(pg_val, 5),
               "kl_loss": round(kl_val, 6), "grad_norm": round(float(gnorm), 4),
               "mean_max_phi": round(sum(step_maxphi) / len(step_maxphi), 4),
               "success_rate": round(step_succ / len(step_maxphi), 4), "B": B, "n_examples": len(examples),
               "frac_zero_grad": round(frac_zero_gradient(all_rw), 4), "step_time": round(time.time() - t0, 1)}
        prog.write(json.dumps(rec) + "\n"); prog.flush()
        print(f"  step {step}/{args.steps} pg={rec['pg_loss']:.4f} kl={rec['kl_loss']:.4f} "
              f"gnorm={rec['grad_norm']:.1f} meanΦ={rec['mean_max_phi']:.3f} succ={rec['success_rate']:.2f} "
              f"nex={rec['n_examples']} fzg={rec['frac_zero_grad']:.2f} {rec['step_time']:.0f}s")

        if args.ckpt_every and step % args.ckpt_every == 0 and step < args.steps:
            model.save_pretrained(str(run_dir / f"adapter_step{step}"))
            print(f"  [ckpt] adapter_step{step}")

    model.save_pretrained(str(run_dir / "adapter"))
    prog.close(); roll_f.close()
    print(f"[mt-grpo] DONE {tag} -> {run_dir}")


if __name__ == "__main__":
    main()
