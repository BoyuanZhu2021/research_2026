"""OOD / in-domain eval (runs ON the H20): does the dense- vs sparse-trained attacker generalize
better? H1's actual metric (OOD ASR), and the in-domain learning-gate (Step 3).

Loads one policy (base / dense-adapter / sparse-adapter) onto Qwen3-8B, runs S episodes per goal
(batched via `rollout_batch` — same fast path as training) against the local vLLM 27B victim, and
writes per-(goal, seed) rows so the paired powered analysis (`h1_mt_powered_analyze.py`) can align
arms goal-by-goal. RNG-fixed goals reproduce across processes -> all arms hit the SAME goals (paired).

  VICTIM_URL=... python h1_mt_ood_eval.py --adapter base                      --split ood --n 150 --seeds 4 --tag base
  VICTIM_URL=... python h1_mt_ood_eval.py --adapter runs/mt-dense-s0/adapter  --split ood --n 150 --seeds 1 --tag dense-s0
  VICTIM_URL=... python h1_mt_ood_eval.py --adapter runs/mt-dense-s0/adapter  --split indomain --n 40 --seeds 1 --tag gate-dense   # learning gate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))              # so `import h1_mt_grpo_train` works
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))

from src.domains.extraction_multifield import MultiFieldExtractionDomain  # noqa: E402
from src.mt_rollout import rollout_batch  # noqa: E402
from h1_mt_grpo_train import make_gen_batch_fn, make_victim_batch_fn  # noqa: E402

import os  # noqa: E402
ATTACKER_MODEL = os.environ.get("ATTACKER_MODEL", "Qwen/Qwen3.5-4B")   # tool-use reframe: lighter


def load_policy(adapter: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(ATTACKER_MODEL, quantization_config=bnb,
                                                torch_dtype=torch.bfloat16, device_map={"": 0})
    if adapter == "base":
        return base
    from peft import PeftModel
    return PeftModel.from_pretrained(base, adapter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="'base' or path to a saved LoRA adapter")
    ap.add_argument("--split", default="ood", choices=["ood", "indomain"])
    ap.add_argument("--n", type=int, default=150, help="number of goals")
    ap.add_argument("--seeds", type=int, default=1, help="episodes (samples) per goal")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--defense", default="light")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gen-chunk", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="/root/autodl-tmp/h1mt/ood_eval")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    tag = args.tag or Path(args.adapter).name

    dom = MultiFieldExtractionDomain(K=args.K, tau=args.tau, defense_tier=args.defense)
    goals = dom.load_goals(args.split, seed=0, n=args.n)     # reproducible -> paired across arms
    tok = AutoTokenizer.from_pretrained(ATTACKER_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_policy(args.adapter)
    dev = next(model.parameters()).device
    gen_batch_fn = make_gen_batch_fn(model, tok, args.max_new, args.temperature, dev, args.gen_chunk)
    victim_batch_fn = make_victim_batch_fn(dom, args.workers)

    items, meta = [], []
    for g in goals:
        for s in range(args.seeds):
            items.append(g); meta.append((g.id, s))

    t0 = time.time()
    results = rollout_batch(dom, items, gen_batch_fn, victim_batch_fn, T=args.T, tau=args.tau)
    rows = [{"goal": gid, "seed_idx": s, "success": int(r["success"]), "max_phi": r["max_phi"]}
            for (gid, s), r in zip(meta, results)]

    n_ep = len(rows) or 1
    succ = sum(r["success"] for r in rows)
    maxphis = [r["max_phi"] for r in rows]
    hist = {}
    for v in maxphis:
        hist[round(v, 1)] = hist.get(round(v, 1), 0) + 1
    summary = {"tag": tag, "adapter": args.adapter, "split": args.split, "n_goals": len(goals),
               "seeds": args.seeds, "n_episodes": len(rows), "ASR": round(succ / n_ep, 4),
               "mean_max_phi": round(sum(maxphis) / n_ep, 4), "hist": dict(sorted(hist.items())),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ood_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"ood_{tag}_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
