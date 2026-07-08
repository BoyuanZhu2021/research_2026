"""Sharp Phase-B eval: dense vs sparse head-to-head on DEEP (depth>=min) held-out OOD goals.

Evaluates BASE (K samples), a dense-trained adapter, and a sparse-trained adapter on the SAME deep
OOD goals via the local vLLM victim, and reports the paired dense-vs-sparse difference in ASR and in
mean potential Phi (the H1-relevant contrast) with bootstrap CIs. This is where dense's partial-credit
advantage should be largest; if dense still ties sparse here, that is strong evidence against H1.

  python h1_eval_deep.py --min-depth 4 --n 40 --k 5 \
      --dense-tag M-dense-deep-s0 --sparse-tag M-sparse-deep-s0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import goalpool as GP  # noqa: E402
from agentdojo_client import build_pipeline, make_client  # noqa: E402
from h1_rollout import attacker_messages, is_refusal, rollout  # noqa: E402
from trace import TraceLogger  # noqa: E402
from agentdojo.attacks.base_attacks import FixedJailbreakAttack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-8B"
VICTIM_URL, VICTIM_MODEL = "http://127.0.0.1:8000/v1", "qwen3.6-27b"
RUNS = Path("/root/autodl-tmp/h1/runs")
G = {}


def dsample(items, k, salt):
    if len(items) <= k:
        return items
    return sorted(items, key=lambda s: hashlib.md5((salt + json.dumps(s, sort_keys=True)).encode()).hexdigest())[:k]


def load_attacker(adapter):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                             torch_dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m


def gen(model, tok, goal, suite, k, temp):
    prompt = tok.apply_chat_template(attacker_messages(goal, suite), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=(temp > 0), temperature=(temp or 1.0), top_p=0.95,
                             max_new_tokens=220, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def eval_arm(model, tok, specs, k, temp, name, trace):
    rows = []
    for i, g in enumerate(specs):
        task = GP.build_task(g)
        res = [rollout(G["suite"], G["carrier"], task, {c: p for c in G["cands"]}, G["victim"])
               for p in gen(model, tok, task.GOAL, g["suite"], k, temp)]
        sec = [bool(x["secured"]) for x in res]
        phi = [x["dense"] for x in res]
        rows.append({"depth": g["depth"], "single": int(sec[0]), "bestk": int(any(sec)),
                     "phi1": phi[0], "phi_max": max(phi), "mean_phi": sum(phi) / len(phi)})
        trace.log_record({"kind": "eval_deep", "arm": name, "depth": g["depth"],
                          "secured": sec[0], "phi": phi[0]})
        if i % 10 == 0:
            print(f"  [{name}] {i+1}/{len(specs)}")
    return rows


def paired_ci(a, b, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    d.sort()
    return round(sum(a) / n - sum(b) / n, 4), round(d[int(0.025 * n_boot)], 4), round(d[int(0.975 * n_boot)], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-depth", type=int, default=4)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--dense-tag", default="M-dense-deep-s0")
    ap.add_argument("--sparse-tag", default="M-sparse-deep-s0")
    args = ap.parse_args()

    trace = TraceLogger(RUNS / "eval_deep")
    G["victim"] = build_pipeline(make_client(trace, VICTIM_URL, "EMPTY", role="victim", vllm_local=True),
                                 VICTIM_MODEL, max_iters=15)
    suite = get_suites("v1")["banking"]
    helper = FixedJailbreakAttack("{goal}", suite, G["victim"])
    G["suite"] = suite
    G["carrier"] = next(ut for ut in suite.user_tasks.values() if _inj(helper, ut))
    G["cands"] = list(helper.get_injection_candidates(G["carrier"]))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ood = [json.loads(x) for x in (HERE / "goals_ood.jsonl").read_text().splitlines()]
    deep = dsample([g for g in ood if g["regime"] == "multi" and g["depth"] >= args.min_depth], args.n, "deep")
    print(f"[eval-deep] {len(deep)} OOD goals depth>={args.min_depth}  (depths {sorted(set(g['depth'] for g in deep))})")

    print("== BASE ==")
    base = load_attacker(None)
    b = eval_arm(base, tok, deep, args.k, args.temp, "base", trace)
    del base
    torch.cuda.empty_cache()
    print("== DENSE ==")
    md = load_attacker(str(RUNS / args.dense_tag / "adapter"))
    d = eval_arm(md, tok, deep, 1, 0.0, "dense", trace)
    del md
    torch.cuda.empty_cache()
    print("== SPARSE ==")
    ms = load_attacker(str(RUNS / args.sparse_tag / "adapter"))
    s = eval_arm(ms, tok, deep, 1, 0.0, "sparse", trace)
    del ms
    torch.cuda.empty_cache()

    def asr(rows, f):
        return round(sum(r[f] for r in rows) / len(rows), 4)

    dense_sec = [r["single"] for r in d]
    sparse_sec = [r["single"] for r in s]
    dense_phi = [r["phi1"] for r in d]
    sparse_phi = [r["phi1"] for r in s]
    summary = {
        "n": len(deep), "min_depth": args.min_depth,
        "base": {"asr_1s": asr(b, "single"), "asr_bestk": asr(b, "bestk"), "mean_phi": asr(b, "mean_phi")},
        "dense": {"asr": asr(d, "single"), "mean_phi": asr(d, "phi1")},
        "sparse": {"asr": asr(s, "single"), "mean_phi": asr(s, "phi1")},
        "dense_minus_sparse_ASR": paired_ci(dense_sec, sparse_sec),
        "dense_minus_sparse_PHI": paired_ci(dense_phi, sparse_phi),
        "dense_minus_base_bestk": paired_ci(dense_sec, [r["bestk"] for r in b]),
        "sparse_minus_base_bestk": paired_ci(sparse_sec, [r["bestk"] for r in b]),
    }
    out = RUNS / "eval_deep" / "phaseB_deep_eval.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== SHARP PHASE B (deep dense-vs-sparse) =====")
    print(json.dumps(summary, indent=2))
    trace.close()


def _inj(helper, ut):
    try:
        return bool(helper.get_injection_candidates(ut))
    except Exception:
        return False


if __name__ == "__main__":
    main()
