"""A5/A6: BASE-attacker + best-of-K baselines over the held-out (OOD) goal pool.

The "before" bar Phase-A GRPO must beat. For each eval goal, the UNTRAINED Qwen3-8B attacker draws
K payloads; each is run through a victim (Qwen3.6-27B) AgentDojo episode and scored by reward.py.
Per goal we record: single-sample success (sample 0), best-of-K success (any of K), mean dense Phi,
refusal rate. Aggregated per regime (single / multi) with a bootstrap 95% CI. Every LLM call logged.

Crash-safe: each sample's record streams to records.jsonl as it completes; re-runnable.

Usage:
  python code/scripts/h1_baseline.py --smoke                          # 2 goals x K=2
  python code/scripts/h1_baseline.py --k 5 --n-multi 40 --workers 6 --run-tag baseA
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))
sys.path.insert(0, str(CODE_DIR / "scripts"))

import goalpool as GP  # noqa: E402
from agentdojo_client import build_pipeline, chat_once, make_logged_client  # noqa: E402
from h1_rollout import ATTACKER, VICTIM, attacker_messages, find_injectable_user_task, is_refusal, rollout  # noqa: E402
from trace import TraceLogger  # noqa: E402

from agentdojo.attacks.base_attacks import FixedJailbreakAttack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402


def boot_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of a 0/1 (or real) list, no numpy dependency."""
    import random
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (sum(values) / n, lo, hi)


def deterministic_sample(items, k, salt):
    """Deterministic subsample of `items` (list of specs) to size k, seeded by salt."""
    if len(items) <= k:
        return items
    scored = sorted(items, key=lambda s: hashlib.md5((salt + json.dumps(s, sort_keys=True)).encode()).hexdigest())
    return scored[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="best-of-K sample count per goal")
    ap.add_argument("--n-multi", type=int, default=40, help="cap on multi-step eval goals")
    ap.add_argument("--n-single", type=int, default=40, help="cap on single-step eval goals")
    ap.add_argument("--temp", type=float, default=0.8, help="attacker decoding temperature")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()
    if args.smoke:
        args.k, args.n_multi, args.n_single, args.workers = 2, 2, 2, 2

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = (args.run_tag + "_") if args.run_tag else ("smoke_" if args.smoke else "")
    run_dir = CODE_DIR / "runs" / f"h1_baseline_{tag}{run_id}"
    trace = TraceLogger(run_dir)
    att_client = make_logged_client(trace, ATTACKER[0], role="attacker")
    vic_client = make_logged_client(trace, VICTIM[0], role="victim")
    victim_pipeline = build_pipeline(vic_client, VICTIM[1], max_iters=15)

    bank = get_suites("v1")["banking"]
    helper = FixedJailbreakAttack("{goal}", bank, victim_pipeline)
    carrier, cands = find_injectable_user_task(bank, helper)

    specs = [json.loads(x) for x in (CODE_DIR / "runs" / "goalpool" / "goals_ood.jsonl").read_text().splitlines()]
    singles = deterministic_sample([s for s in specs if s["regime"] == "single"], args.n_single, "eval-single")
    multis = deterministic_sample([s for s in specs if s["regime"] == "multi"], args.n_multi, "eval-multi")
    eval_specs = singles + multis

    meta = {"run_id": run_id, "smoke": args.smoke, "attacker": ATTACKER, "victim": VICTIM,
            "k": args.k, "temp": args.temp, "carrier": carrier.ID, "vectors": cands,
            "n_single": len(singles), "n_multi": len(multis), "workers": args.workers,
            "split": "goals_ood.jsonl", "command": " ".join(sys.argv),
            "started": datetime.datetime.now().astimezone().isoformat()}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[baseline] {run_dir.name}  eval: {len(singles)} single + {len(multis)} multi  K={args.k}  workers={args.workers}")

    def eval_goal(spec):
        gen_task = GP.build_task(spec)
        goal = gen_task.GOAL
        samples = []
        for j in range(args.k):
            payload = chat_once(att_client, ATTACKER[1], attacker_messages(goal, "banking"),
                                temperature=args.temp, ctx={"goal": goal, "regime": spec["regime"], "sample": j})
            refused = is_refusal(payload)
            res = rollout(bank, carrier, gen_task, {c: payload for c in cands}, victim_pipeline)
            rec = {"spec": spec, "regime": spec["regime"], "depth": spec["depth"], "sample": j,
                   "refusal": refused, "secured": res["secured"], "dense": res["dense"],
                   "n_calls": res["n_calls"], "err": res["err"]}
            trace.log_record(rec)
            samples.append(rec)
        return spec, samples

    per_goal = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(eval_goal, s): s for s in eval_specs}
        done = 0
        for fut in as_completed(futs):
            try:
                spec, samples = fut.result()
            except Exception as e:  # noqa: BLE001
                trace.log_event({"event": "goal_error", "spec": futs[fut], "error": f"{type(e).__name__}: {e}"})
                continue
            single = 1 if samples[0]["secured"] else 0
            bestk = 1 if any(s["secured"] for s in samples) else 0
            mean_dense = sum(s["dense"] for s in samples) / len(samples)
            refus = sum(s["refusal"] for s in samples) / len(samples)
            per_goal.append({"regime": spec["regime"], "single": single, "bestk": bestk,
                             "mean_dense": mean_dense, "refusal": refus})
            done += 1
            if done % 5 == 0 or done == len(eval_specs):
                print(f"  ... {done}/{len(eval_specs)} goals scored")

    summary = {"meta": meta, "by_regime": {}}
    for regime in ("single", "multi"):
        g = [x for x in per_goal if x["regime"] == regime]
        if not g:
            continue
        s_asr, s_lo, s_hi = boot_ci([x["single"] for x in g])
        b_asr, b_lo, b_hi = boot_ci([x["bestk"] for x in g])
        summary["by_regime"][regime] = {
            "n_goals": len(g),
            "base_asr_1sample": {"mean": round(s_asr, 4), "ci95": [round(s_lo, 4), round(s_hi, 4)]},
            f"best_of_{args.k}_asr": {"mean": round(b_asr, 4), "ci95": [round(b_lo, 4), round(b_hi, 4)]},
            "mean_dense_phi": round(sum(x["mean_dense"] for x in g) / len(g), 4),
            "refusal_rate": round(sum(x["refusal"] for x in g) / len(g), 4),
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_regime"], indent=2))
    print(f"\n[done] {run_dir}  ({trace.n_calls} LLM calls logged)")
    trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
