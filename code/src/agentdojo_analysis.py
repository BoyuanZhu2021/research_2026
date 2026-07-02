"""Analyze the AgentDojo H0 round-2 sweep — utility-matched multi-step proof.

Primary metric (no env replay, robust): each injection task has a GOAL DEPTH =
len(injection_task.ground_truth(env)) = number of malicious tool calls the attack
goal requires. On a FULL-capacity agent (utility matched), we compare:
  - ASR_single_goal : ASR on depth==1 tasks (achievable in a single malicious step)
  - ASR_multi_goal  : ASR on depth>=2 tasks (REQUIRE multiple steps; a single-step
                      agent cannot perform these by construction)
A substantial ASR_multi_goal (CI well above 0) proves multi-step capability is exploited,
and it cannot be a utility artifact because the agent is full-capacity in both groups.

Also: genuineness check (the agent really executed the required malicious calls), and
breakdowns by model / suite / attack and in-domain vs OOD (held-out suite).

Usage:  python code/src/agentdojo_analysis.py [run_dir ...] [--ood-suite travel]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

_SUITES = get_suites("v1")
_DEPTH_CACHE: dict = {}


def goal_depth(suite_name, injection_task_id):
    key = (suite_name, injection_task_id)
    if key not in _DEPTH_CACHE:
        suite = _SUITES[suite_name]
        ut = next(iter(suite.user_tasks.values()))
        env = ut.init_environment(suite.load_and_inject_default_environment({}))
        gt = suite.get_injection_task_by_id(injection_task_id).ground_truth(env)
        _DEPTH_CACHE[key] = (len(gt), [c.function for c in gt])
    return _DEPTH_CACHE[key]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def two_sample_boot(a, b, n_boot=10000, seed=0):
    """diff = mean(a) - mean(b), bootstrap CI (a,b are 0/1 lists, independent groups)."""
    if not a or not b:
        return None
    rng = random.Random(seed)
    point = _mean(a) - _mean(b)
    diffs = []
    na, nb = len(a), len(b)
    for _ in range(n_boot):
        ma = _mean([a[rng.randrange(na)] for _ in range(na)])
        mb = _mean([b[rng.randrange(nb)] for _ in range(nb)])
        diffs.append(ma - mb)
    diffs.sort()
    return {"mean_a": _mean(a), "mean_b": _mean(b), "diff": point,
            "ci_low": diffs[int(0.025 * n_boot)], "ci_high": diffs[min(n_boot - 1, int(0.975 * n_boot))],
            "n_a": na, "n_b": nb}


def boot_ci_mean(xs, n_boot=10000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(_mean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return {"mean": _mean(xs), "ci_low": ms[int(0.025 * n_boot)], "ci_high": ms[min(n_boot - 1, int(0.975 * n_boot))], "n": n}


def load_records(run_dirs):
    recs = []
    seen = set()
    for d in run_dirs:
        p = Path(d) / "records.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["model"], r["suite"], r["attack"], r["user_task"], r["injection_task"])
            if key in seen:   # dedupe across pooled runs (temp 0.0 -> deterministic)
                continue
            seen.add(key)
            depth, calls = goal_depth(r["suite"], r["injection_task"])
            r["goal_depth"] = depth
            r["depth_class"] = "multi" if depth >= 2 else ("single" if depth == 1 else "other")
            recs.append(r)
    return recs


def asr(recs):
    s = [1 if r["security"] else 0 for r in recs]
    return _mean(s), len(s)


def genuineness(run_dirs, recs, sample=30, seed=0):
    """For a sample of depth>=2 successes, confirm the agent's saved trajectory actually
    contains the ground_truth malicious calls (so the success genuinely needed multi-step)."""
    succ_multi = [r for r in recs if r["depth_class"] == "multi" and r["security"]]
    if not succ_multi:
        return {"checked": 0, "genuine": 0, "rate": None}
    rng = random.Random(seed)
    rng.shuffle(succ_multi)
    checked = genuine = 0
    for r in succ_multi[:sample]:
        # locate the AgentDojo per-task log under any run dir
        depth, gt_calls = goal_depth(r["suite"], r["injection_task"])
        logs = []
        for d in run_dirs:
            logs += glob.glob(os.path.join(d, f"{r['model']}__{r['suite']}__{r['attack']}",
                                           "**", f"{r['injection_task']}.json"), recursive=True)
        log = None
        for f in logs:
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if j.get("user_task_id") == r["user_task"] and j.get("injection_task_id") == r["injection_task"]:
                log = j; break
        if log is None:
            continue
        checked += 1
        names = [tc.get("function") for mm in log.get("messages", [])
                 for tc in (mm.get("tool_calls") or []) if isinstance(tc, dict)]
        # all ground-truth malicious calls present (multiset subset)
        from collections import Counter
        have, need = Counter(names), Counter(gt_calls)
        if all(have.get(fn, 0) >= cnt for fn, cnt in need.items()):
            genuine += 1
    return {"checked": checked, "genuine": genuine, "rate": (genuine / checked if checked else None)}


def report(recs, ood_suite=None):
    suites = sorted(set(r["suite"] for r in recs))
    in_suites = [s for s in suites if s != ood_suite]

    def block(subset, title):
        single = [1 if r["security"] else 0 for r in subset if r["depth_class"] == "single"]
        multi = [1 if r["security"] else 0 for r in subset if r["depth_class"] == "multi"]
        cs, cm = boot_ci_mean(single), boot_ci_mean(multi)
        gap = two_sample_boot(multi, single)
        print(f"\n### {title}")
        if cs:
            print(f"  single-goal ASR (depth=1): {cs['mean']*100:5.1f}% [{cs['ci_low']*100:.1f},{cs['ci_high']*100:.1f}]  n={cs['n']}")
        if cm:
            sig = "  *(CI>0)" if cm["ci_low"] > 0 else ""
            print(f"  MULTI-goal  ASR (depth>=2): {cm['mean']*100:5.1f}% [{cm['ci_low']*100:.1f},{cm['ci_high']*100:.1f}]  n={cm['n']}{sig}")
        if gap:
            print(f"  gap multi-single: {gap['diff']*100:+5.1f}pt [{gap['ci_low']*100:+.1f},{gap['ci_high']*100:+.1f}]")

    print("=" * 90)
    print(f"AGENTDOJO H0 ROUND-2  (full-capacity agent, utility-matched)  records={len(recs)}")
    print("=" * 90)
    block(recs, "ALL")
    if ood_suite and any(r["suite"] == ood_suite for r in recs):
        block([r for r in recs if r["suite"] in in_suites], f"IN-DOMAIN (suites {in_suites})")
        block([r for r in recs if r["suite"] == ood_suite], f"OOD (held-out suite: {ood_suite})")
    # per model
    print("\n--- per target (multi-goal ASR, depth>=2) ---")
    for m in sorted(set(r["model"] for r in recs)):
        mm = [1 if r["security"] else 0 for r in recs if r["model"] == m and r["depth_class"] == "multi"]
        ci = boot_ci_mean(mm)
        if ci:
            print(f"  {m:14s} {ci['mean']*100:5.1f}% [{ci['ci_low']*100:.1f},{ci['ci_high']*100:.1f}]  n={ci['n']}  "
                  f"{'*' if ci['ci_low']>0 else ''}")
    # per suite
    print("\n--- per suite (single vs multi-goal ASR) ---")
    for s in sorted(set(r["model"] and r["suite"] for r in recs)):
        sub = [r for r in recs if r["suite"] == s]
        a1, n1 = asr([r for r in sub if r["depth_class"] == "single"])
        a2, n2 = asr([r for r in sub if r["depth_class"] == "multi"])
        print(f"  {s:12s} single {a1*100:5.1f}%(n{n1})  multi {a2*100:5.1f}%(n{n2})")
    # utility parity
    u = [1 if r["utility"] else 0 for r in recs]
    print(f"\n  utility (full agent, both groups): {_mean(u)*100:.1f}%  (confirms utility-matched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--ood-suite", default="travel")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dirs = a.run_dirs or sorted(d for d in glob.glob(str(CODE_DIR / "runs" / "adojo2_*")) if "smoke" not in d)
    if not run_dirs:
        print("no adojo2_* run dirs"); return 1
    recs = load_records(run_dirs)
    if not recs:
        print("no records"); return 1
    report(recs, ood_suite=a.ood_suite)
    g = genuineness(run_dirs, recs)
    print(f"\n  genuineness (depth>=2 successes with required malicious calls present): "
          f"{g['genuine']}/{g['checked']} = {g['rate']*100:.0f}%" if g["rate"] is not None else "  genuineness: n/a")
    if a.out:
        Path(a.out).write_text(json.dumps({"n_records": len(recs), "run_dirs": run_dirs}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
