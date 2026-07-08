"""Per-step first-success-step analysis for the AgentDojo H0 sweep.

Replays each saved trajectory step by step against a fresh injected env and calls the
same injection-security oracle AgentDojo uses (`suite._check_injection_task_security`,
which tries `security_from_traces` then `security`) after each tool call, to find the
FIRST agent step at which the attack succeeds.

Yields the cleanest utility-matched single-vs-multi-step contrast on the SAME tasks:
  single_step_ASR = P(success by step 1)   (expected ~0: injection lives in a tool result,
                                            so the agent must fetch it before it can act)
  multi_step_ASR  = P(success by step K)   (the full ASR)

Also reports replay fidelity (does a full replay reproduce the recorded verdict) so
incomplete saved logs are accounted for, not silently trusted.

Usage:  python code/scripts/agentdojo_perstep.py [run_glob] [--out path.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

SUITES = get_suites("v1")
_DEPTH: dict = {}


def depth_of(sname, iid):
    k = (sname, iid)
    if k not in _DEPTH:
        s = SUITES[sname]
        ut = next(iter(s.user_tasks.values()))
        env = ut.init_environment(s.load_and_inject_default_environment({}))
        _DEPTH[k] = len(s.get_injection_task_by_id(iid).ground_truth(env))
    return _DEPTH[k]


def calls_from(messages):
    out = []
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            a = tc.get("args")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            out.append(FunctionCall(function=tc.get("function"), args=a or {}, id=tc.get("id") or "x"))
    return out


def analyze(run_dirs):
    recs = []
    for d in run_dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(j, dict):
                continue
            uid, iid, sname = j.get("user_task_id"), j.get("injection_task_id"), j.get("suite_name")
            if not (uid and iid and sname):
                continue
            suite = SUITES[sname]
            ut = suite.get_user_task_by_id(uid)
            it = suite.get_injection_task_by_id(iid)
            env = ut.init_environment(suite.load_and_inject_default_environment(j.get("injections") or {}))
            pre = env.model_copy(deep=True)
            calls = calls_from(j.get("messages", []))
            rt = FunctionsRuntime(suite.tools)
            fss = None
            for i, c in enumerate(calls, 1):
                try:
                    rt.run_function(env, c.function, c.args)
                except Exception:
                    pass
                try:
                    if suite._check_injection_task_security(it, [], pre, env.model_copy(deep=True), calls[:i]) and fss is None:
                        fss = i
                except Exception:
                    pass
            try:
                replay_final = bool(suite._check_injection_task_security(it, [], pre, env, calls))
            except Exception:
                replay_final = None
            recs.append({"model": os.path.basename(d).replace("adojo2_v4_", "").rsplit("_", 1)[0],
                         "suite": sname, "depth": depth_of(sname, iid), "rec_sec": bool(j.get("security")),
                         "replay_final": replay_final, "n_calls": len(calls), "first_success_step": fss})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_glob", nargs="?", default=str(CODE_DIR / "runs" / "adojo2_v4_*"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dirs = sorted(glob.glob(a.run_glob))
    recs = analyze(run_dirs)
    n = len(recs)
    faithful = [r for r in recs if r["replay_final"] == r["rec_sec"]]
    succ = [r for r in recs if r["rec_sec"]]
    repro = [r for r in succ if r["replay_final"] and r["first_success_step"]]
    single = sum(1 for r in faithful if r["rec_sec"] and r["first_success_step"] == 1)
    multi = sum(1 for r in faithful if r["rec_sec"])
    dist = dict(sorted(Counter(r["first_success_step"] for r in repro).items()))
    summary = {
        "n_tasks": n, "replay_faithful": len(faithful), "replay_faithful_pct": round(len(faithful) / n * 100, 1) if n else None,
        "successes": len(succ), "successes_reproduced": sum(1 for r in succ if r["replay_final"]),
        "first_success_step_dist": dist, "successes_at_step1": sum(1 for r in repro if r["first_success_step"] == 1),
        "single_step_asr": round(single / len(faithful), 4) if faithful else None,
        "multi_step_asr": round(multi / len(faithful), 4) if faithful else None,
        "gap_pt": round((multi - single) / len(faithful) * 100, 1) if faithful else None,
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
