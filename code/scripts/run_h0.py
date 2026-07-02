"""H0 runner (confound-aware, per the harness audit).

For each target x arm x seed x OOD goal x sample, run an episode (fully logged,
crash-safe records), then compute the confound-aware analysis per target:
  - per-arm ASR-all / ASR-valid / ASR-delivered / delivery rate
  - best-of-K single control (attempt-matched to multi)
  - contrasts: multi vs single, multi vs best-of-K, each all/delivered
  - PRIMARY H0 verdict = multi vs best-of-K among delivered (attempt + delivery matched)
  - first-success-turn distribution, per-attack-type, tool-level bootstrap

Examples:
  python code/scripts/run_h0.py --smoke
  python code/scripts/run_h0.py --n-ood-goals 30 --targets qwen2.5-7b-instruct,qwen3.6-27b
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from src import analysis                              # noqa: E402
from src.domains.injecagent import InjecAgentDomain   # noqa: E402
from src.episode import run_episode                   # noqa: E402
from src.providers import PROJECT_ROOT, mask, resolve_provider  # noqa: E402
from src.trace import LoggedClient, TraceLogger       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CODE_DIR / "configs" / "h0_pilot.json"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--setting", default="base", choices=["base", "enhanced"])
    p.add_argument("--n-ood-goals", type=int, default=None)
    p.add_argument("--n-single", type=int, default=None)
    p.add_argument("--n-multi", type=int, default=None)
    p.add_argument("--best-of-k", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--targets", default=None)
    p.add_argument("--max-workers", type=int, default=16)
    return p.parse_args()


def git_sha():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "no-git"


def pct(x):
    return "  n/a " if x is None else f"{x * 100:5.1f}%"


def fmt_contrast(c):
    if not c:
        return "(no paired units)"
    star = " *" if c["excludes_zero"] else "  "
    return (f"{c['mean_a'] * 100:5.1f}% vs {c['mean_b'] * 100:5.1f}%  "
            f"diff {c['diff'] * 100:+5.1f} [{c['ci_low'] * 100:+5.1f},{c['ci_high'] * 100:+5.1f}]pt{star} "
            f"(n={c['n_units']})")


def main() -> int:
    a = parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))

    if a.smoke:
        cfg.update(seeds=[0], n_samples_single=3, n_samples_multi=2, best_of_k=3, max_turns=3)
        cfg["n_ood_goals"] = 4
        cfg["targets"] = cfg["targets"][:1]
    for flag, key in [(a.seeds, "seeds"), (a.n_single, "n_samples_single"), (a.n_multi, "n_samples_multi"),
                      (a.best_of_k, "best_of_k"), (a.max_turns, "max_turns"), (a.n_ood_goals, "n_ood_goals")]:
        if flag is not None:
            cfg[key] = [int(x) for x in flag.split(",")] if key == "seeds" else int(flag)
    if a.targets:
        keep = set(a.targets.split(","))
        cfg["targets"] = [t for t in cfg["targets"] if t["name"] in keep]
    k = cfg["best_of_k"]
    if cfg["n_samples_single"] < k:
        print(f"[warn] n_samples_single ({cfg['n_samples_single']}) < best_of_k ({k}); best-of-K limited.")

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = CODE_DIR / "runs" / f"h0_{'smoke_' if a.smoke else ''}{run_id}"
    trace = TraceLogger(run_dir)
    client = LoggedClient(trace)

    domain = InjecAgentDomain(setting=a.setting, attack="dh")
    goals = domain.load_goals(split="ood", seed=cfg["seeds"][0], n=cfg.get("n_ood_goals"))

    used_providers = {cfg["attacker"]["provider"]} | {t["provider"] for t in cfg["targets"]}
    prov_info = {}
    for name in sorted(used_providers):
        try:
            base, key = resolve_provider(name)
            prov_info[name] = {"base_url": base, "key": mask(key)}
        except Exception as e:  # noqa: BLE001
            prov_info[name] = {"error": str(e)}

    meta = {
        "run_id": run_id, "smoke": a.smoke, "started": datetime.datetime.now().astimezone().isoformat(),
        "experiment": cfg.get("experiment"), "discussion": cfg.get("discussion"),
        "config": cfg, "setting": a.setting, "domain": domain.name, "attack": "dh",
        "ood_split": {"strategy": "stratified_by_attack_type", "holdout_attacker_tools": sorted(domain._ood),
                      "n_ood_tools": len(domain._ood), "n_goals_loaded": len(goals),
                      "goal_ids": [g.id for g in goals]},
        "repro": {"git_sha": git_sha(), "python": platform.python_version(), "platform": platform.platform(),
                  "providers": prov_info,
                  "note": "ASR is NOT bit-reproducible: temperatures>0 and API seed is best-effort/provider-dependent."},
        "command": " ".join(sys.argv),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run] {run_dir.name}  goals={len(goals)} (OOD tools={len(domain._ood)}, stratified)  "
          f"targets={[t['name'] for t in cfg['targets']]}  seeds={cfg['seeds']}  "
          f"n_single={cfg['n_samples_single']} n_multi={cfg['n_samples_multi']} K={k} maxturns={cfg['max_turns']}")

    jobs = []
    for t in cfg["targets"]:
        for seed in cfg["seeds"]:
            for g in goals:
                for s in range(cfg["n_samples_single"]):
                    jobs.append((t, "single", seed, g, s))
                for s in range(cfg["n_samples_multi"]):
                    jobs.append((t, "multi", seed, g, s))

    def work(job):
        t, arm, seed, g, s = job
        try:
            res = run_episode(domain, g, t, cfg["attacker"], arm=arm, seed=seed * 1000 + s,
                              max_turns=cfg["max_turns"], sample_idx=s, client=client, trace=trace)
            c = g.meta
            rec = {"target": t["name"], "arm": arm, "seed": seed, "sample": s, "goal_id": g.id,
                   "attacker_tool": c["Attacker Tools"][0], "user_tool": c["User Tool"],
                   "attack_type": c["Attack Type"], "success": bool(res["success"]),
                   "delivered_attack": res["delivered_attack"], "any_valid": res["any_valid"],
                   "final_eval": res["final_eval"], "n_invalid": res["n_invalid"],
                   "first_success_turn": res["first_success_turn"], "error": False}
        except Exception as e:  # noqa: BLE001 — never let one episode lose the run
            trace.log_event({"event": "job_error", "target": t["name"], "arm": arm, "seed": seed,
                             "goal_id": g.id, "sample": s, "error": f"{type(e).__name__}: {e}"})
            rec = {"target": t["name"], "arm": arm, "seed": seed, "sample": s, "goal_id": g.id,
                   "success": False, "delivered_attack": False, "any_valid": False,
                   "final_eval": "job_error", "error": True}
        trace.log_record(rec)
        return rec

    records, done = [], 0
    with ThreadPoolExecutor(max_workers=a.max_workers) as ex:
        for r in ex.map(work, jobs):
            records.append(r)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} episodes")

    clean = [r for r in records if not r.get("error")]
    n_err = len(records) - len(clean)

    results = {}
    for t in cfg["targets"]:
        tr = [r for r in clean if r["target"] == t["name"]]
        results[t["name"]] = analysis.analyze_target(tr, k=k, n_boot=cfg.get("n_boot", 10000))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*88}\nRESULTS  (errors excluded: {n_err})  [* = 95% CI excludes 0]\n{'='*88}")
    for name, res in results.items():
        a_s, a_m = res["arms"]["single"], res["arms"]["multi"]
        print(f"\n### target: {name}")
        if a_s:
            print(f"  single : ASR-all {pct(a_s['asr_all'])}  ASR-valid {pct(a_s['asr_valid'])}  "
                  f"ASR|delivered {pct(a_s['asr_delivered'])}  delivered {pct(a_s['delivered_rate'])}")
        if a_m:
            print(f"  multi  : ASR-all {pct(a_m['asr_all'])}  ASR-valid {pct(a_m['asr_valid'])}  "
                  f"ASR|delivered {pct(a_m['asr_delivered'])}  delivered {pct(a_m['delivered_rate'])}")
        fst = res["first_success_turn"]
        print(f"  multi first-success-turn: {fst['counts']}  (frac turn1 = {pct(fst['frac_turn1'])} of "
              f"{fst['n_successes']} successes)")
        print(f"  contrasts (goal-level paired bootstrap):")
        for cname, c in res["contrasts"].items():
            print(f"    {cname:28s} {fmt_contrast(c)}")
        print(f"    [tool-level] multi_vs_bestof{k}__delivered {fmt_contrast(res['tool_level'].get(f'multi_vs_bestof{k}__delivered'))}")
        pv = res["primary_verdict"]
        print(f"  >>> PRIMARY (attempt+delivery matched): H0 adaptation-helps = "
              f"{'SUPPORTED' if pv['h0_supported'] else 'NOT supported'}  (diff "
              f"{pv['diff']*100:+.1f}pt, CI [{pv['ci'][0]*100:+.1f},{pv['ci'][1]*100:+.1f}])" if pv['ci'] else
              f"  >>> PRIMARY: insufficient paired data")

    print(f"\n[done] traces: {run_dir}  ({trace.n_calls} LLM calls, {trace.n_events} events)")
    print("[note] 3 targets/contrasts share alpha=0.05 — apply Holm/Bonferroni if claiming 'holds for any target'.")
    trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
