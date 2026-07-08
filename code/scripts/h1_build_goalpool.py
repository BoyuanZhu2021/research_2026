"""Build + validate + freeze the H1 Phase-A goal pool (A1).

Enumerates the parametrized banking goal grid, validates every spec's oracle against the AgentDojo
runtime (reward-module golden checks), splits into train / OOD (fixed SPLIT_SEED, stratified by
family x depth), and writes goals_train.jsonl / goals_ood.jsonl + a summary. CPU-only, no API, no GPU.

Usage:  python code/scripts/h1_build_goalpool.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "src"))

import goalpool as G  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

OUT = CODE_DIR / "runs" / "goalpool"


def main() -> int:
    bank = get_suites("v1")["banking"]
    ut = next(iter(bank.user_tasks.values()))

    specs = G.iter_specs()
    depths, regimes, bad = {}, {}, []
    for s in specs:
        key = json.dumps(s, sort_keys=True)
        ok, trace, secured = G.validate_spec(s, bank, ut)
        d = len(G.build_task(s).ground_truth(ut.init_environment(bank.load_and_inject_default_environment({}))))
        depths[key], regimes[key] = d, G.regime_of(d)
        if not ok:
            bad.append((s, trace, secured))

    if bad:
        print(f"[FAIL] {len(bad)}/{len(specs)} specs failed oracle validation:")
        for s, trace, sec in bad[:10]:
            print("  -", json.dumps(s), "trace=", trace, "secured=", sec)
        return 1

    split = G.split_specs(specs, depths)
    OUT.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        p = OUT / name
        with open(p, "w", encoding="utf-8") as f:
            for s in rows:
                key = json.dumps(s, sort_keys=True)
                f.write(json.dumps({**s, "depth": depths[key], "regime": regimes[key]}) + "\n")
        return p

    dump("goals_train.jsonl", split["train"])
    dump("goals_ood.jsonl", split["ood"])

    def breakdown(rows):
        c = Counter(regimes[json.dumps(s, sort_keys=True)] for s in rows)
        dd = Counter(depths[json.dumps(s, sort_keys=True)] for s in rows)
        return dict(c), dict(sorted(dd.items()))

    tr_reg, tr_depth = breakdown(split["train"])
    od_reg, od_depth = breakdown(split["ood"])
    summary = {
        "split_seed": G.SPLIT_SEED, "ood_frac": G.OOD_FRAC,
        "total_specs": len(specs), "all_validated": True,
        "families": dict(Counter(s["family"] for s in specs)),
        "train": {"n": len(split["train"]), "regime": tr_reg, "depth": tr_depth},
        "ood": {"n": len(split["ood"]), "regime": od_reg, "depth": od_depth},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nvalidated {len(specs)}/{len(specs)} specs; wrote train/ood jsonl + summary to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
