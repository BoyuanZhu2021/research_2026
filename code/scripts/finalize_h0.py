"""Finalize H0 results across one or more run dirs.

A run may be split (e.g. one target completed in run A, another in run B, after an
interruption). This merges per-target, taking each target from the run dir where it
has the MOST records (avoids double-counting a target that appears partially in two
dirs), runs the confound-aware analysis, and writes a combined results file.

Usage:
  python code/scripts/finalize_h0.py [run_dir ...] [--k 5] [--out path.json]
  (no run_dir => all non-smoke dirs under code/runs)
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
from src import analysis  # noqa: E402


def load_records(run_dir):
    p = Path(run_dir) / "records.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(x):
    return " n/a " if x is None else f"{x * 100:5.1f}%"


def fmt(c):
    if not c:
        return "(no paired units)"
    return (f"{c['mean_a']*100:5.1f}% vs {c['mean_b']*100:5.1f}%  diff {c['diff']*100:+5.1f} "
            f"[{c['ci_low']*100:+5.1f},{c['ci_high']*100:+5.1f}]pt{' *' if c['excludes_zero'] else '  '} (n={c['n_units']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = a.run_dirs or sorted(d for d in glob.glob(str(CODE_DIR / "runs" / "h0_*")) if "smoke" not in d)
    if not dirs:
        print("no run dirs found"); return 1

    best = {}  # target -> (count, dir, records)
    for d in dirs:
        recs = [r for r in load_records(d) if not r.get("error")]
        for tgt, cnt in Counter(r["target"] for r in recs).items():
            if tgt not in best or cnt > best[tgt][0]:
                best[tgt] = (cnt, d, [r for r in recs if r["target"] == tgt])

    results = {}
    for tgt, (cnt, d, recs) in sorted(best.items()):
        results[tgt] = {"source_dir": os.path.basename(d), "n_records": cnt,
                        "analysis": analysis.analyze_target(recs, k=a.k, n_boot=10000)}

    out = a.out or str(CODE_DIR / "runs" / "results_combined.json")
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}\n{'='*92}\nH0 COMBINED RESULTS  [* = 95% CI excludes 0]\n{'='*92}")
    for tgt, res in results.items():
        an = res["analysis"]; s, m = an["arms"]["single"], an["arms"]["multi"]
        print(f"\n### {tgt}  (n={res['n_records']} episodes, from {res['source_dir']})")
        if s:
            print(f"  single: ASR-all {pct(s['asr_all'])} ASR-valid {pct(s['asr_valid'])} "
                  f"ASR|deliv {pct(s['asr_delivered'])} deliv {pct(s['delivered_rate'])}")
        if m:
            print(f"  multi : ASR-all {pct(m['asr_all'])} ASR-valid {pct(m['asr_valid'])} "
                  f"ASR|deliv {pct(m['asr_delivered'])} deliv {pct(m['delivered_rate'])}")
        fst = an["first_success_turn"]
        print(f"  multi first-success-turn {fst['counts']} (frac turn1 {pct(fst['frac_turn1'])} of {fst['n_successes']})")
        for cn, c in an["contrasts"].items():
            print(f"    {cn:28s} {fmt(c)}")
        tl = an["tool_level"].get(f"multi_vs_bestof{a.k}__delivered")
        print(f"    {'[tool] mvb%d_delivered' % a.k:28s} {fmt(tl)}")
        pv = an["primary_verdict"]
        verdict = "SUPPORTED" if pv["h0_supported"] else "NOT supported"
        ci = f"diff {pv['diff']*100:+.1f}pt CI[{pv['ci'][0]*100:+.1f},{pv['ci'][1]*100:+.1f}]" if pv["ci"] else "n/a"
        print(f"  >>> PRIMARY (multi vs best-of-{a.k}, delivered): H0 adaptation-helps = {verdict}  ({ci})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
