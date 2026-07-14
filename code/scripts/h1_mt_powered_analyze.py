"""Powered paired analysis of the H1 verification eval (Step 4). Reuses `analysis.bootstrap_diff`
(paired bootstrap over shared goals, returns excludes_zero) and adds a small bootstrap p-value + Holm
correction across the {dense-sparse, dense-base, sparse-base} family.

Reads per-(goal,seed) rows written by h1_mt_ood_eval.py (`ood_<tag>_rows.jsonl`) and groups by arm via
tag prefix: `base*` -> base (best-of-K control), `dense*` -> dense (pool seeds), `sparse*` -> sparse.

  python code/scripts/h1_mt_powered_analyze.py --pull        # sftp the ood_eval dir from the H20 then analyze
  python code/scripts/h1_mt_powered_analyze.py               # analyze already-pulled local rows

H1 supported iff: paired dense-sparse OOD ASR diff > 0 AND excludes 0 (Holm-significant) AND both
trained arms beat base(best-of-K).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from analysis import bootstrap_diff  # noqa: E402

REMOTE = "/root/autodl-tmp/h1mt/ood_eval"
LOCAL = CODE / "runs" / "h1mt_powered"


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pull():
    sys.path.insert(0, str(CODE / "src"))
    import remote as RM
    LOCAL.mkdir(parents=True, exist_ok=True)
    cli = RM.connect(); sftp = cli.open_sftp()
    rc, out, _ = RM.run(cli, f"ls {REMOTE}/*_rows.jsonl 2>/dev/null", timeout=30)
    got = []
    for path in out.split():
        name = path.rsplit("/", 1)[-1]
        try:
            sftp.get(path, str(LOCAL / name)); got.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"  (miss {name}: {type(e).__name__})")
    sftp.close(); cli.close()
    print(f"pulled {len(got)} row files -> {LOCAL}")


def load_arms():
    """arm -> list of rows, grouped by tag prefix of each ood_<tag>_rows.jsonl file."""
    arms = defaultdict(list)
    for p in sorted(LOCAL.glob("ood_*_rows.jsonl")):
        tag = p.name[len("ood_"):-len("_rows.jsonl")]
        arm = "base" if tag.startswith("base") else "dense" if tag.startswith("dense") \
            else "sparse" if tag.startswith("sparse") else tag
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                arms[arm].append(json.loads(line))
    return arms


def asr_rate(rows):
    """{goal: mean success over that goal's rows} (pools seeds/samples)."""
    d = defaultdict(list)
    for r in rows:
        d[r["goal"]].append(r["success"])
    return {g: _mean(v) for g, v in d.items()}


def phi_rate(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["goal"]].append(r["max_phi"])
    return {g: _mean(v) for g, v in d.items()}


def bestofk_rate(rows):
    """{goal: 1 if ANY of that goal's attempts succeeded} — attempt-matched base control."""
    d = defaultdict(int)
    for r in rows:
        d[r["goal"]] = max(d[r["goal"]], int(r["success"]))
    return dict(d)


def paired_boot_p(a, b, n_boot=10000, seed=0):
    """2-sided bootstrap p that mean(a)-mean(b) (paired over shared goals) != 0."""
    units = [u for u in a if u in b]
    if len(units) < 2:
        return 1.0
    point = _mean([a[u] for u in units]) - _mean([b[u] for u in units])
    rng = random.Random(seed); n = len(units)
    tail = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = _mean([a[units[i]] for i in idx]) - _mean([b[units[i]] for i in idx])
        if (d <= 0) if point >= 0 else (d >= 0):
            tail += 1
    return min(1.0, 2.0 * tail / n_boot)


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, adj = len(items), 0.0, {}
    for rank, (name, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adj[name] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="sftp the ood_eval rows from the H20 first")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    if args.pull:
        pull()

    arms = load_arms()
    for a in ("base", "dense", "sparse"):
        if a not in arms:
            print(f"MISSING arm '{a}' — need ood_{a}*_rows.jsonl in {LOCAL}"); return 1
    print("arms:", {a: {"rows": len(arms[a]), "goals": len(asr_rate(arms[a]))} for a in ("base", "dense", "sparse")})

    d_asr, s_asr = asr_rate(arms["dense"]), asr_rate(arms["sparse"])
    d_phi, s_phi = phi_rate(arms["dense"]), phi_rate(arms["sparse"])
    b_1 = asr_rate(arms["base"])                # base 1-sample ASR (fair, same scoring as dense/sparse)
    b_bok = bestofk_rate(arms["base"])          # base best-of-K (stricter attempt-matched control)
    b_phi = phi_rate(arms["base"])

    # Primary + the "did training help" gate use the FAIR 1-sample base; Holm over these three.
    contrasts = {
        "dense-sparse_ASR": (d_asr, s_asr),
        "dense-base1_ASR":  (d_asr, b_1),
        "sparse-base1_ASR": (s_asr, b_1),
    }
    stats, pvals = {}, {}
    for name, (a, b) in contrasts.items():
        stats[name] = bootstrap_diff(a, b); pvals[name] = paired_boot_p(a, b)
    adj = holm(pvals)
    # stricter best-of-K controls (reported, not gating)
    for name, (a, b) in {"dense-baseK_ASR": (d_asr, b_bok), "sparse-baseK_ASR": (s_asr, b_bok)}.items():
        stats[name] = bootstrap_diff(a, b)

    print("\n=== paired ASR contrasts (paired bootstrap over shared OOD goals; Holm over the 3 gating) ===")
    for name in contrasts:
        st = stats[name]
        print(f"  {name:18s} diff={st['diff']:+.4f}  CI[{st['ci_low']:+.4f},{st['ci_high']:+.4f}] "
              f"excl0={st['excludes_zero']}  p={pvals[name]:.4f}  p_holm={adj[name]:.4f}")
    print("  -- stricter best-of-K base control (informative) --")
    for name in ("dense-baseK_ASR", "sparse-baseK_ASR"):
        st = stats[name]
        print(f"  {name:18s} diff={st['diff']:+.4f}  CI[{st['ci_low']:+.4f},{st['ci_high']:+.4f}] excl0={st['excludes_zero']}")
    print("\n=== mean-Phi contrasts (secondary) ===")
    for name, (a, b) in {"dense-sparse_Phi": (d_phi, s_phi), "dense-base_Phi": (d_phi, b_phi),
                         "sparse-base_Phi": (s_phi, b_phi)}.items():
        st = bootstrap_diff(a, b)
        print(f"  {name:18s} diff={st['diff']:+.4f}  CI[{st['ci_low']:+.4f},{st['ci_high']:+.4f}] excl0={st['excludes_zero']}")

    ds = stats["dense-sparse_ASR"]
    h1 = (ds["diff"] > 0 and ds["excludes_zero"] and adj["dense-sparse_ASR"] < args.alpha
          and stats["dense-base1_ASR"]["diff"] > 0 and stats["sparse-base1_ASR"]["diff"] > 0)
    print(f"\n=== H1 VERDICT ===\n  PRIMARY dense-sparse OOD ASR = {ds['diff']:+.4f} "
          f"(CI [{ds['ci_low']:+.4f},{ds['ci_high']:+.4f}], Holm p={adj['dense-sparse_ASR']:.4f})")
    print(f"  training helped (vs base 1-sample): dense-base1={stats['dense-base1_ASR']['diff']:+.4f}, "
          f"sparse-base1={stats['sparse-base1_ASR']['diff']:+.4f}")
    print(f"  H1 (per-step > terminal on OOD) {'SUPPORTED' if h1 else 'NOT supported'} at alpha={args.alpha}")
    (LOCAL / "verdict.json").write_text(json.dumps(
        {"contrasts": {k: stats[k] for k in stats}, "pvals": pvals, "holm": adj, "h1_supported": h1},
        indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
