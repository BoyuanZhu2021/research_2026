"""Pull + analyze the full-trajectory Gate-2-redo run (h1mt) from the H20.

Gate 2 asks: does the trained policy now CONTROL the graded progress so per-step (dense) and terminal
(sparse) rewards produce a divergent signal? Reads progress.jsonl (per-step mean_max_phi / success)
+ each arm's rollouts.jsonl (per-rollout phi_trace), reports the learning trend per arm and the
dense-vs-sparse comparison.

  python code/scripts/h1_mt_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import remote as RM  # noqa: E402

REMOTE = "/root/autodl-tmp/h1mt"
LOCAL = CODE / "runs" / "h1mt_gate2redo"
ARMS = ["dense", "sparse"]


def pull():
    LOCAL.mkdir(parents=True, exist_ok=True)
    cli = RM.connect(); sftp = cli.open_sftp(); got = []
    for remote_rel, local_name in [("progress.jsonl", "progress.jsonl")] + [
            (f"runs/mt-{a}-s0/rollouts.jsonl", f"rollouts_{a}.jsonl") for a in ARMS]:
        try:
            sftp.get(f"{REMOTE}/{remote_rel}", str(LOCAL / local_name)); got.append(local_name)
        except Exception as e:  # noqa: BLE001
            print(f"  (missing {remote_rel}: {type(e).__name__})")
    sftp.close(); cli.close()
    print(f"pulled {got} -> {LOCAL}")


def _load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _win(vals, frac=0.4):
    """mean of the first and last `frac` window."""
    if not vals:
        return (0.0, 0.0)
    h = max(1, int(len(vals) * frac))
    return (sum(vals[:h]) / h, sum(vals[-h:]) / h)


def main():
    pull()
    prog = _load(LOCAL / "progress.jsonl")
    print("\n=== per-step learning (progress.jsonl) ===")
    summ = {}
    for arm in ARMS:
        rows = [r for r in prog if r.get("arm") == arm]
        mp = [r["mean_max_phi"] for r in rows]
        sr = [r["success_rate"] for r in rows]
        f_mp, l_mp = _win(mp); f_sr, l_sr = _win(sr)
        summ[arm] = {"steps": len(rows), "first_phi": f_mp, "last_phi": l_mp, "last_sr": l_sr,
                     "all_phi": mp}
        print(f"  {arm:6s} steps={len(rows)}  mean_max_phi first40%={f_mp:.3f} -> last40%={l_mp:.3f} "
              f"(Δ={l_mp-f_mp:+.3f})   success last40%={l_sr:.2f}")

    print("\n=== per-rollout Phi (rollouts.jsonl, last 40% of rollouts) ===")
    for arm in ARMS:
        rr = _load(LOCAL / f"rollouts_{arm}.jsonl")
        if not rr:
            print(f"  {arm}: none"); continue
        maxphi = [max(r["phi_trace"]) if r["phi_trace"] else 0.0 for r in rr]
        h = max(1, int(len(maxphi) * 0.4))
        late = maxphi[-h:]
        hist = {}
        for v in late:
            hist[round(v, 1)] = hist.get(round(v, 1), 0) + 1
        succ = sum(1 for r in rr[-h:] if r.get("success"))
        print(f"  {arm:6s} n={len(rr)}  late mean_maxphi={sum(late)/len(late):.3f}  "
              f"late success={succ}/{h}  late hist={dict(sorted(hist.items()))}")

    print("\n=== Gate 2 (redo) read ===")
    d, s = summ.get("dense", {}), summ.get("sparse", {})
    if d and s:
        d_rose = d["last_phi"] - d["first_phi"]
        s_rose = s["last_phi"] - s["first_phi"]
        gap = d["last_phi"] - s["last_phi"]
        print(f"  dense learned: Δphi={d_rose:+.3f} (first {d['first_phi']:.3f} -> last {d['last_phi']:.3f})")
        print(f"  sparse learned: Δphi={s_rose:+.3f} (first {s['first_phi']:.3f} -> last {s['last_phi']:.3f})")
        print(f"  final gap dense-sparse = {gap:+.3f}")
        print("  PASS if both arms move (harness learns on-policy) AND they diverge "
              "(dense reaches higher / rises where sparse stalls).")


if __name__ == "__main__":
    main()
