"""Pull + analyze the Gate-2 dense-vs-sparse run from the H20 (no GPU).

Gate 2 asks: do per-step (dense) vs terminal (sparse) rewards produce a usable, DIVERGENT training
signal on the frozen victim? PASS if (a) the training reward moves (harness learns) and (b) the arms
diverge in behaviour — dense chases intermediate Phi (partial disclosure), sparse only the full breach.

Pulls progress.jsonl (per-step metrics, both arms) + each arm's train_rollouts.jsonl (per-rollout
Phi/reward/turns) to code/runs/h1x_gate2/, then reports per-arm reward trend + Phi behaviour.

  python code/scripts/h1_gate2_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import remote as RM  # noqa: E402

REMOTE = "/root/autodl-tmp/h1x"
LOCAL = CODE / "runs" / "h1x_gate2"
ARMS = ["dense", "sparse"]


def pull():
    LOCAL.mkdir(parents=True, exist_ok=True)
    cli = RM.connect()
    sftp = cli.open_sftp()
    got = []
    for remote_rel, local_name in [("progress.jsonl", "progress.jsonl")] + [
            (f"runs/extract-{a}-s0/records.jsonl", f"rollouts_{a}.jsonl") for a in ARMS]:
        try:
            sftp.get(f"{REMOTE}/{remote_rel}", str(LOCAL / local_name))
            got.append(local_name)
        except Exception as e:  # noqa: BLE001
            print(f"  (missing {remote_rel}: {type(e).__name__})")
    sftp.close(); cli.close()
    print(f"pulled {got} -> {LOCAL}")


def _load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _trend(vals: list[float]) -> str:
    if len(vals) < 4:
        return "n/a"
    h = max(1, len(vals) // 3)
    return f"first{h}={sum(vals[:h])/h:.3f} last{h}={sum(vals[-h:])/h:.3f} delta={sum(vals[-h:])/h - sum(vals[:h])/h:+.3f}"


def analyze():
    prog = _load(LOCAL / "progress.jsonl")
    print("\n=== per-step training reward (progress.jsonl) ===")
    for arm in ARMS:
        rows = [r for r in prog if r.get("arm") == arm]
        if not rows:
            print(f"  {arm}: no steps logged"); continue
        rewards = [r.get("reward", 0.0) for r in rows]
        zstd = [r.get("frac_reward_zero_std", 0.0) for r in rows]
        ent = [r.get("entropy", 0.0) for r in rows]
        print(f"  {arm:6s} steps={len(rows)}  reward: {_trend(rewards)}")
        print(f"         zero_std_frac: {_trend(zstd)}   entropy: {_trend(ent)}")

    print("\n=== per-rollout Phi behaviour (train_rollouts.jsonl) ===")
    for arm in ARMS:
        rr = _load(LOCAL / f"rollouts_{arm}.jsonl")
        if not rr:
            print(f"  {arm}: no rollouts"); continue
        n = len(rr)
        mp = [r.get("max_phi", 0.0) for r in rr]
        succ = sum(1 for r in rr if r.get("success"))
        partial = sum(1 for r in rr if 0.0 < r.get("max_phi", 0.0) < 1.0)
        refused = sum(1 for r in rr if r.get("refused"))
        # early vs late mean max_phi (behavioural drift over training)
        h = max(1, n // 3)
        early, late = sum(mp[:h]) / h, sum(mp[-h:]) / h
        hist = {}
        for v in mp:
            hist[round(v, 1)] = hist.get(round(v, 1), 0) + 1
        print(f"  {arm:6s} n={n}  mean_maxphi={sum(mp)/n:.3f}  success={succ}({succ/n:.0%})  "
              f"partial={partial}({partial/n:.0%})  refused={refused}({refused/n:.0%})")
        print(f"         maxphi early->late: {early:.3f} -> {late:.3f} ({late-early:+.3f})   "
              f"hist={dict(sorted(hist.items()))}")

    print("\n=== Gate 2 read ===")
    print("  PASS if: both arms' reward moves (harness learns) AND dense reaches higher/rising "
          "mean_maxphi (chases partial Phi) while sparse's partial credit stays flat/low.")


if __name__ == "__main__":
    pull()
    analyze()
