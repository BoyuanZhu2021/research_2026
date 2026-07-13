"""Multi-turn GRPO core: per-turn potential reward + return-to-go group-relative advantage.

Implements method.md §2 (H1 full-trajectory multi-turn, DRAFT pending PI approval). Pure functions
over Phi traces — NO LLM, NO GPU — so the credit-assignment math is unit-testable on CPU before it
is wired into the training loss. The rollout (policy generates all turns, local vLLM victim responds)
and the PG loss are built separately once the H20 is up + the math is approved.

Arms (matching src/reward.py: dense = potential Phi, sparse = terminal):
  r_dense_t  = Phi_t - Phi_{t-1}                         (per-turn potential gain, >= 0)
  r_sparse_t = 1[Phi_t >= tau and Phi_{t-1} < tau]       (fires once, at the tau crossing)
Telescoping: sum_t r_dense = Phi_T ;  sum_t r_sparse = 1[Phi_T >= tau]  (same terminal return).

Advantage (per goal, over G on-policy trajectories):
  G_{i,t} = sum_{k>=t} r_{i,k}                           (return-to-go)
  b_t, sigma_t = mean/std over trajectories present at turn t
  A_{i,t} = (G_{i,t} - b_t) / (sigma_t + eps)
"""
from __future__ import annotations

import math


def per_turn_rewards(phi_trace: list[float], tau: float, arm: str) -> list[float]:
    """Per-turn reward from a Phi trace [Phi_1, .., Phi_T] (Phi_0 := 0 implied).

    dense = potential gain ΔΦ_t (>=0 since Phi is monotone). sparse = 1 exactly at the turn that
    first reaches tau, else 0.
    """
    if arm not in ("dense", "sparse"):
        raise ValueError(f"unknown arm {arm!r}")
    rewards = []
    prev = 0.0
    crossed = False
    for phi in phi_trace:
        if arm == "dense":
            rewards.append(max(0.0, phi - prev))
        else:
            hit = (not crossed) and (phi >= tau) and (prev < tau)
            rewards.append(1.0 if hit else 0.0)
            crossed = crossed or phi >= tau
        prev = phi
    return rewards


def returns_to_go(rewards: list[float]) -> list[float]:
    """G_t = sum_{k>=t} r_k (undiscounted; gamma=1 matches potential telescoping)."""
    out = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc += rewards[t]
        out[t] = acc
    return out


def group_advantages(traj_rewards: list[list[float]], eps: float = 1e-6) -> list[list[float]]:
    """GRPO group-relative, per-turn-position advantages over G trajectories for ONE goal.

    traj_rewards[i] = per-turn rewards of trajectory i (variable length; episodes end early on
    success). Returns A[i][t] = (G_{i,t} - b_t)/(sigma_t+eps), with b_t/sigma_t computed over the
    trajectories that REACHED turn t. If a position has < 2 trajectories or zero spread, advantages
    there are 0 (no usable gradient) — this is exactly how sparse degenerates when success is rare.
    """
    rtg = [returns_to_go(r) for r in traj_rewards]
    maxT = max((len(r) for r in rtg), default=0)
    adv = [[0.0] * len(r) for r in rtg]
    for t in range(maxT):
        col = [(i, rtg[i][t]) for i in range(len(rtg)) if t < len(rtg[i])]
        if len(col) < 2:
            continue
        vals = [v for _, v in col]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(var)
        if sigma < eps:            # no spread -> no relative signal (degenerate group)
            continue
        for i, v in col:
            adv[i][t] = (v - mean) / (sigma + eps)
    return adv


def frac_zero_gradient(traj_rewards: list[list[float]]) -> float:
    """Diagnostic: fraction of (trajectory,turn) decisions with zero advantage (no learning signal).
    High for sparse when success is rare — the quantity Claim 1 predicts separates the arms."""
    adv = group_advantages(traj_rewards)
    total = sum(len(a) for a in adv) or 1
    zero = sum(1 for a in adv for x in a if abs(x) < 1e-9)
    return zero / total


# --------------------------------------------------------------------------- golden self-test
def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def _selftest() -> int:
    fails = []

    # 1. dense telescopes to Phi_T; sparse fires once at tau
    tr = [0.2, 0.2, 0.6, 1.0]
    d = per_turn_rewards(tr, tau=1.0, arm="dense")
    s = per_turn_rewards(tr, tau=1.0, arm="sparse")
    if not _approx(sum(d), 1.0):
        fails.append(f"dense sum {sum(d)} != Phi_T 1.0")
    if d != [0.2, 0.0, 0.4, 0.4] and not all(_approx(x, y) for x, y in zip(d, [0.2, 0.0, 0.4, 0.4])):
        fails.append(f"dense per-turn wrong: {d}")
    if s != [0, 0, 0, 1.0]:
        fails.append(f"sparse should fire once at the tau=1 crossing: {s}")
    # tau=0.6 -> sparse fires at turn 3 (Phi 0.2->0.6)
    s06 = per_turn_rewards(tr, tau=0.6, arm="sparse")
    if s06 != [0, 0, 1.0, 0]:
        fails.append(f"sparse tau=0.6 crossing wrong: {s06}")

    # 2. return-to-go
    if returns_to_go([0.2, 0.0, 0.4, 0.4]) != [1.0, 0.8, 0.8, 0.4]:
        fails.append(f"rtg wrong: {returns_to_go([0.2,0.0,0.4,0.4])}")

    # 3. THE mechanism: a group of failed-but-partial trajectories.
    #    3 trajectories, none reaches tau=1.0 (max Phi 0.6/0.4/0.0) -> sparse has NO signal anywhere,
    #    dense still separates the progress-makers from the zero one.
    traces = [[0.2, 0.4, 0.6], [0.2, 0.4, 0.4], [0.0, 0.0, 0.0]]
    dense_g = [per_turn_rewards(tr, 1.0, "dense") for tr in traces]
    sparse_g = [per_turn_rewards(tr, 1.0, "sparse") for tr in traces]
    fz_dense = frac_zero_gradient(dense_g)
    fz_sparse = frac_zero_gradient(sparse_g)
    if fz_sparse != 1.0:
        fails.append(f"sparse should have 100% zero-gradient on all-failed group, got {fz_sparse}")
    if fz_dense >= 1.0:
        fails.append(f"dense should have SOME gradient on partial-progress group, got {fz_dense}")
    adv_dense = group_advantages(dense_g)
    # trajectory 0 (most progress) should get positive advantage at turn 0; trajectory 2 (none) negative
    if not (adv_dense[0][0] > 0 > adv_dense[2][0]):
        fails.append(f"dense advantage should rank progress: t0={adv_dense[0][0]:.2f} t2={adv_dense[2][0]:.2f}")

    # 4. sparse DOES get signal when the group has mixed success (some reach tau)
    mixed = [[0.5, 1.0], [0.5, 0.5], [0.0, 0.0]]  # traj0 succeeds
    sparse_mixed = [per_turn_rewards(tr, 1.0, "sparse") for tr in mixed]
    if frac_zero_gradient(sparse_mixed) >= 1.0:
        fails.append("sparse should have signal when some trajectories succeed")

    for f in fails:
        print("  [FAIL]", f)
    if not fails:
        print("  [ok] dense telescopes to Phi_T; sparse fires once at tau")
        print("  [ok] return-to-go correct")
        print(f"  [ok] all-failed-partial group: sparse zero-grad=100%, dense zero-grad={fz_dense:.0%} "
              f"(dense still learns, sparse doesn't) <- Claim 1")
        print("  [ok] dense advantage ranks partial progress; sparse gets signal only on mixed-success")
        print("mt_grpo golden: ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
