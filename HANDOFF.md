# HANDOFF — live state & next action

> **Single source of "where we are right now."** Any new session/agent reads this FIRST (after
> `python tools/session_check.py`), then continues. **Keep it current: update it whenever the active
> task or its next action changes** (end of session, when a run finishes, when blocked/unblocked).
> Detailed history lives in `LOGS/`, `Discussion.md`, `method.md`; this file is only the pointer.

**Last updated:** 2026-07-15 · **Active issue:** [`DISC-2026W28-001`](Discussion.md) (Open) · **Mode:** newbie

---

## Current phase

**H1 verification** — does a per-step (dense, ΔΦ) reward beat a terminal (sparse) reward on **OOD ASR**
for the multi-turn extraction attacker? Plan: [`docs/plans/h1-verification-plan.md`](docs/plans/h1-verification-plan.md).
Latest experiment: [`EXP-2026W29-001`](LOGS/2026-W29.md).

## Status: TRAINING DONE, verdict eval BLOCKED on GPU

- ✅ Built + CPU-validated: batched `rollout_batch` (5.5× throughput, equivalence-tested), KL-to-ref
  stabilizer (`disable_adapter`), paired powered analysis (`bootstrap_diff` + Holm).
- ✅ Trained **dense + sparse** (LR 1e-5 + KL 0.01, 40 steps, B=48, seed 0). grad_norm ~20 (was 76 →
  stable, no collapse). H1 mechanism visible in training: dense ~180 loss-examples/step vs **sparse
  signal-starved ~24/step** (full 5/5 disclosure only ~2%). mean_phi noisy ~0.24–0.40 both arms.
- ✅ Adapters saved on H20 persistent disk: `/root/autodl-tmp/h1mt/runs/mt-{dense,sparse}-s0/adapter`.
  Local backup `code/runs/h1mt_adapters/`: **dense complete (349 MB)**, **sparse truncated** (gateway
  flaked — re-pull the full copy from the H20 when the connection is stable).
- ⛔ **BLOCKED:** AutoDL **reclaimed the GPU** mid-eval (2026-07-15 ~20:00). Container up, but
  `nvidia-smi` → "No devices found"; vLLM/victim dead. User: GPU back in a few hours.

## ▶ Immediate next action (when the GPU is back)

Run the one-command resume, which re-serves the victim, re-pulls the full sparse adapter, and launches
the eval chain:

```
python code/scripts/h1_resume_eval.py            # from the local repo (drives the H20 over SSH)
```

Then, after the eval chain finishes (~1 h; monitor via the printed commands):

```
python code/scripts/h1_mt_powered_analyze.py --pull      # paired dense−sparse OOD ASR + Holm verdict
```

Finally: record the verdict as an EXP + Discussion post, then **power off the H20** (`shutdown -h now`
over SSH; confirm "stopped" in the AutoDL console).

**What the eval does** (already staged inside `h1_resume_eval.py`): base(best-of-K, seeds=4) + dense
(seeds=3) + sparse(seeds=3) on the SAME n=150 held-out OOD goals → `ood_eval/`; plus base/dense/sparse
on n=48 in-domain → `indomain_eval/` (the learning gate: did training beat base at all?).

## Honest caveats to carry forward

- Training Φ was **noisy (~0.24–0.40 for both arms)**; the OOD verdict may still land **inconclusive**.
  If it does, that points at the **testbed's low ceiling** (base already ~17% OOD ASR against a hard
  27B victim that won't exfiltrate) rather than more hyperparameter tuning — report it straight.
- Budget: ran **1 training seed/arm** (not the 3 the plan scoped) to fit ~one H20 session; if dense>sparse
  shows, harden with seeds 1–2 later. `--seed` in the trainer sets it.
- `progress.jsonl` on the H20 is polluted by killed-run rows (rm-races); use the per-run
  `runs/mt-*/rollouts.jsonl` and the adapters for anything authoritative.

## Owed housekeeping

- **W28 Weekly Retro** (§6) — scan `LOGS/2026-W28.md`, post a retro under the active issue, set
  `MODE.md::last_retro = 2026-W29`. `tools/session_check.py` will keep flagging this until done.

## Key files

| What | Where |
|---|---|
| Live state (this file) | `HANDOFF.md` |
| Protocol (rules) | `AGENTS.md` |
| Plans (durable, in-repo) | `docs/plans/` |
| Method formalization | `method.md` (§2 per-turn potential + KL, PI-approved) |
| Experiment log (this week) | `LOGS/2026-W29.md` |
| Discussion + registry | `Discussion.md` (active) · `Discussion/INDEX.md` |
| Trainer / rollout / eval / analyze | `code/scripts/h1_mt_grpo_train.py` · `code/src/mt_rollout.py` · `code/scripts/h1_mt_ood_eval.py` · `code/scripts/h1_mt_powered_analyze.py` |
| Remote/serve helpers | `code/src/remote.py` · `code/scripts/h1_serve_victim.py` · `code/scripts/h1_deploy_mt.py` |
| Compute + serving notes | memory `compute-env`, `vllm-qwen36-serving` |
