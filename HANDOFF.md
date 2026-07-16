# HANDOFF — live state & next action

> **Single source of "where we are right now."** Any new session/agent reads this FIRST (after
> `python tools/session_check.py`), then continues. **Keep it current: update it whenever the active
> task or its next action changes.** History lives in `LOGS/`, `Discussion.md`, `method.md`,
> `docs/plans/`; this file is only the pointer.

**Last updated:** 2026-07-15 · **Active issue:** [`DISC-2026W28-001`](Discussion.md) (Open) · **Mode:** newbie

---

## Current phase — TOOL-USE reframe (fast turnaround)

**Pivot (user directive 2026-07-15):** lighter models + make **tool-use central**. New conditions:
- **Attacker = Qwen3.5-4B (int4 QLoRA), Victim = Qwen3.5-9B (FP8/int8)** — faster turnaround.
- **ASR judged by TOOL-CHAIN EXECUTION** (not chat disclosure): Φ = fraction of the target malicious
  tool-chain the victim was induced to execute, causally gated. dense = per-tool ΔΦ, sparse = terminal.
- **Channel** = indirect prompt injection (InjecAgent). **RL** = GRPO (per-turn potential, built).
- **Data**: InjecAgent `ds` (m=2) fast gate → extend m≥3.
- Full design: [`docs/plans/h1-tooluse-plan.md`](docs/plans/h1-tooluse-plan.md) (grilled 2026-07-15).

The previous chat-disclosure powered run (EXP-2026W29-001) is **abandoned** in favor of this reframe.

## Status

- ✅ **Stage 0 built + CPU-validated (no GPU):** `code/src/domains/tooluse_oracle.py` (m-step gated
  tool-chain oracle, golden all-pass — m=2 reproduces the `ds` scores, m=3 gives 5 graded states) +
  `code/src/domains/tooluse_injection.py` (`ToolUseInjectionDomain`, thin InjecAgent subclass, Stage-0
  test all-pass). Plugs into the existing `mt_rollout`/`mt_grpo` harness.
- ✅ **Model configs updated** (env-overridable): attacker `Qwen/Qwen3.5-4B`, victim serve `Qwen/Qwen3.5-9B`
  (fp8) in `h1_mt_grpo_train.py` / `h1_mt_ood_eval.py` / `h1_serve_victim.py`.
- ⛔ **SERVER UNREACHABLE:** H20 SSH fails with a protocol-banner error → **container down or
  `.env::REMOTE_HOST` creds rotated** (AutoDL issues a new host/port/password on restart). All
  server-side steps are blocked.

## ▶ Immediate next actions

**First — restore connectivity (needs the user):** restart the AutoDL instance and **paste the new SSH
command** so `.env::REMOTE_HOST` + `REMOTE_PASSWORD` can be updated. Verify with `python code/src/remote.py`.

**Then — provision the new models (server up):**
1. Confirm exact HF repo IDs for `Qwen3.5-4B` / `Qwen3.5-9B` (try to download; adjust suffix if 404).
   The user said a mirror is saved and old models may be deleted.
2. Free disk: delete `hf_home/hub/models--Qwen--Qwen3-8B` and `...Qwen3.6-27B-FP8` (~47G); download
   the 4B + 9B.
3. Serve the 9B victim: `python code/scripts/h1_serve_victim.py` (fp8; **do NOT set `HF_HUB_OFFLINE`**
   until fully cached — IncompleteSnapshot lesson). Spot-check it EXECUTES tool-chains under injection.

**Then — the experiment (still needs building + running):**
4. **Wire the tool-use victim into the harness** (NOT done yet): the trainer/eval need a ReAct
   tool-loop `victim_batch_fn` (adapt `direct_extraction_episode.run_extraction_victim`) + a
   `--domain tooluse` switch to use `ToolUseInjectionDomain`. CPU-testable with a mock victim.
5. **Gate 1′ (no GPU beyond the served victim):** `h1_defense_sweep`-style over `ds` with the 9B victim
   → does it rest at 0<Φ<1 (partial tool-chain) AND is full reachable? Pick+freeze the defense tier.
6. **Gate 2 → scale:** GRPO dense-vs-sparse divergence → paired OOD tool-chain ASR + Holm verdict.

## Owed housekeeping

- **W28 Weekly Retro** (§6) — still owed (scan `LOGS/2026-W28.md`, post retro, set `MODE.md::last_retro`).

## Key files

| What | Where |
|---|---|
| Live state (this file) | `HANDOFF.md` |
| Tool-use plan (grilled) | `docs/plans/h1-tooluse-plan.md` |
| New oracle / domain | `code/src/domains/tooluse_oracle.py` · `code/src/domains/tooluse_injection.py` |
| Stage-0 test | `code/scripts/h1_tooluse_stage0_test.py` |
| Trainer / rollout / eval / analyze (reused) | `code/scripts/h1_mt_grpo_train.py` · `code/src/mt_rollout.py` · `code/scripts/h1_mt_ood_eval.py` · `code/scripts/h1_mt_powered_analyze.py` |
| Victim serve (9B) / deploy / remote | `code/scripts/h1_serve_victim.py` · `code/scripts/h1_deploy_mt.py` · `code/src/remote.py` |
| Protocol / discussion registry | `AGENTS.md` · `Discussion/INDEX.md` |
| Compute notes | memory `compute-env` (⚠ SiliconFlow balance exhausted; H20 creds rotate on restart) |
