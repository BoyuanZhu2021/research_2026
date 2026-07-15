# Plan — Effectively verify H1 (per-step/dense vs terminal/sparse on OOD)

## Context

**Why this plan.** The full-trajectory multi-turn GRPO harness now works (EXP-2026W28-009: on-policy all turns, per-turn potential credit, local vLLM 27B victim, CPU+GPU validated). But three Gate-2 rounds never *decided* dense vs sparse, and the OOD probe (EXP-2026W28-010) came back a **clean null**: the 20-step run's adapters were indistinguishable from the untrained base on OOD (base 8/48, dense 9/48, sparse 7/48 — all within a ±7pt CI). The root cause is not the method — it is that every run was **noise-dominated** (only 12 rollouts/step) and **unstable** (dense rose to ~0.38 then collapsed; grad-norm spiked to 76 with no KL), and Gate 2 measured *in-domain training reward* rather than H1's actual metric (**OOD ASR**).

**Goal.** Make a decisive H1 test affordable and trustworthy by fixing the three blockers in order: (1) rollout **throughput** (the prerequisite that lets us afford enough rollouts to beat the noise floor), (2) training **stability**, (3) a cheap **learning-gate** so we never spend the big run on a dead policy, then (4) the **powered, paired OOD eval** that is the real verification. Scope chosen by PI: **standard powered = 3 seeds/arm, n≥150 paired OOD goals, base(best-of-K)+dense+sparse, Holm across contrasts.**

**Discipline (unchanged).** Build + CPU-validate everything with NO GPU first; GPU needs the PI to restart the H20; the learning-gate (Step 3) precedes the powered run (Step 4). Serve the victim **without** `HF_HUB_OFFLINE` (the `IncompleteSnapshotError` lesson). RNG reproducibility bug already fixed in `extraction_multifield.py`.

## Step 1 — Batched turn-synchronized rollout (throughput; CPU-buildable, no GPU)

The trainer is fully sequential today (`h1_mt_grpo_train.py:154` loops `for i in range(G)`; each turn is one blocking `model.generate` + one blocking victim POST). Restructure to advance **all B = n_goals × G trajectories in lockstep**, batching the GPU generate and firing victim calls concurrently. Expected ~5–10× → afford 40–64 rollouts/step at similar wall-clock.

- **MODIFY `code/src/mt_rollout.py`** — add `rollout_batch(domain, items, gen_batch_fn, victim_batch_fn, *, T, tau)` that keeps `rollout_trajectory` intact (still used for CPU tests + as the correctness reference) and advances a list of trajectories together:
  - per-trajectory state (attacker_messages, conversation, victim_texts, phi_prev, `active` flag, turn records);
  - each turn: gather the **active** trajectories' `attacker_messages` → `gen_batch_fn([...])` returns raw texts (one batched generate) → `victim_batch_fn([(goal, conversation), ...])` returns replies (concurrent) → score each via `domain.score`, append per-turn record + phi, deactivate on `success` (Φ≥τ), else append `domain.feedback` for the next turn;
  - returns a list of per-trajectory dicts with the **same shape** `rollout_trajectory` returns.
- **MODIFY `code/scripts/h1_mt_grpo_train.py`**:
  - `gen_batch_fn`: **left-pad** the B tokenized prompts + attention mask → one `model.generate` → split outputs; record per-trajectory `(prompt_ids, resp_ids)` in per-trajectory buffers (so `turn_logprob` attribution stays correct). Reuse the batched-generate template at `code/scripts/h1_eval.py:60-67`.
  - `victim_batch_fn`: reuse **`code/src/llm_client.chat_batch`** (`llm_client.py:99`) or a `ThreadPoolExecutor` (as in `h1_grpo_train_extract.py:70`) over the B victim POSTs; the local vLLM victim's `--max-num-seqs 128` serves them in parallel.
  - handle variable-length trajectories via the `active` mask (early-break on success stays).
- **Do NOT serve the attacker via a second vLLM** — Explore confirmed it hits GPU-budget (victim already 0.45 util + QLoRA trainer on GPU 0) and on-policy-staleness (LoRA updates every step) problems. In-process batched HF generate keeps it on-policy and is simpler.
- **MODIFY `code/scripts/h1_mt_pipeline_test.py`** — add a CPU equivalence test: `rollout_batch` (mock gen/victim) yields the **identical** phi_traces/rewards as looping `rollout_trajectory` over the same items. This is the correctness gate before any GPU.

## Step 2 — Stabilize training (MODIFY `code/scripts/h1_mt_grpo_train.py`)

- **KL-to-reference penalty** (fixes the dense collapse). With PEFT the reference policy = **adapter disabled** — no second model needed: compute `logp_ref` under `with model.disable_adapter(): torch.no_grad()` and add `beta_kl * KL` per turn-example to the loss (k3 estimator `exp(Δ)-Δ-1`, Δ = logp_ref − logp_θ). `beta_kl≈0.02` (tunable). This is a **loss-form change → `method.md §3` addendum + changelog; PI approves via this plan.** The KL is identical for both arms, so it does not bias the dense-vs-sparse contrast.
- **Lower LR** 1e-5 → **3e-6**; keep grad-clip 1.0; **more steps** 60–80.
- **Bigger effective batch** (now affordable): `n_goals` 2→**8**, `G` 6→**8** (~64 rollouts/step).
- **Checkpoint the adapter at ~step 30** for the learning gate.

## Step 3 — Learning-check gate (cheap; before the powered run)

- Run ~30 steps for one arm, then eval the step-30 checkpoint vs base on the **IN-DOMAIN** split (small n≈40). **MODIFY `code/scripts/h1_mt_ood_eval.py`** to accept `--split {indomain,ood}` (it already loads either).
- **PASS** if the trained arm's in-domain ASR (or mean Φ) is meaningfully above base. **FAIL → stop and debug** (do not spend the powered OOD run). This is the single biggest cost-saver after three inconclusive rounds.

## Step 4 — Powered paired OOD eval (the H1 verdict)

- **MODIFY `code/scripts/h1_mt_ood_eval.py`**: add `--seeds` (repeat episodes/seeds), a **best-of-K** control for base (K≈4), and keep writing per-goal rows keyed by `goal.id` (already does; RNG now reproducible so all arms hit the **same** goals — paired).
- Run **base(best-of-K) + dense(3 seeds) + sparse(3 seeds)** on the **same n≥150 OOD goals**.
- **NEW `code/scripts/h1_mt_powered_analyze.py`** — reuse the existing stats, add only Holm:
  - `analysis.per_unit_rate` (`code/src/analysis.py:35`) → per-goal `{goal_id: rate}` per arm; `analysis.per_unit_best_of_k` (`analysis.py:50`) for the base control;
  - `analysis.bootstrap_diff` (`analysis.py:71`, paired, returns `excludes_zero`) for {dense−sparse, dense−base, sparse−base} on **ASR** and **mean-Φ**;
  - **NEW ~10-line Holm** helper (none exists in repo) to correct the 3-contrast family (bootstrap p-values → Holm).
- **H1 supported iff**: paired **dense−sparse OOD ASR** CI **excludes 0 and >0** after Holm, AND both trained arms beat base(best-of-K). Report the honest verdict either way.

## Files

- MODIFY `code/src/mt_rollout.py` (add `rollout_batch`), `code/scripts/h1_mt_grpo_train.py` (batched gen + concurrent victim + KL-to-ref + LR/steps/batch + step-30 ckpt), `code/scripts/h1_mt_ood_eval.py` (`--split`, `--seeds`, base best-of-K), `code/scripts/h1_mt_pipeline_test.py` (batched≡sequential test), `method.md §3` (KL addendum + changelog).
- NEW `code/scripts/h1_mt_powered_analyze.py` (+ a small Holm helper, in `analysis.py` or the script).
- REUSE (no new code): `llm_client.chat_batch`, `h1_eval.py:60` batched-generate template, `analysis.bootstrap_diff` / `per_unit_rate` / `per_unit_best_of_k`, `h1_serve_victim.py` + `remote.py` (serve/deploy), `mt_grpo.py` (advantages), `extraction_multifield.py` (frozen light victim), `h1_deploy_mt.py` (ship code).

## Compute budget

- Each training run = one arm × one seed; with the Step-1 throughput fix a 60–80 step run is ~2–3h → **well under the 20h/cycle** §10 ceiling. 3 seeds × 2 arms = **6 runs** (~12–18h total) + the powered eval — sequence them, **run a 5-step throughput smoke first to measure s/step** and size steps/seeds to fit; may span two H20 sessions (each run individually ≤20h, so no §10 breach). Shut the H20 after each session.

## Verification

- **CPU (no GPU):** `rollout_batch` ≡ `rollout_trajectory` equivalence test passes; `mt_grpo.py`, the `mt` pipeline test, and `extraction_oracle` goldens still pass; KL term sanity (adapter-disabled logprob differs from adapter-on).
- **GPU throughput smoke (5 steps):** measure rollouts/step + s/step; confirm **≥5×** vs the 450s/12-rollout baseline and no OOM (victim 0.45 + QLoRA trainer + KL-ref forward on one H20).
- **Learning gate:** step-30 in-domain ASR > base, else stop.
- **Powered verdict:** `h1_mt_powered_analyze.py` prints the paired dense−sparse OOD ASR diff + Holm-corrected CI → H1 supported / not, recorded as an EXP + Discussion post (W28 Weekly Retro also still owed).
