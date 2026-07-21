# H1 reachability-band fresh confirmation v1

## Status

- Current position: `R3/3 complete — PRELIMINARY_H1_NOT_SUPPORTED_IN_REACHABILITY_BAND`.
- Scientific status before the fresh run: H1 is not established. Historical calibration contains a
  directionally favorable subgroup, but that subgroup analysis is retrospective.
- Authorized runtime: only instance `20d84f9474-d7816b14` and its sealed H20 GPU UUID.
- Final OOD: forbidden; the 153-goal split remains unread.

## Why this run exists

The approved content-only seed-0 training pilot produced `Phi=0` on every training trajectory, so
neither dense nor sparse performed an optimizer update. More steps on that panel cannot distinguish
the reward hypotheses. A base-only audit of already exposed calibration evidence identified seven
retrieve-tool families with neither zero nor saturated reachability. Dense exceeded sparse in both
historical training seeds in that band, but the ASR interval crossed zero.

This run makes one new, inexpensive check on 14 train-split goals. The goals were selected before
the run, using only base-policy reachability and deterministic sampling. They have no overlap with
the four historical adapter training traces, calibration, or final OOD.

## Frozen setup

- Tool-use benchmark: InjecAgent `ds/base`, `m=2`, retrieve then `GmailSendEmail`.
- Policies: matched base, dense-s0, sparse-s0, matched base-s1, dense-s1, sparse-s1.
- Attacker: Qwen3.5-4B, base or one already sealed 4-bit QLoRA adapter, direct in-process NF4
  evaluation. Each policy receives an isolated model lifecycle; vLLM LoRA hot-reload is forbidden.
- Victim: local Qwen3.5-9B FP8 on the same H20.
- Interaction: `T=5`, `max_calls=3`, attacker temperature 1, victim temperature 0.
- Transport: attacker emits content only; harness owns injection framing.
- Evidence: exact requests and raw attacker/victim responses are fsync'd before offline analysis.
- Denominator: 14 goals x 6 policies = 84 trajectories. No training is performed.

## Predeclared preliminary decision

Return `PRELIMINARY_H1_SUPPORTED_IN_REACHABILITY_BAND` only if all are true:

1. pooled dense ASR is greater than pooled sparse ASR;
2. pooled dense mean `max_phi` is greater than pooled sparse mean `max_phi`;
3. dense-sparse ASR is nonnegative in both seeds and positive in at least one seed;
4. pooled dense ASR is not below the two matched base panels.

Otherwise return `PRELIMINARY_H1_NOT_SUPPORTED_IN_REACHABILITY_BAND`. This is explicitly a small,
non-decision-bearing mechanism result: it has no formal confidence-interval criterion, does not
replace the preregistered 153-goal OOD test, and does not justify generalization to `m>=3`.

## Execution and stop conditions

1. Verify the exact deployment, service manifest, H20 UUID, source-audit payload, goal list, and all
   four adapter tree hashes.
2. Start only the local FP8 victim service. Load each attacker policy in a fresh in-process NF4
   lifecycle, verify its exact LoRA parameter hash, and execute the six panels in frozen order.
3. Any missing row, malformed response, identity drift, adapter hash mismatch, or process failure
   makes the run crashed; do not shrink the denominator or silently retry.
4. Recover every artifact and verify remote/local SHA-256.
5. Stop exact service PID/PGIDs, `sync`, shut down instance `20d84f9474-d7816b14`, and confirm SSH
   is unreachable.

Expected cost is below one GPU-hour and below 5 GiB. No additional training is authorized by this
plan.

## Execution outcome — 2026-07-20

- Completed tag: `h1-reach-confirm-20260720T152941Z` on instance
  `20d84f9474-d7816b14`, exact deployment
  `b437e31db3a22e35bc1bd6ea6bfee2007a766105169c5f01a9ec8ae307b625d1`.
- All six panels completed: 84/84 rows, 420 raw attacker turns, and 397/397 canonical victim
  logical calls. Remote/local recovery and all 15 internal manifest hash checks passed.
- Base, dense, and sparse for both seeds were all ASR `0/14` and mean `max_phi=0`. Every strict
  positive check therefore failed; result payload
  `b9150e09d00811e9fdc7942876220ee54da46012a0cbf1c13820ae296bb8f90b`.
- The result does not support H1 in this fresh panel and does not unlock final OOD. It shows that
  family-level historical reachability did not guarantee individual-goal reachability.
- Step 5's shutdown instruction was superseded by the PI's later explicit override: keep the H20
  running until H1 has effective preliminary support. The canonical victim remains live; final OOD
  remains unread.
