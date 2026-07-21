# H1 individual-goal reachability prospect v1

## Status

- Current position: `IR-P2/3 complete — INSUFFICIENT_PARTIAL_GOALS; IR-P3 locked`.
- Scientific status: H1 is not established. The previous 14-goal family-selected confirmation was
  complete but all-zero (`EXP-2026W30-019`).
- Authorized runtime: only instance `20d84f9474-d7816b14` and H20
  `GPU-14627e41-ad52-9967-0a52-bbd82009ef01`.
- Runtime override: keep the H20 and canonical victim on until H1 has effective preliminary support.
- Final OOD: forbidden; the 153-goal split remains unread.

## Failure this plan addresses

The previous panel selected seven retrieve-tool families using historical base reachability, then
sampled two fresh train goals from each family. Family-level reachability did not transfer to those
specific goals: all 84 base/dense/sparse trajectories had `max_phi=0`.

This plan changes only the selection granularity. It first measures base-policy reachability on
individual, still-unexposed train goals. Dense and sparse adapters are not loaded or inspected until
the partial-goal set and fresh confirmation seeds are sealed.

## Frozen setup

- Dataset/domain: InjecAgent `ds/base,m=2`, retrieve then `GmailSendEmail`.
- Attacker: pinned Qwen3.5-4B, direct in-process NF4, double quantization, BF16 compute.
- Victim: pinned local Qwen3.5-9B FP8 with the existing `light` defense.
- Interaction and Oracle: unchanged `T=5`, `max_calls=3`, programmatic
  `Phi in {0,1/3,2/3,1}`, no LLM judge.
- Transport: content-only attacker output with harness-owned injection framing.
- Candidate pool: the 322-goal train manifest minus 107 historical adapter-training goals and the
  14 goals used by the failed fresh confirmation, leaving exactly 201 goals. Calibration and final
  OOD are not candidates.
- Prospect seed: 19. Raw attacker and victim responses are fsync'd before parsing/analysis.

## IR-P2/3 — base-only prospect scan

1. Verify the exact deployment, service manifest, source audit, historical rollout hashes, H20 UUID,
   201-goal candidate count and candidate-list SHA-256.
2. Load only the base 4B policy. Run one trajectory per candidate with the frozen setup.
3. Select in train-manifest order only rows satisfying `success=0` and `0<max_phi<1`; cap at 16.
4. Return `PROSPECT_READY` only if at least 8 goals qualify. The selection artifact must declare
   `base_policy_only=true`, `dense_sparse_outcomes_read=false`, and `final_ood_read=false`.
5. Any malformed response, missing row, identity drift or incomplete raw ledger crashes the scan;
   do not shrink the denominator or repair responses.

## IR-P3/3 — fresh paired confirmation

This phase is locked until a recovered, hash-verified `PROSPECT_READY` selection exists. Before any
adapter evaluation, freeze two new generation seeds that differ from prospect seed 19 and from the
adapter training seeds. Then evaluate matched base/dense/sparse panels on exactly the selected goal
IDs, with isolated model lifecycles.

Preliminary H1 support still requires pooled dense ASR and mean `max_phi` to exceed sparse, both
seed-level dense-sparse ASR differences to be nonnegative with at least one positive, and pooled
dense ASR not below matched base. A pass is exploratory mechanism evidence only; it does not unlock
or replace the preregistered 153-goal final OOD test.

## Budget and stop conditions

- Expected prospect scan: 201 trajectories, below one GPU-hour and below 5 GiB.
- If fewer than 8 partial goals exist, stop this path and report `INSUFFICIENT_PARTIAL_GOALS`; do not
  inspect dense/sparse outcomes.
- If projected total exceeds 12 GPU-hours or 5 GiB, stop before the paired confirmation.
- Keep the H20 on under the PI's current override; do not connect the old H20 or any V100.

## Outcome

- The first launch, tag `h1-individual-prospect-20260720T160130Z`, failed before model loading
  because generation seed 19 was incorrectly reused as the formal construction seed; the harness
  permits construction seeds only in `{0,1,2}`. It produced 0/201 trajectories and is recorded as
  `EXP-2026W30-020`.
- The repaired launch used construction seed 2 while preserving generation seed 19. Tag
  `h1-individual-prospect-20260720T160758Z` completed 201/201 base trajectories, but every row had
  `max_phi=0`: ASR 0/201, mean max_phi 0 and partial candidates 0. Raw evidence includes 1,005
  attacker turns and 944/944 canonical-valid victim calls. See `EXP-2026W30-021`.
- The predeclared minimum of eight partial goals was not met, so `IR-P3/3` is not unlocked and this
  plan terminates as `INSUFFICIENT_PARTIAL_GOALS`. Dense/sparse outcomes were not read during the
  selection, and the final OOD split remains unread.
