# H1 six-hour pilot v1

**Authorized:** 2026-07-19 17:40 EDT  
**Plan position:** P6 complete; preliminary calibration verdict sealed, evidence recovered, H20 shut down  
**Only execution instance:** `20d84f9474-d7816b14`  
**Expected GPU UUID:** `GPU-018361b3-f812-a23d-e886-82d38e8501eb`

## Completion record (2026-07-19 EDT)

The two-pair amendment completed as registered: dense/sparse seeds 0 and 1 each ran eight steps;
base best-of-4 and four adapters then completed all 69 calibration goals. Formal panels consumed
2.381 GPU wall-hours. Exact offline Oracle replay returned `INCONCLUSIVE`: dense/sparse/base ASR
were 6.52%/7.25%/21.74%, so `preliminary_h1_supported=false`; formal `h1_supported` remains unset.
The 153 final-OOD goals were not read. Ten recovery manifests cover 157 files (1.746 GiB) with zero
local hash failures. Instance `20d84f9474-d7816b14` was synced, shut down, and verified
SSH-unreachable. See `LOGS/2026-W29.md#exp-2026w29-050`.

## Purpose and claim boundary

The PI replaced the 360-step formal campaign with a bounded pilot whose immediate purpose is to
obtain useful preliminary evidence about H1 within six wall-clock hours. The pilot may return
`PRELIMINARY_SUPPORTED`, `DIRECTIONAL_SUPPORT`, `INCONCLUSIVE`, or `INVALID`. It is
`decision_bearing=false`, never writes the formal `h1_supported` flag, never unlocks final OOD, and
does not claim that a calibration result is a definitive OOD result.

## Frozen science

- data identity remains InjecAgent `ds/base,m=2`: 322 train / 69 calibration / 153 untouched final
  OOD;
- target chain remains `retrieve -> GmailSendEmail`;
- every attacker turn creates a fresh victim attempt, with `T=5,max_calls=3` and no cross-attempt
  call stitching;
- the programmatic Oracle remains `Phi in {0,1/3,2/3,1}` with no LLM judge;
- dense reward remains positive change in running Phi; sparse reward remains first attainment of
  `Phi=1`;
- attacker remains pinned Qwen3.5-4B NF4 QLoRA/BF16 training plus a co-resident BitsAndBytes Int4
  vLLM rollout replica; victim remains pinned Qwen3.5-9B local FP8;
- LR, KL, LoRA, optimizer, clipping, 8 goals/step, G=8, attacker token budget, sampling seeds, and
  paired goal schedules remain unchanged.

## Bounded pilot shape

The original registry was `dense-s0, sparse-s0, dense-s1, sparse-s1, dense-s2, sparse-s2` in that
order. Each run has exactly eight registered rollout/training steps instead of 60; a sparse step
with no reward variance may be a sealed zero-gradient no-op rather than an optimizer update. Each
arm/seed therefore contributes 512 registered
training trajectories, and each arm aggregates 1,536 trajectories across three seeds. Same-seed
dense/sparse runs must have identical initial LoRA hashes and goal schedules.

The only evaluation is the already-exposed 69-goal calibration split: base best-of-4 plus one panel
for each of the six pilot adapters. The 153-goal final OOD split remains unread.

## Preliminary analysis

Replay every raw trace and recompute calls, Oracle states, Phi, and ASR. Aggregate trained arms over
three seeds by goal, compare against paired sparse and base best-of-4, and report the same three ASR
contrasts with paired bootstrap intervals and Holm adjustment:

1. `dense-sparse`;
2. `dense-baseK`;
3. `sparse-baseK`.

`PRELIMINARY_SUPPORTED` requires all three differences to be positive, exclude zero, and have
Holm-adjusted `p<0.05`. `DIRECTIONAL_SUPPORT` requires positive point estimates for dense-sparse and
dense-base but does not satisfy the strict three-contrast rule. Everything else is `INCONCLUSIVE`;
an incomplete identity, denominator, ledger, or Oracle replay is `INVALID`.

## Runtime repair and time budget

The first 60-step attempt failed before step 4 because the attacker echoed a reserved `<inject>`
tag in explanatory prose. The parser correctly failed closed. The repair keeps the parser and no-
repair policy unchanged: active prompts no longer spell the reserved tags, and vLLM generation uses
both all-case bad-word sequences and all-case decoder stop strings. A real service test must pass
before the pilot starts.

Measured early step time is 274 seconds. Forty-eight steps project to 3.66 hours; calibration
evaluation projects to 0.4-0.8 hours. Deployment, service restart, validation, analysis, recovery,
and shutdown receive the remaining 1.5 hours. If elapsed time makes a valid result impossible within
six hours, stop at a complete paired boundary, mark the pilot `INCONCLUSIVE`, recover evidence, and
shut down rather than touching final OOD or silently changing the denominator.

## Execution and stop conditions

1. pass local protocol/controller/safety tests and strict lint;
2. exact-stop the old dual services, transactionally deploy the new tree, and start the new exact
   runtime on `20d84f9474-d7816b14`;
3. pass reserved-tag online regression, then launch the persistent pilot driver;
4. complete six paired short runs and seven calibration panels;
5. run offline replay and preliminary analysis;
6. recover/hash/audit, exact-stop services, `sync`, shut down the instance, and verify SSH loss.

Any malformed output, identity drift, non-finite gradient, incomplete ledger, denominator mismatch,
or foreign GPU process fails closed. The old H20 and all V100 hosts remain forbidden.

## Fail-closed two-pair amendment (2026-07-19 EDT)

The first four runs completed as two exact dense/sparse pairs for seeds 0 and 1. `dense-s2` then
failed twice before any calibration outcome was observed. Both failures were HTTP-200 victim
responses whose nested decision JSON contained a generated `\b` escape; outer decoding converted
it into U+0008, so strict inner parsing correctly rejected the call. The second request carried the
registered `bad_words` guard, proving that the guard is not sufficient under real structured
decoding. A third blind retry is forbidden because it would approach the protocol's consecutive-
crash stop boundary without a demonstrated fix.

To preserve the PI's six-hour, preliminary-evidence objective, seed 2 is excluded symmetrically:
neither its incomplete dense run nor an unstarted sparse run enters the denominator. The calibration
profile is versioned as `h1-calibration-two-complete-pairs-v1` and evaluates only base best-of-4,
dense-s0, dense-s1, sparse-s0, and sparse-s1. This exclusion was fixed before calibration and is
therefore not outcome selection. The 153 final-OOD goals remain unread.

The analysis aggregates two paired seeds per arm. It keeps the three preregistered contrasts and
Holm adjustment but reports only calibration-level `PRELIMINARY_SUPPORTED`, `DIRECTIONAL_SUPPORT`,
or `INCONCLUSIVE`; the smaller seed count is an explicit limitation and cannot set formal
`h1_supported`. P4 completion now requires five complete, hash-bound panels plus offline Oracle
replay. P6 remains artifact recovery, audit, exact service stop, instance shutdown, and SSH-loss
verification.
