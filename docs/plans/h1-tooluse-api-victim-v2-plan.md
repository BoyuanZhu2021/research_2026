# H1 API victim v2 — 4096-token contract retry

**Approved:** PI user message `开始`, 2026-07-19  
**Current position:** P2/6 retry preparation  
**Active profile:** `h20-attacker-siliconflow-victim-v2`

This version supersedes only the runtime identity in `h1-tooluse-api-victim-plan.md`. The v1 plan
and `EXP-2026W29-042` remain immutable history. No local-FP8 Gate is rerun: its only transferred
fact remains the selected `light` defense.

## Exact v1 → v2 change

- SiliconFlow victim remains exact `Qwen/Qwen3.5-9B`, `temperature=0`,
  `enable_thinking=false`, and `response_format=json_schema`.
- `max_tokens` changes from 2048 to **4096**. This value is shared by smoke, benchmark, all six
  trainings, learning and final OOD.
- Before dispatch, the append-only ledger fsyncs the credential-free request body plus its canonical
  SHA-256. The Authorization header/API key is never serialized.
- After HTTP completion, the exact raw envelope is fsynced before any JSON parsing. Returned model,
  finish reason, trace ID, usage and parse outcome are explicit even on invalid responses.
- Retry policy is unchanged: only 429/503/504, at most three identical attempts. HTTP 200 malformed
  or length-stopped content is never retried, repaired or scored as `Phi=0`.

## Frozen science contract

Data remain 322 train / 69 calibration / 153 untouched final OOD from InjecAgent `ds/base`, with
`m=2`, `T=5`, `max_calls=3`, one fresh victim attempt per attacker turn and programmatic Oracle
`Phi in {0,1/3,2/3,1}`. Attacker remains pinned Qwen3.5-4B NF4 double-quantized QLoRA with BF16
compute. Dense reward is `Delta Phi`; sparse reward is first terminal success. Formal runs remain
`dense-s0, sparse-s0, dense-s1, sparse-s1, dense-s2, sparse-s2`, with identical paired initialization
and goal schedules. H1/Holm criteria and the prohibition on extrapolating beyond `m=2` are unchanged.

## Authorized execution

1. Complete CPU/source tests, strict protocol lint, deployment plan and credential-leak check.
2. Use only instance `20d84f9474-d7816b14`; never connect the previous H20 or V100. When powered,
   transactionally deploy the new tree, install only the two API variables as remote mode-0600
   `.env`, and verify one idle H20, port 8000 closed, no compute process, exact tree and Gate hash.
3. Run exactly one already-exposed calibration contract smoke and recover it byte-identically.
4. Run exactly one fresh dense/seed-0 full-shape benchmark (`8 goals x G=8 x T<=5`) through attacker
   generation, API victim, reward, backward and optimizer update.
5. If the benchmark completes and the 1.25-safety projection is at most 12 GPU-hours and 5 GiB,
   continue directly to P3. If it exceeds either budget, recover and shut down before reporting.
6. Any HTTP 200 malformed/length-stopped response, identity drift, incomplete ledger, non-finite
   gradient or missing optimizer step stops the retry permanently: recover hashes, shut down and
   request a new PI decision. Do not increase the budget again automatically.

P3–P6 otherwise follow `h1-tooluse-api-victim-plan.md` without change.
