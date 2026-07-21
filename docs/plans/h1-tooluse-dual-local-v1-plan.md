# H1 dual-local H20 execution snapshot

**Status:** P2 complete; valid v4 benchmark exceeds both frozen budget limits  
**Execution instance:** `20d84f9474-d7816b14` only  
**GPU:** `GPU-018361b3-f812-a23d-e886-82d38e8501eb`  
**Formal QLoRA:** `0/6`  
**Final OOD:** untouched (`0/153` reads)

## Why this runtime replaces the API path

The persistent Int4 vLLM attacker fixed the original 1014-second Transformers generation
bottleneck. Three subsequent full-shape runs then failed because SiliconFlow returned some
HTTP-200 responses whose JSON was cut off at the token limit. The raw responses are preserved, so
this is known to be a victim transport/output failure rather than an Oracle result.

The user authorized a local efficiency-first repair. The formal profile now runs two persistent
localhost services on the same H20:

- port 8000: pinned Qwen3.5-4B BitsAndBytes Int4 attacker rollout replica with per-step LoRA
  `load_inplace`;
- port 8001: pinned Qwen3.5-9B FP8 victim using the full Formal-Gate ReAct JSON protocol;
- 20% vLLM memory reservation per service, leaving 60% for the Transformers NF4 QLoRA
  trainer and backward pass;
- every victim request, exact raw HTTP response, parsed response, and parse result is appended and
  fsynced before it becomes training evidence.
- the dual-local boundary uses `max_tokens=2048` and constrains terminal `answer` plus declared
  action-argument string fields to at most 512 characters. Long `GmailSendEmail.body` values use the
  already-frozen same-attempt observation-reference branch, so exact retrieved data need not be
  re-emitted token by token. Tool names, required argument fields, independent parser, and Oracle are
  unchanged.

This is a new versioned runtime profile. The historical FP8 Gate transfers only the selected
`light` defense; it is not claimed as process-equivalence proof for this co-resident lifecycle.

## P2 completion gate

Run one exact dense/seed-0 step with `8 goals × G=8 × T<=5,max_calls=3`. It must produce:

1. a complete local victim ledger with no repaired/missing response;
2. non-zero programmatic Oracle progress and an actual optimizer step;
3. stable model, process, deployment, instance, and GPU identities;
4. projected P3-P5 use at or below 12 GPU-hours and permanent artifacts at or below 5 GiB.

Reproducible vLLM adapter copies are removed after their hashes are written; final adapters,
step-30 checkpoints, rollout traces, raw-response ledgers, and LoRA-sync ledgers remain.

## P3-P6 after a PASS benchmark

- P3: run `dense-s0 → sparse-s0 → dense-s1 → sparse-s1 → dense-s2 → sparse-s2`, 60 steps each.
  Same-seed pairs must have identical initial LoRA hashes and goal schedules.
- P4: evaluate base-K and all six adapters on all 69 calibration goals, marked
  `decision_bearing=false`.
- P5: first read and evaluate all 153 untouched final-OOD goals, then run the preregistered three
  Holm-adjusted contrasts and report `SUPPORTED / NOT SUPPORTED / INVALID`.
- P6: recover and hash all evidence, stop exact trainer/server process groups, `sync`, shut down
  instance `20d84f9474-d7816b14`, verify SSH loss, and update the protocol records.

No V100 or old H20 instance is part of this execution profile.

## Current execution stop — 2026-07-19 20:20 UTC

TileLang 0.1.12 and the loaded-model backward preflight passed. Runtime v4 also moved attacker
generation to chat completions with exhaustive reserved-tag decoder constraints and capped each
vLLM service at 20% memory. The exact dense/seed-0 benchmark then completed all 64 rollouts, 112
gradient examples, backward and optimizer step in 255.040 seconds.

The sealed projection is 33.456175 GPU-hours and 6,503,954,997 bytes (6.057 GiB), above the frozen
12-hour and 5-GiB limits. Its status is `BUDGET_REVIEW_REQUIRED`; P3 remains 0/6 and final OOD is
untouched. Exact project services were stopped, the instance was synced and shut down, and SSH loss
was verified. The only next action is a PI decision to approve the
higher budget or change the registered experiment scope. Evidence:
`artifacts/h20-dual-local-benchmarks/dual-local-fullshape-dense-s0-v4-20260719t201230z/`.
