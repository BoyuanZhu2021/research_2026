# H1 v1 runtime behavioral canary

**Authorized:** 2026-07-19 EDT by PI reply “好的继续”  
**Plan position:** R3/3 complete — runtime gate FAIL, evidence recovered, H20 shut down  
**Only execution instance:** `20d84f9474-d7816b14`  
**Expected GPU UUID:** `GPU-018361b3-f812-a23d-e886-82d38e8501eb`

## Purpose

Resolve the two runtime confounders found after `EXP-2026W29-050` before spending more GPU time:

1. determine whether one fixed adapter and one byte-identical request are exactly reproducible when
   served serially;
2. prove behaviorally that vLLM same-name `load_inplace=true` serves new LoRA bytes rather than a
   stale adapter.

This is an engineering gate, not an H1 experiment. It is `decision_bearing=false`, does not change
the frozen dataset, Oracle, reward, training loss, model pins, or H1 statement, and must not read the
153-goal final OOD split.

## R1/3 — harness implementation

Upgrade the existing `h1_vllm_attacker_contract_smoke.py` rather than introduce a parallel service
interface. The smoke uses the production request and LoRA synchronization paths with one worker and
one fixed request seed:

1. serialize and load original adapter A;
2. submit the exact same request four times sequentially and require identical prompt token IDs,
   response token IDs, and text;
3. make a large, test-only mutation to one LoRA-B parameter, serialize/load adapter B under the same
   production adapter name, and require behavior to change;
4. restore the exact original parameter bytes, serialize/load A again, and require the original
   behavior to return exactly;
5. bind A-B-A in-memory parameter hashes, serialized tree hashes, `load_inplace` flags, request
   hashes, raw response hashes, token IDs, and output fingerprints in a sealed result.

Any mismatch fails closed. The mutation is restored in a `finally` block and is never a training
artifact. Raw HTTP responses remain in the append-only attacker ledger.

## R2/3 — one real H20 gate

Only after local tests pass:

1. connect only to `20d84f9474-d7816b14`; verify instance ID, expected GPU UUID, deployment tree,
   GPU inventory, ports, and absence of foreign compute processes;
2. if the host is off or identity drifts, stop without using another host;
3. transactionally deploy the changed harness while the service is stopped;
4. start only the exact attacker vLLM service, run the six-request behavioral canary, and stop the
   exact sealed PID/PGID immediately afterward;
5. recover the complete canary directory and service records and compare remote/local SHA-256.

PASS requires all of the following:

- four original-A outputs are behaviorally identical;
- all six request SHA-256 values are identical;
- B has different parameter/tree hashes and a different behavioral output;
- restored A has the original parameter/tree hashes and exact original behavioral output;
- B and restored A both use `load_inplace=true`;
- artifact identity and remote/local hashes are complete.

FAIL or CRASH keeps training locked. No automatic retries with changed seeds, prompts, mutation, or
concurrency are allowed.

## R3/3 — decision

- PASS: propose a separate, bounded 20–30-step matched-K checkpoint plan. Do not start it under this
  authorization.
- FAIL/CRASH: preserve the exact evidence, keep H1 training locked, and diagnose the serving stack.
- In either case, keep final OOD untouched. Shut down the H20 after evidence recovery unless the PI
  explicitly asks to keep it running for an immediately following approved action.

## Budget

The online gate uses six attacker generations and one QLoRA construction with no victim calls or
backward pass. It is expected to be far below one GPU-hour and 5 GiB of new artifacts. If setup or
execution projects beyond two GPU-hours, stop and report before continuing.

## Completion record (2026-07-20 EDT)

The four serial A requests were exactly reproducible. A large, text-path LoRA-B mutation changed both
the in-memory parameter hash and serialized tree hash, and restored A returned to both original hashes.
However, B produced exactly the same token IDs and text as A, so the behavioral reload condition failed.
The service was stopped, all evidence recovered with matching hashes, and the H20 was synced, shut down,
and verified SSH-unreachable. See `EXP-2026W30-001` and
`artifacts/h20-dual-local-campaigns/h1-runtime-canary-fail-audit-20260720.md`.
