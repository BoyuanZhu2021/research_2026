# H1 v2 runtime behavioral canary

**Authorized:** 2026-07-20 EDT by PI reply “继续”  
**Plan position:** V2-R3/3 complete — runtime gate FAIL, evidence recovered, H20 shut down  
**Only execution instance:** `20d84f9474-d7816b14`  
**Expected GPU UUID:** `GPU-018361b3-f812-a23d-e886-82d38e8501eb`

## Purpose

Resolve the failed same-name/same-ID adapter refresh observed in `EXP-2026W30-001`. The v1 canary
proved that the serialized LoRA bytes changed and restored, but vLLM generated the same output for a
large LoRA-B mutation. V2 therefore gives every adapter version a unique runtime identity and removes
the previous project adapter before loading the next one.

This remains an engineering gate, not an H1 experiment. It is `decision_bearing=false`, does not
change the frozen dataset, Oracle, reward, training loss, model pins, or H1 statement, and must not
read the 153-goal final OOD split. It does not authorize training.

## V2-R1/3 — harness repair

Reuse the existing production request and adapter synchronization paths:

1. derive each training adapter name from its registered step and serialized LoRA SHA-256;
2. before each load, list runtime models, fail if more than one project LoRA is present, explicitly
   unload the previous project LoRA by name, and verify it is absent;
3. load the new adapter with a unique name and `load_inplace=false`, then verify that exact name is
   present and bind all generation requests to it;
4. apply the same explicit-unload/unique-name lifecycle to evaluation adapters;
5. preserve raw vLLM responses and both request hashes: the full request hash includes the adapter
   name, while the semantic request hash excludes only that expected identity change;
6. cover the lifecycle and audit rules with local unit tests and static compilation.

## V2-R2/3 — one real H20 gate

After local checks pass:

1. connect only to `20d84f9474-d7816b14`; verify the expected GPU UUID, clean GPU inventory, closed
   project ports, absent foreign compute processes, and stopped project service;
2. transactionally deploy the changed harness while no project service is running;
3. start only the attacker vLLM service and run the six-generation A→B→A canary with one worker and
   a fixed request seed;
4. use three unique adapter names: original A, mutated B, and restored A; explicitly unload the
   previous name before each transition and never use `load_inplace`;
5. stop the exact sealed service PID/PGID immediately after the gate and recover the complete
   canary and control evidence with remote/local SHA-256 verification.

PASS requires all of the following:

- original A is exactly reproducible across four serial generations;
- all six semantic request SHA-256 values are identical;
- there are exactly three full request SHA-256 values, one per unique A/B/restored-A runtime name;
- the unload chain is `[none, original-A, mutated-B]`, and all three loads use `load_inplace=false`;
- B has different parameter/tree hashes and a different behavioral output;
- restored A has the original parameter/tree hashes and the exact original behavioral output;
- raw responses, runtime identity, and remote/local artifact hashes are complete.

FAIL or CRASH keeps training locked. There is no automatic retry with a changed seed, prompt,
mutation, concurrency, quantization, or model.

## V2-R3/3 — decision and closure

- PASS: the vLLM per-step adapter refresh path is behaviorally usable. Draft a separate bounded
  20–30-step matched-K H1 pilot; do not start it under this authorization.
- FAIL/CRASH: preserve exact evidence, keep H1 training locked, and diagnose or replace the serving
  mechanism before another training plan.
- In either case, keep final OOD untouched, stop exact project processes, run `sync`, shut down
  instance `20d84f9474-d7816b14`, and verify SSH becomes unreachable.

## Budget

The gate uses six attacker generations and no victim calls or backward pass. It is expected to stay
below one GPU-hour and 5 GiB of new artifacts. If setup plus execution projects beyond two GPU-hours,
stop and report before continuing.

## Completion record (2026-07-20 EDT)

The unique-name lifecycle worked as designed: three distinct adapter names, two successful explicit
unloads, three successful non-in-place loads, one semantic request hash, and three expected full
request hashes. A→B→A parameter and tree hashes changed and restored. Nevertheless, the large B
mutation produced the exact same text and token IDs as A, so the behavioral gate failed. Evidence was
recovered with zero hash mismatches; the service was stopped, and instance
`20d84f9474-d7816b14` was synced, shut down, and verified SSH-unreachable. See
`EXP-2026W30-002` and
`artifacts/h20-dual-local-campaigns/h1-runtime-canary-v2-fail-audit-20260720.md`.
