# H1 v3 static-start LoRA behavioral canary

**Authorized:** 2026-07-20 EDT by PI reply “继续”  
**Plan position:** V3-S3/3 complete — `CRASHED_BEFORE_GENERATION`  
**Only execution instance:** `20d84f9474-d7816b14`  
**Expected GPU UUID:** `GPU-917cce20-b276-41bf-76e6-a24528ca0100`

## Purpose

Determine whether the failure in `EXP-2026W30-001..002` is confined to vLLM's dynamic LoRA
loading. V3 never calls the dynamic load or unload endpoints. Instead, it follows the documented
static server contract and supplies exactly one adapter through `--lora-modules name=path` when each
vLLM process starts.

This remains a `decision_bearing=false` engineering gate. It does not authorize H1 training, change
the frozen scientific contract, call a victim, read calibration, or read the 153-goal final OOD split.

## V3-S1/3 — isolated controller and local checks

1. Prepare original A, a deliberately large B mutation, and restored A in a short-lived QLoRA child
   process. The child must exit and the H20 must return to zero memory before serving starts.
2. Reuse the production attacker request builder so static and training transports cannot drift.
3. Register three independent static vLLM lifecycles. Each command, adapter tree, process PID/PGID,
   start ticks, environment, model registry, raw response, and stop record is sealed.
4. Add the controller and its tests to the transactional deployment manifest and GPU-guarded entrypoint
   list. Extend the existing recovery tool rather than introduce another transfer path.
5. Pass targeted unit tests, deployment safety tests, static compilation, diff checks, and the plan-only
   deployment manifest before connecting to the H20.

## V3-S2/3 — one real cross-process A→B→A gate

1. Connect only to `20d84f9474-d7816b14`; verify instance/GPU identity, 0 MiB, no compute process,
   closed ports 8000/8001, and no live project manifest.
   The 2026-07-20 powered-on preflight observed that the provider reassigned this instance from
   `GPU-018361b3-f812-a23d-e886-82d38e8501eb` back to its previously observed
   `GPU-917cce20-b276-41bf-76e6-a24528ca0100`; V3 is freshly bound to the latter identity and does
   not reuse proof from either earlier GPU lifecycle.
2. Transactionally deploy the exact locally hashed tree.
3. Prepare and seal the three adapters from one exposed train case, then run:
   - lifecycle 1: start with static original A and submit four identical serial requests;
   - lifecycle 2: fully restart with static mutated B and submit one identical semantic request;
   - lifecycle 3: fully restart with static restored A and submit one identical semantic request.
4. After every lifecycle, stop the exact sealed PID/PGID, require the port to close and GPU to return
   to zero memory, and preserve its separate log and stop record.

PASS requires all of the following:

- original A is exactly reproducible across four serial generations;
- all six semantic request SHA-256 values are identical and there are exactly three expected full
  request hashes, one per static model name;
- three lifecycle manifest hashes and three static LoRA names are unique;
- A→B→A parameter and adapter-tree hashes change and restore exactly;
- B changes the behavioral fingerprint, and restored A exactly restores the original fingerprint;
- all six raw responses, lifecycle records, and recovery hashes are complete.

FAIL or CRASH is final for this static vLLM route. There is no automatic change to prompt, seed,
mutation, model, quantization, or concurrency.

## V3-S3/3 — recovery and decision

- PASS: draft a separate 20–30-step matched-K pilot that fully restarts static attacker serving after
  each optimizer update. Do not start training under this authorization.
- FAIL/CRASH: retire vLLM LoRA serving for H1 and use direct in-process PEFT generation in the next
  costed plan.
- In either case, recover and hash all files, stop exact project processes, run `sync`, shut down
  `20d84f9474-d7816b14`, verify SSH-unreachable, and update EXP/Discussion/HANDOFF.

## Budget

The gate uses three vLLM starts, six attacker generations, one QLoRA construction, no victim calls,
and no backward pass. Expected cost is below one GPU-hour and 1 GiB of new artifacts. Stop and report
before continuing if execution projects beyond two GPU-hours or 5 GiB.

## Completion record

V3 ran once on 2026-07-20 as `vllm-static-lora-canary-20260720T053213Z`. The preparation child
loaded 426/426 Qwen3.5-4B weight shards but crashed before saving any adapter because the tokenizer
returned `tokenizers.Encoding` and the controller called `.detach()` on it. Consequently there were
0/3 static adapters, 0/3 vLLM lifecycles and 0/6 attacker requests/raw responses. This is a controller
compatibility crash, not a behavioral FAIL and not an H1 result.

The single recovered remote file passed its remote/local SHA-256 check. The current H20 was then
verified idle with closed ports and no project process, `sync` and shutdown returned 0, and the
post-shutdown SSH connection failed as expected. Under this plan's frozen CRASH rule, the static
route was not patched and retried. Evidence and analysis are recorded in
`LOGS/2026-W30.md#exp-2026w30-003` and
`artifacts/h20-dual-local-campaigns/h1-runtime-canary-v3-crash-audit-20260720.md`.
