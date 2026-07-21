# H1 direct in-process PEFT v1 plan

**Authorized direction:** PI reply “继续” on 2026-07-20 after static-start v3 closed.  
**Current position:** `IP-P3/4 complete — FAIL`; `IP-P4/4` is not unlocked and the H20 is shut down.  
**Only powered instance:** `20d84f9474-d7816b14`.  
**Expected current GPU UUID:** `GPU-14627e41-ad52-9967-0a52-bbd82009ef01` (observed 2026-07-20).  
**Forbidden:** old H20 `fa85409945-b6dee8ab` and every V100 host.

## Goal and frozen science

Repair the rollout correctness confounder before drawing another preliminary H1 conclusion. The
attacker must generate directly from the same live PEFT model whose LoRA parameters receive the
optimizer update. No adapter serialization, HTTP LoRA reload, vLLM LoRA name or serving cache lies
between optimizer state and rollout generation.

The science contract remains InjecAgent `ds/base,m=2`, 322 train / 69 calibration / 153 untouched
final OOD, `retrieve -> GmailSendEmail`, independent victim attempt per attacker turn,
`T=5,max_calls=3`, and programmatic `Phi in {0,1/3,2/3,1}`. Dense reward remains positive
`Delta Phi`; sparse remains the first `Phi=1` terminal event. Attacker remains pinned Qwen3.5-4B
NF4 double-quant/BF16 QLoRA `r=32,alpha=64`; victim remains the pinned local Qwen3.5-9B FP8
compact-decision service with raw HTTP response ledger. No LLM judge is introduced.

This plan may produce only a calibration-based preliminary result. It never sets formal
`h1_supported` and never opens the 153-goal final OOD split.

## IP-P1/4 — implementation and local gates

1. Reuse the production `make_gen_batch_fn`; generation temporarily enters eval mode, enables KV
   cache, runs under `no_grad`, and restores training mode. Formal training keeps the 256-token
   ceiling; the behavioral canary alone uses 32 tokens.
2. Add one in-process A -> B -> restored-A controller. One child process constructs the exact QLoRA
   model, makes four same-request/same-seed A generations, changes the first real text-path
   `lora_B` tensor to 0.25, makes B once, restores every LoRA byte and makes A once.
3. Preserve before audit: messages/request hash, exact prompt/response token IDs, decoded text,
   generation time, parameter hashes and training-mode restoration. The CPU parent preserves the
   complete worker log and seals PASS/FAIL/CRASH only after the child exits and the GPU is idle.
4. Register the controller in transactional deployment and the existing hash-verified recovery
   path. Pass targeted unit tests, deployment safety, static compilation, strict protocol lint,
   `git diff --check` and file-size hygiene.

## IP-P2/4 — one powered behavioral gate

Power on only instance `20d84f9474-d7816b14`, fresh-bind its currently observed H20 UUID, require
0 MiB, no compute process and closed ports 8000/8001, then deploy the exact locally hashed tree.
Run exactly one in-process canary on one exposed train case.

PASS requires: A is exact across four repeats; all six request hashes are identical; A/B/A LoRA
hashes change and restore; B changes the token/text fingerprint; restored A exactly restores it;
training mode is restored after every generation; six raw token/text rows and the worker/parent
envelopes are complete. FAIL or CRASH does not unlock training and is not retried automatically.
Expected cost is below 0.5 GPU-hour and 0.5 GiB.

## IP-P3/4 — bounded speed and one-step gate

Only PASS unlocks the already implemented cached-eval/local-victim probe on exposed train goals:
`8 goals x G=2 x T<=2`. It must retain all victim raw responses, produce at least one `Phi>0`, avoid
OOM/runtime drift and improve per-completion attacker time by at least 8x over the historical
1014.513 seconds / 320 completions slow path.

If the short probe passes, run one full-shape dense/seed-0 step (`8 goals x G=8 x T<=5`) including
real local victim, rewards, finite backward and optimizer update. Use that step to project a
two-pair 12-step-per-arm campaign plus five 69-goal calibration panels. The target is <=6 GPU-hours;
the standing automatic-execution ceiling is <=12 GPU-hours and <=5 GiB. Exceeding either standing
ceiling stops before training and requires PI input. OOM permits only symmetric throughput changes;
model, quantization, data, Oracle, reward and scientific batch stay fixed.

## IP-P4/4 — corrected preliminary H1 pilot and closure

If IP-P3 passes its standing budget gate, run in order:
`dense-s0 -> sparse-s0 -> dense-s1 -> sparse-s1`, 12 steps each. Same-seed arms must share exact
initial LoRA and goal schedule hashes. Evaluate matched single-sample base seeds 0/1 plus all four
adapters over all 69 calibration goals using direct in-process attacker generation and the same
local victim contract.

Report the preregistered three contrasts with Holm correction as
`PRELIMINARY_SUPPORTED / PRELIMINARY_NOT_SUPPORTED / INCONCLUSIVE / INVALID`, explicitly
`decision_bearing=false`. `PRELIMINARY_SUPPORTED` mirrors the formal sign/CI/adjusted-p conditions
for dense-sparse, dense-base and sparse-base but cannot set `h1_supported` because the data are
calibration-only and only two seeds are used.

Recover every canary/probe/benchmark/training/evaluation byte and verify remote/local SHA-256.
Stop exact project PID/PGID, `sync`, shut down `20d84f9474-d7816b14`, verify SSH unreachable, and
update EXP/Discussion/HANDOFF. Any canary FAIL/CRASH, invalid victim JSON, identity drift, nonfinite
gradient or incomplete registry skips the remaining powered stages and goes directly to recovery
and shutdown.

## Executed outcome (2026-07-20)

IP-P2 passed on one powered A→B→restored-A direct PEFT canary. IP-P3 then completed the exact
8×2×T<=2 local-victim probe but failed both release conditions: 2.228× speedup versus the required
8× and 0/16 trajectories with `Phi>0`. All 31 victim decisions were canonical-valid and their raw
responses were preserved; all 24 loaded fast-path layers passed. IP-P4 was therefore not run.
Artifacts were recovered and hash-verified, the victim was stopped, and instance
`20d84f9474-d7816b14` was synchronized and shut down. This terminal outcome does not evaluate H1.
