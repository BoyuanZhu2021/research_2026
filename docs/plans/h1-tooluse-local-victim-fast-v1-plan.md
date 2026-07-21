# H1 local-victim fast v1 repair plan

**Approved direction:** PI/user directive on 2026-07-19 permits a faster attacker runtime,
low-precision local victim, workflow simplification, and debugging.  This snapshot replaces the
API-victim v3 runtime for future work; it does not rewrite the historical API benchmark.

**Current position:** `P2R-2/7` — implementation and CPU contracts are in progress. Formal QLoRA
is `0/6`; the 153-goal final OOD split is untouched.

## 1. What failed

The full-shape API benchmark `api-victim-fullshape-dense-s0-20260719T151028Z` completed one
registered rollout step but required 1068.995 seconds. Of that, the 4B attacker generation used
1014.513 seconds and the API victim plus harness used only 54.346 seconds. The step projected to
146.498 GPU-hours and therefore failed the 12-hour budget gate.

The same run also had no learning signal: all 320 victim calls were canonical-valid compact JSON,
but every call selected the terminal branch, all 64 trajectories had five zeros in their Phi trace,
and `optimizer_step=false`. Therefore moving only the API or changing only victim latency cannot
repair the experiment.

The controller log gives two concrete attacker causes:

1. QLoRA preparation enabled gradient checkpointing and set `use_cache=false`; rollout generation
   then ran while the model remained in training mode, so Transformers disabled the autoregressive
   KV cache and recomputed the growing prefix at every output token.
2. The Qwen3.5 Gated DeltaNet fast libraries were absent, so Transformers used its documented slow,
   memory-hungry PyTorch fallback. The failed run also reserved 78.8 GB while only 17.0 GB was live,
   which is hostile to a co-located victim.

The attacker was already NF4 4-bit with double quantization and BF16 compute. “Changing it to
int4” is not a new optimization.

## 2. Frozen science contract

This repair may change runtime engineering but not the experiment being tested:

- Data remain 322 train / 69 calibration / 153 untouched final OOD from InjecAgent `ds/base`,
  restricted to `m=2` and the `retrieve -> GmailSendEmail` chain.
- Formal interaction remains `T=5,max_calls=3`; each attacker turn starts an independent victim
  attempt and calls from different attempts are never stitched.
- The programmatic Oracle remains the only judge and returns
  `Phi in {0,1/3,2/3,1}`. Dense reward is positive `Delta Phi`; sparse reward is the first
  `Phi=1` terminal event.
- Formal training remains dense/sparse by seeds 0,1,2, 60 steps each, with 8 goals, group size 8,
  attacker max-new-tokens 256, and paired initial LoRA/schedule identity.
- H1 and its Holm-corrected three-contrast decision rule do not change. No result extends to
  `m>=3`.

## 3. New runtime profile

Profile ID: `h20-attacker-cached-nf4-local-fp8-victim-v1`.

- Authorized hardware remains only instance `20d84f9474-d7816b14`; old H20
  `fa85409945-b6dee8ab` and every V100 are forbidden.
- Attacker remains pinned `Qwen/Qwen3.5-4B`, NF4 4-bit, double quantization, BF16 compute, LoRA
  `r=32/alpha=64`. Backward keeps gradient checkpointing.
- During rollout generation only, the same model temporarily enters evaluation mode, uses
  `use_cache=true`, runs under `no_grad`, then restores the exact prior training state. After
  generated tensors are copied to CPU, `torch.cuda.empty_cache()` releases unused allocator
  reservations before local victim calls. Sampling seed, temperature, top-p and 256-token ceiling
  remain unchanged.
- The attacker environment must pin `fla-core==0.5.1` and
  `causal-conv1d==1.6.2.post1`. A loaded-model preflight requires every Qwen3.5 Gated DeltaNet layer
  to bind causal-convolution, chunk-delta and recurrent-delta fast callables; package installation
  alone is insufficient evidence.
- Victim is the same pinned `Qwen/Qwen3.5-9B` served locally by the existing vLLM 0.24.0 H20
  launcher in FP8. The historical Formal Gate contributes only the selected `light` defense and
  evidence that local victim partial/full behavior is reachable; a new runtime profile does not
  claim byte-equivalence with that Gate lifecycle.
- The local victim uses a separate compact-terminal JSON protocol: actions are unchanged and the
  terminal wire object is exactly `{"kind":"final"}`. The harness renders `[terminal]`; the model
  cannot spend tokens on a terminal answer.
- Every local request is logged before sending. The exact raw HTTP response is fsynced before JSON
  or canonical decision parsing, followed by parsed envelope and canonical parse outcome. HTTP 200
  malformed JSON is not retried, repaired, or scored as zero.

## 4. Execution gates

### P2R-1 — offline diagnosis and plan

Complete when the failed benchmark is audited, runtime causes are tied to code/log evidence, and
this versioned plan plus HANDOFF/Discussion pointers are written.

### P2R-2 — implementation and CPU contracts

Complete when cached generation, compact local protocol, raw local ledger, fast-kernel preflight
and the short probe pass CPU tests, deployment dry-run, strict protocol lint, diff and credential
checks. Historical profiles must retain their prior defaults.

### P2R-3 — one short H20 probe

Deploy transactionally, install the two pinned kernel distributions, validate their functions on
the loaded 4B model, start the exact local 9B FP8 service, and run only the non-decision-bearing
shape `8 goals x G=2 x T<=2` on exposed train goals. The probe passes only if:

- all local victim calls have complete raw-response ledgers and canonical decisions;
- attacker generation is at least 8x faster per completion than EXP-045's failed benchmark;
- at least one trajectory reaches `Phi>0`;
- no identity drift, OOM, non-finite value, or allocator/victim collision occurs.

Failure branches are bounded. Missing/broken fast kernels stop before sampling. OOM permits only a
symmetric generation-chunk/concurrency reduction. Speed below 8x permits one throughput-only
diagnostic (chunk 16/32/64, same prompts/seeds); no smaller scientific shape becomes formal
evidence. Zero reachability stops the local profile rather than forcing tool-only output.

### P2R-4 — one formal full-shape benchmark

Only a passing short probe unlocks one exact `8 x G=8 x T=5` dense/seed-0 step including real local
victim, rewards, backward and optimizer update. A valid benchmark requires nonzero advantage,
finite gradient, `optimizer_step=true`, complete artifacts, total projection <=12 GPU-hours, and
new artifacts <=5 GiB. It is the only artifact that can unlock training.

### P3–P6 — unchanged experiment

Run six paired QLoRA arms; produce the non-decision-bearing 69-goal learning report; unlock and
evaluate the untouched 153-goal final OOD only after complete registry evidence; apply the frozen
Holm decision; recover and hash all artifacts; exact-stop the project processes, `sync`, shut down
instance `20d84f9474-d7816b14`, and verify SSH is unreachable.

## 5. Audit files

- Historical failure: `LOGS/2026-W29.md#exp-2026w29-045`
- Historical raw benchmark audit:
  `artifacts/h20-api-victim-benchmarks/api-victim-fullshape-dense-s0-20260719T151028Z.audit.md`
- Cached generation: `code/src/generation_runtime.py`
- Local compact decision: `code/src/local_victim_decision_protocol.py`
- Local raw ledger/profile: `code/src/local_vllm_victim.py`
- Fast-kernel preflight: `code/src/qwen35_fast_kernels.py`
- Short probe: `code/scripts/h1_local_fast_probe.py`
