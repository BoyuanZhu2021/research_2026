# H1 API victim v3 — compact terminal contract through P6

**Approved:** PI instruction to continue the whole plan and autonomously debug through P6,
2026-07-19  
**Current position:** P2/6 implementation and one retry  
**Active profile:** `h20-attacker-siliconflow-victim-v3`  
**Only execution instance:** `20d84f9474-d7816b14`

This plan supersedes v2 for new execution while preserving v1/v2 plans and EXP records as history.
The existing local-FP8 Gate remains evidence only for selecting `light`; it is not rerun and does
not claim behavioral equivalence with the API victim.

## Exact v2 → v3 change

- SiliconFlow remains exact `Qwen/Qwen3.5-9B` with `temperature=0`,
  `enable_thinking=false`, `max_tokens=4096`, and JSON Schema response format.
- The API-only terminal branch changes from unbounded
  `{"kind":"final","answer":"<free text>"}` to exact `{"kind":"final"}`.
- The harness deterministically renders the latter as `Final Answer: [terminal]` for existing trace
  and feedback plumbing. No model-authored terminal text is recovered or invented.
- Tool-action branches, tool schemas, step-bound observation references, resolver, programmatic
  Oracle, `Phi`, dense/sparse rewards, data splits and H1/Holm decision rule are unchanged.
- The historical shared local-FP8 protocol is not modified. The API compact-terminal protocol has a
  separate ID and payload hash so v2 artifacts cannot unlock v3.
- Every request body and exact raw API envelope is still fsynced before parsing. HTTP 200 malformed
  content is not retried, repaired, or scored as `Phi=0`; API keys are never logged.

## P2 execution and budget gate

1. Run CPU protocol/integration/artifact/deployment gates, strict lint, diff and credential scan.
2. On instance `20d84f9474-d7816b14`, preflight an idle H20, closed port 8000, exact deployment tree
   and frozen defense-selection hash; transactionally deploy v3 and install only the two API
   variables as remote mode-0600 `.env`.
3. Run exactly one exposed-calibration compact-terminal smoke and recover all raw evidence.
4. Only after smoke PASS, run one dense/seed-0 full-shape benchmark:
   `8 goals × G=8 × T≤5`, including attacker generation, real API victim, reward, backward and one
   optimizer step.
5. Apply the registered 1.25 safety factor. At projected total `≤12 GPU-hours` and new artifacts
   `≤5 GiB`, continue directly to P3; otherwise recover and shut down.

## P3–P6 unchanged

- **P3:** `dense-s0 → sparse-s0 → dense-s1 → sparse-s1 → dense-s2 → sparse-s2`, each 60 steps,
  paired initial LoRA hash and goal schedule exact; NF4/double quantization/BF16 and all scientific
  settings frozen. OOM may change only symmetric throughput knobs.
- **P4:** base-K plus all six adapters on all 69 calibration goals; `decision_bearing=false` and no
  seed/hyperparameter selection.
- **P5:** only after complete authorization evidence, first read all 153 final OOD goals, replay raw
  traces, recompute the Oracle and execute the preregistered three-comparison Holm analysis.
- **P6:** recover and hash all artifacts, stop exact project processes, `sync`, shut down instance
  `20d84f9474-d7816b14`, verify SSH unreachable, and finalize EXP/Discussion/HANDOFF/lint.

Autonomous debugging may repair engineering faults while preserving model identities, data,
precision, Oracle, rewards, step counts, denominators and H1 criteria. Any change to those scientific
invariants still requires an explicit PI decision.
