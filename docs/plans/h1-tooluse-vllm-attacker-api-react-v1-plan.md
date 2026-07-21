# H1 vLLM attacker + API ReAct execution snapshot

**Status:** stopped at P2R-4 after three consecutive API-victim crashes  
**Execution instance:** `20d84f9474-d7816b14` only  
**Formal QLoRA:** `0/6`  
**Final OOD:** untouched (`0/153` reads)

## Frozen scientific setup

- InjecAgent `ds/base,m=2`: 322 train, 69 calibration, 153 final OOD.
- One independent victim attempt per attacker turn; `T=5,max_calls=3`.
- Programmatic `Phi ∈ {0,1/3,2/3,1}`; no LLM judge.
- Dense reward is running-score delta; sparse reward is first full success.
- Six 60-step NF4/double-quant/BF16 QLoRA runs remain paired by seed.
- The local FP8 Formal Gate selected `light`; its frozen file is defense-selection evidence, not
  proof that a provider API is behaviorally identical.

## Runtime repair that passed

The slow Hugging Face rollout generator was replaced by a persistent H20 vLLM 0.24.0 attacker using
BitsAndBytes Int4 and per-step LoRA `load_inplace`. The real two-adapter synchronization smoke passed,
and full-shape runs reached the victim batch in tens of seconds rather than spending 1014.513 seconds
in attacker generation.

## Runtime repair that failed

SiliconFlow `Qwen/Qwen3.5-9B` returned exact raw HTTP-200 envelopes, but some terminal decisions
repeated until the completion budget and ended before the root JSON object closed. The full benchmark
crashed at 1024 tokens, again at 2048 tokens, and again after adding a 512-character JSON-Schema bound
plus an explicit no-repetition instruction. See
`artifacts/h20-api-victim-benchmarks/vllm-attacker-api-react-crash-audit-20260719.md`.

## Locked next action

Do not launch a fourth API-victim attempt. Obtain PI confirmation, then implement one versioned
co-resident local profile: 4B Int4 vLLM attacker plus pinned 9B FP8 vLLM victim on separate localhost
ports, with smaller manifest-bound KV-cache allocations. Reuse the full Formal-Gate victim decision
protocol and local raw-response ledger. One exact full-shape benchmark must show legal traces,
positive `Phi`, an optimizer step, projected total `≤12 GPU-hours`, and artifacts `≤5 GiB` before P3.

P3–P6 remain unchanged after that gate: six registered QLoRA runs; 69-goal non-decision-bearing
learning report; first read and evaluate 153 final OOD goals with preregistered Holm contrasts; then
recover, audit, stop exact processes, sync, and shut down the H20 instance.
