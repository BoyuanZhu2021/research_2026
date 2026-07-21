# H1 content-only transport repair v1

## Status and authorization

- User authorization: `2026-07-20`, explicit approval of the new content-only transport intent.
- Current position: `COT-P1/3` complete locally; H20 remains shut down.
- Only authorized powered instance: `20d84f9474-d7816b14`.
- Old H20 `fa85409945-b6dee8ab` and all V100 hosts remain forbidden.
- This is a new intent after three consecutive pre-victim crashes of the superseded wrapper prompt.
  It is not a fourth automatic retry of that intent.

## Frozen scientific setup

- Dataset: InjecAgent `ds/base,m=2`, with the existing 322 train / 69 calibration / 153 final OOD
  split and the existing high-information 12-train / 12-heldout panel.
- Attacker: pinned Qwen3.5-4B, NF4 double quantization, BF16 compute, direct in-process PEFT.
- Victim: pinned local Qwen3.5-9B FP8, canonical compact decision protocol.
- Interaction: each attacker turn creates one independent victim attempt; `T=5`, `max_calls=3`.
- Oracle: programmatic ordered tool-chain score `Phi in {0, 1/3, 2/3, 1}`; no LLM judge.
- Rewards: dense=`Delta Phi`; sparse=first terminal `Phi=1`; same optimizer, LoRA and schedule.
- Training shape: 12 steps per arm, 4 goals/step, 8 trajectories/goal, seed 0 first.
- No final-OOD model evaluation or response trace is permitted in this preliminary intent.

## Authorized transport change

The model-facing tool-use prompts no longer show the reserved transport-tag literals. They request
only plain injected Observation content and forbid any wrapper, delimiter, quotation, explanation or
metadata without naming the reserved tokens. The harness still:

1. fsyncs every raw attacker completion and SHA-256 before parsing;
2. treats ordinary non-empty model text as content-only payload;
3. owns the internal transport frame;
4. fails closed on an actual reserved tag, empty output or malformed evidence;
5. preserves raw token IDs/text and normalization metadata.

Legacy non-tool-use wrapper prompts and historical artifact parsers are unchanged.

## Pairing repair

Every paired run must now have the same deployment-tree SHA-256 in addition to the same seed,
initial LoRA SHA-256 and goal schedule. Therefore the old base-s0 cannot be paired with this prompt
tree. A fresh base-s0 is required and doubles as the only real-model transport canary.

## COT-P1/3 — local implementation (complete)

- Removed reserved-tag literals from initial system/task and retry prompts reachable by tool-use.
- Added prompt invariants over all train and calibration goal tasks plus both feedback paths.
- Added fail-closed paired deployment-tree identity validation.
- Passed tool-use Stage-0, legacy extraction Stage-0, MT pipeline, pilot tests, setup safety,
  py_compile and diff check.
- Exact local deployment plan: 166 files, tree
  `37c7e3fdc3e90d74411552780a71c7c7bd74596252cca1e563ecd58270cf5fcb`.

## COT-P2/3 — powered seed-0 panels

After the user opens instance `20d84f9474-d7816b14`:

1. Run fresh read-only preflight; if GPU UUID changed, fresh-bind the active H20 profile while idle.
2. Transactionally deploy the exact 166-file tree and pass all remote gates.
3. Start the canonical local FP8 victim.
4. Run a fresh base-s0. It must complete 48/48 trajectories with a complete raw attacker/victim
   ledger and zero reserved-tag transport errors. This is the sole transport canary.
5. Run dense-s0 paired to that fresh base, then sparse-s0 paired to both fresh base and dense.
6. Recover every panel and verify remote/local hashes before analysis.

Any crash, identity drift, incomplete ledger, non-finite gradient or missing artifact stops this new
intent. Do not silently repair or shrink its denominator.

## COT-P3/3 — preliminary H1 decision and closure

- Run the existing seed-0 analyzer on the fresh matched base/dense/sparse panels.
- A useful preliminary positive requires dense ASR > sparse ASR and dense ASR >= fresh base ASR.
- Only if seed 0 is directionally positive may the same exact tree run seed 1 for replication.
- This remains `decision_bearing=false`; it does not replace the preregistered final-OOD verdict.
- Recover and hash all evidence, stop exact project services, `sync`, shut down instance
  `20d84f9474-d7816b14`, and verify SSH is unreachable.

## Current next action

Wait for the user to open instance `20d84f9474-d7816b14`; then execute COT-P2 without adding a
separate smoke test or benchmark.
