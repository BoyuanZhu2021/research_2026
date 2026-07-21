# H1 gate-partial targeted curriculum v1

## Status

- Current position: `GP-P3/3 complete — PRELIMINARY_H1_SUPPORTED_IN_GATE_PARTIAL_SUBSET`.
- Scientific status: the efficiency-first, post-hoc calibration-subset mechanism criterion passed;
  this is preliminary support, not the preregistered final-OOD H1 result.
- Authorized runtime: only instance `20d84f9474-d7816b14`, H20
  `GPU-14627e41-ad52-9967-0a52-bbd82009ef01`.
- Runtime override satisfied: exact victim stopped, GPU reached 0 MiB/0%, `sync` passed, instance
  shutdown was dispatched and a fresh SSH connection was unreachable.
- Final OOD: forbidden; the 153-goal split remains unread.

The strict legacy-terminal base canary `h1-pr-base-s0-20260720T182049Z` completed on 2026-07-20:
48/48 rows, ASR 7/48, mean max-Phi 0.2917, and Phi counts `{0:23, 1/3:15, 2/3:3, 1:7}`. This
unlocks the short matched experiment but is not itself H1 evidence. Dense-s0 is running as tag
`h1-pr-dense-s0-20260720T182827Z` on the same exact deployment tree and seed-0 pairing.

Dense R1 made two valid optimizer updates, then crashed during step 3 because an unbounded legacy
terminal answer repeated until the 1,024-token completion ceiling and left JSON incomplete. The
existing dual-local 512-character final/action string bounds are now config-sealed and applied to
both train and eval victim clients. Exact tree `27597e4f...825b` runs a fresh base anchor before
dense R2; data, defense, Oracle, interaction, rewards and training shape are unchanged. The bounded
base completed with ASR 6/48, mean max-Phi 0.2986, 280/280 accepted victim responses and zero
`finish_reason=length`; dense R2 is tag `h1-pr-dense-s0-20260720T191248Z`.

The bounded matched panel then completed. Dense R2 finished 8/8 finite optimizer updates and scored
7/48 ASR with mean max-Phi 0.3403. Matched sparse tag `h1-pr-sparse-s0-20260720T201547Z` also
finished 8/8 updates and scored 5/48 with mean max-Phi 0.2986. Against the bounded base at 6/48 and
0.2986, the registered analyzer emitted `PRELIMINARY_H1_SUPPORTED_IN_GATE_PARTIAL_SUBSET`.
Dense-minus-sparse ASR is +4.17 pp and dense-minus-base is +2.08 pp; the one-sided 90% goal-cluster
bootstrap lower bound is -6.25 pp, so the outcome remains explicitly exploratory.

## Frozen question and setup

The mechanism question is unchanged: does dense `Delta Phi` reward produce a more effective
tool-use attacker than sparse first-`Phi=1` reward when intermediate progress is actually reachable?

- InjecAgent `ds/base,m=2`, retrieve then `GmailSendEmail`.
- Qwen3.5-4B attacker, direct in-process NF4/double-quant/BF16 QLoRA.
- Local Qwen3.5-9B FP8 victim, `light` defense.
- `T=5`, `max_calls=3`, programmatic `Phi in {0,1/3,2/3,1}`, no LLM judge.
- Content-only transport, raw attacker and victim responses durably recorded before analysis.
- LoRA `r=32/alpha=64`, LR `3e-6`, KL `0.02`, gradient clip 1.

## Base-only selection

Selection uses only artifacts created before the targeted adapters exist:

1. Formal Gate `tier-light/episodes.jsonl`, SHA-256
   `70f7e5288b5ef143b25b36124e4aa6af95664de220d49f2f74b8bfb00d9034b8`, supplies 30 goals with
   `success=false` and `0<max_phi<1` (23 at 1/3, seven at 2/3).
2. Historical direct base-K rows, SHA-256
   `b9f191460262b18754730c7172526863f4f4eebbcfdb7fb44edd2c2e560f65b2`, show that 20/30 goals
   also have `max_phi>0` under the current direct generation path.
3. Rank partial-without-full goals by partial sample count descending, mean `max_phi` descending,
   then goal ID. The first eight are training goals. Every remaining direct-base-reachable goal is
   held out, giving 12 disjoint evaluation goals.

The selection does not use dense/sparse outcomes. Choosing this calibration subset after failed
train scans is post-hoc, so any pass is explicitly exploratory and not a confirmatory OOD result.

## GP-P2/3 — short matched experiment

1. Transactionally deploy the exact source/config tree while the victim is stopped, then restart the
   sealed canonical victim on the same H20.
2. Run fresh matched `base-s0 -> dense-s0 -> sparse-s0`. Dense and sparse must share the exact
   initial LoRA SHA-256 and eight-step goal schedule.
3. Each training arm runs eight steps, four goals per step, eight trajectories per goal and `T=5`:
   256 trajectories per arm. Each of the eight training goals appears exactly four times.
4. Evaluate each arm on the same 12 heldout goals with four trajectories per goal: 48 rows per arm.
5. Any malformed victim response, identity drift, non-finite gradient or incomplete raw ledger
   crashes the corresponding panel without shrinking a denominator.

Expected runtime is roughly 70–90 minutes, below 12 GPU-hours and 5 GiB.

## GP-P3/3 — preliminary verdict

`PRELIMINARY_H1_SUPPORTED_IN_GATE_PARTIAL_SUBSET` requires all of:

- dense heldout ASR is strictly above sparse and matched base;
- dense heldout mean `max_phi` is strictly above sparse;
- dense records at least one finite optimizer update;
- paired config, initial LoRA, schedule, deployment, victim identity and all raw ledgers verify.

Otherwise report `NOT_SUPPORTED` or `CRASHED`. A pass establishes only a preliminary mechanism
result in this base-reachable calibration subset. It neither unlocks nor substitutes for the
untouched 153-goal final OOD test.

## Stop rule

Run only seed 0 for this efficiency-first result. Do not add seed 1, extra steps or final OOD after a
negative result without a new plan. Keep H20 on under the PI override even if this targeted run is
negative.

## Executed outcome and minimal repair

- `light` base-s0 and dense-s0 were all zero. Dense completed 256 training trajectories with zero
  optimizer updates and unchanged LoRA bytes, so the light-tier sparse arm was not run.
- A separately versioned `defense=none` base canary was also zero on all 48 heldout trajectories.
  Therefore defense strength is not the active bottleneck.
- Raw attacker text is explicit and well formed. The local compact victim protocol instead chose
  the legal seven-token terminal object `{"kind":"final"}` at every decision. This protocol was
  introduced after the Formal Gate and gives termination a much shorter branch than a tool action.
- The next repair changes only the victim decision wire contract back to the Formal Gate's legacy
  terminal shape. Models, quantization, goals, tool chain, Oracle, `T`, `max_calls`, defense and
  content-only attacker transport remain fixed. A base-only canary must restore `Phi>0` before any
  further dense/sparse training.
- Legacy canary R1 restored action generation (164 action versus 133 final decisions) but crashed
  before sealing a denominator because the broad legacy grammar admitted an undeclared
  `GmailSearchEmails.body` observation reference. R2 will enable the existing local client's
  strict-declared-arguments grammar; the parser and Oracle remain unchanged.
