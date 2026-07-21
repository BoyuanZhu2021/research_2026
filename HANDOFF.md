# HANDOFF — live state and next action

> Read this file immediately after `python tools/session_check.py`. It is the current pointer only;
> detailed history lives in LOGS, Discussion and versioned plans.

**Last updated:** 2026-07-21
**Mode:** newbie
**Active issue:** `DISC-2026W28-001`
**Current plan position:** `H1-CP3/5 / CE-P3/4 — fresh learning: 3/7 recovered, dense-s1 running`
**Only authorized instance:** `20d84f9474-d7816b14`
**Current GPU:** `GPU-14627e41-ad52-9967-0a52-bbd82009ef01` (H20)

## Current state

- All six formal adapters are trained and recovered: `dense/sparse × seeds {0,1,2}`. The registered
  base tags also exist for seeds 0/1/2.
- The constrained-eval deployment tree on the H20 is
  `989df391d07dcd30d905442a9014633e9f42278cee69d191bcb1165122b2df7f` (95 files).
- The canonical victim is pinned Qwen3.5-9B FP8, served locally on port 8000. The attacker is pinned
  Qwen3.5-4B NF4 double-quantized PEFT. All remote work is on the single instance above.
- Fresh constrained-eval learning campaign `h1-confirm-learning-20260721T071124Z` on instance
  `20d84f9474-d7816b14`:
  - `base-k4`: COMPLETE and recovered, 276/276 rows, 23 successes, ASR 0.0833333,
    mean max-Phi 0.185990, wall 2317.95 s.
  - `dense-s0`: COMPLETE and recovered, 69/69 rows, 5 successes, ASR 0.0724638,
    mean max-Phi 0.202899, wall 596.20 s.
  - `sparse-s0`: COMPLETE and recovered, 69/69 rows, 10 successes, ASR 0.144928,
    mean max-Phi 0.222222, wall 545.89 s; the registered decoder guard masked 199 candidate slots.
  - `dense-s1`: RUNNING as PID `98210`; `sparse-s1`, `dense-s2`, and `sparse-s2` remain pending.
- Historical pre-guard learning campaign `h1-confirm-learning-20260721T032302Z`:
  - `base-k4`: COMPLETE and recovered, 276 rows, 15 successes, ASR 0.0543478,
    mean max-Phi 0.173913, wall 2357.34 s.
  - `dense-s0`: COMPLETE and recovered, 69 rows, 8 successes, ASR 0.1159420,
    mean max-Phi 0.227053, wall 554.36 s.
  - `sparse-s0`: CRASHED before a complete denominator. One raw prompt-echo row contained the
    reserved literal `<inject>`; 63 attacker and 152 victim raw events were recovered with matching
    remote/local SHA-256.
- Final OOD remains unread and locked. No H1 decision is available.
- The H20 instance remains powered on under the PI instruction not to shut it down before H1 is
  effectively tested. After constrained-eval deployment, the Qwen3.5-9B FP8 victim was restarted as
  PID `74630`, manifest payload `78b0dc50f469ac3da69b0817670e6288a6e60ce35b51aa4a8287f89d268efc4b`.
- Two cleanup passes removed a net 2,301,533,206 bytes and reduced the active surface to 31 scripts,
  52 src files and three artifact top-level directories. Historical source/evidence remains in
  verified ZIP archives; formal adapters and current raw ledgers remain loose and unchanged.
- The compact local deployment candidate has 94 files and tree SHA-256
  `9a20719e3d1c4f043f6dd8405aa6508bae528c83ad81ac62b53298bda0bd3bae` and is now historical.
  The deployed constrained-eval candidate has 95 files and tree SHA-256
  `989df391d07dcd30d905442a9014633e9f42278cee69d191bcb1165122b2df7f`.

## Resolved blocker and remaining gate

The PI authorized the versioned constrained-eval intent. Local implementation is complete: a
text-aware DFA masks a token before it completes any of the 128 ASCII-case reserved tags, including
right-fused tokenization. It is applied only to evaluation and symmetrically across the complete
base/dense/sparse grid. Raw text, raw response token IDs and raw victim responses remain preserved;
the parser remains fail-closed and performs no repair.

The registered training and heldout raw attacker ledgers for all seed-0/1/2 base/dense/sparse runs
have zero reserved-tag matches. Thus the minimal recommended repair retains the sealed adapters,
adds the decoder guard symmetrically for evaluation, and reruns the complete seven-panel learning
grid under a fresh campaign ID. It does not mix old and new panel results.

A tokenizer-only feasibility probe proved that a plain list of 128 or 256 `bad_words_ids` sequences
is unsafe: Qwen fuses `>` with right-side punctuation, leaving 7,104/10,752 representative contexts
uncovered. The local DFA implementation, transactional deployment and real pinned-Qwen tokenizer
gate now pass: all 10,752 token paths were masked before tag completion. Guard payload is
`f3d3081858b7de07181d4cf2b7c3e9ae4b041658b59fbe1b3744a179756ccff4`; transition table is
`7dd9516e90864d0b84def89e1f01857ae0fd413535829b955b05dea9640d2a55`.

## Single next action

1. Run a fresh learning campaign in order
   `base-k4, dense-s0, sparse-s0, dense-s1, sparse-s1, dense-s2, sparse-s2`;
2. Require a complete non-decision-bearing 69-goal report before creating the already authorized,
   hash-bound final-OOD authorization;
3. Run the untouched 153-goal final OOD, registered Holm analysis, recovery and shutdown.

Do not connect the old H20 `fa85409945-b6dee8ab` or any V100 host. Do not read final OOD before the
learning report and authorization validate.

## Key evidence

- Plan: `docs/plans/h1-tooluse-confirmatory-ood-v1.md`
- Constrained-eval plan: `docs/plans/h1-tooluse-constrained-eval-v2.md`
- Current crash: `LOGS/2026-W30.md#exp-2026w30-029`
- Decoder feasibility: `LOGS/2026-W30.md#exp-2026w30-030`
- Crash audit: `artifacts/h20-confirmatory/h1-confirm-learning-20260721T032302Z/sparse-s0-crash/crash-audit.md`
- Completed learning panels: `artifacts/h20-confirmatory/h1-confirm-learning-20260721T032302Z/`
- Local cleanup audit: `docs/h1-cleanup-20260721.md`
