# H1 final-only C0 transport post-exposure confirmation

**Approved by PI:** 2026-07-21

> “批准 `h1-victim-final-c0-canonicalization-v1`，我立即实现回归测试、部署新
> profile，完整重跑七组 learning，创建新 final 授权并完成 153 条 post-exposure OOD。”

## Scope

- Authorized instance: `20d84f9474-d7816b14` only.
- Preserve the frozen 322 train / 69 calibration / 153 final-OOD split, six sealed adapters,
  base-K4, light defense, attacker decoder guard, interaction shape and programmatic Oracle.
- The new final result is a **post-exposure confirmation**. The first untouched read was consumed by
  engineering-invalid campaign `h1-confirm-final-20260721T090531Z`.
- Do not reuse that campaign or its authorization; do not mix panels across profiles/campaigns.

## Registered transport

Transport ID: `h1-victim-final-c0-canonicalization-v1`  
Policy payload SHA-256: `6fb06e5896d28fd05f097f50ef98cbd7752773e6c1fe86795090e6e4361e5ebd`

1. Persist the exact outer HTTP response before parsing.
2. Parse nested content strictly first; clean responses stay byte-identical.
3. Only after `JSONDecodeError(msg="Invalid control character at")`, allow tolerant parsing when
   the semantic object has exactly `answer` and `kind`, `kind="final"`, and a bounded string answer.
4. Canonicalize that same semantic object to strict JSON, and record original/canonical hashes,
   policy identity, error position and control codepoints.
5. Action decisions, tool names, arguments, structural errors, identity drift and length stops remain
   fail-closed. No HTTP retry and no conversion to `Phi=0` are allowed.

## Execution gates

1. Exact-crash regression and clean/action negative tests pass locally and remotely.
2. Transactional deployment passes all existing CPU/protocol gates; the service is restarted under
   the new deployment identity.
3. Run a fresh seven-panel 69-goal learning campaign and require complete artifact/ledger identity,
   offline Oracle replay and `decision_bearing=false`.
4. Create a new hash-bound final authorization for a new campaign ID.
5. Run the complete seven-panel 153-goal post-exposure final campaign and the preregistered Holm
   analysis. Report `SUPPORTED`, `NOT_SUPPORTED`, or `INVALID` without substituting partial panels.
6. Recover and hash-check all evidence, update the protocol records, stop the exact project service,
   sync, and shut down instance `20d84f9474-d7816b14`.
