# H1 constrained-eval content-only confirmation v2

**PI authorization:** `继续`（承接 `HANDOFF.md` 中唯一建议动作）  
**Frozen:** 2026-07-21  
**Current position:** CE-P1/4 implementation complete; remote deployment pending  
**Only instance:** `20d84f9474-d7816b14`  
**Expected GPU:** H20 `GPU-14627e41-ad52-9967-0a52-bbd82009ef01`

## What changes and what stays frozen

The six sealed adapters, 322/69/153 split, `ds/base,m=2` task, `T=5`, `max_calls=3`, local
Qwen3.5-9B FP8 victim, Qwen3.5-4B NF4 attacker, programmatic Oracle, dense `Delta Phi`, sparse
first-success reward and registered Holm decision rule remain unchanged. This version changes only
the attacker decoding distribution used by confirmatory evaluation.

The predecessor profile hard-failed when one raw completion echoed the harness-reserved literal
`<inject>`. Fixed `bad_words_ids` cannot safely prevent this because Qwen may fuse `>` with the
following punctuation. The new profile therefore applies a text-aware token-level DFA before
sampling. For the current decoded generated-text suffix, it masks every candidate token whose
decoded text would complete any of the 128 ASCII-case variants of `<inject>` or `</inject>`.

Post-generation trimming, tag stripping, prefix salvage and scoring malformed output as zero remain
forbidden. The independent parser still fails closed. Raw attacker text, raw response token IDs and
raw victim responses are durably recorded before parsing or Oracle scoring.

## CE-P1/4 — versioned implementation and identity

- Profile: `h1-gate-partial-confirmatory-constrained-eval-v2`.
- Guard: `h1-content-only-reserved-tag-dfa-v1`.
- The run config seals the profile-config SHA-256, tokenizer revision, vocabulary size, explicit
  case-variant hash, DFA states, per-state blocked-token counts and transition-table SHA-256.
- The result repeats the exact guard identity. The analyzer rejects any profile, deployment,
  service, GPU, config or guard drift across the seven panels.
- The new profile cannot load or authorize panels from the predecessor profile.

## CE-P2/4 — exact deployment and tokenizer/runtime gate

Stop only the registered project victim process, transactionally deploy the exact local tree, run
all remote CPU gates and reconstruct the same pinned FP8 victim. Before a scientific panel, build
the guard against the pinned Qwen3.5-4B tokenizer and require:

1. all 128 variants are forbidden as complete generated text;
2. cross-token completion is masked;
3. right-fused candidates such as the token that decodes to `>"` are masked;
4. the Transformers logits-processor interface works on the real installed stack;
5. deployment, tokenizer revision, GPU UUID and victim lifecycle are sealed.

Any failure stops the campaign without reading final OOD.

## CE-P3/4 — fresh learning and final OOD

Create a fresh campaign and run exactly:

`base-k4 -> dense-s0 -> sparse-s0 -> dense-s1 -> sparse-s1 -> dense-s2 -> sparse-s2`.

All seven 69-goal learning panels must be complete under one guard identity; predecessor results are
not mixed. Build the non-decision-bearing learning report and then create the already authorized,
hash-bound final-OOD authorization. Only then load the untouched 153-goal split and run the same
seven-panel order under the identical constrained-eval profile.

## CE-P4/4 — decision, recovery and shutdown

Replay every Oracle trace and apply the frozen 20,000-sample goal-cluster bootstrap plus Holm
correction to `dense-sparse`, `dense-baseK` and `sparse-baseK`. H1 is supported only if all three
point estimates are positive, every 95% interval excludes zero and every adjusted p-value is below
0.05. Otherwise a complete run is `NOT_SUPPORTED`; missing/inconsistent evidence is `INVALID`.

Recover and hash-verify all artifacts, stop exact project PIDs/PGIDs, `sync`, shut down instance
`20d84f9474-d7816b14`, verify SSH loss, and update LOGS/Discussion/HANDOFF. No claim extends to
`m>=3`.
