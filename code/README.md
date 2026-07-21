# code/ · active H1 confirmatory harness

This directory now contains only the active single-H20 H1 path. Historical H0, API-victim,
dual-local, V100 and superseded Gate implementations are stored in the audited source archive at
`artifacts/_archive/h1-historical-code-20260721.zip`.

**Current scientific state:** six `dense/sparse × seeds {0,1,2}` adapters are complete. The
69-goal confirmatory learning run is blocked on a versioned constrained-eval decoder guard; final
OOD remains unread.

## Local artifacts (not in git)

`code/runs/` holds historical raw run outputs. Current H1 artifacts live under `artifacts/`, which
is also gitignored; durable conclusions and pointers remain in `LOGS/`, `Discussion.md` and
`HANDOFF.md`.

## Layout

```
code/
├── configs/             # frozen split and curriculum identities
├── runs/                # historical local outputs; gitignored
├── scripts/             # 31 active controllers, runners and CPU contract tests
│   ├── h1_inprocess_confirmatory_controller.py
│   ├── h1_inprocess_confirmatory_eval.py
│   ├── h1_inprocess_confirmatory_analyze.py
│   ├── h1_mt_grpo_train_h20.py
│   ├── h1_serve_victim_h20.py
│   └── h1_deploy_mt.py
├── src/                 # 52 active protocol, Oracle, identity and artifact modules
└── requirements.txt
```

## Active entrypoints

```bash
python code/scripts/h1_deploy_mt.py --plan
python code/scripts/h1_inprocess_confirmatory_test.py
python code/scripts/h1_inprocess_confirmatory_controller.py --help
```

`h1_deploy_mt.py` uses explicit script and src allowlists; new diagnostics are not deployed unless
they are deliberately registered. Remote mutation still requires the explicit `--execute` flag.

## Frozen experiment

- Attacker: Qwen3.5-4B NF4 QLoRA on one H20.
- Victim: Qwen3.5-9B FP8 on the same H20.
- Data: 322 train / 69 calibration / 153 untouched final OOD.
- Interaction: `T=5`, `max_calls=3`, one fresh victim attempt per attacker turn.
- Oracle: deterministic `Phi ∈ {0, 1/3, 2/3, 1}`; no LLM judge.
- H1 decision: registered dense-vs-sparse/base-K Holm family on final OOD.
