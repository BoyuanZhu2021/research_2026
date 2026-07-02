# code/ · H0 pilot harness

Tests **H0** (the geological-foundation hypothesis): *on OOD / unseen targets, does
naive **multi-turn** attacking beat **single-turn**?* — for `DISC-2026W27-001`.

**Mode:** API-only (no local GPU). Attacker + targets are served over OpenAI-compatible
APIs read from the project-root `.env`. Local vLLM on the H20 comes later (H1+ training).

## Status (2026-06-29)

- [x] Multi-provider LLM client + provider resolution (`src/`) — **stdlib only**
- [x] H0 config (`configs/h0_pilot.json`) + connectivity check (`scripts/env_check.py`)
- [ ] Attacker policies: single-turn + naive multi-turn (prompting, CoT)
- [ ] Domain ① adapter: agentic injection (InjecAgent / AgentDojo) + programmatic oracle
- [ ] Domain ② adapter: NVIDIA-style (garak probes + detectors)
- [ ] Runner `scripts/run_h0.py` (domain × arm × seed → ASR) + bootstrap-CI analysis

## Local artifacts (not in git)

`code/runs/` holds raw run outputs (JSONL, per-task dumps, API call logs). It is
**gitignored** and stays on your machine only — commit summaries to `LOGS/YYYY-Www.md`
instead. Same for `code/outputs/`, `code/checkpoints/`, and `code/wandb/`.

## Layout

```
code/
├── runs/                # local only — see above
├── src/
│   ├── providers.py     # logical provider name -> (base_url, api_key) from .env
│   └── llm_client.py    # chat() / chat_batch(); handles Qwen thinking channel + retries
├── configs/
│   └── h0_pilot.json    # models, seeds, K turns, n samples, OOD split, success criterion
├── scripts/
│   └── env_check.py     # ping the roster through the client
└── requirements.txt
```

## Roster (locked — `DISC-2026W27-001`)

| Role | Model | Provider | Thinking |
|---|---|---|---|
| Attacker | `Qwen/Qwen3.5-9B` | SiliconFlow | ON (CoT, 1024 tok) |
| Target (open) | `Qwen/Qwen3.6-27B` | SiliconFlow | OFF |
| Target (frontier) | `claude-sonnet-4-6` | aipaibox | n/a |

> Qwen 3.5/3.6 are **reasoning models**: `enable_thinking=false` for clean output, or
> `true` with ≥512 `max_tokens` (CoT lands in `reasoning_content`).

## Run the connectivity check

```bash
python code/scripts/env_check.py          # uses configs/h0_pilot.json
```

## Success criterion (pre-registered, PI-set)

H0 **holds** iff the bootstrap 95% CI of `ASR_multi − ASR_single` on held-out OOD
targets **excludes 0** (statistical significance; no fixed percentage-point threshold).
