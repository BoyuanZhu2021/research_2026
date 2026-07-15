# Discussion registry — every issue, nothing missing

> Complete list of all discussion issues (active + archived). The **one active** issue lives at
> the repo root `Discussion.md` (protocol §7: one active at a time); closed issues are archived
> here under `Archive/` by `python tools/new_disc.py close "<slug>"`. This index is the durable
> registry — update it whenever an issue is opened or closed.

| Issue | Status | Title / scope | File |
|---|---|---|---|
| `DISC-2026W27-001` | Resolved | H0 foundation (attacker/victim harness, oracle, feasibility) | [Archive/DISC-2026W27-001-h0-foundation.md](Archive/DISC-2026W27-001-h0-foundation.md) |
| `DISC-2026W27-002` | Resolved | H1 dense-vs-sparse **UNTESTED** (round-1 AgentDojo atomic-by-construction, underpowered) | [Archive/DISC-2026W27-002-h1-dense-vs-sparse-untested.md](Archive/DISC-2026W27-002-h1-dense-vs-sparse-untested.md) |
| `DISC-2026W28-001` | **Open (active)** | H1 round-2: interactive/gated → **decoupled multi-turn extraction** → Gate 1′ pass → Gate 2 (3 inconclusive rounds) → **full-trajectory powered run** (built + trained; verdict eval pending GPU) | [../Discussion.md](../Discussion.md) |

## Notes on the active issue

`DISC-2026W28-001` is long-running and spans several phases (it deliberately stayed open across the
whole H1 round-2 arc). Its phase-by-phase record is in the `Posts` section of `../Discussion.md`,
cross-linked to `LOGS/2026-W28.md` (`EXP-2026W28-004`..`010`) and `LOGS/2026-W29.md`
(`EXP-2026W29-001`). Current state + next action: see [`../HANDOFF.md`](../HANDOFF.md).

When it is finally resolved, `tools/new_disc.py close` will move `Discussion.md` into
`Archive/DISC-2026W28-001-<slug>.md`; **add the row here at that time** so the registry stays complete.
