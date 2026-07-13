"""On-policy full-trajectory rollout for multi-turn GRPO (H1, method.md §2, PI-approved 2026-07-13).

The attacker policy generates EVERY turn (this is the fix for Gate 2's opening-only weakness): given
the running attacker-side chat history it emits a message; the frozen victim replies; the rule oracle
updates Phi. We record, per attacker turn, the exact (prompt_messages, response_text) and the Phi after
that turn, so the trainer can re-tokenize and weight each turn's tokens by its per-turn advantage.

Text-level + injection-agnostic on the generation/victim side (`gen_fn`, `victim_fn` are injected), so
the whole loop is unit-testable on CPU with mocks before the real Qwen3-8B policy + vLLM 27B victim.
"""
from __future__ import annotations

from collections.abc import Callable

from .attacker import build_initial_messages, parse_injection
from .domains.base import Domain, Goal

# gen_fn(messages: list[dict]) -> raw attacker text (may wrap the message in <inject>...</inject>)
GenFn = Callable[[list[dict]], str]
# victim_fn(goal, conversation: list[dict]) -> (victim_reply_text, tool_calls)
VictimFn = Callable[[Goal, list[dict]], "tuple[str, list]"]


def rollout_trajectory(domain: Domain, goal: Goal, gen_fn: GenFn, victim_fn: VictimFn,
                       *, T: int, tau: float) -> dict:
    """Roll out ONE on-policy trajectory. Returns turns (each with prompt/response/phi/phi_gain),
    the Phi trace, and success. The attacker adapts each turn via `domain.feedback`."""
    attacker_messages = build_initial_messages(domain, goal)
    conversation: list[dict] = []          # victim-facing dialogue
    victim_texts: list[str] = []
    calls: list = []
    turns: list[dict] = []
    phi_trace: list[float] = []
    prev_phi = 0.0
    success = False

    for t in range(1, T + 1):
        raw = gen_fn(attacker_messages)                     # POLICY generates this turn
        msg = parse_injection(raw)
        conversation.append({"role": "user", "content": msg})
        reply, new_calls = victim_fn(goal, conversation)
        conversation.append({"role": "assistant", "content": reply})
        victim_texts.append(reply)
        calls = calls + list(new_calls)
        score = domain.score(goal, victim_texts, calls)
        phi = float(score["phi"])
        turns.append({
            "turn": t,
            "prompt_messages": [dict(m) for m in attacker_messages],
            "response": raw,
            "phi": phi,
            "phi_gain": max(0.0, phi - prev_phi),
            "refused": len(msg.strip()) < 3,
        })
        phi_trace.append(phi)
        if bool(score["security"]):                          # phi >= tau
            success = True
            break
        prev_phi = phi
        attacker_messages = attacker_messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": domain.feedback(goal, score, reply)}]

    return {"turns": turns, "phi_trace": phi_trace, "success": success,
            "max_phi": max(phi_trace) if phi_trace else 0.0, "n_turns": len(turns)}
