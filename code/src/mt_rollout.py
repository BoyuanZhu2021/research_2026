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


# gen_batch_fn(batch_messages: list[list[dict]]) -> list[dict], each {"text": raw, "prompt_ids"?, "resp_ids"?}
GenBatchFn = Callable[[list], list]
# victim_batch_fn(batch_items: list[tuple[Goal, list[dict]]]) -> list[tuple[str, list]] (aligned)
VictimBatchFn = Callable[[list], list]


def rollout_batch(domain: Domain, goals: list, gen_batch_fn: GenBatchFn, victim_batch_fn: VictimBatchFn,
                  *, T: int, tau: float) -> list[dict]:
    """Advance B on-policy trajectories in LOCKSTEP (one per element of `goals`; repeats allowed, e.g.
    each goal x G). Same per-trajectory logic as `rollout_trajectory`, but each turn batches the active
    trajectories' attacker generation into ONE `gen_batch_fn` call and fires the victim replies via ONE
    `victim_batch_fn` call (concurrent). Because every trajectory's gen/victim depend only on its OWN
    running state, this is exactly equivalent to looping `rollout_trajectory` when gen/victim are pure
    per-item (asserted on CPU with deterministic mocks). Returns a list of per-trajectory result dicts,
    each the SAME shape `rollout_trajectory` returns (turns also carry optional prompt_ids/resp_ids).
    """
    B = len(goals)
    st = [{"goal": g, "attacker_messages": build_initial_messages(domain, g),
           "conversation": [], "victim_texts": [], "calls": [], "turns": [], "phi_trace": [],
           "prev_phi": 0.0, "success": False, "active": True} for g in goals]

    for t in range(1, T + 1):
        active = [i for i in range(B) if st[i]["active"]]
        if not active:
            break
        gens = gen_batch_fn([st[i]["attacker_messages"] for i in active])   # batched attacker generation
        pend = []
        for k, i in enumerate(active):
            g = gens[k]
            raw = g["text"] if isinstance(g, dict) else g
            st[i]["_raw"], st[i]["_gen"] = raw, (g if isinstance(g, dict) else {"text": raw})
            st[i]["_msg"] = parse_injection(raw)
            st[i]["conversation"].append({"role": "user", "content": st[i]["_msg"]})
            pend.append((st[i]["goal"], list(st[i]["conversation"])))
        replies = victim_batch_fn(pend)                                     # concurrent victim replies
        for k, i in enumerate(active):
            s = st[i]
            reply, new_calls = replies[k]
            s["conversation"].append({"role": "assistant", "content": reply})
            s["victim_texts"].append(reply)
            s["calls"] = s["calls"] + list(new_calls)
            score = domain.score(s["goal"], s["victim_texts"], s["calls"])
            phi = float(score["phi"])
            s["turns"].append({
                "turn": t, "prompt_messages": [dict(m) for m in s["attacker_messages"]],
                "response": s["_raw"], "phi": phi, "phi_gain": max(0.0, phi - s["prev_phi"]),
                "refused": len(s["_msg"].strip()) < 3,
                "prompt_ids": s["_gen"].get("prompt_ids"), "resp_ids": s["_gen"].get("resp_ids")})
            s["phi_trace"].append(phi)
            if bool(score["security"]):                                     # phi >= tau
                s["success"] = True
                s["active"] = False
            else:
                s["prev_phi"] = phi
                s["attacker_messages"] = s["attacker_messages"] + [
                    {"role": "assistant", "content": s["_raw"]},
                    {"role": "user", "content": domain.feedback(s["goal"], score, reply)}]

    return [{"turns": s["turns"], "phi_trace": s["phi_trace"], "success": s["success"],
             "max_phi": max(s["phi_trace"]) if s["phi_trace"] else 0.0, "n_turns": len(s["turns"])}
            for s in st]
