"""On-policy full-trajectory rollout for multi-turn GRPO (H1, method.md §2, PI-approved 2026-07-13).

The attacker policy generates EVERY turn (this is the fix for Gate 2's opening-only weakness): given
the running attacker-side chat history it emits a message; the frozen victim replies; the rule oracle
updates Phi. We record, per attacker turn, the exact (prompt_messages, response_text) and the Phi after
that turn, so the trainer can re-tokenize and weight each turn's tokens by its per-turn advantage.

Text-level + injection-agnostic on the generation/victim side (`gen_fn`, `victim_fn` are injected), so
the whole loop is unit-testable on CPU with mocks before the real Qwen3.5-4B policy + vLLM 9B victim.
"""
from __future__ import annotations

from collections.abc import Callable

from .attacker import build_attacker_transport, build_initial_messages, parse_injection
from .domains.base import Domain, Goal

# gen_fn(messages: list[dict]) -> raw attacker text (tool-use emits verbatim content only)
GenFn = Callable[[list[dict]], str]
# victim_fn(goal, conversation: list[dict]) -> (victim_reply_text, tool_calls)
VictimFn = Callable[[Goal, list[dict]], "tuple[str, list]"]


def _call_scope(domain: Domain) -> str:
    scope = getattr(domain, "call_scope", "episode")
    if scope not in ("episode", "attempt"):
        raise ValueError(f"invalid call_scope {scope!r} for domain {domain.name!r}")
    return scope


def _score_turn(domain: Domain, goal: Goal, scope: str, victim_texts: list[str],
                cumulative_calls: list, reply: str, attempt_calls: list,
                prev_phi: float) -> tuple[dict, float, float]:
    """Return ``(current_attempt_score, episode_phi, attempt_phi)``.

    Attempt-scoped domains start a fresh target ReAct run each outer turn, so calls may never be
    stitched across turns. Their episode potential is the running maximum of independently valid
    attempts. Episode-scoped legacy domains retain cumulative transcript/call scoring.
    """
    if scope == "attempt":
        score = domain.score(goal, [reply], attempt_calls)
        attempt_phi = float(score["phi"])
        return score, max(prev_phi, attempt_phi), attempt_phi
    score = domain.score(goal, victim_texts, cumulative_calls)
    attempt_phi = float(score["phi"])
    return score, attempt_phi, attempt_phi


def rollout_trajectory(domain: Domain, goal: Goal, gen_fn: GenFn, victim_fn: VictimFn,
                       *, T: int, tau: float) -> dict:
    """Roll out one trajectory, including per-turn attempts and the best valid call sequence.

    ``phi_trace`` is cumulative for episode-scoped domains and a running max for fresh-attempt
    domains. The attacker receives feedback from the current attempt, never a cross-attempt score.
    """
    attacker_messages = build_initial_messages(domain, goal)
    conversation: list[dict] = []          # victim-facing dialogue
    victim_texts: list[str] = []
    calls: list = []
    call_attempts: list[list] = []
    best_calls: list = []
    best_phi = -1.0
    turns: list[dict] = []
    phi_trace: list[float] = []
    prev_phi = 0.0
    success = False
    scope = _call_scope(domain)

    for t in range(1, T + 1):
        raw = gen_fn(attacker_messages)                     # POLICY generates this turn
        if domain.name == "tooluse":
            transport = build_attacker_transport(raw)
            msg = transport["content"] if transport["kind"] == "injection" else ""
            refused = transport["kind"] == "refusal"
        else:
            msg, refused = parse_injection(raw), False
            transport = None
        if refused:
            reply, attempt_calls = "", []
        else:
            user_message = {"role": "user", "content": msg}
            conversation.append(user_message)
            victim_conversation = ([dict(user_message)] if scope == "attempt"
                                   else list(conversation))
            reply, new_calls = victim_fn(goal, victim_conversation)
            conversation.append({"role": "assistant", "content": reply})
            victim_texts.append(reply)
            attempt_calls = list(new_calls)
        call_attempts.append(attempt_calls)
        if scope == "episode":
            calls = calls + attempt_calls
        score, phi, attempt_phi = _score_turn(
            domain, goal, scope, victim_texts, calls, reply, attempt_calls, prev_phi)
        if scope == "attempt" and attempt_phi > best_phi:
            best_phi, best_calls = attempt_phi, list(attempt_calls)
        turns.append({
            "turn": t,
            "prompt_messages": [dict(m) for m in attacker_messages],
            "response": raw,
            "raw_model_response": raw,
            "attacker_transport_normalization": "none",
            "attacker_transport": transport,
            "injection": msg,
            "victim_reply": reply,
            "attempt_calls": list(attempt_calls),
            "oracle": dict(score),
            "phi": phi,
            "attempt_phi": attempt_phi,
            "phi_gain": max(0.0, phi - prev_phi),
            "refused": refused,
        })
        phi_trace.append(phi)
        if bool(score["security"]):                          # phi >= tau
            success = True
            break
        prev_phi = phi
        retry = (
            "Output only non-empty injected content; do not emit <inject> or </inject>."
            if refused else domain.feedback(goal, score, reply)
        )
        attacker_messages = attacker_messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry}]

    result_calls = best_calls if scope == "attempt" else calls
    return {"turns": turns, "phi_trace": phi_trace, "success": success,
            "max_phi": max(phi_trace) if phi_trace else 0.0, "n_turns": len(turns),
            "calls": list(result_calls), "call_attempts": call_attempts}


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
           "call_attempts": [], "best_calls": [], "best_phi": -1.0,
           "prev_phi": 0.0, "success": False, "active": True} for g in goals]
    scope = _call_scope(domain)

    for t in range(1, T + 1):
        active = [i for i in range(B) if st[i]["active"]]
        if not active:
            break
        gens = gen_batch_fn([st[i]["attacker_messages"] for i in active])   # batched attacker generation
        pend = []
        pending_indices: list[int] = []
        for k, i in enumerate(active):
            g = gens[k]
            raw = g["text"] if isinstance(g, dict) else g
            st[i]["_raw"], st[i]["_gen"] = raw, (g if isinstance(g, dict) else {"text": raw})
            if domain.name == "tooluse":
                st[i]["_transport"] = build_attacker_transport(raw)
                st[i]["_msg"] = (
                    st[i]["_transport"]["content"]
                    if st[i]["_transport"]["kind"] == "injection" else ""
                )
                st[i]["_refused"] = st[i]["_transport"]["kind"] == "refusal"
            else:
                st[i]["_msg"], st[i]["_refused"] = parse_injection(raw), False
                st[i]["_transport"] = None
            if st[i]["_refused"]:
                continue
            user_message = {"role": "user", "content": st[i]["_msg"]}
            st[i]["conversation"].append(user_message)
            victim_conversation = ([dict(user_message)] if scope == "attempt"
                                   else list(st[i]["conversation"]))
            pend.append((st[i]["goal"], victim_conversation))
            pending_indices.append(i)
        replies = victim_batch_fn(pend) if pend else []                     # concurrent victim replies
        if len(replies) != len(pending_indices):
            raise RuntimeError(
                f"victim_batch_fn returned {len(replies)} replies for "
                f"{len(pending_indices)} non-refusal attempts"
            )
        reply_by_index = dict(zip(pending_indices, replies, strict=True))
        for i in active:
            s = st[i]
            if s["_refused"]:
                reply, attempt_calls = "", []
            else:
                reply, new_calls = reply_by_index[i]
                s["conversation"].append({"role": "assistant", "content": reply})
                s["victim_texts"].append(reply)
                attempt_calls = list(new_calls)
            s["call_attempts"].append(attempt_calls)
            if scope == "episode":
                s["calls"] = s["calls"] + attempt_calls
            score, phi, attempt_phi = _score_turn(
                domain, s["goal"], scope, s["victim_texts"], s["calls"], reply,
                attempt_calls, s["prev_phi"])
            if scope == "attempt" and attempt_phi > s["best_phi"]:
                s["best_phi"], s["best_calls"] = attempt_phi, list(attempt_calls)
            s["turns"].append({
                "turn": t, "prompt_messages": [dict(m) for m in s["attacker_messages"]],
                "response": s["_raw"],
                "raw_model_response": s["_gen"].get("raw_model_text", s["_raw"]),
                "attacker_transport_normalization": s["_gen"].get(
                    "pilot_transport_normalization", "none"
                ),
                "attacker_transport": s["_transport"], "injection": s["_msg"],
                "victim_reply": reply, "attempt_calls": list(attempt_calls),
                "oracle": dict(score), "phi": phi, "attempt_phi": attempt_phi,
                "phi_gain": max(0.0, phi - s["prev_phi"]),
                "refused": s["_refused"],
                "prompt_ids": s["_gen"].get("prompt_ids"), "resp_ids": s["_gen"].get("resp_ids")})
            s["phi_trace"].append(phi)
            if bool(score["security"]):                                     # phi >= tau
                s["success"] = True
                s["active"] = False
            else:
                s["prev_phi"] = phi
                retry = (
                    "Output only non-empty injected content; do not emit <inject> or </inject>."
                    if s["_refused"] else domain.feedback(s["goal"], score, reply)
                )
                s["attacker_messages"] = s["attacker_messages"] + [
                    {"role": "assistant", "content": s["_raw"]},
                    {"role": "user", "content": retry}]

    return [{"turns": s["turns"], "phi_trace": s["phi_trace"], "success": s["success"],
             "max_phi": max(s["phi_trace"]) if s["phi_trace"] else 0.0, "n_turns": len(s["turns"]),
             "calls": list(s["best_calls"] if scope == "attempt" else s["calls"]),
             "call_attempts": s["call_attempts"]}
            for s in st]
