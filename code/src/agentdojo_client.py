"""Reusable logged OpenAI-compatible client + AgentDojo pipeline builder for H1.

Factors the H0 harness's `make_logged_client` / `build_pipeline` (run_agentdojo_h0_v2.py) into a
shared module, adding a `role` label so ATTACKER and VICTIM calls are logged distinctly (the H0
copy hard-codes role="target_agent"). Same normalization: enable_thinking=false for Qwen3,
developer->system, content-part flatten, malformed tool-call JSON repair. Every call -> TraceLogger.
"""
from __future__ import annotations

import json

import openai

from providers import resolve_provider
from trace import TraceLogger


def make_logged_client(trace: TraceLogger, provider: str, role: str = "agent"):
    """OpenAI client (by logical provider name) whose calls are logged under `role`."""
    base, key = resolve_provider(provider)
    return make_client(trace, base, key, role)


def make_client(trace: TraceLogger, base_url: str, api_key: str, role: str = "agent", vllm_local: bool = False):
    """OpenAI client (by explicit base_url/key, e.g. a local vLLM server) with logging under `role`.

    `vllm_local=True`: disable Qwen3 thinking via vLLM's `chat_template_kwargs` (SiliconFlow instead
    takes a top-level `enable_thinking`), matched case-insensitively so `qwen3.6-27b` is covered.
    """
    client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=4, timeout=180)
    orig = client.chat.completions.create

    def wrapped(*args, **kwargs):
        ctx = kwargs.pop("_ctx", None)  # our own field; must not reach the OpenAI API
        eb = dict(kwargs.get("extra_body") or {})
        if "qwen3" in str(kwargs.get("model", "")).lower():
            if vllm_local:
                ct = dict(eb.get("chat_template_kwargs") or {})
                ct.setdefault("enable_thinking", False)
                eb["chat_template_kwargs"] = ct
            else:
                eb.setdefault("enable_thinking", False)
        if eb:
            kwargs["extra_body"] = eb
        for mm in (kwargs.get("messages") or []):
            if not isinstance(mm, dict):
                continue
            if mm.get("role") == "developer":
                mm["role"] = "system"
            c = mm.get("content")
            if isinstance(c, list):
                parts = [(p.get("text") or "") if isinstance(p, dict) and p.get("type") == "text"
                         else (p if isinstance(p, str) else "") for p in c]
                joined = "".join(parts)
                mm["content"] = joined if joined else (None if mm.get("tool_calls") else "")
        resp = orig(*args, **kwargs)
        try:  # repair malformed tool-call JSON (weaker models emit invalid args -> AgentDojo crashes)
            for ch in resp.choices:
                for tc in (getattr(ch.message, "tool_calls", None) or []):
                    a = getattr(tc.function, "arguments", None)
                    if isinstance(a, str):
                        try:
                            json.loads(a)
                        except Exception:
                            tc.function.arguments = "{}"
        except Exception:
            pass
        try:
            msg = resp.choices[0].message
            trace.log_call({"role": role, "model": kwargs.get("model"),
                            "messages": kwargs.get("messages"), "tools_present": bool(kwargs.get("tools")),
                            "response": {"content": msg.content,
                                         "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]},
                            "usage": resp.usage.model_dump() if resp.usage else None,
                            "context": ctx})
        except Exception as e:  # noqa: BLE001
            trace.log_event({"event": "log_error", "role": role, "error": str(e)})
        return resp

    client.chat.completions.create = wrapped
    return client


def build_pipeline(client, model_path: str, max_iters: int):
    """AgentDojo pipeline over `client`, victim tool loop capped at `max_iters`."""
    from agentdojo.agent_pipeline import AgentPipeline, OpenAILLM, ToolsExecutionLoop
    from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig

    llm = OpenAILLM(client, model_path)
    cfg = PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None)
    pipeline = AgentPipeline.from_config(cfg)
    for el in pipeline.elements:
        if isinstance(el, ToolsExecutionLoop):
            el.max_iters = max_iters
    pipeline.name = f"local-{model_path}"
    return pipeline


def chat_once(client, model_path: str, messages: list[dict], temperature: float = 0.7,
              max_tokens: int = 1024, ctx: dict | None = None) -> str:
    """Single completion (for the attacker policy). Returns the text content."""
    resp = client.chat.completions.create(
        model=model_path, messages=messages, temperature=temperature, max_tokens=max_tokens, _ctx=ctx,
    )
    return resp.choices[0].message.content or ""
