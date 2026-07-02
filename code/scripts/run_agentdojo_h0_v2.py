"""H0 round-2 sweep: utility-matched multi-step injection ASR across suites x attacks x models.

Runs the agent at FULL capacity (max_iters=15) so utility is matched and high (fixes the
capped-agent confound from EXP-2026W27-004). The single-step vs multi-step contrast is
computed POST-HOC by agentdojo_analysis.py, which replays each saved trajectory step by
step and checks AgentDojo's env-state security oracle to find the first step at which the
attack succeeds. This runner just collects, at full capacity, the per-task security/utility
+ the saved AgentDojo trajectories (for replay) + every LLM call (calls.jsonl).

Shardable: launch one process per model (or per suite) for parallelism.

Usage:
  python code/scripts/run_agentdojo_h0_v2.py --smoke
  python code/scripts/run_agentdojo_h0_v2.py --models qwen3-8b --suites workspace,banking,slack \
      --attacks important_instructions_no_names,ignore_previous,direct --n-user 8 --run-tag sweepA
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import openai

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from src.providers import resolve_provider  # noqa: E402
from src.trace import TraceLogger  # noqa: E402

from agentdojo.agent_pipeline import AgentPipeline, OpenAILLM, ToolsExecutionLoop  # noqa: E402
from agentdojo.agent_pipeline.agent_pipeline import PipelineConfig  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.benchmark import benchmark_suite_with_injections  # noqa: E402
from agentdojo.logging import OutputLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

MODELS = {
    "qwen3-8b": ("siliconflow", "Qwen/Qwen3-8B"),
    "qwen3-14b": ("siliconflow", "Qwen/Qwen3-14B"),
    "qwen3.5-4b": ("siliconflow", "Qwen/Qwen3.5-4B"),
    "qwen2.5-14b": ("siliconflow", "Qwen/Qwen2.5-14B-Instruct"),
    "qwen3.6-27b": ("siliconflow", "Qwen/Qwen3.6-27B"),
    "qwen2.5-32b": ("siliconflow", "Qwen/Qwen2.5-32B-Instruct"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
}


def make_logged_client(trace: TraceLogger, provider: str):
    base, key = resolve_provider(provider)
    client = openai.OpenAI(base_url=base, api_key=key, max_retries=4, timeout=120)
    orig = client.chat.completions.create

    def wrapped(*args, **kwargs):
        eb = dict(kwargs.get("extra_body") or {})
        if "Qwen3" in str(kwargs.get("model", "")):   # enable_thinking is Qwen3-only
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
            trace.log_call({"role": "target_agent", "model": kwargs.get("model"),
                            "messages": kwargs.get("messages"), "tools_present": bool(kwargs.get("tools")),
                            "response": {"content": msg.content,
                                         "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]},
                            "usage": resp.usage.model_dump() if resp.usage else None,
                            "context": kwargs.get("_ctx")})
        except Exception as e:  # noqa: BLE001
            trace.log_event({"event": "log_error", "error": str(e)})
        return resp

    client.chat.completions.create = wrapped
    return client


def build_pipeline(client, model_path, max_iters):
    llm = OpenAILLM(client, model_path)
    cfg = PipelineConfig(llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None)
    pipeline = AgentPipeline.from_config(cfg)
    for el in pipeline.elements:
        if isinstance(el, ToolsExecutionLoop):
            el.max_iters = max_iters
    pipeline.name = f"local-{model_path}"   # 'local' satisfies the attack's model-name lookup
    return pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="qwen3-8b,qwen3.6-27b")
    ap.add_argument("--suites", default="workspace,banking,slack")
    ap.add_argument("--attacks", default="important_instructions_no_names,ignore_previous,direct")
    ap.add_argument("--n-user", type=int, default=8, help="user-task subset per suite")
    ap.add_argument("--max-iters", type=int, default=15)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    models = args.models.split(",")
    suites_sel = args.suites.split(",")
    attacks_sel = args.attacks.split(",")
    if args.smoke:
        models, suites_sel, attacks_sel, args.n_user = ["qwen3-8b"], ["slack"], ["important_instructions_no_names"], 2

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = (args.run_tag + "_") if args.run_tag else ("smoke_" if args.smoke else "")
    run_dir = CODE_DIR / "runs" / f"adojo2_{tag}{run_id}"
    trace = TraceLogger(run_dir)
    suites = get_suites("v1")

    meta = {"run_id": run_id, "smoke": args.smoke, "models": {m: MODELS[m] for m in models},
            "suites": suites_sel, "attacks": attacks_sel, "n_user": args.n_user, "max_iters": args.max_iters,
            "oracle": "AgentDojo env-state security (full-capacity agent; per-step via post-hoc replay)",
            "started": datetime.datetime.now().astimezone().isoformat(), "command": " ".join(sys.argv)}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[run] {run_dir.name}  models={models}  suites={suites_sel}  attacks={attacks_sel}  "
          f"n_user={args.n_user}  max_iters={args.max_iters}")

    clients = {}
    for m in models:
        provider = MODELS[m][0]
        if provider not in clients:
            clients[provider] = make_logged_client(trace, provider)

    n_cells = 0
    for m in models:
        provider, model_path = MODELS[m]
        pipeline = build_pipeline(clients[provider], model_path, args.max_iters)
        for sname in suites_sel:
            suite = suites[sname]
            user_ids = list(suite.user_tasks.keys())[: args.n_user]
            for aname in attacks_sel:
                attack = load_attack(aname, suite, pipeline)
                logsub = run_dir / f"{m}__{sname}__{aname}"
                try:
                    with OutputLogger(str(logsub), live=None):
                        r = benchmark_suite_with_injections(pipeline, suite, attack, logdir=logsub,
                                                            force_rerun=True, user_tasks=user_ids, verbose=False)
                except Exception as e:  # noqa: BLE001 — one bad cell must not kill the shard
                    trace.log_event({"event": "cell_error", "model": m, "suite": sname, "attack": aname,
                                     "error": f"{type(e).__name__}: {e}"})
                    print(f"  [cell-error] {m} {sname} {aname}: {type(e).__name__}: {str(e)[:80]}")
                    continue
                for (ut, it), sec in r["security_results"].items():
                    rec = {"model": m, "suite": sname, "attack": aname, "user_task": ut, "injection_task": it,
                           "security": bool(sec), "utility": bool(r["utility_results"].get((ut, it)))}
                    trace.log_record(rec)
                sec_vals = list(r["security_results"].values())
                util_vals = list(r["utility_results"].values())
                asr = sum(sec_vals) / len(sec_vals) if sec_vals else 0
                util = sum(util_vals) / len(util_vals) if util_vals else 0
                n_cells += 1
                print(f"  {m:12s} {sname:10s} {aname:28s}  ASR={asr*100:5.1f}%  util={util*100:5.1f}%  n={len(sec_vals)}")
    print(f"\n[done] {run_dir}  ({n_cells} cells, {trace.n_calls} LLM calls logged)")
    trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
