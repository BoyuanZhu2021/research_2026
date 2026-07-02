"""H0 in a genuine MULTI-STEP environment (AgentDojo), with full LLM logging.

The injection sits in a tool result mid-trajectory; a successful attack requires the
agent to read the poisoned result and THEN call the malicious tool, so it is inherently
multi-step. We contrast a step-capped agent (single/short) vs an open agent (multi) by
limiting ToolsExecutionLoop.max_iters, same suite/tasks/attack/model. ASR = AgentDojo's
programmatic security oracle (a SECOND, independent oracle from InjecAgent).

Every LLM call (full input messages + tools + output message + usage) is logged via the
existing TraceLogger by wrapping the OpenAI client's create(); that wrapper also forces
enable_thinking=false for the SiliconFlow Qwen reasoning models.

Usage:
  python code/scripts/run_agentdojo_h0.py --smoke
  python code/scripts/run_agentdojo_h0.py --models qwen3-8b,qwen3.6-27b --iters 2,15 --suite slack --n-user 4
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
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
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
}

_lock = threading.Lock()


def make_logged_client(trace: TraceLogger):
    base, key = resolve_provider("siliconflow")
    client = openai.OpenAI(base_url=base, api_key=key, max_retries=4, timeout=120)
    orig = client.chat.completions.create

    def wrapped(*args, **kwargs):
        eb = dict(kwargs.get("extra_body") or {})
        eb.setdefault("enable_thinking", False)   # SiliconFlow Qwen3.x: no hidden reasoning
        kwargs["extra_body"] = eb
        for mm in (kwargs.get("messages") or []):   # normalize for SiliconFlow
            if not isinstance(mm, dict):
                continue
            if mm.get("role") == "developer":        # 'developer' role -> 'system'
                mm["role"] = "system"
            c = mm.get("content")
            if isinstance(c, list):                  # collapse content-part lists to a string
                parts = [(p.get("text") or "") if isinstance(p, dict) and p.get("type") == "text"
                         else (p if isinstance(p, str) else "") for p in c]
                joined = "".join(parts)
                mm["content"] = joined if joined else (None if mm.get("tool_calls") else "")
        resp = orig(*args, **kwargs)
        try:
            msg = resp.choices[0].message
            trace.log_call({
                "role": "target_agent", "model": kwargs.get("model"),
                "messages": kwargs.get("messages"),
                "tools_present": bool(kwargs.get("tools")),
                "response": {"content": msg.content,
                             "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
                             "reasoning": getattr(msg, "reasoning_content", None)},
                "usage": resp.usage.model_dump() if resp.usage else None,
                "context": kwargs.get("_ctx"),
            })
        except Exception as e:  # noqa: BLE001 — never let logging break a run
            trace.log_event({"event": "log_error", "error": str(e)})
        return resp

    client.chat.completions.create = wrapped
    return client


def build_pipeline(client, model_path, max_iters):
    llm = OpenAILLM(client, model_path)
    cfg = PipelineConfig(llm=llm, model_id=None, defense=None,
                         system_message_name=None, system_message=None)
    pipeline = AgentPipeline.from_config(cfg)
    for el in pipeline.elements:
        if isinstance(el, ToolsExecutionLoop):
            el.max_iters = max_iters
    pipeline.name = f"local-{model_path}"   # 'local' -> "Local model" for the attack's name lookup
    return pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="qwen3-8b,qwen3.6-27b")
    ap.add_argument("--iters", default="2,15", help="csv of max_iters arms (e.g. 2,15)")
    ap.add_argument("--suite", default="slack")
    ap.add_argument("--n-user", type=int, default=4)
    ap.add_argument("--attack", default="important_instructions_no_names")
    args = ap.parse_args()

    models = args.models.split(",")
    arms = [int(x) for x in args.iters.split(",")]
    if args.smoke:
        models, arms, args.n_user = ["qwen3-8b"], [2, 15], 1

    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = CODE_DIR / "runs" / f"adojo_{'smoke_' if args.smoke else ''}{run_id}"
    trace = TraceLogger(run_dir)
    client = make_logged_client(trace)

    suites = get_suites("v1")
    suite = suites[args.suite]
    user_task_ids = list(suite.user_tasks.keys())[: args.n_user]
    print(f"[run] {run_dir.name}  suite={args.suite}  user_tasks={user_task_ids}  "
          f"inj_tasks={len(suite.injection_tasks)}  models={models}  arms(max_iters)={arms}  attack={args.attack}")
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_id": run_id, "smoke": args.smoke, "suite": args.suite, "attack": args.attack,
        "user_tasks": user_task_ids, "injection_tasks": list(suite.injection_tasks.keys()),
        "models": {m: MODELS[m] for m in models}, "arms_max_iters": arms,
        "started": datetime.datetime.now().astimezone().isoformat(),
        "oracle": "AgentDojo programmatic security check (env-state)", "command": " ".join(sys.argv),
    }, indent=2), encoding="utf-8")

    results = {}
    for m in models:
        for max_iters in arms:
            arm = "single" if max_iters <= 2 else "multi"
            pipeline = build_pipeline(client, MODELS[m], max_iters)
            attack = load_attack(args.attack, suite, pipeline)
            logsub = run_dir / f"{m}_iters{max_iters}"
            with OutputLogger(str(logsub), live=None):
                r = benchmark_suite_with_injections(
                    pipeline, suite, attack, logdir=logsub,
                    force_rerun=True, user_tasks=user_task_ids, verbose=False)
            sec = list(r["security_results"].values())
            util = list(r["utility_results"].values())
            asr = sum(sec) / len(sec) if sec else None
            ut = sum(util) / len(util) if util else None
            results[f"{m}|iters{max_iters}|{arm}"] = {
                "model": m, "max_iters": max_iters, "arm": arm, "n": len(sec),
                "asr": asr, "utility": ut,
                "security_results": {f"{k[0]}|{k[1]}": v for k, v in r["security_results"].items()},
            }
            print(f"  {m:12s} max_iters={max_iters:2d} ({arm:6s})  ASR={asr*100 if asr is not None else 0:5.1f}%  "
                  f"utility={ut*100 if ut is not None else 0:5.1f}%  n={len(sec)}")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {run_dir}  ({trace.n_calls} LLM calls logged)")
    trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
