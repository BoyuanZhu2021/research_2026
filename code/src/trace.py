"""Append-only experiment trace. Records EVERY LLM call (full input + output) and
every episode-level event, so the whole run is auditable / re-analyzable offline.

Two JSONL streams per run dir (both flushed after each write, thread-safe):
  - calls.jsonl   one record per LLM call: role, context, request params, full
                  messages (input), content + reasoning (output), usage, latency.
  - events.jsonl  one record per episode-level event: episode start, per-turn
                  summary, oracle verdict, episode result.

Plus run_meta.json (config snapshot + repro metadata) written by the runner.

`LoggedClient` wraps llm_client.chat so a call is recorded even if it later raises.
"""
from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path

try:
    from . import llm_client
except ImportError:  # imported as a top-level module (src on sys.path)
    import llm_client


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


class TraceLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._calls = open(self.run_dir / "calls.jsonl", "a", encoding="utf-8")
        self._events = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        self._records = open(self.run_dir / "records.jsonl", "a", encoding="utf-8")
        self.n_calls = 0
        self.n_events = 0

    def _write(self, fh, rec: dict):
        with self._lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    def log_call(self, rec: dict):
        self._write(self._calls, {"ts": _now_iso(), "kind": "llm_call", **rec})
        self.n_calls += 1

    def log_event(self, rec: dict):
        self._write(self._events, {"ts": _now_iso(), **rec})
        self.n_events += 1

    def log_record(self, rec: dict):
        """One per-episode analysis record, written crash-safely as it completes."""
        self._write(self._records, rec)

    def close(self):
        with self._lock:
            self._calls.close()
            self._events.close()
            self._records.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_REQ_KEYS = ("provider", "model", "max_tokens", "temperature", "enable_thinking", "seed")


class LoggedClient:
    """Every .chat() is logged to the TraceLogger with full input + output.

    `role` (e.g. 'attacker', 'target') and `context` (episode_id, turn, etc.) are
    attached for slicing during analysis.
    """

    def __init__(self, trace: TraceLogger):
        self.trace = trace

    def chat(self, *, role: str, context: dict | None = None, **kwargs) -> dict:
        t0 = time.time()
        rec: dict = {
            "role": role,
            "context": context or {},
            "request": {k: kwargs.get(k) for k in _REQ_KEYS},
            "messages": kwargs.get("messages"),
        }
        try:
            r = llm_client.chat(**kwargs)
            rec["ok"] = True
            rec["response"] = {
                "content": r["content"],
                "reasoning": r["reasoning"],
                "usage": r["usage"],
                "finish_reason": r.get("finish_reason"),
                "latency": r["latency"],
            }
            return r
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["elapsed"] = time.time() - t0
            raise
        finally:
            self.trace.log_call(rec)
