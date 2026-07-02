"""Connectivity check for the H0 roster. Pings the attacker + every target in
the config through the unified client. Reusable / cheap (one tiny call each).

Usage:  python code/scripts/env_check.py [path/to/config.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from src.llm_client import chat  # noqa: E402
from src.providers import ENV_PATH, mask, resolve_provider, PROVIDERS  # noqa: E402

DEFAULT_CFG = CODE_DIR / "configs" / "h0_pilot.json"


def ping(role: str, spec: dict) -> bool:
    try:
        r = chat(
            spec["provider"], spec["model"],
            [{"role": "user", "content": "Reply with exactly the two characters: OK"}],
            max_tokens=spec.get("max_tokens", 64),
            enable_thinking=spec.get("enable_thinking"),
            temperature=0.0,
        )
        print(f"  [OK]   {role:22s} {spec['provider']}/{spec['model']}  "
              f"{r['latency']:.1f}s  ctok={r['usage'].get('completion_tokens','?')}  "
              f"reasoning={len(r['reasoning'])}c  content={r['content'][:30]!r}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {role:22s} {spec['provider']}/{spec['model']}  ->  {e}")
        return False


def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CFG
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print(f"env file: {ENV_PATH}  (exists={ENV_PATH.exists()})")
    print("providers:")
    for name in PROVIDERS:
        try:
            base, key = resolve_provider(name)
            print(f"  {name:18s} {base}  key={mask(key)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:18s} UNRESOLVED: {e}")
    print(f"\nroster ping ({cfg_path.name}):")
    ok = ping("attacker", cfg["attacker"])
    for t in cfg["targets"]:
        ok = ping(f"target:{t['name']}", t) and ok
    print("\n==> ALL OK" if ok else "\n==> SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
