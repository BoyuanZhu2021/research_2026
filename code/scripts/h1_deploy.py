"""Ship the H1 training code + goal pool to the H20 (flat /root/autodl-tmp/h1/) and install agentdojo.

Flat layout on purpose: every module sits next to h1_grpo_train.py, whose sys.path insert makes the
cross-imports (reward/goalpool/h1_rollout/trace/agentdojo_client) resolve without a package.
Run after provisioning; re-runnable (overwrites).
"""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import remote as RM  # noqa: E402

FILES = {
    "src/reward.py": "reward.py",
    "src/goalpool.py": "goalpool.py",
    "src/agentdojo_client.py": "agentdojo_client.py",
    "src/providers.py": "providers.py",
    "src/trace.py": "trace.py",
    "src/llm_client.py": "llm_client.py",
    "scripts/h1_rollout.py": "h1_rollout.py",
    "scripts/h1_goalgen_poc.py": "h1_goalgen_poc.py",
    "scripts/h1_grpo_train.py": "h1_grpo_train.py",
    "runs/goalpool/goals_train.jsonl": "goals_train.jsonl",
    "runs/goalpool/goals_ood.jsonl": "goals_ood.jsonl",
}


def main():
    cli = RM.connect()
    RM.run(cli, "mkdir -p /root/autodl-tmp/h1/runs", timeout=20)
    for loc, rem in FILES.items():
        RM.put_file(cli, str(CODE / loc), f"/root/autodl-tmp/h1/{rem}")
    print(f"transferred {len(FILES)} files -> /root/autodl-tmp/h1/")
    RM.run_bg(cli, "python -m pip install -q agentdojo && "
                   "python -c \"import agentdojo;print('AGENTDOJO_OK',agentdojo.__version__)\" && echo H1_ADOJO_DONE",
              "/root/autodl-tmp/h1_adojo.log")
    print("installing agentdojo (base env) -> h1_adojo.log")
    cli.close()


if __name__ == "__main__":
    main()
