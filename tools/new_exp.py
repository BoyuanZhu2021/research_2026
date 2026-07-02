#!/usr/bin/env python3
"""Append a fresh EXP block to the current week's LOGS file.

Auto-fills: experiment ID (next NNN), git commit hash, hostname,
and the active discussion ID from Discussion.md (falls back to the
`DISC-YYYYWww-NNN` stub when no issue is open).

Usage:
    python tools/new_exp.py "测试 Lipschitz 正则对 PGD 的鲁棒性"
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO / "LOGS"


def current_week_id() -> tuple[str, str]:
    today = date.today()
    year, week, _ = today.isocalendar()
    return f"{year}-W{week:02d}", f"{year}W{week:02d}"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{out}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def active_disc() -> str:
    """Active discussion ID from Discussion.md's Issue Header, or the stub."""
    try:
        text = (REPO / "Discussion.md").read_text(encoding="utf-8")
    except OSError:
        return "DISC-YYYYWww-NNN"
    header = text.split("## Open Questions")[0]
    m = re.search(r"DISC-\d{4}W\d{2}-\d{3}", header)
    return m.group(0) if m else "DISC-YYYYWww-NNN"


def next_exp_number(text: str, week_short: str) -> int:
    pattern = re.compile(rf"EXP-{re.escape(week_short)}-(\d{{3}})")
    nums = [int(m.group(1)) for m in pattern.finditer(text)]
    return max(nums, default=0) + 1


def render_block(exp_id: str, vibe: str, commit: str, host: str, disc: str) -> str:
    return f"""

### {exp_id}

- 源意图 (Original Vibe):    {vibe}
- 假设 (Hypothesis):
- 是否被驳斥 (Falsified?):   Y / N / 部分 / Crashed
- 驳斥/支持原因 (Why):
- Agent 动作 (What changed):
- 复现信息 (Repro):
  - commit:                 {commit}
  - seed:
  - dataset / version:
  - env:
  - hardware:               {host}
  - command:
- 关键指标 (Metrics):
- 日志路径 (Artifacts):
- 结论 (Conclusion, 1–3 句):
- 下一步 (Next):
- 关联议题 (Discussion):     {disc}
"""


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: python tools/new_exp.py "源意图"', file=sys.stderr)
        return 1
    vibe = sys.argv[1]

    week_id, week_short = current_week_id()
    target = LOGS_DIR / f"{week_id}.md"
    if not target.exists():
        print(f"[err] {target.relative_to(REPO)} not found. Run `python tools/new_week.py` first.", file=sys.stderr)
        return 2

    text = target.read_text(encoding="utf-8")
    nnn = next_exp_number(text, week_short)
    exp_id = f"EXP-{week_short}-{nnn:03d}"

    block = render_block(exp_id, vibe, git_commit(), socket.gethostname(), active_disc())
    with target.open("a", encoding="utf-8") as f:
        f.write(block)
    print(f"[ok] appended {exp_id} to {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
