#!/usr/bin/env python3
"""Create the LOGS file for the current ISO week if it does not exist.

Usage:
    python tools/new_week.py            # current week
    python tools/new_week.py 2026-W11   # explicit week
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO / "LOGS"

TEMPLATE = """\
# 🧪 实验周志 · {week_id}

- **起止日期**：{start} ～ {end}
- **本周核心 Vibe**：（一句话写清本周想验证的核心问题）

---

## 1. 本周概览（Weekly Overview）

- **关联议题**：`Discussion.md` 中当前议题号 `DISC-{week_short}-NNN`
- **本周里程碑**：
  - [ ] 里程碑 1
  - [ ] 里程碑 2

---

## 2. 实验记录（Experiments）

> 每跑完一个实验，追加一块下面这样的记录。**所有字段必填**（无关项写 `N/A`）。
> 使用 `python tools/new_exp.py "源意图"` 可自动追加骨架。

---

## 3. 本周回顾（Weekly Retro，周末由 Agent 汇总）

- 已驳斥假设：
- 已支持假设：
- 悬而未决：
- 与 idea.md / method.md 的冲突：
"""


def parse_week_id(arg: str | None) -> tuple[str, str, date, date]:
    """Return (week_id, week_short, monday, sunday)."""
    if arg:
        year_str, week_str = arg.split("-W")
        year, week = int(year_str), int(week_str)
    else:
        today = date.today()
        year, week, _ = today.isocalendar()
    week_id = f"{year}-W{week:02d}"
    week_short = f"{year}W{week:02d}"
    # ISO week starts Monday
    monday = date.fromisocalendar(year, week, 1)
    sunday = monday + timedelta(days=6)
    return week_id, week_short, monday, sunday


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    week_id, week_short, monday, sunday = parse_week_id(arg)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    target = LOGS_DIR / f"{week_id}.md"
    if target.exists():
        print(f"[skip] {target.relative_to(REPO)} already exists")
        return 0

    target.write_text(
        TEMPLATE.format(
            week_id=week_id,
            week_short=week_short,
            start=monday.isoformat(),
            end=sunday.isoformat(),
        ),
        encoding="utf-8",
    )
    print(f"[ok] created {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
