#!/usr/bin/env python3
"""Session-start self-check — the single mechanical implementation of the
protocol's "session begin" checks (AGENTS.md § 2 / § 2.1 / § 6).

All agents (Claude Code via SessionStart hook, Codex, Cursor, ...) run this
before the first reply and act on its output:

- bootstrap.md present -> initialization instructions (guide text lives in
  bootstrap.md itself; this script never duplicates it)
- initialized          -> mode + active issue + current week LOGS + retro due

Usage:
    python tools/session_check.py          # plain text briefing (any agent)
    python tools/session_check.py --hook   # JSON envelope for Claude Code SessionStart hook
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO / "LOGS"

DISC_RE = re.compile(r"DISC-\d{4}W\d{2}-\d{3}")
WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def mode_fields() -> dict[str, str]:
    """Parse the `- `key`: `value`` bullets at the top of MODE.md."""
    fields: dict[str, str] = {}
    for raw in read(REPO / "MODE.md").splitlines():
        m = re.match(r"^-\s*`?(\w+)`?\s*[:：]\s*`?([^`]*)`?\s*$", raw.strip())
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def parse_week(s: str) -> tuple[int, int] | None:
    m = WEEK_RE.match(s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def week_str(yw: tuple[int, int]) -> str:
    return f"{yw[0]}-W{yw[1]:02d}"


def latest_past_week_log(now: tuple[int, int]) -> tuple[int, int] | None:
    """Most recent LOGS/YYYY-Www.md strictly before the current ISO week."""
    best: tuple[int, int] | None = None
    if not LOGS_DIR.is_dir():
        return None
    for p in LOGS_DIR.glob("*.md"):
        yw = parse_week(p.stem)
        if yw and yw < now and (best is None or yw > best):
            best = yw
    return best


def discussion_line() -> str:
    text = read(REPO / "Discussion.md")
    header = text.split("## Open Questions")[0]
    m = DISC_RE.search(header)
    if not m:
        return (
            "active 议题：无（Discussion.md 处于模板态）→ Plan 前置：先与用户对齐议题，"
            '再 `python tools/new_disc.py open "<标题>"`'
        )
    status_m = re.search(r"\*\*状态 \(Status\)\*\*\s*\|([^|\n]*)\|", header)
    status = status_m.group(1).strip().strip("`") if status_m else "?"
    title_m = re.search(r"\*\*标题 \(Title\)\*\*\s*\|([^|\n]*)\|", header)
    title = title_m.group(1).strip() if title_m else ""
    return f"active 议题：{m.group(0)} [{status}] {title}"


def briefing() -> str:
    if (REPO / "bootstrap.md").exists():
        return (
            "[未初始化] 检测到 bootstrap.md。\n"
            "请立即读取 bootstrap.md，向用户完整呈现其 § 1–2（协议简介 + 模式选择），"
            "并要求用户仅回复 A 或 B。\n"
            "完成初始化前：可回答关于协议/仓库本身的元问题，但不得执行任何 P-E-R 动作；"
            "每次回复末尾重复 A/B 选择引导。（AGENTS.md § 2）"
        )

    fields = mode_fields()
    mode = fields.get("mode", "")
    if mode not in {"newbie", "expert"}:
        return (
            "[协议异常] bootstrap.md 不存在，且 MODE.md::mode 不是 newbie/expert。\n"
            "请提示用户从模板重建 bootstrap.md 并重新初始化。（AGENTS.md § 2.2）"
        )

    now: tuple[int, int] = date.today().isocalendar()[:2]
    lines = [
        f"[会话简报] mode = {mode}（策略见 MODE.md / AGENTS.md § 3）",
        f"- {discussion_line()}",
    ]

    week_file = LOGS_DIR / f"{week_str(now)}.md"
    if week_file.exists():
        lines.append(f"- 当周日志：LOGS/{week_str(now)}.md 已存在")
    else:
        lines.append(
            f"- 当周日志：LOGS/{week_str(now)}.md 不存在 → 首次实验前先 `python tools/new_week.py`"
        )

    last_retro = parse_week(fields.get("last_retro", "")) or (0, 0)
    target = latest_past_week_log(now)
    if target and last_retro < now:
        lines.append(
            f"- ⚠ 欠 Weekly Retro：本会话先按 AGENTS.md § 6 扫描 LOGS/{week_str(target)}.md 完成回顾，"
            f"然后把 MODE.md::last_retro 更新为 {week_str(now)}"
        )
    else:
        lines.append("- Weekly Retro：无欠账")
    lines.append("- Plan 前必读：idea.md + Discussion.md（AGENTS.md § 4.2 前置检查）")
    return "\n".join(lines)


def main() -> int:
    text = briefing()
    if "--hook" in sys.argv[1:]:
        payload = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": text,
            }
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
