#!/usr/bin/env python3
"""Open / close Discussion issues with mechanical ID allocation (AGENTS.md § 7).

Subcommands:
    open "<标题>" [--owner "PI @张三"]   在模板态 Discussion.md 上开启新议题（自动分配议题号）
    close "<slug>"                       校验 Resolution → 归档到 Discussion/Archive/DISC-YYYYWww-NNN-<slug>.md
                                         → 从 tools/templates/Discussion.template.md 重置
    next                                 仅打印下一个可用议题号

议题号按周编号：DISC-YYYYWww-NNN，扫描当前 Discussion.md + Archive/ 取 max+1，避免手工撞号。
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DISCUSSION = REPO / "Discussion.md"
ARCHIVE = REPO / "Discussion" / "Archive"
TEMPLATE = REPO / "tools" / "templates" / "Discussion.template.md"

DISC_RE = re.compile(r"DISC-\d{4}W\d{2}-\d{3}")

# 关闭议题时 Resolution 段不允许残留的模板占位片段
TEMPLATE_RESIDUE = (
    "一段话说明最终结论",
    "为什么这是结论",
    "§ X",
    "§ Y",
    "YYYY-MM-DD",
    "@姓名",
    "EXP-YYYYWww-NNN",
)


def week_short() -> str:
    y, w, _ = date.today().isocalendar()
    return f"{y}W{w:02d}"


def next_disc_id() -> str:
    ws = week_short()
    pat = re.compile(rf"DISC-{ws}-(\d{{3}})")
    sources: list[str] = []
    if DISCUSSION.exists():
        sources.append(DISCUSSION.read_text(encoding="utf-8"))
    if ARCHIVE.is_dir():
        for p in ARCHIVE.glob("*.md"):
            sources.append(p.name)
            sources.append(p.read_text(encoding="utf-8", errors="replace"))
    nums = [int(m.group(1)) for s in sources for m in pat.finditer(s)]
    return f"DISC-{ws}-{max(nums, default=0) + 1:03d}"


def active_id(text: str) -> str | None:
    header = text.split("## Open Questions")[0]
    m = DISC_RE.search(header)
    return m.group(0) if m else None


def row_value(text: str, label: str) -> str:
    m = re.search(rf"^\|\s*\*\*{re.escape(label)}\*\*\s*\|(.*)\|\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def replace_row(text: str, label: str, value: str) -> str:
    pattern = re.compile(rf"^\|\s*\*\*{re.escape(label)}\*\*\s*\|.*\|\s*$", re.MULTILINE)
    return pattern.sub(f"| **{label}** | {value} |", text, count=1)


def cmd_open(title: str, owner: str | None) -> int:
    text = DISCUSSION.read_text(encoding="utf-8")
    existing = active_id(text)
    if existing:
        if "Resolved" in row_value(text, "状态 (Status)"):
            print(f"[err] {existing} 已 Resolved 但未归档。先运行 close。", file=sys.stderr)
        else:
            print(f"[err] 已存在 active 议题 {existing}（一议题一主线）。先 close 再 open。", file=sys.stderr)
        return 2

    disc_id = next_disc_id()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = replace_row(text, "议题号 (ID)", f"`{disc_id}`")
    text = replace_row(text, "标题 (Title)", title)
    text = replace_row(text, "状态 (Status)", "`Open`")
    text = replace_row(text, "开题时间 (Opened)", f"`{now}`")
    if owner:
        text = replace_row(text, "发起人 (Owner)", f"`{owner}`")
    DISCUSSION.write_text(text, encoding="utf-8")
    print(f"[ok] opened {disc_id}: {title}")
    print("[next] 在 Open Questions 填入待决问题，并在 Posts 写首发帖。")
    return 0


def cmd_close(slug: str) -> int:
    text = DISCUSSION.read_text(encoding="utf-8")
    disc_id = active_id(text)
    if not disc_id:
        print("[err] Discussion.md 没有 active 议题（仍是模板态），无可归档。", file=sys.stderr)
        return 2
    if "Resolved" not in row_value(text, "状态 (Status)"):
        print(f"[err] {disc_id} 的 Status 不是 Resolved。先在 Issue Header 把状态切到 `Resolved`。", file=sys.stderr)
        return 2

    resolution = text.split("## Resolution", 1)[-1].split("\n## ", 1)[0]
    residues = [r for r in TEMPLATE_RESIDUE if r in resolution]
    if residues:
        print(f"[err] Resolution 段仍有未填模板占位：{residues}。全部填完才能关闭（AGENTS.md § 7）。", file=sys.stderr)
        return 2

    if not TEMPLATE.exists():
        print(f"[err] 模板缺失：{TEMPLATE.relative_to(REPO)}。", file=sys.stderr)
        return 2

    slug = re.sub(r"[^\w一-鿿-]+", "-", slug).strip("-") or "untitled"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE / f"{disc_id}-{slug}.md"
    if dest.exists():
        print(f"[err] {dest.relative_to(REPO)} 已存在，换一个 slug。", file=sys.stderr)
        return 2

    shutil.move(str(DISCUSSION), str(dest))
    shutil.copy(str(TEMPLATE), str(DISCUSSION))
    print(f"[ok] archived -> {dest.relative_to(REPO)}")
    print("[ok] Discussion.md 已从 tools/templates/Discussion.template.md 重置")
    print("[next] 按 Resolution.Propagated to，在 method.md / idea.md 受影响章节追加 changelog 并链回归档路径；"
          '随后可 `python tools/new_disc.py open "<标题>"` 开启下一议题。')
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    cmd, rest = args[0], args[1:]
    if cmd == "next":
        print(next_disc_id())
        return 0
    if cmd == "open":
        if not rest:
            print('usage: python tools/new_disc.py open "<标题>" [--owner "PI @张三"]', file=sys.stderr)
            return 1
        owner = None
        if "--owner" in rest:
            i = rest.index("--owner")
            if i + 1 >= len(rest):
                print("[err] --owner 后缺少值", file=sys.stderr)
                return 1
            owner = rest[i + 1]
            rest = rest[:i] + rest[i + 2:]
        return cmd_open(rest[0], owner)
    if cmd == "close":
        if not rest:
            print('usage: python tools/new_disc.py close "<slug>"', file=sys.stderr)
            return 1
        return cmd_close(rest[0])
    print(f"[err] unknown subcommand: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
