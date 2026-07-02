#!/usr/bin/env python3
"""Lint protocol Markdown files.

Errors（任何状态下都报，exit 1）:
- 坏格式 EXP id（应为 EXP-YYYYWww-NNN）
- 坏格式 DISC id（应为 DISC-YYYYWww-NNN）
- Discussion 文件标记 Resolved 但 Decision 缺内容

Warnings（仅初始化后，即 bootstrap.md 已删除；默认 exit 0，--strict 时升级为 error）:
- 未填的 `[填写...]` 占位符
- 残留的 YYYY-MM-DD / YYYY-Www / NNN 字面占位
- LOGS 周文件中 EXP 块必填字段为空（假设 / 是否被驳斥 / 结论 / command / 关联议题）

永久模板文件（AGENTS.md、各 README、bootstrap.md、Discussion.md、tools/templates/）
豁免 warning；格式类 error 对所有文件生效。
含 `例：` / `示例` 的行跳过 warning 检查。

Usage:
    python tools/lint_protocol.py                     # lint all（warning 不阻塞）
    python tools/lint_protocol.py --strict LOGS/      # Reflect 后置：warning 也算失败
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# 这些文件本身就是模板/协议文档，占位符是刻意保留的 —— 永久豁免 warning。
# Discussion.md 的生命周期包含"关闭后重置回模板态"，因此同样豁免占位 warning
# （格式 error 与 Resolved/Decision 检查仍然生效）。
ALWAYS_TEMPLATE = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "bootstrap.md",
    "Discussion.md",
    "tools/templates/Discussion.template.md",
    "LOGS/README.md",
    "tools/README.md",
    "ref/README.md",
    "ref/notes/README.md",
    "code/README.md",
    "baseline/README.md",
    "Discussion/Archive/README.md",
}

PLACEHOLDER_RE = re.compile(r"\\?\[填写[^\]]*\\?\]")
DATE_PLACEHOLDER_RE = re.compile(r"YYYY-MM-DD")
WEEK_PLACEHOLDER_RE = re.compile(r"YYYY-Www|YYYYWww")
NNN_RE = re.compile(r"\bNNN\b")
EXP_BAD_RE = re.compile(r"EXP-(?!\d{4}W\d{2}-\d{3})\S+")
DISC_BAD_RE = re.compile(r"DISC-(?!\d{4}W\d{2}-\d{3})\S+")
STUBS = ("YYYYWww", "YYYY-Www", "NNN", "...", "ID", "*")
EXAMPLE_MARKERS = ("例：", "例:", "示例")
WEEK_FILE_RE = re.compile(r"^LOGS/\d{4}-W\d{2}\.md$")

EXP_REQUIRED = ("假设", "是否被驳斥", "结论", "command", "关联议题")
EMPTY_VALUES = {
    "",
    "Y / N / 部分",
    "Y / N / 部分 / Crashed",
    "DISC-NNN",
    "DISC-YYYYWww-NNN",
    "bash ...",
}


def relevant_md_files(roots: list[Path]) -> list[Path]:
    files = []
    for root in roots:
        if root.is_file() and root.suffix == ".md":
            files.append(root)
        elif root.is_dir():
            files.extend(p for p in root.rglob("*.md") if ".git" not in p.parts)
    return files


def format_errors(rel: str, text: str) -> list[str]:
    """坏格式 EXP / DISC id（占位 stub 除外）。"""
    issues = []
    for bad_re, expect in ((EXP_BAD_RE, "EXP-YYYYWww-NNN"), (DISC_BAD_RE, "DISC-YYYYWww-NNN")):
        for m in bad_re.finditer(text):
            token = m.group(0)
            if any(stub in token for stub in STUBS):
                continue
            issues.append(f"{rel}: 格式错误 {token!r}（应为 {expect}）")
    return issues


def resolved_errors(rel: str, text: str) -> list[str]:
    """Discussion 文件：Status=Resolved 但 Decision 没有实际内容。"""
    if not rel.startswith("Discussion"):
        return []
    status_m = re.search(r"\*\*状态 \(Status\)\*\*\s*\|([^|\n]*)\|", text)
    cell = status_m.group(1) if status_m else ""
    # 模板态的 Status 单元格同时含 Open / Resolved，跳过
    if "Resolved" not in cell or "Open" in cell:
        return []
    m = re.search(r"\*\*Decision\*\*[:：](.*)", text)
    value = m.group(1).strip() if m else ""
    if value and "一段话说明最终结论" not in value:
        return []
    return [f"{rel}: 状态为 Resolved 但 Decision 缺内容"]


def placeholder_warnings(rel: str, text: str) -> list[str]:
    """初始化后仍残留的占位符（每行最多报一条；示例行跳过）。"""
    warns = []
    checks = (
        (PLACEHOLDER_RE, "未填占位符 [填写...]"),
        (DATE_PLACEHOLDER_RE, "字面日期占位 YYYY-MM-DD"),
        (WEEK_PLACEHOLDER_RE, "字面周占位 YYYY-Www"),
        (NNN_RE, "未填序号占位 NNN"),
    )
    for i, line in enumerate(text.splitlines(), 1):
        if any(mark in line for mark in EXAMPLE_MARKERS):
            continue
        for regex, label in checks:
            if regex.search(line):
                warns.append(f"{rel}:{i}: {label}")
                break
    return warns


def exp_block_warnings(rel: str, text: str) -> list[str]:
    """LOGS 周文件：每个 EXP 块的必填字段非空（§ 5.2「所有字段必填」的机械防线）。"""
    warns = []
    blocks = re.split(r"(?=^### EXP-)", text, flags=re.MULTILINE)
    for block in blocks:
        if not block.startswith("### EXP-"):
            continue
        exp_id = block.splitlines()[0].removeprefix("### ").strip()
        for key in EXP_REQUIRED:
            line = next(
                (l for l in block.splitlines() if ":" in l and key in l.split(":", 1)[0]),
                None,
            )
            if line is None:
                warns.append(f"{rel}: {exp_id} 缺字段 '{key}'")
                continue
            value = line.split(":", 1)[1].strip().strip("`")
            if value in EMPTY_VALUES or value.startswith("<"):
                warns.append(f"{rel}: {exp_id} 字段 '{key}' 未填")
    return warns


def main() -> int:
    args = sys.argv[1:]
    strict = "--strict" in args
    paths = [a for a in args if a != "--strict"]
    roots = [REPO / p for p in paths] if paths else [REPO]
    initialized = not (REPO / "bootstrap.md").exists()

    files = relevant_md_files(roots)
    errors: list[str] = []
    warnings: list[str] = []
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        text = f.read_text(encoding="utf-8", errors="replace")
        errors.extend(format_errors(rel, text))
        errors.extend(resolved_errors(rel, text))
        if initialized and rel not in ALWAYS_TEMPLATE:
            warnings.extend(placeholder_warnings(rel, text))
            if WEEK_FILE_RE.match(rel):
                warnings.extend(exp_block_warnings(rel, text))

    for line in errors:
        print(f"[err]  {line}")
    for line in warnings:
        print(f"[warn] {line}")

    failed = bool(errors) or (strict and bool(warnings))
    if not errors and not warnings:
        print(f"[ok] scanned {len(files)} file(s), no issues")
    else:
        tag = "fail" if failed else "ok-with-warnings"
        print(f"\n[{tag}] {len(errors)} error(s), {len(warnings)} warning(s) across {len(files)} file(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
