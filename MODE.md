# Research Mode State

- `mode`: `newbie`
- `updated_at`: `2026-07-08 03:20`
- `set_by`: `bootstrap.md`
- `last_retro`: `2026-W30`

---

## 状态判定（与 AGENTS.md § 2.2 一致）

| `bootstrap.md` | `MODE.md::mode` | 行为 |
|---|---|---|
| 存在 | 任意 | **未初始化**，必须走 bootstrap 向导 |
| 不存在 | `newbie` / `expert` | 按该模式正常工作 |
| 不存在 | `unset` 或文件缺失 | 协议异常，提示用户从模板重建 `bootstrap.md` |

---

## 模式说明

### newbie
- 一步一引导，默认给最小下一步。
- 每步同步更新文档，帮助建立流程感。

### expert
- 结论先行，减少教学式解释。
- 支持批量推进，优先效率与实验吞吐。

---

## 切换规则

- 首次由 `bootstrap.md` 写入。
- 后续若用户明确说"切换为新手/老手模式"，立即更新本文件三字段（`mode / updated_at / set_by=user`）并生效。
- Agent **不得**自行切换模式，必须由用户发起。
- `last_retro`：周回顾的唯一触发依据（`tools/session_check.py` 据此判定是否欠账）。每次完成 Weekly Retro（AGENTS.md § 6）后，由 Agent 更新为当周 ISO 周（如 `2026-W24`）；初始化时由 bootstrap 写为当周。
