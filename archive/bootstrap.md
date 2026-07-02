# 欢迎 · 项目初始化向导

> 这是仓库的**首次入口**。完成本向导后，本文件会被自动删除，模式状态保存在 `MODE.md`。

---

## 1. 这套协议是干什么的（30 秒版）

用 **Markdown 文件驱动科研过程**：你给方向，Agent 负责重复执行（写代码、跑实验、整理日志），关键状态全部落盘成可读文件，方便多人协作和半年后复现。

**核心文件一览**：

| 文件 | 作用 | 何时读 |
|---|---|---|
| `AGENTS.md` | 协议正文（唯一来源） | 每次会话开头 |
| `idea.md` | 研究问题、动机、目标 | 立项 + 每次 Plan |
| `method.md` | 方法的数学形式化 + 变更日志 | 改公式 / 改假设时 |
| `Discussion.md` | **当前 active 议题**（多人主战场，一次一个议题） | 每次 Plan + 议题更新时 |
| `Discussion/Archive/` | 已关闭议题归档 | 复盘 / 找历史决策 |
| `LOGS/YYYY-Www.md` | 周实验日志（按周分文件） | 每次实验跑完 |
| `MODE.md` | 当前协作模式（newbie/expert） | 每次会话开头 |
| `code/` | 主代码 | Execute 阶段 |
| `baseline/` | Baseline 代码（可没有） | 对比实验时 |
| `ref/` | 资料/论文（默认不读，被显式引用时必读） | 用户点名时 |
| `tools/` | 协议辅助脚本 | 新建周志/新建实验时 |

**日常工作循环（P-E-R）**：
1. **Plan** — 读 `MODE.md`+`idea.md`+`Discussion.md`，锁定本轮唯一问题。
2. **Execute** — 改 `code/`，必要时同步 `method.md`。
3. **Reflect** — 把实验追加到当周 `LOGS/YYYY-Www.md`，把结论回写 `Discussion.md`。

---

## 2. 选择协作模式（**仅回复 A 或 B**）

### A · 科研新手（newbie）
- 适合：第一次用这套流程，希望一步一步引导。
- 策略：最小下一步、边做边写、先 Pilot 再扩展。

### B · 科研老手（expert）
- 适合：熟悉流程，追求推进速度。
- 策略：结论先行、批量执行、少教学多产出。

> 模式后续可随时切换：对 Agent 说"切换为新手/老手模式"即可。
> 选择前也可以先就协议本身提问——Agent 会回答，但在你回复 A/B 之前不会执行任何研究动作（不建议题、不改代码、不写日志）。

---

## 3. Agent 收到 A/B 后必须做的事

1. 把选择写入 `MODE.md`：
   - `mode: newbie | expert`
   - `updated_at: <当前时间，UTC+8>`
   - `set_by: bootstrap.md`
   - `last_retro: <当前 ISO 周，如 2026-W24>`（初始化当周视为已回顾，避免立刻欠账）
2. **按所选模式执行一次"零号动作"**：
   - **newbie**：引导用户填 `idea.md` 中的 **One-Sentence Summary** 和 **Primary Metric** 两项（其余可留白）；并解释 P-E-R 的第一步该怎么做。
   - **expert**：跳过教学，直接询问"当前要解决的第一个议题是什么？"，并按 `AGENTS.md § 7` 在 `Discussion.md` 创建议题 `DISC-YYYYWww-001`。
3. 删除 `bootstrap.md`。
4. （可选）若 `Discussion/Archive/` 目录不存在，顺手创建。

---

## 4. 出错怎么办

- 若误删 `bootstrap.md` 但仍想重新初始化：把本文件从 git 历史里 `git checkout` 回来即可。
- 若 `MODE.md` 内容损坏：手动改回 `mode: unset` 并把本文件 restore，下次会话会重新触发向导。
- 若 hook 未触发（非 Claude Code 环境）：按 `AGENTS.md § 2.1` 手动执行等价初始化。
