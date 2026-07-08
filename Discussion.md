# 💬 Discussion · 当前 Active 议题

> **协议要求**：本文件同一时刻只承载 **1 个 active 议题**。议题关闭后整体迁移到 `Discussion/Archive/DISC-YYYYWww-NNN-<slug>.md`，本文件从 `tools/templates/Discussion.template.md` 重置或立即承载下一个议题。

---

## Issue Header

| 字段 | 值 |
|---|---|
| **议题号 (ID)** | `DISC-YYYYWww-NNN` |
| **标题 (Title)** | 一句话写清要解决的"一个具体问题" |
| **状态 (Status)** | `Open` / `Resolved` |
| **发起人 (Owner)** | `PI|Lead|Collab @姓名` |
| **开题时间 (Opened)** | `YYYY-MM-DD HH:MM` |
| **关联 idea/method** | `idea.md §2.3` / `method.md §3` |
| **关联实验** | `EXP-YYYYWww-001`, ... |

---

## Open Questions（待决清单）

> 维护一组**收敛性问题**，每条带 owner 与到期，防止讨论绕圈。

- [ ] Q1：…… (owner: `Lead`, due: `YYYY-MM-DD`)
- [ ] Q2：…… (owner: `Agent`, due: `YYYY-MM-DD`)
- [ ] Q3：…… (owner: `PI`, due: `YYYY-MM-DD`)

---

## Posts（回帖区）

> **格式**：每条以 `【角色@姓名】【YYYY-MM-DD HH:MM】` 开头。
> **角色**：`PI` / `Lead` / `Collab` / `Agent`。
> **Agent 发言必须**：(a) 链接到 `LOGS/...#EXP-...`；(b) 直接贴关键数字，不让读者跳页。

---

【PI @TODO】【YYYY-MM-DD HH:MM】
（首发：把问题、当前怀疑、希望验证什么，写清楚）

---

【Agent @TODO】【YYYY-MM-DD HH:MM】
（回帖示例）
- 已跑 `EXP-2026W10-001` ([LOGS/2026-W10.md](LOGS/2026-W10.md))，结果：
  - metric_a = 0.823 (+1.2% over baseline)
  - metric_b = 0.471 (-3.5%, **未支持原假设**)
- 建议：是否将 Q1 调整为"区分 metric_a / metric_b 增益来源"？

---

## Resolution（关闭议题时必填）

> Status 切到 `Resolved` 时，本节必须全部填好；否则不许关闭。

- **Decision**：(一段话说明最终结论)
- **Rationale**：(为什么这是结论，基于哪些 EXP / 哪些论证)
- **Propagated to**：
  - `method.md § X`（写明改了什么）
  - `idea.md § Y`（如有）
  - `EXP-YYYYWww-NNN`（受影响的实验）
- **Closed by**：`PI @姓名`
- **Closed at**：`YYYY-MM-DD HH:MM`

---

## 关闭流程（Agent 执行）

1. 确认 `Resolution` 各字段已填（Decision / Rationale / Propagated to / Closed by / Closed at），且 Status 已切到 `Resolved`。
2. 运行 `python tools/new_disc.py close "<slug>"`：脚本校验 Resolution → 归档为 `Discussion/Archive/DISC-YYYYWww-NNN-<slug>.md`（文件名含议题号，可反查）→ 从 `tools/templates/Discussion.template.md` 重置本文件。
3. 在 `method.md` / `idea.md` 受影响章节追加 changelog 条目，并反向链回归档路径。
4. 开启下一个议题：`python tools/new_disc.py open "<标题>"`（编号由脚本分配，避免撞号）。
