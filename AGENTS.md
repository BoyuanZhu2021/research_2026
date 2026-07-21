# Agent-Centric Research Protocol

> 本文件是协议的 **single source of truth**。`CLAUDE.md` 仅是指向本文件的引用。

---

## 0. 核心哲学（必须保留）

- **Agent First**：除高阶科学决策外，重复执行任务由 Agent 主导。
- **Everything is Markdown**：研究过程即文档，关键状态必须落盘到 Markdown。
- **Vibe Alignment**：用户给意图与方向，Agent 负责把意图转成实现与实验。
- **Math-Guided Improvement**：优先让数学指导方法改进；至少要能数学描述与解释。

---

## 1. 核心锚点文件

- `HANDOFF.md`：**接手点（会话/换 Agent 必先读）**——当前阶段、进行中任务、下一步（含可直接跑的命令）、产物位置、阻塞项。`tools/session_check.py` 会在会话简报首行surface它。**任务或其下一步变化时（会话结束、跑完一段、阻塞/解阻塞）必须更新本文件。** 详细历史仍在 LOGS/Discussion/method；本文件只是指针。
- `idea.md`：研究问题、动机、目标、实验大图。
- `method.md`：方法的数学形式化（损失、假设、推论/定理、解释）+ 变更日志。
- `Discussion.md`：**当前唯一 active 议题**（多人协作主战场，一议题一主线）。
- `Discussion/INDEX.md`：**议题总登记表**（active + 已归档，确保无议题遗漏）；`Discussion/Archive/`：已关闭议题归档（每议题一个 md）。开/关议题时同步更新 INDEX。
- `docs/plans/`：已批准的计划快照（跨 Agent 持久，任何工具可读）。
- `LOGS/`：按周记录实验结果（每周一个 `YYYY-Www.md`）。
- `MODE.md`：当前协作模式（newbie/expert）、策略与 `last_retro`（上次周回顾的 ISO 周）。
- `code/`：项目主代码目录。
- `baseline/`：Baseline 代码目录（可不存在）。
- `ref/`：论文/资料/笔记（默认不主动扫描；被显式引用时必读，读后笔记沉淀到 `ref/notes/`，详见 § 8）。
- `tools/`：协议辅助脚本（会话自检、新建周志/实验块/议题、占位符 lint）。

---

## 2. 首次初始化（bootstrap 机制）

1. 若仓库存在 `bootstrap.md`，优先执行初始化向导：读取 `bootstrap.md` 并向用户完整呈现其 § 1–2。**引导文本以 `bootstrap.md` 为唯一来源**，hook / 脚本只负责触发，不另存一份文案。
2. 用户在向导中**仅回复 `A` 或 `B`** 选择模式：A=newbie，B=expert。
3. **初始化完成前的边界**：Agent 可以回答关于协议/仓库本身的元问题，但**不得执行任何 P-E-R 动作**（不创建议题、不改 `code/`、不写 LOGS）；每次回复末尾重复 A/B 选择引导，直到用户完成选择。
4. Agent 将选择结果写入 `MODE.md`（`mode / updated_at / set_by / last_retro`）。
5. 完成初始化后删除 `bootstrap.md`。
6. 删除后，后续策略以 `MODE.md` 为准。

### 2.1 跨 Agent 兼容触发规则

- **所有 Agent**（Claude Code / Codex / Cursor / 其他）在进入仓库后的第一条有效回复前，必须先运行并遵循：

  ```bash
  python3 tools/session_check.py
  ```

  该脚本是「会话开始检查」的**唯一实现**，输出三种之一：
  1. `bootstrap.md` 存在 → 初始化指引（读取 `bootstrap.md` 并要求用户仅回复 A 或 B）；
  2. 已初始化 → 会话简报：mode、active 议题、当周 LOGS 状态、是否欠 Weekly Retro（§ 6）；
  3. 协议异常（§ 2.2 第三行）→ 提示用户从模板重建 `bootstrap.md`。
- Claude Code 的 `SessionStart` hook（`.claude/settings.json`）调用的是**同一脚本**，因此不存在"hook 文案"与"手动检查文案"两份拷贝需要维护。
- 若环境没有 python3：退化为手动检查——`bootstrap.md` 存在则读取并向用户呈现其 § 1–2，要求仅回复 A 或 B。
- 用户完成模式选择前，不得进入 P-E-R 循环（允许回答元问题，边界见 § 2 第 3 条）。

### 2.2 模式状态判定（消歧）

| `bootstrap.md` | `MODE.md::mode` | 行为 |
|---|---|---|
| 存在 | 任意 | **未初始化**，必须先走 bootstrap |
| 不存在 | `newbie` 或 `expert` | 按该模式正常工作 |
| 不存在 | `unset` 或缺失 | 视为协议异常，提示用户从模板重建 `bootstrap.md` |

---

## 3. 模式策略（由 MODE.md 驱动）

### 3.1 科研新手模式（newbie）
- 交互：一步一引导，不一次抛太多选项。
- 执行：默认给出最小下一步（single next action）。
- 文档：每完成一步同步写入对应文件。
- 实验：先做小规模验证（Pilot），再扩展。

### 3.2 科研老手模式（expert）
- 交互：结论先行，少解释，多直接执行。
- 执行：以任务批次推进，减少确认轮次。
- 文档：只保留关键变更与结论，避免冗长。
- 实验：直接进入主实验/消融，不强制教学式拆解。

---

## 4. 日常工作循环（P-E-R）

### 4.1 三步循环

1. **Plan**：读取 `MODE.md` + `idea.md` + `Discussion.md`，明确当前唯一问题。
2. **Execute**：修改 `code/`（或 `baseline/` 对照），必要时更新 `method.md`。
3. **Reflect**：在当周 `LOGS/YYYY-Www.md` 追加实验块，并回写 `Discussion.md` 共识。

### 4.2 前置 / 后置检查

| 阶段 | 必须先做 / 必须后做 |
|---|---|
| **Plan 前置** | 若 `Discussion.md` 无 active 议题，先与用户对齐一个并按 § 7 创建；若 `idea.md` 关键字段（One-Sentence Summary / Primary Metric）为空，先填这两项再继续。新建函数、脚本、配置、测试或入口前，必须先按 § 4.3 搜索并判断能否复用现有实现。 |
| **Execute 后置** | 若 `code/` 非空且存在测试入口，运行一次 lint + 单测；任何 method 公式/假设的实质改动必须同步进 `method.md` 并写入其 Changelog（§ 9）。按 § 4.3 完成残留清理与工作树核对后，Execute 才算完成。 |
| **Reflect 后置** | 当周 `LOGS/YYYY-Www.md` 至少追加一条完整 EXP 块（按 § 5 字段全填）；运行 `python tools/lint_protocol.py --strict LOGS/<当周>.md`，error 与 warning 清零后 Reflect 才算完成；若该实验影响当前议题结论，必须在 `Discussion.md` 该议题下回帖。 |

### 4.3 代码复用与残留清理（强制）

目标是让仓库始终只有仍在使用、可以解释其用途的实现。不得用不断新增平行脚本的方式推进任务，也不得把临时代码留给后续 Agent 猜测。

1. **先搜索，后新增。** 每个新任务开始时，先用 `rg` 检查相关函数、类、脚本、配置、测试、调用点和文档。能扩展、组合或参数化现有实现时，必须复用；没有完成搜索和复用判断前，不得新建平行入口。
2. **禁止复制式迭代。** 不得通过复制旧文件并添加 `old`、`new`、`v2`、`copy`、`patch`、`debug`、`final` 等后缀来规避修改现有实现。确需并存的版本必须有不同的长期职责、正式命名、调用入口和文档说明，否则只保留一个正式实现。
3. **临时代码必须有结束点。** 一次性探针、调试脚本、临时测试、手工补丁和中间输出优先放到系统临时目录，不得放进正式源码目录。若因工具限制暂时写入仓库，必须在同一任务结束前删除；有长期价值的测试要整理成正式回归测试并接入现有测试入口。
4. **替换完成就清旧实现。** 新实现接管后，必须同步更新调用方、测试、配置和文档，并删除已经失去用途的旧代码、旧脚本、旧补丁、注释死代码和重复入口。历史由 Git、LOGS 或 Discussion 保存，不靠仓库内旧副本保存。
5. **交付前逐项核对。** 完成前运行 `git status --short` 并检查本次 diff；对本次新增或修改的每个文件说明长期用途，再用 `rg` 核对被替换入口是否仍有引用。无法说明用途的 Agent 新建文件必须删除，发现未完成迁移或平行实现时不得宣称完成。
6. **清理后必须验证。** 删除或合并后运行与影响范围相称的 lint、单测或最小调用验证，确认没有残留引用和入口断裂。验证失败时恢复到可用状态并报告，不得把破损清理包装成完成。
7. **不盲删用户文件。** Agent 可以清理自己在当前任务中创建且已确认无用途的临时内容；对会话开始前已经存在、用途不明或可能承载实验证据的文件，只能列出引用证据、替代关系和影响，得到用户确认后再删除。任何已有 LOGS/EXP 均受 § 5 约束，不属于临时代码清理对象。

**完成门槛：** 工作树中不得留下本任务产生的临时 script、临时测试、一次性 patch、重复实现、注释死代码或无法解释用途的文件；否则任务状态只能写“尚未完成”。

---

## 5. LOGS 约定（周维度）

### 5.1 命名 & 组织
- `LOGS/` 每周一个文件：`YYYY-Www.md`（例：`2026-W10.md`）。
- 同一周所有实验追加在该文件。
- 实验 ID 统一格式：**`EXP-YYYYWww-NNN`**（例：`EXP-2026W10-001`，三位序号）。

### 5.2 每条 EXP 块必填字段

```
### EXP-YYYYWww-NNN

- 源意图 (Original Vibe):
- 假设 (Hypothesis):
- 是否被驳斥 (Falsified?):      Y / N / 部分 / Crashed
- 驳斥/支持原因 (Why):
- Agent 动作 (What changed):
- 复现信息 (Repro):
  - commit:                   <git sha 或 dirty>
  - seed:                     <int 或 N/A>
  - dataset / version:
  - env:                      <python/cuda/key libs>
  - hardware:                 <GPU/CPU>
  - command:                  `bash ...`
- 关键指标 (Metrics):
- 日志路径 (Artifacts):       <wandb / 文件路径>
- 结论 (Conclusion, 1–3 句):
- 下一步 (Next):
- 关联议题 (Discussion):       DISC-YYYYWww-NNN
```

**负结果原则**：`Falsified=Y` 的实验同样需要完整记录，**不允许悄悄删除或重命名**。负结果与正结果同等重要。

**崩溃同样记录**：跑崩 / 不收敛而中止的实验记 `Falsified=Crashed`，块内容可精简，但 `commit / command / Why（崩溃现象）` 必填。§ 10 的"连续 3 次跑崩"以 LOGS 中同一意图下连续的 `Crashed` 记录为准——机械可数，不依赖 Agent 跨会话记忆。

---

## 6. 周回顾（Weekly Retro）

- **触发**：机械判定，由 `tools/session_check.py` 在会话开始时给出——当 `MODE.md::last_retro` 早于当前 ISO 周、且存在更早一周的 `LOGS/YYYY-Www.md` 时，会话简报会标注"欠 Weekly Retro"，Agent 必须在该会话先完成回顾再进入其他工作。
- **动作**：
  1. 扫描上一周 `LOGS/YYYY-W(N-1).md` 所有 EXP；
  2. 按 `Falsified=Y / N / 部分` 分组汇总；
  3. 在 `Discussion.md` 当前 active 议题底部追加一条 `【Agent】【日期】Weekly Retro:` 回帖，列出 (a) 已驳斥假设 (b) 已支持假设 (c) 悬而未决问题；
  4. 若发现与 `idea.md` / `method.md` 的冲突，必须在 retro 里明确指出并建议对齐方案；
  5. 完成后将 `MODE.md::last_retro` 更新为当前 ISO 周（如 `2026-W24`）。

---

## 7. Discussion.md 多人协作约定（主战场）

详细模板见 `Discussion.md` 本身，此处仅规定不变量：

- **一议题一主线**：根目录 `Discussion.md` 同一时刻只承载 **1 个 active 议题**。
- **议题号**：`DISC-YYYYWww-NNN`，可被 LOGS / method.md / idea.md 反向引用；由 `python tools/new_disc.py open "<标题>"` 统一分配，避免手工撞号。
- **状态机**：`Open → Resolved`。无 `Decision` 段不得关闭。
- **角色化发言**：`【PI|Lead|Collab|Agent @名字】【YYYY-MM-DD HH:MM】`。
- **Agent 发言必带硬证据**：链接到 `LOGS/...#EXP-...` 并贴关键数字。
- **关闭流程**：用 `python tools/new_disc.py close "<slug>"` 执行——脚本校验 Resolution 非空后，归档到 `Discussion/Archive/DISC-YYYYWww-NNN-<slug>.md`（文件名含议题号，可反查），并从 `tools/templates/Discussion.template.md` 重置根目录 `Discussion.md`。
- **结论反哺**：议题 `Decision` 必须在 `Resolution.Propagated to` 字段列出反写到 `method.md §` / `idea.md §` / 哪些 `EXP-...`。

---

## 8. ref/ 使用规则

- 默认 **不主动扫描** `ref/`，避免污染上下文。
- **强制读取条件**：当 `idea.md` / `method.md` / `Discussion.md` 中出现 `ref/<path>` 形式的显式引用，Agent 必须读取被引用文件并在下一次发言里反映其内容。
- 用户口头要求时同样必须读。
- **读后沉淀**：每次因引用读取 `ref/<file>` 后，必须在 `ref/notes/<file>.md` 写入/更新要点笔记（核心结论、方法要点、与本项目的关联、可复用公式/数字）。同一文件再次被引用时**先读笔记**，笔记不足再回读原文——避免重复读全文污染上下文。

---

## 9. 决策边界：高阶 vs Agent 自治

| 类别 | 例子 | 谁拍板 |
|---|---|---|
| 修改 loss / 正则形式 | 加入 $\beta\|\nabla f\|^2$ | **用户** |
| 修改假设 / 定理陈述 | 改写 `method.md` § 2 / § 3 | **用户** |
| 引入新 baseline / 新数据集 | 加 NCFM-v2、换 ImageNet→CIFAR | **用户** |
| 改 `idea.md` 研究目标 | 改 Objective 1 | **用户** |
| 关闭 / 开启 Discussion 议题 | 任意 DISC-* | **用户**（Agent 可起草 Decision 草稿） |
| 调超参 / lr / batch / 网格搜索 | 任意 sweep | Agent |
| 跑现有 baseline / 复现 | 复现 NCFM | Agent |
| 数据预处理脚本 | tokenizer、resize | Agent |
| 写 / 修单元测试 | `tests/*` | Agent |
| 整理日志、补 EXP 字段 | 任意 LOGS 编辑 | Agent |
| 起草 `method.md` Changelog 条目 | 待用户审核 | Agent |

---

## 10. Stop / Escalate 触发条件

满足任一条件时，Agent **必须停下来征求用户**，不得自行继续：

- 任一实验连续 3 次跑崩 / 不收敛（以当周 LOGS 中同一意图下连续的 `Falsified=Crashed` 记录计，见 § 5.2）。
- 与目标 SOTA 的关键指标差距超过用户在 `idea.md` 设定的阈值（若未设，按 ≥10% 触发）。
- 需要修改 `method.md § 2 形式化定义` 或 § 3 关键结论。
- 需要修改 `idea.md` § 2.3 研究目标 或 § 4.3 主指标。
- 单次实验预估开销超过当前预算（用户未设预算时，按"单跑 >2h GPU 或 >5GB 写盘"触发）。
- 涉及"删除已有 EXP 记录"或"重写已 Resolved 议题结论"。

---

## 11. 质量与边界

- 涉及定理/推导/SOTA 对比：给出处或推导过程，避免幻觉。
- 若 `idea.md`、`method.md`、`LOGS` 结论冲突：先指出冲突，再建议对齐方案。
- 默认不读取 `ref/`，除非 § 8 条件触发。
