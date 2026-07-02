# LOGS 说明（按周记录）

- `LOGS/` 采用周维度记录实验结果。
- 一周一个文件，命名为 `YYYY-Www.md`（例：`2026-W10.md`）。
- 同一周内所有实验都追加在该文件中。

## 实验 ID 格式（唯一标准）

- **`EXP-YYYYWww-NNN`**（例：`EXP-2026W10-001`，三位序号）。
- `YYYY` = 年（4 位），`ww` = ISO 周序（2 位），`NNN` = 当周内序号（3 位，从 001 起）。
- 全协议（`AGENTS.md`、`method.md`、`Discussion.md`、`LOGS/*`）统一使用此格式。

## 每条实验必填字段（强制）

```
### EXP-YYYYWww-NNN

- 源意图 (Original Vibe):
- 假设 (Hypothesis):
- 是否被驳斥 (Falsified?):   Y / N / 部分 / Crashed
- 驳斥/支持原因 (Why):
- Agent 动作 (What changed):
- 复现信息 (Repro):
  - commit:                 <git sha 或 dirty>
  - seed:                   <int 或 N/A>
  - dataset / version:
  - env:                    <python/cuda/key libs>
  - hardware:               <GPU/CPU>
  - command:                `bash ...`
- 关键指标 (Metrics):
- 日志路径 (Artifacts):     <wandb / 文件路径>
- 结论 (Conclusion, 1–3 句):
- 下一步 (Next):
- 关联议题 (Discussion):     DISC-YYYYWww-NNN
```

## 负结果原则

- `Falsified=Y` 的实验同样必须完整记录，**不允许悄悄删除或重命名**。
- 负结果与正结果同等重要——半年后回看，最有价值的往往是"我们当时验证了 X 不行"。
- **跑崩也要记**：跑崩 / 不收敛而中止的实验记 `Falsified=Crashed`，块内容可精简，但 `commit / command / Why（崩溃现象）` 必填。`AGENTS.md § 10` 的"连续 3 次跑崩"就以这里的连续 `Crashed` 记录机械计数。

## 新建周志 / 实验块

使用 `tools/` 下脚本：
- `python tools/new_week.py` — 新建当周 `LOGS/YYYY-Www.md`（如果不存在）。
- `python tools/new_exp.py "源意图"` — 在当周文件追加一个填好骨架的 EXP 块（自动写入下一个 `NNN`、commit hash、主机名、当前 active 议题号）。
- `python tools/lint_protocol.py --strict LOGS/` — 检查 EXP 块字段完整性与占位符（Reflect 后置要求清零，见 `AGENTS.md § 4.2`）。
