# Method（数学方法说明）

## 1. 方法定位

- **方法名**：\[填写\]
- **核心改进目标**：\[例如：提高鲁棒性/稳定性/泛化\]
- **数学层级**：
  - 最优：数学先指导方法设计（含 theorem / corollary）
  - 中等：可数学解释为什么奏效
  - 基础：可数学描述但暂无完整推导

---

## 2. 形式化定义

- 任务目标：
  \[
  \min_{\theta}\ \mathbb{E}_{(x,y)\sim\mathcal D}\big[\mathcal L(f_\theta(x),y)\big]
  \]
- 总损失（示例）：
  \[
  \mathcal L_{total}
  = \alpha\mathcal L_{task}
  + \beta\mathcal L_{reg}
  + \gamma\mathcal L_{defense}
  \]
- 关键假设：\[A1, A2, A3...\]

---

## 3. 关键结论（可逐步补完）

- **Theorem 1（草稿）**：\[填写自然语言结论 + 形式化表达\]
- **Corollary 1（草稿）**：\[填写\]
- **Proof Sketch**：\[填写 3-5 步证明思路\]

---

## 4. 为什么奏效

- 与 baseline 的本质差异：\[填写\]
- 理论解释（几何/信息论/优化视角）：\[填写\]
- 可被实验验证的结论：\[填写\]

---

## 5. 与实验日志对齐

- 对应 `LOGS/YYYY-Www.md` 中实验 ID：\[EXP-...\]
- 每次方法有实质变更时，更新本文件并在 § 6 Changelog 追加一行。

---

## 6. Changelog（每次实质变更必填）

> 实质变更 = 改公式 / 改假设 / 改 theorem 陈述 / 替换 loss 项 / 改超参定义。**纯文字润色不必记录**。
> 分工：本表只记**公式级**变更；研究方向级演进（目标、范围、瓶颈）记在 `idea.md § 6`，不要两边重复。

| 日期 | 改动摘要 | 触发原因 | 关联议题 | 关联实验 | commit | 操作人 |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | 例：在 $\mathcal L_{defense}$ 中加入 Lipschitz 约束 $\lambda \|\nabla_x f\|$ | 例：`EXP-2026W10-003` 显示原始项对 PGD 不鲁棒 | `DISC-2026W10-002` | `EXP-2026W10-003,004` | `abc1234` | `PI @TODO` |
