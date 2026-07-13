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

> 【PI 已批准 2026-07-13】H1 全轨迹多轮 per-step 奖励。触发：Gate 2（EXP-2026W28-008）证「只训开场白」杠杆太弱、两臂不分化 → 改全轨迹多轮、per-turn potential 信用分配。

**Episode（交互多轮提取）**：一次 episode 是有上限 `T` 轮的对话。第 `t` 轮攻击者策略 `π_θ` 依历史 `h_{t-1}` 生成消息 `a_t ~ π_θ(·|h_{t-1})`；冻结 victim 回复；纯规则 oracle（`score_disclosure`，无 LLM judge）给出势函数
\[
\Phi_t \;=\; \frac{\#\{\text{已披露的 field canary}\}}{K}\ \in[0,1],\qquad \Phi_0=0,\quad \Phi_{t}\ge\Phi_{t-1}\ (\text{披露单调累积}).
\]
终局成功：`Φ_T ≥ τ`。

**两臂 per-turn 奖励**（沿用 `reward.py`：dense=势 Φ / sparse=终局）：
\[
r^{\text{dense}}_t=\Delta\Phi_t=\Phi_t-\Phi_{t-1}\ \ge 0,\qquad
r^{\text{sparse}}_t=\mathbb{1}[\Phi_t\ge\tau\ \wedge\ \Phi_{t-1}<\tau].
\]
telescoping：`Σ_t r^dense_t = Φ_T`，`Σ_t r^sparse_t = 1[Φ_T≥τ]`（两臂**同终局回报**，仅**中途稠密度**不同 → 隔离过程奖励效应）。

**GRPO 组相对、per-turn return-to-go 优势**：每个 goal `g` on-policy 采 `G` 条轨迹 `{τ^i}`。第 `i` 轨第 `t` 轮 return-to-go 与逐位置基线：
\[
G_{i,t}=\sum_{k\ge t} r_{i,k},\qquad
b_t=\operatorname{mean}_{i}G_{i,t},\quad \sigma_t=\operatorname{std}_i G_{i,t},\qquad
A_{i,t}=\frac{G_{i,t}-b_t}{\sigma_t+\varepsilon}.
\]
策略梯度（GRPO/REINFORCE，PPO-clip 可选）：
\[
\nabla_\theta J=\mathbb E\Big[\sum_t A_{i,t}\,\nabla_\theta\log\pi_\theta(a_{i,t}\mid h_{i,t-1})\Big].
\]

**关键假设**：A1 victim 冻结（两臂同一 victim，Gate 1′ 冻结的 light 档）；A2 `Φ` 由规则 oracle 决定、单调、`Φ=1⇔`全披露（可解释、无 judge）；A3 所有 attacker turn 都是 `π_θ` 输出（on-policy 全轨迹）。

---

## 3. 关键结论（可逐步补完）

> 【PI 已批准 2026-07-13】

- **Claim 1（H1 机制）**：在硬/OOD goal 上（终局成功稀疏），
  - dense 的 `G_{i,t}=Φ_{T,i}-Φ_{t-1,i}>0` 只要**未来有任何披露**——即使该轨迹**从未达 τ**，凡有部分进度的轮次都得正学习信号（部分信用）；
  - sparse 的 `G_{i,t}=1[Φ_{T,i}\ge\tau]`——凡**未达 τ 的轨迹其所有轮次梯度恒 0**。
  故 success 稀疏时 sparse 近乎无梯度、dense 仍从部分进度学习 → **dense 在 OOD 泛化更好**（H1 主张）。
- **Corollary（与 Gate 2 失败对齐）**：若被训策略**不控制**分级进度（如「只训开场白+固定 base 跟进」），`A_{i,t}` 与被训 token 解耦，dense≈sparse——正是 EXP-2026W28-008 所见。全轨迹（A3）是 dense 生效的前提。
- **Proof Sketch**：(1) Φ 单调 ⇒ `r^dense_t≥0`、return-to-go 分级；(2) 势塑形 telescoping ⇒ 两臂同终局回报，dense 仅加中途密度（Ng et al. 1999 不改最优策略、加速信用分配）；(3) 稀疏 success 下 sparse 的组内回报多为全 0（`σ_t→0`，优势退化），dense 组内回报有方差 ⇒ 有效梯度；(4) OOD 上 success 更稀疏 ⇒ 差距放大。

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
| 2026-07-13 | §2/§3 填入 H1 全轨迹多轮 per-turn potential 奖励：`r^dense_t=ΔΦ_t` / `r^sparse_t=终局`，return-to-go 组相对优势 `A_{i,t}=(G_{i,t}-b_t)/σ_t`；Claim 1 机制（`mt_grpo.py` golden 验证） | Gate 2「只训开场白」杠杆太弱、两臂不分化 | `DISC-2026W28-001` | `EXP-2026W28-008` | dirty | **PI 已批准 @bzhu11 2026-07-13** |
