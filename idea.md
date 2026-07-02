# 可组合攻击技能驱动的多轮红队：面向 OOD / 难目标的小模型攻击策略

> 模式：idea 草稿（基于讨论投影到模板）

## 1. 一句话总结（必填）

- 一句话：把红队建模成「学习一个**可验证攻击技能库**」——攻击者把任意对抗目标**分解成可组合、可程序化检查的子技能**并跨轮执行，从而让稀疏终局奖励下学不动的 **OOD / 强对齐目标**，因「技能重组」而变得可泛化。


## 2. 一段话总结（必填）

- 一段话总结：

  现有自动红队存在一个共同盲点：攻击成功稀有且只在**末端**可见，攻击者只拿到稀疏 0/1 信号，于是记住的是 per-goal 的表面套路、而非可迁移的攻击能力——表现为在**未见目标(OOD)**与**强对齐目标**上停滞（单轮 rubric 的 NVIDIA 工作 OOD 仅 ~29%；多轮终局分类器的 DialTree 自承 >5 轮稀疏乏力）。本 idea 面向 **AI 红队 / 模型安全评测**研究者，提出从**策略侧**切入：将攻击目标分解为一组**可验证的子技能/子状态**（如"诱导目标承认拥有某能力""使 agent 调用前置工具""泄露部分 canary"），攻击者学习**规划+执行**一条技能序列，新目标 = 已知技能的**重新组合**。中间状态由 garak / AgentDojo 这类**程序化 oracle** 判定，从而获得技能级（per-turn）的可信信用分配。用 8B 开源小模型 + GRPO 即可训练，保持可本地、可复现、低成本。现在值得做：多轮红队与可验证奖励两条线 2025–2026 同时成熟，但**"可组合可验证技能 × 多轮 × 小模型"这个交集仍空**，且两篇 SOTA 的弱点恰好落在这里。


## 3. 一页纸总结

### 3.1 背景 & 问题

- 当前现状 / 痛点：
  - 自动红队的两条主线各有锁死的局限——**单轮 + 生成式 rubric 奖励**（NVIDIA, 2604.23067）；**多轮 + 末端学习型分类器奖励、仅有害内容**（DialTree, 2510.02286, ICLR'26）。
  - 共同根因：**奖励稀疏且只在终局可见** → 学不到"朝目标的渐进推进"，无法迁移到从未拿过奖励的 OOD 目标。
- 为什么现有做法不够好：
  - NVIDIA：in-domain ASR 85% 近饱和，但 **OOD 仅 29.35%**（基线 20.87%），且单轮一发打不穿强对齐目标。
  - DialTree：多轮，但奖励 = 单一 HarmAug-Guard 分类器、**仅末端**、**只做有害内容**；作者明言长程稀疏限制 >5 轮。
  - 两者都把"攻击"当**整体**学，没有把它拆成**可迁移、可组合的能力单元**。

### 3.2 目标 & 成功标准

- 成功时对谁有改变：给安全评测者一个**对未见攻击目标也能泛化**的小模型红队器；给防御方一个能产出"技能级"诊断（哪类攻击技能最有效）的审计工具。
- 粗略成功指标：
  - **主指标**：OOD（未见目标）ASR 显著高于 NVIDIA / DialTree 基线（攻 ~29% 这块洼地）。
  - **泛化曲线**：随"已学技能数"增加，对新目标的组合成功率上升（compositional generalization 证据）。
  - **难目标**：在强对齐 / 前沿目标上的 ASR 提升幅度 > 单轮 baseline。
  - **可信度**：成功由程序化 oracle 判定（非自评 judge），抗 reward-hacking。

### 3.3 核心方案（High-level）

- 关键思路 / 机制：
  1. **技能库**：定义一组可复用的攻击原语/子技能（角色铺垫、前提注入、混淆、分步诱导、工具诱导…），每个技能有**可程序化判定的完成条件**。
  2. **目标分解**：给定任意目标，自动生成一条**子技能 / 子状态序列**（把 NVIDIA 的"终局 rubric"改造成"可验证子目标分解"）。
  3. **规划-执行（多轮）**：攻击者跨轮选择并执行技能，利用目标上一轮回应决定下一步；多轮是底座而非卖点。
  4. **技能级信用分配**：每个子技能用 oracle 判定是否达成 → 密集、可验证的 per-turn/技能级奖励；GRPO 在轨迹层做 group-relative。
  5. **OOD = 重组**：新目标用已学技能的新组合达成 → 泛化由机制涌现。
- 关键设计决策（含假设）：
  - 域选择被机制**逼定**在**可验证/agentic 域**（提示注入、数据泄露、agentic 工具滥用），因为只有这些域有可检查的中间状态；纯有害内容因"中间进度模糊"暂不作主战场。
  - 基座：8B 开源（如 Qwen3-8B / Llama-3.1-8B）+ GRPO，复用两篇 baseline 的设置以便公平对照。
  - 技能的"完成 oracle"复用 garak detectors / AgentDojo・InjecAgent 的程序化判定。

### 3.4 风险 & 开放问题

- 最大不确定性：
  - **技能如何形式化与监督**——子技能边界、完成条件是否能自动、稳健地生成？
  - 可验证 oracle 的**覆盖面**：能否覆盖足够多有意义的攻击，避免只剩玩具任务。
  - **与 DialTree 的区分度**必须在 intro 讲死：攻击域不同（可验证/agentic vs 有害内容）、奖励性质不同（可组合技能级可验证 vs 末端学习分类器）、解决问题不同（OOD 组合泛化 vs 单纯多轮）。
- 需先验证的关键假设（按先后）：
  - **H0（地基，最先验）：在 OOD/未见目标上，多轮攻击本身就比单轮提高 ASR。** 若不成立，技能库 / 可验证奖励都无从谈起，需重定方向。
  - H1：稀疏终局奖励是 OOD 停滞的主因（可用"密集技能奖励 vs 仅末端奖励"消融验证）。
  - H2：技能可组合 → 对未见目标的成功率随技能数上升（compositional 泛化曲线）。
  - H3：程序化 oracle 奖励比学习型 judge 更抗 reward-hacking 且更稳。


## 4. 相关工作 & 参考文献

- 直接 baseline / 动机：
  - [Training a General Purpose Automated Red Teaming Model (NVIDIA)](https://arxiv.org/abs/2604.23067) — **单轮**红队，Qwen3-8B+GRPO，目标专属 rubric 奖励；OOD 仅 29.35% = 本 idea 要攻的洼地，设为 baseline。
  - [Tree-based Dialogue Reinforced Policy Optimization (DialTree, ICLR 2026)](https://arxiv.org/abs/2510.02286) — **多轮**红队，Llama-3.1-8B+GRPO+树搜索，但奖励=HarmAug-Guard 学习分类器、仅末端、仅有害内容；设为 baseline 并明确区分。
- 可验证攻击域 / oracle 来源：
  - [AgentDojo (NeurIPS 2024 D&B)](https://openreview.net/forum?id=m1YYAQjO3w) — agentic 注入环境，成功=注入任务被完整执行（程序化）→ 子技能 oracle 与多轮环境来源。
  - [Tensor Trust (ICLR 2024)](https://openreview.net/forum?id=fsW7wJGLBd) — 密钥提取，精确匹配可验证。
  - [InjecAgent (ACL 2024 Findings)](https://aclanthology.org/2024.findings-acl.624/) — 间接提示注入，程序化判成功。
  - [garak](https://arxiv.org/abs/2406.11036) — 程序化 LLM 漏洞探针/detector（作者含 NVIDIA 论文作者 Derczynski），可直接当训练 oracle。
- 可验证奖励范式（方法学支撑）：
  - [Tulu 3](https://arxiv.org/abs/2411.15124) — RLVR：用 verification function 取代 reward model。
  - [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) — 两篇 baseline 都在用的 RL 算法，配可验证奖励。
- 近期 RL 红队景观（定位/差异化坐标）：
  - [PISmith / RL-Hammer](https://arxiv.org/abs/2603.13026) — GRPO 攻提示注入防御，针对奖励稀疏联合训练。
  - [CHASE](https://arxiv.org/abs/2606.05523) — 红蓝对抗 RL（如转防御闭环可参考）。
  - [A Systematic Investigation of RL-Jailbreaking](https://arxiv.org/abs/2605.07032) — 发现奖励函数设计+episode 长度是越狱成败主因（支持"奖励该可验证/密集"）。
- 多轮越狱先验（非学习/搜索式，用于区分）：
  - Tree of Attacks (TAP) / Crescendo — 多轮越狱但多为 prompt 工程/搜索，非训练出的可组合技能策略。（引用前再核实具体出处）
- 方法学旁参：HRL / option-skill discovery、process reward model（PRM，对照"学习型 vs 可验证"过程奖励）。


## 5. TODO & 下一步

- 接下来的小步骤（按优先级：先验地基前提，再搭机制）：
  1. **【最优先 · 验证地基 / H0】在 OOD（未见目标）上，多轮攻击是否比单轮提高 ASR？** 用最小代价验：沿用 NVIDIA 单轮设置作对照，**先不引入技能库 / 可验证奖励**（甚至可先用 prompting + 少量 rollout 的朴素多轮），只回答"多轮本身在 OOD 上有没有增益"。有增益 → 前提成立，继续；无增益 → idea 前提不立，重定方向。
  2. 选定 1–2 个**可验证攻击域**作主战场（建议 agentic 注入：AgentDojo / InjecAgent 现成环境 + 程序化 oracle），盘清可用的子技能与其完成条件。
  3. 设计**目标→可验证子技能序列**的自动分解方案（把 NVIDIA "生成终局 rubric" 改造为 "生成可验证子目标链"），先小规模验证可生成性，再跑"仅末端 vs 技能级密集奖励"消融骨架（验证 H1–H2）。
- 需要先回答的问题：
  - 子技能的粒度与"完成 oracle"能否自动、稳健生成？覆盖面够不够撑起一篇主结果？
  - 主战场到底押 agentic 注入域，还是更广的"通用可验证目标"？（影响环境/oracle 选型）
  - 难目标上探到哪一档（开源中端 vs 前沿强对齐）——决定成本与冲击力的平衡。
