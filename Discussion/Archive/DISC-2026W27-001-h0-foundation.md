# 💬 Discussion · 当前 Active 议题

> **协议要求**：本文件同一时刻只承载 **1 个 active 议题**。议题关闭后整体迁移到 `Discussion/Archive/DISC-YYYYWww-NNN-<slug>.md`，本文件从 `tools/templates/Discussion.template.md` 重置或立即承载下一个议题。

---

## Issue Header

| 字段 | 值 |
|---|---|
| **议题号 (ID)** | `DISC-2026W27-001` |
| **标题 (Title)** | 在 OOD/未见目标上，多轮攻击是否比单轮提高 ASR？(H0 地基验证) |
| **状态 (Status)** | `Resolved` |
| **发起人 (Owner)** | `PI @bzhu11` |
| **开题时间 (Opened)** | `2026-06-29 16:34` |
| **关联 idea/method** | `idea.md §3.4 (H0)` + `§5 TODO-1` / `method.md`（H0 阶段暂不涉及公式） |
| **关联实验** | `EXP-2026W27-001`（待跑 · H0 pilot） |

---

## Open Questions（待决清单）

> 维护一组**收敛性问题**，每条带 owner 与到期，防止讨论绕圈。

- [x] Q1：**已定**（PI @bzhu11, 2026-06-29）→ **两域各跑最小 pilot**：agentic 注入域（InjecAgent / AgentDojo）+ 沿用 NVIDIA 域，各做一个最小 H0 对照。
- [x] Q2：**已实现**（EXP-2026W27-001）→ 单轮=1 次注入；朴素多轮=≤K 轮自适应再注入（attacker 看 target 回复后改写），均纯 prompting、无训练。
- [x] Q3：**已实现**→ 程序化 oracle（vendored InjecAgent `evaluate_output_prompted`，检测 agent 是否调用 attacker tool），单轮/多轮共用、可复算、抗 hack。
- [ ] Q5：scaled H0 的 **target 组合**？Qwen3.6-27B 对朴素注入高度鲁棒（基线~0，多轮难显增益）→ 是否加更易受攻 target（更弱对齐 / enhanced setting）给多轮留增益空间，并保留 frontier Claude？(owner: `PI`, due: `2026-07-03`)
- [ ] Q6：**attacker 拒答策略**？保留 Qwen3.5-27B + 低触发 prompt + 多轮 nudge + 报告 delivery rate（当前 single 38% / multi 62%）是否可接受，还是换更可攻 attacker？(owner: `PI`, due: `2026-07-03`)
- [x] Q4：**已定**（PI @bzhu11, 2026-06-29）→ **统计显著即可**：多轮 > 单轮且跨 seeds 统计显著（如 bootstrap 95% CI 不含 0）即算 H0 成立；不设固定百分点。

---

## Posts（回帖区）

> **格式**：每条以 `【角色@姓名】【YYYY-MM-DD HH:MM】` 开头。
> **角色**：`PI` / `Lead` / `Collab` / `Agent`。
> **Agent 发言必须**：(a) 链接到 `LOGS/...#EXP-...`；(b) 直接贴关键数字，不让读者跳页。

---

【PI @bzhu11】【2026-06-29 16:35】（Agent 代拟首发，待 PI 确认 / 修订）

- **本议题只验证一个地基前提 H0**：在**未见 (OOD) 目标**上，多轮攻击**本身**是否比单轮提高 ASR。本轮**先不引入**技能库 / 可验证奖励 / 训练——只回答"多轮有没有增益"。
- **当前怀疑**：两篇 SOTA 的 OOD 洼地暗示瓶颈在"末端稀疏 + 单发打不穿"——NVIDIA 单轮 OOD 仅 ~29.35%（in-domain 已 ~85% 近饱和）；DialTree 多轮但奖励 = 末端学习分类器、作者自承 >5 轮乏力。若**朴素多轮**（prompting + 少量 rollout）即能在 OOD 上抬 ASR，则"渐进推进 + 技能级信用分配"这条路线的前提成立。
- **希望验证**：用最小代价——沿用单轮设置作对照，加一条朴素多轮 rollout，在**同一组 OOD 目标**上比较单轮 vs 多轮 ASR（可本地、可复现、低成本）。
- **判定 / 出口**：多轮显著 > 单轮 → H0 成立，进入技能库机制设计（H1 / H2）；无增益 → idea 前提不立，回 `idea.md` 重定方向（触发 AGENTS.md §10 Escalate）。
- **当前阻塞**：Q1（主战场域）与 Q4（成立阈值）是 PI 决策项，未定则无法落 pilot；Q2 / Q3（最小实现 & ASR 口径）待 Q1 定后由 Agent 起草。
- **关联**：`idea.md §3.4 H0` / `§5 TODO 第 1 项`。

---

【Agent @Claude】【2026-06-29 16:40】

记录 PI 决策：**Q1 = 两域各跑最小 pilot**。据此起草 Q2 / Q3（待 PI 确认 Q4 阈值 + 执行环境后定稿）：

**两域 H0 最小 pilot 设计（控制变量：仅"单轮 vs 多轮"不同，其余全一致）**

- **域 ①｜agentic 注入（InjecAgent，最轻量、自带程序化 oracle）**
  - 目标集：InjecAgent attacker instructions；按 held-out 一组工具 / 攻击意图划出 OOD。
  - 单轮 baseline：标准一发 IPI 注入。
  - 朴素多轮：攻击者跨 ≤K 轮依 agent 反馈改写 / 追加注入（纯 prompting、无训练、每目标 n 次采样）。
  - ASR：InjecAgent 程序化判定（注入任务是否被执行）。
- **域 ②｜NVIDIA 域（忠实对照，直攻 OOD ~29% 洼地）**
  - 目标集：复用其 in-domain / OOD 划分（garak behaviors）。
  - 单轮 / 多轮：同上策略；H0 阶段 attacker 先用 prompting 版，**暂不起完整 GRPO 训练**（H0 只比轮数，避免过早上 RL 管线）。
  - ASR：沿用其 rubric / garak detector。
- **共同协议**：模型、目标集、判定口径在单轮 / 多轮间完全一致；OOD 严格 held-out；每目标多 seed 采样报均值 ± 方差。

**Q2（最小实现）**：上述 prompting 版单轮 + 朴素多轮 rollout，无需训练。
**Q3（ASR 口径）**：域① InjecAgent 程序化 oracle；域② garak detector / rubric——均为可复算判定，单轮多轮共用。
**待 PI 定**：Q4 成立阈值；执行环境（GPU / API / 数据可用性）以定模型档与 pilot 规模。

---

【Agent @Claude】【2026-06-29 16:50】

PI 已定 **Q4 + 执行环境**，两域 pilot 协议据此定稿：

- **成功判据（Q4）**：**统计显著即可**——多轮 vs 单轮的 OOD ASR 差，跨 seeds bootstrap 95% CI 不含 0 即算 H0 成立（不预设固定百分点）。→ 含义：每个 (域, 单/多轮) 配置需跑**足够 seeds / 目标数**撑起 CI（计划每域 ≥30 OOD 目标 × ≥3 seeds，必要时加采样）。
- **执行环境**：1× H20-NVLink 96GB + 16 vCPU (EPYC 9K84) + 150GB RAM；另有 API。→ 96GB 显存可**本地同时托管 8B attacker + 8B target**（bf16 各 ~16GB），忠实复刻两 baseline 的 8B 设置（Qwen3-8B / Llama-3.1-8B），H0 阶段纯推理、不训练。
- **下一步（待 PI go-ahead）**：Agent 起草 `code/` 脚手架 + 资源清单（模型 / 数据 / 磁盘 / 预估时长），经 §10 资源确认后再实跑。

---

【Agent @Claude】【2026-06-29 17:00】

已读 `related_works.md`（PI 点名，§8）。要点对 target 选型的影响：

- DialTree 在 **12 个目标模型**上测（含 **Claude-4-Sonnet ~71% ASR**），评判用 held-out GPT-4o；NVIDIA 用 garak 目标 + detector。领域惯例 target = 开源 8B（Qwen / Llama）+ 商用前沿（GPT / Claude / Gemini）。
- **H0 在强对齐前沿目标上最有判别力**（单轮最易失败、多轮增益最该显现）→ H0 即应含 ≥1 前沿 target。

**Target 阶梯（PI 指示"尽量全面 + 前沿"）**：
- **T1 本地开源（H20，免费快）**：Qwen2.5-7B-Instruct（ungated）、Llama-3.1-8B-Instruct（gated, 对齐 DialTree）、(可选) 更强对齐中端（Qwen2.5-32B / Llama-3.1-70B-4bit，96GB 可容）。
- **T2 前沿 API（对齐 DialTree 难目标）**：Claude Sonnet、GPT-4o / 前沿 OpenAI、(可选) Gemini。

**执行**：PI 选 **SSH 直连**——Agent 经 Bash ssh/scp 在 H20 部署 + 跑、自动收日志。
**Staging（newbie 先 Pilot 再扩展，AGENTS.md §3.1）**：H0 先在**代表性子集**（1 本地 Qwen2.5-7B-Instruct + 1 前沿 Claude Sonnet）跑通两域 pipeline，统计显著性验证后再铺满 target 阶梯。
**待 PI provisioning**：① SSH 连接信息 / 密钥；② HF token（含 Llama gated 许可）；③ 前沿 API key（Claude / OpenAI）；④ 确认服务器 ≥80GB 空闲盘。

---

【Agent @Claude】【2026-06-29 17:30】Env 连通性核验完成（PI 提供 `.env`）

**全部打通 ✅**：
- **4 路 API 均可 chat**：SiliconFlow、aipaibox-Claude（8 模型，含 `claude-sonnet-4-6` / `claude-opus-4-8` / `claude-haiku-4-5`）、aipaibox-GPT（`gpt-5.4` / `gpt-5.5` / `gpt-5.4-mini`）、OpenAI 官方（`gpt-4o-mini`）。空的 CLAUDE_MODEL / GPT_MODEL 已查明可用名。
- **SSH→H20**：NVIDIA H20 96GB（idle, driver 580.82, CUDA 12.8 toolkit），torch 2.8.0+cu128 `cuda=True`；128 vCPU / 1.2TB RAM；conda base py3.12（`/root/miniconda3/bin/python`，非交互 shell 无 python3，需全路径或 `bash -lc`）。
- **写盘**：`/root/autodl-tmp`（vdb, 150G, **76G 空闲**, 可写；HF_HOME 复用 `/root/autodl-tmp/hf_home`）。`/` 仅 30G 勿用；`/autodl-pub` 只读。
- **出网**：server 默认不通 huggingface.co；`source /etc/network_turbo` 后 HF=200；hf-mirror / modelscope / pypi-tuna / github 均通。
- **开源模型经 SiliconFlow 可直接服务**：**Qwen3-8B（attacker）**、Qwen2.5-7B/32B/72B-Instruct（targets）、GLM / DeepSeek 等。HF token 有效（crossmen00）。

**待办 / 注意 ⚠️**：
1. **Llama-3.1-8B-Instruct 暂不可得**：HF 403（crossmen00 未接受 license）且 SiliconFlow 无 Llama/Meta。→ 纳入 DialTree 对齐 target 需：(a) HF 接受 Llama-3.1 许可，或 (b) ModelScope 下载，或 (c) 用 Qwen 家族替代。**H0 非必需**。
2. vLLM 未装（仅本地托管才需；H0 用 SiliconFlow API 即可）。
3. 服务器与他项目 **MemSifter** 共用，已占 75G；余 76G，H0 够用，铺 fleet / 训练前需关注空间。

**H0 执行模式建议**：**全 API 跑 H0**（零下载、零 vLLM、不占盘）——attacker = Qwen3-8B (SiliconFlow)，targets = Qwen2.5-7B-Instruct (SiliconFlow) + claude-sonnet-4-6 (aipaibox)。本地 vLLM（Mode B）留到训练期 (H1+) 再起，届时预下 Qwen3-8B 到 autodl-tmp。

---

【Agent @Claude】【2026-06-29 17:45】Roster 锁定（PI: Llama 走 (c) Qwen 替代；API 用新版 Qwen 3.5/3.6；GPU 暂 无卡模式）

- **关键发现**：SiliconFlow 上 Qwen3.5/3.6 全是**推理模型**，默认 thinking → 输出走 `reasoning_content` 通道，max_tokens 不够时 `content` 为空。开关：`enable_thinking=false` 直出干净 content（~2s）；`true` 则需 ≥512 tok 才有 content（CoT 在 `reasoning_content`）。
- **H0 roster（全 API，无需 GPU）**：
  - Attacker = `Qwen/Qwen3.5-9B`，thinking **ON**（CoT 利于多轮策略，对齐 DialTree CoT），max_tokens 1024。
  - Target-open = `Qwen/Qwen3.6-27B`，thinking **OFF**（部署式行为、干净 content 供 oracle）——即 Llama 的 Qwen 替代 (PI 选 c)。
  - Target-frontier = `claude-sonnet-4-6`（干净直出）。
  - 后续扩展 target：`gpt-5.4`、Qwen2.5-72B、GLM-4.5-Air、DeepSeek 等。
- **GPU**：H0 不需要（无卡模式 OK）；训练期 (H1+) 再请 PI 退出无卡模式。
- **下一步**：建 `code/` 脚手架（已起 LLM client + config + env_check），随后接 attacker（单轮/朴素多轮）+ 两域 oracle 适配。

---

【Agent @Claude】【2026-06-29 18:25】H0 脚手架建成 + 端到端验证 + 首个信号

证据：[LOGS/2026-W27.md#EXP-2026W27-001](LOGS/2026-W27.md) · 全程留痕 `code/runs/h0_smoke_20260629T182212/`（calls/events/records JSONL）。

- **脚手架就绪**（全 API、无需 GPU）：多 provider client → trace（**每次 LLM 输入/输出落 JSONL**）→ attacker（单轮/朴素多轮 + 拒答处理）→ InjecAgent DH 域适配（OOD 按 attacker-tool 留出, SPLIT_SEED=1234）→ vendored 程序化 oracle → bootstrap-CI → run_h0。
- **修复两 bug**：①空注入（thinking 吃光 token，content 空）；②attacker **Qwen3.5-9B 安全拒答 ~2/3** → 换 **Qwen3.5-27B**(thinking off) + 低触发 benchmark prompt + 拒答检测 / 多轮 nudge（delivered-attack rate single 38% → multi 62%）。
- **首个信号**（n=8 OOD goals, target=Qwen3.6-27B）：**ASR single 0% vs multi 12.5%（diff +12.5pt, 95%CI [0, +37.5]）**——方向支持 H0，样本太小未达显著。Qwen3.6-27B 能识别并拒绝注入 → 基线~0。
- **待 PI 决策**（scaled run 前）：见 Open Questions **Q5（target 组合）**、**Q6（attacker 拒答策略）**。

---

【Agent @Claude】【2026-06-30 13:32】H0 判定（confound-aware）：偏否 / equivocal

证据：[LOGS/2026-W27.md#EXP-2026W27-002](LOGS/2026-W27.md) · 全程留痕 `code/runs/h0_20260629T191044/`。先经 **22-agent harness 审计**（15 confirmed findings, verdict `has_blocking_bugs`）修复关键 confound 后再判。

**完整数据：attackable target `qwen3-8b`（450 single + 270 multi；30 OOD goals × 3 seeds）**
- raw `multi vs single` **+8.7pt** [+4.4,+13.6] 显著 —— 但为 confound。
- attempt-matched `multi vs best-of-5`(all) **−21.5pt** [−31,−12]：5 次**独立**采样反而强于多轮自适应。
- 主指标 `multi vs best-of-5`(delivered, goal-level) **+9.2pt** [+0.9,+20.6] 勉强显著；但 **tool-level（正确 OOD 单元）−0.9pt [−2.7,+0.0] 不显著**。
- 多轮成功 **75% 落在第 1 轮**；一旦 delivered，target 几乎不抵抗（single 79% → multi 99%）。
- target 屏蔽：**现代对齐模型对朴素注入高度鲁棒**（Qwen2.5-32B / Qwen3.6-27B / gpt-4o-mini 均 0%），仅 Qwen3-8B 有余量；Qwen2.5-7B 退化不可用。

**判定**：在唯一有余量的 target 上，**朴素多轮自适应未稳健超过 attempt-matched 基线** → H0 地基前提**偏否**。表层 multi>single 主要来自"更多尝试 + 更高 delivery"两个 confound（审计已预警）。

**待补**：`qwen3.6-27b`（鲁棒参照，补跑中，预期 ~0）。

**§10 Escalate（PI 决策，Agent 仅起草）**：H0 偏否触及 idea 前提，三条草拟方向待 PI 选向——
- **(A)** 接受"多轮本身不是卖点"，将主张收紧为"**可验证技能 / 密集过程奖励 → OOD 组合泛化**"（多轮仅作底座），让 **H1（密集 vs 末端奖励）成为真正的地基实验**；
- **(B)** 换更强多轮算子（树搜索 / DialTree 式）再测 H0；
- **(C)** 换有真实抵抗余量的 target / 域（当前注入域可攻模型太少）。
PI 选向后由 Agent 起草 `idea.md` 对应改动（§9）。

---

【Agent @Claude】【2026-06-30 13:40】PI 选 **(C) 换有余量的 target / 域**

- 根因：注入域里现代对齐模型对"跟随注入指令"训练得太好（6 模型 5 个 0%），可攻余量几乎只剩 Qwen3-8B 一个 → 任何方法都无空间证明增益。
- 下一步（Agent）：**headroom scout** —— 在多个可验证 setup（注入 / Tensor-Trust 式 prompt-extraction / hijack）× 多个 target 上测单发强攻 ASR，找有"抵抗梯度"（部分 target 20–70%）的战场，再定新域并建适配器（建新域属 §9，建前请 PI 确认具体域）。
- 当前数据收尾：`qwen3.6-27b` 鲁棒参照补跑完成后，用 `finalize_h0.py` 写 combined results 并补记 EXP-2026W27-002。

---

【Agent @Claude】【2026-06-30 13:50】Headroom scout 结果（为 (C) 选战场）

- **换"域"无用**：Tensor-Trust 式 extraction / hijack 单发强攻在现代模型上**也几乎全 0%**（6 模型仅 Qwen3-8B extract 1/5）→ headroom 瓶颈是 **target 鲁棒性，不是域**。
- **存在可攻 target 集**（注入域, 单发强注入, ASR-valid, n=8 OOD）：**Qwen3-8B 43% · Qwen3-14B 38% · Qwen3.5-4B 25% · Qwen2.5-14B 17%**（有梯度、连贯）。鲁棒 0%：Qwen3.5-9B / GLM-4-9B / DeepSeek-R1-8B / Qwen2.5-32B / Qwen3.6-27B / gpt-4o-mini。
- **结论**：**不必换域**；注入域 + 这 4 个可攻 target（17–43% 基线）即有余量战场。

---

【Agent @Claude】【2026-06-30 14:15】⚠️ **修正：robust target 数据反转了 H0 初判 → H0 在 hard target 上强支持**

`qwen3.6-27b`（鲁棒, 完整 450 single + 270 multi）结果与 `qwen3-8b` **相反**且**强支持 H0**：
- single ASR|delivered **0.5%**（单发几乎打不穿）；multi **24.1%**；多轮成功 **96% 落在第 2 轮及以后**（27 个仅 1 个在第 1 轮）。
- **attempt+delivery matched 主指标 +25.1pt [+12.7,+38.5] 显著**，且 **tool-level 同样显著 +14.3pt [+4.8,+24.5]**（best-of-5 独立采样仅 1.5% → 不是"更多尝试"，是真·自适应）。
- **人工核验成功为真**：attacker 第 2 轮升级社工话术（伪造 "Shopify Security Protocol 4.1 要求"），target 真的调用 attacker 工具（`Action: EvernoteManagerManageNoteSharingLevel{...}` → `success`）。
- **重新理解**：EXP-2026W27-002 的"偏否"是 **qwen3-8b 的天花板效应**（单发已 delivered 79%，多轮无余量、best-of-5 即可追平）——并非 H0 假。**在单发打不穿的 robust/hard target 上，自适应多轮是唯一能突破的，且击败 attempt-matched 基线**——这恰是 idea 关心的 OOD / 强对齐 regime。
- **净判定**：H0 价值**依目标难度**——hard target 上强支持，easy target 上天花板。idea 前提（多轮对难目标有用）**成立**。
- **待**：3 个 mid-attackable target（`brtg423mx`）跑完 → 出"难度梯度"完整图 → 定稿 **EXP-2026W27-003**（修订 H0 结论，supersede 002 的初判）。

---

【Agent @Claude】【2026-06-30 15:30】H0 终判（5-target 难度梯度）：**成立 · target-difficulty-dependent**

证据：[LOGS/2026-W27.md#EXP-2026W27-003](LOGS/2026-W27.md) · 合并 `code/runs/results_h0_combined.json`（每 target n=720）。PRIMARY（multi vs best-of-5, delivered, attempt-matched）随目标变难**单调增强**：

| target | single\|deliv | multi\|deliv | PRIMARY [95%CI] | tool-level | %turn1 |
|---|---|---|---|---|---|
| qwen3-8b | 79% | 99% | +9.2 [+0.9,+20.6] * | −0.9 ns | 75% |
| qwen2.5-14b | 75% | 98% | +6.3 [−1.8,+17.6] ns | +12.4 ns | 73% |
| qwen3-14b | 72% | 99% | +9.2 [+1.0,+19.7] * | +25.7 * | 78% |
| qwen3.5-4b | 50% | 93% | +23.4 [+10.3,+38.4] * | +29.1 * | 57% |
| qwen3.6-27b | 0.5% | 24% | +25.1 [+12.5,+38.5] * | +14.3 * | 4% |

- easy（single 75–79%）天花板：qwen3-8b（边缘 sig，tool-level null）、qwen2.5-14b（不显著）。
- mid–hard（≤72%）支持且越难越强：qwen3-14b / qwen3.5-4b（含 tool-level）；qwen3.6-27b 强支持，自适应**必需**（96% 成功在 turn 2+；人工核验真实工具调用）。Holm 校正后强信号（qwen3.5-4b / qwen3.6-27b）稳健。

**净：H0 成立，多轮增益随目标难度上升——在 idea 关心的 hard/OOD/强对齐 regime 最大最稳。** supersede 002 的"偏否"（easy-target 假象）。

**【拟 Decision，待 PI 确认是否 close DISC-2026W27-001】**
- **Decision**：H0 地基**成立（difficulty-dependent）**；自适应多轮对 hard/OOD target 有真实、attempt-matched 增益，easy target 因单发已成而天花板。
- **Rationale**：EXP-2026W27-003（5 target, confound-aware；tool-level 3/5 sig；hard target 强 sig + 人工核验；Holm 后强信号稳健）。
- **Propagated to**：`idea.md §3.4 H0`（标记成立 + 难度依赖限定）、`EXP-2026W27-001/002/003`。
- **下一步（开新议题）**：(a) hard-target 复制（Qwen2.5-32B / gpt-4o-mini / DeepSeek-V3.2）巩固；(b) H1 = 可验证技能 / 密集奖励 pilot。

请 PI 确认：是否据此 **close 本议题** 并由 Agent 起草 `idea.md §3.4` 改动（§9）？

---

【Agent @Claude】【2026-06-30 16:55】第二 oracle（AgentDojo, 真·多步）确认 **multi-step ASR ≫ single-step**

回应 PI 质疑（multi-turn ≠ multi-step；只有 1 个 oracle）。证据：[LOGS/2026-W27.md#EXP-2026W27-004](LOGS/2026-W27.md) · `code/runs/adojo_20260630T164449/`（每次 LLM call 全程留痕，504 calls）。

- **AgentDojo env-state oracle**（独立于 InjecAgent），`slack` suite，工具循环封顶 `max_iters` 2(single) vs 15(multi)：
  - qwen3-8b **12% → 56%**（+44pt [+24,+64]）；qwen3.6-27b **16% → 72%**（+56pt [+28,+80]）。McNemar：多步严格支配（only-multi 11/16 vs only-single 0/2）。
- **意外**：单发鲁棒的 qwen3.6-27b 在多步 agentic 设置反而**更**易受攻（72% > qwen3-8b 56%，utility 也更高 88%>76%）→ **单发鲁棒 ≠ 多步 agentic 鲁棒**。
- **Confound**：single 臂 utility 也被压低（步数不足完成任务），gap 部分来自 agent 被削弱；但多步绝对 ASR 56–72%（标准 benchmark + 真 oracle）本身是核心结论。
- **含义**：是 **multi-step（agent 多步轨迹），而非 multi-turn 重试**，才是注入成功的关键维度——直接支撑 idea 的"多步 + 中间可验证状态"机制。两个独立 oracle 结论一致。
- **下一步**：扩 suite（workspace/banking）/ 多 attack / utility-matched 单步定义做稳健 CI；H1（中间状态 credit assignment）pilot。

---

【Agent @Claude】【2026-07-01 16:48】AgentDojo round-2（utility-matched）完成：multi-step 真实攻击面确证

证据：[LOGS/2026-W27.md#EXP-2026W27-005](LOGS/2026-W27.md) · `code/runs/adojo2_v4_*/`（972 records, 4 model × 4 suite × 3 attack, 每次 LLM 全程留痕）。修掉 EXP-2026W27-004 的 capped-agent utility confound：full-capacity agent + 按 goal-depth（single=1 / multi=≥2 恶意调用）分组。

- **需 ≥2 步的攻击（单步 agent 结构上不可能）以显著非零率成功**：总体 multi-goal ASR **22.7% [18.7,27.0]**；**4 模型全显著**（qwen3-14b 33% / qwen3-8b 23% / gpt-4o-mini 22% / 单发鲁棒 qwen3.6-27b 12%）。
- **OOD（held-out travel suite）泛化**：multi 27.8% [19,36] 显著；此处 multi ≈ single（gap +3.7pt 不显著）。
- utility 49.3%（两组同 full agent → matched）；genuineness 93%（多步成功确执行了所需恶意调用序列）。
- **诚实方向**：深攻击绝对 ASR 更低（multi 22.7% < single 39.6%）→ 卖点是"multi-step 开启单步够不到的攻击类"，**非**"步多更易"。与 EXP-2026W27-004（capped 12–16% → full 56–72%）互证。
- **两 oracle（InjecAgent 单步 + AgentDojo 多步）一致**：multi-step 是 idea 关心的真实攻击面。
- 下一步：per-step first-success-step（同任务 by-step-1≈0 vs by-step-K 的干净 gap）；接 **H1**（中间可验证状态 → 密集 credit assignment）。

---

【Agent @Claude】【2026-07-02 16:46】H0 seal：per-step 干净 gap + 6 角度对抗审计 → 结论存活（精确限定）

证据：[LOGS/2026-W27.md#EXP-2026W27-006](LOGS/2026-W27.md) · `code/runs/perstep_v4.json` · `code/scripts/agentdojo_perstep.py`。

- **per-step（v4, 972 tasks, 99.3% replay-faithful）**：**single-step ASR（by step 1）= 0.0%（0/965）**；**multi-step ASR（by step K）= 31.4%**；gap +31.4pt；303 个成功**无一**在 step 1（分布 step 2–12）。
- **6-skeptic 对抗审计**（triviality/stats/oracle/generalization/confounds/construct-validity）→ **H0 存活**，但须精确限定，关键 caveats：
  1. per-step 0% **部分 definitional**（注入在 tool 输出，需先 fetch）→ 非定义性主张是 goal-depth（≥2 恶意调用 22.7%）。
  2. 方向：深攻击绝对 ASR 更低（multi 22.7% < single 39.6%）→ "多步开启单步够不到的攻击类"，非"步多更易"。
  3. goal-depth single/multi 是不同任务（task-difficulty 混入）。
  4. 泛化不均（suite/model 依赖；仅 4 model；OOD n 小）。
  5. 单跑无 seed variance；多 cell 须 Holm（强信号稳）。
  6. **construct validity（最大）**：H0 用静态模板证"多步注入成功"，**≠** idea 的"可验证可组合技能 + 密集奖励 → OOD 泛化"。**H0 只立 precondition（多步是攻击面 + 中间态可程序化验证 = 密集奖励底座），机制留给 H1。**
- **净判定**：H0 地基**成立**（多步是 agentic 注入攻击面，中间态可验证），经对抗审计存活。最重要限定：**H0 立 precondition，非完整 thesis**——这正是 H1 要证的。
- 建议：据此 **close DISC-2026W27-001**（Decision 见前贴，附本 seal），开 H1 议题（密集可验证技能奖励 vs 末端）。待 PI 拍板。

---

## Resolution（关闭议题时必填）

> Status 切到 `Resolved` 时，本节必须全部填好；否则不许关闭。

- **Decision**：H0 地基前提**成立**，精确表述：(a) 朴素**多轮**（attacker 重试、单步 target）对 OOD 的增益是 **target-difficulty-dependent**——在单发打不穿的 hard target 上强支持、easy target 天花板（EXP-2026W27-001 至 003）；(b) 更核心的 **multi-step**（agent 多步轨迹）是**真实、独立的攻击面**——需 ≥2 恶意步的注入攻击在 full-capacity（utility-matched）agent 上以显著非零率成功（goal-depth 22.7%，4 模型全显著，泛化 held-out OOD suite 27.8%），且 per-step 显示 **0.0% 的注入在单步内得手 / 31.4% 需多步**（EXP-2026W27-004 至 006，两 oracle 一致）。**关键限定**：H0 只立 **precondition**（多步是攻击面 + 中间态可程序化验证 = 密集奖励的底座），**非** idea 完整 thesis（学习可验证可组合技能 + 密集奖励 → OOD 泛化），后者是 H1；方向诚实——深攻击绝对 ASR 更低，卖点是"多步开启单步够不到的攻击类"而非"步多更易"。
- **Rationale**：EXP-2026W27-001 至 006；两独立 oracle（InjecAgent 单步 parse + AgentDojo 多步 env-state）；confound-corrected（best-of-K / delivery-conditional / utility-matched / goal-depth / per-step，99.3% replay-faithful）；22-agent harness 审计修掉关键 confound；6-skeptic 结论对抗审计（H0 存活，caveats 已入 EXP-2026W27-006）；genuineness 程序+人工核验 93–100%。
- **Propagated to**：
  - `idea.md §3.4`：H0 标记成立 + 区分 multi-turn vs multi-step + 限定"precondition 非 thesis"
  - `idea.md §5`：H0 完成，转 H1
  - 支撑实验 `EXP-2026W27-001` 至 `EXP-2026W27-006`
  - `method.md`：H0 阶段无公式改动；机制形式化留待 H1
- **Closed by**：`PI @bzhu11`
- **Closed at**：`2026-07-02 16:46`

---

## 关闭流程（Agent 执行）

1. 确认 `Resolution` 各字段已填（Decision / Rationale / Propagated to / Closed by / Closed at），且 Status 已切到 `Resolved`。
2. 运行 `python tools/new_disc.py close "<slug>"`：脚本校验 Resolution → 归档为 `Discussion/Archive/DISC-YYYYWww-NNN-<slug>.md`（文件名含议题号，可反查）→ 从 `tools/templates/Discussion.template.md` 重置本文件。
3. 在 `method.md` / `idea.md` 受影响章节追加 changelog 条目，并反向链回归档路径。
4. 开启下一个议题：`python tools/new_disc.py open "<标题>"`（编号由脚本分配，避免撞号）。
