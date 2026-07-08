# Week 2 实验报告（零基础可读版 + 图表版）

## H1 Phase A：训练「攻击写手」并对比两种打分方式

> **时间**：2026-07-02 ～ 2026-07-07  
> **议题**：[`DISC-2026W27-002`](../Discussion.md)  
> **原始日志**：[`2026-W27.md`](2026-W27.md) · [`2026-W28.md`](2026-W28.md) · Week 1 见 [`report_week1.md`](report_week1.md)

---

## 图表导航（建议看图顺序）

| # | 图 | 在哪一节 | 帮你看懂什么 |
|---|-----|----------|--------------|
| 1 | [总流程图](#fig-1) | §一 | Attacker / Victim / 注入 / 判分 怎么串起来 |
| 2 | [时序图：一次攻击](#fig-2) | §一 | 从读题到成功/失败逐步发生什么 |
| 3 | [研究路线图 H0→H1](#fig-3) | §二 | Week 1 和 Week 2 各回答什么 |
| 4 | [Week 2 时间线](#fig-4) | §二 | 哪天在搭 lab、哪天在 GPU 训练 |
| 5 | [Sparse vs Dense 进度条](#fig-5) | §三 | Φ 和 0/1 打分随「完成几步」怎么变 |
| 6 | [GRPO 训练循环](#fig-6) | §三 | 5 份 payload 怎么比、怎么更新模型 |
| 7 | [Phase A 三格 + Q10 读法](#fig-7) | §四 | 三门课各自测什么、期望 vs 实际 |
| 8 | [372 题怎么切 train/OOD](#fig-8) | §五 | 题库从哪来、考试集占多少 |
| 9 | [基线 ASR 柱状图](#fig-9) | §五 | 未训练时单步 vs 多步、写 1 次 vs 5 次 |
| 10 | [H20 双角色显存](#fig-10) | §六 | 一张卡如何同时跑 Victim + Attacker |
| 11 | [训练 reward 上升曲线](#fig-11) | §六 | 三格训练过程中分数涨没涨 |
| 12 | [OOD 考试 ASR 对比](#fig-12) | §六 | **核心结果**：训前训后、dense vs sparse |
| 13 | [GRPO 提升幅度 Δ](#fig-13) | §六 | Phase A 门是否通过 |
| 14 | [H1 预览：dense ≈ sparse](#fig-14) | §六 | 为什么 H1 暂未支持 |
| 15 | [Phase B 决策树](#fig-15) | §十一 | 下一步 PI 可选路径 |
| 16 | [论文大故事总览](#fig-16) | §十三 | idea → H0 → H1 → 未来 |

---

## 写在最前面

如果你**没有** RL、AgentDojo、红队背景——本报告用**比喻 + 图表**从零讲起；文字表格保留，但**建议先看上面导航里的图**。

---

## 一、用图理解整个项目在干什么

<a id="fig-1"></a>

### 图 1 · 总流程（Attacker 写纸条 → Victim 中招 → 自动判分）

```mermaid
flowchart TB
    subgraph input ["输入"]
        G["攻击目标 goal<br/>例：分 3 笔转 $6 给黑客"]
    end

    subgraph attacker_box ["Attacker（被训练 · Qwen3-8B）"]
        A["写 payload 小纸条"]
    end

    subgraph env ["AgentDojo 沙盒（假银行）"]
        INJ["注入：纸条藏进账单/邮件"]
        UT["用户正常任务<br/>「帮我查账单」"]
        VICT["Victim（冻结 · Qwen3.6-27B）<br/>读数据 → 调工具"]
        INJ --> VICT
        UT --> VICT
    end

    subgraph judge ["Oracle 自动判卷"]
        SP["Sparse：全成才 1 分"]
        DN["Dense：Φ = 完成几步 / 总几步"]
    end

    G --> A
    A --> INJ
    VICT --> SP
    VICT --> DN

    SP --> R1{"ASR<br/>成功?"}
    DN --> R2{"Φ 进度?"}

    style attacker_box fill:#ffe6cc
    style env fill:#e6f3ff
    style judge fill:#e6ffe6
```

**Attacker** = 我们要教的「写小纸条的人」。**Victim** = 不动的「考场 AI」。**Oracle** = 不靠人感觉、靠程序查账本来判成功。

---

<a id="fig-2"></a>

### 图 2 · 时序图：一道 depth=3 的题从头到尾

```mermaid
sequenceDiagram
    autonumber
    participant T as 攻击目标
    participant A as Attacker
    participant E as 沙盒环境
    participant V as Victim AI
    participant O as Oracle

    T->>A: 「分 3 笔共转 $6」
    A->>E: 注入 payload 到账单文本
    V->>E: 用户任务：查本月账单
    E->>V: 返回被污染的账单内容
    loop Victim 最多 15 步工具调用
        V->>E: send_money / read_file / ...
        E->>V: 工具结果
    end
    E->>O: 轨迹 + 账户最终状态
    Note over O: Sparse=0 或 1<br/>Dense=Φ∈[0,1]
    O-->>A: 一个分数（GRPO 用来更新）
```

**要点**：Attacker **只写一次**纸条；「多步」是 **Victim 在环境里连续调工具**，不是 Attacker 和 Victim 聊天多轮。

---

## 二、Week 1 vs Week 2：两阶段问题

<a id="fig-3"></a>

### 图 3 · 研究路线图（H0 前提 → H1 机制）

```mermaid
flowchart LR
    subgraph W1 ["Week 1 · H0 ✅"]
        Q0["多步链式攻击<br/>真存在吗?"]
        A0["是 · 单步内 0%<br/>多步 31%"]
    end

    subgraph W2 ["Week 2 · H1 进行中"]
        Q1["过程分 Φ<br/>比期末分 0/1<br/>更 help OOD?"]
        A1a["Phase A ✅<br/>GRPO 能训"]
        A1b["预览 ⚠️<br/>dense≈sparse"]
    end

    subgraph W3 ["Week 3+ · 待定"]
        PB["Phase B<br/>深题 / 难 OOD"]
    end

    Q0 --> A0
    A0 --> Q1
    Q1 --> A1a
    Q1 --> A1b
    A1b --> PB

    style W1 fill:#d4edda
    style A1a fill:#d4edda
    style A1b fill:#fff3cd
```

| 阶段 | 人话 | 比喻 |
|------|------|------|
| **H0** | 多步攻击有没有用？ | 先证明「偷保险柜要过三道门」 |
| **H1** | 过程分是否更会上新题？ | 对比「只给期末分」vs「按章节给分」 |

---

<a id="fig-4"></a>

### 图 4 · Week 2 时间线

```mermaid
flowchart LR
    subgraph w27 ["7/2 - 7/3 基建 无GPU"]
        E7["EXP-007 造题 PoC"]
        E8["EXP-008 reward"]
        E9["EXP-009 基线"]
        E7 --> E8 --> E9
    end

    M1["7/3 PI 批 GPU"]

    subgraph w28 ["7/6 - 7/7 训练"]
        PA["Phase A 三格 GRPO 约14h"]
        EV["EXP-028 OOD eval"]
        PA --> EV
    end

    PB["7/8+ Phase B 待定"]

    E9 --> M1 --> PA
    EV --> PB

    style E7 fill:#d4edda
    style E8 fill:#d4edda
    style E9 fill:#d4edda
    style PA fill:#d4edda
    style EV fill:#d4edda
    style PB fill:#fff3cd
    style M1 fill:#e6f3ff
```

**时间轴（文字版）**：

```
07-02 ── EXP-007 造题 PoC
07-02 ── EXP-008 reward 模块
07-03 ── EXP-009 基线 + harness  ──►  Ready for GPU
07-03 ── PI 批准 H20 (<20h)
07-06 ── Phase A 三格 GRPO 训练 (~14h)
07-07 ── EXP-2026W28-001 OOD 评估
07-08+ ─ Phase B 深目标（待定）
```

---

## 三、核心概念（配图表）

### 3.1 AgentDojo = 安全版飞行模拟器

```mermaid
flowchart TB
    AD["AgentDojo 沙盒"]

    AD --> B["banking 银行"]
    AD --> S["slack 办公"]
    AD --> T["travel 旅行"]
    AD --> O["Oracle 自动判分"]

    B --> B1["读账单 / 转账"]
    B --> B2["Week 2 主场景"]

    S --> S1["发消息等"]

    T --> T1["订票等"]
    T --> T2["Phase B 难 OOD 计划"]

    O --> O1["程序查账 非人类感觉"]

    style B fill:#e6f3ff
    style B2 fill:#cce5ff
    style O fill:#e6ffe6
```

### 3.2 depth：一步 vs 多步（三道门比喻）

```mermaid
flowchart LR
    subgraph d1 ["depth = 1"]
        A1["骗 1 次转账"] --> OK1["攻击完成"]
    end

    subgraph d3 ["depth = 3"]
        B1["第 1 笔转"] --> B2["第 2 笔转"] --> B3["第 3 笔转"] --> OK3["攻击完成"]
    end
```

---

<a id="fig-5"></a>

### 图 5 · Sparse vs Dense：同一道题上分数怎么变（depth=3 例题）

**Victim 每多完成一步恶意操作，两种打分的变化：**

| 完成步数 | Sparse（期末 0/1） | Dense Φ（过程分） |
|----------|-------------------|-------------------|
| 0 步 | 0 | 0 |
| 1 步 | 0 | 0.33 |
| 2 步 | 0 | 0.67 |
| 3 步全成 | 1 | 1.0 |

```
步数:     0      1      2      3(全成)
Sparse:   ░░░░░  ░░░░░  ░░░░░  █████  (只在终局给分)
Dense Φ:  ░░░░░  ██░░░  ████░  █████  (每步累积进度)
          0      0.33   0.67   1.0
```

```mermaid
flowchart LR
    S0["0 步<br/>Sparse=0 Φ=0"] --> S1["1 步<br/>Sparse=0 Φ=0.33"]
    S1 --> S2["2 步<br/>Sparse=0 Φ=0.67"]
    S2 --> S3["3 步<br/>Sparse=1 Φ=1.0"]

    style S0 fill:#ffcccc
    style S1 fill:#ffe6cc
    style S2 fill:#ffffcc
    style S3 fill:#ccffcc
```

**H1 要比的就是**：训练时用右边「过程分」的模型，是否在**新题**上比用左边「期末分」的更强。

---

<a id="fig-6"></a>

### 图 6 · GRPO 怎么训练 Attacker（一题 5 份作文比高低）

```mermaid
flowchart TB
    GOAL["抽一道攻击题"] --> GEN["Attacker 写 5 份 payload<br/>G = 5"]
    GEN --> R1["纸条 1 → Victim → 得分 r₁"]
    GEN --> R2["纸条 2 → Victim → 得分 r₂"]
    GEN --> R3["..."]
    GEN --> R5["纸条 5 → Victim → 得分 r₅"]
    R1 & R2 & R3 & R5 --> CMP["组内比：高的强化、低的抑制"]
    CMP --> UPD["更新 LoRA 权重"]
    UPD --> GOAL

    style GEN fill:#ffe6cc
    style CMP fill:#e6ffe6
```

**QLoRA / int4**：在一张 H20 上「精简装修」式微调 8B 模型，否则显存不够。

---

### 3.3 Attacker vs Victim（谁动、谁不动）

```mermaid
flowchart LR
    subgraph train ["会学习"]
        ATK["Attacker Qwen3-8B<br/>🎓 学生"]
    end
    subgraph frozen ["冻结不动"]
        VIC["Victim Qwen3.6-27B<br/>📋 考场"]
    end
    ATK -->|"投递 payload"| VIC
    VIC -->|"成功/失败/Φ"| ATK
```

---

### 3.4 OOD：温和 vs 困难（Phase A vs Phase B）

```mermaid
flowchart TB
    subgraph train_pool ["训练见过的"]
        T1["题型 staged / exfil"]
        T2["深度 1～5"]
        T3["具体参数组合 A,B,C..."]
    end

    subgraph ood_mild ["Phase A OOD · 温和"]
        O1["同题型同深度"]
        O2["新参数组合 D,E"]
    end

    subgraph ood_hard ["Phase B OOD · 困难（计划）"]
        O3["未见题型"]
        O4["travel 等新场景"]
    end

    train_pool --> ood_mild
    train_pool -.->|"还没测"| ood_hard
```

---

## 四、Phase A / Phase B 与三格实验设计

<a id="fig-7"></a>

### 图 7 · Phase A 三格 + 我们期望 vs 实际看到的

```mermaid
flowchart TB
    subgraph cells ["Phase A 三个训练格"]
        C1["S-sparse<br/>单步 + 期末分"]
        C2["M-dense<br/>多步 + 过程分 Φ"]
        C3["M-sparse<br/>多步 + 期末分"]
    end

    subgraph expect ["期望（利于 H1）"]
        E1["✔ 能训"]
        E2["✔ 能训"]
        E3["✘ 训不动"]
    end

    subgraph actual ["Week 2 实际"]
        A1["✔ ASR 83.9%"]
        A2["✔ ASR 72.5%"]
        A3["✔ ASR 75.0%"]
    end

    C1 --> E1 --> A1
    C2 --> E2 --> A2
    C3 --> E3 --> A3

    style E3 fill:#ffcccc
    style A3 fill:#fff3cd
```

```mermaid
flowchart LR
    PA["Phase A ✅<br/>GRPO 能训"] --> PB["Phase B 待定<br/>dense vs sparse 谁 OOD 更好?"]
    PA -.->|"M-sparse 也成功了"| H1warn["H1 预览 ⚠️<br/>dense 未赢 sparse"]
```

| 代号 | 练什么 | 怎么打分 | 实际 OOD ASR |
|------|--------|----------|--------------|
| S-sparse | 单步 | 期末 0/1 | **83.9%** |
| M-dense | 多步 | 过程 Φ | **72.5%** |
| M-sparse | 多步 | 期末 0/1 | **75.0%** |

---

## 五、上周：搭实验室（7/2～7/3）

<a id="fig-8"></a>

### 图 8 · 372 道题从哪来、怎么切

```mermaid
flowchart LR
    POOL["372 题 oracle 校验通过"]
    POOL --> TR["train 261<br/>70%"]
    POOL --> OOD["OOD 111<br/>30%"]

    style TR fill:#cce5ff
    style OOD fill:#ffe6cc
```

**比例示意（ASCII）**：

```
train 261  ████████████████████████████████░░░░░░░░  70%
OOD  111   ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  30%
           └─ 372 题全部 oracle 校验通过 ─┘
```

```mermaid
flowchart TB
    OFF["官方 AgentDojo 27 题 多步约8"] --> GEN["程序生成器"]
    GEN --> F1["staged_transfer 分笔转账链"]
    GEN --> F2["exfil_facts 分笔藏 secret 词"]
    F1 --> POOL["372 题"]
    F2 --> POOL
    POOL --> TR["train 261"]
    POOL --> OOD["OOD 111"]
```

**两族例题**：

| 族 | 故事 | depth 例 |
|----|------|----------|
| staged_transfer | 分几笔小额转给黑客 | $6 / 每笔≤$2 → **3 步** |
| exfil_facts | 每笔备注藏一词 | 3 个词 → **3 步** |

---

<a id="fig-9"></a>

### 图 9 · 未训练 Attacker 基线（EXP-009 baseB）

**抽 1 次 ASR (%)**

| 条件 | n | ASR |
|------|---|-----|
| 单步 | 31 | **54.8%** |
| 多步 | 40 | **20.0%** |

```
单步 n=31  ███████████████████████████░░░░░░░░░░░░░░░  54.8%
多步 n=40  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20.0%
           0%                                      100%
```

**同题抽 5 次取最好 ASR (%)**

| 条件 | ASR |
|------|-----|
| 单步 | **87.1%** |
| 多步 | **65.0%** |

```
单步  ████████████████████████████████████████████░  87.1%
多步  ████████████████████████████████░░░░░░░░░░░░░  65.0%
      0%                                          100%
```

**读图**：
- 左图：多步 **20%** ≪ 单步 **55%** → 多步难很多  
- 右图：多写几次（5 次）→ 多步 **65%** → GRPO 目标是「训完写 1 次 ≈ 未训写 5 次最好」

```mermaid
flowchart LR
    B1["读题"] --> B2["写 payload"]
    B2 --> B3["注入沙盒"]
    B3 --> B4["Victim 跑"]
    B4 --> B5["Oracle 判分"]
    B5 --> B6["日志存档"]
```

---

## 六、本周：Phase A 训练 + OOD 考试（7/6～7/7）

<a id="fig-10"></a>

### 图 10 · 一张 H20 显卡干两件事

```mermaid
flowchart TB
    subgraph H20["一张 H20 显卡 96GB"]
        direction TB
        V["Victim 服务 vLLM FP8<br/>Qwen3.6-27B 约 43GB<br/>localhost:8000"]
        A["Attacker 训练 int4 + LoRA<br/>Qwen3-8B 约 9GB<br/>GRPO 更新权重"]
    end

    V -.->|"同卡共存"| A

    style H20 fill:#f0f0f0
    style V fill:#e6f3ff
    style A fill:#ffe6cc
```

**显存示意（ASCII）**：

```
┌────────────── H20 96GB ──────────────┐
│  Victim vLLM FP8     [████████████░░] ~43GB  考场 AI  │
│  Attacker int4 LoRA  [██░░░░░░░░░░░░] ~9GB   学生    │
│  其余                [░░░░░░░░░░░░░░] ~44GB  缓冲    │
└──────────────────────────────────────┘
```

**训练 ~14h**（三格顺序跑完，<20h PI 预算）。

---

<a id="fig-11"></a>

### 图 11 · 训练过程中平均 reward 上升（train 集上）

| 训练格 | 开课前 | 结课后 |
|--------|--------|--------|
| S-sparse | 0.50 | **0.85** |
| M-dense | 0.15 | **0.80** |
| M-sparse | 0.30 | **0.75** |

```
S-sparse  开 0.50 ████████████░░░░░░░░  →  结 0.85 █████████████████░░░
M-dense   开 0.15 ███░░░░░░░░░░░░░░░░░  →  结 0.80 ████████████████░░░░
M-sparse  开 0.30 ██████░░░░░░░░░░░░░░  →  结 0.75 ███████████████░░░░░
          0.0 ─────────────────────────────── 1.0
```

→ 三格**都学动了** = Phase A「培训班有效」✅

---

<a id="fig-12"></a>

### 图 12 · OOD 考试 ASR（核心结果 · 写 1 次）

**OOD 攻击成功率 ASR (%) · 抽 1 次**

| 条件 | ASR |
|------|-----|
| BASE 单步 | 48.4% |
| BASE 多步 | 20.0% |
| S-sparse 训后 | **83.9%** |
| M-dense 训后 | **72.5%** |
| M-sparse 训后 | **75.0%** |

```
BASE单步      ███████████████████░░░░░░░░░░░░░░░░░░░░  48.4%
BASE多步      ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20.0%
S-sparse训后  █████████████████████████████████░░░░░░░  83.9%
M-dense训后   █████████████████████████████░░░░░░░░░░  72.5%
M-sparse训后  ██████████████████████████████░░░░░░░░░  75.0%
              0%                                      100%
```

**多步 OOD 平均进度 Φ（仅多步相关）**

| 条件 | mean Φ |
|------|--------|
| BASE 多步 | 0.23 |
| M-dense 训后 | **0.73** |
| M-sparse 训后 | **0.75** |

```
BASE多步      █████░░░░░░░░░░░░░░  0.23
M-dense训后   ███████████████░░░░  0.73
M-sparse训后  ███████████████░░░░  0.75
              0.0 ───────────── 1.0
```

---

<a id="fig-13"></a>

### 图 13 · GRPO 提升了多少？（Δ_raw，相对未训写 1 次）

| 对比 | Δ 百分点 |
|------|----------|
| S-sparse vs BASE 单步 | **+35.5** |
| M-dense vs BASE 多步 | **+52.5** |
| M-sparse vs BASE 多步 | **+55.0** |

```
S-sparse vs BASE单步   ████████████████████████████████░░░░░░░░░░░░  +35.5 pt
M-dense vs BASE多步    ████████████████████████████████████████████  +52.5 pt
M-sparse vs BASE多步   █████████████████████████████████████████████ +55.0 pt
                       0 pt                                    60 pt
```

**结论 1 ✅**：GRPO **确实有效**——菜鸟 ~20% → 培训后 ~73–84%。

---

<a id="fig-14"></a>

### 图 14 · H1 预览：多步 OOD 上 Dense vs Sparse 头对头

**多步 OOD · 训后 ASR 对比（H1 关键）**

| 训练格 | ASR |
|--------|-----|
| M-dense 过程分 Φ | **72.5%** |
| M-sparse 期末分 | **75.0%** |

```
M-dense 过程分Φ   █████████████████████████████░░░░░░░░░░  72.5%
M-sparse 期末分   ██████████████████████████████░░░░░░░░░  75.0%
                  65%                                    80%
```

```mermaid
flowchart TB
    Q["H1 问：过程分更 help 新题?"]
    Q --> R["实际：72.5% vs 75.0%<br/>dense 未赢 · 差距在噪声内"]
    R --> C["结论 ⚠️<br/>当前设置不支持 H1"]
    C --> N["但还不能判 H1 死刑<br/>→ Phase B 深题再测"]

    style C fill:#fff3cd
```

**结论 2 ⚠️**：Sparse **也能**训好多步 → 没有出现「只有过程分才行」的故事。

---

### 图 15 · 和「写 5 次取最好」比（更严的 bar）

| 训练格 | Δ_learn（训后1次 − 未训 best-of-5） |
|--------|-------------------------------------|
| S-sparse | **+19.4 pt** |
| M-dense | **+10.0 pt** |
| M-sparse | **+12.5 pt** |

```
S-sparse   ████████████████████░░░░░░░░░░░░░░░░░░░░░░  +19.4 pt
M-dense    ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  +10.0 pt
M-sparse   ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  +12.5 pt
           -15 pt              0 pt              +25 pt
```

虚线含义：0 = 刚好等于「未训写 5 次最好」；**三格都未显著超过**（CI 含 0），但约 **5× 采样效率**。

---

## 七、Week 2 结论一张图

```mermaid
flowchart TB
    subgraph confirmed ["已确认 高置信"]
        C1["GRPO 能训 Attacker"]
        C2["管线能跑通"]
    end

    subgraph uncertain ["暂不确定 低置信"]
        U1["Dense 优于 Sparse"]
        U2["H1 成立"]
        U3["H1 彻底失败"]
    end

    subgraph pending ["待 Phase B"]
        P1["深题 depth 4+ 头对头"]
        P2["更难 OOD + 多 seed"]
    end

    confirmed --> pending
    uncertain --> pending

    style confirmed fill:#d4edda
    style uncertain fill:#fff3cd
    style pending fill:#e6f3ff
```

| 问题 | 答案 | 信心 |
|------|------|------|
| 链路能跑吗？ | ✅ | 高 |
| GRPO 能训吗？ | ✅ +35～+55pt | 高 |
| Dense > Sparse？ | ⚠️ 72.5 vs 75 | 中 |
| H1 成立？ | ❌ 还不行 | — |
| H1 死刑？ | ❌ 还不行 | — |

---

## 八、为何还不能一锤定音？

```mermaid
flowchart TB
    ROOT["Phase A 限制"]

    ROOT --> D1["题太浅"]
    D1 --> D1a["depth 2-3 为主"]
    D1 --> D1b["过程分优势小"]

    ROOT --> D2["OOD 温和"]
    D2 --> D2a["只换新数字"]
    D2 --> D2b["未换场景"]

    ROOT --> D3["样本少"]
    D3 --> D3a["1 seed"]
    D3 --> D3b["multi n=40"]

    ROOT --> D4["Victim 偏松"]
    D4 --> D4a["未训 5 次已有 65%"]

    style ROOT fill:#fff3cd
```

**需要 Phase B**：更深链（depth≥4）+ 更难 OOD + 多 seed。

---

## 九、和你心中的 Dense 对照

```mermaid
flowchart LR
    subgraph yes ["✅ 一致"]
        Y1["按子步骤算进度 Φ"]
        Y2["不是只在终局给分"]
        Y3["只改 reward 其它对齐"]
    end
    subgraph no ["❌ 容易误解"]
        N1["Attacker 多轮对话"]
        N2["每步即时发 reward"]
        N3["Week2已证dense更好"]
    end
```

---

## 十、踩坑与修复

```mermaid
flowchart TB
    P1["payload 弄崩 YAML"] --> F1["crash-safe → 0分继续"]
    P2["Attacker 拒答 ~55%"] --> F2["换 Qwen3-8B prompt → 0%"]
    P3["Victim API 慢贵"] --> F3["本地 vLLM · 14h训完"]
    P4["官方题太少"] --> F4["生成 372 题"]
```

---

<a id="fig-15"></a>

## 十一、下一步：Phase B 决策树

```mermaid
flowchart TD
    START["Phase A 完成<br/>GRPO ✅ · dense≈sparse ⚠️"]
    START --> PI{"PI 拍板"}

    PI -->|A 小而尖| DEEP["depth≥4 头对头<br/>h1_eval_deep.py"]
    PI -->|A' 完整| FULL["+ 未见族 + travel + 多 seed"]
    PI -->|B 重审| REV["换 victim / 改主张"]

    DEEP --> D{"dense >> sparse?"}
    FULL --> D

    D -->|是| WIN["H1 可能成立<br/>写 Resolution"]
    D -->|否| LOSE["H1 可能不成立<br/>pivot / 负结果"]

    style START fill:#e6f3ff
    style WIN fill:#d4edda
    style LOSE fill:#fff3cd
```

**Agent 建议**：先做 **A（深题）**——在 dense 理论上最该赢的地方测。

---

## 十二、工程师附录

### 12.1 实验与产物

| EXP | 内容 |
|-----|------|
| EXP-2026W27-007 | 造题 PoC |
| EXP-2026W27-008 | reward.py |
| EXP-2026W27-009 | 未训基线 |
| **EXP-2026W28-001** | Phase A 三格 + eval |

远程：`/root/autodl-tmp/h1/runs/{S-sparse-s0,M-dense-s0,M-sparse-s0}/`

### 12.2 复现命令

```bash
# 本地基线
python code/scripts/h1_build_goalpool.py
python code/scripts/h1_rollout.py --smoke
python code/scripts/h1_baseline.py --k 5 --n-single 31 --n-multi 40 --workers 6 --run-tag baseB
python code/src/reward.py

# 远程 Phase A
bash /root/autodl-tmp/h1/phaseA.sh
python h1_eval.py --k 5 --n-single 31 --n-multi 40
```

---

<a id="fig-16"></a>

## 十三、论文大故事总览

```mermaid
flowchart TB
    IDEA["idea.md<br/>可验证技能 + 过程分<br/>→ OOD 组合泛化"]
    H0["H0 Week1 ✅<br/>多步攻击面存在"]
    PA["Phase A Week2 ✅<br/>GRPO 能训 attacker"]
    H1["H1 预览 ⚠️<br/>dense ≈ sparse"]
    PB["Phase B 待定"]
    H2["H2 未开始<br/>技能组合曲线"]
    H3["H3 未开始<br/>oracle vs judge"]

    IDEA --> H0 --> PA --> H1 --> PB
    PB --> H2
    PB --> H3

    style H0 fill:#d4edda
    style PA fill:#d4edda
    style H1 fill:#fff3cd
```

---

## 十四、FAQ（简）

| 问 | 答 |
|----|-----|
| 在教 AI 做坏事？ | **仿真红队**，为防御测漏洞 |
| Attacker = 聊天？ | **只写一段注入文本** |
| Φ vs ASR？ | Φ=进度条；ASR=全成功了吗 |
| Phase A 过 H1 没过？ | **课能教** ✔ · **哪种教法更好** 尚未 ✔ |
| 72.5 vs 75 差大吗？ | n=40 下**可能是噪声** |

更多讨论：[`Discussion.md`](../Discussion.md) · [`2026-W28.md`](2026-W28.md)

---

*报告版本：2026-07-07 v4（图表版：16 组 mermaid 图 + 保留零基础文字；数据同 EXP-2026W28-001）*
