# baseline/ · Baseline 代码（可选）

放置对比实验所用的 baseline 项目。本目录不强制存在——若你的研究无 baseline 对比，可以删掉。

## 建议结构

```
baseline/
├── <baseline-name-1>/     一个 baseline 一个子目录
│   ├── ...原始项目结构...
│   └── PATCH.md           （可选）记录你对该 baseline 的任何改动
├── <baseline-name-2>/
└── README.md              本文件
```

## 与协议的对接

- 每个 baseline 子目录尽量保持其**原始项目结构**，便于追溯出处与版本。
- 在 `idea.md § 3.3 基准对比` 中列出每个 baseline 的论文/链接。
- 跑 baseline 也是 Agent 可自治的任务（见 `AGENTS.md § 9` 决策边界表）。
- 引入**新** baseline 是用户决策，Agent 不得自行添加。
