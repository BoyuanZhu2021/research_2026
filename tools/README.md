# tools/ · 协议辅助脚本

零依赖，纯 Python 3.9+。

| 脚本 | 作用 | 用法 |
|---|---|---|
| `session_check.py` | **会话开始自检（唯一实现）**：未初始化 → 输出 bootstrap 指引；已初始化 → 输出 mode / active 议题 / 当周日志 / 是否欠 Weekly Retro。Claude Code 的 SessionStart hook 与其他 Agent（Codex / Cursor）共用此脚本 | `python tools/session_check.py`（`--hook` 输出 Claude Code hook 的 JSON 信封） |
| `new_week.py` | 新建当周 `LOGS/YYYY-Www.md`（已存在则跳过） | `python tools/new_week.py` 或 `python tools/new_week.py 2026-W11` |
| `new_exp.py` | 往当周 LOGS 追加一个 EXP 骨架（自动算 NNN / commit hash / 主机名 / 当前 active 议题号） | `python tools/new_exp.py "源意图一句话"` |
| `new_disc.py` | 开启/关闭 Discussion 议题：自动分配 `DISC-YYYYWww-NNN`（扫描 Archive 取 max+1，避免撞号）；close 时校验 Resolution → 归档为 `Archive/DISC-YYYYWww-NNN-<slug>.md` → 从模板重置 | `python tools/new_disc.py open "<标题>" [--owner "PI @张三"]` / `close "<slug>"` / `next` |
| `lint_protocol.py` | **error**（始终阻塞）：坏格式 EXP/DISC ID、Resolved 议题缺 Decision；**warning**（仅初始化后）：未填占位符、LOGS 中 EXP 块必填字段为空。`--strict` 把 warning 升级为 error | `python tools/lint_protocol.py [--strict] [路径]` |
| `install_hooks.sh` | 把 git 指向 `.githooks/`，启用 pre-commit 自动 lint（每个新 clone 跑一次即可） | `bash tools/install_hooks.sh` |

`templates/` 存放协议文件的标准模板（当前为 `Discussion.template.md`，议题关闭后由 `new_disc.py` 用其重置根目录 `Discussion.md`）。

## pre-commit 行为

装上后每次 `git commit` 会自动跑 `lint_protocol.py`（默认模式）：
- 只有 warning → 打印提醒，commit 正常进行（允许提交进行中的实验骨架）
- 存在 error → commit 被中止，必须先修复坏 ID / 议题 Decision
- Reflect 后置（`AGENTS.md § 4.2`）要求 Agent 对当周日志跑 `--strict`，warning 也要清零
- 紧急绕过：`git commit --no-verify`（不推荐）

## 设计原则

- **零依赖**：只用标准库，避免污染科研项目自身的环境。
- **不破坏现有文件**：`new_week.py` 已存在则跳过；`new_exp.py` 只追加不重写；`new_disc.py` 归档前先校验。
- **可被 Agent 直接调用**：所有脚本以非零退出码报错，便于 Agent 检测。
- **一份逻辑，多处复用**：会话检查只活在 `session_check.py`，hook 与跨 Agent 约定都指向它；引导文案只活在 `bootstrap.md`。
