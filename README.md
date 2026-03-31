# Meta-Learning（灵敏自进化学习系统）

一个面向 Agent 的三层学习系统：在任务执行前做风险评估、任务结束后捕获学习信号、再通过近线/离线流水线把经验沉淀为分类树与技能卡片。

## 架构总览

**完整产品融合架构**（Nanobot / DeskClaw × MCP × 三层学习）见 [`meta_learning_chain_fusion.drawio`](meta_learning_chain_fusion.drawio)（用 draw.io 打开）。若只需看不含对话生态框图的**核心流水线**，可参考 [`meta_learning_chain.drawio`](meta_learning_chain.drawio)。

自上而下分为：

1. **入口层** — MCP Server / OpenClaw 插件 / CLI 三种接入方式
2. **Layer 1 在线实时层** — QuickThink 风险评估（4 项检测维度）→ Agent 执行 → 信号捕获（4 类触发条件）→ 信号缓冲区
3. **Layer 2 近线整合层** — 触发门控 → 信号物化 → 经验池 → 经验聚类 → 错误分类提炼 → 技能演化
4. **Layer 3 离线深度学习层** — 跨任务根因挖掘 / 能力盲区识别 / 记忆架构优化
5. **反馈回路** — 错误分类词更新回注 QuickThink 风险索引；技能卡片注入下一次 Agent 执行

## 架构定位：能力边界

Meta-learning 是**策略与流程学习系统**，通过积累任务执行经验来帮助 Agent 在后续类似任务中更快、更准确地完成工作。

### 可学习的模式（策略层）

- **工具调用顺序与策略选择**：何时拒绝、何时接受、先验证再行动
- **多步流程完整性**：批量操作不遗漏、多意图追踪
- **政策合规**：不同场景适用不同规则（multi-rule discrimination）
- **验证纪律**：不主动越权、核实用户声明后再执行

### 不可学习的模式（参数层）

- **实例特定参数**：具体航班号、数据库主键、精确金额
- **数值计算**：加减乘除、汇率换算
- **数据库状态匹配**：评估系统内部的 ground truth 比对

### 信号来源约束

Meta-learning 仅使用 **agent 可观测数据**：

| 信号类型 | 可用性 | 说明 |
|---------|--------|------|
| 完整对话 (conversation) | 可用 | 用户与 agent 的全部消息 |
| 工具调用日志 (tool_calls) | 可用 | 名称、参数、返回结果 |
| 粗粒度结果 (reward) | 可用 | 成功/失败 |
| 用户/监督反馈 (user_corrections) | 可用 | 用户纠正或 QA 策略层反馈 |
| 评估系统内部 (db_check, reward_breakdown) | **不可用** | 生产环境无法获取 |

## 当前代码实现概览

### Layer 1（在线）

- `QuickThinkIndex`：基于 `TaskContext` 进行风险评估，当前包含 4 类检测维度（与 [`meta_learning_chain_fusion.drawio`](meta_learning_chain_fusion.drawio) 中「混合检索」一致）
  - `keyword_taxonomy_hit`：命中 `ErrorTaxonomy`——优先用**关键词子串**匹配；未命中且配置启用 `layer1.quick_think.vector_fallback_enabled`、MCP/运行时提供了 `embedding_fn` 时，对由 `task_description`、错误信息与工具名拼成的检索文本做**向量相似度**回退（阈值、top-k 见 `vector_similarity_threshold` / `vector_top_k`）
  - `irreversible_operation`：命中不可逆操作关键词（如 `rm -rf`、`drop table`、`force push`）
  - `recent_failure_pattern`：命中近期失败签名（需先注册）
  - `new_tool_usage`：命中新工具使用（基于 `new_tools` / 已知工具集合）
- 风险等级由 `_assess_risk_level()` 计算：
  - `high`：出现不可逆操作
  - `medium`：命中 2 个及以上信号
  - `low`：命中 1 个信号
  - `none`：未命中

- `SignalCapture`：按优先级捕获学习信号并写入 `signal_buffer/*.yaml`
  1. `error_recovery`
  2. `user_correction`
  3. `new_tool`
  4. `efficiency_anomaly`（默认阈值：`step_count > average_step_count * 2.0`，即 `>10`）

### Layer 2（近线整合）

`Layer2Orchestrator` 按顺序执行：

1. `Materialize`：将待处理 `Signal` 物化为 `Experience`
2. `Consolidate`：聚类经验并形成 `ExperienceCluster`
3. `Taxonomy`：从成熟聚类提炼 `ErrorTaxonomy`
4. `Skill Evolve`：根据新 taxonomy 条目创建/更新 `skills/*/SKILL.md`

触发逻辑（`should_trigger()`）：
- 待处理信号数 >= `layer2.trigger.min_pending_signals`（默认 2），或
- 距上次运行超过 `layer2.trigger.max_hours_since_last`（默认 8h）且有待处理信号

### Layer 3（离线深度学习）

`Layer3Orchestrator` 按顺序执行：

1. `CrossTaskMiner`：跨任务类型根因模式挖掘
2. `NewCapabilityDetector`：识别 taxonomy 覆盖盲区
3. `MemoryArchitect`：产出记忆架构建议（`extract` / `consolidate` / `prune`）

结果持久化为 `Layer3Result`，并更新 `layer3_state.json` 的最近运行时间。

## 运行方式

### 1) Python CLI

入口：`python -m meta_learning`

```bash
# 查看状态
venv/bin/python3 -m meta_learning status --workspace ~/.openclaw/workspace

# 运行 Layer 2（仅当触发条件满足）
venv/bin/python3 -m meta_learning run-layer2 --workspace ~/.openclaw/workspace

# 强制运行 Layer 2
venv/bin/python3 -m meta_learning run-layer2 --workspace ~/.openclaw/workspace --force

# 运行 Layer 3
venv/bin/python3 -m meta_learning run-layer3 --workspace ~/.openclaw/workspace
```

可选参数：
- `--workspace`：指定工作区根目录
- `--config`：指定配置文件（默认使用 `MetaLearningConfig` 默认值）

### 2) MCP Server

入口：`python -m meta_learning.mcp_server`

已暴露 MCP tools（以当前代码为准）：
- `quick_think`
- `capture_signal`
- `run_layer2`
- `run_layer3`
- `sync_taxonomy_to_nobot`
- `status`

已暴露 MCP resources：
- `meta-learning://taxonomy`
- `meta-learning://config`

环境变量：
- `META_LEARNING_WORKSPACE`：覆盖工作区路径（meta-learning 数据目录，如 signal_buffer、taxonomy）
- `META_LEARNING_CONFIG`：指定配置文件路径
- `META_LEARNING_SESSIONS_ROOT`：覆盖会话 JSONL 目录（MCP 启动时生效）。当工作区指向子目录（例如 `.../meta-learning-data`）而 DeskClaw 会话在仓库根下时，必须用此项或 `config.yaml` 里的 `sessions_root` 指向真实 sessions 目录。

项目内示例配置见 `.cursor/mcp.json`。

#### DeskClaw（Nanobot）会话与 meta-learning 对齐

DeskClaw 的会话文件目录一般为：

`~/.deskclaw/nanobot/workspace/sessions/`

常见文件命名：`{channel}_{chat_id}.jsonl`，例如 `agent_main_desk-e3982f34.jsonl`（`channel` 如 `agent`；`chat_id` 逻辑上多为 `main:desk-...`，落盘时常把 `:` 融入/转为下划线后与 channel 拼接，具体以你机器上的文件名为准）。

JSONL 行类型可包含：`metadata`、`user`、`assistant`、`tool` 等。另有一套按设备区分的日志在 `~/.deskclaw/logs/` 下，**物化（materialize）读上下文用的是 `workspace/sessions` 下的 JSONL**，不是 logs 目录。

**重要**：若 `META_LEARNING_WORKSPACE` 设为 `~/.deskclaw/nanobot/workspace/meta-learning-data`（推荐，与数据与 skills 分离），则默认的 `workspace_root/sessions` 会指向 `meta-learning-data/sessions`，通常**不是**聊天会话所在位置。请同时配置：

- `META_LEARNING_SESSIONS_ROOT=~/.deskclaw/nanobot/workspace/sessions`，或  
- 在 `meta-learning-data/config.yaml` 中设置 `sessions_root: ~/.deskclaw/nanobot/workspace/sessions`

并保证写入 Signal 的 `session_id` 与磁盘上的 basename（不含 `.jsonl`）一致；若 id 中含 `:`，解析时会额外尝试将 `:` 换成 `_` 的文件名（见 `resolve_session_file`）。

### 3) OpenClaw 插件（`openclaw-plugin/`）

- `before_prompt_build`：执行 Quick Think，并将风险提示注入上下文
- `agent_end`：分析对话并写入信号文件
- `before_tool_call` / `session_start`：跟踪工具调用计数

插件配置声明见 `openclaw-plugin/openclaw.plugin.json`。

## 配置说明

默认配置文件：`config.yaml`（可通过 CLI `--config` 或 MCP 环境变量覆盖）

关键配置项：
- 路径：`workspace_root`、`sessions_root`、`signal_buffer_dir`、`experience_pool_dir`、`error_taxonomy_path`、`skills_dir`
- Layer 1：不可逆关键词、效率异常阈值
- Layer 2：触发条件、初始置信度、taxonomy/skill 阈值
- Layer 3：跨任务挖掘门槛、盲区最小出现次数、记忆裁剪阈值
- `llm.provider`：`stub`（默认）或 `openai`

## 开发与测试

### 依赖

- Python `>=3.12`
- 运行时依赖见 `pyproject.toml`：`pydantic`、`pyyaml`、`httpx`
- MCP 依赖（可选）：`mcp`（`[project.optional-dependencies].mcp`）

### 安装（开发模式）

```bash
venv/bin/python3 -m pip install -e ".[dev,mcp]"
```

### 运行测试

```bash
venv/bin/python3 -m pytest tests/
```

### 运行集成演示脚本

```bash
venv/bin/python3 scripts/main_test.py
```

## 目录结构

- `src/meta_learning/`：Python 主体实现（Layer1/2/3、MCP、CLI）
- `openclaw-plugin/`：OpenClaw 插件实现（TS）
- `tests/`：单元测试与 mock 资源
- `schemas/`：YAML schema
- `scripts/`：集成演示脚本
- `.cursor/mcp.json`：本地 MCP Server 启动配置示例
