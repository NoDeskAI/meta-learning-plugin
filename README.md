# Meta-Learning MCP Plugin

面向 Nobot / DeskClaw 的自进化学习 MCP 插件。通过任务前风险评估、任务后信号捕获、近线/离线流水线，将 Agent 的执行经验沉淀为错误分类体系（Error Taxonomy）和技能卡片（SKILL.md），实现"犯过的错不再犯"。

## 给 Nobot 安装

### 前提

- Python >= 3.12
- DeskClaw / Nobot 已安装（`~/.deskclaw/` 存在）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/NoDeskAI/meta-learning-plugin ~/.deskclaw/plugins/meta-learning

# 2. 运行安装脚本（自动创建 venv、安装依赖、生成配置）
cd ~/.deskclaw/plugins/meta-learning
bash install.sh
```

`install.sh` 执行完毕后会输出一段 JSON，将其添加到 `~/.deskclaw/nanobot/config.json` 的 `tools.mcp_servers` 中：

```json
"meta-learning": {
  "type": "stdio",
  "command": "~/.deskclaw/plugins/meta-learning/.venv/bin/meta-learning-mcp",
  "args": [],
  "tool_timeout": 120,
  "env": {
    "META_LEARNING_WORKSPACE": "~/.deskclaw/nanobot/workspace/meta-learning-data",
    "META_LEARNING_CONFIG": "~/.deskclaw/nanobot/workspace/meta-learning-data/config.yaml",
    "META_LEARNING_SESSIONS_ROOT": "~/.deskclaw/nanobot/workspace/sessions"
  }
}
```

然后重启 DeskClaw / Nobot 使插件生效。

### 卸载

```bash
cd ~/.deskclaw/plugins/meta-learning
bash uninstall.sh
```

脚本会自动移除 MCP 注册、数据目录、技能文件和虚拟环境，完成后重启 DeskClaw 即可。

## MCP 工具一览

| 工具 | 用途 |
|------|------|
| `quick_think` | 任务执行前风险评估，命中已知错误模式时返回风险提示 |
| `capture_signal` | 任务结束后捕获学习信号（错误恢复、用户纠正、效率异常等） |
| `run_layer2` | 手动触发 Layer 2 近线整合流水线（物化→聚类→分类→技能演化） |
| `layer2_status` | 查询 Layer 2 流水线当前状态（idle/running/completed/failed） |
| `run_layer3` | 触发 Layer 3 离线深度学习（跨任务根因挖掘、盲区识别） |
| `sync_taxonomy_to_nobot` | 将最新错误分类同步为 Nobot 的 SKILL.md 和分类规则文件 |
| `status` | 查看系统状态（信号数、经验数、分类条目数等） |

## 架构概览

```
Layer 1（在线实时）
  QuickThink 风险评估 → Agent 执行 → 信号捕获 → 信号缓冲区

Layer 2（近线整合，capture_signal 达到阈值后自动触发）
  信号物化 → 经验池 → 经验聚类 → 错误分类提炼 → 技能演化

Layer 3（离线深度学习，手动触发）
  跨任务根因挖掘 / 能力盲区识别 / 记忆架构优化

反馈回路
  错误分类 → 更新 QuickThink 风险索引
  技能卡片 → 注入下一次 Agent 执行
```

### 信号触发条件

| 触发类型 | 说明 |
|---------|------|
| `user_correction` | 用户纠正了 Agent 的行为（最高优先级信号） |
| `unresolved_error` | Agent 遇到错误但未解决（高学习价值） |
| `self_recovery` | Agent 遇到错误并自行恢复 |
| `new_tool` | Agent 首次使用某个工具 |
| `efficiency_anomaly` | 步骤数显著超过平均值 |

## 配置说明

运行时配置文件：`~/.deskclaw/nanobot/workspace/meta-learning-data/config.yaml`（由 `install.sh` 从 `config.deskclaw.yaml` 生成）。

关键配置项：

| 配置路径 | 默认值 | 说明 |
|---------|--------|------|
| `llm.provider` | `openai` | LLM 提供方（`stub` 为测试模式） |
| `llm.model` | `minimax-m2.7` | 用于信号物化和分类提取的模型 |
| `layer2.trigger.min_pending_signals` | `2` | 触发 Layer 2 的最小待处理信号数 |
| `layer2.trigger.max_hours_since_last` | `8` | 超过此时间且有待处理信号则触发 |
| `layer2.taxonomy.min_confidence_for_skill` | `0.8` | 分类条目提升为 SKILL 的最低置信度 |

LLM API（内部 gateway）和多模态 Embedding（DashScope）的凭证已内置于代码中，安装后无需额外配置。如需覆盖，可设置环境变量 `META_LEARNING_LLM_BASE_URL` / `META_LEARNING_LLM_API_KEY` / `DASHSCOPE_API_KEY`。

## 开发与测试

```bash
# 安装开发依赖
pip install -e ".[dev,mcp]"

# 运行测试
pytest tests/

# 本地启动 MCP server
python -m meta_learning.mcp_server
```

## 目录结构

```
src/meta_learning/       Python 主体实现
  ├── layer1/            在线层（QuickThink、SignalCapture）
  ├── layer2/            近线层（Materialize、Consolidate、Taxonomy、Orchestrator）
  ├── layer3/            离线层（CrossTaskMiner、NewCapabilityDetector、MemoryArchitect）
  ├── shared/            共享模型、IO、LLM 接口
  └── mcp_server.py      MCP Server 入口
tests/                   单元测试
config.yaml              通用配置模板
config.deskclaw.yaml     DeskClaw 专用配置模板
install.sh               自动化安装脚本
uninstall.sh             自动化卸载脚本
```
