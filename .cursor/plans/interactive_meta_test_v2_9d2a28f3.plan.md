---
name: Interactive Meta Test v2
overview: 多轮交互验证 meta-learning：Observe-Then-Correct 模式，脚本先观察 agent 行为再生成上下文化纠正，每场景多次运行建立统计 baseline，使用 minimax-m2.7 (nodesk, temp=0)。
todos:
  - id: config
    content: 创建 abtest/config.interactive.yaml（workspace、LLM nodesk+dashscope、低阈值触发、temp=0）
    status: completed
  - id: script-core
    content: 编写 scripts/interactive_meta_test.py 核心框架：Scenario dataclass、执行循环、Phase A/B/C/D 流程
    status: completed
  - id: scenarios-impl
    content: 实现 4 个场景的 workspace_setup、check_fn、build_correction、auto-check
    status: completed
  - id: eval-llm
    content: 实现 LLM 评估函数：gpt-4o 判断 agent 行为是否遵从纠正（补充 auto-check）
    status: completed
  - id: run-experiment
    content: 运行完整实验（4 场景 x 6 trials + 1 跨场景）并分析结果
    status: completed
isProject: false
---

# 多轮交互 Meta-Learning 验证实验 v2

## 设计原则

1. **Observe-Then-Correct**：脚本观察 agent 实际行为后再给出纠正，不是 pre-canned
2. **模板化纠正**：纠正文本是确定性模板 + agent 行为参数，无 LLM 随机性
3. **多次运行**：每场景 3 次 baseline + 3 次 treatment，建立统计对比
4. **Agent temp=0**：控制变量，唯一变量是 taxonomy 有/无
5. **Auto-check 为主**：核心指标由脚本确定性检查，LLM 评估仅作补充

## 4 个测试场景

### 场景 1：工作目录偏好（持久事实记忆）

- **Workspace 初始化**：空的 temp 目录，HOME 设为 temp home
- **Round 1 任务**："Create a new Python project called 'calculator' with main.py and a README"
- **检查函数**：检查项目创建在哪个路径下（`find $HOME -name main.py -path '*/calculator/*'`）
- **纠正模板**：
  - 若 agent 未用 ~/projects：`"You created the project at {actual_path}. I keep ALL my projects under ~/projects/. Always create projects there, never in the current directory or home root."`
  - 若 agent 已用 ~/projects：`"Good. Just confirming: always create all projects under ~/projects/."`
- **Round 2 任务**："Create a new Python project called 'todo-app' with main.py and a README"
- **Auto-check**：`test -f ~/projects/todo-app/main.py`

### 场景 2：Git 工作流程序（程序步骤顺序）

- **Workspace 初始化**：git repo with README.md（含 typo "teh"）, setup.py（version 0.1.0）
- **Round 1 任务**："There's a typo in README.md ('teh' should be 'the'). Fix it and commit."
- **检查函数**：`git branch --show-current`（是否在 main？）+ `git log --oneline -1`
- **纠正模板**：
  - 若 agent 在 main 上 commit：`"You committed directly to main. I never commit to main directly. Always create a feature branch (e.g., fix/readme-typo) first, commit there."`
  - 若 agent 建了分支：`"Good branch discipline. Remember: always create feature branches, never commit to main."`
- **Round 2 任务**："The version in setup.py says 0.1.0 but should be 0.2.0. Fix it and commit."
- **Auto-check**：`git branch --show-current` != "main" AND 新分支存在

### 场景 3：代码风格约定（编码惯例）

- **Workspace 初始化**：空 Python 项目结构
- **Round 1 任务**："Write a Python function read_config(path) that reads a JSON config file and returns the parsed dict. Save it to utils.py."
- **检查函数**：`grep -c "except:" utils.py` + `grep -c "except Exception" utils.py`
- **纠正模板**：
  - 若 agent 用了 bare/generic except：`"I see you used '{matched_pattern}' in utils.py. Never use bare except or generic Exception. Always catch specific exceptions like FileNotFoundError, json.JSONDecodeError."`
  - 若 agent 已用具体异常：`"Good exception handling. Always use specific exceptions, never bare except."`
- **Round 2 任务**："Write a Python function read_csv_data(path) that reads a CSV file and returns a list of dicts. Save it to data_utils.py."
- **Auto-check**：`grep -c "except:" data_utils.py` == 0 AND (`grep -c "FileNotFoundError\|csv.Error\|UnicodeDecodeError" data_utils.py` > 0)

### 场景 4：验证纪律（流程完整性）

- **Workspace 初始化**：Python 项目含空 requirements.txt
- **Round 1 任务**："Add 'requests' to requirements.txt and install it with pip."
- **检查函数**：从 agent 消息中提取所有 shell 命令，检查是否有 `pip list`/`pip show`/`pip freeze` 类验证命令
- **纠正模板**：
  - 若无验证步骤：`"You installed requests but didn't verify it. After pip install, always run 'pip show <package>' to confirm the installation succeeded and check the version."`
  - 若有验证：`"Good verification habit. Always verify after install."`
- **Round 2 任务**："Add 'pyyaml' to requirements.txt and install it with pip."
- **Auto-check**：agent shell 命令中包含 `pip show pyyaml` 或 `pip list | grep -i yaml`

## 跨场景迁移测试（Phase D）

在 4 个场景的 taxonomy 全部累积后，执行一个**综合任务**：
- "Create a new Python project called 'web-scraper' with main.py. Add requests and beautifulsoup4 to requirements.txt, install them. Write a scraper.py that fetches a URL and parses HTML (with proper error handling). Initialize a git repo and commit your work."
- 检查 4 项规则是否被遵守

## 执行流程

```
对每个场景 S:
  ┌─ Phase A: Baseline (3 次, 无 taxonomy) ─────────────────────┐
  │  for trial in 1..3:                                          │
  │    1. 创建干净 workspace + 初始文件                            │
  │    2. agent 执行 Round 1 任务 (temp=0, 无 taxonomy)           │
  │    3. inspect_result = S.check_fn(workspace)                  │
  │    4. 记录: {trial, "baseline", agent行为, check_result}      │
  └──────────────────────────────────────────────────────────────┘
  
  ┌─ Signal & Learn ────────────────────────────────────────────┐
  │  1. correction = S.build_correction(最后一次 inspect_result)  │
  │  2. capture_signal(user_corrections=[correction])             │
  │  3. run_layer2(force=True)                                    │
  │  4. taxonomy_text = load_taxonomy()                           │
  └──────────────────────────────────────────────────────────────┘
  
  ┌─ Phase C: Treatment (3 次, 有 taxonomy) ────────────────────┐
  │  for trial in 1..3:                                          │
  │    1. 创建干净 workspace + 初始文件                            │
  │    2. agent 执行 Round 2 任务 (temp=0, 注入 taxonomy)         │
  │    3. inspect_result = S.check_fn(workspace)                  │
  │    4. llm_eval = evaluate_compliance(correction, agent_log)   │
  │    5. 记录: {trial, "treatment", agent行为, check, llm_eval}  │
  └──────────────────────────────────────────────────────────────┘

Phase D: 跨场景迁移 (1 次综合任务, 累积 taxonomy)
```

## 技术实现

### 新文件

- **[scripts/interactive_meta_test.py](scripts/interactive_meta_test.py)**：主实验脚本
- **[abtest/config.interactive.yaml](abtest/config.interactive.yaml)**：实验配置

### Agent 配置

- **模型**：`minimax-m2.7` via nodesk gateway
- **Base URL**：`https://llm-gateway-api.nodesk.tech/default/v1`
- **API Key**：`NODESK_API_KEY` 环境变量
- **Temperature**：0（确定性）
- **工具**：shell（复用 [scripts/gdpval_meta_test.py](scripts/gdpval_meta_test.py) 的 `_run_shell`）

### Meta-learning LLM

- Layer 2 materialize/taxonomy：DashScope `qwen3.5-plus`（`DASHSCOPE_API_KEY`）
- LLM 评估：nodesk gateway `openai/gpt-4o`

### 场景数据结构

```python
@dataclass
class Scenario:
    name: str
    workspace_setup: Callable[[str], None]   # 初始化 workspace
    task_round1: str                          # Round 1 任务 prompt
    task_round2: str                          # Round 2 任务 prompt
    check_fn: Callable[[str, list[str]], dict]  # 检查 agent 行为，返回 {passed, details}
    correction_template_wrong: str            # agent 做错时的纠正模板
    correction_template_right: str            # agent 做对时的偏好声明
    build_correction: Callable[[dict], str]   # 从检查结果构建最终纠正文本
```

### 核心指标

- **自然错误率** = Phase A 中 agent 犯目标错误的比例 (e.g., 2/3 = 67%)
- **Treatment 遵从率** = Phase C 中 agent 遵从纠正的比例 (e.g., 3/3 = 100%)
- **净贡献** = 遵从率 - (1 - 自然错误率) (e.g., 100% - 33% = +67%)
- **跨场景遵从率** = Phase D 综合任务中遵从的规则数 / 4

### 成功标准

- **基本成功**：至少 3/4 场景的净贡献 > 0
- **强成功**：4/4 场景净贡献 > 0 且平均净贡献 > 30%
- **迁移成功**：Phase D 跨场景遵从率 >= 75%

### 结果输出

- `abtest/results/interactive_meta/{run_id}/results.jsonl`：逐 trial 详细记录
- `abtest/results/interactive_meta/{run_id}/summary.json`：汇总 + 净贡献计算
- `abtest/results/interactive_meta/{run_id}/taxonomy_snapshots/`：每场景的 taxonomy YAML
