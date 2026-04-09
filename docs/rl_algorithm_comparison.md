RL 算法对比：AReaL / OpenClaw-RL / MetaClaw / SAVR (Ours)
版本: v2 — 整合 session_context 验证  
核心修正: R_taxonomy (TGRS) 降级为 ablation，创新点转向 Signal-Adaptive Verifiable Reward (SAVR)


---

Part A：三个参考方法详解

A.1 AReaL — 数学推理 RL

任务领域：GSM8K 数学应用题（人工标注答案作为 GT）

训练数据：数据集自带标准答案（answer 字段），reward 函数直接比对。无需 LLM 判断。

奖励函数：gsm8k_reward_fn → math_verify_worker.verify(completions, answer) → 二值 0/1


areal/reward/gsm8k.py (简化)
def gsm8k_reward_fn(completions, answer):
    return [1.0 if math_verify(c, answer) else 0.0 for c in completions]

**训练算法**：PPO（GRPO 风格变体），on-policy 生成多候选后按奖励排名更新。

**举例**：
输入:  "A farmer has 12 apples and buys 15 more. How many total?"
GT:    27
候选1: "12 + 15 = 27" → reward = 1.0 ✓
候选2: "12 × 15 = 180" → reward = 0.0 ✗
候选3: "12 + 15 = 28" → reward = 0.0 ✗
候选4: "The answer is 27" → reward = 1.0 ✓

**关键特点**：GT 是客观数学答案，reward 零噪声，验证成本极低。

---

### A.2 OpenClaw-RL — Agent 策略优化（PRM + OPD 双信号）

**任务领域**：Agent 与环境的实时多轮对话，无预定义数据集。

**训练数据**：Agent 在真实环境中执行任务的轨迹（tool calls、outputs、errors）。

**优化目标由两个互补机制共同构成**：

#### 机制 1：Binary RL（基于 PRM 的标量奖励）

将 next-state 中的**评估性信号 (Evaluative signals)** 转化为标量过程奖励。

```python
openclaw_api_server.py L94-136 (简化)
def _build_prm_judge_prompt(response_text, next_state_text):
    """next-state 是客观环境状态，作为 judge 的证据"""
    return f"""
    ## Agent Response
    {response_text}
    ## Next State (what actually happened)
    {next_state_text}
    Did the response help the user? Answer: \\boxed{{1}} / \\boxed{{-1}} / \\boxed{{0}}
    """

L620-662: 3 次投票取多数
reward = majority_vote([judge(prompt) for _ in range(3)])  # {-1, 0, +1}

PRM 模型**不是训练出来的**，`PRM_MODEL_PATH` 默认为 `HF_CKPT`（同一个基座模型 zero-shot 做 judge）。覆盖面广，适用于所有有反馈的交互。

#### 机制 2：OPD — Hindsight-Guided On-Policy Distillation（核心创新）

当 next-state 包含**指令性信号 (Instructive signals)**（如用户纠正、具体错误提示）时，从中提取 Hint，构建增强上下文，让模型重新计算动作概率分布。通过对比"增强分布"（teacher）和"原始分布"（student），得到 **token 级别**的优势函数。

```python
OPD 核心思路 (简化)
1. 从 next-state 提取 instructive hint
hint = extract_hint(next_state)  # e.g. 用户纠正: "应该先 ls 再删"

2. 构建增强上下文 → teacher 分布
s_enhanced = concat(original_state, hint)
log_pi_teacher = model.log_prob(action, context=s_enhanced)

3. 原始上下文 → student 分布
log_pi_student = model.log_prob(action, context=original_state)

4. Token 级优势 = teacher 与 student 的 log prob 差
A_opd = log_pi_teacher - log_pi_student  # per-token!

**联合优势函数**：
A_t = w_binary · r_final + w_opd · (log π_teacher(a_t | s_enhanced) - log π_θ(a_t | s_t))
默认 `w_binary = w_opd = 1`，二者等权。

**OPD 的价值**：PRM 只给出 +1/-1 标量，丢失了"哪些 token 该改"的细粒度信息。OPD 通过 hindsight 恢复了这些信息——当用户说"应该先 ls 再删"时，模型在增强上下文下会给"ls"更高概率，OPD 利用这种概率差为每个 token 构建监督信号。

**举例**：
场景: Agent 直接 rm -rf，用户纠正"以后先 ls 确认"

Binary RL (PRM):
  Agent 回复 vs next-state → Judge: \boxed{-1}（标量信号，仅知道"做错了"）

OPD (token 级):
  hint = "以后先 ls 确认再删"
  增强上下文: [原始对话 + hint]
  teacher 给 "ls -la" 高概率, "rm -rf" 低概率
  student 给 "rm -rf" 高概率, "ls -la" 低概率
  → 每个 token 都有精确的更新方向

**关键特点**：
- PRM 提供广覆盖的评估性信号（什么做对了/做错了）
- OPD 提供深度的指令性信号（具体该怎么改），恢复标量奖励丢失的 token 级信息
- 二者互补：PRM 覆盖面广但粗糙，OPD 精细但仅在有 instructive signal 时可用

---

### A.3 MetaClaw — Agent 持续进化（PRM + Skill-Driven 双循环）

**任务领域**：同 OpenClaw-RL，Agent 与环境的多轮对话。

**训练数据**：Agent 执行轨迹，格式化为 (instruction, response) 对。

**持续进化由两个互补机制构成**：

#### 机制 1：PRM（RL 梯度更新的奖励信号）

外部 LLM API（如 GPT-5.2）做 judge，对每一步行为细粒度评估。在用户空闲时段触发 LoRA 微调（机会主义策略优化），配合严格的数据版本控制防止过时奖励污染。

```python
prm_scorer.py (简化)
class PRMScorer:
    def evaluate(self, response, instruction):
        """3 次独立采样 → 多数投票"""
        votes = [self.llm_judge(response, instruction) for _ in range(3)]
        return majority_vote(votes)  # {-1, 0, +1}

PRM 模型为外部 LLM API，不是训练出来的专用模型。

#### 机制 2：Skill-Driven Rapid Adaptation（非梯度的即时适应）

LLM Evolver 分析失败轨迹 → 识别失败模式 → 合成新技能指令 → 注入技能库。实现零停机的即时改进，无需等待 RL 批量训练。
失败轨迹 → LLM Evolver 分析 → 新 Skill 指令 → 注入技能库 → Agent 立即获得新能力

**双循环的互补关系**：
- **PRM RL（慢循环）**：梯度更新优化策略 → 产出更高质量轨迹
- **Skill Adaptation（快循环）**：零停机即时改进 → 更丰富的技能库帮助获取更高奖励
- 正反馈：RL 提升轨迹质量 → 为 Skill 合成提供更好素材 → 技能库丰富 → 探索更广 → RL 获得更多训练信号

**举例**：
PRM (慢循环):
  Instruction: "帮用户配置 nginx 反向代理"
  Response:    "建议你去看 nginx 文档"
  → GPT-5.2 judge: \boxed{-1} → LoRA 梯度更新（下次空闲时）

Skill Adaptation (快循环):
  LLM Evolver 分析失败: "Agent 未实际执行配置操作"
  → 合成 Skill: "当用户要求配置服务时，应直接操作而非推诿"
  → 立即注入技能库 → Agent 下次立刻改进（无需等 RL）

**关键特点**：
- PRM 依赖外部 LLM 判断，没有客观事实锚定，judge 可能幻觉
- Skill-Driven Adaptation 提供了非梯度的快速改进通道
- 双循环互补：RL 负责策略级优化，Skill 负责即时适应

---

## Part B：三方法关键维度对比

| 维度 | AReaL | OpenClaw-RL | MetaClaw |
|------|-------|-------------|----------|
| **训练数据** | GSM8K 数学题（人工标注） | Agent 实时轨迹（无预定义） | Agent 实时轨迹（无预定义） |
| **GT 来源** | 数据集标注答案 | next-state（环境状态） | 无（LLM 主观判断） |
| **GT 质量** | 完美（数学答案唯一） | 中等（单步状态快照） | 低（LLM 可能幻觉） |
| **RL 奖励** | `math_verify` → 0/1 | PRM: LLM judge → {-1, 0, +1} | PRM: 外部 LLM judge → {-1, 0, +1} |
| **额外优化机制** | 无 | **OPD**: hindsight 蒸馏 → token 级优势 | **Skill-Driven**: LLM Evolver → 技能注入 |
| **优势函数粒度** | 候选级（binary） | **token 级**（PRM 标量 + OPD token 差） | 候选级（PRM 标量） |
| **非梯度学习** | 无 | 无 | **有**（Skill Adaptation 零停机即时改进） |
| **PRM 实现** | 无（不需要） | zero-shot 同模型 judge | zero-shot 外部 LLM API |
| **算法** | PPO (GRPO 风格) | GRPO + OPD | GRPO (IS) |
| **训练目标** | 数学推理能力 | Agent 策略 | Agent 策略 |
| **硬件** | 多节点 GPU 集群 | 8×A100/H100 | 8×A100 |
| **核心创新** | 大规模 RL infra | OPD: 从标量恢复 token 级监督 | 双循环: RL + 非梯度 Skill 互补 |
| **核心劣势** | 仅适用于有标准答案的域 | OPD 仅在有 instructive signal 时可用 | PRM 无客观验证；Skill 合成依赖 LLM 质量 |

---

## Part C：我们的 SAVR 方案

### C.1 训练目标

让 **Qwen3.5-32B-A3B**（MoE: 32B 总参 / 3B 激活）在 **materialize** 任务上超过通用大模型 API。

具体目标：给定一个学习信号（Signal）和会话上下文（Session Context），生成高质量的结构化错误分析（MaterializeResult）。

### C.2 数据结构说明

**Signal** — Layer 1 捕获的学习信号：
```yaml
signal_id: "sig-20260401-001"
trigger_reason: user_correction  # 5种之一
session_id: "sess-abc123"
task_summary: "用户要求删除旧项目目录"
error_snapshot: null              # 系统错误消息（如有）
user_feedback: "以后删除前先列出内容让我确认"  # 用户原文
keywords: ["rm", "delete", "确认"]
step_count: 3

**Session Context** — 完整会话日志：
```jsonl
{"role": "user", "content": "帮我删除 ~/old-project"}
{"role": "assistant", "tool_calls": [{"name": "bash", "args": "rm -rf ~/old-project"}]}
{"role": "tool", "content": "directory removed"}
{"role": "user", "content": "以后删除前先 ls 列出内容，让我确认后再删"}

MaterializeResult — 模型需要生成的结构化分析：
{
  "scene": "用户要求删除目录，Agent 直接执行 rm -rf 未经确认",
  "failure_signature": "destructive_op_without_confirmation",
  "root_cause": "执行 rm -rf ~/old-project 时未先 ls 列出目录内容供用户确认",
  "resolution": "先 ls -la 列出内容 → 等用户确认 → 再 rm -rf",
  "meta_insight": "任何不可逆操作（删除、覆盖、DROP）前必须先展示内容并等待确认",
  "task_type": "devops"
}

Experience — 从 MaterializeResult 包装而来，增加元数据：
id: "exp-20260401-001"
task_type: devops
source_signal: "sig-20260401-001"
confidence: 0.8
scene: "..."       # 同 MaterializeResult
root_cause: "..."  # 同 MaterializeResult
resolution: "..."  # 同 MaterializeResult
meta_insight: "..." # 同 MaterializeResult

**TaxonomyEntry** — 从多个 Experience 聚合而来的错误分类知识：
```yaml
id: "tax-destructive-op-001"
name: "不可逆操作未确认"
trigger: "当 Agent 即将执行删除、覆盖、DROP 等不可逆操作时"
fix_sop: "1. 列出受影响内容 2. 等待用户确认 3. 执行操作"
prevention: "在 prompt 中注入: 任何不可逆操作前必须先展示内容"
confidence: 0.85
keywords: ["rm", "delete", "drop", "覆盖", "truncate"]

C.3 奖励函数 — Signal-Adaptive Verifiable Reward (SAVR)

核心思想：不同信号类型拥有不同质量的 GT，奖励函数应自适应地选择能用的验证手段。

五个奖励组分

R_format — 格式合规（门槛奖励，RLVR 风格）
- JSON 解析失败 → -1.0
- 每个非空必填字段 → +0.04（5 字段 = 0.20）
- task_type 非法值 → -0.05
- 范围: [-1.0, +0.2]
- 类型: rule-based
R_keyword_anchor — 关键词锚定（仅 user_correction）
- resolution 中是否出现 user_feedback 的关键词
- 计算方式: keyword_overlap(resolution, user_feedback) × 0.15
- 范围: [0, +0.15]
- 类型: rule-based
R_entity_anchor — 实体锚定（所有 error 类信号）
- root_cause 是否引用 error_snapshot 或 session_context 中的具体实体
  - 实体引用率（命令/路径/错误码）× 0.15
  - 步骤锚定（"执行 X 时"、"第 N 步"）× 0.08
  - 因果链（"因为"、"导致"）× 0.07
- 范围: [0, +0.30]
- 类型: rule-based
R_align — Resolution 语义对齐（仅 user_correction）
- cosine_sim(embed(resolution), embed(user_feedback)) × 0.35
- 使用 bge-small-zh embedding
- 范围: [-0.1, +0.35]
- 类型: embedding
R_prm — LLM-as-Judge（以 session_context 为证据）
- 独立 LLM 评审：3 次采样 → 多数投票 → {-1, 0, +1} × 0.20
- 关键: session_context（完整会话轨迹）作为客观证据
- 三个变体:
  - prm: 评估整体分析准确性（user_correction, self_recovery）
  - prm_diag: 仅评估诊断，不评估 resolution（unresolved_error）
  - prm_weak: 弱约束评估（new_tool, efficiency_anomaly）
- 范围: [-0.20, +0.20]
- 类型: LLM judge
按信号类型选择激活组分 + 按 GT 质量分层加权

EVAL_COMPONENTS = {
    USER_CORRECTION:    ["format", "keyword_anchor", "entity_anchor", "align", "prm"],
    SELF_RECOVERY:      ["format", "entity_anchor", "prm"],
    UNRESOLVED_ERROR:   ["format", "entity_anchor", "prm_diag"],
    NEW_TOOL:           ["format", "prm_weak"],
    EFFICIENCY_ANOMALY: ["format", "prm_weak"],
}

SIGNAL_TYPE_WEIGHT = {
    USER_CORRECTION:    1.0,   # 有用户原文作为硬 GT
    SELF_RECOVERY:      0.8,   # session 轨迹提供强验证（v2 上调）
    UNRESOLVED_ERROR:   0.5,   # 有 error_snapshot，但无正确答案
    NEW_TOOL:           0.25,  # 仅格式 + 弱 PRM
    EFFICIENCY_ANOMALY: 0.2,   # 仅格式 + 弱 PRM
}

**设计逻辑**：
- `user_correction` 拥有最丰富的 GT（用户原文 + session），所有组分全开
- `self_recovery` 无用户反馈但有完整 session 轨迹可验证分析准确性，权重上调到 0.8
- `unresolved_error` 有 error_snapshot 可锚定，但无 resolution GT，PRM 只评估诊断
- `new_tool` / `efficiency_anomaly` 缺乏硬 GT，仅做格式检查 + 弱 PRM

### C.4 R_prm Prompt 设计

以 session_context 为证据，与 OpenClaw-RL 的 next-state 思路同构：
系统提示:
你是一个分析质量评估器。你会看到一段 Agent 会话日志和一份结构化错误分析。
请判断：这份分析是否准确反映了会话中实际发生的事？
评分：\boxed{1}（准确）/ \boxed{-1}（矛盾/错误）/ \boxed{0}（模糊）

用户提示:
会话日志
{session_context}
错误分析
{MaterializeResult JSON}
请先分析会话中实际发生了什么，再判断这份分析是否准确。

**与 OpenClaw-RL PRM 的对照**：

| | OpenClaw-RL (PRM 部分) | SAVR (Ours) |
|---|---|---|
| **评估对象** | (Agent 回复, next-state) | (MaterializeResult, session_context) |
| **评估问题** | "回复是否帮助了用户？" | "分析是否准确描述了会话中发生的事？" |
| **证据来源** | 单条 next-state | 完整会话轨迹 |
| **证据丰富度** | 低（当前步状态快照） | 高（所有 tool calls + outputs + errors） |

### C.5 SAVR 与 OPD / Skill-Driven Adaptation 的关系

**与 OpenClaw-RL OPD 的对比**：

OPD 解决的核心问题是"标量 PRM reward 丢失了 token 级信息"——+1/-1 只告诉模型整体对/错，不告诉哪些 token 该改。OPD 通过 hindsight 蒸馏恢复这些信息。

SAVR 面临类似但不完全相同的问题。差异在于：
- **OPD 适用场景**：Agent 实时交互，next-state 经常包含 instructive signals（用户纠正、错误提示），OPD 可以从中提取 hint 构建 teacher 分布
- **SAVR 适用场景**：离线错误分析生成，输入是完整的 Signal + Session，输出是结构化 JSON。user_feedback 作为 hint 的价值已经通过 R_keyword_anchor 和 R_align 捕获——这两个组分本质上在做 OPD 的"锚定"工作，只是在 reward 层面而非 token 层面

**OPD 作为潜在增强方向**：对于 user_correction 信号，可以考虑将 user_feedback 作为 hint 注入增强上下文，让模型在"看到正确答案后"重新生成，以此构建 token 级 teacher 分布。这会比当前的标量 SAVR 提供更精细的监督。但代价是：(1) 每个训练样本需要额外一次 forward pass 生成 teacher 分布；(2) 仅对有 user_feedback 的 ~15% 数据有效。可列为后续 ablation 实验。

**与 MetaClaw Skill-Driven Adaptation 的对比**：

MetaClaw 的 Skill-Driven Adaptation（LLM Evolver → 失败分析 → 技能合成 → 注入）与我们的 Layer 2 pipeline（Signal → Materialize → Taxonomy → Skill → Sync）高度同构：

| | MetaClaw Skill-Driven | 我们的 Layer 2 |
|---|---|---|
| **触发** | 失败轨迹 | Signal（5 种类型） |
| **分析** | LLM Evolver | Materializer (Qwen3.5-32B) |
| **知识提取** | 识别失败模式 | Experience → Taxonomy |
| **注入方式** | 技能指令 → 技能库 | SKILL.md + taxonomy index → Agent 配置 |
| **生效速度** | 即时（非梯度） | 即时（非梯度） |

关键区别：
- MetaClaw 的 Skill 合成和 RL 训练是两个并列的学习通道，互相增强
- 我们的 Layer 2 是主要学习通道，RL 训练是用来**提升 Layer 2 中 Materializer 的质量**——RL 不直接优化 Agent 行为，而是优化"分析错误的能力"
- 这意味着我们的 RL 和非梯度学习是**串联关系**（RL 提升 materialize → materialize 产出更好的 Skill），而 MetaClaw 是**并联关系**（RL 和 Skill 各自独立地改善 Agent）

### C.6 数据量估算（v2 更新）

假设总数据量 N 条信号：

| 信号类型 | 占比 | 权重 | 等效有效数据 |
|---------|------|------|-------------|
| user_correction | ~15% | × 1.0 | 15% |
| **self_recovery** | **~45%** | **× 0.8** | **36%（最大贡献者）** |
| unresolved_error | ~20% | × 0.5 | 10% |
| new_tool + efficiency | ~20% | × 0.2 | 4% |
| **合计** | 100% | | **~65% 等效有效数据** |

self_recovery 是最大贡献者——v2 将其权重从 0.5 上调到 0.8，因为 session_context 提供了强验证能力。

### C.6 训练过程举例

**场景**: 用户说"帮我删除 ~/old-project"，Agent 直接 rm -rf，用户纠正"以后先列出内容让我确认"

**Step 1 — 构造 Prompt**:
Signal: user_correction, user_feedback="以后删除前先 ls 列出内容，让我确认后再删"
Session: [user: "帮我删除...", assistant: rm -rf, tool: removed, user: "以后先列出..."]
→ 拼接为 prompt

**Step 2 — 生成 N=4 候选 MaterializeResult**:
候选1: root_cause="执行 rm -rf 时未先 ls 列出内容" resolution="先 ls → 确认 → 再 rm"
候选2: root_cause="直接删除未确认"              resolution="删除前应确认"
候选3: root_cause="操作流程不规范"              resolution="建议用 trash-cli"
候选4: (JSON 格式错误)

**Step 3 — SAVR 打分** (user_correction → 全组分激活):
候选1: F=0.20 + K=0.12 + E=0.22 + A=0.32 + P=0.20  = 1.06 × 1.0 = 1.06
候选2: F=0.20 + K=0.05 + E=0.05 + A=0.18 + P=0.00  = 0.48 × 1.0 = 0.48
候选3: F=0.20 + K=0.00 + E=0.02 + A=0.06 + P=-0.20 = 0.08 × 1.0 = 0.08
候选4: F=-1.0 + (其余跳过)                           = -1.0  × 1.0 = -1.0

**Step 4 — GRPO 标准化 + 策略更新**:
优势: A₁ = +1.05, A₂ = +0.04, A₃ = -0.49, A₄ = -0.60
→ 候选1 类型的输出概率上升，候选4 类型下降

---

**self_recovery 信号举例**（v2 新增的高权重场景）:

**场景**: Agent 执行 git push 失败，自行检测到 remote rejected，改用 git pull --rebase 后重新 push 成功。

**Session Context** 提供了完整证据链：
```jsonl
{"role": "assistant", "tool_calls": [{"name": "bash", "args": "git push origin main"}]}
{"role": "tool", "content": "error: failed to push, remote rejected (non-fast-forward)"}
{"role": "assistant", "tool_calls": [{"name": "bash", "args": "git pull --rebase origin main"}]}
{"role": "tool", "content": "Successfully rebased and updated refs/heads/main"}
{"role": "assistant", "tool_calls": [{"name": "bash", "args": "git push origin main"}]}
{"role": "tool", "content": "main -> main (push successful)"}

SAVR 打分（self_recovery → format + entity_anchor + prm）:
- R_entity_anchor: root_cause 引用了 "remote rejected (non-fast-forward)" → 高分
- R_prm: session 轨迹清楚地显示了 push → 失败 → pull rebase → 成功 → 重新 push，LLM judge 可以验证分析是否准确

---

Part D：四方法对比总表

维度
AReaL
OpenClaw-RL
MetaClaw
SAVR (Ours)
任务域
数学推理
Agent 策略
Agent 策略
错误分析生成
算法
PPO (GRPO 风格)
GRPO + OPD
GRPO (IS)
GRPO
训练数据
GSM8K（人工标注）
实时轨迹
实时轨迹
Signal + Session
GT 来源
标准答案
next-state
无
user_feedback + error_snapshot + session_context
GT 质量
完美
中等
低
分层（按信号类型）
RL 奖励
math_verify (0/1)
PRM: LLM judge (±1)
PRM: 外部 LLM (±1)
SAVR 五组分（自适应激活）
额外优化机制
无
OPD: hindsight → token 级优势
Skill-Driven: LLM Evolver → 技能注入
Layer 2 pipeline (非梯度即时学习)
优势函数粒度
候选级
token 级（PRM + OPD）
候选级
候选级（5 组分标量）
RL 与非梯度关系
仅 RL
仅 RL (OPD 在 RL 内部)
并联（RL ∥ Skill 各自改善 Agent）
串联（RL → 提升 materialize → 更好的 Skill）
PRM 实现
无
zero-shot 同模型
zero-shot 外部 LLM
zero-shot LLM + session_context 证据
PRM 证据
N/A
单步 next-state
无客观证据
完整会话轨迹
KL 控制
0.0~0.1
0.01
OPD KL
0.01
核心创新
大规模 RL infra
OPD: 从标量恢复 token 级监督
双循环: RL + Skill 互补
信号自适应多源验证 + RL-pipeline 串联闭环
自进化闭环
无
无
有（RL ↔ Skill 互补）
有（RL 优化分析 → 更好 Skill → Agent 改善）
硬件
多节点集群
8×A100/H100
8×A100
8×A100, LoRA r=32

关键差异总结

1. 优势函数粒度：OpenClaw-RL 的 OPD 通过 hindsight 蒸馏提供 token 级监督，是四个方法中最精细的。AReaL / MetaClaw / SAVR 都是候选级标量奖励。SAVR 通过拆分为 5 个可独立验证的标量组分来缓解信息丢失，但粒度仍不及 OPD。OPD 思路可作为 SAVR 后续增强方向（见 C.5）。
2. RL 与非梯度学习的耦合方式：MetaClaw 的 RL 和 Skill-Driven 是并联关系——各自独立地改善 Agent。我们的 RL 和 Layer 2 是串联关系——RL 不直接优化 Agent 行为，而是优化 materialize 质量，间接通过更好的 Skill 改善 Agent。串联架构的优势在于 RL 训练目标更聚焦（结构化分析生成 vs 通用 Agent 策略），劣势在于改善链路更长。
3. GT 分层策略：AReaL 有完美 GT，OpenClaw-RL / MetaClaw 对所有数据统一处理。SAVR 按信号类型分层——有硬 GT 的给高权重，无硬 GT 的降权但不丢弃。
4. PRM 证据丰富度：OpenClaw-RL 的 PRM 用单步 next-state（但 OPD 从 instructive signal 中提取更多信息），MetaClaw 无客观证据。SAVR 用完整 session_context，作为 PRM 证据信息量最大。
5. 自进化闭环：MetaClaw 通过 RL ↔ Skill 的双循环互补实现持续进化。我们通过 RL 提升 materialize → 更好的 Experience/Taxonomy → 更好的 Skill → Agent 改善实现串联闭环。