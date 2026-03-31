# Meta-Learning 问题定义：Agent「学不会」的困境

> 本文档与 [claw失忆/问题定义.md](claw失忆/问题定义.md) 形成对照。Claw 失忆文档定义的是 Agent **记不住**的问题（记忆存储/检索/注入链路缺陷）；本文档定义的是 Agent **学不会**的问题（从交互错误中提炼可迁移策略、结构化沉淀并前置应用的链路缺陷）。两类问题互补但问题域不同。

---

## 一、问题现象：Agent 在策略层面无法从经验中学习

Nanobot（DeskClaw）用户在日常使用中会遭遇以下问题，这些问题**不是**记忆丢失，而是 Agent 缺乏从过去的错误与纠正中提炼规律并应用于未来的能力：

- **同类策略错误反复犯**：Agent 在任务 A 中因"未先验证就执行不可逆操作"而失败，用户纠正后，Agent 在任务 B 中遇到类似场景时仍犯同样的策略错误。即使 Agent "记得"任务 A 的对话，它也无法将纠正抽象为"遇到不可逆操作应先验证"的通用规则。

- **失败后知识不成体系**：用户反复纠正 Agent 的错误（如"批量操作不要遗漏项目""先检查权限再调用工具"），但这些纠正被记录为散乱的记忆片段或对话历史，从未被整合为结构化的「遇到 X 场景 → 应该采取 Y 策略」可执行规程。

- **仅事后 RAG/记忆搜索无法形成可执行规程**：即使 Agent 通过 memory_search 检索到历史失败记录，它也缺乏将检索结果转化为行动指导的机制。从"上次这么做失败了"到"这次应该怎么做"之间存在断层。

- **风险操作无前置拦截**：Agent 执行 `rm -rf`、`force push`、`drop table` 等不可逆操作前，没有任何基于历史失败模式的预警机制。所有学习都是事后的——先犯错，再补救。

- **纠正信号稀薄或延迟**：生产环境中 Agent 仅能获得粗粒度反馈（任务成功/失败），缺乏策略级别的诊断依据。用户纠正是最有价值的信号，但频率低且表述随意，难以自动化利用。

- **多用户/多场景下经验无法复用与治理**：用户 A 踩过的坑，用户 B 依然会踩。不同业务场景（前端开发 vs DevOps vs 客服）的最佳实践各不相同，但行业级经验无法跨用户沉淀、聚合与分发。

### 与「失忆」问题的对照

| 维度 | Claw 失忆（记不住） | Meta-Learning（学不会） |
|------|-------------------|----------------------|
| 核心缺陷 | 记忆存储/检索/注入链路 | 经验提炼/结构化/前置应用链路 |
| 典型现象 | 隔夜忘记昨天教的内容 | 纠正后下次同类任务仍犯同样策略错误 |
| 知识形态 | 原始对话片段、MEMORY.md | 分类化的错误模式、可执行的策略规程 |
| 时机 | 事后回忆 | 事前拦截 + 事后学习 |
| 作用范围 | 单用户单会话 | 可跨用户、跨场景、跨任务迁移 |

---

## 二、根本原因：从信号到策略的全链路断裂

以下根因分析映射到 [`meta_learning_chain_fusion.drawio`](meta_learning_chain_fusion.drawio)（Nobot × Meta-Learning 融合架构）中的具体模块与 [`src/meta_learning/`](src/meta_learning/) 代码实现。

### （一）信号与诊断粒度不足

**问题**：当 Agent 任务失败时，系统能获取的信号往往仅是粗粒度的成功/失败（reward=0/1）。LLM 在物化（materialize）阶段依赖这些信号来提取策略级教训，但信号太粗糙时无法准确定位"哪一步做错了"以及"应该采取什么正确策略"。

**代码映射**：

- [`TaskContext`](src/meta_learning/shared/models.py) 定义了 Agent 可观测数据的边界：`task_description`、`tools_used`、`errors_encountered`、`user_corrections`、`step_count`。其中 `user_corrections` 是最高价值信号，但生产环境中频率最低。
- [`SignalCapture`](src/meta_learning/layer1/signal_capture.py) 按优先级捕获 4 种触发条件：`error_recovery` > `user_correction` > `new_tool` > `efficiency_anomaly`。触发是 first-match-wins——如果存在 `errors_encountered`，即使用户同时提供了纠正，也只会产生 `error_recovery` 信号，`user_correction` 被吞没。（注：Layer 2 对 `USER_CORRECTION` 类信号有 fast-track 路径——跳过聚类直接生成 taxonomy entry，但前提是信号被捕获为该类型。）
- 物化阶段（Layer 2 `Materialize`）需要读取会话 JSONL 文件获取完整上下文。若 session 文件不存在或被截断，整个物化会失败（raise 异常），阻塞 Layer 2 流水线。

**实验证据**：在 GDPVal 和 tau-bench airline:35 实验中，仅依赖 reward=0 信号时，LLM 物化产出的 `failure_signature` 和 `root_cause` 质量显著低于有 NL assertion 反馈（相当于 QA 监督信号）时的产出。

### （二）学习时效与冷启动

**问题**：Layer 2 流水线有门控条件（默认 ≥5 条 pending signals 或 ≥24h 且有 pending），在实际使用中用户每天可能只触发 1-2 个信号，需要 3-5 天才能首次触发 Layer 2。在此期间系统没有任何学习能力。

**代码映射**：

- [`Layer2Orchestrator.should_trigger()`](src/meta_learning/layer2/orchestrator.py) 实现了触发逻辑：`min_pending_signals`（默认 5）或距上次运行超过 `max_hours_since_last`（默认 24h）。
- 系统没有预置种子 taxonomy（`ErrorTaxonomy` 初始为空），QuickThink 在 taxonomy 为空时无关键词可匹配、无向量可比对，风险评估功能完全失效。

**对比**：AutoSkill 从第一次对话即开始提取 skill；MetaClaw 的 skill-driven adaptation 无需等待累积。

### （三）知识结构化 vs 复杂度的未证命题

**问题**：系统采用 `ErrorTaxonomy`（domain → subdomain → TaxonomyEntry）三级层级结构来组织错误模式，并通过 QuickThink 混合检索（关键词优先 + 向量语义回退）在执行前匹配风险。这比 flat skill list + embedding 检索复杂得多，但效果优势未经 A/B 验证。

**代码映射**：

- [`ErrorTaxonomy`](src/meta_learning/shared/models.py)：`dict[str, dict[str, list[TaxonomyEntry]]]` — 第一层 key 为 domain，第二层 key 为 subdomain。
- [`QuickThinkIndex.evaluate()`](src/meta_learning/layer1/quick_think.py)：先做关键词子串匹配 → 未命中时调用 `_check_vector_match()`（需 `embedding_fn` 和 `vector_fallback_enabled`）→ cosine 相似度 ≥ 阈值的 top-k 条目。
- Taxonomy 的层级由 Layer 2 中 LLM `extract_taxonomy` 输出决定——LLM 输出是非确定性的，分类粒度不可控。

**风险**：taxonomy 增加了系统复杂度（更多中间状态文件、更多 LLM 调用），但它的结构化价值相比简单的 embedding flat 检索未被证明。在 B2B 行业沉淀场景下，按 domain 聚合的优势**理论上**更大，但需要验证。

### （四）闭环与度量缺失

**问题**：整个系统的隐含假设是「signal → experience → taxonomy → skill → QuickThink 命中 → Agent 表现变好」，但从头到尾没有度量这个因果链是否成立。

**代码映射**：

- `QuickThinkResult` 包含 `hit`、`matched_signals`、`risk_level`，但没有记录 Agent 是否采纳了风险提示、采纳后任务是否成功。
- `TaxonomyEntry` 有 `confidence` 和 `verification_count` 字段，但 `verification_count` 仅在 Layer 2 聚类时被设置，没有基于实际任务结果的衰减或强化机制。
- Layer 3（`CrossTaskMiner`、`NewCapabilityDetector`、`MemoryArchitect`）产出的结果保存为 `Layer3Result` 文件，但**不回流到 Layer 1/2**。Layer 3 的跨任务挖掘和盲区检测结果是死胡同——对 QuickThink 检索和 Taxonomy 更新没有任何影响。

**对比**：AutoSkill 有 A/B 对比实验（有 skill 注入 vs 无注入的任务成功率）；MetaAgent 在 GAIA/WebWalkerQA 上有 benchmark 评分；MetaClaw 报告了 +32% 准确率提升。

### （五）多用户愿景与单机实现的结构性鸿沟

**问题**：若将 meta-learning 作为公司产品安装到每个 Nanobot 客户端、并期望实现行业经验沉淀，则当前架构存在根本性缺失。

**代码映射**：

- `Signal` 模型无 `user_id` 字段——无法区分不同用户的信号。
- `TaxonomyEntry` 模型无 `source_scope`（personal / team / industry）字段——无法区分个人偏好与行业共识。
- 全部数据存储为本地 YAML 文件（`signal_buffer/*.yaml`、`experience_pool/*.yaml`、`error_taxonomy.yaml`），无并发写入保护、无事务性。
- 无信号上行通道（客户端 → 云端）、无 taxonomy 下行通道（云端 → 客户端）。
- 无隐私/匿名化/用户 consent 机制——`Signal` 中的 `task_summary`、`error_snapshot`、`user_corrections` 可能包含敏感业务数据。
- 无跨用户信号可信度验证——单个用户的纠正不一定正确，但当前所有 `user_correction` 被等权对待。
- 无 personal vs industry taxonomy 合并与冲突策略。

---

## 三、影响分析

1. **用户重复试错**：同类策略错误不收敛，Agent 的任务完成效率无法随使用时长提升。用户被迫在每次新任务中重复相同的纠正。

2. **组织知识资产为零**：每个 Agent 实例独立摸索，行业经验不积累。公司无法从用户基的集体智慧中构建竞争壁垒。

3. **风险不可控**：已知高风险模式（如不可逆操作、权限误用）无前置拦截机制。只能事后复盘，无法事前预防。

4. **度量不可报告**：无法向管理层或客户证明「学习系统确实提升了 Agent 表现」。缺乏 QuickThink 命中率、采纳率、命中后任务成功率等可报告指标。

5. **多用户产品化工程量被低估**：从本地单用户原型到支持多租户、行业沉淀、隐私合规的产品化系统，需要补建的基础设施（数据模型扩展、上行/下行通道、匿名化管道、冲突策略、分布式存储）远超当前代码规模。

6. **学习系统自身无法验证有效性**：Layer 3 输出不回流、无闭环度量、无 A/B 实验框架，导致无法判断系统的学习是否真的在改善 Agent 表现——有可能是在积累噪音而非知识。
