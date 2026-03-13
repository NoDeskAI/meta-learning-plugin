# DeskClaw Meta-Learning (灵敏自进化学习系统)

> **核心使命：** 实现从“被动执行”到“主动进化”的跃迁，通过 Action-Think-Refine 循环持续优化 Agent 性能。

本项目是 OpenClaw 灵敏 Agent 的元学习 (Meta-Learning) 核心实现。它采用三层自进化架构，能够从对话轨迹中自动识别、提炼并固化知识。

## 🌟 三层架构 (Layered Architecture)

### Layer 1: 在线微学习 (Online Micro-learning)

#### QuickThink — 两级触发机制 (Two-Level Trigger)

`QuickThinkIndex` 在任务执行前进行实时风险评估，采用两级触发架构：

**Level 1 — 多维信号扫描**：`evaluate()` 对 `TaskContext` 并行执行四项独立检测，延迟上限 50ms：

| 检测器 | 触发条件 | 信号标识 |
|---|---|---|
| `_check_keyword_match` | 任务文本命中 `ErrorTaxonomy` 关键词索引（历史错误模式） | `keyword_taxonomy_hit` |
| `_check_irreversible_ops` | 检测到不可逆操作（`rm -rf`、`DROP TABLE`、`force push` 等） | `irreversible_operation` |
| `_check_recent_failures` | 匹配近期已注册的失败签名 | `recent_failure_pattern` |
| `_check_new_tools` | 检测到首次使用的工具 | `new_tool_usage` |

**Level 2 — 复合风险定级**：基于 Level 1 收集到的信号集合，`_assess_risk_level()` 进行风险定级：
- **`high`** — 检测到不可逆操作（立即阻断）
- **`medium`** — 2+ 信号同时命中（交叉风险）
- **`low`** — 单一信号命中
- **`none`** — 无命中

最终输出 `QuickThinkResult(hit, matched_signals, matched_taxonomy_entries, risk_level)`，供下游决策使用。

#### SignalCapture — 优先级信号捕获 (Priority-Based Signal Capture)

`SignalCapture` 在任务执行过程中实时捕获学习信号，采用优先级触发 + 结构化快照模式：

**触发判定**：`_determine_trigger()` 按优先级顺序逐级评估，首个命中即返回：

1. **`ERROR_RECOVERY`**（最高优先级）— 存在已修复的错误 (`errors_fixed=True` 且 `errors_encountered` 非空)
2. **`USER_CORRECTION`** — 用户主动纠正 (`user_corrections` 非空)
3. **`NEW_TOOL`** — 使用了新工具 (`new_tools` 非空)
4. **`EFFICIENCY_ANOMALY`**（最低优先级）— 步数超过平均值的 `threshold` 倍（默认 2.0×，即 step_count > 10）

**信号构建**：触发后构建 `Signal` 对象，包含：
- 自增信号 ID（通过 `next_signal_id` 原子生成）
- 关键词自动提取：优先从错误消息中提取大写/含数字 token，回退到任务描述
- 上下文快照：错误快照（≤500 字符）、解决方案快照（≤200 字符）、用户反馈

**持久化**：信号以 YAML 格式写入 `signal_buffer_path`，供 Layer 2 近线整合阶段异步消费。

### Layer 2: 近线整合 (Near-line Consolidation)

`Layer2Orchestrator` 按序执行四阶段流水线（Materialize → Consolidate → Taxonomy → Skill Evolve），触发条件：待处理信号数 ≥ `min_pending_signals`，或距上次运行超过 `max_hours_since_last` 小时且存在待处理信号。

#### Materialize — 信号物化 (Signal → Experience)

`Materializer.materialize_all_pending()` 消费 Layer 1 写入 `signal_buffer_path` 的 YAML 信号，将每条轻量级 `Signal` 物化为结构化的 `Experience`：

1. **信号加载**：通过 `list_pending_signals()` 读取所有 `processed=False` 的待处理信号
2. **上下文补全**：若 `signal.session_id` 有效（非 `"unknown"`），调用 `read_session_context()` 加载完整会话上下文
3. **LLM 结构化提取**：`llm.materialize_signal(signal, session_context)` 返回 `MaterializeResult`，包含：
   - `scene`（场景描述）、`failure_signature`（错误签名）、`root_cause`（根因分析）
   - `resolution`（解决方案）、`meta_insight`（元认知洞察）、`task_type`（任务分类）
4. **Experience 构建**：分配自增 ID（`next_experience_id`），初始置信度取自 `config.layer2.materialize.initial_confidence`，`verification_count=1`
5. **持久化 & 标记**：Experience 写入 `experience_pool_path`，原始 Signal 标记为 `processed`

#### Consolidate — 语义聚类 (BFS Clustering)

`Consolidator.consolidate()` 对 Experience 池执行基于语义相似度的图聚类：

**Step 1 — 置信度衰减**：若 `config.confidence.decay_enabled`，按时间指数衰减：
```
decayed_confidence = original_confidence × decay_base ^ days_old
```
低于 `prune_threshold` 的 Experience 直接剔除，不参与后续聚类。

**Step 2 — 按 TaskType 分组**：将有效 Experience 按 `task_type`（`coding` / `devops` / `debugging` 等）分桶，各组独立聚类。

**Step 3 — 组内语义聚类**（`_cluster_within_group`）：
1. **构建相似度图**：对组内所有 Experience 两两组合（`combinations(experiences, 2)`），调用 `llm.judge_same_class(exp_a, exp_b)` 判定语义等价性。若 `same_class=True`，双向添加邻接边
2. **BFS 连通分量发现**：遍历邻接表，对每个未访问节点执行 BFS（`_bfs_component`），提取完整连通分量
3. **聚类过滤**：仅保留 `|component| ≥ 2` 的连通分量，生成 `ExperienceCluster`（代表性签名取自排序后首个 Experience 的 `failure_signature`）

**Taxonomy 就绪判定**：`get_clusters_ready_for_taxonomy()` 筛选满足 `len(experience_ids) ≥ min_cluster_size_for_taxonomy` 且 `promoted_to_taxonomy is None` 的聚类。

#### Taxonomy — 错误分类树构建

`TaxonomyBuilder.build_from_clusters()` 将成熟聚类提炼为 `ErrorTaxonomy` 树形结构：

**树形结构**：`ErrorTaxonomy.taxonomy` 为三层嵌套 — `domain → subdomain → list[TaxonomyEntry]`：
- **domain**：取自 `cluster.task_type.value`（如 `"coding"`、`"devops"`）
- **subdomain**：由 `_infer_subdomain()` 基于 `failure_signature` 关键词推断（`"typescript"` / `"python"` / `"docker"` / `"git"` / `"general"`）

**单条 TaxonomyEntry 生成流程**：
1. 加载聚类内所有 Experience（从 `experience_pool_path` 按 ID 匹配 `*.yaml`）
2. `llm.extract_taxonomy(experiences)` 提取：`name`（分类名）、`trigger`（触发条件）、`fix_sop`（修复 SOP）、`prevention`（预防措施）、`keywords`（索引关键词）
3. **ID 生成**：`next_taxonomy_id(taxonomy, prefix)` 以 `"{domain前3字符}-{subdomain前3字符}"` 为前缀
4. **置信度计算**：`avg(experiences.confidence) + min(count × 0.05, 0.2)`，上限 1.0
5. **晋升标记**：Experience 标记 `promoted_to = taxonomy_id`，Cluster 标记 `promoted_to_taxonomy = taxonomy_id`，防止重复处理

生成的关键词通过 `ErrorTaxonomy.all_keywords()` 建立反向索引，供 Layer 1 QuickThink 的 `_check_keyword_match` 实时查询。

#### Skill Evolve — 技能卡片进化

`SkillEvolver.evolve_from_taxonomy()` 将新 TaxonomyEntry 转化为可执行的 Skill 卡片（`SKILL.md`）：

**匹配策略**（`_find_matching_skill`）：按优先级在 `skills_path` 目录下检索现有 Skill：
1. 内容包含 `"Taxonomy ID: {entry.id}"` → 精确匹配
2. 目录名 == `entry.name` 规范化形式 → 名称匹配
3. 目录名与 `entry.keywords` 有交集 → 关键词匹配

**更新决策**（`llm.evaluate_skill_update`）：返回 `SkillEvolveResult`，包含 6 种操作：

| Action | 触发条件 | 行为 |
|---|---|---|
| `CREATE` | 无匹配 + `confidence ≥ 0.8` + `source_exps ≥ 5` | 创建 `{skill_name}/SKILL.md` |
| `APPEND` | 已有匹配 Skill | 追加新知识段落到 `SKILL.md` 末尾 |
| `REPLACE` | 已有匹配 Skill（内容需重写） | 覆写 `SKILL.md` 全部内容 |
| `MERGE` / `SPLIT` | LLM 判定需合并/拆分 | （预留，由 LLM 驱动） |
| `NONE` | 证据不足或无需更新 | 跳过 |

### Layer 3: 离线深度学习 (Offline Deep Learning)

`Layer3Orchestrator` 按序执行三阶段离线流水线（Cross-Task Mining → Capability Gap Detection → Memory Optimization），产出 `Layer3Result` 并持久化，实现系统级自进化闭环。

#### CrossTaskMiner — 跨领域根因发现 (Cross-Domain Root Cause Discovery)

`CrossTaskMiner.mine_patterns()` 从 Experience 池中挖掘跨任务类型的共享根因模式：

**Step 1 — 前置门控**：Experience 总数 < `min_experiences_for_cross_task`（默认 3）时直接跳过，避免数据不足时的噪声分析。

**Step 2 — 跨类型分组**（`_build_cross_type_groups`）：
1. 将所有 Experience 按 `task_type`（`coding` / `devops` / `debugging` 等）分桶
2. 若桶数 < 2（即只有单一任务类型），直接返回空——跨域分析至少需要两种任务类型
3. 对所有任务类型两两组合（`combinations(task_types, 2)`），对每对类型执行 `_find_shared_root_cause_pairs`

**Step 3 — 根因重叠检测**（`_root_causes_overlap`）：
- 将两条 Experience 的 `root_cause` 文本分词，过滤长度 ≤ 3 的短词
- 计算词集交集占比：`|overlap| / min(|words_a|, |words_b|)`
- **阈值 ≥ 0.3** 判定为根因重叠
- 每对类型中，重叠配对去重后合并，仅保留 **≥ 2 条** Experience 的组

**Step 4 — LLM 模式提炼**：将每个跨类型组送入 `llm.analyze_cross_task_patterns()`：
- 置信度公式：`min(group_size × 0.15, 0.9)`（组内 Experience 越多，置信度越高，上限 0.9）
- 仅保留 `confidence ≥ min_pattern_confidence`（默认 0.6，即至少 4 条 Experience）的模式

**输出 `CrossTaskPattern`**：包含 `pattern_id`（`ctp-001` 格式）、`description`、`shared_root_cause`、`affected_task_types`（涉及的任务类型列表）、`source_experience_ids`、`confidence`、`meta_strategy`（面向下游的预防策略）。

#### NewCapabilityDetector — 盲区识别与补全建议 (Capability Gap Detection)

`NewCapabilityDetector.detect_gaps()` 通过分析"落选"Experience（未被 Taxonomy 吸收的孤立经验）识别系统的认知盲区：

**Step 1 — 盲区定位**：
1. 加载全量 Experience 池与当前 `ErrorTaxonomy`
2. 筛选 `promoted_to is None` 的 Experience——即未被任何 Taxonomy 条目吸收的"孤儿经验"
3. 这些孤儿经验代表了 **Taxonomy 未能覆盖的知识空白**

**Step 2 — LLM 盲区分析**：将孤儿经验和现有 Taxonomy 关键词一并送入 `llm.analyze_capability_gaps()`：
- 按 `task_type` 统计孤儿经验出现频次
- 仅对频次 **≥ 3** 的任务类型生成 Gap（避免偶发噪声）
- 优先级公式：`min(count × 0.2, 1.0)`

**Step 3 — Gap 类型推断**（`_infer_gap_type`）：
| Gap 类型 | 判定条件 |
|---|---|
| `failure` | 匹配的 Experience 中存在 `failure_signature` |
| `efficiency` | 无失败签名，但平均置信度 < 0.5 |
| `frequency` | 无失败、置信度正常——纯粹的高频未覆盖场景 |

**输出 `CapabilityGap`**：包含 `gap_id`（`gap-001` 格式）、`gap_type`、`description`、`evidence_ids`、`suggested_action`（建议创建的 Skill 方向）、`priority`。

#### MemoryArchitect — 长期记忆优化 (Memory Architecture Optimization)

`MemoryArchitect.optimize()` 对 Experience 池执行置信度分层，生成记忆架构调整建议：

- **高置信度**（`≥ confidence.promote_threshold`）→ `EXTRACT`：提升至长期记忆（MEMORY.md），内容取自 `meta_insight`
- **低置信度**（`< layer3.prune_confidence_threshold`，默认 0.5）→ `PRUNE`：标记为待清理候选
- 此外支持 `CONSOLIDATE` 操作（合并冗余记忆）

#### 离线自进化闭环

三阶段串行执行构成完整的自进化反馈闭环：

```
Experience Pool (Layer 2 产出)
    │
    ├─→ [CrossTaskMiner]    跨类型根因挖掘 → CrossTaskPattern（跨域预防策略）
    │                         ↓ 反哺 Layer 1 风险评估
    ├─→ [CapabilityDetector] 孤儿经验聚类 → CapabilityGap（盲区补全建议）
    │                         ↓ 驱动 Taxonomy/Skill 扩展
    └─→ [MemoryArchitect]   置信度分层 → MemoryRecommendation（记忆提升/裁剪）
                              ↓ 优化知识存储效率
```

Layer 3 的产出通过 `save_layer3_result()` 持久化为 `Layer3Result`，同时记录 `last_l3_run` 时间戳至 `layer3_state.json`，供调度系统判定下次执行窗口。整体形成 **Layer 1 捕获 → Layer 2 整合 → Layer 3 深挖** 的三层递进式自进化体系。

## 🚀 迭代状态 (Iteration Progress)

本项目已通过 **Ralph Loop** 完成两轮深度自迭代模拟：
- **Iteration 1**: 修复了关键词提取噪声、聚类算法边界问题、ID 生成竞态条件。
- **Iteration 2**: 实现了 Layer 3 核心组件及跨任务关联分析逻辑。
- **集成测试结果**: Quick Think 命中率从 **1/19** 提升至 **17/19**（精准命中）。

## 🛠️ 安装与测试

### 环境依赖
- Python 3.12+
- Pytest (用于单元测试)
- PyYAML

### 运行集成测试
```bash
# 运行全链路集成测试（包含三层流水线模拟）
python3 scripts/main_test.py
```

### 运行单元测试
```bash
python3 -m pytest tests/
```

## 📂 目录结构
- `src/meta_learning/`: 核心源码
- `schemas/`: 数据模型定义 (YAML)
- `tests/`: 单元测试与 Mock 资源
- `scripts/`: 集成测试脚本

---
Developed by **灵敏 (OpenClaw Agent)**
