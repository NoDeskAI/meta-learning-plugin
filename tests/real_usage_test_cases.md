# Meta-Learning 真实场景测试用例

> 前置条件：nanobot 已重启，meta-learning MCP server 已加载新代码。
> 测试方法：通过 DeskClaw 客户端（Telegram / CLI）与 agent 对话。

---

## TC-01: 单次用户纠正触发即时学习

**目标**：验证用户第一次纠正就能走通完整学习链路。

**步骤**：

1. 向 DeskClaw 发送：`帮我在 ~/projects 下创建一个 Python 项目`
2. DeskClaw 执行后，纠正它：`不对，项目应该创建在 ~/workspace 下，不是 ~/projects`
3. 观察 DeskClaw 是否调用 `capture_signal`（查看对话中的工具调用提示）
4. 观察 `capture_signal` 返回值是否包含 `[Action Required]`
5. 观察 DeskClaw 是否随后调用 `run_layer2`

**预期**：

- `capture_signal` 被调用，`trigger=user_correction`
- 返回 `[Action Required] Layer 2 trigger conditions met (1 pending signal(s))`
- DeskClaw 调用 `run_layer2`，pipeline 执行成功
- `signal_buffer/` 中信号文件被标记为 processed

**验证**：

```bash
# 检查 signal_buffer 中是否有新信号
ls ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/

# 检查 layer2_state.json 是否更新了 last_run
cat ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/layer2_state.json

# 检查 SKILL.md 是否包含新规则
cat ~/.deskclaw/nanobot/workspace/skills/meta-learning/SKILL.md
```

---

## TC-02: 纠正信号优先级（errors + user_corrections 共存）

**目标**：验证用户纠正在 errors 同时存在时不被吞没。

**步骤**：

1. 向 DeskClaw 发送一个会出错的任务：`帮我运行 python nonexistent_script.py`
2. DeskClaw 遇到错误后尝试修复
3. 在它修复过程中纠正：`不用修复了，那个脚本不存在，你应该先用 ls 确认文件再执行`
4. 观察 `capture_signal` 的 `trigger` 值

**预期**：

- `trigger=user_correction`
- 信号文件中 `trigger_reason: user_correction`
- `[Action Required]` 出现在返回值中

**验证**：

```bash
# 查看最新信号文件的 trigger_reason
cat ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/sig-*.yaml | grep trigger_reason
```

---

## TC-03: 学习成果在下次任务中生效

**目标**：验证 Layer 2 产出的规则通过 SKILL.md 注入到 agent 的 system prompt。

**步骤**：

1. 先完成 TC-01（确保一条规则已学到）
2. 发送 `/new` 开始新会话
3. 向 DeskClaw 发送一个类似任务：`帮我创建一个新的 Node.js 项目`
4. 观察 DeskClaw 是否主动确认目录，或提到"先确认目录"

**预期**：

- DeskClaw 在执行前询问 "在哪个目录创建？" 或类似确认行为
- 如果调用了 `quick_think`，结果中应包含相关规则匹配

**验证**：

```bash
# 确认 SKILL.md 中包含学到的规则
grep -i "directory\|目录\|workspace" ~/.deskclaw/nanobot/workspace/skills/meta-learning/SKILL.md
```

---

## TC-04: quick_think 匹配已学规则

**目标**：验证学到的 taxonomy entry 能被 `quick_think` 检索命中。

**步骤**：

1. 确保 TC-01 的学习已完成（taxonomy 中有相关条目）
2. 向 DeskClaw 发送：`帮我删除 ~/old-project 文件夹`
3. 观察 DeskClaw 是否在执行前调用 `quick_think`
4. `quick_think` 是否返回了风险提示

**预期**：

- `quick_think` 被调用
- 至少匹配到 "irreversible operation" 内置规则
- 如果 taxonomy 中有相关条目，也应命中

**验证**：

```bash
# 检查 error_taxonomy.yaml 中有条目
cat ~/.deskclaw/nanobot/workspace/meta-learning-data/error_taxonomy.yaml | head -30
```

---

## TC-05: 非纠正信号的阈值触发（>=2 条 error_recovery）

**目标**：验证 2 条 error_recovery 信号能触发 Layer 2。

**步骤**：

1. 发送 `/new` 开始新会话
2. 让 DeskClaw 做一个会出错的任务（不要纠正它，让它自己修复）
   - 例如：`帮我编辑文件 /tmp/test_nonexist.txt，把第一行改成 hello`
3. 任务结束后，观察是否有 `capture_signal`（trigger=error_recovery）
4. 重复第 2-3 步，制造第二次错误修复
5. 第二次 `capture_signal` 后观察是否出现 `[Action Required]`

**预期**：

- 第 1 次：`capture_signal` 成功，无 `[Action Required]`（仅 1 条）
- 第 2 次：`capture_signal` 成功，**出现** `[Action Required]`（>= 2 条）
- DeskClaw 调用 `run_layer2`

---

## TC-06: Heartbeat 兜底触发

**目标**：验证 agent 忘记调用 `run_layer2` 时，Heartbeat 能补偿执行。

**步骤**：

1. 手动写入一条 USER_CORRECTION 信号（模拟 agent 忘记响应提示）：
   ```bash
   cat > ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/sig-manual-test.yaml << 'EOF'
   signal_id: sig-manual-test
   timestamp: '2026-03-31T10:00:00'
   session_id: unknown
   trigger_reason: user_correction
   keywords: [test, manual]
   task_summary: Manual test signal
   user_feedback: This is a test correction
   step_count: 3
   EOF
   ```
2. 等待 30 分钟（Heartbeat 间隔），或检查 nanobot 日志中 Heartbeat tick
3. 观察 Heartbeat 是否触发了 `run_layer2`

**预期**：

- Heartbeat 读取 HEARTBEAT.md -> LLM 判断有任务 -> 执行 `run_layer2`
- 手动信号被处理

**验证**：

```bash
# 检查信号是否被 processed（文件中会有 processed 标记或被移动）
ls ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/

# 检查 nanobot 日志中的 heartbeat 相关输出
# （日志位置取决于 nanobot 配置）
```

---

## TC-07: 跨会话学习迁移

**目标**：验证在会话 A 中学到的规则在会话 B 中生效。

**步骤**：

1. **会话 A**：让 DeskClaw 做一个任务，然后纠正：`你不应该直接修改 .env 文件，应该先备份`
2. 等待 `capture_signal` + `run_layer2` 完成
3. 发送 `/new` 开始新会话
4. **会话 B**：向 DeskClaw 发送：`帮我修改 .env 文件，添加一个新的环境变量`
5. 观察 DeskClaw 是否在修改前提到备份

**预期**：

- 会话 B 中 DeskClaw 的行为体现了会话 A 的学习成果
- SKILL.md 中包含关于备份的规则

---

## TC-08: capture_signal 未被调用时的降级

**目标**：验证 agent 在某些场景下未调用 `capture_signal` 的行为。

**步骤**：

1. 和 DeskClaw 进行一次**完全正常**的对话（无错误、无纠正）：
   - `现在几点了？` 或 `帮我搜索 Python 3.12 的新特性`
2. 观察是否有 `capture_signal` 被调用

**预期**：

- **不应该**调用 `capture_signal`（没有触发条件）
- 没有多余的学习噪音产生

**验证**：

```bash
# signal_buffer 中不应有新文件
ls -lt ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/
```

---

## TC-09: Layer 2 fast-track 路径完整性

**目标**：验证 USER_CORRECTION 信号走 fast-track（跳过聚类直接生成 taxonomy）。

**步骤**：

1. 清空现有数据（可选）：
   ```bash
   rm ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/sig-*.yaml 2>/dev/null
   rm ~/.deskclaw/nanobot/workspace/meta-learning-data/signal_buffer/layer2_state.json 2>/dev/null
   ```
2. 向 DeskClaw 发送一个任务，然后纠正
3. 等待 `run_layer2` 完成
4. 检查 `audit/layer2_trace.jsonl`

**预期**：

- trace 中有 `fast_track` 事件
- `step_start: {"step": "fast_track", "count": 1}` 出现在 trace 中
- taxonomy 中有对应条目

**验证**：

```bash
# 检查 Layer 2 trace 日志
cat ~/.deskclaw/nanobot/workspace/meta-learning-data/audit/layer2_trace.jsonl | python3 -m json.tool --no-ensure-ascii 2>/dev/null || cat ~/.deskclaw/nanobot/workspace/meta-learning-data/audit/layer2_trace.jsonl

# 检查 taxonomy
cat ~/.deskclaw/nanobot/workspace/meta-learning-data/error_taxonomy.yaml
```

---

## TC-10: 连续纠正的增量学习

**目标**：验证多次纠正产生的规则能增量累积，不丢失已有规则。

**步骤**：

1. **第一次纠正**：`帮我写一个 Python 脚本` -> 纠正：`Python 文件用 4 空格缩进，不要用 tab`
2. 等待 Layer 2 完成，记录当前 SKILL.md 中的规则数量
3. **第二次纠正**：`帮我提交代码` -> 纠正：`提交前要先运行测试，不要直接 git push`
4. 等待 Layer 2 完成，检查 SKILL.md

**预期**：

- SKILL.md 中包含**两条**规则（缩进 + 提交前测试）
- 第一条规则未被第二次 sync 覆盖
- `error_taxonomy.yaml` 中有 2 个 taxonomy entries

**验证**：

```bash
# 检查 SKILL.md 规则数量
grep "^- " ~/.deskclaw/nanobot/workspace/skills/meta-learning/SKILL.md

# 检查 taxonomy entries 数量
grep "id:" ~/.deskclaw/nanobot/workspace/meta-learning-data/error_taxonomy.yaml | wc -l
```

---

## 测试矩阵总结

| ID | 场景 | 触发类型 | 关注链路 |
|----|------|----------|----------|
| TC-01 | 单次用户纠正 | USER_CORRECTION | capture -> trigger -> L2 -> skill |
| TC-02 | 纠正+错误共存 | USER_CORRECTION (优先) | 信号分类优先级 |
| TC-03 | 下次任务生效 | - | SKILL.md 注入 -> agent 行为变化 |
| TC-04 | quick_think 命中 | - | taxonomy -> QuickThink 检索 |
| TC-05 | 非纠正信号累积 | SELF_RECOVERY x2 | >=2 阈值触发 |
| TC-06 | Heartbeat 兜底 | USER_CORRECTION | HEARTBEAT.md -> 定时检查 |
| TC-07 | 跨会话迁移 | USER_CORRECTION | 会话 A 学习 -> 会话 B 受益 |
| TC-08 | 正常对话无噪音 | 无触发 | 不应产生信号 |
| TC-09 | fast-track 路径 | USER_CORRECTION | 跳过聚类直接 taxonomy |
| TC-10 | 增量累积 | USER_CORRECTION x2 | 多规则不丢失 |
