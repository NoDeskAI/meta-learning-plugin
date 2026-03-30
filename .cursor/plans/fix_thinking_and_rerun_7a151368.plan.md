---
name: Fix thinking and rerun
overview: "在 gdpval_meta_test.py 的 OpenSpaceConfig 中添加 `llm_kwargs={\"extra_body\": {\"enable_thinking\": False}}` 关闭 qwen3.5-plus 的 thinking 模式，然后重新执行 within-task 实验。"
todos:
  - id: fix-thinking
    content: 在 gdpval_meta_test.py 的 OpenSpaceConfig 中添加 llm_kwargs 关闭 thinking
    status: pending
  - id: smoke-test
    content: 冒烟测试：1 task, 2 iterations，确认无超时
    status: pending
  - id: rerun-within-task
    content: 重跑 within-task 实验（1 task x 3 rounds）
    status: pending
isProject: false
---

# 关闭 qwen3.5-plus thinking 并重跑实验

## 修改点

唯一需要改的地方：[scripts/gdpval_meta_test.py](scripts/gdpval_meta_test.py) 中 `_execute_task` 函数的 `OpenSpaceConfig` 构造：

```python
config = OpenSpaceConfig(
    llm_model=AGENT_MODEL,
    llm_timeout=300.0,
    llm_kwargs={"extra_body": {"enable_thinking": False}},  # 新增
    ...
)
```

同时也要给 meta-learning 的 LLM 调用（Layer 2 materialize/taxonomy）关闭 thinking，在 `abtest/config.gdpval.yaml` 中无需改动（meta-learning 用 httpx 直接调用，不走 litellm，不受影响）。

## 验证

修改后先跑一次快速冒烟测试（1 task, 2 iterations），确认 LLM 调用在 5-15 秒内返回，无超时。

## 重跑实验

确认无超时后，执行 within-task 实验（1 task x 3 rounds），继续推进剩余 todo。
