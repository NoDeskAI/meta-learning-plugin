---
name: meta-learning-l2
version: 1.0.0
description: 触发元学习 Layer 2 近线整合流水线（信号物化 → 语义聚类 → 错误分类树 → 技能进化）
metadata:
  openclaw:
    requires:
      bins: ["python3"]
---

# Meta-Learning Layer 2 — 近线整合

## 何时激活

在以下任一条件满足时使用此技能：

1. 心跳检查发现 `signal_buffer/` 目录中有 >= 5 个未处理的 `.yaml` 信号文件
2. 距上次 Layer 2 运行已超过 24 小时，且存在未处理信号
3. 用户明确要求运行学习整合

## 检查是否需要运行

```bash
# 统计未处理信号数量
ls signal_buffer/sig-*.yaml 2>/dev/null | wc -l
```

如果计数 >= 5，则触发 Layer 2。

## 运行 Layer 2

```bash
cd "$META_LEARNING_PROJECT_DIR"
source .env.paths 2>/dev/null
PYTHON_BIN="${VIRTUAL_ENV:-venv}/bin/python3"
[ "$(uname)" != "Linux" ] && [ "$(uname)" != "Darwin" ] && PYTHON_BIN="${VIRTUAL_ENV:-venv}/Scripts/python.exe"
$PYTHON_BIN -m meta_learning run-layer2 --workspace "${OPENCLAW_WORKSPACE_ROOT:-~/.openclaw/workspace}"
```

## 运行 Layer 3（可选，低频）

仅在 experience_pool 中有足够经验（>= 10 条）时运行：

```bash
cd "$META_LEARNING_PROJECT_DIR"
source .env.paths 2>/dev/null
PYTHON_BIN="${VIRTUAL_ENV:-venv}/bin/python3"
[ "$(uname)" != "Linux" ] && [ "$(uname)" != "Darwin" ] && PYTHON_BIN="${VIRTUAL_ENV:-venv}/Scripts/python.exe"
$PYTHON_BIN -m meta_learning run-layer3 --workspace "${OPENCLAW_WORKSPACE_ROOT:-~/.openclaw/workspace}"
```

## 产出

- `experience_pool/` — 结构化经验碎片
- `error_taxonomy.yaml` — 错误分类树（供 Quick Think 使用）
- `skills/` — 自动生成/更新的技能卡片
