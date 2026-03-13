# DeskClaw Meta-Learning (灵敏自进化学习系统)

> **核心使命：** 实现从“被动执行”到“主动进化”的跃迁，通过 Action-Think-Refine 循环持续优化 Agent 性能。

本项目是 OpenClaw 灵敏 Agent 的元学习 (Meta-Learning) 核心实现。它采用三层自进化架构，能够从对话轨迹中自动识别、提炼并固化知识。

## 🌟 三层架构 (Layered Architecture)

### Layer 1: 在线微学习 (Online Micro-learning)
*   **QuickThink**: 在任务执行前进行风险预测与已知错误模式匹配。
*   **SignalCapture**: 实时监控并捕获错误恢复、用户纠错、新工具使用及效率异常信号。

### Layer 2: 近线整合 (Near-line Consolidation)
*   **Materialize**: 将轻量级信号物化为结构化的经验碎片 (Experiences)。
*   **Consolidate**: 基于语义相似度进行经验聚类。
*   **Taxonomy**: 构建错误分类树，将碎片化经验提炼为标准 SOP。
*   **Skill Evolve**: 自动生成或更新 Skill 卡片，实现能力的闭环进化。

### Layer 3: 离线深度学习 (Offline Deep Learning)
*   **Cross-Task Miner**: 挖掘跨任务类型的深层模式（如不同领域任务共享的根因）。
*   **Capability Gap Detection**: 识别系统能力的盲区并提出补全建议。
*   **Memory Architecture**: 优化长期记忆 (MEMORY.md) 的存储与索引结构。

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
