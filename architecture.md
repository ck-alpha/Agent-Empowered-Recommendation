# DualAgent-Rec 架构文档

## 1. 项目结构概览

```text
Dual-Agents-Recommendation/
├── src/
│   ├── dualagent_rec.py                    # 主编排器：双智能体 + 协调器 + 约束处理的优化闭环
│   ├── run_experiments.py                  # 实验入口：主对比、消融、结果汇总与导出
│   ├── data_utils.py                       # Amazon 数据加载、过滤、用户画像与历史构建
│   ├── llm_utils.py                        # 本地 Ollama LLM 接口封装
│   ├── agents/
│   │   ├── base_agent.py                   # 个体定义、CDP支配、非支配排序、环境选择
│   │   ├── exploitation_agent.py           # 开发智能体：偏准确率，DE/pbest/1 风格进化
│   │   └── exploration_agent.py            # 探索智能体：偏多样性/新颖性，DE/rand/1 风格进化
│   ├── llm_coordinator/
│   │   └── coordinator.py                  # LLM/启发式协调器：动态分配 α（利用/探索预算）
│   ├── constraints/
│   │   └── constraint_handler.py           # 约束计算与 ε 松弛/衰减/自适应阈值
│   └── evaluation/
│       └── objectives.py                   # 三目标计算（相关性/多样性/新颖性）与多目标指标(HV等)
├── data/                                   # 数据文件（Amazon Reviews）
├── results/                                # 结果输出目录
├── README.md
└── requirements.txt
```

## 2. 核心模块映射（代码 -> 论文三大机制）

### A. 双智能体架构（Dual-Agent Architecture）
- `src/agents/exploitation_agent.py`
  - 对应“开发智能体”：可行域内强化准确率，使用 CDP 约束优先与 `pbest` 引导变异。
- `src/agents/exploration_agent.py`
  - 对应“探索智能体”：弱化硬约束压力，强调决策空间多样性和新颖性。
- `src/agents/base_agent.py`
  - 提供 CDP 支配比较、非支配排序、拥挤度与环境选择等共同进化基元。
- `src/dualagent_rec.py::_cross_population_breeding`
  - 对应知识迁移：在两种群之间交叉，融合精度基因与多样性基因。

### B. LLM 协调器（LLM-Based Coordination）
- `src/llm_coordinator/coordinator.py`
  - `get_resource_allocation(...)`：按代数、性能、约束状态输出 exploitation 比例 `α`。
  - `_llm_allocation(...)`：通过提示词将全局状态交给 LLM 决策。
  - `_heuristic_allocation(...)`：LLM 失败回退策略，保证鲁棒性与可复现性。
- `src/dualagent_rec.py`
  - 在主循环中按 `α` 切分子代预算：`exploit_offspring = αN`，`explore_offspring = (1-α)N`。

### C. 自适应约束处理（Adaptive Constraint Handling）
- `src/constraints/constraint_handler.py`
  - `calculate_violations(...)`：三类硬约束违反量（公平性、卖家覆盖、新品曝光）。
  - `update_epsilon(...)`：ε 随代衰减并按可行率自适应回调。
  - `calibrate_epsilon(...)`：根据初始违反水平自校准衰减参数。
- `src/dualagent_rec.py`
  - 每代基于整体可行率调用 `update_epsilon(...)`，实现“先宽后严”的可行域收缩。

## 3. 调用依赖关系（Dependency Graph）

### 3.1 运行时主链路
1. 数据输入
- `run_experiments.py::load_amazon_data` 调用 `data_utils.AmazonDataLoader` 与 `UserBehaviorProcessor`。
- 产出 `candidate_items / user_histories / item_features / user_profiles`。

2. 优化入口
- `run_experiments.py::run_single_experiment` 创建 `DualAgentRec(config)`。
- 调用 `DualAgentRec.optimize(...)` 进入主循环。

3. 双智能体评估
- `dualagent_rec.py::_evaluate_populations`
  - `ExploitationAgent.evaluate_population(...)`
  - `ExplorationAgent.evaluate_population(...)`
- 两者都依赖 `ObjectivesCalculator.calculate(...)` 和 `ConstraintHandler.calculate_violations(...)`。

4. 协调与资源分配
- `dualagent_rec.py` 汇总 `exploit_metrics / explore_metrics / constraint_metrics`。
- 调用 `LLMCoordinator.get_resource_allocation(...)` 得到 `α`。
- 将子代预算切分给两个智能体进化。

5. 进化与迁移
- exploitation: `evolve(...)`（偏 `pbest`）
- exploration: `evolve(...)`（偏随机发散）
- 跨群迁移: `_cross_population_breeding(...)`
- 环境选择: `BaseAgent.environmental_selection(...)`

6. 约束收紧与结果输出
- 每代调用 `constraint_handler.update_epsilon(...)`。
- 用 `non_dominated_sort(...)` 更新 Pareto 前沿。
- 最终计算 `hypervolume / spacing / feasibility_rate` 并输出结果。

### 3.2 依赖图（简化）
```text
run_experiments.py
  -> data_utils.py
  -> dualagent_rec.py
       -> agents/exploitation_agent.py
       -> agents/exploration_agent.py
       -> agents/base_agent.py
       -> constraints/constraint_handler.py
       -> evaluation/objectives.py
       -> llm_coordinator/coordinator.py
            -> llm_utils.py (Ollama)
```

## 4. 后续优化建议（Optimization Guide）

### 场景 A：替换或增强 LLM 协调策略
- 修改文件：
  - `src/llm_coordinator/coordinator.py`（模型名、提示词、解析与回退逻辑）
  - `src/llm_utils.py`（切换推理后端/API 协议）
  - `src/dualagent_rec.py`（调用频率 `llm_update_frequency` 与输入上下文字段）
- 可做项：
  - 引入结构化输出校验（schema）与重试机制。
  - 将 `constraint_metrics` 的时间窗趋势加入 prompt（不仅看单代）。

### 场景 B：修改多目标评分公式
- 修改文件：
  - `src/evaluation/objectives.py`（`_calculate_relevance/_calculate_diversity/_calculate_novelty`）
  - `src/agents/exploitation_agent.py`、`src/agents/exploration_agent.py`（fitness 聚合方式）
- 可做项：
  - 引入用户长期兴趣漂移项。
  - 将新颖性拆分为“流行度逆向 + 时间新鲜度”双分量。

### 场景 C：增强约束机制（硬约束/软约束混合）
- 修改文件：
  - `src/constraints/constraint_handler.py`（新约束定义、ε 更新规则、自适应阈值策略）
  - `src/dualagent_rec.py`（约束指标聚合与更新时机）
- 可做项：
  - 分约束独立 epsilon（`epsilon_fair`, `epsilon_seller`, `epsilon_new`）。
  - 引入“分层可行率”信号，分别驱动 exploitation/exploration 的惩罚强度。

### 场景 D：调整并行与性能（大规模实验）
- 修改文件：
  - `src/run_experiments.py`（按用户并行、按类别并行、结果聚合）
  - `src/agents/base_agent.py`（排序/距离计算向量化）
  - `src/evaluation/objectives.py`（缓存 embedding 相似度、批量化计算）
- 可做项：
  - 对用户粒度实验使用多进程并行。
  - 为 `calculate_decision_space_diversity` 增加采样近似，降低 O(n^2) 开销。

### 场景 E：把“单用户优化”扩展到“在线服务”
- 修改文件：
  - 新增服务入口（如 `src/service/`）
  - `src/dualagent_rec.py`（暴露增量更新接口）
  - `src/data_utils.py`（实时特征更新与缓存）
- 可做项：
  - 将 `optimize(...)` 拆成“离线初始化 + 在线轻量重排”两阶段。
  - 增加模型与约束配置热更新能力。
