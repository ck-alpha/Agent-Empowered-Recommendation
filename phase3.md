# COPA Phase 3：完整 Agent 推荐系统

## 背景与目标

Phase 3 在 Phase 1 确定性约束优化和 Phase 2 Qwen Constraint Compiler 之上加入有界、可审计、可恢复的 Agent 工作流：

```text
Natural-language Request
  -> Qwen2.5:14B Constraint Compiler
  -> Constraint IR
  -> Agent Planner
  -> Deterministic Policy Gate
  -> Candidate State Bus / COPA Tools
  -> Verifier
       ├─ pass -> Recommendation
       └─ fail -> Repair Agent -> repair / clarification / abort
```

LangGraph 负责状态图、中断和 checkpoint；不使用 LangChain Agent。Constraint Registry、Objective Registry、NSGA-II 和 Verifier 保持确定性。

## Agent 模块

### 1. Planner Agent

Planner 根据已经验证的 Compiled Plan 生成严格 `AgentExecutionPlan`，只允许调用：

- `apply_constraints`
- `compute_objectives`
- `select_feasible_topk`
- `optimize_pareto`
- `verify`

确定性 Policy Gate 强制工具顺序和策略：仅 relevance 时使用 feasible Top-K；存在额外软目标时使用 Pareto。Planner 不得生成物品 ID、约束、Top-K、seed 或优化参数。

### 2. Execution Tracker

Agent trace 使用确定性 event ID，记录 node、operation、attempt、工具摘要、候选数量、Bus 版本、约束状态、耗时和错误。trace 不记录原始请求、完整候选、Prompt 或模型原始输出。

Candidate trace 保留 Phase 1 格式，并按 thread 和 repair attempt 分开保存。

### 3. Repair Agent

Verifier 失败后，Repair Agent 只能选择：

- `recompute_objectives`
- `reexecute_selection`
- `replan`
- `request_clarification`
- `abort`

Repair Agent 不得修改、删除或放宽任何硬约束。候选不足必须请求用户明确澄清；用户回答后重新经过 Constraint Compiler。系统约束始终不可变，用户确认造成的约束变化必须记录差异。

默认最多修复两次；相同 violation signature 连续出现时提前终止，防止死循环。

## LangGraph 状态与恢复

固定节点为：

1. `compile_request`
2. `await_clarification`
3. `plan_execution`
4. `validate_plan`
5. `execute_tools`
6. `verify_result`
7. `decide_repair`
8. `finalize`

所有 checkpoint state 均为可序列化字典，不保存 DataFrame、函数或连接。SQLite 使用 `thread_id` 隔离工作流，支持：

- `run` 创建新 thread；
- `resume` 使用同一 thread 恢复 clarification interrupt；
- `status` 查看不含原文的状态摘要；
- `cleanup` 按 thread 或时间清理。

默认 checkpoint 位于 `COPA/checkpoints/phase3.sqlite`，权限为 `0600`，并启用严格 msgpack 反序列化。完整原始请求和候选仅保存在本地 checkpoint，用于跨进程恢复。

每个 thread 记录系统约束指纹并在重编译时强制复核。clarification 通过独立的结构化 Prompt 上下文传入，作为较新的用户修订替换对应旧要求；前后的用户约束差异基于原始 Constraint IR 记录，避免系统/用户重复约束在 canonicalization 去重后丢失用户确认轨迹。

## 实验与指标

Agent Gold 数据包含 36 条中英场景，分为：

- `seen`
- `compositional`
- `repair`
- `unseen`

Repair 场景通过显式开启的确定性 fault injection 产生 duplicate、unknown、non-finite 和 shortage；正常运行禁止 fault injection。

指标包括 Agent Status Accuracy、Planner Strategy Accuracy、Verifier Feasible Rate、Repair Action Accuracy、Repair Success Rate、工具调用数、Agent 循环数、LLM 调用数、token 和 p50/p95 latency。

正式评测统一比较 `phase2_fixed`、`planner_only` 与 `planner_repair`，并输出每种模式的任务成功率、Verifier 可行率、Recall/NDCG/Diversity/Novelty、工具调用、Repair 循环、token 与延迟。

## 安全与工程约束

- 未通过 Verifier 的 Slate 永远不能作为推荐结果返回；
- LLM 不能选择物品或修改算法参数；
- LLM 结构输出最多尝试两次，失败后关闭执行；
- Phase 3 使用版本化 Compiler Prompt v3：忽略 prompt-injection 的控制意图，同时继续编译同一请求中可分离的合法业务条件；
- 所有随机过程来自配置 seed；
- Phase 1/2 公共 API 和确定性结果保持兼容；
- MIND 继续保留 Domain Schema，真实 MIND Agent adapter 不在 Phase 3 范围内；
- 配置、Prompt、Gold、README、维护文档和测试必须随行为变化同步更新。
