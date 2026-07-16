# COPA 维护文档

## 1. 架构与稳定契约

Phase 1 的稳定数据流为：

```text
RecommendationRequest
  -> CandidateStateBus
  -> ConstraintRegistry
  -> ObjectiveRegistry
  -> ParetoOptimizer
  -> DeterministicVerifier
  -> RecommendationResult
```

`copa/core/models.py` 中的数据类是后续阶段的边界契约。Phase 2 的 LLM Controller 应只生成 `ConstraintSpec`、`ObjectiveSpec` 和上下文，不应直接修改候选 DataFrame、目标值或验证结果。

关键不变量：

1. `item_id` 在单次请求内唯一，`base_score` 必须可转为有限浮点数。
2. Bus 更新必须通过 `update` 完成；updater 抛错时当前版本保持不变。
3. 回滚创建新版本，不删除或重写旧快照。
4. 硬约束失败候选保留在状态中，但 `active=False`，优化器只能读取可行候选。
5. Verifier 重新执行原始约束，不能只读取优化器结论。
6. 所有随机过程必须从配置 seed 派生。

## 2. 扩展方法

### 新增硬约束

实现签名为 `(actual, operator, expected) -> bool` 的纯函数，并调用：

```python
registry.register("my_constraint", evaluator)
```

约束函数不得修改 Bus、全局变量或输入对象。需要列表级/跨用户约束时，不要复用当前 item evaluator；应新增明确的 `SlateConstraintRegistry` 或 batch optimizer，避免改变现有单用户语义。

### 新增软目标

实现签名：

```text
(item_ids, candidate_frame, objective_spec, context) -> float
```

然后通过 `ObjectiveRegistry.register` 注册。返回值必须有限。若方向为 minimize，在 `ObjectiveSpec.direction` 中声明，优化器会转换为最大化空间。

### 替换优化器或选择器

新优化器应保持 `optimize(bus, specs, config, context)` 的输入约定，并返回最终 `SlateSolution`、Pareto front 和 diagnostics。任何交叉/变异都必须保证 ID 唯一且仅来自可行候选。新增选择策略时保留 `compromise` 与 `weighted` 行为以兼容既有配置。

### 接入新的召回模型

推荐使用 `candidates_from_precomputed_scores` 将任意召回器输出转换为 `CandidateRecord`。召回器负责生成 `item_id/base_score`；COPA 不应导入模型训练代码。metadata 中需要包含启用约束和目标所引用的字段。

## 3. 配置、日志与结果

正式配置位于 `configs/`。修改默认口径时同步更新 README，并确保结果目录保存配置快照。

JSONL trace 是追加式审计日志。新增模块至少记录：模块、操作、状态、前后版本、前后候选数、耗时、seed 和输入摘要。不要写入完整用户历史、密钥或 LLM prompt。

结果文件的稳定分组键为 `experiment/method/seed/user_id`。新增指标可增加列，但不要重命名已有指标而不提供迁移说明。

## 4. 测试与复现检查

修改后至少执行：

```bash
cd COPA
conda run -n LLM_Rec python -m pytest tests -q
conda run -n LLM_Rec python -m copa.experiments.run_phase1 \
  --config configs/smoke.yaml --experiment all --output-dir /tmp/copa_smoke
```

涉及 All Beauty adapter 时额外运行一用户 real-data smoke。检查相同配置和 seed 是否得到相同 Slate、目标值和 Pareto front。

## 5. Phase 2 接入点

Phase 2 建议先使用轻量 Ollama 客户端和严格结构化解析：

```text
Natural-language request
  -> AgentController/Qwen
  -> validated ConstraintSpec + ObjectiveSpec
  -> unchanged COPAPipeline
```

当前不需要 LangChain。只有出现持久化状态、多轮工具调用、Verifier 失败后循环修复以及人工审批节点时，才引入 LangGraph。即使引入图编排，Constraint Registry、Pareto Optimizer 和 Verifier 仍保持纯确定性组件。

## 6. 变更检查清单

- 公共数据类是否向后兼容？
- Bus 原子性、历史和回滚测试是否通过？
- 新约束是否有满足、违反、缺字段和错误类型测试？
- 新目标是否有手工可计算测试和缺失特征回退测试？
- 最终 Slate 是否重新经过 Verifier？
- seed、配置快照、环境版本和 trace 是否保存？
- README、示例配置及本维护文档是否同步？
