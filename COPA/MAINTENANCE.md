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

`copa/core/models.py` 是确定性执行边界。Phase 2 的 `ConstraintIR` 经语义编译后只生成 `ConstraintSpec`、`ObjectiveSpec` 和上下文，不能直接修改候选 DataFrame、目标值或验证结果。

关键不变量：

1. `item_id` 在单次请求内唯一，`base_score` 必须可转为有限浮点数。
2. Bus 更新必须通过 `update` 完成；updater 抛错时当前版本保持不变。
3. 回滚创建新版本，不删除或重写旧快照。
4. 硬约束失败候选保留在状态中，但 `active=False`，优化器只能读取可行候选。
5. Verifier 重新执行原始约束，不能只读取优化器结论。
6. 所有随机过程必须从配置 seed 派生。
7. Compiler 的 `failed` 和 `clarification_required` 状态都不得调用 Phase 1。
8. LLM 输出必须同时通过 Pydantic 结构校验和 Domain Schema 语义校验。
9. Planner 和 Repair 输出必须通过各自的 Pydantic Schema 与确定性 Policy Gate。
10. Repair Agent 不能修改系统或用户硬约束；只有用户 resume 回答可触发重新编译。
11. 未通过 Verifier 的 Slate 不得出现在 Phase 3 成功结果中。

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

同一 evaluator 的多个实例使用唯一 `ObjectiveSpec.name`，并通过 `params.registry_name` 指向实际 Registry evaluator。例如 `brand_diversity` 与 `category_diversity` 都复用 `diversity`，但使用不同 attribute。

### 扩展自然语言字段或目标

在 `copa.phase2.domain` 中构造 `DomainSchema`，再通过 `DomainSchemaRegistry.register(name, schema, aliases=...)` 注册。不要在 `get_domain_schema` 或 Compiler 中增加领域判断分支。

`AttributeCapability` 的稳定字段包括 logical/executable name、kind、operators、aliases、value aliases、executor type、parameter schema、description 和可选 deterministic transform。`ObjectiveCapability` 包含 IR name、Registry evaluator name、参数、别名、参数 Schema 和描述。

扩展时遵守：

- logical name 和 aliases 只负责自然语言规范化；
- executable attribute 必须真实存在于 `CandidateRecord.metadata`；
- operator 必须是 Phase 1 Registry 已支持的闭集；
- 特殊表示转换必须由确定性 semantic compiler 完成，不能放进 Prompt 猜测。

新增能力后同步扩展 gold JSONL、语义单测和至少一个真实 Qwen smoke 样例。

内置 Registry 注册名为 `synthetic/all_beauty/mind`。MIND adapter 契约为：

- `topic -> category`
- `subtopic -> subcategory`
- `entity -> entity_ids`，由 `title_entities/abstract_entities` 解析 Wikidata ID；
- `freshness -> age_hours`，由可信推荐时间与发布时间计算；
- `popularity -> popularity`，由训练曝光统计归一化。

原始 MIND `news.tsv` 没有可信发布时间。没有上游时间字段时禁止构造 `age_hours`，涉及 freshness 的请求应返回阻塞 issue。

### 替换优化器或选择器

新优化器应保持 `optimize(bus, specs, config, context)` 的输入约定，并返回最终 `SlateSolution`、Pareto front 和 diagnostics。任何交叉/变异都必须保证 ID 唯一且仅来自可行候选。新增选择策略时保留 `compromise` 与 `weighted` 行为以兼容既有配置。

### 接入新的召回模型

推荐使用 `candidates_from_precomputed_scores` 将任意召回器输出转换为 `CandidateRecord`。召回器负责生成 `item_id/base_score`；COPA 不应导入模型训练代码。metadata 中需要包含启用约束和目标所引用的字段。

## 3. 配置、日志与结果

正式配置位于 `configs/`。修改默认口径时同步更新 README，并确保结果目录保存配置快照。

JSONL trace 是追加式审计日志。新增模块至少记录：模块、操作、状态、前后版本、前后候选数、耗时、seed 和输入摘要。不要写入完整用户历史、密钥或 LLM prompt。

Compiler audit 与 Candidate trace 分离。Compiler audit 只记录 request SHA-256、长度、domain、prompt 版本、模型、attempt、状态、issue code、耗时与 Ollama usage。除非研究数据已经公开且显式开启，不得记录任意用户原文或完整模型响应。

结果文件的稳定分组键为 `experiment/method/seed/user_id`。新增指标可增加列，但不要重命名已有指标而不提供迁移说明。

核心论文实验额外固定 `cohort_seed`，优化随机性只由 `seed` 控制。无约束方法仍必须通过原始业务约束做独立严评；`strict_constraint_satisfaction_rate` 不得由训练请求中的空约束集合计算。跨方法 Hypervolume 必须使用共享边界与相同 Sobol seed，不能用每个 front 自己的观测 min-max 作正式比较。

## 4. 测试与复现检查

修改后至少执行：

```bash
cd COPA
conda run -n LLM_Rec python -m pytest tests -q
conda run -n LLM_Rec python -m copa.experiments.run_phase1 \
  --config configs/smoke.yaml --experiment all --output-dir /tmp/copa_smoke
```

涉及 All Beauty adapter 时额外运行一用户 real-data smoke。检查相同配置和 seed 是否得到相同 Slate、目标值和 Pareto front。

Phase 2 修改后额外执行：

```bash
python -m copa.phase2.cli --config configs/phase2_qwen.yaml evaluate \
  --gold evaluation/constraint_compiler_gold.jsonl \
  --case-ids zh01,zh11,zh15,amb01,en01,en11,en15,amb06 \
  --output-dir /tmp/copa_phase2_live_eval8
```

Gold JSONL 的稳定字段为 `id/language/scenario/text/expected_status/constraints/objectives/top_k`，其中 scenario 只能是 `seen/compositional/unseen`。Parsing Accuracy 要求状态、硬约束、目标和 Top-K 同时精确匹配；Constraint Execution Accuracy 比较 Gold 与预测约束在固定候选集上产生的完整可行 ID 集合。不要用 `semantic_executable_rate` 代替执行准确率。

## 5. Phase 2 编译链路

```text
Natural-language request
  -> Ollama structured ConstraintIR
  -> Pydantic validation
  -> Domain Schema canonicalization
  -> conflict and feasible-candidate preflight
  -> validated ConstraintSpec + ObjectiveSpec + provenance
  -> unchanged COPAPipeline
```

Prompt 位于版本化模板；当前默认是 `constraint_compiler_v2`，v1 保留用于复现实验。修改语义规则时必须新建版本、更新 gold case 或新增 case，并在配置中修改 `compiler.prompt_version`；不要原地覆盖历史模板，也不要通过放宽 Pydantic `extra=forbid` 或允许未知字段来提高表面成功率。

Phase 3 的 `phase3_agent.yaml` 使用 `constraint_compiler_v3`。v3 只增加 prompt-injection 与合法业务子句的确定性语义分离规则；v2 文件和 Phase 2 配置保持不变，用于复现实验。禁止把恶意控制文本转换成约束，也禁止因拒绝恶意子句而静默丢弃同一请求中的合法价格、类别等要求。

Phase 3 使用 LangGraph 的状态图、interrupt 和 SQLite checkpoint，不使用 LangChain Agent。`langchain-core` 是 LangGraph 的传递依赖，不得在 COPA 中用它替换现有 Ollama transport 或确定性 Registry。Constraint Registry、Pareto Optimizer 和 Verifier 始终保持纯确定性组件。

## 6. Phase 3 Agent 与 checkpoint

`COPAExecutionSession` 是 Phase 1 和 Phase 3 共用的逐步执行边界。修改它时必须证明 `COPAPipeline.run()` 的 Pareto 路径不回归。Agent 新工具只能委托 session 或现有 Registry，不能复制约束和目标实现。

LangGraph 固定节点为 compile、clarification、plan、policy validation、tools、verification、repair 和 finalize。新增节点或路由时必须更新 Agent Gold、状态迁移测试和 `phase3.md`。

Planner 的合法序列固定为：

```text
apply_constraints -> compute_objectives ->
  (select_feasible_topk | optimize_pareto) -> verify
```

Repair Schema 不包含 constraint、item ID 或算法参数字段。任何新增 repair action 都必须先在确定性 `RepairPolicy` 中定义可处理的 violation code，并验证 constraint fingerprint 不变。

每个 thread 在首次运行时保存系统约束 SHA-256 指纹；每次 Compiler 重编译都会重新核对 executable plan 中 `system` provenance 的约束。用户 clarification 通过独立的 `<phase3_user_clarifications>` 上下文传给 v3 Prompt，作为较新的用户修订替换对应旧要求，不能通过字符串拼接形成两个冲突阈值。差异比较原始 `ConstraintIR.hard_constraints`，而不是去重后的 executable constraints，因此即使用户约束与系统约束重复，审计仍会保留用户明确确认的变化；Compiler audit 的哈希和长度覆盖原请求与 clarification，但不保存两者原文。

Checkpoint state 必须保持 JSON/MessagePack 可序列化，禁止保存 DataFrame、函数、Registry、数据库连接或活动客户端。SQLite 文件权限必须为 `0600`；生产日志不记录 state、原文、完整候选或模型响应。interrupt 节点在调用 `interrupt()` 前不得执行非幂等副作用。

Agent trace event ID 来自 thread/node/operation/attempt/sequence/status。修改事件身份字段时需提供迁移说明，避免恢复后重复事件无法去重。

Phase 3 修改后运行：

```bash
python -m pytest tests/test_phase3_agents.py tests/test_phase3_workflow.py -q
python -m copa.phase3.cli --config configs/phase3_agent.yaml evaluate \
  --gold evaluation/agent_workflow_gold.jsonl \
  --case-ids seen_zh01,seen_zh05,comp_en01,repair_zh01,unseen_zh01 \
  --output-dir /tmp/copa_phase3_live5
```

正式评测默认生成 Planner+Repair 逐例结果，并进一步运行 Phase 2 fixed 与 Planner-only，产出 `mode_comparison.csv` 和 `mode_comparison_summary.json`。重复执行同一 Gold/seed 时，评测器只清理自身的确定性 evaluation thread，不触碰普通用户 thread。

## 7. 变更检查清单

- 公共数据类是否向后兼容？
- Bus 原子性、历史和回滚测试是否通过？
- 新约束是否有满足、违反、缺字段和错误类型测试？
- 新目标是否有手工可计算测试和缺失特征回退测试？
- 最终 Slate 是否重新经过 Verifier？
- seed、配置快照、环境版本和 trace 是否保存？
- README、示例配置及本维护文档是否同步？
- Prompt 是否仍禁止 SQL、item ID、推荐排序和静默丢弃子句？
- `failed/clarification_required` 是否始终失败关闭？
- Compiler audit 是否没有原始私人文本？
- 中英 gold、Mock retry 和 live Qwen smoke 是否更新？
- 新 Domain 是否通过 Registry 注册，并有 alias 冲突测试？
- Gold 是否包含合法 scenario，执行准确率是否基于候选级真值？
- MIND freshness 是否来自可信时间字段而非推测？
- Planner 是否仍不能输出约束、物品 ID 或算法参数？
- Repair 是否保持系统/用户硬约束不变，并受两次循环上限控制？
- checkpoint 是否可以跨 pipeline 实例恢复且权限为 `0600`？
- Agent trace 是否去重且不包含原始请求和候选 payload？
- 正常 run 是否禁用 fault injection？

## 8. 2026-07-18 核心实验交接快照

本节是供后续新对话、论文复盘和继续实验使用的当前事实快照。完整结果根目录为：

```text
COPA/results/copa_core_20260716T174120Z
```

正式执行已生成 Phase 1 的 2,400 条逐用户记录、Phase 2 的 180 条 run-case 和 Phase 3 的 324 条 mode-run-case。核心实现新增 `run_core.py`、固定权重 `WeightedGeneticOptimizer`、共享边界 Hypervolume/Spacing、严格约束后评估、可恢复 supervisor、资源监控、统计分析与 PNG/PDF 制图。最近一次完整测试为 46 项通过。

### 8.1 Phase 1 当前协议

- 数据：Amazon Reviews'23 All Beauty；按用户时间排序，最后一次交互为 ground truth，其余为训练历史。
- 队列：`cohort_seed=42` 固定 100 名用户；优化 seeds 为 `42/43/44`。
- 每用户 100 个候选，Top-10；进化方法 population=100、generations=50。
- A/Hard-only：`unconstrained_relevance` 对比 `feasible_relevance`。
- B/Soft-only：`unconstrained_relevance`、等权 `unconstrained_weighted_ga`、`unconstrained_copa`。
- C/Joint：`feasible_relevance`、`feasible_weighted_ga`、完整 `copa`。
- 三个优化目标为 relevance、brand diversity、inverse-popularity novelty；正式 All Beauty 不启用 fairness。

All Beauty adapter 当前使用启发式候选分数：

```text
base_score = 0.65 * popularity
           + 0.25 * preferred_brand
           + 0.10 * preferred_category
```

这三个权重只是确定性工程启发式，当前没有离线调参、学习排序、概率模型或外部文献能够证明 0.65/0.25/0.10 为最优。论文不得把它描述为有理论依据的最优召回器。若保留，必须做权重敏感性分析；更推荐用训练折内的 BPR、矩阵分解、SASRec、双塔或 LambdaMART 产生 `base_score`，COPA 仅消费预计算候选分数。

当前候选池由用户偏好品牌/类别匹配商品、预算内热门商品和全局热门商品合并，先排除训练历史，再按 base score 排序；约 60% 容量留给 relevance-oriented 候选，其余优先补预算内商品。该过程没有显式注入 ground truth，也不保证固定召回率。正式结果的 Candidate Recall 仅为 0.10，Retrieval Loss 为 0.90，因此当前 Phase 1 的首要瓶颈是召回，而不是 Pareto 重排。

### 8.2 当前硬约束及已知有效性问题

每用户三条原始业务约束为：

1. `price_filled <= budget_high`；`budget_high` 当前由用户全部交互商品价格中位数上浮 20% 合成。
2. `inventory_initial > 0`；库存由流行度、固定系数和高斯噪声合成。
3. 训练历史 `item_id not_in seen_items`。

必须保留以下限制说明：

- 预算特征在全量 reviews 上生成，可能包含 leave-one-out ground truth 的价格，存在测试信息泄漏风险；应只用训练历史重算。
- 候选生成器本身使用预算内商品池，使 retrieval 与 constraint filtering 部分耦合。
- 库存生成器要求 `min_inventory >= 1`，本次数据库存范围为 5--122，所以 `inventory_initial > 0` 恒真，不能作为有效库存约束证据。
- seen-items 已在候选生成阶段排除，严格后评估中的同名约束主要是防御性不变量。
- 因而本次 All Beauty 中实质生效的主要是预算约束，统一 violation rate 还会被两条几乎恒真的约束稀释。

库存若无真实补货/销量数据，应明确称为 synthetic stress-test protocol，而不能声称模拟了真实库存。推荐改为可校准的离散时间过程：用训练期交互估计商品需求率及过度离散程度，采用 Poisson/Negative-Binomial demand；给定可解释的 service level、lead time 和 base-stock/reorder policy 生成库存；用低/中/高压力场景报告敏感性。若无法校准这些参数，主论文真实数据实验应删除库存有效性结论，只保留合成故障注入验证。

### 8.3 Phase 1 结果解释边界

- `candidate_recall=0.10`、`retrieval_loss=0.90`、`constraint_filter_loss=0.03`、`target_in_feasible_domain=0.07`。
- constrained 方法的 ranking loss 约 0.05--0.06，最终 Recall@10 约 0.01--0.02；所有方法 `actual_k_ratio=1` 且无 shortage。
- 所有 Recall/NDCG 配对比较经 Holm 校正后不显著，不能声称 COPA 提升准确率。
- COPA 相比 relevance Top-K 提升 novelty 与共享 Hypervolume；固定权重 GA 的 novelty 更高，而 COPA 的优势主要是产生更宽的多目标 trade-off set。
- `unconstrained_copa` 的最大 relevance 略高于 `copa` 是因为它可以使用预算外候选；它不是合法业务可行域中的公平胜利。完整 COPA 用更小的可行域换取 100% 严格约束满足。
- COPA 的 Pareto size 在约第 15 代接近 population=100，同时 diversity 几乎恒为 1。这更可能意味着 diversity 饱和和弱支配导致选择压力不足，而不应直接解释为高质量收敛。
- 当前 convergence 图缺少逐代 shared Hypervolume、selected-compromise 稳定性、跨 seed CI 和早停判据；累计 evaluations 只是预算消耗曲线。

### 8.4 Phase 2/3 评测口径

`Gold` 指人工预先标注且可作为 ground truth 的评测样例，不是训练数据。Phase 2 的 60 条双语 Gold 分为：

- `seen`：Prompt/能力表覆盖的常见单约束或已知表达组合；
- `compositional`：已知原子能力的新组合；
- `unseen`：歧义、冲突、未知值、非法范围或安全边界表达；
- `parsing_exact`：状态、硬约束集合、目标集合和 Top-K 必须同时精确匹配 Gold。

Phase 2 和 Phase 3 正式 Gold 评测都使用 seed=42 的 30 个 synthetic candidates，而不是 All Beauty 用户候选。原因是需要精确计算可行 ID 集合、冲突、shortage 和故障修复真值；它们验证的是 compiler/agent correctness，不是现实推荐效果。

Phase 2 的 `execution_exact` 只在期望状态为 success 的样例上比较预测约束与 Gold 约束产生的完整可行 ID 集合。P50/P95 分别是延迟分布的第 50/95 百分位，不是置信区间。

Phase 3 图表当前用 `status == "success"` 生成 success bar，会把正确的 `clarification_required` 当成未成功。后续主表应以 `status_correct = (status == expected_status)` 作为任务正确率，并分别报告 Slate delivery rate、clarification rate、infrastructure failure rate 和 delivered-slate verifier pass rate。当前 mode comparison 只向 Planner 路径注入 repair fault，`phase2_fixed` 没有面对同一故障，因此 raw quality 不能用于声称 fixed pipeline 优于完整 Agent。

### 8.5 图表、原子性与后续优先级

仅展示框架本体的论文风格图使用：

```text
COPA/docs/assets/copa_framework_topconf_v2.png
```

Bus 的“原子提交”是指 updater 先在候选状态副本上完成全部变更与校验；只有全部成功时才一次性把副本发布为新版本。任一字段、约束或数值校验失败时，当前版本完全不变，不会留下半过滤、半更新的候选状态。

后续实验优先级：

1. 修复预算 leave-one-out 泄漏并移除候选生成中的预算耦合。
2. 接入真实召回模型，同时报告真实 candidate recall 与 oracle-candidate 重排实验。
3. 删除或重新校准库存生成协议，并报告逐约束违反贡献。
4. 替换饱和的品牌差异 diversity，增加类别覆盖、熵或 embedding ILD。
5. 重画逐代 shared HV/selected-slate 曲线，并使用跨用户/seed 置信区间。
6. 修正 Phase 3 completion 与 McNemar 的 `status_correct` 口径，并对三种模式施加相同故障条件。
