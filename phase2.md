# COPA Phase 2：加入 Qwen2.5:14B Constraint Compiler

## 背景与职责边界

服务器已通过 Ollama 部署 `qwen2.5:14b`。Phase 2 在 Phase 1 的确定性推荐闭环前增加自然语言约束编译层：

```text
Natural-language Request
  -> Qwen2.5:14B Constraint Compiler
  -> Strict Constraint IR
  -> Domain Schema / Semantic Validation
  -> ConstraintSpec + ObjectiveSpec
  -> Phase 1 COPAPipeline
```

LLM 只负责自然语言约束和偏好理解，不负责推荐排序，不生成 SQL、Python、工具调用、物品 ID 或算法参数。约束执行、Pareto 优化与结果验证均由确定性模块完成。

## 新增模块

### 1. LLM Controller

通过 Ollama `/api/generate` 调用本地 Qwen，支持：

- 版本化 Prompt 管理；
- JSON Schema structured output；
- Pydantic 严格二次校验；
- 有限网络重试和一次结构修复；
- 不记录原始私人请求的审计日志。

两次结构或服务调用失败后返回 `failed`；存在未知或模糊需求时返回 `clarification_required`。两种状态均不得执行推荐。

### 2. Constraint IR

Constraint IR 是 Natural Language 与 Executable Constraint 之间的严格中间表示。

用户请求：

> 推荐 50 美元以内的电子产品，最好品牌不要太集中。

结构化输出示例：

```json
{
  "schema_version": "1.0",
  "hard_constraints": [
    {
      "attribute": "price",
      "operator": "<=",
      "value": 50,
      "currency": "USD"
    },
    {
      "attribute": "category",
      "operator": "==",
      "value": "electronics",
      "currency": null
    }
  ],
  "soft_objectives": [
    {
      "objective": "brand_diversity",
      "direction": "maximize"
    }
  ],
  "top_k": 10,
  "unresolved_requirements": []
}
```

LLM 不生成执行类型或 Constraint ID。Compiler 根据 Domain Schema 确定性生成 `ConstraintSpec`，具体执行由 Phase 1 `ConstraintRegistry` 完成。

### 3. Constraint Schema Registry

使用 `DomainSchemaRegistry`（公共别名为 `ConstraintSchemaRegistry`）管理不同领域允许 LLM 使用的字段和目标。每个字段能力包含：

- 逻辑字段名与真实 metadata 字段名；
- 字段类型与参数 Schema；
- 合法 operator；
- 执行器类型或特殊转换；
- 别名与描述。

Schema Registry 负责“允许理解什么”，Phase 1 Constraint Registry 负责“如何确定性执行”，二者保持解耦。

当前领域契约：

- Synthetic：`price/category/brand/group/popularity/availability`；
- Amazon All Beauty：`price/category/brand/seller/popularity/availability`；
- MIND：`topic/subtopic/entity/freshness/popularity`。

MIND 映射为 `topic→category`、`subtopic→subcategory`、`entity→entity_ids`、`freshness→age_hours`。原始 MIND 新闻文件不包含可靠发布时间，因此候选适配器必须从上游时间上下文提供 `age_hours`；字段缺失时 Compiler 必须阻塞，不能猜测 freshness。

## Phase 2 实验

增加 Constraint Compilation Evaluation，测试：

```text
Natural Language -> Constraint IR -> ConstraintSpec -> Candidate Feasibility
```

测试场景分为：

- `seen`：Schema 支持的单一或已覆盖表达；
- `compositional`：多条约束、目标和冲突组合；
- `unseen`：未覆盖表达、未知字段、模糊要求、边界输入和 Prompt Injection。

主要指标：

- Parsing Accuracy：状态、硬约束、软目标和 Top-K 全部精确匹配的样例比例；
- Constraint Execution Accuracy：编译约束与 Gold 约束产生完全相同可行候选集合的比例；
- Candidate Feasibility Precision / Recall / F1；
- Schema Valid Rate、Hard-Constraint F1、Objective F1、Top-K Accuracy；
- Clarification F1、Retry Rate、p50/p95 Latency。

执行真值由 Gold Constraint IR 在固定合成候选集上确定性执行得到，避免把候选 ID 冗余写入 Gold 数据并随 fixture 变化而失效。评测同时输出总体指标和按场景分层指标。

## 工程要求

- 在 Phase 1 代码基础上扩展，不改变推荐、NSGA-II 和 Verifier 的确定性契约；
- Qwen 不参与物品选择和 Pareto 排序；
- 不引入 LangChain 或 LangGraph；
- 更新配置、实验 CLI、公共 API、测试、README 和维护文档；
- 新领域通过 Domain Schema Registry 注册，不在 Compiler 中增加领域判断分支。
