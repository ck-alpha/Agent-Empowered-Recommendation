# COPA Phase 1

COPA（Constraint-Oriented Pareto Optimization Agent for Recommendation）第一阶段实现的是不依赖 LLM 的确定性约束推荐闭环：

```text
Candidate Generation
  -> Candidate State Bus
  -> Hard Constraint Registry
  -> Soft Objective Registry
  -> NSGA-II Pareto Optimization
  -> Deterministic Verifier
```

Phase 1 不需要 LangChain、LangGraph 或 Ollama。LLM 在后续阶段只作为约束解析和工作流控制层，不参与确定性目标计算与验证。

## 安装

在仓库根目录、`LLM_Rec` 环境中执行：

```bash
conda activate LLM_Rec
python -m pip install -e "COPA[test]"
```

也可以使用：

```bash
python -m pip install -r COPA/requirements.txt
python -m pip install -e COPA
```

## 快速运行

```bash
python -m copa.experiments.run_phase1 \
  --config COPA/configs/smoke.yaml \
  --experiment all \
  --output-dir /tmp/copa_smoke
```

运行独立实验：

```bash
python -m copa.experiments.run_phase1 --config COPA/configs/experiment_a.yaml --experiment A
python -m copa.experiments.run_phase1 --config COPA/configs/experiment_b.yaml --experiment B
python -m copa.experiments.run_phase1 --config COPA/configs/experiment_c.yaml --experiment C
```

正式配置默认使用现有 `data/processed/beauty_scenario1_*.parquet`，执行 100 用户、3 个 seed、Top-10、100 个候选、100 个个体和 50 代优化。建议先运行 smoke。

## Python API

核心入口是 `COPAPipeline.run(RecommendationRequest) -> RecommendationResult`。完整示例见 `examples/run_synthetic.py`。

主要可扩展接口：

- `CandidateStateBus.initialize/update/query/rollback`
- `ConstraintRegistry.register/evaluate/apply`
- `ObjectiveRegistry.register/evaluate/annotate_candidates`
- `ParetoOptimizer.optimize`
- `DeterministicVerifier.verify`

约束输入示例：

```python
ConstraintSpec("budget", "numeric", "price", "<=", 50)
ConstraintSpec("brand", "categorical", "brand_id", "not_in", ["blocked"])
```

## 实验产物

每次 CLI 运行产生：

- `config_snapshot.yaml`：实际配置快照
- `environment.json`：Python 与关键依赖版本
- `per_user_metrics.csv`：逐用户、逐 seed 指标
- `summary.csv`、`results.json`：方法汇总
- `pareto_fronts.json`：前沿和最终选择
- `traces/*.jsonl`：Bus、约束、优化代次和验证轨迹

指标包括 Recall@K、NDCG@K、约束满足率、违反率、Diversity、Novelty、Hypervolume、Spacing、运行时间和 Python 峰值内存。

## 测试

```bash
cd COPA
conda run -n LLM_Rec python -m pytest tests -q
```

All Beauty smoke test 在数据缺失时会显示为 skip；不会用合成数据伪装真实数据通过。

维护和扩展说明见 [MAINTENANCE.md](MAINTENANCE.md)。
