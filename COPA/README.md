# COPA Phase 1–3

COPA（Constraint-Oriented Pareto Optimization Agent for Recommendation）第一阶段实现的是不依赖 LLM 的确定性约束推荐闭环：

```text
Candidate Generation
  -> Candidate State Bus
  -> Hard Constraint Registry
  -> Slate Constraint Registry + exact MILP preflight
  -> Soft Objective Registry
  -> NSGA-II Pareto Optimization
  -> Deterministic Verifier
```

Phase 2 在该闭环前加入 Qwen2.5:14B Constraint Compiler：

```text
Natural-language Request
  -> Ollama/Qwen Structured Constraint IR
  -> Semantic Validation + Domain Schema Registry
  -> ConstraintSpec + ObjectiveSpec
  -> unchanged Phase-1 Pipeline
```

Phase 1/2 中 LLM 只解析自然语言，不生成 SQL、物品 ID、排序结果或优化参数。Phase 3 使用独立 LangGraph 组织可恢复 Agent 状态图，但不使用 LangChain Agent；确定性工具仍负责候选选择和验证。

Phase 2/3 的当前协议使用 `constraint_compiler_v4` 与 Constraint IR v1.1。IR 以 `scope=item|slate` 区分逐物品条件和列表聚合；v1.0 输入继续按 item scope 兼容读取。旧 v1–v3 Prompt 文件仍保留用于历史复现。

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

### 核心论文实验与自动报告

强化主对比将固定用户队列与优化 seed 解耦，运行 Hard-only、Soft-only、Joint 三组共八个方法单元，并使用原始业务约束进行统一严评：

```bash
python -m copa.experiments.run_core \
  --config COPA/configs/core_experiment.yaml \
  --suite core --workers 4 --cohort-seed 42 \
  --output-dir COPA/results/core_run --resume
```

完整三阶段实验由可恢复 supervisor 顺序执行测试、smoke、Phase 1、三次 Phase 2、三次 Phase 3、统计和制图，适合在 tmux 中运行：

```bash
python -m copa.experiments.supervisor \
  --output-dir COPA/results/copa_core_TIMESTAMP \
  --workers 4 --cohort-seed 42 --repeats 3 --resume
```

仅从已保存原始结果重建统计、PNG/PDF 和中文报告：

```bash
python -m copa.experiments.analyze_core --input-dir COPA/results/copa_core_TIMESTAMP
```

supervisor 会保存 `run_manifest.json`、`stage_status.jsonl`、`resource_usage.csv`、逐阶段日志和精确命令。失败后使用同一输出目录及 `--resume`，已完成阶段不会重跑，Phase 1 还会复用逐用户/seed 的原子 checkpoint。

## 成熟召回层：RecBole BPR、ItemKNN 与 SASRec

正式召回实验自 2026-07-18 起统一使用 `recbole==1.2.1`。仓库原有 PyTorch BPR 是可复现的 legacy 自研基线，不再参与正式方法选择；`src/run_scenario1_baselines.py` 中由 `implicit` 提供的成熟 Item-KNN 变体必须显式命名为 `itemknn_bm25`、`itemknn_tfidf` 或 `itemknn_cosine`，不能与 RecBole ItemKNN 合并报告。SASRec 使用 RecBole 自带的 TransformerEncoder，不需要安装 Hugging Face `transformers`。

安装并校验独立依赖：

```bash
conda run -n LLM_Rec python -m pip install -r COPA/requirements-retrieval.txt
conda run -n LLM_Rec python -m pip check
```

三模型最小训练、full-sort 和 artifact 集成 smoke：

```bash
cd COPA
conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
  smoke --output-dir results/retrieval_smoke
```

正式 Beauty/Electronics 去重 5-core、冻结 train 的时序切分与三 seed 重训（从仓库根目录执行）：

```bash
conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
  suite --config COPA/configs/retrieval_extension.yaml \
  --output-dir COPA/results/retrieval_formal --resume
```

训练、validation-only slate calibration、正式矩阵和核心消融可以用可恢复脚本顺序执行；长任务应放入 tmux：

```bash
tmux new-session -d -s copa_slate_multi_positive_v2 \
  "bash COPA/scripts/run_slate_multi_positive_v2.sh \
   COPA/results/slate_multi_positive_v2 4 2>&1 | tee /tmp/copa_slate_multi_positive_v2.log"
```

已有 v2 召回 artifact 时，kernel-v2 优化重跑不重新训练模型。门控脚本依次执行全量测试、两数据集严格校准、artifact/multi-positive 复核、8+8 用户满预算性能 pilot，全部通过后才启动正式矩阵：

```bash
tmux new-session -d -s copa_slate_multi_positive_v2_optimized \
  "bash COPA/scripts/run_slate_multi_positive_v2_optimized.sh \
   COPA/results/slate_multi_positive_v2_20260719 \
   COPA/results/slate_multi_positive_v2_optimized_20260721 12"
```

运行状态见新目录的 `runner_status.json`；各 evaluation 子目录另有原子 `task_results/` 与持续更新的 `progress.json`。正式进程固定 `OMP/OPENBLAS/MKL/NUMEXPR` 单线程，12 个 worker 不会在每个 worker 内再次过度并行。校准或 pilot 任一门槛失败都会保留搜索表/诊断并停止，旧结果不会进入新汇总。

每用户最后两条交互是同一 test positive set，倒数第三条是 validation，其余为 train。请求统计、seen set、流行度、预算和 SASRec 查询历史只读取 train；validation 只选模型，test 不参与构造。候选目录仅允许 train 中出现的物品，test-only 目标以空 score/rank 记录为 model-catalog loss。RecBole 的 checkpoint-selection evaluator 同样屏蔽 validation/test-only 目录物品，避免随机初始化的冷物品 embedding 影响 early stopping。

三模型对全部 Beauty 253 / Electronics 33,138 用户执行统一 multi-positive full-sort；下游优化固定使用 Beauty 全部 253 用户和 `cohort_seed=42` 的 Electronics 500 用户，名单保存为 `downstream_users.txt`，不按 coverage、item eligibility 或 slate opportunity 过滤。SASRec 的两个未来正例共享同一冻结查询，不会被错误拆成两条不同历史；显式 `item_id_list` 与 target `item_id` 强制共享同一 RecBole token vocabulary，并按每个 trial 的 `MAX_ITEM_LIST_LENGTH` 保留最近历史。

suite 还会生成 `popularity_full_sort.csv`。正式请求统计只使用 train；协议是严格的用户内时间冻结，并不是全局时间切分，因此不能宣称消除了所有 global temporal leakage。

也可以分别执行数据转换与单模型训练：

```bash
conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension prepare \
  --interactions data/processed/beauty_scenario1_interactions.parquet \
  --items data/processed/beauty_scenario1_items.parquet \
  --dataset-name beauty_5core --output-root COPA/results/retrieval_atomic --k-core 5

conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension train \
  --data-root COPA/results/retrieval_atomic --dataset-name beauty_5core \
  --source-interactions data/processed/beauty_scenario1_interactions.parquet \
  --source-items data/processed/beauty_scenario1_items.parquet \
  --model sasrec --seed 42 --epochs 100 --stopping-step 10 --candidate-k 500 \
  --output-dir COPA/results/retrieval_sasrec
```

对一个候选 artifact 执行真实 K 扫描、Oracle/受控召回以及 COPA 端到端实验：

```bash
conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension evaluate \
  --manifest COPA/results/retrieval_sasrec/manifest.json \
  --split COPA/results/retrieval_atomic/beauty_5core/beauty_5core_split.parquet \
  --items data/processed/beauty_scenario1_items.parquet \
  --candidate-ks 50,100,200,500 --end-to-end-candidate-k 100 \
  --optimizer-seeds 42,43,44 --population-size 100 --generations 50 \
  --interventions --output-dir COPA/results/retrieval_sasrec_evaluation
```

正式 suite 完成后，一条命令执行全部三 seed 召回扫描、seed-42 三召回器 medium 端到端矩阵、loose/medium/tight sweep、Oracle/六档 pair-level 受控敏感性实验，以及 item-only 和无 MILP seed/feasible operators 消融：

```bash
conda run -n LLM_Rec python -m copa.experiments.run_retrieval_extension \
  evaluate-suite --suite-dir COPA/results/retrieval_formal \
  --workers 4 --resume
```

端到端任务按 dataset/retriever/condition/user/method/optimizer-seed 保存原子 checkpoint；中断后 `--resume` 不会重算签名一致的 population=100、generations=50 优化任务。签名绑定 kernel version、校准、cohort、方法、seed、搜索预算、受控比例和 intervention seed；kernel-v1 或不同协议的未完成任务会安全失效。

候选 parquet 的稳定字段为：

```text
user_id, item_id, raw_score, base_score, retrieval_rank,
retriever, backend, model_seed
```

Artifact v2 的 `targets.parquet` 是长表：

```text
user_id, target_item_id, target_order, target_timestamp,
target_raw_score, target_base_score, target_full_rank, target_model_covered
```

model-uncovered 目标的三项 score/rank 必须为空；covered 目标必须有限并与候选排序严格一致。Reader 可把 v1 单目标 artifact 提升为单元素集合，但新 writer 只生成 v2。Manifest 绑定 split policy、positive set、train catalog、request cutoff、calibration、benchmark、checkpoint 和 artifact hashes。

下游逐用户计算 `P→I→M→C→U→H`，将 item constraint、model catalog、retrieval、slate constraint、ranking/selection 五类损失和 `H/|P|` 严格加和为 1；同时报告 multi-positive Recall/NDCG、legacy first-positive 对照、eligible rates、opportunity recall、constraint-aware NDCG、full-K 严格 CSR、delivered-slate Verifier pass、逐约束 violation magnitude、2 秒 preflight/5 秒 opportunity solver cost、P50/P95、shared HV/Spacing 与配对 user bootstrap。

## Phase 2：Qwen Constraint Compiler

启动本地 Ollama（该服务器的模型目录）：

```bash
OLLAMA_MODELS=/home/linchengkai/ollama/models \
  /home/linchengkai/ollama/bin/ollama serve
```

编译自然语言，不执行推荐：

```bash
python -m copa.phase2.cli --config COPA/configs/phase2_qwen.yaml compile \
  --domain synthetic \
  --text "推荐50美元以内且有货的商品，最好品牌多样，给我5个"
```

编译并调用 Phase 1 确定性推荐：

```bash
python -m copa.phase2.cli --config COPA/configs/phase2_qwen.yaml recommend \
  --domain synthetic \
  --text "推荐60美元以内的商品，尽量品牌多样，给我5个"
```

完整双语 Compiler 评测：

```bash
python -m copa.phase2.cli --config COPA/configs/phase2_qwen.yaml evaluate \
  --gold COPA/evaluation/constraint_compiler_gold.jsonl \
  --output-dir COPA/results/phase2_compiler_eval
```

使用 `--case-ids zh01,zh11,en01,en11` 可运行小规模 live smoke。Qwen 或 IR 校验失败后最多重试一次；最终状态为 `failed` 或 `clarification_required` 时不会执行推荐。

使用 `--scenarios seen,compositional` 可以只运行指定场景；无过滤时汇总 JSON 会同时包含总体指标和三类场景指标。

## Phase 3：Checkpointed Agent Workflow

启动一个 Agent thread：

```bash
python -m copa.phase3.cli --config COPA/configs/phase3_agent.yaml run \
  --thread-id demo-001 --domain synthetic \
  --text "推荐5个价格不超过60美元的商品，品牌尽量多样"
```

如果返回 `clarification_required`，使用同一 thread 恢复：

```bash
python -m copa.phase3.cli --config COPA/configs/phase3_agent.yaml resume \
  --thread-id demo-001 --answer "最高价格是60美元"
```

状态与清理：

```bash
python -m copa.phase3.cli --config COPA/configs/phase3_agent.yaml status --thread-id demo-001
python -m copa.phase3.cli --config COPA/configs/phase3_agent.yaml cleanup --thread-id demo-001
python -m copa.phase3.cli --config COPA/configs/phase3_agent.yaml cleanup --older-than-days 7
```

运行 44 条中英 Agent Gold（含 8 条列表级故障）：

```bash
python -m copa.phase3.cli --config COPA/configs/phase3_agent.yaml evaluate \
  --gold COPA/evaluation/agent_workflow_gold.jsonl \
  --output-dir COPA/results/phase3_agent_eval
```

正式评测默认同时运行 Phase 2 fixed、Planner-only 和 Planner+Repair 三种模式；使用 `--skip-comparison` 可只跑完整 Agent，适合小规模 Qwen live smoke。

正常 `run` 禁止 fault injection；只有 `evaluate` 根据配置启用可复现的 Verifier 故障。

## Python API

Phase 1 核心入口是 `COPAPipeline.run(RecommendationRequest) -> RecommendationResult`。Phase 2 核心入口是 `ConstraintCompiler.compile(...)` 和 `NaturalLanguageCOPAPipeline.run(...)`。Phase 3 使用 `AgentCOPAPipeline.run/resume/get_state/delete_thread`。完整示例见 `examples/`。

主要可扩展接口：

- `CandidateStateBus.initialize/update/query/rollback`
- `ConstraintRegistry.register/evaluate/apply`
- `ObjectiveRegistry.register/evaluate/annotate_candidates`
- `ParetoOptimizer.optimize`
- `DeterministicVerifier.verify`
- `ConstraintCompiler.compile`
- `DomainSchemaRegistry.register/get`
- `NaturalLanguageCOPAPipeline.run`
- `COPAExecutionSession.initialize/apply_constraints/compute_objectives/select_feasible_topk/optimize_pareto/verify`
- `AgentCOPAPipeline.run/resume/get_state/delete_thread`
- `AgentToolRegistry.register/execute`
- `PlannerAgent.plan`
- `RepairAgent.decide`

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

Phase 2 额外输出隐私安全的 Compiler JSONL audit、逐样例 CSV 和汇总 JSON。编译评测包括 Parsing Accuracy、Constraint Execution Accuracy、候选可行性 Precision/Recall/F1、schema valid rate、hard/objective F1、Top-K accuracy、clarification F1、重试率及 p50/p95 latency，并按 `seen/compositional/unseen` 分层汇总。默认 audit 只保存请求哈希与长度，不保存原始私人请求。

### Domain Schema Registry

`DomainSchemaRegistry`（兼容别名 `ConstraintSchemaRegistry`）管理自然语言层允许使用的字段、operator、参数格式、执行器键和描述；Phase 1 `ConstraintRegistry` 只负责确定性执行。内置注册名及别名如下：

- `synthetic`（`demo`）
- `all_beauty`（`beauty`、`allbeauty`）
- `mind`（`news`）

MIND Schema 支持 `topic/subtopic/entity/freshness/popularity`。适配器需将实体 JSON 解析为 `entity_ids`，并在存在可靠参考时间时提供 `age_hours`。MIND 原始新闻表不含发布时间，COPA 不会自行猜测 freshness；缺少所需 metadata 时编译失败关闭。

### Phase 3 persistence 与日志

SQLite checkpoint 默认位于 `COPA/checkpoints/phase3.sqlite`，保存恢复所需的原始请求、候选和结构化状态，权限为 `0600`。Agent JSONL trace 只记录哈希、计数、工具摘要和时延，不记录原始请求或模型输出。checkpoint 默认不自动删除，使用 cleanup CLI 显式管理。

Phase 3 评测输出 `agent_case_results.csv` 与 `agent_summary.json`，包括状态、Planner 策略、Verifier、Repair、工具调用、token 和延迟指标，并按 `seen/compositional/repair/unseen` 分组。默认三模式评测还输出 `mode_comparison.csv` 和 `mode_comparison_summary.json`，统一汇总成功率、可行率、Recall/NDCG/Diversity/Novelty、工具数、Repair 循环、LLM 调用、token 与 p50/p95 latency。

## 测试

```bash
cd COPA
conda run -n LLM_Rec python -m pytest tests -q
```

All Beauty smoke test 在数据缺失时会显示为 skip；不会用合成数据伪装真实数据通过。

维护和扩展说明见 [MAINTENANCE.md](MAINTENANCE.md)。
