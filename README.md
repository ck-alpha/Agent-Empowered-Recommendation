# LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems

## 1) 项目简介 / Project Overview
**中文**  
本仓库实现了 **DualAgent-Rec**：一个面向电商推荐的、受约束的多目标优化框架。系统由两个进化智能体（Exploitation / Exploration）和一个可选的 LLM 协调器组成，在准确性、多样性、新颖性与约束可行性之间进行联合优化。  

**English**  
This repository implements **DualAgent-Rec**, a constrained multi-objective recommendation framework with two evolutionary agents (exploitation/exploration) and an optional LLM coordinator.

核心能力 / Key capabilities:
- 双智能体协同优化（利用 + 探索）
- LLM 资源分配协调（可开关）
- 硬约束处理（公平性、卖家覆盖、新品曝光）
- 代理指标与真实离线指标并行评估（`avg_accuracy` + `real_ndcg@10`）

---

## 2) 快速安装 / Quick Setup
**中文**
```bash
cd /home/linchengkai/Dual-Agents-Recommendation

# 建议 Python 3.10+
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

如需启用 LLM 协调器（Ollama）：
```bash
# 安装 Ollama 后执行
ollama pull qwen2.5:14b
ollama serve
```

**English**  
Create a Python virtual environment, install dependencies from `requirements.txt`, and (optionally) run Ollama with `qwen2.5:14b` for LLM coordination.

---

## 3) CLI 参数（当前有效） / Valid CLI Options
入口脚本 / Entry script: `src/run_experiments.py`

```bash
python src/run_experiments.py --help
```

当前有效参数（与代码一致）：
- `--data_dir`
- `--output_dir`
- `--categories`（`nargs='+'`，可传多个类别）
- `--n_users`
- `--use_llm`
- `--num_runs`
- `--max_reviews`
- `--experiment`（`all | main | ablation | quick`）

说明 / Notes:
- 本 README 已移除过时参数示例（如 `--pop_size`、`--llm_model` 的 CLI 透传写法等）。
- `--categories` 正确写法是空格分隔，例如：`--categories All_Beauty Electronics`。

---

## 4) 复现实验命令 / Reproducible Commands
### 4.1 快速检查 / Quick smoke run
```bash
python src/run_experiments.py \
  --experiment quick \
  --data_dir data \
  --categories All_Beauty \
  --n_users 1 \
  --max_reviews 3000 \
  --output_dir /tmp/dualagent_quick
```

### 4.2 主实验 / Main comparison
```bash
python src/run_experiments.py \
  --experiment main \
  --data_dir data \
  --categories All_Beauty \
  --n_users 20 \
  --num_runs 3 \
  --max_reviews 20000 \
  --output_dir results/main_all_beauty_$(date +%Y%m%d_%H%M%S)
```

### 4.3 Ablation / 消融实验
```bash
python src/run_experiments.py \
  --experiment ablation \
  --data_dir data \
  --categories All_Beauty \
  --n_users 20 \
  --max_reviews 20000 \
  --output_dir results/ablation_all_beauty_$(date +%Y%m%d_%H%M%S)
```

### 4.4 全流程 / All-in-one
```bash
python src/run_experiments.py \
  --experiment all \
  --data_dir data \
  --categories All_Beauty \
  --n_users 20 \
  --num_runs 3 \
  --max_reviews 20000 \
  --output_dir results/all_all_beauty_$(date +%Y%m%d_%H%M%S)
```

### 4.5 user=100 + LLM（推荐）
```bash
python src/run_experiments.py \
  --experiment main \
  --data_dir data \
  --categories All_Beauty \
  --n_users 100 \
  --num_runs 1 \
  --use_llm \
  --max_reviews 120000 \
  --output_dir results/main_all_beauty_u100_$(date +%Y%m%d_%H%M%S)
```

### 4.6 tmux 运行与查看（推荐长任务）
```bash
# 1) 后台启动 Ollama
tmux new-session -d -s ollama "ollama serve 2>&1 | tee -a logs/ollama_$(date +%Y%m%d_%H%M%S).log"

# 2) 后台启动实验
tmux new-session -d -s exp100 "
cd /home/linchengkai/Dual-Agents-Recommendation
TS=\$(date +%Y%m%d_%H%M%S)
OUT=results/main_all_beauty_u100_\$TS
LOG=logs/exp_u100_\$TS.log
python src/run_experiments.py --experiment main --data_dir data --categories All_Beauty --n_users 100 --num_runs 1 --use_llm --max_reviews 120000 --output_dir \$OUT 2>&1 | tee \$LOG
"

# 3) 查看会话/日志
tmux ls
tmux attach -t exp100
# 退出但不断开进程: Ctrl+b 然后 d
tail -f "$(ls -t logs/exp_u100_*.log | head -1)"
```

---

## 5) 输出产物说明 / Output Artifacts
主实验目录（`--output_dir`）下的核心文件：
- `main_comparison.json`
- `results_table.tex`
- `tradeoff_scatter.png`
- `acc_div_grouped_bar.png`
- `hv_feasibility_grouped_bar.png`
- `performance_radar.png`
- `dualagent_convergence_subplots.png`
- `real_ndcg_bar.png`

说明 / Notes:
- 图表由 `plot_paper_figures(...)` 自动生成（主实验流程中会调用）。
- 若运行 `ablation`，会额外输出 `ablation_table.tex` 等对应产物。

---

## 6) 指标与约束口径 / Metrics & Constraint Protocol
### 指标 / Metrics
- 代理优化指标（proxy）：`avg_accuracy`、`avg_diversity`、`avg_novelty`
- 真实离线指标（offline）：`real_ndcg@10`、`real_hr@10`

### 合法率口径 / Feasibility protocol
- `feasibility_rate`：统一严评口径（统一阈值、`epsilon=0`）
- `internal_feasibility_rate`：方法内部训练/搜索阶段的原始可行率（用于审计对照）

### 当前主实验阈值 / Current main thresholds
- `fairness_threshold = 0.30`
- `seller_coverage_threshold = 0.40`
- `new_item_threshold = 0.25`

---

## 7) 常见问题 / FAQ
**Q1: `--categories` 为什么不能写逗号字符串？**  
A: 因为该参数在代码中使用 `nargs='+'`，请用空格分隔，例如：  
`--categories All_Beauty Electronics`

**Q2: 默认 `--data_dir` 为什么可能找不到数据？**  
A: 默认值是 `../data`。若你在仓库根目录运行，推荐显式写 `--data_dir data`。

**Q3: 如何确认真的调用了 Ollama？**  
A: 需要同时满足：  
1) 运行命令带 `--use_llm`；  
2) `ollama serve` 正常；  
3) 日志中可见 `LLM Coordinator initialized ...` 或相关调用信息。  
否则会回退到启发式协调。

---

## 8) 结果解读声明 / Result Interpretation
本仓库 README 不承诺固定数值结论。  
建议按上述命令复现实验，并基于生成的 `main_comparison.json` 和图表进行解释。

This README intentionally avoids fixed headline numbers.  
Please reproduce experiments and interpret results from generated artifacts.

---

## License
MIT License. See [LICENSE](LICENSE).
