# Results Directory

## 中文说明
本目录用于存放 `src/run_experiments.py` 运行后的实验输出。

### 主实验（`--experiment main`）典型输出
- `main_comparison.json`
- `results_table.tex`
- `tradeoff_scatter.png`
- `acc_div_grouped_bar.png`
- `hv_feasibility_grouped_bar.png`
- `performance_radar.png`
- `dualagent_convergence_subplots.png`
- `real_ndcg_bar.png`

### 其他实验输出
- `--experiment ablation`：会生成消融相关结果（例如 `ablation_table.tex`）。
- `--experiment all`：包含主实验 + 消融实验产物。

说明：
- 图表由主实验流程自动调用 `plot_paper_figures(...)` 生成。
- 当前项目中不存在 `--visualize` 参数，请勿使用旧文档中的可视化命令。

---

## English
This folder stores artifacts produced by `src/run_experiments.py`.

### Typical outputs for `--experiment main`
- `main_comparison.json`
- `results_table.tex`
- `tradeoff_scatter.png`
- `acc_div_grouped_bar.png`
- `hv_feasibility_grouped_bar.png`
- `performance_radar.png`
- `dualagent_convergence_subplots.png`
- `real_ndcg_bar.png`

Notes:
- Figures are generated automatically in the main pipeline via `plot_paper_figures(...)`.
- The `--visualize` flag is not available in the current codebase.
