"""Aggregate COPA core experiments, statistics, figures, and Chinese report."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


PHASE1_METRICS = [
    "recall_at_k",
    "ndcg_at_k",
    "objective_diversity",
    "objective_novelty",
    "strict_constraint_satisfaction_rate",
    "strict_violation_rate",
    "shortage",
    "actual_k_ratio",
    "strict_candidate_filter_rate",
    "candidate_recall",
    "target_in_feasible_domain",
    "retrieval_loss",
    "constraint_filter_loss",
    "ranking_loss",
    "shared_hypervolume",
    "shared_spacing",
    "pareto_front_size",
    "runtime_seconds",
    "peak_memory_mb",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a COPA core result root")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    return series.map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else float(str(value).strip().casefold() in {"true", "1", "1.0"})
        )
    )


def _holm(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def phase1_statistics(frame: pd.DataFrame, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = [metric for metric in PHASE1_METRICS if metric in frame.columns]
    user_means = frame.groupby(["experiment", "method", "user_id"], as_index=False)[available].mean()
    rng = np.random.default_rng(42)
    aggregate = []
    for (experiment, method), group in user_means.groupby(["experiment", "method"], sort=True):
        for metric in available:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            indices = rng.integers(0, len(values), size=(samples, len(values)))
            boot = values[indices].mean(axis=1)
            aggregate.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "user_count": len(values),
                }
            )

    comparisons = {
        "A": [("feasible_relevance", "unconstrained_relevance")],
        "B": [
            ("unconstrained_copa", "unconstrained_relevance"),
            ("unconstrained_copa", "unconstrained_weighted_ga"),
        ],
        "C": [
            ("copa", "feasible_relevance"),
            ("copa", "feasible_weighted_ga"),
        ],
    }
    tests = []
    test_metrics = [
        "recall_at_k",
        "ndcg_at_k",
        "objective_diversity",
        "objective_novelty",
        "strict_constraint_satisfaction_rate",
        "strict_violation_rate",
        "shared_hypervolume",
    ]
    for experiment, pairs in comparisons.items():
        current = user_means[user_means["experiment"] == experiment]
        for left, right in pairs:
            for metric in test_metrics:
                if metric not in current:
                    continue
                pivot = current.pivot(index="user_id", columns="method", values=metric)
                if left not in pivot or right not in pivot:
                    continue
                differences = (pivot[left] - pivot[right]).dropna().to_numpy(float)
                nonzero = differences[differences != 0]
                if not len(nonzero):
                    statistic, p_value, effect = 0.0, 1.0, 0.0
                else:
                    result = wilcoxon(differences, zero_method="pratt", alternative="two-sided")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
                    positive = float(ranks[nonzero > 0].sum())
                    negative = float(ranks[nonzero < 0].sum())
                    effect = (positive - negative) / (positive + negative)
                tests.append(
                    {
                        "test": "paired_wilcoxon",
                        "experiment": experiment,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "n": len(differences),
                        "mean_difference": float(np.mean(differences)),
                        "statistic": statistic,
                        "p_value": p_value,
                        "effect_rank_biserial": effect,
                    }
                )
    tests_frame = pd.DataFrame(tests)
    if len(tests_frame):
        tests_frame["p_holm"] = _holm(tests_frame["p_value"].tolist())
    return pd.DataFrame(aggregate), tests_frame


def load_repeats(root: Path, filename: str) -> pd.DataFrame:
    records = []
    for path in sorted(root.glob(f"repeat_*/{filename}")):
        repeat = int(path.parent.name.split("_")[-1])
        frame = pd.read_csv(path)
        frame.insert(0, "repeat", repeat)
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def phase2_aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    booleans = [
        "parsing_exact",
        "schema_valid",
        "status_correct",
        "hard_exact",
        "objective_exact",
        "top_k_correct",
        "execution_exact",
    ]
    for column in booleans:
        if column in frame:
            frame[column] = _bool_series(frame[column])
    rows = []
    for dimensions in [("scenario",), ("language",)]:
        for keys, group in frame.groupby(list(dimensions), sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = {"group_type": "+".join(dimensions), "group": "+".join(map(str, keys))}
            for metric in booleans:
                if metric in group:
                    rows.append({**base, "metric": metric, "mean": float(group[metric].mean()), "n": len(group)})
            rows.extend(
                [
                    {**base, "metric": "retry_rate", "mean": float((group["attempts"] > 1).mean()), "n": len(group)},
                    {**base, "metric": "latency_p50_seconds", "mean": float(group["latency_seconds"].quantile(0.5)), "n": len(group)},
                    {**base, "metric": "latency_p95_seconds", "mean": float(group["latency_seconds"].quantile(0.95)), "n": len(group)},
                    {**base, "metric": "mean_prompt_tokens", "mean": float(group["prompt_tokens"].mean()), "n": len(group)},
                    {**base, "metric": "mean_output_tokens", "mean": float(group["output_tokens"].mean()), "n": len(group)},
                ]
            )
    signature_columns = [
        column
        for column in ["status", "predicted_constraints", "predicted_objectives"]
        if column in frame
    ]
    signatures = frame[signature_columns].astype(str).agg("|".join, axis=1)
    consistency = frame.assign(_signature=signatures).groupby("id")["_signature"].nunique()
    consistency_frame = consistency.rename("unique_outputs").reset_index()
    consistency_frame["exact_repeat_consistent"] = consistency_frame["unique_outputs"] == 1
    return pd.DataFrame(rows), consistency_frame


def phase3_aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame["success"] = (frame["status"] == "success").astype(float)
    frame["verified_feasible"] = _bool_series(frame["verified_feasible"])
    metrics = [
        "success",
        "verified_feasible",
        "recall_at_k",
        "ndcg_at_k",
        "diversity",
        "novelty",
        "tool_call_count",
        "repair_count",
        "llm_call_count",
        "prompt_tokens",
        "output_tokens",
        "latency_seconds",
    ]
    rows = []
    for mode, group in frame.groupby("mode", sort=True):
        for metric in metrics:
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=1)),
                    "n": len(group),
                }
            )
    signatures = frame.assign(_signature=frame["status"].astype(str)).groupby(["id", "mode"])["_signature"].nunique()
    consistency = signatures.rename("unique_statuses").reset_index()
    consistency["status_repeat_consistent"] = consistency["unique_statuses"] == 1

    per_repeat = frame.groupby(["id", "mode"], as_index=False)["success"].mean()
    tests = []
    for left, right in combinations(sorted(frame["mode"].unique()), 2):
        pivot = per_repeat.pivot(index="id", columns="mode", values="success")
        left_success = pivot[left] >= 0.5
        right_success = pivot[right] >= 0.5
        left_only = int((left_success & ~right_success).sum())
        right_only = int((~left_success & right_success).sum())
        discordant = left_only + right_only
        p_value = float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue) if discordant else 1.0
        tests.append(
            {
                "test": "paired_mcnemar_exact",
                "left": left,
                "right": right,
                "left_only_success": left_only,
                "right_only_success": right_only,
                "discordant": discordant,
                "p_value": p_value,
            }
        )
    tests_frame = pd.DataFrame(tests)
    if len(tests_frame):
        tests_frame["p_holm"] = _holm(tests_frame["p_value"].tolist())
    return pd.DataFrame(rows), consistency, tests_frame


def _save_figure(fig: plt.Figure, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _bar_panels(aggregate: pd.DataFrame, metrics: Sequence[str], figures: Path, name: str) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        data = aggregate[aggregate["metric"] == metric].copy()
        labels = [f"{row.experiment}:{row.method}" for row in data.itertuples()]
        errors = np.vstack(
            [data["mean"] - data["ci95_low"], data["ci95_high"] - data["mean"]]
        ) if len(data) else None
        axis.bar(np.arange(len(data)), data["mean"], yerr=errors, capsize=2)
        axis.set_xticks(np.arange(len(data)), labels, rotation=65, ha="right", fontsize=7)
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, figures, name)


def phase1_figures(root: Path, frame: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    figures = root / "figures"
    _bar_panels(
        aggregate,
        ["recall_at_k", "ndcg_at_k", "objective_diversity", "objective_novelty"],
        figures,
        "phase1_quality",
    )
    _bar_panels(
        aggregate,
        ["strict_constraint_satisfaction_rate", "strict_violation_rate", "shortage", "candidate_recall"],
        figures,
        "phase1_constraints_coverage",
    )
    _bar_panels(
        aggregate,
        ["retrieval_loss", "constraint_filter_loss", "ranking_loss", "actual_k_ratio"],
        figures,
        "phase1_loss_decomposition",
    )
    _bar_panels(
        aggregate,
        ["shared_hypervolume", "shared_spacing", "pareto_front_size", "runtime_seconds"],
        figures,
        "phase1_pareto_runtime",
    )

    generation_path = root / "phase1" / "generation_metrics.csv"
    if generation_path.exists():
        generation = pd.read_csv(generation_path)
        grouped = generation.groupby(["method", "generation"], as_index=False).mean(numeric_only=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for method, group in grouped.groupby("method"):
            axes[0, 0].plot(group["generation"], group["pareto_size"], label=method)
            axes[0, 1].plot(group["generation"], group["evaluations"], label=method)
            if "relevance_max" in group:
                axes[1, 0].plot(group["generation"], group["relevance_max"], label=method)
            if "diversity_max" in group:
                axes[1, 1].plot(group["generation"], group["diversity_max"], label=method)
        titles = ["Pareto Size", "Cumulative Evaluations", "Max Relevance", "Max Diversity"]
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title)
            axis.set_xlabel("Generation")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        _save_figure(fig, figures, "phase1_convergence")

    front_path = root / "phase1" / "pareto_fronts.jsonl"
    if front_path.exists():
        points = []
        with front_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= 1200:
                    break
                record = json.loads(line)
                if record["method"] not in {"copa", "feasible_weighted_ga", "unconstrained_copa", "unconstrained_weighted_ga"}:
                    continue
                for solution in record["front"][:20]:
                    values = solution["objective_values"]
                    points.append({"method": record["method"], **values})
        if points:
            point_frame = pd.DataFrame(points)
            fig, axis = plt.subplots(figsize=(7, 5))
            for method, group in point_frame.groupby("method"):
                axis.scatter(group["relevance"], group["diversity"], s=8, alpha=0.35, label=method)
            axis.set_xlabel("Relevance")
            axis.set_ylabel("Diversity")
            axis.set_title("Final-Population Nondominated Projections")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
            _save_figure(fig, figures, "phase1_pareto_projection")


def phase2_figures(root: Path, aggregate: pd.DataFrame, consistency: pd.DataFrame) -> None:
    figures = root / "figures"
    metrics = ["parsing_exact", "execution_exact", "status_correct", "top_k_correct"]
    data = aggregate[(aggregate["group_type"] == "scenario") & aggregate["metric"].isin(metrics)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot = data.pivot(index="group", columns="metric", values="mean")
    pivot.plot(kind="bar", ax=axes[0])
    language = aggregate[(aggregate["group_type"] == "language") & aggregate["metric"].isin(metrics)]
    language.pivot(index="group", columns="metric", values="mean").plot(kind="bar", ax=axes[1])
    for axis, title in zip(axes, ["By Scenario", "By Language"]):
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, figures, "phase2_compiler_quality")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    latency = aggregate[(aggregate["group_type"] == "scenario") & aggregate["metric"].str.startswith("latency")]
    latency.pivot(index="group", columns="metric", values="mean").plot(kind="bar", ax=axes[0])
    axes[0].set_title("Compiler Latency")
    axes[0].set_ylabel("Seconds")
    rate = float(consistency["exact_repeat_consistent"].mean()) if len(consistency) else 0.0
    axes[1].bar(["Exact repeat consistency"], [rate])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Three-Run Reproducibility")
    _save_figure(fig, figures, "phase2_latency_consistency")

    cases = pd.read_csv(root / "analysis" / "phase2_all_cases.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    issue_counts = cases["issue_codes"].fillna("none").replace("", "none").value_counts().head(8)
    issue_counts.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Top Issue Signatures")
    axes[0].tick_params(axis="x", labelrotation=65, labelsize=7)
    axes[1].bar(["first attempt", "retry"], [(cases["attempts"] == 1).mean(), (cases["attempts"] > 1).mean()])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Retry Distribution")
    axes[2].bar(["prompt", "output"], [cases["prompt_tokens"].mean(), cases["output_tokens"].mean()])
    axes[2].set_title("Mean Tokens per Case")
    _save_figure(fig, figures, "phase2_errors_retries_tokens")


def phase3_figures(
    root: Path,
    aggregate: pd.DataFrame,
    consistency: pd.DataFrame,
    agent_cases: pd.DataFrame,
) -> None:
    figures = root / "figures"
    quality = aggregate[aggregate["metric"].isin(["success", "verified_feasible", "recall_at_k", "ndcg_at_k", "diversity", "novelty"])]
    fig, axis = plt.subplots(figsize=(10, 5))
    quality.pivot(index="mode", columns="metric", values="mean").plot(kind="bar", ax=axis)
    axis.set_title("Phase 3 Mode Quality and Safety")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, figures, "phase3_mode_quality")

    cost = aggregate[aggregate["metric"].isin(["llm_call_count", "tool_call_count", "repair_count", "latency_seconds"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cost[cost["metric"] != "latency_seconds"].pivot(index="mode", columns="metric", values="mean").plot(kind="bar", ax=axes[0])
    tokens = aggregate[aggregate["metric"].isin(["prompt_tokens", "output_tokens"])]
    tokens.pivot(index="mode", columns="metric", values="mean").plot(kind="bar", ax=axes[1])
    cost[cost["metric"] == "latency_seconds"].set_index("mode")["mean"].plot(kind="bar", ax=axes[2])
    axes[0].set_title("Tools and LLM Calls")
    axes[1].set_title("Mean Tokens per Case")
    axes[2].set_title("End-to-End Latency")
    axes[2].set_ylabel("Seconds")
    _save_figure(fig, figures, "phase3_cost_latency")

    rates = consistency.groupby("mode")["status_repeat_consistent"].mean()
    fig, axis = plt.subplots(figsize=(7, 4))
    rates.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Exact status consistency")
    axis.set_title("Phase 3 Three-Run Reproducibility")
    _save_figure(fig, figures, "phase3_consistency")

    if len(agent_cases):
        expected = agent_cases["expected_repair_action"].fillna("none").astype(str)
        actual = agent_cases["repair_actions"].fillna("[]").map(
            lambda value: (json.loads(value)[0] if json.loads(value) else "none")
        )
        confusion = pd.crosstab(expected, actual)
        confusion.to_csv(root / "analysis" / "phase3_repair_confusion.csv")
        fig, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(confusion.to_numpy(), cmap="Blues")
        axis.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(confusion.index)), confusion.index)
        axis.set_xlabel("Predicted first repair action")
        axis.set_ylabel("Expected repair action")
        axis.set_title("Phase 3 Repair Action Confusion")
        for row in range(len(confusion.index)):
            for column in range(len(confusion.columns)):
                axis.text(column, row, int(confusion.iloc[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=axis)
        _save_figure(fig, figures, "phase3_repair_confusion")


def resource_figures(root: Path) -> None:
    path = root / "resource_usage.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if not len(frame):
        return
    frame["elapsed_minutes"] = (pd.to_datetime(frame["timestamp"]) - pd.to_datetime(frame["timestamp"]).min()).dt.total_seconds() / 60
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    mappings = [
        ("cpu_percent", "CPU %"),
        ("rss_mb", "Experiment RSS MB"),
        ("gpu_util_percent", "GPU %"),
        ("gpu_memory_mb", "GPU Memory MB"),
    ]
    for axis, (column, title) in zip(axes.flat, mappings):
        if column in frame:
            axis.plot(frame["elapsed_minutes"], frame[column])
        axis.set_title(title)
        axis.set_xlabel("Elapsed minutes")
        axis.grid(alpha=0.25)
    _save_figure(fig, root / "figures", "resource_timeline")
    status_path = root / "stage_status.jsonl"
    if status_path.exists():
        records = [json.loads(line) for line in status_path.read_text().splitlines() if line.strip()]
        completed = pd.DataFrame([record for record in records if "duration_seconds" in record])
        if len(completed):
            fig, axis = plt.subplots(figsize=(10, 5))
            completed.groupby("stage")["duration_seconds"].sum().sort_values().plot(kind="barh", ax=axis)
            axis.set_xlabel("Wall-clock seconds")
            axis.set_title("Stage Runtime Breakdown")
            _save_figure(fig, root / "figures", "stage_runtime_breakdown")


def write_report(
    root: Path,
    phase1: pd.DataFrame,
    phase1_aggregate: pd.DataFrame,
    phase2: pd.DataFrame,
    phase2_consistency: pd.DataFrame,
    phase3: pd.DataFrame,
    phase3_consistency: pd.DataFrame,
) -> None:
    lines = [
        "# COPA 核心主实验报告",
        "",
        "> 本报告由保存的原始 CSV/JSONL 确定性生成；图中文字使用英文，PNG 与 PDF 同时保留。",
        "",
        "## 工程验证与数据完整性",
        "",
        f"- Phase 1 记录数：{len(phase1)}。",
        f"- Phase 2 记录数：{len(phase2)}。",
        f"- Phase 3 模式比较记录数：{len(phase3)}。",
    ]
    if len(phase1_aggregate):
        selected = phase1_aggregate[
            phase1_aggregate["metric"].isin(
                ["recall_at_k", "ndcg_at_k", "strict_constraint_satisfaction_rate", "shared_hypervolume"]
            )
        ][["experiment", "method", "metric", "mean", "ci95_low", "ci95_high"]]
        lines.extend(["", "## Phase 1 主结果", "", _markdown_table(selected)])
    if len(phase2):
        lines.extend(
            [
                "",
                "## Phase 2 编译器",
                "",
                f"- 三次运行完全一致样例比例：{phase2_consistency['exact_repeat_consistent'].mean():.4f}。",
                f"- 总体 parsing accuracy：{_bool_series(phase2['parsing_exact']).mean():.4f}。",
                f"- 总体 constraint execution accuracy：{_bool_series(phase2['execution_exact']).mean():.4f}。",
            ]
        )
    if len(phase3):
        mode_table = phase3.assign(success=(phase3["status"] == "success").astype(float)).groupby("mode").agg(
            completion_rate=("success", "mean"),
            verified_rate=("verified_feasible", lambda values: _bool_series(values).mean()),
            latency_seconds=("latency_seconds", "mean"),
        ).reset_index()
        lines.extend(
            [
                "",
                "## Phase 3 Agent 模式比较",
                "",
                _markdown_table(mode_table),
                "",
                f"- 三次运行状态一致率：{phase3_consistency['status_repeat_consistent'].mean():.4f}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 结果支持单用户静态候选池上的可验证约束重排与多目标优化，不代表跨用户库存或全局曝光约束。",
            "- All Beauty 的 diversity 是品牌多样性，不应解释为语义多样性或社会公平性。",
            "- Qwen 仅负责约束编译、受限规划和修复决策，不直接生成物品或排序。",
            "- 本实验未包含 Electronics、MIND、外部召回模型或大规模超参数消融。",
            "",
            "## 可复现入口",
            "",
            "所有聚合表位于 `analysis/`，图表位于 `figures/`；重新运行：",
            "",
            "```bash",
            f"python -m copa.experiments.analyze_core --input-dir {root}",
            "```",
        ]
    )
    (root / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(root: Path, *, bootstrap_samples: int, allow_partial: bool) -> dict[str, Any]:
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    phase1_path = root / "phase1" / "per_user_metrics.csv"
    if not phase1_path.exists() and not allow_partial:
        raise FileNotFoundError(phase1_path)
    phase1 = pd.read_csv(phase1_path) if phase1_path.exists() else pd.DataFrame()
    phase1_aggregate = phase1_tests = pd.DataFrame()
    if len(phase1):
        phase1_aggregate, phase1_tests = phase1_statistics(phase1, bootstrap_samples)
        phase1_aggregate.to_csv(analysis / "phase1_aggregate_metrics.csv", index=False)
        phase1_tests.to_csv(analysis / "phase1_statistical_tests.csv", index=False)
        phase1_figures(root, phase1, phase1_aggregate)

    phase2 = load_repeats(root / "phase2", "compiler_case_results.csv")
    phase2_aggregate_frame = phase2_consistency = pd.DataFrame()
    if len(phase2):
        phase2.to_csv(analysis / "phase2_all_cases.csv", index=False)
        phase2_aggregate_frame, phase2_consistency = phase2_aggregate(phase2)
        phase2_aggregate_frame.to_csv(analysis / "phase2_aggregate_metrics.csv", index=False)
        phase2_consistency.to_csv(analysis / "phase2_repeat_consistency.csv", index=False)
        phase2_figures(root, phase2_aggregate_frame, phase2_consistency)
    elif not allow_partial:
        raise FileNotFoundError(root / "phase2" / "repeat_1" / "compiler_case_results.csv")

    phase3 = load_repeats(root / "phase3", "mode_comparison.csv")
    phase3_agent = load_repeats(root / "phase3", "agent_case_results.csv")
    phase3_aggregate_frame = phase3_consistency = phase3_tests = pd.DataFrame()
    if len(phase3):
        phase3.to_csv(analysis / "phase3_all_mode_cases.csv", index=False)
        phase3_agent.to_csv(analysis / "phase3_all_agent_cases.csv", index=False)
        phase3_aggregate_frame, phase3_consistency, phase3_tests = phase3_aggregate(phase3)
        phase3_aggregate_frame.to_csv(analysis / "phase3_aggregate_metrics.csv", index=False)
        phase3_consistency.to_csv(analysis / "phase3_repeat_consistency.csv", index=False)
        phase3_tests.to_csv(analysis / "phase3_mcnemar_tests.csv", index=False)
        phase3_figures(root, phase3_aggregate_frame, phase3_consistency, phase3_agent)
    elif not allow_partial:
        raise FileNotFoundError(root / "phase3" / "repeat_1" / "mode_comparison.csv")
    resource_figures(root)
    write_report(
        root,
        phase1,
        phase1_aggregate,
        phase2,
        phase2_consistency,
        phase3,
        phase3_consistency,
    )
    result = {
        "phase1_records": len(phase1),
        "phase2_records": len(phase2),
        "phase3_mode_records": len(phase3),
        "figure_png_count": len(list((root / "figures").glob("*.png"))),
        "figure_pdf_count": len(list((root / "figures").glob("*.pdf"))),
    }
    (analysis / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        Path(args.input_dir),
        bootstrap_samples=args.bootstrap_samples,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
