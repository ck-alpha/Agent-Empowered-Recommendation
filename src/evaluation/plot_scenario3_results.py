"""
Scenario-3 plotting utilities for MIND news baselines.

Recommended:
  python src/evaluation/plot_scenario3_results.py \
    --metrics_json results/scenario3_metrics_mind_smoke.json \
    --output_dir results/figures_scenario3_mind_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LOGGER = logging.getLogger(__name__)
METHOD_ORDER = ["raw_ranker", "hard_filter", "postprocessing", "inprocessing"]
METHOD_LABELS = {
    "raw_ranker": "Raw",
    "hard_filter": "Hard",
    "postprocessing": "Post",
    "inprocessing": "In",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scenario-3 MIND news baseline results.")
    parser.add_argument("--metrics_json", default="results/scenario3_metrics_mind_smoke.json")
    parser.add_argument("--output_dir", default="results/figures_scenario3_mind_smoke")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_metrics(metrics_json: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(metrics_json)
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    summaries = payload.get("summaries_by_method", {})
    if not records:
        raise ValueError("Metrics JSON has no records.")
    if not summaries:
        raise ValueError("Metrics JSON has no summaries_by_method.")
    summary_df = pd.DataFrame([dict(value, method=key) for key, value in summaries.items()])
    records_df = pd.DataFrame(records)
    exposure = payload.get("category_exposure_by_method", {})
    exposure_rows = []
    for method, counts in exposure.items():
        total = sum(float(value) for value in counts.values())
        for category, count in counts.items():
            exposure_rows.append(
                {
                    "method": method,
                    "category": category,
                    "count": float(count),
                    "share": float(count) / total if total > 0 else 0.0,
                }
            )
    exposure_df = pd.DataFrame(exposure_rows)
    for df in [summary_df, records_df]:
        df["method"] = pd.Categorical(df["method"].astype(str), categories=METHOD_ORDER, ordered=True)
        df["method_label"] = df["method"].astype(str).map(lambda value: METHOD_LABELS.get(value, value))
    if not exposure_df.empty:
        exposure_df["method"] = pd.Categorical(exposure_df["method"].astype(str), categories=METHOD_ORDER, ordered=True)
        exposure_df["method_label"] = exposure_df["method"].astype(str).map(lambda value: METHOD_LABELS.get(value, value))
    summary_df = summary_df.sort_values("method").reset_index(drop=True)
    records_df = records_df.sort_values("method").reset_index(drop=True)
    exposure_df = exposure_df.sort_values(["method", "share"], ascending=[True, False]).reset_index(drop=True) if not exposure_df.empty else exposure_df
    return payload.get("summary", {}), summary_df, records_df, exposure_df


def setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.05,
        rc={
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.fontsize": 8,
        },
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, fmt: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.{fmt}"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure: %s", path)
    return path


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_accuracy(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["ndcg_at_10", "mrr_at_10", "hit_at_10", "recall_at_10"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "ndcg_at_10": "NDCG@10",
            "mrr_at_10": "MRR@10",
            "hit_at_10": "Hit@10",
            "recall_at_10": "Recall@10",
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    sns.barplot(data=plot_df, x="method_label", y="Score", hue="Metric", ax=ax)
    ax.set_title("Scenario-3 Accuracy Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.05, float(plot_df["Score"].max()) * 1.2 if plot_df["Score"].notna().any() else 0.05))
    ax.legend(title="", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario3_accuracy_comparison", fmt, dpi)


def plot_constraint_compliance(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["feasible_rate", "freshness_violation_rate", "topn_topic_violation_rate", "candidate_shortage"]
    df = _numeric(summary_df, metrics).copy()
    df["freshness_pass_rate"] = 1.0 - df["freshness_violation_rate"]
    df["topn_topic_pass_rate"] = 1.0 - df["topn_topic_violation_rate"]
    plot_df = df.melt(
        id_vars="method_label",
        value_vars=["feasible_rate", "freshness_pass_rate", "topn_topic_pass_rate", "candidate_shortage"],
        var_name="Metric",
        value_name="Rate",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "feasible_rate": "CSR",
            "freshness_pass_rate": "Freshness",
            "topn_topic_pass_rate": "Top-N Topic",
            "candidate_shortage": "Shortage",
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    sns.barplot(data=plot_df, x="Metric", y="Rate", hue="method_label", ax=ax)
    ax.set_title("Scenario-3 Constraint Compliance")
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.legend(title="Method", loc="lower right", frameon=True)
    return save_figure(fig, output_dir, "scenario3_constraint_compliance", fmt, dpi)


def plot_ecology(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["topic_entropy", "topic_coverage_at_10", "avg_word_count", "load_deviation"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "topic_entropy": "Topic entropy",
            "topic_coverage_at_10": "Topic coverage",
            "avg_word_count": "Avg word count",
            "load_deviation": "Load deviation",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    for ax, metric in zip(axes.ravel(), ["Topic entropy", "Topic coverage", "Avg word count", "Load deviation"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(data=sub, x="method_label", y="Value", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    fig.suptitle("Scenario-3 List Ecology", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario3_list_ecology", fmt, dpi)


def plot_tradeoff(records_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    required = ["ndcg_at_10", "augmented_lagrangian_penalty", "method_label"]
    missing = [col for col in required if col not in records_df.columns]
    if missing:
        raise ValueError(f"Records missing columns for tradeoff plot: {missing}")
    df = _numeric(records_df, ["ndcg_at_10", "augmented_lagrangian_penalty"]).dropna(subset=required[:2])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    sns.scatterplot(
        data=df,
        x="augmented_lagrangian_penalty",
        y="ndcg_at_10",
        hue="method_label",
        alpha=0.35,
        s=18,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("Scenario-3 Utility vs. Soft Penalty")
    ax.set_xlabel("ALM soft penalty")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(-0.02, max(1.0, float(df["ndcg_at_10"].max()) + 0.05 if not df.empty else 1.0))
    ax.legend(title="Method", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario3_utility_penalty_tradeoff", fmt, dpi)


def plot_search_diagnostics(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["num_swaps", "search_steps", "final_list_size", "mean_base_score"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "num_swaps": "Swaps",
            "search_steps": "Search steps",
            "final_list_size": "List size",
            "mean_base_score": "Mean base score",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    for ax, metric in zip(axes.ravel(), ["Swaps", "Search steps", "List size", "Mean base score"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(data=sub, x="method_label", y="Value", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    fig.suptitle("Scenario-3 Algorithm Diagnostics", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario3_algorithm_diagnostics", fmt, dpi)


def plot_metric_distributions(records_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["ndcg_at_10", "feasible_rate", "topic_entropy", "augmented_lagrangian_penalty"]
    plot_df = _numeric(records_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "ndcg_at_10": "NDCG@10",
            "feasible_rate": "CSR",
            "topic_entropy": "Topic entropy",
            "augmented_lagrangian_penalty": "ALM penalty",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    for ax, metric in zip(axes.ravel(), ["NDCG@10", "CSR", "Topic entropy", "ALM penalty"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.boxplot(data=sub, x="method_label", y="Value", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    fig.suptitle("Scenario-3 Per-Impression Metric Distributions", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario3_metric_distributions", fmt, dpi)


def plot_paired_delta(records_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["ndcg_at_10", "feasible_rate", "topic_entropy", "augmented_lagrangian_penalty"]
    df = _numeric(records_df, metrics)
    rows = []
    for metric in metrics:
        pivot = df.pivot_table(index="impression_id", columns="method", values=metric, aggfunc="first", observed=False)
        if "raw_ranker" not in pivot.columns:
            continue
        for method in METHOD_ORDER:
            if method == "raw_ranker" or method not in pivot.columns:
                continue
            delta = pivot[method] - pivot["raw_ranker"]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "Metric": metric,
                    "Mean delta": float(delta.mean()),
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError("Cannot build paired delta plot without raw_ranker and comparison methods.")
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "ndcg_at_10": "NDCG@10",
            "feasible_rate": "CSR",
            "topic_entropy": "Topic entropy",
            "augmented_lagrangian_penalty": "ALM penalty",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    for ax, metric in zip(axes.ravel(), ["NDCG@10", "CSR", "Topic entropy", "ALM penalty"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(data=sub, x="method_label", y="Mean delta", ax=ax)
        ax.axhline(0.0, color="#4A4A4A", linewidth=1.0)
        ax.set_title(f"Delta vs Raw: {metric}")
        ax.set_xlabel("Method")
        ax.set_ylabel("Mean delta")
    fig.suptitle("Scenario-3 Paired Changes Relative to Raw Ranker", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario3_paired_delta_vs_raw", fmt, dpi)


def plot_category_exposure(exposure_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int, top_categories: int = 8) -> Path:
    if exposure_df.empty:
        raise ValueError("Metrics JSON has no category_exposure_by_method for category exposure plot.")
    top = (
        exposure_df.groupby("category", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(top_categories)["category"]
        .tolist()
    )
    df = exposure_df.copy()
    df["category_plot"] = df["category"].where(df["category"].isin(top), "other")
    plot_df = (
        df.groupby(["method_label", "category_plot"], as_index=False)["share"]
        .sum()
        .sort_values(["method_label", "share"], ascending=[True, False])
    )
    pivot = plot_df.pivot(index="method_label", columns="category_plot", values="share").fillna(0.0)
    ordered_methods = [METHOD_LABELS[method] for method in METHOD_ORDER if METHOD_LABELS[method] in pivot.index]
    pivot = pivot.loc[ordered_methods]
    ordered_cols = [cat for cat in top if cat in pivot.columns] + (["other"] if "other" in pivot.columns else [])
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bottom = pd.Series(0.0, index=pivot.index)
    palette = sns.color_palette("tab20", n_colors=len(pivot.columns))
    for color, category in zip(palette, pivot.columns):
        ax.bar(pivot.index, pivot[category], bottom=bottom, label=category, color=color)
        bottom = bottom + pivot[category]
    ax.set_title("Scenario-3 Final Recommendation Category Exposure")
    ax.set_xlabel("Method")
    ax.set_ylabel("Exposure share")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)
    return save_figure(fig, output_dir, "scenario3_category_exposure", fmt, dpi)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    setup_style()
    _, summary_df, records_df, exposure_df = load_metrics(args.metrics_json)
    output_dir = Path(args.output_dir)
    plot_accuracy(summary_df, output_dir, args.format, args.dpi)
    plot_constraint_compliance(summary_df, output_dir, args.format, args.dpi)
    plot_ecology(summary_df, output_dir, args.format, args.dpi)
    plot_tradeoff(records_df, output_dir, args.format, args.dpi)
    plot_search_diagnostics(summary_df, output_dir, args.format, args.dpi)
    plot_metric_distributions(records_df, output_dir, args.format, args.dpi)
    plot_paired_delta(records_df, output_dir, args.format, args.dpi)
    if exposure_df.empty:
        LOGGER.warning("Skipping category exposure plot because metrics JSON has no category_exposure_by_method.")
    else:
        plot_category_exposure(exposure_df, output_dir, args.format, args.dpi)


if __name__ == "__main__":
    main()
