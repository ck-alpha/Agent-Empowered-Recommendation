"""
Scenario-2 plotting utilities for MIND news baselines.

Recommended:
  python src/evaluation/plot_scenario2_results.py \
    --metrics_json results/scenario2_metrics_mind_smoke.json \
    --output_dir results/figures_scenario2_mind_smoke
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
METHOD_ORDER = ["raw_ranker", "postprocessing", "inprocessing"]
METHOD_LABELS = {
    "raw_ranker": "Raw",
    "postprocessing": "Post",
    "inprocessing": "In",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scenario-2 MIND news baseline results.")
    parser.add_argument("--metrics_json", default="results/scenario2_metrics_mind_smoke.json")
    parser.add_argument("--output_dir", default="results/figures_scenario2_mind_smoke")
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
    exposure_rows = []
    for method, counts in payload.get("category_exposure_by_method", {}).items():
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
    if not exposure_df.empty:
        exposure_df = exposure_df.sort_values(["method", "share"], ascending=[True, False]).reset_index(drop=True)
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
    ax.set_title("Scenario-2 News Accuracy")
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.05, float(plot_df["Score"].max()) * 1.2 if plot_df["Score"].notna().any() else 0.05))
    ax.legend(title="", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario2_news_accuracy_comparison", fmt, dpi)


def plot_entropy(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["topic_entropy", "topic_coverage_at_10", "diversity_penalty", "augmented_lagrangian_penalty"]
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
            "diversity_penalty": "Diversity penalty",
            "augmented_lagrangian_penalty": "ALM penalty",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    for ax, metric in zip(axes.ravel(), ["Topic entropy", "Topic coverage", "Diversity penalty", "ALM penalty"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(data=sub, x="method_label", y="Value", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    fig.suptitle("Scenario-2 News Topic Diversity", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario2_news_topic_diversity", fmt, dpi)


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
    ax.set_title("Scenario-2 News Utility vs. Entropy Penalty")
    ax.set_xlabel("ALM entropy penalty")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(-0.02, max(1.0, float(df["ndcg_at_10"].max()) + 0.05 if not df.empty else 1.0))
    ax.legend(title="Method", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario2_news_utility_penalty_tradeoff", fmt, dpi)


def plot_search_diagnostics(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["num_swaps", "search_steps", "final_list_size", "mean_base_score", "final_objective"]
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
            "final_objective": "Final objective",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0))
    for ax, metric in zip(axes.ravel(), ["Swaps", "Search steps", "List size", "Mean base score", "Final objective"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(data=sub, x="method_label", y="Value", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Scenario-2 News Algorithm Diagnostics", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario2_news_algorithm_diagnostics", fmt, dpi)


def plot_metric_distributions(records_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["ndcg_at_10", "topic_entropy", "diversity_penalty", "augmented_lagrangian_penalty"]
    plot_df = _numeric(records_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "ndcg_at_10": "NDCG@10",
            "topic_entropy": "Topic entropy",
            "diversity_penalty": "Diversity penalty",
            "augmented_lagrangian_penalty": "ALM penalty",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    for ax, metric in zip(axes.ravel(), ["NDCG@10", "Topic entropy", "Diversity penalty", "ALM penalty"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.boxplot(data=sub, x="method_label", y="Value", ax=ax, fliersize=1.5)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    fig.suptitle("Scenario-2 News Metric Distributions", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario2_news_metric_distributions", fmt, dpi)


def plot_category_exposure(exposure_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    if exposure_df.empty:
        raise ValueError("No category exposure data available.")
    top_categories = (
        exposure_df.groupby("category", observed=False)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )
    df = exposure_df.loc[exposure_df["category"].isin(top_categories)].copy()
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    sns.barplot(data=df, x="category", y="share", hue="method_label", ax=ax)
    ax.set_title("Scenario-2 News Category Exposure")
    ax.set_xlabel("Category")
    ax.set_ylabel("Exposure share")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Method", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario2_news_category_exposure", fmt, dpi)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    setup_style()
    _, summary_df, records_df, exposure_df = load_metrics(args.metrics_json)
    output_dir = Path(args.output_dir)
    plot_accuracy(summary_df, output_dir, args.format, args.dpi)
    plot_entropy(summary_df, output_dir, args.format, args.dpi)
    plot_tradeoff(records_df, output_dir, args.format, args.dpi)
    plot_search_diagnostics(summary_df, output_dir, args.format, args.dpi)
    plot_metric_distributions(records_df, output_dir, args.format, args.dpi)
    if not exposure_df.empty:
        plot_category_exposure(exposure_df, output_dir, args.format, args.dpi)


if __name__ == "__main__":
    main()
