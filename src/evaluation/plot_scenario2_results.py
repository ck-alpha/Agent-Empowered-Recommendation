"""
Scenario-2 plotting utilities for Djinni recruitment baselines.

Recommended:
  python src/evaluation/plot_scenario2_results.py \
    --metrics_json results/scenario2_metrics_djinni_smoke.json \
    --output_dir results/figures_scenario2_djinni_smoke
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
METHOD_ORDER = ["postprocessing", "inprocessing"]
METHOD_LABELS = {"postprocessing": "Post", "inprocessing": "In"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scenario-2 Djinni recruitment baseline results.")
    parser.add_argument("--metrics_json", default="results/scenario2_metrics_djinni_smoke.json")
    parser.add_argument("--output_dir", default="results/figures_scenario2_djinni_smoke")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_metrics(metrics_json: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
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
    for df in [summary_df, records_df]:
        df["method"] = pd.Categorical(df["method"].astype(str), categories=METHOD_ORDER, ordered=True)
        df["method_label"] = df["method"].astype(str).map(lambda value: METHOD_LABELS.get(value, value))
    summary_df = summary_df.sort_values("method").reset_index(drop=True)
    return payload.get("summary", {}), summary_df, records_df


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
    metrics = ["raw_hit_at_10", "raw_ndcg_at_10", "final_hit_at_10", "final_ndcg_at_10"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "raw_hit_at_10": "Raw HR@10",
            "raw_ndcg_at_10": "Raw NDCG@10",
            "final_hit_at_10": "Final HR@10",
            "final_ndcg_at_10": "Final NDCG@10",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    sns.barplot(data=plot_df, x="method_label", y="Score", hue="Metric", ax=ax)
    ax.set_title("Scenario-2 Accuracy Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.05, float(plot_df["Score"].max()) * 1.2 if plot_df["Score"].notna().any() else 0.05))
    ax.legend(title="", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario2_accuracy_comparison", fmt, dpi)


def plot_constraint_compliance(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["qualification_pass_rate", "capacity_satisfied", "fully_repaired"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Rate",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "qualification_pass_rate": "Qualification",
            "capacity_satisfied": "Capacity",
            "fully_repaired": "Overall CSR",
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    sns.barplot(data=plot_df, x="Metric", y="Rate", hue="method_label", ax=ax)
    ax.set_title("Scenario-2 Constraint Compliance")
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.legend(title="Method", loc="lower right", frameon=True)
    return save_figure(fig, output_dir, "scenario2_constraint_compliance", fmt, dpi)


def plot_congestion(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["capacity_violation_total", "over_capacity_job_count", "max_capacity_overflow", "congestion_rate"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "capacity_violation_total": "Total overflow",
            "over_capacity_job_count": "Over-cap jobs",
            "max_capacity_overflow": "Max overflow",
            "congestion_rate": "Congestion rate",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    left = plot_df[plot_df["Metric"].isin(["Total overflow", "Over-cap jobs", "Max overflow"])]
    right = plot_df[plot_df["Metric"] == "Congestion rate"]
    sns.barplot(data=left, x="Metric", y="Value", hue="method_label", ax=axes[0])
    axes[0].set_title("Capacity Overflow")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="Method", frameon=True)
    sns.barplot(data=right, x="method_label", y="Value", ax=axes[1])
    axes[1].set_title("Congestion Rate")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, max(1.0, float(right["Value"].max()) * 1.2 if not right.empty else 1.0))
    return save_figure(fig, output_dir, "scenario2_congestion_reduction", fmt, dpi)


def plot_mismatch_tradeoff(records_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    required = ["final_ndcg_at_10", "salary_mismatch_penalty", "skill_mismatch_penalty", "method_label"]
    missing = [col for col in required if col not in records_df.columns]
    if missing:
        raise ValueError(f"Records missing columns for mismatch tradeoff: {missing}")
    df = _numeric(records_df, ["final_ndcg_at_10", "salary_mismatch_penalty", "skill_mismatch_penalty"])
    df["mismatch"] = 0.5 * df["salary_mismatch_penalty"] + 0.5 * df["skill_mismatch_penalty"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    sns.scatterplot(data=df, x="mismatch", y="final_ndcg_at_10", hue="method_label", alpha=0.35, s=18, linewidth=0, ax=ax)
    ax.set_title("Scenario-2 Mismatch vs. Utility")
    ax.set_xlabel("Salary/skill mismatch penalty")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(-0.02, max(1.0, float(df["final_ndcg_at_10"].max()) + 0.05))
    ax.legend(title="Method", loc="upper right", frameon=True)
    return save_figure(fig, output_dir, "scenario2_mismatch_tradeoff", fmt, dpi)


def plot_exposure_distribution(records_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["job_exposure_gini", "mean_job_utilization"]
    plot_df = _numeric(records_df, metrics).dropna(subset=metrics)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    sns.boxplot(data=plot_df, x="method_label", y="job_exposure_gini", ax=axes[0])
    axes[0].set_title("Job Exposure Gini")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Gini")
    sns.boxplot(data=plot_df, x="method_label", y="mean_job_utilization", ax=axes[1])
    axes[1].set_title("Mean Job Utilization")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Exposure / capacity")
    return save_figure(fig, output_dir, "scenario2_job_exposure_distribution", fmt, dpi)


def plot_method_summary(summary_df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int) -> Path:
    metrics = ["final_ndcg_at_10", "fully_repaired", "reciprocal_penalty", "mismatch_penalty"]
    plot_df = _numeric(summary_df, metrics).melt(
        id_vars="method_label",
        value_vars=metrics,
        var_name="Metric",
        value_name="Value",
    )
    plot_df["Metric"] = plot_df["Metric"].map(
        {
            "final_ndcg_at_10": "NDCG@10",
            "fully_repaired": "CSR",
            "reciprocal_penalty": "Reciprocal penalty",
            "mismatch_penalty": "Mismatch penalty",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    for ax, metric in zip(axes.ravel(), ["NDCG@10", "CSR", "Reciprocal penalty", "Mismatch penalty"]):
        sub = plot_df[plot_df["Metric"] == metric]
        sns.barplot(data=sub, x="method_label", y="Value", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Method")
    fig.suptitle("Scenario-2 Method Constraint-Utility Summary", y=1.02, fontweight="bold")
    return save_figure(fig, output_dir, "scenario2_method_constraint_utility_comparison", fmt, dpi)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    setup_style()
    _, summary_df, records_df = load_metrics(args.metrics_json)
    output_dir = Path(args.output_dir)
    plot_accuracy(summary_df, output_dir, args.format, args.dpi)
    plot_constraint_compliance(summary_df, output_dir, args.format, args.dpi)
    plot_congestion(summary_df, output_dir, args.format, args.dpi)
    plot_mismatch_tradeoff(records_df, output_dir, args.format, args.dpi)
    plot_exposure_distribution(records_df, output_dir, args.format, args.dpi)
    plot_method_summary(summary_df, output_dir, args.format, args.dpi)


if __name__ == "__main__":
    main()
