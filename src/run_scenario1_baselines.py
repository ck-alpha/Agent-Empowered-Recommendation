"""
Scenario-1 offline runner: recall-only diagnostics or recall -> constrained e-commerce processing.

Recommended recall-only smoke command:
/home/username/conda/envs/dualagent/bin/python src/run_scenario1_baselines.py \
  --recall_only --recall_strategy all --test_users 50 --recall_k 100 --iterations 5 --factors 16 \
  --output_json results/scenario1_metrics_recall_suite_smoke.json

Recommended recall-only full command:
/home/username/conda/envs/dualagent/bin/python src/run_scenario1_baselines.py \
  --recall_only --recall_strategy all --test_users 0 --recall_k 200 \
  --min_user_interactions 6 --min_item_interactions 6 \
  --output_json results/scenario1_metrics_recall_suite_full.json

This script evaluates a rigorous recommendation funnel:
1) temporal train/test split with each user's last interaction as ground truth;
2) recall strategies: BPR, Item-KNN, popularity;
3) inverse mapping from internal contiguous item ids back to original item_id;
4) feature join with scenario-1 synthesized item table;
5) EcommercePostProcessingAgent constrained reranking;
6) final constraint and accuracy report, persisted as JSON for plotting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    torch = None
    F = None

try:
    from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    BM25Recommender = None
    CosineRecommender = None
    TFIDFRecommender = None

# Allow running as: python src/run_scenario1_baselines.py
sys.path.insert(0, os.path.dirname(__file__))

from agents import (
    EcommerceInProcessingAgent,
    EcommerceInProcessingConfig,
    EcommercePostProcessingAgent,
    EcommercePostProcessingConfig,
)
from constraints import EcommerceConstraintConfig, EcommerceConstraintHandler

LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
EPS = 1e-9
SINGLE_STRATEGIES = ["bpr", "item_knn", "pop"]
BPR_STRATEGIES = {"bpr"}
ITEM_KNN_STRATEGIES = {"item_knn"}
REQUIRED_ITEM_COLUMNS = [
    "item_id",
    "inventory_initial",
]


@dataclass
class IdMappings:
    """Continuous integer id mappings required by matrix-based recall models."""

    user_id_to_idx: Dict[str, int]
    idx_to_user_id: Dict[int, str]
    item_id_to_idx: Dict[str, int]
    idx_to_item_id: Dict[int, str]


@dataclass
class UserHistoryProfile:
    """Lightweight user content profile from training interactions."""

    brand_ids: set
    seller_ids: set


@dataclass
class RecallCache:
    """Precomputed item indexes for fast large-catalog recall."""

    items_by_id: pd.DataFrame
    all_item_ids_np: np.ndarray
    popular_item_ids_np: np.ndarray
    popular_scores_np: np.ndarray
    price_sorted_item_ids: np.ndarray
    price_sorted_prices: np.ndarray
    price_sorted_popularity: np.ndarray
    brand_to_items: Dict[str, Tuple[np.ndarray, np.ndarray]]
    seller_to_items: Dict[str, Tuple[np.ndarray, np.ndarray]]
    global_median_price: float


@dataclass
class EvalRecord:
    """Per-user evaluation record for final aggregation and plotting."""

    strategy: str
    method: str
    user_id: str
    ground_truth_item_id: str
    recall_count: int
    recall_hit_at_k: float
    raw_top10_hit_at_10: float
    raw_top10_ndcg_at_10: float
    raw_top10_capacity_satisfaction_rate: float
    raw_top10_capacity_violation_total: float
    raw_top10_over_capacity_item_count: int
    raw_top10_fully_repaired: bool
    agent_capacity_satisfaction_rate: float
    agent_capacity_violation_total: float
    agent_over_capacity_item_count: int
    agent_max_capacity_overflow: float
    agent_mean_item_utilization: float
    fully_repaired: bool
    num_swaps: int
    search_steps: int
    local_search_moves: int
    candidate_shortage: bool
    final_list_size: int
    final_utility: float
    raw_final_overlap_rate: float
    changed_item_count: int
    final_hit_at_10: float
    final_ndcg_at_10: float
    raw_item_ids: List[str]
    final_item_ids: List[str]


@dataclass
class RecallEvalRecord:
    """Per-user recall-only evaluation record."""

    strategy: str
    user_id: str
    ground_truth_item_id: str
    history_length: int
    history_bucket: str
    ground_truth_in_train_items: bool
    recall_count: int
    recall_hit_at_k: float
    recall_ndcg_at_k: float
    recall_mrr_at_k: float
    ground_truth_rank: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scenario-1 recall diagnostics or constrained processing baselines."
    )
    parser.add_argument("--processed_dir", default="data/processed", help="Directory containing scenario-1 parquet files.")
    parser.add_argument("--output_prefix", default="beauty_scenario1", help="Scenario-1 parquet filename prefix.")
    parser.add_argument(
        "--output_json",
        default="results/scenario1_metrics_recall_suite.json",
        help="Path to save detailed metrics JSON.",
    )
    parser.add_argument(
        "--test_users",
        type=int,
        default=1000,
        help="Maximum number of eligible users to evaluate. Use 0 for all eligible users.",
    )
    parser.add_argument("--recall_k", type=int, default=200, help="Recall size before reranking.")
    parser.add_argument("--top_k", type=int, default=10, help="Final recommendation list length.")
    parser.add_argument("--factors", type=int, default=128, help="BPR latent factor dimension.")
    parser.add_argument("--iterations", type=int, default=100, help="BPR training epochs.")
    parser.add_argument("--bpr_learning_rate", type=float, default=0.001, help="BPR optimizer learning rate.")
    parser.add_argument("--bpr_regularization", type=float, default=0.0001, help="BPR explicit L2 regularization strength.")
    parser.add_argument("--bpr_batch_size", type=int, default=8192, help="BPR pairwise training batch size.")
    parser.add_argument(
        "--bpr_optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="BPR optimizer for the PyTorch implementation.",
    )
    parser.add_argument(
        "--bpr_negative_sampler",
        choices=["uniform", "popularity", "mixed"],
        default="uniform",
        help="BPR negative sampler: uniform, popularity^0.75, or a 50/50 mix.",
    )
    parser.add_argument("--item_knn_neighbors", type=int, default=100, help="Item-KNN neighbor count.")
    parser.add_argument(
        "--item_knn_weighting",
        choices=["bm25", "cosine", "tfidf"],
        default="bm25",
        help="Item-KNN sparse weighting/similarity model.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for sampling, BPR, and random recall.")
    parser.add_argument("--min_user_interactions", type=int, default=3, help="Minimum user interactions before temporal split.")
    parser.add_argument(
        "--min_item_interactions",
        type=int,
        default=1,
        help="Minimum item interactions before temporal split. Use 6 to filter items with <=5 interactions.",
    )
    parser.add_argument(
        "--recall_backend",
        choices=["legacy", "fast"],
        default="fast",
        help="Recall implementation backend. Use legacy to reproduce prior Beauty results; use fast for large catalogs.",
    )
    parser.add_argument(
        "--recall_strategy",
        choices=SINGLE_STRATEGIES + ["all"],
        default="all",
        help="Recall strategy. Use 'all' to run BPR, Item-KNN, and popularity.",
    )
    parser.add_argument("--hybrid_bpr_weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--baseline_mode",
        choices=["postprocessing", "inprocessing", "both"],
        default="both",
        help="Constrained baseline layer to evaluate.",
    )
    parser.add_argument("--recall_only", action="store_true", help="Only evaluate recall metrics; skip processing agents.")
    parser.add_argument("--show_progress", action="store_true", help="Show BPR / Item-KNN training progress.")
    return parser.parse_args()


def resolve_strategies(strategy: str) -> List[str]:
    return SINGLE_STRATEGIES.copy() if strategy == "all" else [strategy]


def resolve_methods(baseline_mode: str) -> List[str]:
    return ["postprocessing", "inprocessing"] if baseline_mode == "both" else [baseline_mode]


def ensure_torch_available(required: bool = True) -> None:
    """Fail fast when the PyTorch BPR backend is required but unavailable."""
    if required and torch is None:
        raise RuntimeError(
            "Missing dependency module: torch. Install the LLM_Rec PyTorch environment, for example:\n"
            "  conda run -n LLM_Rec python -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )


def ensure_implicit_available(required: bool = True, *, require_item_knn: bool = False) -> None:
    """Fail fast with an actionable message when implicit is required but not installed."""
    if not required:
        return
    missing = []
    if require_item_knn and BM25Recommender is None:
        missing.append("implicit.nearest_neighbours")
    if missing:
        raise RuntimeError(
            f"Missing optional dependency modules: {missing}. Install project dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or install it directly with:\n"
            "  pip install 'implicit>=0.7.0'"
        )


def load_scenario1_tables(processed_dir: str, output_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load scenario-1 item/user/interaction parquet tables."""
    base = Path(processed_dir)
    paths = {
        "items": base / f"{output_prefix}_items.parquet",
        "users": base / f"{output_prefix}_users.parquet",
        "interactions": base / f"{output_prefix}_interactions.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scenario-1 parquet files: {missing}")

    items = pd.read_parquet(paths["items"])
    users = pd.read_parquet(paths["users"])
    interactions = pd.read_parquet(paths["interactions"])

    missing_item_cols = [col for col in REQUIRED_ITEM_COLUMNS if col not in items.columns]
    if missing_item_cols:
        raise ValueError(f"Items table missing required columns: {missing_item_cols}")
    missing_user_cols = [col for col in ["user_id", "target_budget", "budget_tolerance"] if col not in users.columns]
    if missing_user_cols:
        raise ValueError(f"Users table missing required columns: {missing_user_cols}")
    missing_inter_cols = [col for col in ["user_id", "item_id", "timestamp"] if col not in interactions.columns]
    if missing_inter_cols:
        raise ValueError(f"Interactions table missing required columns: {missing_inter_cols}")

    items = items.copy()
    items["item_id"] = items["item_id"].astype(str)
    items["brand_id"] = items["brand_id"].astype(str)
    items["seller_id"] = items["seller_id"].astype(str)
    items["price_filled"] = pd.to_numeric(items["price_filled"], errors="coerce")

    users = users.copy()
    users["user_id"] = users["user_id"].astype(str)
    users["target_budget"] = pd.to_numeric(users["target_budget"], errors="coerce")
    users["budget_tolerance"] = pd.to_numeric(users["budget_tolerance"], errors="coerce")

    interactions = interactions.dropna(subset=["user_id", "item_id", "timestamp"]).copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
    interactions = interactions.dropna(subset=["timestamp"]).copy()
    return items, users, interactions


def _density(rows: int, users: int, items: int) -> float:
    denominator = users * items
    return float(rows / denominator) if denominator else 0.0


def _interaction_count_stats(counts: pd.Series, prefix: str) -> Dict[str, float]:
    if counts.empty:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    counts = counts.astype(float)
    return {
        f"{prefix}_min": float(counts.min()),
        f"{prefix}_mean": float(counts.mean()),
        f"{prefix}_median": float(counts.median()),
        f"{prefix}_p90": float(counts.quantile(0.90)),
        f"{prefix}_p95": float(counts.quantile(0.95)),
        f"{prefix}_max": float(counts.max()),
    }


def make_data_stat(
    step: str,
    interactions: pd.DataFrame,
    *,
    iteration: Optional[int] = None,
    min_user_interactions: Optional[int] = None,
    min_item_interactions: Optional[int] = None,
    removed_interactions: int = 0,
    removed_users: int = 0,
    removed_items: int = 0,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    sampled_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Build one row for the scenario-1 data filtering/split audit trail."""
    rows = int(len(interactions))
    users = int(interactions["user_id"].nunique()) if rows else 0
    items = int(interactions["item_id"].nunique()) if rows else 0
    user_counts = interactions.groupby("user_id").size() if rows else pd.Series(dtype=int)
    item_counts = interactions.groupby("item_id").size() if rows else pd.Series(dtype=int)

    out: Dict[str, Any] = {
        "step": step,
        "iteration": iteration,
        "rows": rows,
        "users": users,
        "items": items,
        "density": _density(rows, users, items),
        "removed_interactions": int(removed_interactions),
        "removed_users": int(removed_users),
        "removed_items": int(removed_items),
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "users_below_min_interactions": None,
        "items_below_min_interactions": None,
        "train_interactions": None,
        "train_users": None,
        "train_items": None,
        "test_interactions": None,
        "test_users": None,
        "test_items": None,
        "test_only_item_count": None,
        "cold_start_unrecallable_rate": None,
        "train_item_coverage_rate": None,
        "evaluated_users": None,
        "test_user_sampling_rate": None,
    }
    out.update(_interaction_count_stats(user_counts, "user_interactions"))
    out.update(_interaction_count_stats(item_counts, "item_interactions"))

    if min_user_interactions is not None:
        out["users_below_min_interactions"] = int((user_counts < min_user_interactions).sum())
    if min_item_interactions is not None:
        out["items_below_min_interactions"] = int((item_counts < min_item_interactions).sum())

    if train_df is not None:
        train_items = set(train_df["item_id"].astype(str))
        out["train_interactions"] = int(len(train_df))
        out["train_users"] = int(train_df["user_id"].nunique()) if len(train_df) else 0
        out["train_items"] = int(train_df["item_id"].nunique()) if len(train_df) else 0
    else:
        train_items = set()

    if test_df is not None:
        test_items = set(test_df["item_id"].astype(str))
        out["test_interactions"] = int(len(test_df))
        out["test_users"] = int(test_df["user_id"].nunique()) if len(test_df) else 0
        out["test_items"] = int(test_df["item_id"].nunique()) if len(test_df) else 0
        if train_df is not None and len(test_df):
            covered = test_df["item_id"].astype(str).isin(train_items)
            cold_start_rate = float((~covered).mean())
            out["test_only_item_count"] = int(len(test_items - train_items))
            out["cold_start_unrecallable_rate"] = cold_start_rate
            out["train_item_coverage_rate"] = float(1.0 - cold_start_rate)

    if sampled_test is not None:
        evaluated_users = int(sampled_test["user_id"].nunique()) if len(sampled_test) else 0
        eligible_users = int(out["test_users"] or 0)
        out["evaluated_users"] = evaluated_users
        out["test_user_sampling_rate"] = float(evaluated_users / eligible_users) if eligible_users else 0.0

    return out


def iterative_k_core_filter(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Iteratively remove users/items below the configured interaction thresholds."""
    if min_user_interactions < 2:
        raise ValueError("min_user_interactions must be >= 2 for temporal split.")
    if min_item_interactions < 1:
        raise ValueError("min_item_interactions must be >= 1.")

    stats: List[Dict[str, Any]] = [
        make_data_stat(
            "loaded",
            interactions,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )
    ]
    current = interactions.copy()
    iteration = 0
    while True:
        iteration += 1
        before_rows = len(current)
        before_users = current["user_id"].nunique()
        before_items = current["item_id"].nunique()

        user_counts = current["user_id"].value_counts()
        item_counts = current["item_id"].value_counts()
        keep_users = set(user_counts[user_counts >= min_user_interactions].index.astype(str))
        keep_items = set(item_counts[item_counts >= min_item_interactions].index.astype(str))
        current = current[
            current["user_id"].isin(keep_users) & current["item_id"].isin(keep_items)
        ].copy()

        removed_interactions = int(before_rows - len(current))
        stats.append(
            make_data_stat(
                f"kcore_iter_{iteration}",
                current,
                iteration=iteration,
                min_user_interactions=min_user_interactions,
                min_item_interactions=min_item_interactions,
                removed_interactions=removed_interactions,
                removed_users=int(before_users - current["user_id"].nunique()),
                removed_items=int(before_items - current["item_id"].nunique()),
            )
        )

        if current.empty:
            raise ValueError(
                "Scenario-1 k-core filtering removed all interactions. "
                f"min_user_interactions={min_user_interactions}, "
                f"min_item_interactions={min_item_interactions}"
            )
        if removed_interactions == 0:
            break

    stats.append(
        make_data_stat(
            "final_filtered",
            current,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )
    )
    return current.reset_index(drop=True), stats


def trim_feature_tables_to_interactions(
    items: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep item/user feature tables aligned with the filtered interaction universe."""
    item_ids = set(interactions["item_id"].astype(str))
    user_ids = set(interactions["user_id"].astype(str))
    filtered_items = items[items["item_id"].isin(item_ids)].copy().reset_index(drop=True)
    filtered_users = users[users["user_id"].isin(user_ids)].copy().reset_index(drop=True)
    LOGGER.info(
        "Trimmed feature tables: items %s -> %s, users %s -> %s",
        len(items),
        len(filtered_items),
        len(users),
        len(filtered_users),
    )
    return filtered_items, filtered_users


def temporal_train_test_split(interactions: pd.DataFrame, min_user_interactions: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split: last interaction is ground truth, previous interactions train recall."""
    if min_user_interactions < 2:
        raise ValueError("min_user_interactions must be >= 2 for temporal split.")

    counts = interactions["user_id"].value_counts()
    eligible_users = counts[counts >= min_user_interactions].index
    eligible = interactions[interactions["user_id"].isin(eligible_users)].copy()
    eligible = eligible.sort_values(["user_id", "timestamp"])

    last_idx = eligible.groupby("user_id", sort=False).tail(1).index
    test = eligible.loc[last_idx, ["user_id", "item_id", "timestamp"]].copy()
    train = eligible.drop(index=last_idx).copy()

    LOGGER.info(
        "Temporal split: train_interactions=%s, test_users=%s, train_users=%s, train_items=%s",
        len(train),
        len(test),
        train["user_id"].nunique(),
        train["item_id"].nunique(),
    )
    return train, test


def build_id_mappings(train_interactions: pd.DataFrame) -> IdMappings:
    """Build continuous user/item id mappings for matrix-based recall models."""
    user_ids = train_interactions["user_id"].dropna().astype(str).drop_duplicates().tolist()
    item_ids = train_interactions["item_id"].dropna().astype(str).drop_duplicates().tolist()

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

    if not user_id_to_idx or not item_id_to_idx:
        raise ValueError("Cannot build BPR mappings from empty train interactions.")
    return IdMappings(user_id_to_idx, idx_to_user_id, item_id_to_idx, idx_to_item_id)


def build_user_item_matrix(train_interactions: pd.DataFrame, mappings: IdMappings) -> csr_matrix:
    """Build CSR user-item feedback matrix."""
    mapped = train_interactions[["user_id", "item_id"]].copy()
    mapped["user_idx"] = mapped["user_id"].map(mappings.user_id_to_idx)
    mapped["item_idx"] = mapped["item_id"].map(mappings.item_id_to_idx)
    mapped = mapped.dropna(subset=["user_idx", "item_idx"])

    rows = mapped["user_idx"].astype(np.int32).to_numpy()
    cols = mapped["item_idx"].astype(np.int32).to_numpy()
    data = np.ones(len(mapped), dtype=np.float32)
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(mappings.user_id_to_idx), len(mappings.item_id_to_idx)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


class TorchBPRRecommender:
    """PyTorch BPR recommender exposed with an implicit-like recommend API."""

    def __init__(self, user_factors: Any, item_factors: Any, device: str) -> None:
        self.device = torch.device(device)
        self.user_factors = user_factors.detach().to(self.device)
        self.item_factors = item_factors.detach().to(self.device)

    def recommend(
        self,
        user_idx: int,
        user_items: csr_matrix,
        N: int,
        filter_already_liked_items: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            user_vector = self.user_factors[int(user_idx)]
            scores = torch.mv(self.item_factors, user_vector)
            if filter_already_liked_items:
                liked = getattr(user_items, "indices", np.array([], dtype=np.int64))
                if len(liked):
                    liked_tensor = torch.as_tensor(liked, dtype=torch.long, device=self.device)
                    scores[liked_tensor] = -torch.inf
            top_n = min(max(1, int(N)), int(scores.numel()))
            values, indices = torch.topk(scores, k=top_n)
        return indices.detach().cpu().numpy(), values.detach().cpu().numpy()


def _build_user_positive_sets(user_item_matrix: csr_matrix) -> List[set]:
    matrix = user_item_matrix.tocsr()
    return [
        set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]].astype(int).tolist())
        for user_idx in range(matrix.shape[0])
    ]


def _make_negative_sampler(
    user_item_matrix: csr_matrix,
    user_positive_sets: Sequence[set],
    sampler: str,
    seed: int,
) -> Any:
    rng = np.random.default_rng(seed)
    num_items = int(user_item_matrix.shape[1])
    sampler = str(sampler).lower()
    item_counts = np.asarray(user_item_matrix.sum(axis=0)).reshape(-1).astype(float)
    popularity = np.power(item_counts + EPS, 0.75)
    popularity = popularity / max(float(popularity.sum()), EPS)

    def draw(size: int) -> np.ndarray:
        if sampler == "popularity":
            return rng.choice(num_items, size=size, replace=True, p=popularity).astype(np.int64)
        if sampler == "mixed":
            mask = rng.random(size) < 0.5
            out = rng.integers(0, num_items, size=size, dtype=np.int64)
            if mask.any():
                out[mask] = rng.choice(num_items, size=int(mask.sum()), replace=True, p=popularity).astype(np.int64)
            return out
        if sampler != "uniform":
            raise ValueError(f"Unsupported BPR negative sampler: {sampler}")
        return rng.integers(0, num_items, size=size, dtype=np.int64)

    def sample(users: np.ndarray) -> np.ndarray:
        negatives = draw(len(users))
        conflicts = np.array(
            [int(item) in user_positive_sets[int(user)] for user, item in zip(users, negatives)],
            dtype=bool,
        )
        attempts = 0
        while conflicts.any() and attempts < 20:
            negatives[conflicts] = draw(int(conflicts.sum()))
            conflicts = np.array(
                [int(item) in user_positive_sets[int(user)] for user, item in zip(users, negatives)],
                dtype=bool,
            )
            attempts += 1
        if conflicts.any():
            for idx in np.where(conflicts)[0]:
                positives = user_positive_sets[int(users[idx])]
                if len(positives) >= num_items:
                    continue
                candidate = int(rng.integers(0, num_items))
                while candidate in positives:
                    candidate = int(rng.integers(0, num_items))
                negatives[idx] = candidate
        return negatives

    return sample


def train_bpr_model(
    user_item_matrix: csr_matrix,
    factors: int,
    iterations: int,
    learning_rate: float,
    regularization: float,
    seed: int,
    show_progress: bool,
    batch_size: int = 8192,
    optimizer_name: str = "adam",
    negative_sampler: str = "uniform",
) -> Tuple[Any, Dict[str, Any]]:
    """Train the project BPR recall model with PyTorch."""
    ensure_torch_available(required=True)
    if F is None:
        raise RuntimeError("torch.nn.functional is unavailable.")
    user_item_matrix = user_item_matrix.tocsr()
    num_users, num_items = user_item_matrix.shape
    if user_item_matrix.nnz <= 0:
        raise ValueError("Cannot train BPR from an empty user-item matrix.")

    factors = max(1, int(factors))
    iterations = max(1, int(iterations))
    batch_size = max(1, int(batch_size))
    optimizer_name = str(optimizer_name).lower()
    negative_sampler = str(negative_sampler).lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    coo = user_item_matrix.tocoo()
    pair_users = coo.row.astype(np.int64)
    pair_items = coo.col.astype(np.int64)
    user_positive_sets = _build_user_positive_sets(user_item_matrix)
    sample_negatives = _make_negative_sampler(user_item_matrix, user_positive_sets, negative_sampler, seed + 17)

    user_embedding = torch.nn.Embedding(num_users, factors).to(device)
    item_embedding = torch.nn.Embedding(num_items, factors).to(device)
    torch.nn.init.normal_(user_embedding.weight, mean=0.0, std=0.01)
    torch.nn.init.normal_(item_embedding.weight, mean=0.0, std=0.01)
    params = list(user_embedding.parameters()) + list(item_embedding.parameters())
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=float(learning_rate))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=float(learning_rate))
    else:
        raise ValueError(f"Unsupported BPR optimizer: {optimizer_name}")

    LOGGER.info(
        "Training BPR (PyTorch): users=%s, items=%s, nnz=%s, factors=%s, epochs=%s, "
        "batch_size=%s, optimizer=%s, learning_rate=%.6f, regularization=%.6f, sampler=%s, device=%s",
        num_users,
        num_items,
        user_item_matrix.nnz,
        factors,
        iterations,
        batch_size,
        optimizer_name,
        learning_rate,
        regularization,
        negative_sampler,
        device,
    )
    diagnostics: Dict[str, Any] = {
        "model": "bpr",
        "implementation": "pytorch",
        "users": int(num_users),
        "items": int(num_items),
        "nnz": int(user_item_matrix.nnz),
        "factors": int(factors),
        "epochs": int(iterations),
        "batch_size": int(batch_size),
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "regularization": float(regularization),
        "negative_sampler": negative_sampler,
        "device": device,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    rng = np.random.default_rng(seed)
    epoch_iter = range(1, iterations + 1)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training BPR", leave=False)
    start = time.time()
    for epoch in epoch_iter:
        order = rng.permutation(len(pair_users))
        total_loss = 0.0
        total_auc = 0.0
        total_examples = 0
        for start_idx in range(0, len(order), batch_size):
            batch_idx = order[start_idx : start_idx + batch_size]
            users_np = pair_users[batch_idx]
            pos_np = pair_items[batch_idx]
            neg_np = sample_negatives(users_np)

            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(pos_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(neg_np, dtype=torch.long, device=device)

            user_vec = user_embedding(users)
            pos_vec = item_embedding(positives)
            neg_vec = item_embedding(negatives)
            pos_scores = (user_vec * pos_vec).sum(dim=1)
            neg_scores = (user_vec * neg_vec).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            if regularization > 0:
                l2 = user_vec.pow(2).sum() + pos_vec.pow(2).sum() + neg_vec.pow(2).sum()
                loss = loss + float(regularization) * l2 / max(1, len(batch_idx))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = int(len(batch_idx))
            total_loss += float(loss.detach().cpu()) * batch_n
            total_auc += float((pos_scores.detach() > neg_scores.detach()).float().sum().cpu())
            total_examples += batch_n

        row = {
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(1, total_examples)),
            "sampled_train_auc": float(total_auc / max(1, total_examples)),
            "elapsed_seconds": float(time.time() - start),
        }
        if epoch == 1:
            diagnostics["first_epoch"] = row
        diagnostics["last_epoch"] = row
        if show_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f"{row['train_loss']:.4f}", auc=f"{row['sampled_train_auc']:.4f}")

    diagnostics["fit_seconds"] = float(time.time() - start)
    model = TorchBPRRecommender(user_embedding.weight, item_embedding.weight, device)
    return model, diagnostics


def train_item_knn_model(
    user_item_matrix: csr_matrix,
    neighbors: int,
    weighting: str,
    show_progress: bool,
) -> Tuple[Any, Dict[str, Any]]:
    """Train an implicit Item-KNN recall model."""
    ensure_implicit_available(required=True, require_item_knn=True)
    neighbors = max(1, int(neighbors))
    weighting = str(weighting).lower()
    if weighting == "bm25":
        model = BM25Recommender(K=neighbors)
    elif weighting == "cosine":
        model = CosineRecommender(K=neighbors)
    elif weighting == "tfidf":
        model = TFIDFRecommender(K=neighbors)
    else:
        raise ValueError(f"Unsupported item_knn_weighting: {weighting}")

    LOGGER.info(
        "Training Item-KNN: users=%s, items=%s, nnz=%s, neighbors=%s, weighting=%s",
        user_item_matrix.shape[0],
        user_item_matrix.shape[1],
        user_item_matrix.nnz,
        neighbors,
        weighting,
    )
    start = time.time()
    model.fit(user_item_matrix, show_progress=show_progress)
    diagnostics = {
        "model": "item_knn",
        "users": int(user_item_matrix.shape[0]),
        "items": int(user_item_matrix.shape[1]),
        "nnz": int(user_item_matrix.nnz),
        "neighbors": int(neighbors),
        "weighting": weighting,
        "fit_seconds": float(time.time() - start),
    }
    return model, diagnostics


def build_popularity_scores(train_interactions: pd.DataFrame) -> pd.Series:
    """Build log-normalized popularity scores from training interactions."""
    counts = train_interactions["item_id"].value_counts().astype(float)
    scores = np.log1p(counts)
    max_score = float(scores.max()) if len(scores) else 1.0
    if max_score <= 0:
        return counts * 0.0
    return scores / max_score


def _minmax(series: pd.Series) -> pd.Series:
    """Min-Max scaling with epsilon denominator to avoid NaN on constant scores."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if len(values) == 0:
        return pd.Series(dtype=float, index=series.index)
    min_v = float(values.min())
    max_v = float(values.max())
    return (values - min_v) / (max_v - min_v + EPS)


def _rank_fallback_score(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Use min-max score, but preserve sorted rank when all values are constant."""
    score = _minmax(df[score_col])
    if len(score) > 1 and float(score.max()) <= EPS:
        return pd.Series(np.linspace(1.0, 0.0, len(score), endpoint=False), index=df.index)
    return score


def _stable_user_seed(user_id: str, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}:{user_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _empty_recall() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "item_id",
            "base_score",
            "bpr_score",
            "item_knn_score",
            "pop_score",
            "budget_score",
            "history_score",
            "random_score",
        ]
    )


def _finalize_recall(df: pd.DataFrame, recall_k: int) -> pd.DataFrame:
    if df.empty:
        return _empty_recall()
    df = df.copy()
    score_columns = ["bpr_score", "item_knn_score", "pop_score", "budget_score", "history_score", "random_score"]
    for col in score_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "base_score" not in df.columns:
        score_cols = [col for col in score_columns if df[col].abs().sum() > 0]
        if score_cols:
            df["base_score"] = sum(_minmax(df[col]) for col in score_cols) / len(score_cols)
        else:
            df["base_score"] = np.linspace(1.0, 0.0, len(df), endpoint=False)
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce").fillna(0.0)
    return df.sort_values(["base_score", "item_id"], ascending=[False, True]).head(recall_k).reset_index(drop=True)


def build_item_catalog(items: pd.DataFrame, popularity_scores: pd.Series) -> pd.DataFrame:
    """Build an item feature view used by content-aware recall strategies."""
    catalog = items[["item_id", "price_filled", "brand_id", "seller_id"]].copy()
    # Recall-layer popularity must come only from the temporal train split. The
    # item table's precomputed popularity is built from full interactions and can leak test-period signal.
    catalog["item_popularity"] = catalog["item_id"].map(popularity_scores).fillna(0.0)
    catalog["price_filled"] = pd.to_numeric(catalog["price_filled"], errors="coerce")
    catalog["item_popularity_norm"] = _minmax(catalog["item_popularity"])
    return catalog.sort_values(["item_popularity_norm", "item_id"], ascending=[False, True]).reset_index(drop=True)


def _build_group_index(catalog: pd.DataFrame, group_col: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build group -> popularity-sorted item arrays for fast history recall."""
    index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for group_id, group in catalog.groupby(group_col, sort=False):
        if pd.isna(group_id):
            continue
        index[str(group_id)] = (
            group["item_id"].astype(str).to_numpy(),
            group["item_popularity_norm"].astype(float).to_numpy(),
        )
    return index


def build_recall_cache(items: pd.DataFrame, popularity_scores: pd.Series) -> RecallCache:
    """Precompute indexes that avoid per-user full-catalog scans in fast backend."""
    catalog = build_item_catalog(items, popularity_scores)
    items_by_id = items[REQUIRED_ITEM_COLUMNS].copy()
    items_by_id["item_id"] = items_by_id["item_id"].astype(str)
    items_by_id = items_by_id.drop_duplicates("item_id", keep="first").set_index("item_id", drop=False)

    price_catalog = catalog.dropna(subset=["price_filled"]).copy()
    price_catalog = price_catalog.sort_values(
        ["price_filled", "item_popularity_norm", "item_id"],
        ascending=[True, False, True],
    )
    global_median_price = float(price_catalog["price_filled"].median()) if not price_catalog.empty else 0.0

    return RecallCache(
        items_by_id=items_by_id,
        all_item_ids_np=catalog["item_id"].astype(str).to_numpy(),
        popular_item_ids_np=catalog["item_id"].astype(str).to_numpy(),
        popular_scores_np=catalog["item_popularity_norm"].astype(float).to_numpy(),
        price_sorted_item_ids=price_catalog["item_id"].astype(str).to_numpy(),
        price_sorted_prices=price_catalog["price_filled"].astype(float).to_numpy(),
        price_sorted_popularity=price_catalog["item_popularity_norm"].astype(float).to_numpy(),
        brand_to_items=_build_group_index(catalog, "brand_id"),
        seller_to_items=_build_group_index(catalog, "seller_id"),
        global_median_price=global_median_price,
    )


def build_train_history_by_user(train_df: pd.DataFrame, target_users: Optional[set] = None) -> Dict[str, set]:
    """Build user -> seen item set for recall filtering."""
    source = train_df
    if target_users is not None:
        source = train_df[train_df["user_id"].isin(target_users)]
    return source.groupby("user_id")["item_id"].agg(lambda values: set(map(str, values))).to_dict()


def build_train_history_lengths(train_df: pd.DataFrame, target_users: Optional[set] = None) -> Dict[str, int]:
    """Build user -> temporal-train interaction count for recall diagnostics."""
    source = train_df
    if target_users is not None:
        source = train_df[train_df["user_id"].isin(target_users)]
    return source.groupby("user_id").size().astype(int).to_dict()


def build_user_history_profiles(
    train_df: pd.DataFrame,
    items: pd.DataFrame,
    target_users: Optional[set] = None,
) -> Dict[str, UserHistoryProfile]:
    """Build user brand/seller profiles from training interactions only."""
    item_features = items[["item_id", "brand_id", "seller_id"]].copy()
    source = train_df
    if target_users is not None:
        source = train_df[train_df["user_id"].isin(target_users)]
    merged = source[["user_id", "item_id"]].merge(item_features, on="item_id", how="left")
    profiles: Dict[str, UserHistoryProfile] = {}
    for user_id, group in merged.groupby("user_id"):
        brand_ids = set(group["brand_id"].dropna().astype(str))
        seller_ids = set(group["seller_id"].dropna().astype(str))
        profiles[str(user_id)] = UserHistoryProfile(brand_ids=brand_ids, seller_ids=seller_ids)
    return profiles


def _normalize_recommend_result(item_indices: Any, scores: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize recommend output into flat arrays."""
    item_indices = np.asarray(item_indices).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(item_indices) != len(scores):
        raise ValueError("Recommend returned item ids and scores with different lengths.")
    return item_indices, scores


def recall_bpr_candidates(
    model: Any,
    user_id: str,
    mappings: IdMappings,
    user_item_matrix: csr_matrix,
    recall_k: int,
) -> pd.DataFrame:
    """Recall BPR candidates and inverse-map internal item ids to original item_id."""
    user_idx = mappings.user_id_to_idx.get(user_id)
    if user_idx is None or model is None:
        return pd.DataFrame(columns=["item_id", "bpr_score"])

    internal_item_indices, scores = model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=recall_k,
        filter_already_liked_items=True,
    )
    internal_item_indices, scores = _normalize_recommend_result(internal_item_indices, scores)

    original_item_ids: List[str] = []
    valid_scores: List[float] = []
    for internal_idx, score in zip(internal_item_indices, scores):
        # Critical inverse transform: models return internal contiguous ids, not original item_id.
        item_id = mappings.idx_to_item_id.get(int(internal_idx))
        if item_id is None:
            LOGGER.warning("Skip unmapped internal item id from BPR: %s", internal_idx)
            continue
        original_item_ids.append(item_id)
        valid_scores.append(float(score))

    return pd.DataFrame({"item_id": original_item_ids, "bpr_score": valid_scores})


def recall_item_knn_candidates(
    model: Any,
    user_id: str,
    mappings: IdMappings,
    user_item_matrix: csr_matrix,
    recall_k: int,
) -> pd.DataFrame:
    """Recall Item-KNN candidates and inverse-map internal item ids to original item_id."""
    user_idx = mappings.user_id_to_idx.get(user_id)
    if user_idx is None or model is None:
        return pd.DataFrame(columns=["item_id", "item_knn_score"])

    internal_item_indices, scores = model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=recall_k,
        filter_already_liked_items=True,
    )
    internal_item_indices, scores = _normalize_recommend_result(internal_item_indices, scores)

    original_item_ids: List[str] = []
    valid_scores: List[float] = []
    for internal_idx, score in zip(internal_item_indices, scores):
        item_id = mappings.idx_to_item_id.get(int(internal_idx))
        if item_id is None:
            LOGGER.warning("Skip unmapped internal item id from Item-KNN: %s", internal_idx)
            continue
        original_item_ids.append(item_id)
        valid_scores.append(float(score))

    return pd.DataFrame({"item_id": original_item_ids, "item_knn_score": valid_scores})


def recall_pop_candidates(
    user_id: str,
    train_by_user: Mapping[str, set],
    popularity_scores: pd.Series,
    recall_k: int,
) -> pd.DataFrame:
    """Recall globally popular items, excluding this user's training history."""
    seen = train_by_user.get(user_id, set())
    rows: List[Tuple[str, float]] = []
    for item_id, score in popularity_scores.items():
        item_id = str(item_id)
        if item_id in seen:
            continue
        rows.append((item_id, float(score)))
        if len(rows) >= recall_k:
            break
    return pd.DataFrame(rows, columns=["item_id", "pop_score"])


def recall_random_candidates(
    user_id: str,
    train_by_user: Mapping[str, set],
    item_catalog: pd.DataFrame,
    recall_k: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministic random lower-bound recall, excluding training history."""
    seen = train_by_user.get(user_id, set())
    pool = item_catalog.loc[~item_catalog["item_id"].isin(seen), "item_id"].astype(str).to_numpy()
    if len(pool) == 0:
        return pd.DataFrame(columns=["item_id", "random_score"])
    rng = np.random.default_rng(_stable_user_seed(user_id, seed, "random_recall"))
    size = min(recall_k, len(pool))
    chosen = rng.choice(pool, size=size, replace=False)
    scores = rng.random(size=size)
    return pd.DataFrame({"item_id": chosen.astype(str), "random_score": scores})


def recall_budget_candidates(
    user_id: str,
    train_by_user: Mapping[str, set],
    item_catalog: pd.DataFrame,
    user_budget_info: pd.Series,
    recall_k: int,
) -> pd.DataFrame:
    """Recall items close to the user's target budget, with popularity as tie-breaker."""
    seen = train_by_user.get(user_id, set())
    target = float(pd.to_numeric(pd.Series([user_budget_info.get("target_budget", np.nan)]), errors="coerce").iloc[0])
    tolerance = float(pd.to_numeric(pd.Series([user_budget_info.get("budget_tolerance", np.nan)]), errors="coerce").iloc[0])
    if not np.isfinite(target):
        target = float(item_catalog["price_filled"].median())
    if not np.isfinite(tolerance) or tolerance <= 0:
        tolerance = max(abs(target) * 0.2, 1.0)

    df = item_catalog.loc[~item_catalog["item_id"].isin(seen)].copy()
    df = df.dropna(subset=["price_filled"])
    if df.empty:
        return pd.DataFrame(columns=["item_id", "budget_score"])
    distance = (df["price_filled"].astype(float) - target).abs()
    price_affinity = 1.0 / (1.0 + distance / (tolerance + EPS))
    df["budget_score"] = 0.75 * price_affinity + 0.25 * df["item_popularity_norm"].astype(float)
    return df.sort_values(["budget_score", "item_popularity_norm", "item_id"], ascending=[False, False, True])[
        ["item_id", "budget_score"]
    ].head(recall_k).reset_index(drop=True)


def recall_history_candidates(
    user_id: str,
    train_by_user: Mapping[str, set],
    item_catalog: pd.DataFrame,
    user_profiles: Mapping[str, UserHistoryProfile],
    popularity_scores: pd.Series,
    recall_k: int,
) -> pd.DataFrame:
    """Recall popular items sharing the user's historical brand/seller preferences."""
    profile = user_profiles.get(user_id, UserHistoryProfile(brand_ids=set(), seller_ids=set()))
    seen = train_by_user.get(user_id, set())
    df = item_catalog.loc[~item_catalog["item_id"].isin(seen)].copy()
    if df.empty:
        return pd.DataFrame(columns=["item_id", "history_score"])

    brand_match = df["brand_id"].isin(profile.brand_ids).astype(float)
    seller_match = df["seller_id"].isin(profile.seller_ids).astype(float)
    df["history_score"] = 0.55 * brand_match + 0.35 * seller_match + 0.10 * df["item_popularity_norm"].astype(float)
    matched = df[df["history_score"] > 0.10].copy()

    if len(matched) < recall_k:
        fallback = recall_pop_candidates(user_id, train_by_user, popularity_scores, recall_k).rename(
            columns={"pop_score": "history_score"}
        )
        matched = pd.concat([matched[["item_id", "history_score"]], fallback], ignore_index=True)
        matched = matched.drop_duplicates("item_id", keep="first")
    else:
        matched = matched[["item_id", "history_score"]]

    return matched.sort_values(["history_score", "item_id"], ascending=[False, True]).head(recall_k).reset_index(drop=True)


def _select_unseen_from_arrays(
    item_ids: np.ndarray,
    scores: np.ndarray,
    seen: set,
    recall_k: int,
) -> Tuple[List[str], List[float]]:
    selected_items: List[str] = []
    selected_scores: List[float] = []
    for item_id, score in zip(item_ids, scores):
        item_id = str(item_id)
        if item_id in seen:
            continue
        selected_items.append(item_id)
        selected_scores.append(float(score))
        if len(selected_items) >= recall_k:
            break
    return selected_items, selected_scores


def recall_pop_candidates_fast(
    user_id: str,
    train_by_user: Mapping[str, set],
    recall_cache: RecallCache,
    recall_k: int,
) -> pd.DataFrame:
    """Fast popularity recall from pre-sorted arrays."""
    seen = train_by_user.get(user_id, set())
    item_ids, scores = _select_unseen_from_arrays(
        recall_cache.popular_item_ids_np,
        recall_cache.popular_scores_np,
        seen,
        recall_k,
    )
    return pd.DataFrame({"item_id": item_ids, "pop_score": scores})


def recall_random_candidates_fast(
    user_id: str,
    train_by_user: Mapping[str, set],
    recall_cache: RecallCache,
    recall_k: int,
    seed: int,
) -> pd.DataFrame:
    """Fast deterministic random recall without per-user full-catalog filtering."""
    all_items = recall_cache.all_item_ids_np
    if len(all_items) == 0:
        return pd.DataFrame(columns=["item_id", "random_score"])

    seen = train_by_user.get(user_id, set())
    rng = np.random.default_rng(_stable_user_seed(user_id, seed, "random_recall_fast"))
    selected: List[str] = []
    selected_set = set(seen)
    batch_size = min(len(all_items), max(recall_k * 4, 1024))

    for _ in range(10):
        indices = rng.integers(0, len(all_items), size=batch_size)
        for idx in indices:
            item_id = str(all_items[int(idx)])
            if item_id in selected_set:
                continue
            selected_set.add(item_id)
            selected.append(item_id)
            if len(selected) >= recall_k:
                break
        if len(selected) >= recall_k:
            break

    if len(selected) < recall_k:
        filler_items, _ = _select_unseen_from_arrays(
            recall_cache.popular_item_ids_np,
            recall_cache.popular_scores_np,
            selected_set,
            recall_k - len(selected),
        )
        selected.extend(filler_items)

    scores = rng.random(size=len(selected))
    return pd.DataFrame({"item_id": selected, "random_score": scores})


def recall_budget_candidates_fast(
    user_id: str,
    train_by_user: Mapping[str, set],
    recall_cache: RecallCache,
    user_budget_info: pd.Series,
    recall_k: int,
) -> pd.DataFrame:
    """Fast budget recall from a price-neighborhood window instead of full catalog scans."""
    seen = train_by_user.get(user_id, set())
    target = float(pd.to_numeric(pd.Series([user_budget_info.get("target_budget", np.nan)]), errors="coerce").iloc[0])
    tolerance = float(pd.to_numeric(pd.Series([user_budget_info.get("budget_tolerance", np.nan)]), errors="coerce").iloc[0])
    if not np.isfinite(target):
        target = recall_cache.global_median_price
    if not np.isfinite(tolerance) or tolerance <= 0:
        tolerance = max(abs(target) * 0.2, 1.0)

    prices = recall_cache.price_sorted_prices
    if len(prices) == 0:
        return pd.DataFrame(columns=["item_id", "budget_score"])

    pool_size = min(len(prices), max(recall_k * 25, 5000))
    center = int(np.searchsorted(prices, target, side="left"))
    start = max(0, center - pool_size // 2)
    end = min(len(prices), start + pool_size)
    start = max(0, end - pool_size)

    item_ids = recall_cache.price_sorted_item_ids[start:end]
    window_prices = prices[start:end]
    popularity = recall_cache.price_sorted_popularity[start:end]
    rows: List[Tuple[str, float]] = []
    for item_id, price, pop_score in zip(item_ids, window_prices, popularity):
        item_id = str(item_id)
        if item_id in seen:
            continue
        price_affinity = 1.0 / (1.0 + abs(float(price) - target) / (tolerance + EPS))
        budget_score = 0.75 * price_affinity + 0.25 * float(pop_score)
        rows.append((item_id, budget_score))

    rows.sort(key=lambda item: (-item[1], item[0]))
    if len(rows) < recall_k:
        selected = {item_id for item_id, _ in rows}
        fallback_seen = set(seen) | selected
        filler_items, filler_scores = _select_unseen_from_arrays(
            recall_cache.popular_item_ids_np,
            recall_cache.popular_scores_np,
            fallback_seen,
            recall_k - len(rows),
        )
        rows.extend((item_id, 0.25 * score) for item_id, score in zip(filler_items, filler_scores))

    return pd.DataFrame(rows[:recall_k], columns=["item_id", "budget_score"])


def recall_history_candidates_fast(
    user_id: str,
    train_by_user: Mapping[str, set],
    recall_cache: RecallCache,
    user_profiles: Mapping[str, UserHistoryProfile],
    recall_k: int,
) -> pd.DataFrame:
    """Fast history recall from brand/seller inverted indexes."""
    profile = user_profiles.get(user_id, UserHistoryProfile(brand_ids=set(), seller_ids=set()))
    seen = train_by_user.get(user_id, set())
    per_group_limit = max(recall_k * 5, 1000)
    components: Dict[str, List[float]] = {}

    def add_group(group_index: Mapping[str, Tuple[np.ndarray, np.ndarray]], group_ids: set, component_idx: int) -> None:
        for group_id in group_ids:
            ids_scores = group_index.get(str(group_id))
            if ids_scores is None:
                continue
            item_ids, pop_scores = ids_scores
            for item_id, pop_score in zip(item_ids[:per_group_limit], pop_scores[:per_group_limit]):
                item_id = str(item_id)
                if item_id in seen:
                    continue
                entry = components.setdefault(item_id, [0.0, 0.0, 0.0])
                entry[component_idx] = 1.0
                entry[2] = max(entry[2], float(pop_score))

    add_group(recall_cache.brand_to_items, profile.brand_ids, 0)
    add_group(recall_cache.seller_to_items, profile.seller_ids, 1)

    rows = [
        (item_id, 0.55 * values[0] + 0.35 * values[1] + 0.10 * values[2])
        for item_id, values in components.items()
    ]
    rows = [row for row in rows if row[1] > 0.10]
    rows.sort(key=lambda item: (-item[1], item[0]))

    if len(rows) < recall_k:
        selected = {item_id for item_id, _ in rows}
        fallback_seen = set(seen) | selected
        filler_items, filler_scores = _select_unseen_from_arrays(
            recall_cache.popular_item_ids_np,
            recall_cache.popular_scores_np,
            fallback_seen,
            recall_k - len(rows),
        )
        rows.extend((item_id, score) for item_id, score in zip(filler_items, filler_scores))

    return pd.DataFrame(rows[:recall_k], columns=["item_id", "history_score"])


def _merge_recall_sources(sources: Sequence[Tuple[pd.DataFrame, str, float]], recall_k: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    weighted_parts: List[pd.Series] = []
    total_weight = 0.0

    for source_df, score_col, weight in sources:
        if source_df.empty or score_col not in source_df.columns:
            continue
        source = source_df[["item_id", score_col]].copy()
        source["item_id"] = source["item_id"].astype(str)
        source[score_col] = pd.to_numeric(source[score_col], errors="coerce").fillna(0.0)
        source = source.drop_duplicates("item_id", keep="first")
        merged = source if merged is None else merged.merge(source, on="item_id", how="outer")
        total_weight += max(float(weight), 0.0)

    if merged is None or merged.empty:
        return _empty_recall()

    merged = merged.fillna(0.0)
    for _, score_col, weight in sources:
        if score_col in merged.columns and weight > 0:
            weighted_parts.append(_minmax(merged[score_col]) * float(weight))
    if weighted_parts and total_weight > 0:
        merged["base_score"] = sum(weighted_parts) / total_weight
    else:
        merged["base_score"] = np.linspace(1.0, 0.0, len(merged), endpoint=False)
    return _finalize_recall(merged, recall_k)


def recall_candidates(
    strategy: str,
    model: Any,
    user_id: str,
    mappings: IdMappings,
    user_item_matrix: csr_matrix,
    train_by_user: Mapping[str, set],
    popularity_scores: pd.Series,
    item_catalog: pd.DataFrame,
    user_profiles: Mapping[str, UserHistoryProfile],
    user_budget_info: pd.Series,
    recall_k: int,
    hybrid_bpr_weight: float,
    seed: int,
    recall_backend: str = "legacy",
    recall_cache: Optional[RecallCache] = None,
) -> pd.DataFrame:
    """Recall candidates using one of the supported strategies."""
    use_fast = recall_backend == "fast"
    if use_fast and recall_cache is None:
        raise ValueError("Fast recall backend requires a RecallCache.")

    if strategy == "bpr":
        bpr_df = recall_bpr_candidates(model, user_id, mappings, user_item_matrix, recall_k)
        if bpr_df.empty:
            return _empty_recall()
        bpr_df["base_score"] = _rank_fallback_score(bpr_df, "bpr_score")
        return _finalize_recall(bpr_df, recall_k)

    if strategy == "item_knn":
        item_knn_df = recall_item_knn_candidates(model, user_id, mappings, user_item_matrix, recall_k)
        if item_knn_df.empty:
            return _empty_recall()
        item_knn_df["base_score"] = _rank_fallback_score(item_knn_df, "item_knn_score")
        return _finalize_recall(item_knn_df, recall_k)

    if strategy == "pop":
        pop_df = (
            recall_pop_candidates_fast(user_id, train_by_user, recall_cache, recall_k)
            if use_fast
            else recall_pop_candidates(user_id, train_by_user, popularity_scores, recall_k)
        )
        if pop_df.empty:
            return _empty_recall()
        pop_df["base_score"] = _rank_fallback_score(pop_df, "pop_score")
        return _finalize_recall(pop_df, recall_k)

    raise ValueError(f"Unsupported recall strategy: {strategy}")


def build_candidate_dataframe(
    recall_df: pd.DataFrame,
    items: pd.DataFrame,
    recall_cache: Optional[RecallCache] = None,
) -> pd.DataFrame:
    """Join recall scores with scenario-1 item features."""
    if recall_df.empty:
        return pd.DataFrame(columns=REQUIRED_ITEM_COLUMNS + ["base_score"])

    recall = recall_df.copy()
    recall["item_id"] = recall["item_id"].astype(str)
    if recall_cache is not None:
        features = recall_cache.items_by_id.reindex(recall["item_id"].to_numpy())
        missing_mask = features["item_id"].isna().to_numpy()
        feature_values = features.reset_index(drop=True).drop(columns=["item_id"])
        candidates = pd.concat([recall.reset_index(drop=True), feature_values], axis=1)
        candidates = candidates.loc[~missing_mask].copy()
    else:
        item_features = items[REQUIRED_ITEM_COLUMNS].copy()
        candidates = recall.merge(item_features, on="item_id", how="inner")

    missing_after_merge = len(recall) - len(candidates)
    if missing_after_merge > 0:
        LOGGER.debug("Dropped %s recalled items missing synthesized features.", missing_after_merge)

    required = REQUIRED_ITEM_COLUMNS + ["base_score"]
    missing_cols = [col for col in required if col not in candidates.columns]
    if missing_cols:
        raise ValueError(f"Merged candidate DataFrame missing columns: {missing_cols}")
    return candidates


def compute_hr_ndcg_at_k(recommended_items: List[str], ground_truth_item_id: str, k: int) -> Tuple[float, float]:
    """Single-ground-truth HR@K and NDCG@K."""
    top_items = recommended_items[:k]
    try:
        rank_idx = top_items.index(ground_truth_item_id)
    except ValueError:
        return 0.0, 0.0
    return 1.0, float(1.0 / np.log2(rank_idx + 2))


def compute_recall_metrics_at_k(
    recommended_items: List[str],
    ground_truth_item_id: str,
    k: int,
) -> Tuple[float, float, float, Optional[int]]:
    """Single-ground-truth HR/NDCG/MRR@K plus one-indexed hit rank."""
    top_items = recommended_items[:k]
    try:
        rank_idx = top_items.index(ground_truth_item_id)
    except ValueError:
        return 0.0, 0.0, 0.0, None
    rank = rank_idx + 1
    return 1.0, float(1.0 / np.log2(rank_idx + 2)), float(1.0 / rank), rank


def history_bucket(history_length: int) -> str:
    if history_length <= 2:
        return "hist=2"
    if history_length == 3:
        return "hist=3"
    if history_length <= 5:
        return "hist=4-5"
    return "hist>=6"


def sample_test_users(test_df: pd.DataFrame, mappings: IdMappings, test_users: int, seed: int) -> pd.DataFrame:
    """Sample eligible test users that exist in train mappings; test_users=0 means all."""
    if test_users < 0:
        raise ValueError("test_users must be >= 0. Use 0 for all eligible users.")
    eligible = test_df[test_df["user_id"].isin(mappings.user_id_to_idx)].copy()
    if eligible.empty:
        raise ValueError("No test users are present in train mappings.")

    if test_users == 0 or test_users >= len(eligible):
        return eligible.reset_index(drop=True)

    return eligible.sample(n=test_users, random_state=seed).reset_index(drop=True)


def build_raw_topk_recommendations(
    candidate_df: pd.DataFrame,
    top_k: int,
    *,
    assume_sorted: bool = False,
) -> pd.DataFrame:
    """Build unconstrained per-user Top-K recommendations before capacity handling."""
    if candidate_df.empty or top_k <= 0:
        out = candidate_df.head(0).copy()
        out["rank"] = pd.Series(dtype=int)
        return out
    required = ["user_id", "item_id", "base_score"]
    missing = [col for col in required if col not in candidate_df.columns]
    if missing:
        raise ValueError(f"candidate_df missing columns for raw Top-K: {missing}")
    if assume_sorted:
        ranked = candidate_df.copy()
    else:
        ranked = candidate_df.sort_values(["user_id", "base_score", "item_id"], ascending=[True, False, True]).copy()
    raw = ranked.groupby("user_id", sort=False).head(top_k).copy()
    raw["rank"] = raw.groupby("user_id", sort=False).cumcount() + 1
    return raw.reset_index(drop=True)


def build_item_ids_by_user(recommendations: pd.DataFrame) -> Dict[str, List[str]]:
    """Build a ranked user -> item-id list index for repeated metric lookups."""
    if recommendations.empty or "user_id" not in recommendations.columns or "item_id" not in recommendations.columns:
        return {}

    ranked = recommendations.copy()
    ranked["user_id"] = ranked["user_id"].astype(str)
    ranked["item_id"] = ranked["item_id"].astype(str)
    if "rank" in ranked.columns:
        ranked["_rank_for_group"] = pd.to_numeric(ranked["rank"], errors="coerce").fillna(float("inf"))
        ranked = ranked.sort_values(["user_id", "_rank_for_group", "item_id"], ascending=[True, True, True])
        ranked = ranked.drop(columns=["_rank_for_group"])
    elif "base_score" in ranked.columns:
        ranked["_score_for_group"] = pd.to_numeric(ranked["base_score"], errors="coerce").fillna(0.0)
        ranked = ranked.sort_values(["user_id", "_score_for_group", "item_id"], ascending=[True, False, True])
        ranked = ranked.drop(columns=["_score_for_group"])
    else:
        ranked = ranked.sort_values(["user_id", "item_id"], ascending=[True, True])
    return ranked.groupby("user_id", sort=False)["item_id"].agg(list).to_dict()


def item_ids_for_user(recommendations: pd.DataFrame, user_id: str) -> List[str]:
    """Return a user's recommended item ids in rank order."""
    if recommendations.empty or "user_id" not in recommendations.columns:
        return []
    rows = recommendations.loc[recommendations["user_id"].astype(str) == str(user_id)].copy()
    if rows.empty:
        return []
    if "rank" in rows.columns:
        rows = rows.sort_values(["rank", "item_id"], ascending=[True, True])
    else:
        rows = rows.sort_values(["base_score", "item_id"], ascending=[False, True])
    return rows["item_id"].astype(str).tolist()


def evaluate_capacity_constraints(recommendations: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate scenario-1 capacity-only hard constraints for a recommendation matrix."""
    handler = EcommerceConstraintHandler(EcommerceConstraintConfig(capacity_col="inventory_initial"))
    return handler.evaluate_all(recommendations)


def run_reranking_evaluation(
    strategy: str,
    methods: Sequence[str],
    model: Any,
    sampled_test: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame,
    mappings: IdMappings,
    user_item_matrix: csr_matrix,
    train_by_user: Mapping[str, set],
    popularity_scores: pd.Series,
    item_catalog: pd.DataFrame,
    user_profiles: Mapping[str, UserHistoryProfile],
    recall_k: int,
    top_k: int,
    hybrid_bpr_weight: float,
    seed: int,
    recall_backend: str,
    recall_cache: Optional[RecallCache],
) -> List[EvalRecord]:
    """Run recall and batch-level capacity-only processing baselines for sampled users."""
    del users
    agents = {
        "postprocessing": EcommercePostProcessingAgent(EcommercePostProcessingConfig(top_k=top_k)),
        "inprocessing": EcommerceInProcessingAgent(EcommerceInProcessingConfig(top_k=top_k, random_seed=seed)),
    }
    selected_methods = [method for method in methods if method in agents]
    if not selected_methods:
        raise ValueError(f"No supported baseline methods selected: {methods}")

    records: List[EvalRecord] = []
    skipped_no_recall = 0
    user_rows: List[Dict[str, Any]] = []
    candidate_frames: List[pd.DataFrame] = []

    desc = f"Collecting candidates [{strategy}]"
    for row in tqdm(sampled_test.itertuples(index=False), total=len(sampled_test), desc=desc):
        user_id = str(row.user_id)
        ground_truth = str(row.item_id)

        recall_df = recall_candidates(
            strategy=strategy,
            model=model,
            user_id=user_id,
            mappings=mappings,
            user_item_matrix=user_item_matrix,
            train_by_user=train_by_user,
            popularity_scores=popularity_scores,
            item_catalog=item_catalog,
            user_profiles=user_profiles,
            user_budget_info=pd.Series(dtype=float),
            recall_k=recall_k,
            hybrid_bpr_weight=hybrid_bpr_weight,
            seed=seed,
            recall_backend=recall_backend,
            recall_cache=recall_cache,
        )
        if recall_df.empty:
            skipped_no_recall += 1
            continue

        candidate_df = build_candidate_dataframe(
            recall_df,
            items,
            recall_cache=recall_cache if recall_backend == "fast" else None,
        )
        if candidate_df.empty:
            skipped_no_recall += 1
            continue
        candidate_df = candidate_df.copy()
        candidate_df["user_id"] = user_id

        recall_hr, _ = compute_hr_ndcg_at_k(recall_df["item_id"].astype(str).tolist(), ground_truth, recall_k)
        user_rows.append(
            {
                "user_id": user_id,
                "ground_truth": ground_truth,
                "recall_count": int(len(recall_df)),
                "recall_hit_at_k": float(recall_hr),
            }
        )
        candidate_frames.append(candidate_df)

    if skipped_no_recall:
        LOGGER.warning("[%s] Skipped %s users because recall/feature join returned no candidates.", strategy, skipped_no_recall)
    if not user_rows or not candidate_frames:
        return records

    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    user_order = [row["user_id"] for row in user_rows]
    raw_recs = build_raw_topk_recommendations(all_candidates, top_k=top_k, assume_sorted=True)
    raw_items_by_user = build_item_ids_by_user(raw_recs)
    raw_constraints = evaluate_capacity_constraints(raw_recs)
    raw_capacity_rate = float(raw_constraints.get("capacity_satisfaction_rate", 0.0))
    raw_capacity_violation = float(raw_constraints.get("capacity_violation_total", 0.0))
    raw_over_capacity_count = int(raw_constraints.get("over_capacity_item_count", 0))
    raw_fully_repaired = bool(raw_constraints.get("all_hard_constraints_satisfied", False))

    raw_accuracy: Dict[str, Tuple[float, float]] = {}
    for info in user_rows:
        raw_items = raw_items_by_user.get(info["user_id"], [])
        raw_accuracy[info["user_id"]] = compute_hr_ndcg_at_k(raw_items, info["ground_truth"], top_k)

    for method in selected_methods:
        LOGGER.info("[%s] Running scenario-1 capacity baseline: %s", strategy, method)
        result = agents[method].recommend_batch(
            candidate_items=all_candidates,
            user_ids=user_order,
            top_k=top_k,
        )
        diagnostics = result.get("diagnostics", {})
        recommendations = result.get("recommendations", pd.DataFrame()).copy()
        final_constraints = diagnostics.get("final_constraints", {})
        agent_capacity_rate = float(final_constraints.get("capacity_satisfaction_rate", 0.0))
        agent_capacity_violation = float(final_constraints.get("capacity_violation_total", 0.0))
        agent_over_capacity_count = int(final_constraints.get("over_capacity_item_count", 0))
        agent_max_overflow = float(final_constraints.get("max_capacity_overflow", 0.0))
        agent_mean_utilization = float(final_constraints.get("mean_item_utilization", 0.0))
        final_items_by_user = build_item_ids_by_user(recommendations)

        final_counts = (
            recommendations["user_id"].astype(str).value_counts().astype(int).to_dict()
            if not recommendations.empty and "user_id" in recommendations.columns
            else {}
        )

        for info in user_rows:
            user_id = info["user_id"]
            ground_truth = info["ground_truth"]
            final_items = final_items_by_user.get(user_id, [])
            final_hr, final_ndcg = compute_hr_ndcg_at_k(final_items, ground_truth, top_k)
            raw_hr, raw_ndcg = raw_accuracy.get(user_id, (0.0, 0.0))
            raw_items = raw_items_by_user.get(user_id, [])
            raw_item_set = set(raw_items)
            final_item_set = set(final_items)
            overlap_count = len(raw_item_set & final_item_set)
            overlap_denominator = max(1, len(raw_item_set))
            records.append(
                EvalRecord(
                    strategy=strategy,
                    method=method,
                    user_id=user_id,
                    ground_truth_item_id=ground_truth,
                    recall_count=int(info["recall_count"]),
                    recall_hit_at_k=float(info["recall_hit_at_k"]),
                    raw_top10_hit_at_10=float(raw_hr),
                    raw_top10_ndcg_at_10=float(raw_ndcg),
                    raw_top10_capacity_satisfaction_rate=raw_capacity_rate,
                    raw_top10_capacity_violation_total=raw_capacity_violation,
                    raw_top10_over_capacity_item_count=raw_over_capacity_count,
                    raw_top10_fully_repaired=raw_fully_repaired,
                    agent_capacity_satisfaction_rate=agent_capacity_rate,
                    agent_capacity_violation_total=agent_capacity_violation,
                    agent_over_capacity_item_count=agent_over_capacity_count,
                    agent_max_capacity_overflow=agent_max_overflow,
                    agent_mean_item_utilization=agent_mean_utilization,
                    fully_repaired=bool(diagnostics.get("fully_repaired", False)),
                    num_swaps=int(diagnostics.get("num_swaps", 0)),
                    search_steps=int(diagnostics.get("search_steps", 0)),
                    local_search_moves=int(diagnostics.get("local_search_moves", 0)),
                    candidate_shortage=int(final_counts.get(user_id, 0)) < top_k,
                    final_list_size=len(final_items),
                    final_utility=float(diagnostics.get("final_utility", 0.0)),
                    raw_final_overlap_rate=float(overlap_count / overlap_denominator),
                    changed_item_count=int(max(len(raw_item_set), len(final_item_set)) - overlap_count),
                    final_hit_at_10=float(final_hr),
                    final_ndcg_at_10=float(final_ndcg),
                    raw_item_ids=raw_items,
                    final_item_ids=final_items,
                )
            )
    return records


def run_recall_only_evaluation(
    strategy: str,
    model: Any,
    sampled_test: pd.DataFrame,
    mappings: IdMappings,
    user_item_matrix: csr_matrix,
    train_by_user: Mapping[str, set],
    train_history_lengths: Mapping[str, int],
    popularity_scores: pd.Series,
    recall_k: int,
) -> List[RecallEvalRecord]:
    """Run recall-only diagnostics without joining item features or invoking processing agents."""
    records: List[RecallEvalRecord] = []
    skipped_no_recall = 0

    desc = f"Evaluating recall [{strategy}]"
    for row in tqdm(sampled_test.itertuples(index=False), total=len(sampled_test), desc=desc):
        user_id = str(row.user_id)
        ground_truth = str(row.item_id)
        recall_df = recall_candidates(
            strategy=strategy,
            model=model,
            user_id=user_id,
            mappings=mappings,
            user_item_matrix=user_item_matrix,
            train_by_user=train_by_user,
            popularity_scores=popularity_scores,
            item_catalog=pd.DataFrame(),
            user_profiles={},
            user_budget_info=pd.Series(dtype=float),
            recall_k=recall_k,
            hybrid_bpr_weight=0.0,
            seed=RANDOM_SEED,
            recall_backend="legacy",
            recall_cache=None,
        )
        if recall_df.empty:
            skipped_no_recall += 1
            continue

        recalled_items = recall_df["item_id"].astype(str).tolist()
        hr, ndcg, mrr, rank = compute_recall_metrics_at_k(recalled_items, ground_truth, recall_k)
        hist_len = int(train_history_lengths.get(user_id, len(train_by_user.get(user_id, set()))))
        records.append(
            RecallEvalRecord(
                strategy=strategy,
                user_id=user_id,
                ground_truth_item_id=ground_truth,
                history_length=hist_len,
                history_bucket=history_bucket(hist_len),
                ground_truth_in_train_items=ground_truth in mappings.item_id_to_idx,
                recall_count=int(len(recall_df)),
                recall_hit_at_k=float(hr),
                recall_ndcg_at_k=float(ndcg),
                recall_mrr_at_k=float(mrr),
                ground_truth_rank=rank,
            )
        )

    if skipped_no_recall:
        LOGGER.warning("[%s] Skipped %s users because recall returned no candidates.", strategy, skipped_no_recall)
    return records


def _mean_dict(records: List[EvalRecord]) -> Dict[str, float]:
    if not records:
        return {}
    keys = [
        "recall_count",
        "recall_hit_at_k",
        "raw_top10_hit_at_10",
        "raw_top10_ndcg_at_10",
        "raw_top10_capacity_satisfaction_rate",
        "raw_top10_capacity_violation_total",
        "raw_top10_over_capacity_item_count",
        "raw_top10_fully_repaired",
        "agent_capacity_satisfaction_rate",
        "agent_capacity_violation_total",
        "agent_over_capacity_item_count",
        "agent_max_capacity_overflow",
        "agent_mean_item_utilization",
        "fully_repaired",
        "num_swaps",
        "search_steps",
        "local_search_moves",
        "candidate_shortage",
        "final_list_size",
        "final_utility",
        "raw_final_overlap_rate",
        "changed_item_count",
        "final_hit_at_10",
        "final_ndcg_at_10",
    ]
    return {key: float(np.mean([getattr(record, key) for record in records])) for key in keys}


def _mean_recall_dict(records: List[RecallEvalRecord]) -> Dict[str, float]:
    if not records:
        return {}
    keys = ["recall_count", "recall_hit_at_k", "recall_ndcg_at_k", "recall_mrr_at_k"]
    return {key: float(np.mean([getattr(record, key) for record in records])) for key in keys}


def make_recall_only_summary(records: List[RecallEvalRecord], strategy: str, recall_k: int) -> Dict[str, Any]:
    summary = _mean_recall_dict(records)
    bucket_metrics: Dict[str, Dict[str, float]] = {}
    for bucket in ["hist=2", "hist=3", "hist=4-5", "hist>=6"]:
        bucket_records = [record for record in records if record.history_bucket == bucket]
        bucket_metrics[bucket] = {
            "num_users": int(len(bucket_records)),
            "recall_hit_at_k": float(np.mean([record.recall_hit_at_k for record in bucket_records]))
            if bucket_records
            else 0.0,
        }

    cold_start_rate = (
        float(np.mean([not record.ground_truth_in_train_items for record in records])) if records else 0.0
    )
    summary.update(
        {
            "num_users": len(records),
            "recall_strategy": strategy,
            "method": "recall_only",
            "recall_only": True,
            "recall_k": recall_k,
            "history_bucket_metrics": bucket_metrics,
            "cold_start_unrecallable_rate": cold_start_rate,
            "train_item_coverage_rate": 1.0 - cold_start_rate,
        }
    )
    return summary


def make_recall_suite_summary(
    summaries_by_strategy: Mapping[str, Dict[str, Any]],
    recall_k: int,
) -> Dict[str, Any]:
    """Build a compact macro summary for multi-strategy recall-only suites."""
    rows = list(summaries_by_strategy.values())
    metrics = ["recall_count", "recall_hit_at_k", "recall_ndcg_at_k", "recall_mrr_at_k"]
    summary = {
        metric: float(np.mean([float(row.get(metric, 0.0)) for row in rows])) if rows else 0.0
        for metric in metrics
    }
    cold_start_rate = (
        float(np.mean([float(row.get("cold_start_unrecallable_rate", 0.0)) for row in rows])) if rows else 0.0
    )
    summary.update(
        {
            "num_users": int(max((int(row.get("num_users", 0)) for row in rows), default=0)),
            "num_strategy_user_records": int(sum(int(row.get("num_users", 0)) for row in rows)),
            "recall_strategy": "all",
            "method": "recall_only",
            "recall_only": True,
            "recall_k": recall_k,
            "cold_start_unrecallable_rate": cold_start_rate,
            "train_item_coverage_rate": 1.0 - cold_start_rate,
        }
    )
    return summary


def make_summary(
    records: List[EvalRecord],
    strategy: str,
    method: Optional[str],
    recall_k: int,
    top_k: int,
    hybrid_bpr_weight: float,
    recall_backend: str,
) -> Dict[str, Any]:
    summary = _mean_dict(records)
    summary.update(
        {
            "num_users": len(records),
            "recall_strategy": strategy,
            "method": method or "mixed",
            "recall_k": recall_k,
            "top_k": top_k,
            "hybrid_bpr_weight": hybrid_bpr_weight,
            "recall_backend": recall_backend,
        }
    )
    return summary


def log_recall_only_report(summary: Dict[str, Any], recall_k: int) -> None:
    """Log aggregate recall-only diagnostics."""
    if not summary:
        LOGGER.warning("No recall-only evaluation records produced; final report is empty.")
        return

    LOGGER.info("\n%s", "=" * 78)
    LOGGER.info("Scenario-1 Recall-Only Report")
    LOGGER.info("%s", "=" * 78)
    LOGGER.info("Evaluated users                    : %d", int(summary.get("num_users", 0)))
    LOGGER.info("Recall strategy                    : %s", summary.get("recall_strategy"))
    LOGGER.info("Recall K                           : %d", recall_k)
    LOGGER.info("Mean recall count                  : %.4f", summary.get("recall_count", 0.0))
    LOGGER.info("Hit Rate@%d (Recall)               : %.4f", recall_k, summary.get("recall_hit_at_k", 0.0))
    LOGGER.info("NDCG@%d (Recall)                   : %.4f", recall_k, summary.get("recall_ndcg_at_k", 0.0))
    LOGGER.info("MRR@%d (Recall)                    : %.4f", recall_k, summary.get("recall_mrr_at_k", 0.0))
    LOGGER.info("Cold-start unrecallable rate       : %.4f", summary.get("cold_start_unrecallable_rate", 0.0))
    for bucket, bucket_summary in summary.get("history_bucket_metrics", {}).items():
        LOGGER.info(
            "%s users / HR@%d                 : %d / %.4f",
            bucket,
            recall_k,
            int(bucket_summary.get("num_users", 0)),
            float(bucket_summary.get("recall_hit_at_k", 0.0)),
        )
    LOGGER.info("%s\n", "=" * 78)


def log_final_report(summary: Dict[str, Any], recall_k: int, top_k: int) -> None:
    """Log final aggregate experiment report."""
    if not summary:
        LOGGER.warning("No evaluation records produced; final report is empty.")
        return

    LOGGER.info("\n%s", "=" * 78)
    LOGGER.info("Scenario-1 Recall -> Constrained Post-processing Report")
    LOGGER.info("%s", "=" * 78)
    LOGGER.info("Evaluated users                    : %d", int(summary.get("num_users", 0)))
    LOGGER.info("Recall strategy                    : %s", summary.get("recall_strategy"))
    LOGGER.info("Baseline method                    : %s", summary.get("method", "unknown"))
    LOGGER.info("Recall backend                     : %s", summary.get("recall_backend"))
    LOGGER.info("Recall K / Final K                 : %d / %d", recall_k, top_k)
    LOGGER.info("Mean recall count                  : %.4f", summary.get("recall_count", 0.0))
    LOGGER.info("Hit Rate@%d (Recall)               : %.4f", recall_k, summary.get("recall_hit_at_k", 0.0))
    LOGGER.info("Raw Top-%d Hit Rate                : %.4f", top_k, summary.get("raw_top10_hit_at_10", 0.0))
    LOGGER.info("Raw Top-%d NDCG                    : %.4f", top_k, summary.get("raw_top10_ndcg_at_10", 0.0))
    LOGGER.info("Final Hit Rate@%d                  : %.4f", top_k, summary.get("final_hit_at_10", 0.0))
    LOGGER.info("Final NDCG@%d                      : %.4f", top_k, summary.get("final_ndcg_at_10", 0.0))
    LOGGER.info("Recall-layer CSR@Top-%d            : %.4f", top_k, summary.get("raw_top10_fully_repaired", 0.0))
    LOGGER.info("Agent CSR@Top-%d                   : %.4f", top_k, summary.get("fully_repaired", 0.0))
    LOGGER.info("Recall-layer capacity satisfaction : %.4f", summary.get("raw_top10_capacity_satisfaction_rate", 0.0))
    LOGGER.info("Agent capacity satisfaction        : %.4f", summary.get("agent_capacity_satisfaction_rate", 0.0))
    LOGGER.info("Recall-layer capacity overflow     : %.4f", summary.get("raw_top10_capacity_violation_total", 0.0))
    LOGGER.info("Agent capacity overflow            : %.4f", summary.get("agent_capacity_violation_total", 0.0))
    LOGGER.info("Agent over-capacity item count     : %.4f", summary.get("agent_over_capacity_item_count", 0.0))
    LOGGER.info("Agent mean item utilization        : %.4f", summary.get("agent_mean_item_utilization", 0.0))
    LOGGER.info("Average swaps                      : %.4f", summary.get("num_swaps", 0.0))
    LOGGER.info("Average search steps               : %.4f", summary.get("search_steps", 0.0))
    LOGGER.info("Average local-search moves         : %.4f", summary.get("local_search_moves", 0.0))
    LOGGER.info("Candidate shortage rate            : %.4f", summary.get("candidate_shortage", 0.0))
    LOGGER.info("Raw/final item overlap             : %.4f", summary.get("raw_final_overlap_rate", 0.0))
    LOGGER.info("Average changed item count         : %.4f", summary.get("changed_item_count", 0.0))
    LOGGER.info("Final DCG-weighted utility         : %.4f", summary.get("final_utility", 0.0))
    LOGGER.info("%s\n", "=" * 78)


def data_stats_csv_path(output_json: str) -> Path:
    output_path = Path(output_json)
    return output_path.with_name(f"{output_path.stem}_data_stats.csv")


def save_data_stats_csv(output_json: str, data_stats: Sequence[Mapping[str, Any]]) -> None:
    if not data_stats:
        return
    path = data_stats_csv_path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(data_stats)).to_csv(path, index=False)
    LOGGER.info("Saved scenario-1 data stats CSV: %s", path)


def save_metrics_json(
    output_json: str,
    args: argparse.Namespace,
    records: Sequence[Any],
    summary: Dict[str, Any],
    summaries_by_strategy: Dict[str, Dict[str, Any]],
    summaries_by_method_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
    completed_strategies: Optional[Sequence[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    data_stats: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
    """Persist per-user records and aggregate summary for plotting."""
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "summary": summary,
        "summaries_by_strategy": summaries_by_strategy,
        "summaries_by_method_strategy": summaries_by_method_strategy or {},
        "completed_strategies": list(completed_strategies or summaries_by_strategy.keys()),
        "diagnostics": diagnostics or {},
        "data_stats": list(data_stats or []),
        "records": [asdict(record) for record in records],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved scenario-1 metrics JSON: %s", output_path)
    if data_stats:
        save_data_stats_csv(output_json, data_stats)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    strategies = resolve_strategies(args.recall_strategy)
    methods = [] if args.recall_only else resolve_methods(args.baseline_mode)

    if any(strategy in BPR_STRATEGIES for strategy in strategies):
        try:
            ensure_torch_available(required=True)
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            sys.exit(1)
    if any(strategy in ITEM_KNN_STRATEGIES for strategy in strategies):
        try:
            ensure_implicit_available(required=True, require_item_knn=True)
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            sys.exit(1)

    items, users, interactions = load_scenario1_tables(args.processed_dir, args.output_prefix)
    interactions, data_stats = iterative_k_core_filter(
        interactions,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    items, users = trim_feature_tables_to_interactions(items, users, interactions)
    train_df, test_df = temporal_train_test_split(interactions, args.min_user_interactions)
    mappings = build_id_mappings(train_df)
    user_item_matrix = build_user_item_matrix(train_df, mappings)
    sampled_test = sample_test_users(test_df, mappings, args.test_users, args.seed)
    data_stats.append(
        make_data_stat(
            "train_test_split",
            interactions,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            train_df=train_df,
            test_df=test_df,
            sampled_test=sampled_test,
        )
    )
    save_data_stats_csv(args.output_json, data_stats)
    sampled_user_ids = set(sampled_test["user_id"].astype(str))
    popularity_scores = build_popularity_scores(train_df)
    train_by_user = build_train_history_by_user(train_df, sampled_user_ids)
    train_history_lengths = build_train_history_lengths(train_df, sampled_user_ids)
    if args.recall_only:
        LOGGER.info("Recall-only mode: skipping item feature cache and processing agents.")
        recall_cache = None
        item_catalog = pd.DataFrame()
        user_profiles = {}
    elif args.recall_backend == "fast":
        LOGGER.info("Building fast recall cache.")
        recall_cache = build_recall_cache(items, popularity_scores)
        item_catalog = pd.DataFrame()
        user_profiles = build_user_history_profiles(train_df, items, sampled_user_ids)
    else:
        LOGGER.info("Using legacy recall backend.")
        recall_cache = None
        item_catalog = build_item_catalog(items, popularity_scores)
        user_profiles = build_user_history_profiles(train_df, items, sampled_user_ids)

    recall_models: Dict[str, Any] = {}
    model_diagnostics: Dict[str, Any] = {}
    if any(strategy in BPR_STRATEGIES for strategy in strategies):
        bpr_model, bpr_diagnostics = train_bpr_model(
            user_item_matrix=user_item_matrix,
            factors=args.factors,
            iterations=args.iterations,
            learning_rate=args.bpr_learning_rate,
            regularization=args.bpr_regularization,
            seed=args.seed,
            show_progress=args.show_progress,
            batch_size=args.bpr_batch_size,
            optimizer_name=args.bpr_optimizer,
            negative_sampler=args.bpr_negative_sampler,
        )
        recall_models["bpr"] = bpr_model
        model_diagnostics["bpr"] = bpr_diagnostics
    if any(strategy in ITEM_KNN_STRATEGIES for strategy in strategies):
        item_knn_model, item_knn_diagnostics = train_item_knn_model(
            user_item_matrix=user_item_matrix,
            neighbors=args.item_knn_neighbors,
            weighting=args.item_knn_weighting,
            show_progress=args.show_progress,
        )
        recall_models["item_knn"] = item_knn_model
        model_diagnostics["item_knn"] = item_knn_diagnostics

    if args.recall_only:
        all_recall_records: List[RecallEvalRecord] = []
        summaries_by_strategy: Dict[str, Dict[str, Any]] = {}
        completed_strategies: List[str] = []
        for strategy in strategies:
            LOGGER.info("Running scenario-1 recall-only strategy: %s", strategy)
            strategy_start = time.time()
            strategy_records = run_recall_only_evaluation(
                strategy=strategy,
                model=recall_models.get(strategy),
                sampled_test=sampled_test,
                mappings=mappings,
                user_item_matrix=user_item_matrix,
                train_by_user=train_by_user,
                train_history_lengths=train_history_lengths,
                popularity_scores=popularity_scores,
                recall_k=args.recall_k,
            )
            strategy_summary = make_recall_only_summary(
                strategy_records,
                strategy=strategy,
                recall_k=args.recall_k,
            )
            summaries_by_strategy[strategy] = strategy_summary
            completed_strategies.append(strategy)
            all_recall_records.extend(strategy_records)
            log_recall_only_report(strategy_summary, args.recall_k)
            LOGGER.info("Finished recall-only strategy %s in %.2f seconds.", strategy, time.time() - strategy_start)

            checkpoint_summary = dict(strategy_summary)
            if len(strategies) > 1:
                checkpoint_summary.update(
                    {
                        "recall_strategy": args.recall_strategy,
                        "primary_strategy": strategy,
                        "num_strategies": len(strategies),
                        "completed_strategy_count": len(completed_strategies),
                    }
                )
            save_metrics_json(
                args.output_json,
                args,
                all_recall_records,
                checkpoint_summary,
                summaries_by_strategy,
                summaries_by_method_strategy={},
                completed_strategies=completed_strategies,
                diagnostics=model_diagnostics,
                data_stats=data_stats,
            )

        if len(strategies) == 1:
            summary = summaries_by_strategy[strategies[0]]
        else:
            summary = make_recall_suite_summary(summaries_by_strategy, recall_k=args.recall_k)
            summary.update(
                {
                    "recall_strategy": args.recall_strategy,
                    "primary_strategy": "all",
                    "num_strategies": len(strategies),
                    "completed_strategy_count": len(completed_strategies),
                }
            )
        save_metrics_json(
            args.output_json,
            args,
            all_recall_records,
            summary,
            summaries_by_strategy,
            summaries_by_method_strategy={},
            completed_strategies=completed_strategies,
            diagnostics=model_diagnostics,
            data_stats=data_stats,
        )
        return

    all_records: List[EvalRecord] = []
    summaries_by_strategy: Dict[str, Dict[str, Any]] = {}
    summaries_by_method_strategy: Dict[str, Dict[str, Any]] = {}
    completed_strategies: List[str] = []
    for strategy in strategies:
        LOGGER.info("Running scenario-1 recall strategy: %s", strategy)
        strategy_start = time.time()
        strategy_records = run_reranking_evaluation(
            strategy=strategy,
            methods=methods,
            model=recall_models.get(strategy),
            sampled_test=sampled_test,
            items=items,
            users=users,
            mappings=mappings,
            user_item_matrix=user_item_matrix,
            train_by_user=train_by_user,
            popularity_scores=popularity_scores,
            item_catalog=item_catalog,
            user_profiles=user_profiles,
            recall_k=args.recall_k,
            top_k=args.top_k,
            hybrid_bpr_weight=args.hybrid_bpr_weight,
            seed=args.seed,
            recall_backend=args.recall_backend,
            recall_cache=recall_cache,
        )
        strategy_method_summaries: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            method_records = [record for record in strategy_records if record.method == method]
            method_summary = make_summary(
                method_records,
                strategy=strategy,
                method=method,
                recall_k=args.recall_k,
                top_k=args.top_k,
                hybrid_bpr_weight=args.hybrid_bpr_weight,
                recall_backend=args.recall_backend,
            )
            strategy_method_summaries[method] = method_summary
            summaries_by_method_strategy[f"{method}::{strategy}"] = method_summary

        primary_method = "postprocessing" if "postprocessing" in strategy_method_summaries else methods[0]
        strategy_summary = strategy_method_summaries[primary_method]
        summaries_by_strategy[strategy] = strategy_summary
        completed_strategies.append(strategy)
        log_final_report(strategy_summary, args.recall_k, args.top_k)
        all_records.extend(strategy_records)
        LOGGER.info("Finished strategy %s in %.2f seconds.", strategy, time.time() - strategy_start)

        checkpoint_summary = dict(strategy_summary)
        if len(strategies) > 1:
            checkpoint_summary.update(
                {
                    "recall_strategy": args.recall_strategy,
                    "primary_strategy": strategy,
                    "num_strategies": len(strategies),
                    "completed_strategy_count": len(completed_strategies),
                    "recall_backend": args.recall_backend,
                }
            )
        save_metrics_json(
            args.output_json,
            args,
            all_records,
            checkpoint_summary,
            summaries_by_strategy,
            summaries_by_method_strategy=summaries_by_method_strategy,
            completed_strategies=completed_strategies,
            data_stats=data_stats,
        )

    if len(strategies) == 1:
        summary = summaries_by_strategy[strategies[0]]
    else:
        primary_method = "postprocessing" if "postprocessing" in methods else methods[0]
        summary = make_summary(
            [record for record in all_records if record.method == primary_method],
            "all",
            primary_method,
            args.recall_k,
            args.top_k,
            args.hybrid_bpr_weight,
            args.recall_backend,
        )
        summary = dict(summary)
        summary.update(
            {
                "recall_strategy": args.recall_strategy,
                "primary_strategy": "all",
                "num_strategies": len(strategies),
                "completed_strategy_count": len(completed_strategies),
                "recall_backend": args.recall_backend,
            }
        )

    save_metrics_json(
        args.output_json,
        args,
        all_records,
        summary,
        summaries_by_strategy,
        summaries_by_method_strategy=summaries_by_method_strategy,
        completed_strategies=completed_strategies,
        diagnostics=model_diagnostics,
        data_stats=data_stats,
    )


if __name__ == "__main__":
    main()
