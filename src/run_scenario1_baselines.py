"""
Scenario-1 offline runner: multi-strategy recall -> constrained e-commerce post-processing.

Recommended smoke command:
/home/username/conda/envs/dualagent/bin/python src/run_scenario1_baselines.py \
  --recall_strategy all --test_users 50 --recall_k 100 --iterations 5 --factors 32 \
  --output_json results/scenario1_metrics_recall_suite_smoke.json

Recommended full command:
/home/username/conda/envs/dualagent/bin/python src/run_scenario1_baselines.py \
  --recall_strategy all --test_users 100000 --recall_k 200 --top_k 10 \
  --factors 128 --iterations 100 \
  --output_json results/scenario1_metrics_recall_suite_full.json

This script evaluates a rigorous recommendation funnel:
1) temporal train/test split with each user's last interaction as ground truth;
2) multiple recall strategies: random, popularity, BPR, hybrid, budget, history, ensemble;
3) inverse mapping from implicit internal item ids back to original item_id;
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

try:
    from implicit.bpr import BayesianPersonalizedRanking
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    BayesianPersonalizedRanking = None

# Allow running as: python src/run_scenario1_baselines.py
sys.path.insert(0, os.path.dirname(__file__))

from agents import EcommercePostProcessingAgent, EcommercePostProcessingConfig

LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
EPS = 1e-9
SINGLE_STRATEGIES = ["random", "pop", "bpr", "hybrid", "budget", "history", "ensemble"]
BPR_STRATEGIES = {"bpr", "hybrid", "ensemble"}
REQUIRED_ITEM_COLUMNS = [
    "item_id",
    "stockout_risk",
    "is_new",
    "seller_id",
    "brand_id",
    "price_filled",
]


@dataclass
class IdMappings:
    """Continuous integer id mappings required by implicit."""

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
class EvalRecord:
    """Per-user evaluation record for final aggregation and plotting."""

    strategy: str
    user_id: str
    ground_truth_item_id: str
    recall_count: int
    recall_hit_at_k: float
    raw_top10_hit_at_10: float
    raw_top10_ndcg_at_10: float
    raw_top10_inventory_safety_rate: float
    raw_top10_fully_repaired: bool
    agent_inventory_safety_rate: float
    fully_repaired: bool
    num_swaps: int
    budget_penalty: float
    entropy_penalty: float
    final_new_item_count: int
    final_seller_count: int
    candidate_shortage: bool
    final_list_size: int
    final_hit_at_10: float
    final_ndcg_at_10: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scenario-1 multi-strategy recall plus constrained post-processing baseline."
    )
    parser.add_argument("--processed_dir", default="data/processed", help="Directory containing scenario-1 parquet files.")
    parser.add_argument("--output_prefix", default="beauty_scenario1", help="Scenario-1 parquet filename prefix.")
    parser.add_argument(
        "--output_json",
        default="results/scenario1_metrics_recall_suite.json",
        help="Path to save detailed metrics JSON.",
    )
    parser.add_argument("--test_users", type=int, default=1000, help="Maximum number of eligible users to evaluate.")
    parser.add_argument("--recall_k", type=int, default=200, help="Recall size before reranking.")
    parser.add_argument("--top_k", type=int, default=10, help="Final recommendation list length.")
    parser.add_argument("--factors", type=int, default=128, help="BPR latent factor dimension.")
    parser.add_argument("--iterations", type=int, default=100, help="BPR training iterations.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for sampling, BPR, and random recall.")
    parser.add_argument("--min_user_interactions", type=int, default=3, help="Minimum interactions before temporal split.")
    parser.add_argument(
        "--recall_strategy",
        choices=SINGLE_STRATEGIES + ["all"],
        default="hybrid",
        help="Recall strategy. Use 'all' to run the full comparison suite.",
    )
    parser.add_argument("--hybrid_bpr_weight", type=float, default=0.5, help="BPR weight in hybrid score fusion.")
    parser.add_argument("--show_progress", action="store_true", help="Show implicit BPR training progress.")
    return parser.parse_args()


def resolve_strategies(strategy: str) -> List[str]:
    return SINGLE_STRATEGIES.copy() if strategy == "all" else [strategy]


def ensure_implicit_available(required: bool = True) -> None:
    """Fail fast with an actionable message when implicit is required but not installed."""
    if required and BayesianPersonalizedRanking is None:
        raise RuntimeError(
            "Missing optional dependency 'implicit'. Install project dependencies with:\n"
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
    """Build continuous user/item id mappings for implicit BPR."""
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
    """Build CSR user-item implicit feedback matrix."""
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


def train_bpr_model(user_item_matrix: csr_matrix, factors: int, iterations: int, seed: int, show_progress: bool):
    """Train implicit BPR recall model."""
    ensure_implicit_available(required=True)
    model = BayesianPersonalizedRanking(
        factors=factors,
        iterations=iterations,
        random_state=seed,
    )
    LOGGER.info(
        "Training BPR: users=%s, items=%s, nnz=%s, factors=%s, iterations=%s",
        user_item_matrix.shape[0],
        user_item_matrix.shape[1],
        user_item_matrix.nnz,
        factors,
        iterations,
    )
    model.fit(user_item_matrix, show_progress=show_progress)
    return model


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
    return pd.DataFrame(columns=["item_id", "base_score", "bpr_score", "pop_score", "budget_score", "history_score", "random_score"])


def _finalize_recall(df: pd.DataFrame, recall_k: int) -> pd.DataFrame:
    if df.empty:
        return _empty_recall()
    df = df.copy()
    for col in ["bpr_score", "pop_score", "budget_score", "history_score", "random_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "base_score" not in df.columns:
        score_cols = [col for col in ["bpr_score", "pop_score", "budget_score", "history_score", "random_score"] if df[col].abs().sum() > 0]
        if score_cols:
            df["base_score"] = sum(_minmax(df[col]) for col in score_cols) / len(score_cols)
        else:
            df["base_score"] = np.linspace(1.0, 0.0, len(df), endpoint=False)
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce").fillna(0.0)
    return df.sort_values(["base_score", "item_id"], ascending=[False, True]).head(recall_k).reset_index(drop=True)


def build_item_catalog(items: pd.DataFrame, popularity_scores: pd.Series) -> pd.DataFrame:
    """Build an item feature view used by content-aware recall strategies."""
    catalog = items[["item_id", "price_filled", "brand_id", "seller_id"]].copy()
    if "popularity" in items.columns:
        catalog["item_popularity"] = pd.to_numeric(items["popularity"], errors="coerce").fillna(0.0)
    else:
        catalog["item_popularity"] = catalog["item_id"].map(popularity_scores).fillna(0.0)
    catalog["price_filled"] = pd.to_numeric(catalog["price_filled"], errors="coerce")
    catalog["item_popularity_norm"] = _minmax(catalog["item_popularity"])
    return catalog.sort_values(["item_popularity_norm", "item_id"], ascending=[False, True]).reset_index(drop=True)


def build_train_history_by_user(train_df: pd.DataFrame) -> Dict[str, set]:
    """Build user -> seen item set for recall filtering."""
    return train_df.groupby("user_id")["item_id"].agg(lambda values: set(map(str, values))).to_dict()


def build_user_history_profiles(train_df: pd.DataFrame, items: pd.DataFrame) -> Dict[str, UserHistoryProfile]:
    """Build user brand/seller profiles from training interactions only."""
    item_features = items[["item_id", "brand_id", "seller_id"]].copy()
    merged = train_df[["user_id", "item_id"]].merge(item_features, on="item_id", how="left")
    profiles: Dict[str, UserHistoryProfile] = {}
    for user_id, group in merged.groupby("user_id"):
        brand_ids = set(group["brand_id"].dropna().astype(str))
        seller_ids = set(group["seller_id"].dropna().astype(str))
        profiles[str(user_id)] = UserHistoryProfile(brand_ids=brand_ids, seller_ids=seller_ids)
    return profiles


def _normalize_recommend_result(item_indices: Any, scores: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize implicit recommend output across minor version differences."""
    item_indices = np.asarray(item_indices).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(item_indices) != len(scores):
        raise ValueError("BPR recommend returned item ids and scores with different lengths.")
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
        # Critical inverse transform: implicit returns internal contiguous ids, not original item_id.
        item_id = mappings.idx_to_item_id.get(int(internal_idx))
        if item_id is None:
            LOGGER.warning("Skip unmapped internal item id from BPR: %s", internal_idx)
            continue
        original_item_ids.append(item_id)
        valid_scores.append(float(score))

    return pd.DataFrame({"item_id": original_item_ids, "bpr_score": valid_scores})


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
) -> pd.DataFrame:
    """Recall candidates using one of the supported strategies."""
    hybrid_bpr_weight = min(1.0, max(0.0, float(hybrid_bpr_weight)))
    pop_weight = 1.0 - hybrid_bpr_weight

    if strategy == "random":
        random_df = recall_random_candidates(user_id, train_by_user, item_catalog, recall_k, seed)
        if random_df.empty:
            return _empty_recall()
        random_df["base_score"] = _rank_fallback_score(random_df, "random_score")
        return _finalize_recall(random_df, recall_k)

    if strategy == "bpr":
        bpr_df = recall_bpr_candidates(model, user_id, mappings, user_item_matrix, recall_k)
        if bpr_df.empty:
            return _empty_recall()
        bpr_df["base_score"] = _rank_fallback_score(bpr_df, "bpr_score")
        return _finalize_recall(bpr_df, recall_k)

    if strategy == "pop":
        pop_df = recall_pop_candidates(user_id, train_by_user, popularity_scores, recall_k)
        if pop_df.empty:
            return _empty_recall()
        pop_df["base_score"] = _rank_fallback_score(pop_df, "pop_score")
        return _finalize_recall(pop_df, recall_k)

    if strategy == "budget":
        budget_df = recall_budget_candidates(user_id, train_by_user, item_catalog, user_budget_info, recall_k)
        if budget_df.empty:
            return _empty_recall()
        budget_df["base_score"] = _rank_fallback_score(budget_df, "budget_score")
        return _finalize_recall(budget_df, recall_k)

    if strategy == "history":
        history_df = recall_history_candidates(user_id, train_by_user, item_catalog, user_profiles, popularity_scores, recall_k)
        if history_df.empty:
            return _empty_recall()
        history_df["base_score"] = _rank_fallback_score(history_df, "history_score")
        return _finalize_recall(history_df, recall_k)

    if strategy == "hybrid":
        bpr_df = recall_bpr_candidates(model, user_id, mappings, user_item_matrix, recall_k)
        pop_df = recall_pop_candidates(user_id, train_by_user, popularity_scores, recall_k)
        return _merge_recall_sources(
            [(bpr_df, "bpr_score", hybrid_bpr_weight), (pop_df, "pop_score", pop_weight)],
            recall_k,
        )

    if strategy == "ensemble":
        bpr_df = recall_bpr_candidates(model, user_id, mappings, user_item_matrix, recall_k)
        pop_df = recall_pop_candidates(user_id, train_by_user, popularity_scores, recall_k)
        budget_df = recall_budget_candidates(user_id, train_by_user, item_catalog, user_budget_info, recall_k)
        history_df = recall_history_candidates(user_id, train_by_user, item_catalog, user_profiles, popularity_scores, recall_k)
        return _merge_recall_sources(
            [
                (bpr_df, "bpr_score", 0.35),
                (pop_df, "pop_score", 0.25),
                (budget_df, "budget_score", 0.20),
                (history_df, "history_score", 0.20),
            ],
            recall_k,
        )

    raise ValueError(f"Unsupported recall strategy: {strategy}")


def build_candidate_dataframe(recall_df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Join recall scores with scenario-1 item features."""
    if recall_df.empty:
        return pd.DataFrame(columns=REQUIRED_ITEM_COLUMNS + ["base_score"])

    item_features = items[REQUIRED_ITEM_COLUMNS].copy()
    candidates = recall_df.merge(item_features, on="item_id", how="inner")
    missing_after_merge = len(recall_df) - len(candidates)
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


def sample_test_users(test_df: pd.DataFrame, mappings: IdMappings, test_users: int, seed: int) -> pd.DataFrame:
    """Sample eligible test users that exist in the recall train user mapping."""
    eligible = test_df[test_df["user_id"].isin(mappings.user_id_to_idx)].copy()
    if eligible.empty:
        raise ValueError("No test users are present in train mappings.")

    n = min(test_users, len(eligible))
    return eligible.sample(n=n, random_state=seed).reset_index(drop=True)


def evaluate_raw_topk(
    candidate_df: pd.DataFrame,
    user_budget_info: pd.Series,
    ground_truth_item_id: str,
    top_k: int,
) -> Dict[str, float]:
    """Evaluate unconstrained recall-layer Top-K before Agent reranking."""
    if candidate_df.empty:
        return {
            "inventory_safety_rate": 0.0,
            "fully_repaired": 0.0,
            "hit_at_10": 0.0,
            "ndcg_at_10": 0.0,
        }
    raw_topk = candidate_df.sort_values("base_score", ascending=False).head(top_k).copy()
    raw_items = raw_topk["item_id"].astype(str).tolist()
    raw_hr, raw_ndcg = compute_hr_ndcg_at_k(raw_items, ground_truth_item_id, top_k)
    agent = EcommercePostProcessingAgent(EcommercePostProcessingConfig(top_k=top_k))
    diagnostics = agent.constraint_handler.evaluate_all(raw_topk, user_budget_info=user_budget_info)
    return {
        "inventory_safety_rate": float(diagnostics.get("inventory_pass_rate", 0.0)),
        "fully_repaired": float(diagnostics.get("all_hard_constraints_satisfied", False)),
        "hit_at_10": float(raw_hr),
        "ndcg_at_10": float(raw_ndcg),
    }


def run_reranking_evaluation(
    strategy: str,
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
) -> List[EvalRecord]:
    """Run recall and constrained post-processing for sampled users."""
    users_by_id = users.set_index("user_id", drop=False)
    agent = EcommercePostProcessingAgent(EcommercePostProcessingConfig(top_k=top_k))
    records: List[EvalRecord] = []
    skipped_no_budget = 0
    skipped_no_recall = 0

    desc = f"Evaluating users [{strategy}]"
    for row in tqdm(sampled_test.itertuples(index=False), total=len(sampled_test), desc=desc):
        user_id = str(row.user_id)
        ground_truth = str(row.item_id)

        if user_id not in users_by_id.index:
            skipped_no_budget += 1
            continue
        user_budget_info = users_by_id.loc[user_id, ["target_budget", "budget_tolerance"]]

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
            user_budget_info=user_budget_info,
            recall_k=recall_k,
            hybrid_bpr_weight=hybrid_bpr_weight,
            seed=seed,
        )
        if recall_df.empty:
            skipped_no_recall += 1
            continue

        candidate_df = build_candidate_dataframe(recall_df, items)
        if candidate_df.empty:
            skipped_no_recall += 1
            continue

        raw_eval = evaluate_raw_topk(
            candidate_df,
            user_budget_info=user_budget_info,
            ground_truth_item_id=ground_truth,
            top_k=top_k,
        )
        result = agent.recommend(
            user_id=user_id,
            candidate_items=candidate_df,
            user_budget_info=user_budget_info,
            top_k=top_k,
        )
        diagnostics = result.get("diagnostics", {})
        final_items = [str(item_id) for item_id in result.get("item_ids", [])]
        final_hr, final_ndcg = compute_hr_ndcg_at_k(final_items, ground_truth, top_k)
        recall_hr, _ = compute_hr_ndcg_at_k(recall_df["item_id"].astype(str).tolist(), ground_truth, recall_k)

        final_constraints = diagnostics.get("final_constraints", {})
        agent_inventory_safety_rate = float(final_constraints.get("inventory_pass_rate", 0.0))

        records.append(
            EvalRecord(
                strategy=strategy,
                user_id=user_id,
                ground_truth_item_id=ground_truth,
                recall_count=int(len(recall_df)),
                recall_hit_at_k=float(recall_hr),
                raw_top10_hit_at_10=float(raw_eval["hit_at_10"]),
                raw_top10_ndcg_at_10=float(raw_eval["ndcg_at_10"]),
                raw_top10_inventory_safety_rate=float(raw_eval["inventory_safety_rate"]),
                raw_top10_fully_repaired=bool(raw_eval["fully_repaired"]),
                agent_inventory_safety_rate=float(agent_inventory_safety_rate),
                fully_repaired=bool(diagnostics.get("fully_repaired", False)),
                num_swaps=int(diagnostics.get("num_swaps", 0)),
                budget_penalty=float(diagnostics.get("final_budget_penalty", 0.0)),
                entropy_penalty=float(diagnostics.get("final_entropy_penalty", 0.0)),
                final_new_item_count=int(diagnostics.get("final_new_item_count", 0)),
                final_seller_count=int(diagnostics.get("final_seller_count", 0)),
                candidate_shortage=bool(diagnostics.get("candidate_shortage", False)),
                final_list_size=len(final_items),
                final_hit_at_10=float(final_hr),
                final_ndcg_at_10=float(final_ndcg),
            )
        )

    if skipped_no_budget:
        LOGGER.warning("[%s] Skipped %s users because budget features were missing.", strategy, skipped_no_budget)
    if skipped_no_recall:
        LOGGER.warning("[%s] Skipped %s users because recall/feature join returned no candidates.", strategy, skipped_no_recall)
    return records


def _mean_dict(records: List[EvalRecord]) -> Dict[str, float]:
    if not records:
        return {}
    keys = [
        "recall_count",
        "recall_hit_at_k",
        "raw_top10_hit_at_10",
        "raw_top10_ndcg_at_10",
        "raw_top10_inventory_safety_rate",
        "raw_top10_fully_repaired",
        "agent_inventory_safety_rate",
        "fully_repaired",
        "num_swaps",
        "budget_penalty",
        "entropy_penalty",
        "final_new_item_count",
        "final_seller_count",
        "candidate_shortage",
        "final_list_size",
        "final_hit_at_10",
        "final_ndcg_at_10",
    ]
    return {key: float(np.mean([getattr(record, key) for record in records])) for key in keys}


def make_summary(records: List[EvalRecord], strategy: str, recall_k: int, top_k: int, hybrid_bpr_weight: float) -> Dict[str, Any]:
    summary = _mean_dict(records)
    summary.update(
        {
            "num_users": len(records),
            "recall_strategy": strategy,
            "recall_k": recall_k,
            "top_k": top_k,
            "hybrid_bpr_weight": hybrid_bpr_weight,
        }
    )
    return summary


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
    LOGGER.info("Recall K / Final K                 : %d / %d", recall_k, top_k)
    LOGGER.info("Mean recall count                  : %.4f", summary.get("recall_count", 0.0))
    LOGGER.info("Hit Rate@%d (Recall)               : %.4f", recall_k, summary.get("recall_hit_at_k", 0.0))
    LOGGER.info("Raw Top-%d Hit Rate                : %.4f", top_k, summary.get("raw_top10_hit_at_10", 0.0))
    LOGGER.info("Raw Top-%d NDCG                    : %.4f", top_k, summary.get("raw_top10_ndcg_at_10", 0.0))
    LOGGER.info("Final Hit Rate@%d                  : %.4f", top_k, summary.get("final_hit_at_10", 0.0))
    LOGGER.info("Final NDCG@%d                      : %.4f", top_k, summary.get("final_ndcg_at_10", 0.0))
    LOGGER.info("Recall-layer CSR@Top-%d            : %.4f", top_k, summary.get("raw_top10_fully_repaired", 0.0))
    LOGGER.info("Agent CSR@Top-%d                   : %.4f", top_k, summary.get("fully_repaired", 0.0))
    LOGGER.info("Recall-layer inventory safety      : %.4f", summary.get("raw_top10_inventory_safety_rate", 0.0))
    LOGGER.info("Agent inventory safety             : %.4f", summary.get("agent_inventory_safety_rate", 0.0))
    LOGGER.info("Average swaps                      : %.4f", summary.get("num_swaps", 0.0))
    LOGGER.info("Mean budget penalty                : %.4f", summary.get("budget_penalty", 0.0))
    LOGGER.info("Mean entropy penalty               : %.4f", summary.get("entropy_penalty", 0.0))
    LOGGER.info("Mean final new item count          : %.4f", summary.get("final_new_item_count", 0.0))
    LOGGER.info("Mean final seller count            : %.4f", summary.get("final_seller_count", 0.0))
    LOGGER.info("Candidate shortage rate            : %.4f", summary.get("candidate_shortage", 0.0))
    LOGGER.info("%s\n", "=" * 78)


def save_metrics_json(
    output_json: str,
    args: argparse.Namespace,
    records: List[EvalRecord],
    summary: Dict[str, Any],
    summaries_by_strategy: Dict[str, Dict[str, Any]],
) -> None:
    """Persist per-user records and aggregate summary for plotting."""
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "summary": summary,
        "summaries_by_strategy": summaries_by_strategy,
        "records": [asdict(record) for record in records],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved scenario-1 metrics JSON: %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    strategies = resolve_strategies(args.recall_strategy)

    if any(strategy in BPR_STRATEGIES for strategy in strategies):
        try:
            ensure_implicit_available(required=True)
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            sys.exit(1)

    items, users, interactions = load_scenario1_tables(args.processed_dir, args.output_prefix)
    train_df, test_df = temporal_train_test_split(interactions, args.min_user_interactions)
    mappings = build_id_mappings(train_df)
    user_item_matrix = build_user_item_matrix(train_df, mappings)
    sampled_test = sample_test_users(test_df, mappings, args.test_users, args.seed)
    popularity_scores = build_popularity_scores(train_df)
    train_by_user = build_train_history_by_user(train_df)
    item_catalog = build_item_catalog(items, popularity_scores)
    user_profiles = build_user_history_profiles(train_df, items)

    model = None
    if any(strategy in BPR_STRATEGIES for strategy in strategies):
        model = train_bpr_model(
            user_item_matrix=user_item_matrix,
            factors=args.factors,
            iterations=args.iterations,
            seed=args.seed,
            show_progress=args.show_progress,
        )

    all_records: List[EvalRecord] = []
    summaries_by_strategy: Dict[str, Dict[str, Any]] = {}
    for strategy in strategies:
        LOGGER.info("Running scenario-1 recall strategy: %s", strategy)
        strategy_records = run_reranking_evaluation(
            strategy=strategy,
            model=model,
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
        )
        strategy_summary = make_summary(
            strategy_records,
            strategy=strategy,
            recall_k=args.recall_k,
            top_k=args.top_k,
            hybrid_bpr_weight=args.hybrid_bpr_weight,
        )
        summaries_by_strategy[strategy] = strategy_summary
        log_final_report(strategy_summary, args.recall_k, args.top_k)
        all_records.extend(strategy_records)

    if len(strategies) == 1:
        summary = summaries_by_strategy[strategies[0]]
    else:
        summary = summaries_by_strategy.get("ensemble", make_summary(all_records, "all", args.recall_k, args.top_k, args.hybrid_bpr_weight))
        summary = dict(summary)
        summary.update({"recall_strategy": args.recall_strategy, "primary_strategy": "ensemble", "num_strategies": len(strategies)})

    save_metrics_json(args.output_json, args, all_records, summary, summaries_by_strategy)


if __name__ == "__main__":
    main()
