"""
Scenario-2 offline runner: MIND news ranker -> entropy-constrained baselines.

Recommended smoke command:
  python src/run_scenario2_baselines.py \
    --data_mode smoke --baseline_mode all \
    --max_eval_impressions 300 \
    --output_json results/scenario2_metrics_mind_smoke.json

This script expects processed files produced by prepare_scenario2_mind.py.
It does not download MIND data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from agents.news_baseline_agents import (  # noqa: E402
    NewsInProcessingAgent,
    NewsInProcessingConfig,
    NewsPostProcessingAgent,
    NewsPostProcessingConfig,
    raw_topk,
)
from constraints.news_constraint_handler import NewsConstraintConfig, NewsConstraintHandler  # noqa: E402


LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
EPS = 1e-9
METHODS = ["raw_ranker", "postprocessing", "inprocessing"]


@dataclass
class RankerContext:
    vectorizer: TfidfVectorizer
    news_matrix: Any
    news_by_id: pd.DataFrame
    news_id_to_idx: Dict[str, int]
    popularity: Dict[str, float]


@dataclass
class Scenario2EvalRecord:
    method: str
    impression_id: str
    user_id: str
    candidate_count: int
    positive_count: int
    final_list_size: int
    ndcg_at_10: float
    mrr_at_10: float
    hit_at_10: float
    recall_at_10: float
    mean_base_score: float
    candidate_shortage: bool
    topic_entropy: float
    topic_coverage_at_10: int
    entropy_target_satisfied: float
    diversity_penalty: float
    augmented_lagrangian_penalty: float
    final_objective: float
    num_swaps: int
    search_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario-2 MIND news baselines.")
    parser.add_argument("--data_dir", default="data/processed/mind", help="Directory containing prepared MIND parquet files.")
    parser.add_argument(
        "--data_mode",
        choices=["smoke", "large_sample", "full"],
        default="smoke",
        help="Prepared data suffix to load.",
    )
    parser.add_argument("--output_json", default="results/scenario2_metrics_mind_smoke.json")
    parser.add_argument("--baseline_mode", choices=METHODS + ["all"], default="all")
    parser.add_argument("--max_train_pairs", type=int, default=200_000, help="Maximum train candidate pairs for ranker. Use 0 for all.")
    parser.add_argument("--max_eval_impressions", type=int, default=300, help="Maximum dev impressions to evaluate. Use 0 for all.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--ranker_mode",
        choices=["learned", "position"],
        default="learned",
        help="Use learned TF-IDF/logistic ranker or the original MIND candidate position as base score.",
    )
    parser.add_argument("--tfidf_max_features", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--target_topic_entropy", type=float, default=1.1)
    parser.add_argument("--lambda_diversity", type=float, default=1.0)
    parser.add_argument("--rho_diversity", type=float, default=1.0)
    return parser.parse_args()


def resolve_methods(mode: str) -> List[str]:
    return METHODS.copy() if mode == "all" else [mode]


def load_processed_paths(data_dir: str, data_mode: str) -> Tuple[pd.DataFrame, Path, Path]:
    base = Path(data_dir)
    paths = {
        "news": base / f"mind_news_{data_mode}.parquet",
        "train": base / f"mind_train_pairs_{data_mode}.parquet",
        "dev": base / f"mind_dev_pairs_{data_mode}.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared MIND files: {missing}. Run src/prepare_scenario2_mind.py first."
        )
    news = pd.read_parquet(paths["news"])
    required_news = ["news_id", "category", "subcategory", "text"]
    missing_news = [col for col in required_news if col not in news.columns]
    if missing_news:
        raise ValueError(f"Prepared news file missing columns: {missing_news}")
    return normalize_news(news), paths["train"], paths["dev"]


def normalize_news(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news["news_id"] = news["news_id"].astype(str)
    news["category"] = news["category"].fillna("unknown").astype(str).str.lower()
    news["subcategory"] = news["subcategory"].fillna("unknown").astype(str).str.lower()
    news["text"] = news["text"].fillna("").astype(str)
    return news.drop_duplicates("news_id", keep="first").reset_index(drop=True)


def normalize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["impression_id"] = pairs["impression_id"].astype(str)
    pairs["user_id"] = pairs["user_id"].astype(str)
    pairs["news_id"] = pairs["news_id"].astype(str)
    pairs["history"] = pairs["history"].fillna("").astype(str)
    pairs["request_time"] = pd.to_datetime(pairs["request_time"], errors="coerce")
    pairs["label"] = pd.to_numeric(pairs["label"], errors="coerce").fillna(-1).astype(int)
    pairs["candidate_position"] = pd.to_numeric(pairs["candidate_position"], errors="coerce").fillna(0).astype(int)
    return pairs


def build_ranker_context(
    news: pd.DataFrame,
    popularity: Dict[str, float],
    tfidf_max_features: int,
) -> RankerContext:
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, min_df=1, stop_words="english", ngram_range=(1, 2))
    news_text = news["text"].fillna("").astype(str).tolist()
    LOGGER.info("Fitting TF-IDF on %s news articles.", len(news_text))
    news_matrix = vectorizer.fit_transform(news_text)
    news_id_to_idx = {news_id: idx for idx, news_id in enumerate(news["news_id"].astype(str))}
    return RankerContext(
        vectorizer=vectorizer,
        news_matrix=news_matrix,
        news_by_id=news.set_index("news_id", drop=False),
        news_id_to_idx=news_id_to_idx,
        popularity=popularity,
    )


def _split_history(history: Any) -> List[str]:
    return [token for token in str(history or "").split() if token]


def _history_profile(history: str, context: RankerContext) -> Dict[str, Any]:
    history_ids = _split_history(history)
    idxs = [context.news_id_to_idx[item_id] for item_id in history_ids if item_id in context.news_id_to_idx]
    if idxs:
        hist_vector = np.asarray(context.news_matrix[idxs].mean(axis=0)).reshape(-1)
    else:
        hist_vector = None
    categories = []
    subcategories = []
    for item_id in history_ids:
        if item_id not in context.news_by_id.index:
            continue
        row = context.news_by_id.loc[item_id]
        categories.append(str(row.get("category", "unknown")))
        subcategories.append(str(row.get("subcategory", "unknown")))
    category_counts = pd.Series(categories).value_counts(normalize=True).to_dict() if categories else {}
    subcategory_counts = pd.Series(subcategories).value_counts(normalize=True).to_dict() if subcategories else {}
    return {
        "hist_vector": hist_vector,
        "category_counts": category_counts,
        "subcategory_counts": subcategory_counts,
    }


def build_pair_features(
    pairs: pd.DataFrame,
    context: RankerContext,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows: List[List[float]] = []
    enriched_records: List[Dict[str, Any]] = []
    profile_cache: Dict[str, Dict[str, Any]] = {}
    iterator = pairs.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(pairs), desc="Build ranker features")
    for row in iterator:
        news_id = str(getattr(row, "news_id"))
        news = context.news_by_id.loc[news_id] if news_id in context.news_by_id.index else None
        if news is None:
            category = "unknown"
            subcategory = "unknown"
        else:
            category = str(news.get("category", "unknown"))
            subcategory = str(news.get("subcategory", "unknown"))

        history = str(getattr(row, "history", ""))
        profile = profile_cache.get(history)
        if profile is None:
            profile = _history_profile(history, context)
            profile_cache[history] = profile

        content_sim = 0.0
        idx = context.news_id_to_idx.get(news_id)
        hist_vector = profile["hist_vector"]
        if idx is not None and hist_vector is not None:
            content_sim = float(context.news_matrix[idx].dot(hist_vector).reshape(-1)[0])

        category_interest = float(profile["category_counts"].get(category, 0.0))
        subcategory_interest = float(profile["subcategory_counts"].get(subcategory, 0.0))
        popularity = float(context.popularity.get(news_id, 0.0))
        position = max(1, int(getattr(row, "candidate_position", 1)))
        position_score = 1.0 / np.log2(position + 1.0)

        rows.append([content_sim, category_interest, subcategory_interest, popularity, position_score])
        record = row._asdict()
        record.update(
            {
                "category": category,
                "subcategory": subcategory,
                "feature_content_sim": content_sim,
                "feature_category_interest": category_interest,
                "feature_subcategory_interest": subcategory_interest,
                "feature_popularity": popularity,
                "feature_position": position_score,
            }
        )
        enriched_records.append(record)

    X = np.asarray(rows, dtype=float)
    y = pd.to_numeric(pairs["label"], errors="coerce").fillna(-1).astype(int).to_numpy()
    return X, y, pd.DataFrame(enriched_records)


def build_position_ranker_frame(pairs: pd.DataFrame, news_by_id: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in pairs.itertuples(index=False):
        news_id = str(getattr(row, "news_id"))
        news = news_by_id.loc[news_id] if news_id in news_by_id.index else None
        if news is None:
            category = "unknown"
            subcategory = "unknown"
        else:
            category = str(news.get("category", "unknown"))
            subcategory = str(news.get("subcategory", "unknown"))
        position = max(1, int(getattr(row, "candidate_position", 1)))
        record = row._asdict()
        record.update(
            {
                "category": category,
                "subcategory": subcategory,
                "base_score": float(1.0 / np.log2(position + 1.0)),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def build_position_ranker_frame_batch(pairs: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()
    features = news[["news_id", "category", "subcategory"]].drop_duplicates("news_id", keep="first")
    out = pairs.merge(features, on="news_id", how="left")
    out["category"] = out["category"].fillna("unknown").astype(str)
    out["subcategory"] = out["subcategory"].fillna("unknown").astype(str)
    position = pd.to_numeric(out["candidate_position"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out["base_score"] = 1.0 / np.log2(position + 1.0)
    return out


def _parquet_batches(path: Path, batch_size: int = 250_000):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def compute_popularity(train_pairs: pd.DataFrame) -> Dict[str, float]:
    if train_pairs.empty:
        return {}
    clicks = train_pairs.loc[train_pairs["label"] == 1, "news_id"].astype(str).value_counts().astype(float)
    if clicks.empty:
        return {}
    counts = pd.Series(clicks, dtype=float)
    pop = np.log1p(counts)
    max_pop = float(pop.max()) if len(pop) else 1.0
    return (pop / max(max_pop, EPS)).to_dict()


def sample_train_pairs(train_pairs_path: Path, max_train_pairs: int, seed: int, batch_size: int = 500_000) -> pd.DataFrame:
    del seed
    if max_train_pairs is None or int(max_train_pairs) <= 0:
        frames = []
        for batch in _parquet_batches(train_pairs_path, batch_size=batch_size):
            batch = normalize_pairs(batch)
            labeled = batch.loc[batch["label"].isin([0, 1])].copy()
            if not labeled.empty:
                frames.append(labeled)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    target = int(max_train_pairs)
    frames = []
    collected = 0
    for batch in _parquet_batches(train_pairs_path, batch_size=batch_size):
        batch = normalize_pairs(batch)
        labeled = batch.loc[batch["label"].isin([0, 1])]
        if labeled.empty:
            continue
        need = target - collected
        frames.append(labeled.head(need).copy())
        collected += int(min(need, len(labeled)))
        if collected >= target:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_ranker(train_pairs: pd.DataFrame, context: RankerContext, args: argparse.Namespace) -> Any:
    if train_pairs.empty:
        raise ValueError("No labeled train pairs available for the scenario-2 news ranker.")
    X_train, y_train, _ = build_pair_features(train_pairs, context, show_progress=True)
    unique = np.unique(y_train)
    if unique.size < 2:
        LOGGER.warning("Train labels contain one class only; using DummyClassifier.")
        model = DummyClassifier(strategy="prior")
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=args.seed)
    LOGGER.info("Training lightweight ranker: pairs=%s positives=%s", len(y_train), int(np.sum(y_train == 1)))
    model.fit(X_train, y_train)
    return model


def predict_click_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(X), dtype=float)
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)].astype(float)
    if len(classes) == 1 and classes[0] == 1:
        return np.ones(X.shape[0], dtype=float)
    return np.zeros(X.shape[0], dtype=float)


def ranking_metrics(recommendations: Sequence[str], relevant: set[str], k: int) -> Dict[str, float]:
    top_items = [str(item) for item in recommendations[:k]]
    if not relevant:
        return {"ndcg_at_10": 0.0, "mrr_at_10": 0.0, "hit_at_10": 0.0, "recall_at_10": 0.0}
    dcg = 0.0
    first_rank: Optional[int] = None
    hits = 0
    for idx, item_id in enumerate(top_items):
        if item_id in relevant:
            hits += 1
            dcg += 1.0 / np.log2(idx + 2)
            if first_rank is None:
                first_rank = idx + 1
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return {
        "ndcg_at_10": float(dcg / idcg) if idcg > 0 else 0.0,
        "mrr_at_10": float(1.0 / first_rank) if first_rank else 0.0,
        "hit_at_10": float(1.0 if hits > 0 else 0.0),
        "recall_at_10": float(hits / max(1, len(relevant))),
    }


def make_constraint_handler(args: argparse.Namespace) -> NewsConstraintHandler:
    return NewsConstraintHandler(
        NewsConstraintConfig(
            target_topic_entropy=args.target_topic_entropy,
            lambda_diversity=args.lambda_diversity,
            rho_diversity=args.rho_diversity,
        )
    )


def make_agents(args: argparse.Namespace) -> Dict[str, Any]:
    common = {
        "top_k": args.top_k,
        "target_topic_entropy": args.target_topic_entropy,
        "lambda_diversity": args.lambda_diversity,
        "rho_diversity": args.rho_diversity,
    }
    return {
        "postprocessing": NewsPostProcessingAgent(NewsPostProcessingConfig(**common)),
        "inprocessing": NewsInProcessingAgent(NewsInProcessingConfig(**common, random_seed=args.seed)),
    }


def evaluate_method(
    method: str,
    group: pd.DataFrame,
    handler: NewsConstraintHandler,
    agents: Mapping[str, Any],
    top_k: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    user_id = str(group["user_id"].iloc[0])
    if method == "raw_ranker":
        recs = raw_topk(group, top_k=top_k, base_score_col="base_score")
        constraints = handler.evaluate_all(recs)
        utility = _position_weighted_utility(recs)
        diagnostics = {
            "num_swaps": 0,
            "search_steps": 0,
            "candidate_shortage": len(recs) < top_k,
            "final_constraints": constraints,
            "final_utility": utility,
            "final_objective": float(utility - float(constraints.get("augmented_lagrangian_penalty", 0.0))),
        }
        return recs, diagnostics
    result = agents[method].recommend(user_id=user_id, candidate_items=group, top_k=top_k)
    return result["recommendations"], result["diagnostics"]


def _position_weighted_utility(recs: pd.DataFrame) -> float:
    if recs.empty:
        return 0.0
    scores = pd.to_numeric(recs.get("base_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = 1.0 / np.log2(np.arange(2, len(scores) + 2, dtype=float))
    return float(np.sum(scores * weights) / max(float(np.sum(weights)), 1e-9))


def record_for_impression(
    method: str,
    group: pd.DataFrame,
    recs: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    top_k: int,
) -> Scenario2EvalRecord:
    relevant = set(group.loc[group["label"] == 1, "news_id"].astype(str))
    rec_ids = recs.sort_values("rank")["news_id"].astype(str).tolist() if not recs.empty else []
    metrics = ranking_metrics(rec_ids, relevant, top_k)
    constraints = diagnostics.get("final_constraints", {})
    mean_score = float(pd.to_numeric(recs.get("base_score", pd.Series(dtype=float)), errors="coerce").mean()) if not recs.empty else 0.0
    return Scenario2EvalRecord(
        method=method,
        impression_id=str(group["impression_id"].iloc[0]),
        user_id=str(group["user_id"].iloc[0]),
        candidate_count=int(len(group)),
        positive_count=int(len(relevant)),
        final_list_size=int(len(rec_ids)),
        ndcg_at_10=metrics["ndcg_at_10"],
        mrr_at_10=metrics["mrr_at_10"],
        hit_at_10=metrics["hit_at_10"],
        recall_at_10=metrics["recall_at_10"],
        mean_base_score=mean_score,
        candidate_shortage=bool(diagnostics.get("candidate_shortage", False)),
        topic_entropy=float(constraints.get("topic_entropy", 0.0)),
        topic_coverage_at_10=int(constraints.get("topic_coverage", 0)),
        entropy_target_satisfied=float(1.0 if constraints.get("entropy_target_satisfied", False) else 0.0),
        diversity_penalty=float(constraints.get("diversity_penalty", 0.0)),
        augmented_lagrangian_penalty=float(constraints.get("augmented_lagrangian_penalty", 0.0)),
        final_objective=float(diagnostics.get("final_objective", 0.0)),
        num_swaps=int(diagnostics.get("num_swaps", 0)),
        search_steps=int(diagnostics.get("search_steps", 0)),
    )


def mean_summary(records: List[Scenario2EvalRecord], method: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    if not records:
        return {}
    numeric_keys = [
        "candidate_count",
        "positive_count",
        "final_list_size",
        "ndcg_at_10",
        "mrr_at_10",
        "hit_at_10",
        "recall_at_10",
        "mean_base_score",
        "candidate_shortage",
        "topic_entropy",
        "topic_coverage_at_10",
        "entropy_target_satisfied",
        "diversity_penalty",
        "augmented_lagrangian_penalty",
        "final_objective",
        "num_swaps",
        "search_steps",
    ]
    summary = {key: float(np.mean([getattr(record, key) for record in records])) for key in numeric_keys}
    summary.update(
        {
            "method": method or "mixed",
            "num_impressions": len({record.impression_id for record in records}),
            "num_records": len(records),
            "data_mode": args.data_mode,
            "top_k": args.top_k,
            "target_topic_entropy": args.target_topic_entropy,
        }
    )
    return summary


def save_metrics(
    output_json: str,
    args: argparse.Namespace,
    records: List[Scenario2EvalRecord],
    summaries_by_method: Dict[str, Dict[str, Any]],
    category_exposure_by_method: Mapping[str, Mapping[str, int]],
) -> None:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    payload = {
        "config": config,
        "summary": mean_summary(records, None, args),
        "summaries_by_method": summaries_by_method,
        "category_exposure_by_method": {
            method: dict(sorted(counts.items()))
            for method, counts in category_exposure_by_method.items()
        },
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved scenario-2 news metrics JSON: %s", path)


def log_report(summaries_by_method: Mapping[str, Dict[str, Any]]) -> None:
    LOGGER.info("\n%s", "=" * 78)
    LOGGER.info("Scenario-2 MIND News Baseline Report")
    LOGGER.info("%s", "=" * 78)
    for method, summary in summaries_by_method.items():
        LOGGER.info("[%s] impressions=%s", method, summary.get("num_impressions", 0))
        LOGGER.info("  NDCG@10 / MRR@10 / Hit@10 : %.4f / %.4f / %.4f", summary.get("ndcg_at_10", 0.0), summary.get("mrr_at_10", 0.0), summary.get("hit_at_10", 0.0))
        LOGGER.info("  Entropy / penalty/objective: %.4f / %.4f / %.4f", summary.get("topic_entropy", 0.0), summary.get("augmented_lagrangian_penalty", 0.0), summary.get("final_objective", 0.0))
    LOGGER.info("%s\n", "=" * 78)


def iter_eval_groups(dev_pairs_path: Path, max_eval_impressions: int, batch_size: int = 250_000):
    carryover = pd.DataFrame()
    yielded = 0
    for batch in _parquet_batches(dev_pairs_path, batch_size=batch_size):
        batch = normalize_pairs(batch)
        batch = batch.loc[batch["label"].isin([0, 1])].copy()
        if carryover.empty:
            df = batch
        else:
            df = pd.concat([carryover, batch], ignore_index=True)
        if df.empty:
            carryover = pd.DataFrame()
            continue

        last_impression = str(df["impression_id"].iloc[-1])
        complete = df.loc[df["impression_id"].astype(str) != last_impression].copy()
        carryover = df.loc[df["impression_id"].astype(str) == last_impression].copy()
        for _, group in complete.groupby("impression_id", sort=False):
            if max_eval_impressions and max_eval_impressions > 0 and yielded >= max_eval_impressions:
                return
            yielded += 1
            yield group.reset_index(drop=True)


def iter_eval_frames(dev_pairs_path: Path, max_eval_impressions: int, batch_size: int = 250_000):
    carryover = pd.DataFrame()
    yielded = 0
    for batch in _parquet_batches(dev_pairs_path, batch_size=batch_size):
        batch = normalize_pairs(batch)
        batch = batch.loc[batch["label"].isin([0, 1])].copy()
        if carryover.empty:
            df = batch
        else:
            df = pd.concat([carryover, batch], ignore_index=True)
        if df.empty:
            carryover = pd.DataFrame()
            continue

        last_impression = str(df["impression_id"].iloc[-1])
        complete = df.loc[df["impression_id"].astype(str) != last_impression].copy()
        carryover = df.loc[df["impression_id"].astype(str) == last_impression].copy()
        if complete.empty:
            continue
        if max_eval_impressions and max_eval_impressions > 0:
            impression_ids = complete["impression_id"].drop_duplicates().astype(str).tolist()
            remaining = max_eval_impressions - yielded
            if remaining <= 0:
                return
            keep_ids = set(impression_ids[:remaining])
            complete = complete.loc[complete["impression_id"].astype(str).isin(keep_ids)].copy()
            yielded += len(keep_ids)
            yield complete.reset_index(drop=True)
            if yielded >= max_eval_impressions:
                return
        else:
            yielded += int(complete["impression_id"].nunique())
            yield complete.reset_index(drop=True)

    if not carryover.empty:
        if max_eval_impressions and max_eval_impressions > 0 and yielded >= max_eval_impressions:
            return
        yield carryover.reset_index(drop=True)

    if not carryover.empty:
        for _, group in carryover.groupby("impression_id", sort=False):
            if max_eval_impressions and max_eval_impressions > 0 and yielded >= max_eval_impressions:
                return
            yielded += 1
            yield group.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    methods = resolve_methods(args.baseline_mode)
    news, train_pairs_path, dev_pairs_path = load_processed_paths(args.data_dir, args.data_mode)

    if args.ranker_mode == "learned":
        train_sample = sample_train_pairs(train_pairs_path, args.max_train_pairs, args.seed)
        popularity = compute_popularity(train_sample)
        context = build_ranker_context(news, popularity, args.tfidf_max_features)
        model = train_ranker(train_sample, context, args)
        news_by_id = context.news_by_id
    else:
        context = None
        model = None
        news_by_id = news.set_index("news_id", drop=False)

    handler = make_constraint_handler(args)
    agents = make_agents(args)
    all_records: List[Scenario2EvalRecord] = []
    summaries_by_method: Dict[str, Dict[str, Any]] = {}
    category_exposure_by_method: Dict[str, Counter] = {method: Counter() for method in methods}

    records_by_method: Dict[str, List[Scenario2EvalRecord]] = {method: [] for method in methods}
    started_at = time.time()
    seen_impressions = 0
    if args.ranker_mode == "position":
        frame_iterator = iter_eval_frames(dev_pairs_path, args.max_eval_impressions)
        for frame in tqdm(frame_iterator, desc="Evaluate scenario-2 news"):
            enriched_frame = build_position_ranker_frame_batch(frame, news)
            for _, eval_enriched in enriched_frame.groupby("impression_id", sort=False):
                seen_impressions += 1
                for method in methods:
                    recs, diagnostics = evaluate_method(method, eval_enriched, handler, agents, args.top_k)
                    if not recs.empty and "category" in recs.columns:
                        category_exposure_by_method[method].update(recs["category"].fillna("unknown").astype(str).tolist())
                    records_by_method[method].append(record_for_impression(method, eval_enriched, recs, diagnostics, args.top_k))
    else:
        group_iterator = iter_eval_groups(dev_pairs_path, args.max_eval_impressions)
        for group in tqdm(group_iterator, desc="Evaluate scenario-2 news"):
            X_eval, _, eval_enriched = build_pair_features(group.copy(), context)
            eval_enriched["base_score"] = predict_click_scores(model, X_eval)
            seen_impressions += 1
            for method in methods:
                recs, diagnostics = evaluate_method(method, eval_enriched, handler, agents, args.top_k)
                if not recs.empty and "category" in recs.columns:
                    category_exposure_by_method[method].update(recs["category"].fillna("unknown").astype(str).tolist())
                records_by_method[method].append(record_for_impression(method, eval_enriched, recs, diagnostics, args.top_k))
    if seen_impressions <= 0:
        raise ValueError("No labeled dev impressions available for scenario-2 news evaluation.")
    LOGGER.info("Finished scenario-2 news evaluation for %s impressions in %.2f seconds.", seen_impressions, time.time() - started_at)

    for method in methods:
        method_records = records_by_method[method]
        summaries_by_method[method] = mean_summary(method_records, method, args)
        all_records.extend(method_records)

    save_metrics(args.output_json, args, all_records, summaries_by_method, category_exposure_by_method)
    log_report(summaries_by_method)


if __name__ == "__main__":
    main()
