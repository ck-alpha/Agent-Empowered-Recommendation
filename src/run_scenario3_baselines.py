"""
Scenario-3 offline runner: MIND news ranker -> constrained baselines.

Recommended smoke command:
  python src/run_scenario3_baselines.py \
    --data_mode smoke --baseline_mode all \
    --max_eval_impressions 300 \
    --output_json results/scenario3_metrics_mind_smoke.json

This script expects processed files produced by prepare_scenario3_mind.py.
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
    hard_constrained_topk,
)
from constraints.news_constraint_handler import NewsConstraintConfig, NewsConstraintHandler  # noqa: E402


LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
EPS = 1e-9
METHODS = ["raw_ranker", "hard_filter", "postprocessing", "inprocessing"]


@dataclass
class RankerContext:
    vectorizer: TfidfVectorizer
    news_matrix: Any
    news_by_id: pd.DataFrame
    news_id_to_idx: Dict[str, int]
    popularity: Dict[str, float]


@dataclass
class Scenario3EvalRecord:
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
    freshness_violation_rate: float
    topn_topic_violation_rate: float
    feasible_rate: float
    candidate_shortage: bool
    topic_entropy: float
    topic_coverage_at_10: int
    avg_age_hours: float
    avg_word_count: float
    load_deviation: float
    diversity_penalty: float
    load_penalty: float
    augmented_lagrangian_penalty: float
    num_swaps: int
    search_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario-3 MIND news baselines.")
    parser.add_argument("--data_dir", default="data/processed/mind", help="Directory containing prepared MIND parquet files.")
    parser.add_argument(
        "--data_mode",
        choices=["smoke", "large_sample", "full"],
        default="smoke",
        help="Prepared data suffix to load.",
    )
    parser.add_argument("--output_json", default="results/scenario3_metrics_mind_smoke.json")
    parser.add_argument("--baseline_mode", choices=METHODS + ["all"], default="all")
    parser.add_argument("--max_train_pairs", type=int, default=200_000, help="Maximum train candidate pairs for ranker. Use 0 for all.")
    parser.add_argument("--max_eval_impressions", type=int, default=300, help="Maximum dev impressions to evaluate. Use 0 for all.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tfidf_max_features", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--max_topic_count", type=int, default=2)
    parser.add_argument("--target_topic_entropy", type=float, default=1.1)
    parser.add_argument("--target_load", type=float, default=0.0, help="Use <=0 to auto-set from news word-count median.")
    parser.add_argument("--load_tolerance", type=float, default=0.0, help="Use <=0 to auto-set from news word-count spread.")
    parser.add_argument("--lambda_diversity", type=float, default=1.0)
    parser.add_argument("--lambda_load", type=float, default=0.02)
    parser.add_argument("--rho_diversity", type=float, default=1.0)
    parser.add_argument("--rho_load", type=float, default=0.001)
    parser.add_argument("--in_population_size", type=int, default=24)
    parser.add_argument("--in_generations", type=int, default=12)
    return parser.parse_args()


def resolve_methods(mode: str) -> List[str]:
    return METHODS.copy() if mode == "all" else [mode]


def load_processed_tables(data_dir: str, data_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    paths = {
        "news": base / f"mind_news_{data_mode}.parquet",
        "train": base / f"mind_train_pairs_{data_mode}.parquet",
        "dev": base / f"mind_dev_pairs_{data_mode}.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared MIND files: {missing}. Run src/prepare_scenario3_mind.py first."
        )
    news = pd.read_parquet(paths["news"])
    train_pairs = pd.read_parquet(paths["train"])
    dev_pairs = pd.read_parquet(paths["dev"])
    required_news = ["news_id", "category", "subcategory", "text", "word_count", "publish_time_proxy"]
    required_pairs = ["impression_id", "user_id", "request_time", "history", "news_id", "label", "candidate_position"]
    missing_news = [col for col in required_news if col not in news.columns]
    missing_train = [col for col in required_pairs if col not in train_pairs.columns]
    missing_dev = [col for col in required_pairs if col not in dev_pairs.columns]
    if missing_news or missing_train or missing_dev:
        raise ValueError(f"Prepared files missing columns. news={missing_news}, train={missing_train}, dev={missing_dev}")
    return normalize_news(news), normalize_pairs(train_pairs), normalize_pairs(dev_pairs)


def normalize_news(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news["news_id"] = news["news_id"].astype(str)
    news["category"] = news["category"].fillna("unknown").astype(str).str.lower()
    news["subcategory"] = news["subcategory"].fillna("unknown").astype(str).str.lower()
    news["text"] = news["text"].fillna("").astype(str)
    news["word_count"] = pd.to_numeric(news["word_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    news["publish_time_proxy"] = pd.to_datetime(news["publish_time_proxy"], errors="coerce")
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


def resolve_load_config(news: pd.DataFrame, args: argparse.Namespace) -> Tuple[float, float]:
    word_count = pd.to_numeric(news["word_count"], errors="coerce").fillna(0.0)
    positive = word_count.loc[word_count > 0]
    target = float(args.target_load)
    tolerance = float(args.load_tolerance)
    if target <= 0.0:
        target = float(positive.median()) if len(positive) else 45.0
    if tolerance <= 0.0:
        q25 = float(positive.quantile(0.25)) if len(positive) else 20.0
        q75 = float(positive.quantile(0.75)) if len(positive) else 70.0
        tolerance = max(10.0, 0.5 * (q75 - q25))
    return target, tolerance


def build_ranker_context(news: pd.DataFrame, train_pairs: pd.DataFrame, tfidf_max_features: int) -> RankerContext:
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, min_df=1, stop_words="english", ngram_range=(1, 2))
    news_text = news["text"].fillna("").astype(str).tolist()
    LOGGER.info("Fitting TF-IDF on %s news articles.", len(news_text))
    news_matrix = vectorizer.fit_transform(news_text)
    news_id_to_idx = {news_id: idx for idx, news_id in enumerate(news["news_id"].astype(str))}
    clicks = train_pairs.loc[train_pairs["label"] == 1, "news_id"].astype(str).value_counts().astype(float)
    pop = np.log1p(clicks)
    max_pop = float(pop.max()) if len(pop) else 1.0
    popularity = (pop / max(max_pop, EPS)).to_dict() if len(pop) else {}
    news_by_id = news.set_index("news_id", drop=False)
    return RankerContext(vectorizer, news_matrix, news_by_id, news_id_to_idx, popularity)


def _split_history(history: Any) -> List[str]:
    return [token for token in str(history or "").split() if token]


def _age_hours(request_time: Any, publish_time: Any) -> float:
    request = pd.to_datetime(request_time, errors="coerce")
    publish = pd.to_datetime(publish_time, errors="coerce")
    if pd.isna(request) or pd.isna(publish):
        return 0.0
    return float(max(0.0, (request - publish).total_seconds() / 3600.0))


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


def build_pair_features(pairs: pd.DataFrame, context: RankerContext) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows: List[List[float]] = []
    enriched_records: List[Dict[str, Any]] = []
    profile_cache: Dict[str, Dict[str, Any]] = {}
    iterator = tqdm(pairs.itertuples(index=False), total=len(pairs), desc="Build ranker features")
    for row in iterator:
        news_id = str(getattr(row, "news_id"))
        news = context.news_by_id.loc[news_id] if news_id in context.news_by_id.index else None
        if news is None:
            category = "unknown"
            subcategory = "unknown"
            word_count = 0.0
            publish_time = pd.NaT
        else:
            category = str(news.get("category", "unknown"))
            subcategory = str(news.get("subcategory", "unknown"))
            word_count = float(news.get("word_count", 0.0))
            publish_time = news.get("publish_time_proxy")

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
        age = _age_hours(getattr(row, "request_time"), publish_time)
        freshness_score = 1.0 / (1.0 + age / 72.0)
        position = max(1, int(getattr(row, "candidate_position", 1)))
        position_score = 1.0 / np.log2(position + 1.0)
        log_word_count = float(np.log1p(max(0.0, word_count)) / 6.0)

        rows.append(
            [
                content_sim,
                category_interest,
                subcategory_interest,
                popularity,
                freshness_score,
                log_word_count,
                position_score,
            ]
        )
        record = row._asdict()
        record.update(
            {
                "category": category,
                "subcategory": subcategory,
                "word_count": word_count,
                "publish_time_proxy": publish_time,
                "age_hours": age,
                "feature_content_sim": content_sim,
                "feature_category_interest": category_interest,
                "feature_subcategory_interest": subcategory_interest,
                "feature_popularity": popularity,
                "feature_freshness": freshness_score,
                "feature_position": position_score,
            }
        )
        enriched_records.append(record)

    X = np.asarray(rows, dtype=float)
    y = pd.to_numeric(pairs["label"], errors="coerce").fillna(-1).astype(int).to_numpy()
    return X, y, pd.DataFrame(enriched_records)


def sample_train_pairs(train_pairs: pd.DataFrame, max_train_pairs: int, seed: int) -> pd.DataFrame:
    labeled = train_pairs.loc[train_pairs["label"].isin([0, 1])].copy()
    if max_train_pairs and max_train_pairs > 0 and len(labeled) > max_train_pairs:
        labeled = labeled.sample(n=max_train_pairs, random_state=seed)
    return labeled.reset_index(drop=True)


def train_ranker(train_pairs: pd.DataFrame, context: RankerContext, args: argparse.Namespace) -> Any:
    sampled = sample_train_pairs(train_pairs, args.max_train_pairs, args.seed)
    if sampled.empty:
        raise ValueError("No labeled train pairs available for the scenario-3 ranker.")
    X_train, y_train, _ = build_pair_features(sampled, context)
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


def select_eval_pairs(dev_pairs: pd.DataFrame, max_eval_impressions: int) -> pd.DataFrame:
    labeled = dev_pairs.loc[dev_pairs["label"].isin([0, 1])].copy()
    impression_ids = labeled["impression_id"].drop_duplicates().astype(str).tolist()
    if max_eval_impressions and max_eval_impressions > 0:
        impression_ids = impression_ids[:max_eval_impressions]
    return labeled.loc[labeled["impression_id"].isin(impression_ids)].reset_index(drop=True)


def raw_ranker_recommend(group: pd.DataFrame, top_k: int) -> pd.DataFrame:
    recs = group.sort_values(["base_score", "news_id"], ascending=[False, True]).head(top_k).copy().reset_index(drop=True)
    recs["rank"] = np.arange(1, len(recs) + 1)
    return recs


def hard_filter_recommend(group: pd.DataFrame, handler: NewsConstraintHandler, top_k: int) -> pd.DataFrame:
    return hard_constrained_topk(group, handler, top_k=top_k, base_score_col="base_score")


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


def make_constraint_handler(args: argparse.Namespace, target_load: float, load_tolerance: float) -> NewsConstraintHandler:
    return NewsConstraintHandler(
        NewsConstraintConfig(
            top_n=args.top_n,
            max_topic_count=args.max_topic_count,
            target_topic_entropy=args.target_topic_entropy,
            target_load=target_load,
            load_tolerance=load_tolerance,
            lambda_diversity=args.lambda_diversity,
            lambda_load=args.lambda_load,
            rho_diversity=args.rho_diversity,
            rho_load=args.rho_load,
        )
    )


def make_agents(args: argparse.Namespace, target_load: float, load_tolerance: float) -> Dict[str, Any]:
    common = {
        "top_k": args.top_k,
        "top_n": args.top_n,
        "max_topic_count": args.max_topic_count,
        "target_topic_entropy": args.target_topic_entropy,
        "target_load": target_load,
        "load_tolerance": load_tolerance,
        "lambda_diversity": args.lambda_diversity,
        "lambda_load": args.lambda_load,
        "rho_diversity": args.rho_diversity,
        "rho_load": args.rho_load,
    }
    return {
        "postprocessing": NewsPostProcessingAgent(NewsPostProcessingConfig(**common)),
        "inprocessing": NewsInProcessingAgent(
            NewsInProcessingConfig(
                **common,
                population_size=args.in_population_size,
                max_generations=args.in_generations,
                random_seed=args.seed,
            )
        ),
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
        recs = raw_ranker_recommend(group, top_k)
        diagnostics = {"num_swaps": 0, "search_steps": 0, "candidate_shortage": len(recs) < top_k}
        diagnostics["final_constraints"] = handler.evaluate_all(recs)
        diagnostics["fully_repaired"] = bool(diagnostics["final_constraints"].get("all_hard_constraints_satisfied", False))
        return recs, diagnostics
    if method == "hard_filter":
        recs = hard_filter_recommend(group, handler, top_k)
        diagnostics = {"num_swaps": 0, "search_steps": 0, "candidate_shortage": len(recs) < top_k}
        diagnostics["final_constraints"] = handler.evaluate_all(recs)
        diagnostics["fully_repaired"] = bool(diagnostics["final_constraints"].get("all_hard_constraints_satisfied", False))
        return recs, diagnostics
    result = agents[method].recommend(user_id=user_id, candidate_items=group, top_k=top_k)
    return result["recommendations"], result["diagnostics"]


def record_for_impression(
    method: str,
    group: pd.DataFrame,
    recs: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    top_k: int,
) -> Scenario3EvalRecord:
    relevant = set(group.loc[group["label"] == 1, "news_id"].astype(str))
    rec_ids = recs.sort_values("rank")["news_id"].astype(str).tolist() if not recs.empty else []
    metrics = ranking_metrics(rec_ids, relevant, top_k)
    constraints = diagnostics.get("final_constraints", {})
    mean_score = float(pd.to_numeric(recs.get("base_score", pd.Series(dtype=float)), errors="coerce").mean()) if not recs.empty else 0.0
    return Scenario3EvalRecord(
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
        freshness_violation_rate=float(constraints.get("freshness_violation_rate", 0.0)),
        topn_topic_violation_rate=float(constraints.get("topn_topic_violation_rate", 0.0)),
        feasible_rate=float(1.0 if constraints.get("all_hard_constraints_satisfied", False) else 0.0),
        candidate_shortage=bool(diagnostics.get("candidate_shortage", False)),
        topic_entropy=float(constraints.get("topic_entropy", 0.0)),
        topic_coverage_at_10=int(constraints.get("topic_coverage", 0)),
        avg_age_hours=float(constraints.get("avg_age_hours", 0.0)),
        avg_word_count=float(constraints.get("avg_word_count", 0.0)),
        load_deviation=float(constraints.get("load_deviation", 0.0)),
        diversity_penalty=float(constraints.get("diversity_penalty", 0.0)),
        load_penalty=float(constraints.get("load_penalty", 0.0)),
        augmented_lagrangian_penalty=float(constraints.get("augmented_lagrangian_penalty", 0.0)),
        num_swaps=int(diagnostics.get("num_swaps", 0)),
        search_steps=int(diagnostics.get("search_steps", 0)),
    )


def mean_summary(records: List[Scenario3EvalRecord], method: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
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
        "freshness_violation_rate",
        "topn_topic_violation_rate",
        "feasible_rate",
        "candidate_shortage",
        "topic_entropy",
        "topic_coverage_at_10",
        "avg_age_hours",
        "avg_word_count",
        "load_deviation",
        "diversity_penalty",
        "load_penalty",
        "augmented_lagrangian_penalty",
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
            "top_n": args.top_n,
            "max_topic_count": args.max_topic_count,
        }
    )
    return summary


def save_metrics(
    output_json: str,
    args: argparse.Namespace,
    target_load: float,
    load_tolerance: float,
    records: List[Scenario3EvalRecord],
    summaries_by_method: Dict[str, Dict[str, Any]],
    category_exposure_by_method: Mapping[str, Mapping[str, int]],
) -> None:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = mean_summary(records, None, args)
    config = vars(args).copy()
    config["resolved_target_load"] = target_load
    config["resolved_load_tolerance"] = load_tolerance
    payload = {
        "config": config,
        "summary": summary,
        "summaries_by_method": summaries_by_method,
        "category_exposure_by_method": {
            method: dict(sorted(counts.items()))
            for method, counts in category_exposure_by_method.items()
        },
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved scenario-3 metrics JSON: %s", path)


def log_report(summaries_by_method: Mapping[str, Dict[str, Any]]) -> None:
    LOGGER.info("\n%s", "=" * 78)
    LOGGER.info("Scenario-3 MIND News Baseline Report")
    LOGGER.info("%s", "=" * 78)
    for method, summary in summaries_by_method.items():
        LOGGER.info("[%s] impressions=%s", method, summary.get("num_impressions", 0))
        LOGGER.info("  NDCG@10 / MRR@10 / Hit@10 : %.4f / %.4f / %.4f", summary.get("ndcg_at_10", 0.0), summary.get("mrr_at_10", 0.0), summary.get("hit_at_10", 0.0))
        LOGGER.info("  CSR / freshness / topN    : %.4f / %.4f / %.4f", summary.get("feasible_rate", 0.0), 1.0 - summary.get("freshness_violation_rate", 0.0), 1.0 - summary.get("topn_topic_violation_rate", 0.0))
        LOGGER.info("  Entropy / load penalty    : %.4f / %.4f", summary.get("topic_entropy", 0.0), summary.get("load_penalty", 0.0))
    LOGGER.info("%s\n", "=" * 78)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    methods = resolve_methods(args.baseline_mode)
    news, train_pairs, dev_pairs = load_processed_tables(args.data_dir, args.data_mode)
    target_load, load_tolerance = resolve_load_config(news, args)
    LOGGER.info("Resolved load target/tolerance: %.3f / %.3f", target_load, load_tolerance)

    context = build_ranker_context(news, train_pairs, args.tfidf_max_features)
    model = train_ranker(train_pairs, context, args)
    eval_pairs = select_eval_pairs(dev_pairs, args.max_eval_impressions)
    if eval_pairs.empty:
        raise ValueError("No labeled dev impressions available for scenario-3 evaluation.")
    X_eval, _, eval_enriched = build_pair_features(eval_pairs, context)
    eval_enriched["base_score"] = predict_click_scores(model, X_eval)
    LOGGER.info(
        "Evaluation pairs ready: impressions=%s pairs=%s",
        eval_enriched["impression_id"].nunique(),
        len(eval_enriched),
    )

    handler = make_constraint_handler(args, target_load, load_tolerance)
    agents = make_agents(args, target_load, load_tolerance)
    all_records: List[Scenario3EvalRecord] = []
    summaries_by_method: Dict[str, Dict[str, Any]] = {}
    category_exposure_by_method: Dict[str, Counter] = {method: Counter() for method in methods}

    groups = list(eval_enriched.groupby("impression_id", sort=False))
    for method in methods:
        LOGGER.info("Running scenario-3 baseline: %s", method)
        started_at = time.time()
        method_records: List[Scenario3EvalRecord] = []
        for _, group in tqdm(groups, desc=f"Evaluate {method}"):
            recs, diagnostics = evaluate_method(method, group.copy(), handler, agents, args.top_k)
            if not recs.empty and "category" in recs.columns:
                category_exposure_by_method[method].update(
                    recs["category"].fillna("unknown").astype(str).tolist()
                )
            method_records.append(record_for_impression(method, group, recs, diagnostics, args.top_k))
        summaries_by_method[method] = mean_summary(method_records, method, args)
        all_records.extend(method_records)
        LOGGER.info("Finished %s in %.2f seconds.", method, time.time() - started_at)

    save_metrics(
        args.output_json,
        args,
        target_load,
        load_tolerance,
        all_records,
        summaries_by_method,
        category_exposure_by_method,
    )
    log_report(summaries_by_method)


if __name__ == "__main__":
    main()
