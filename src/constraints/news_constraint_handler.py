"""
Scenario-3 news recommendation constraints.

This module is intentionally independent from the scenario-1 e-commerce and
scenario-2 recruiting constraint handlers. It evaluates only the compact news
baseline protocol:

Hard constraints:
1) category-aware freshness;
2) Top-N topic concentration.

Soft constraints:
1) topic entropy;
2) reading-load balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd


DEFAULT_FRESHNESS_HOURS: Dict[str, float] = {
    "weather": 24.0,
    "sports": 48.0,
    "news": 72.0,
    "finance": 72.0,
    "entertainment": 168.0,
    "tv": 168.0,
    "music": 168.0,
    "movies": 168.0,
    "video": 168.0,
    "health": 720.0,
    "lifestyle": 720.0,
    "foodanddrink": 720.0,
    "travel": 720.0,
    "autos": 720.0,
}


@dataclass
class NewsConstraintConfig:
    """Configuration for scenario-3 news constraints."""

    top_n: int = 5
    max_topic_count: int = 2
    target_topic_entropy: float = 1.1
    target_load: float = 45.0
    load_tolerance: float = 25.0
    lambda_diversity: float = 1.0
    lambda_load: float = 0.02
    rho_diversity: float = 1.0
    rho_load: float = 0.001
    default_freshness_hours: float = 168.0
    freshness_hours_by_category: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FRESHNESS_HOURS)
    )


class NewsConstraintHandler:
    """Constraint and penalty helper for news recommendation slates."""

    def __init__(self, config: Optional[NewsConstraintConfig] = None):
        self.config = config or NewsConstraintConfig()

    @staticmethod
    def to_dataframe(records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, list):
            return pd.DataFrame(records).copy()
        raise TypeError("records must be a pandas DataFrame or a list of dictionaries.")

    @staticmethod
    def require_columns(df: pd.DataFrame, columns: Iterable[str], method_name: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{method_name} requires columns: {missing}")

    def freshness_thresholds(self, categories: pd.Series) -> pd.Series:
        mapping = {str(key).lower(): float(value) for key, value in self.config.freshness_hours_by_category.items()}
        return categories.fillna("").astype(str).str.lower().map(mapping).fillna(self.config.default_freshness_hours)

    def add_age_hours(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Return a copy with a numeric age_hours column."""
        df = self.to_dataframe(records)
        if df.empty:
            df["age_hours"] = pd.Series(dtype=float)
            return df
        if "age_hours" in df.columns:
            df["age_hours"] = pd.to_numeric(df["age_hours"], errors="coerce").fillna(0.0).clip(lower=0.0)
            return df

        self.require_columns(df, ["request_time", "publish_time_proxy"], "add_age_hours")
        request_time = pd.to_datetime(df["request_time"], errors="coerce")
        publish_time = pd.to_datetime(df["publish_time_proxy"], errors="coerce")
        age = (request_time - publish_time).dt.total_seconds() / 3600.0
        df["age_hours"] = pd.to_numeric(age, errors="coerce").fillna(0.0).clip(lower=0.0)
        return df

    def freshness_mask(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.Series:
        df = self.add_age_hours(records)
        if df.empty:
            return pd.Series(dtype=bool, index=df.index)
        self.require_columns(df, ["category", "age_hours"], "freshness_mask")
        thresholds = self.freshness_thresholds(df["category"])
        return pd.to_numeric(df["age_hours"], errors="coerce").fillna(np.inf) <= thresholds

    def filter_freshness(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        df = self.add_age_hours(records)
        if df.empty:
            return df
        return df.loc[self.freshness_mask(df)].copy()

    def freshness_diagnostics(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        df = self.add_age_hours(records)
        if df.empty:
            return {
                "freshness_satisfied": True,
                "freshness_violation_count": 0,
                "freshness_violation_rate": 0.0,
            }
        mask = self.freshness_mask(df)
        violation_count = int((~mask).sum())
        return {
            "freshness_satisfied": bool(violation_count == 0),
            "freshness_violation_count": violation_count,
            "freshness_violation_rate": float(violation_count / max(1, len(df))),
        }

    def topn_topic_diagnostics(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        df = self.to_dataframe(records)
        if df.empty:
            return {
                "topn_topic_satisfied": True,
                "topn_topic_violation_amount": 0.0,
                "topn_topic_violation_rate": 0.0,
                "max_topn_topic_count": 0,
            }
        self.require_columns(df, ["category"], "topn_topic_diagnostics")
        ordered = self._ordered_slate(df)
        n = min(max(0, int(self.config.top_n)), len(ordered))
        if n == 0:
            return {
                "topn_topic_satisfied": True,
                "topn_topic_violation_amount": 0.0,
                "topn_topic_violation_rate": 0.0,
                "max_topn_topic_count": 0,
            }
        counts = ordered.head(n)["category"].fillna("unknown").astype(str).value_counts()
        max_count = int(counts.max()) if len(counts) else 0
        overflow = float(max(0, max_count - int(self.config.max_topic_count)))
        return {
            "topn_topic_satisfied": bool(overflow <= 0.0),
            "topn_topic_violation_amount": overflow,
            "topn_topic_violation_rate": float(1.0 if overflow > 0.0 else 0.0),
            "max_topn_topic_count": max_count,
        }

    @staticmethod
    def _ordered_slate(df: pd.DataFrame) -> pd.DataFrame:
        if "rank" in df.columns:
            out = df.copy()
            out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(np.inf)
            return out.sort_values(["rank", "news_id"], ascending=[True, True]).reset_index(drop=True)
        return df.reset_index(drop=True)

    def topic_entropy(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        df = self.to_dataframe(records)
        if df.empty:
            return 0.0
        self.require_columns(df, ["category"], "topic_entropy")
        probs = df["category"].fillna("unknown").astype(str).value_counts(normalize=True).to_numpy(dtype=float)
        probs = probs[probs > 0.0]
        return float(-np.sum(probs * np.log(probs))) if probs.size else 0.0

    def diversity_penalty(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        entropy = self.topic_entropy(records)
        return float(max(0.0, self.config.target_topic_entropy - entropy))

    def load_penalty(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        df = self.to_dataframe(records)
        if df.empty:
            return 0.0
        self.require_columns(df, ["word_count"], "load_penalty")
        word_count = pd.to_numeric(df["word_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        avg_load = float(word_count.mean()) if len(word_count) else 0.0
        return float(max(0.0, abs(avg_load - self.config.target_load) - self.config.load_tolerance))

    def augmented_lagrangian_penalty(self, diversity_penalty: float, load_penalty: float) -> float:
        div = max(0.0, float(diversity_penalty))
        load = max(0.0, float(load_penalty))
        return float(
            self.config.lambda_diversity * div
            + 0.5 * self.config.rho_diversity * div ** 2
            + self.config.lambda_load * load
            + 0.5 * self.config.rho_load * load ** 2
        )

    def evaluate_all(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        df = self.add_age_hours(records)
        if not df.empty:
            if "rank" not in df.columns:
                df = df.copy()
                df["rank"] = np.arange(1, len(df) + 1)

        freshness = self.freshness_diagnostics(df)
        topn = self.topn_topic_diagnostics(df)
        entropy = self.topic_entropy(df)
        topic_coverage = int(df["category"].dropna().astype(str).nunique()) if "category" in df.columns else 0
        avg_age = float(pd.to_numeric(df.get("age_hours", pd.Series(dtype=float)), errors="coerce").mean()) if len(df) else 0.0
        word_count = pd.to_numeric(df.get("word_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        avg_word_count = float(word_count.mean()) if len(word_count) else 0.0
        load_deviation = float(abs(avg_word_count - self.config.target_load)) if len(word_count) else 0.0
        div_penalty = self.diversity_penalty(df)
        load_penalty = self.load_penalty(df)
        alm = self.augmented_lagrangian_penalty(div_penalty, load_penalty)
        all_hard = bool(freshness["freshness_satisfied"] and topn["topn_topic_satisfied"])

        return {
            **freshness,
            **topn,
            "all_hard_constraints_satisfied": all_hard,
            "topic_entropy": float(entropy),
            "topic_coverage": int(topic_coverage),
            "avg_age_hours": avg_age,
            "avg_word_count": avg_word_count,
            "load_deviation": load_deviation,
            "diversity_penalty": float(div_penalty),
            "load_penalty": float(load_penalty),
            "augmented_lagrangian_penalty": float(alm),
        }

    def hard_violation_amount(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        diagnostics = self.evaluate_all(records)
        freshness = float(diagnostics.get("freshness_violation_rate", 0.0))
        topn = float(diagnostics.get("topn_topic_violation_amount", 0.0)) / max(1, int(self.config.top_n))
        return float(freshness + topn)
