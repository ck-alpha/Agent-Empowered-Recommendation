"""
Scenario-2 news recommendation constraints.

The MIND news setting uses a single soft constraint: topic diversity measured
by Shannon entropy. There are no hard news-side constraints in this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class NewsConstraintConfig:
    """Configuration for scenario-2 news topic-entropy constraints."""

    topic_col: str = "category"
    target_topic_entropy: float = 1.1
    lambda_diversity: float = 1.0
    rho_diversity: float = 1.0


class NewsConstraintHandler:
    """Lightweight helper for topic entropy and ALM diversity penalty."""

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

    def topic_series(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.Series:
        df = self.to_dataframe(records)
        if df.empty:
            return pd.Series(dtype=object)
        self.require_columns(df, [self.config.topic_col], "topic_series")
        return df[self.config.topic_col].fillna("unknown").astype(str)

    def topic_distribution(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, float]:
        topics = self.topic_series(records)
        if topics.empty:
            return {}
        return topics.value_counts(normalize=True).astype(float).to_dict()

    def topic_counts(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, int]:
        topics = self.topic_series(records)
        if topics.empty:
            return {}
        return topics.value_counts().astype(int).to_dict()

    @staticmethod
    def entropy_from_counts(counts: Dict[str, int], total_count: Optional[int] = None) -> float:
        total = int(sum(counts.values()) if total_count is None else total_count)
        if total <= 0:
            return 0.0
        probs = np.asarray([count / total for count in counts.values() if count > 0], dtype=float)
        return float(-np.sum(probs * np.log(probs))) if probs.size else 0.0

    def topic_entropy(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        return self.entropy_from_counts(self.topic_counts(records))

    def diversity_penalty_from_entropy(self, entropy: float) -> float:
        return float(max(0.0, float(self.config.target_topic_entropy) - float(entropy)))

    def diversity_penalty(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        return self.diversity_penalty_from_entropy(self.topic_entropy(records))

    def augmented_lagrangian_penalty(self, diversity_penalty: float) -> float:
        violation = max(0.0, float(diversity_penalty))
        return float(
            self.config.lambda_diversity * violation
            + 0.5 * self.config.rho_diversity * violation**2
        )

    def evaluate_all(self, records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        df = self.to_dataframe(records)
        if df.empty:
            entropy = 0.0
            coverage = 0
            distribution: Dict[str, float] = {}
        else:
            self.require_columns(df, [self.config.topic_col], "evaluate_all")
            counts = self.topic_counts(df)
            distribution = {topic: float(count) / len(df) for topic, count in counts.items()}
            entropy = self.entropy_from_counts(counts, len(df))
            coverage = int(len(counts))
        penalty = self.diversity_penalty_from_entropy(entropy)
        return {
            "topic_entropy": float(entropy),
            "topic_coverage": int(coverage),
            "topic_distribution": distribution,
            "target_topic_entropy": float(self.config.target_topic_entropy),
            "entropy_target_satisfied": bool(penalty <= 0.0),
            "diversity_penalty": float(penalty),
            "augmented_lagrangian_penalty": self.augmented_lagrangian_penalty(penalty),
        }
