"""Adapter for the repository's existing All Beauty processed parquet files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import numpy as np
import pandas as pd

from copa.core import CandidateRecord, ConstraintSpec
from copa.data.synthetic import UserCase


class AllBeautyAdapter:
    def __init__(self, processed_dir: Path | str, prefix: str = "beauty_scenario1") -> None:
        self.processed_dir = Path(processed_dir)
        self.prefix = prefix
        self.interactions_path = self.processed_dir / f"{prefix}_interactions.parquet"
        self.items_path = self.processed_dir / f"{prefix}_items.parquet"
        self.users_path = self.processed_dir / f"{prefix}_users.parquet"
        for path in (self.interactions_path, self.items_path, self.users_path):
            if not path.exists():
                raise FileNotFoundError(f"Required All Beauty table not found: {path}")
        self.interactions = pd.read_parquet(
            self.interactions_path,
            columns=["user_id", "item_id", "timestamp"],
        )
        item_columns = [
            "item_id",
            "main_category",
            "brand_id",
            "seller_id",
            "price_filled",
            "popularity",
            "inventory_initial",
        ]
        self.items = pd.read_parquet(self.items_path, columns=item_columns)
        self.users = pd.read_parquet(self.users_path, columns=["user_id", "budget_high"])
        self.items["item_id"] = self.items["item_id"].astype(str)
        self.items = self.items.drop_duplicates("item_id").copy()
        self._popular = self.items.sort_values(["popularity", "item_id"], ascending=[False, True]).reset_index(drop=True)
        self._budget_by_user = self.users.set_index(self.users["user_id"].astype(str))["budget_high"].to_dict()

    def iter_user_cases(
        self,
        *,
        num_users: int,
        candidate_k: int,
        seed: int,
        min_history: int = 2,
    ) -> Iterator[UserCase]:
        interactions = self.interactions.copy()
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["item_id"] = interactions["item_id"].astype(str)
        interactions = interactions.sort_values(["user_id", "timestamp", "item_id"])
        counts = interactions["user_id"].value_counts()
        eligible = sorted(counts[counts >= min_history].index.tolist())
        rng = np.random.default_rng(seed)
        if num_users > 0 and num_users < len(eligible):
            eligible = sorted(rng.choice(eligible, size=num_users, replace=False).tolist())
        grouped = {user_id: group for user_id, group in interactions[interactions["user_id"].isin(eligible)].groupby("user_id", sort=False)}
        for user_id in eligible:
            history = grouped[user_id]
            train = history.iloc[:-1]
            ground_truth = str(history.iloc[-1]["item_id"])
            budget_high = float(self._budget_by_user.get(user_id, self.items["price_filled"].median()))
            candidates = self._generate_candidates(train, candidate_k, budget_high)
            if not candidates:
                continue
            constraints = [
                ConstraintSpec("budget_high", "numeric", "price_filled", "<=", budget_high),
                ConstraintSpec("inventory_available", "numeric", "inventory_initial", ">", 0),
                ConstraintSpec("seen_items", "exclusion", "item_id", "not_in", sorted(set(train["item_id"].astype(str)))),
            ]
            yield UserCase(
                user_id=user_id,
                candidates=candidates,
                constraints=constraints,
                relevant_items=[ground_truth],
                context={"source": "all_beauty", "budget_high": budget_high, "seed": seed},
            )

    def _generate_candidates(self, train: pd.DataFrame, candidate_k: int, budget_high: float) -> List[CandidateRecord]:
        seen = set(train["item_id"].astype(str))
        history_features = self.items[self.items["item_id"].isin(seen)]
        preferred_brands = {value for value, _ in Counter(history_features["brand_id"].dropna()).most_common(5)}
        preferred_categories = {value for value, _ in Counter(history_features["main_category"].dropna()).most_common(3)}
        matching = self.items[
            self.items["brand_id"].isin(preferred_brands) | self.items["main_category"].isin(preferred_categories)
        ].nlargest(max(candidate_k * 4, candidate_k), "popularity")
        affordable_catalog = self.items[
            pd.to_numeric(self.items["price_filled"], errors="coerce") <= budget_high
        ].nlargest(max(candidate_k * 4, candidate_k), "popularity")
        pool = pd.concat(
            [matching, affordable_catalog, self._popular.head(candidate_k * 4)], ignore_index=True
        ).drop_duplicates("item_id")
        pool = pool[~pool["item_id"].isin(seen)].copy()
        pool["base_score"] = (
            0.65 * pd.to_numeric(pool["popularity"], errors="coerce").fillna(0.0)
            + 0.25 * pool["brand_id"].isin(preferred_brands).astype(float)
            + 0.10 * pool["main_category"].isin(preferred_categories).astype(float)
        )
        pool = pool.sort_values(["base_score", "item_id"], ascending=[False, True])
        # Preserve a relevance-oriented majority while reserving capacity for the
        # user's feasible price region. This is candidate generation, not repair:
        # hard constraints still make the final decision and remain independently verified.
        relevance_quota = max(1, int(np.ceil(candidate_k * 0.6)))
        relevance_pool = pool.head(relevance_quota)
        affordable_pool = pool[
            (pd.to_numeric(pool["price_filled"], errors="coerce") <= budget_high)
            & ~pool["item_id"].isin(relevance_pool["item_id"])
        ].head(candidate_k - len(relevance_pool))
        selected = pd.concat([relevance_pool, affordable_pool], ignore_index=True).drop_duplicates("item_id")
        if len(selected) < candidate_k:
            fallback = pool[~pool["item_id"].isin(selected["item_id"])].head(candidate_k - len(selected))
            selected = pd.concat([selected, fallback], ignore_index=True)
        pool = selected.sort_values(["base_score", "item_id"], ascending=[False, True]).head(candidate_k)
        records: List[CandidateRecord] = []
        for row in pool.to_dict("records"):
            item_id = str(row.pop("item_id"))
            base_score = float(row.pop("base_score"))
            metadata = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()}
            records.append(CandidateRecord(item_id, base_score, metadata, "all_beauty_adapter"))
        return records


def candidates_from_precomputed_scores(
    frame: pd.DataFrame,
    *,
    item_id_col: str = "item_id",
    score_col: str = "base_score",
    source: str = "precomputed",
) -> List[CandidateRecord]:
    required = {item_id_col, score_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Precomputed candidate frame missing columns: {sorted(missing)}")
    records: List[CandidateRecord] = []
    for row in frame.to_dict("records"):
        item_id = str(row.pop(item_id_col))
        score = float(row.pop(score_col))
        records.append(CandidateRecord(item_id, score, row, source))
    return records
