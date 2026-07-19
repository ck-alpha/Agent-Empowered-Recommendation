"""Leakage-safe temporal popularity full-sort baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd


def _stage_rows(
    split: pd.DataFrame,
    *,
    stage: str,
    visible_splits: Iterable[str],
    candidate_ks: Sequence[int],
) -> list[Dict[str, Any]]:
    visible = split[split["split"].isin(set(visible_splits))]
    targets = split.loc[split["split"] == stage, ["user_id", "item_id"]]
    if targets["user_id"].duplicated().any():
        raise ValueError(f"Popularity baseline requires one {stage} target per user")

    catalog = sorted(split["item_id"].astype(str).unique())
    counts = visible["item_id"].astype(str).value_counts().to_dict()
    ranked_catalog = sorted(catalog, key=lambda item_id: (-int(counts.get(item_id, 0)), item_id))
    global_position = {item_id: index for index, item_id in enumerate(ranked_catalog, start=1)}
    histories = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in visible.groupby("user_id")
    }

    target_ranks: list[int] = []
    covered: list[bool] = []
    for row in targets.itertuples(index=False):
        user_id = str(row.user_id)
        target_id = str(row.item_id)
        seen = histories.get(user_id, set())
        if target_id in seen:
            raise ValueError(f"Popularity {stage} target is already seen for user {user_id}")
        raw_position = global_position[target_id]
        seen_before = sum(global_position[item_id] < raw_position for item_id in seen)
        target_ranks.append(raw_position - seen_before)
        covered.append(target_id in counts)

    ranks = np.asarray(target_ranks, dtype=np.int64)
    coverage = np.asarray(covered, dtype=bool)
    rows: list[Dict[str, Any]] = []
    for candidate_k in sorted({int(value) for value in candidate_ks}):
        if candidate_k <= 0:
            raise ValueError("candidate K values must be positive")
        hits = ranks <= candidate_k
        discounted = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
        reciprocal = np.where(hits, 1.0 / ranks, 0.0)
        rows.append(
            {
                "backend": "temporal_train_popularity-v1",
                "stage": stage,
                "candidate_k": candidate_k,
                "users": int(len(ranks)),
                "catalog_size": int(len(catalog)),
                "model_target_coverage": float(coverage.mean()),
                "candidate_recall": float(hits.mean()),
                "conditional_candidate_recall": (
                    float(hits[coverage].mean()) if coverage.any() else float("nan")
                ),
                "candidate_ndcg": float(discounted.mean()),
                "candidate_mrr": float(reciprocal.mean()),
                "mean_target_full_rank": float(ranks.mean()),
                "median_target_full_rank": float(np.median(ranks)),
            }
        )
    return rows


def evaluate_temporal_popularity(
    split_path: Path | str,
    *,
    candidate_ks: Sequence[int] = (10, 50, 100, 200, 500),
) -> pd.DataFrame:
    """Evaluate popularity using only history visible at each decision time."""

    split = pd.read_parquet(split_path).copy()
    required = {"user_id", "item_id", "split"}
    if not required.issubset(split.columns):
        raise ValueError(f"Protocol split is missing columns: {sorted(required - set(split.columns))}")
    split["user_id"] = split["user_id"].astype(str)
    split["item_id"] = split["item_id"].astype(str)
    if not {"train", "valid", "test"}.issubset(set(split["split"])):
        raise ValueError("Protocol split must contain train, valid, and test rows")
    rows = _stage_rows(
        split,
        stage="valid",
        visible_splits=("train",),
        candidate_ks=candidate_ks,
    )
    rows.extend(
        _stage_rows(
            split,
            stage="test",
            visible_splits=("train", "valid"),
            candidate_ks=candidate_ks,
        )
    )
    return pd.DataFrame(rows)
