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
    target_counts = targets.groupby("user_id")["item_id"].size()
    if stage == "valid" and not (target_counts == 1).all():
        raise ValueError("Popularity validation requires one positive per user")
    if stage == "test" and not (target_counts == 2).all():
        raise ValueError("Popularity test requires two positives per user")

    catalog = sorted(visible["item_id"].astype(str).unique())
    counts = visible["item_id"].astype(str).value_counts().to_dict()
    ranked_catalog = sorted(catalog, key=lambda item_id: (-int(counts.get(item_id, 0)), item_id))
    global_position = {item_id: index for index, item_id in enumerate(ranked_catalog, start=1)}
    histories = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in visible.groupby("user_id")
    }

    user_ranks: list[list[float]] = []
    user_coverage: list[list[bool]] = []
    for user_id, group in targets.groupby("user_id", sort=True):
        user_id = str(user_id)
        seen = histories.get(user_id, set())
        ranks: list[float] = []
        covered: list[bool] = []
        for target_id in group["item_id"].astype(str):
            if target_id in seen:
                raise ValueError(
                    f"Popularity {stage} target is already seen for user {user_id}"
                )
            is_covered = target_id in global_position
            covered.append(is_covered)
            if not is_covered:
                ranks.append(float("inf"))
                continue
            raw_position = global_position[target_id]
            seen_before = sum(
                global_position[item_id] < raw_position
                for item_id in seen
                if item_id in global_position
            )
            ranks.append(float(raw_position - seen_before))
        user_ranks.append(ranks)
        user_coverage.append(covered)
    rows: list[Dict[str, Any]] = []
    for candidate_k in sorted({int(value) for value in candidate_ks}):
        if candidate_k <= 0:
            raise ValueError("candidate K values must be positive")
        recalls = []
        ndcgs = []
        mrrs = []
        coverage_rates = []
        finite_ranks = []
        for ranks, coverage in zip(user_ranks, user_coverage):
            hits = [rank <= candidate_k for rank in ranks]
            recalls.append(sum(hits) / len(ranks))
            hit_ranks = sorted(rank for rank in ranks if rank <= candidate_k)
            dcg = sum(1.0 / np.log2(rank + 1) for rank in hit_ranks)
            ideal = sum(
                1.0 / np.log2(position + 2)
                for position in range(min(len(ranks), candidate_k))
            )
            ndcgs.append(dcg / ideal if ideal else 0.0)
            mrrs.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)
            coverage_rates.append(sum(coverage) / len(coverage))
            finite_ranks.extend(rank for rank in ranks if np.isfinite(rank))
        rows.append(
            {
                "backend": "temporal_train_popularity-v1",
                "stage": stage,
                "candidate_k": candidate_k,
                "users": int(len(user_ranks)),
                "catalog_size": int(len(catalog)),
                "model_target_coverage": float(np.mean(coverage_rates)),
                "candidate_recall": float(np.mean(recalls)),
                "conditional_candidate_recall": (
                    float(
                        np.mean(
                            [
                                sum(
                                    rank <= candidate_k
                                    for rank, covered in zip(ranks, coverage)
                                    if covered
                                )
                                / sum(coverage)
                                for ranks, coverage in zip(user_ranks, user_coverage)
                                if any(coverage)
                            ]
                        )
                    )
                    if any(any(values) for values in user_coverage)
                    else float("nan")
                ),
                "candidate_ndcg": float(np.mean(ndcgs)),
                "candidate_mrr": float(np.mean(mrrs)),
                "mean_target_full_rank": (
                    float(np.mean(finite_ranks)) if finite_ranks else float("nan")
                ),
                "median_target_full_rank": (
                    float(np.median(finite_ranks)) if finite_ranks else float("nan")
                ),
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
            visible_splits=("train",),
            candidate_ks=candidate_ks,
        )
    )
    return pd.DataFrame(rows)
