"""Validation-only calibration for dataset-adapted slate hard constraints."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from copa.constraints import SlateConstraintRegistry
from copa.core import SlateConstraintSpec

from .artifacts import CandidateArtifactManifest, sha256_file


DEFAULT_ALPHA_GRID = tuple(round(0.20 + 0.05 * index, 2) for index in range(27))
DEFAULT_BRAND_CAP_GRID = (1, 2, 3, 4)
DEFAULT_CATEGORY_GRID = tuple(range(2, 11))
TARGET_INTERVALS = {
    "loose": (0.90, 0.98),
    "medium": (0.70, 0.85),
    "tight": (0.45, 0.65),
}


def calibrate_slate_policies(
    validation_candidate_path: Path | str,
    split_path: Path | str,
    item_catalog_path: Path | str,
    output_dir: Path | str,
    *,
    dataset_name: str,
    user_ids: Iterable[str] | None = None,
    candidate_k: int = 100,
    top_k: int = 10,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    brand_cap_grid: Sequence[int] = DEFAULT_BRAND_CAP_GRID,
    category_grid: Sequence[int] = DEFAULT_CATEGORY_GRID,
    solver_time_limit_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Scan the declared validation grid and freeze monotone strength policies."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_users = (
        {str(value) for value in user_ids} if user_ids is not None else None
    )
    read_kwargs: Dict[str, Any] = {
        "columns": ["user_id", "item_id", "retrieval_rank"]
    }
    if selected_users:
        # Electronics validation export contains every retained user (16M+
        # rows).  Predicate pushdown avoids materializing it merely to retain
        # the fixed 500-user downstream cohort.
        read_kwargs["filters"] = [("user_id", "in", sorted(selected_users))]
    candidates = pd.read_parquet(validation_candidate_path, **read_kwargs)
    candidates["user_id"] = candidates["user_id"].astype(str)
    candidates["item_id"] = candidates["item_id"].astype(str)
    candidates = candidates.sort_values(
        ["user_id", "retrieval_rank", "item_id"], kind="mergesort"
    )
    if selected_users is not None:
        candidates = candidates[candidates["user_id"].isin(selected_users)]
    candidates = candidates.groupby("user_id", sort=False).head(int(candidate_k))
    if candidates.empty:
        raise ValueError("Slate calibration has no validation candidates")

    split = pd.read_parquet(split_path)
    split["user_id"] = split["user_id"].astype(str)
    split["item_id"] = split["item_id"].astype(str)
    train = split[split["split"] == "train"]
    catalog = pd.read_parquet(item_catalog_path).copy()
    catalog["item_id"] = catalog["item_id"].astype(str)
    catalog = catalog.drop_duplicates("item_id", keep="last")
    if "price_filled" not in catalog or "brand_id" not in catalog:
        raise ValueError("Calibration requires price_filled and brand_id metadata")
    prices = pd.to_numeric(catalog["price_filled"], errors="coerce")
    global_price = float(prices.median())
    catalog["price_filled"] = prices.fillna(global_price)
    train_prices = train[["user_id", "item_id"]].merge(
        catalog[["item_id", "price_filled"]], on="item_id", how="left"
    )
    medians = (
        train_prices.groupby("user_id")["price_filled"].median().fillna(global_price)
    )
    merged = candidates.merge(catalog, on="item_id", how="left", validate="many_to_one")
    category_attribute = next(
        (
            name
            for name in ("main_category", "category")
            if name in merged.columns and merged[name].notna().any()
        ),
        None,
    )
    electronics = "electronic" in dataset_name.casefold()
    if electronics and category_attribute is None:
        raise ValueError("Electronics calibration requires executable category metadata")

    registry = SlateConstraintRegistry()
    rows: list[Dict[str, Any]] = []
    grouped = {
        str(user_id): group.reset_index(drop=True)
        for user_id, group in merged.groupby("user_id", sort=True)
    }
    for brand_cap, category_min in itertools.product(
        brand_cap_grid, category_grid if electronics else (0,)
    ):
        thresholds: list[float] = []
        unknown = shortage = 0
        solver_costs: list[float] = []
        for user_id, group in grouped.items():
            median = float(medians.get(user_id, global_price))
            item_feasible = group[
                pd.to_numeric(group["price_filled"], errors="coerce") <= 1.2 * median
            ].copy()
            if len(item_feasible) < top_k:
                shortage += 1
                thresholds.append(float("inf"))
                continue
            structural_policy = {
                "brand_cap": int(brand_cap),
                **(
                    {
                        "category_distinct_min": int(category_min),
                        "category_attribute": str(category_attribute),
                    }
                    if electronics
                    else {}
                ),
            }
            specs = _structural_specs(structural_policy)
            objective_scores = {
                str(row["item_id"]): -float(row["price_filled"])
                for row in item_feasible.to_dict("records")
            }
            solved = registry.solve(
                item_feasible,
                specs,
                top_k,
                objective_scores=objective_scores,
                time_limit_seconds=solver_time_limit_seconds,
            )
            solver_costs.append(float(solved.runtime_seconds))
            if solved.status == "optimal":
                minimum_price = -float(solved.objective_value or 0.0)
                thresholds.append(minimum_price / max(top_k * median, 1e-12))
            elif solved.status != "infeasible":
                unknown += 1
                thresholds.append(float("nan"))
            else:
                thresholds.append(float("inf"))
        total = len(grouped)
        threshold_array = np.asarray(thresholds, dtype=float)
        for alpha in alpha_grid:
            feasible = int(np.count_nonzero(threshold_array <= float(alpha) + 1e-12))
            rows.append(
                {
                    "total_budget_alpha": float(alpha),
                    "brand_cap": int(brand_cap),
                    **(
                        {
                            "category_distinct_min": int(category_min),
                            "category_attribute": str(category_attribute),
                        }
                        if electronics
                        else {}
                    ),
                    "users": total,
                    "full_slate_feasible_rate": feasible / max(1, total),
                    "solver_unknown_rate": unknown / max(1, total),
                    "item_pool_shortage_rate": shortage / max(1, total),
                    "minimum_alpha_median": (
                        float(np.median(threshold_array[np.isfinite(threshold_array)]))
                        if np.isfinite(threshold_array).any()
                        else None
                    ),
                    "solver_seconds_mean": float(np.mean(solver_costs)) if solver_costs else 0.0,
                    "solver_seconds_p95": (
                        float(np.quantile(solver_costs, 0.95)) if solver_costs else 0.0
                    ),
                }
            )
    search = pd.DataFrame(rows)
    search.to_csv(output_dir / "slate_calibration_search.csv", index=False)
    selected, selection_error = _select_monotone_policies(
        search, electronics=electronics
    )
    payload: Dict[str, Any] = {
        "dataset": dataset_name,
        "protocol": "validation_only_frozen_train",
        "retriever": "sasrec",
        "model_seed": 42,
        "candidate_k": int(candidate_k),
        "top_k": int(top_k),
        "users": len(grouped),
        "category_boundary": (
            "category coverage enabled"
            if electronics
            else "category coverage omitted because Beauty category metadata is degenerate"
        ),
        "target_intervals": TARGET_INTERVALS,
        "search_grid": {
            "total_budget_alpha": [float(value) for value in alpha_grid],
            "brand_cap": [int(value) for value in brand_cap_grid],
            "category_distinct_min": (
                [int(value) for value in category_grid] if electronics else []
            ),
        },
        "source_hashes": {
            "validation_candidates": sha256_file(validation_candidate_path),
            "protocol_split": sha256_file(split_path),
            "item_catalog": sha256_file(item_catalog_path),
            "calibration_users": hashlib.sha256(
                json.dumps(sorted(grouped), separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "policies": selected,
        "status": "success" if selection_error is None else "failed",
        "failure_reason": selection_error,
        "calibration_algorithm": "minimum_price_threshold_v2",
    }
    payload["calibration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_dir / "slate_calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _policy_specs(
    policy: Mapping[str, Any], train_median: float, top_k: int
) -> list[SlateConstraintSpec]:
    specs = [
        SlateConstraintSpec(
            "slate_total_budget",
            "aggregate_sum",
            "price_filled",
            "<=",
            float(policy["total_budget_alpha"]) * top_k * train_median,
        ),
        SlateConstraintSpec(
            "slate_brand_cap",
            "per_group_count",
            "brand_id",
            "<=",
            int(policy["brand_cap"]),
        ),
    ]
    if "category_distinct_min" in policy:
        specs.append(
            SlateConstraintSpec(
                "slate_category_coverage",
                "distinct_count",
                str(policy["category_attribute"]),
                ">=",
                int(policy["category_distinct_min"]),
            )
        )
    return specs


def _structural_specs(policy: Mapping[str, Any]) -> list[SlateConstraintSpec]:
    specs = [
        SlateConstraintSpec(
            "slate_brand_cap",
            "per_group_count",
            "brand_id",
            "<=",
            int(policy["brand_cap"]),
        )
    ]
    if "category_distinct_min" in policy:
        specs.append(
            SlateConstraintSpec(
                "slate_category_coverage",
                "distinct_count",
                str(policy["category_attribute"]),
                ">=",
                int(policy["category_distinct_min"]),
            )
        )
    return specs


def _select_monotone_policies(
    search: pd.DataFrame, *, electronics: bool
) -> tuple[Dict[str, Dict[str, Any]], str | None]:
    known = search[search["solver_unknown_rate"] == 0.0]
    records = known.to_dict("records")
    centers = {
        name: (bounds[0] + bounds[1]) / 2 for name, bounds in TARGET_INTERVALS.items()
    }

    def monotone(loose: Mapping[str, Any], medium: Mapping[str, Any], tight: Mapping[str, Any]) -> bool:
        if not (
            loose["total_budget_alpha"] >= medium["total_budget_alpha"] >= tight["total_budget_alpha"]
            and loose["brand_cap"] >= medium["brand_cap"] >= tight["brand_cap"]
        ):
            return False
        return not electronics or (
            loose["category_distinct_min"]
            <= medium["category_distinct_min"]
            <= tight["category_distinct_min"]
        )

    eligible = {
        name: [
            row
            for row in records
            if bounds[0] <= row["full_slate_feasible_rate"] <= bounds[1]
        ]
        for name, bounds in TARGET_INTERVALS.items()
    }
    missing = [name for name, values in eligible.items() if not values]
    if missing:
        return {}, f"No calibration policies reached target intervals: {missing}"

    chosen = None
    best_distance = float("inf")
    best_tie_breaker: tuple[Any, ...] | None = None
    for triple in itertools.product(
        eligible["loose"], eligible["medium"], eligible["tight"]
    ):
        if not monotone(triple[0], triple[1], triple[2]):
            continue
        policy_keys = ["total_budget_alpha", "brand_cap"]
        if electronics:
            policy_keys.append("category_distinct_min")
        if len({tuple(row[key] for key in policy_keys) for row in triple}) != 3:
            continue
        distance = sum(
            abs(triple[index]["full_slate_feasible_rate"] - centers[name])
            for index, name in enumerate(("loose", "medium", "tight"))
        )
        tie_breaker = tuple(
            value
            for row in triple
            for value in (
                float(row["total_budget_alpha"]),
                int(row["brand_cap"]),
                int(row.get("category_distinct_min", 0)),
            )
        )
        if distance < best_distance or (
            np.isclose(distance, best_distance)
            and (best_tie_breaker is None or tie_breaker < best_tie_breaker)
        ):
            chosen = triple
            best_distance = distance
            best_tie_breaker = tie_breaker
    if chosen is None:
        return {}, "Calibration grid has no strict monotone loose/medium/tight triple"
    output: Dict[str, Dict[str, Any]] = {}
    for name, row in zip(("loose", "medium", "tight"), chosen):
        lower, upper = TARGET_INTERVALS[name]
        policy_keys = ["total_budget_alpha", "brand_cap"]
        if electronics:
            policy_keys.extend(["category_distinct_min", "category_attribute"])
        output[name] = {
            **{key: row[key] for key in policy_keys},
            "validation_feasible_rate": row["full_slate_feasible_rate"],
            "within_target_interval": lower
            <= row["full_slate_feasible_rate"]
            <= upper,
        }
    return output, None


def attach_calibration_hash(
    manifest_paths: Iterable[Path | str], calibration_sha256: str
) -> None:
    """Bind frozen calibration to existing immutable candidate/checkpoint files."""

    for path_value in manifest_paths:
        path = Path(path_value)
        manifest = CandidateArtifactManifest.read(path)
        hashes = dict(manifest.source_hashes)
        hashes["constraint_calibration"] = str(calibration_sha256)
        replace(manifest, source_hashes=hashes).write(path)
