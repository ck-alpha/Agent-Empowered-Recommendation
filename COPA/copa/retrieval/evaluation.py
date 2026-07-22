"""Leakage-safe recall and COPA evaluation over versioned candidate artifacts."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry
from copa.core import CandidateRecord, ConstraintSpec, SlateConstraintSpec
from copa.data import UserCase
from copa.experiments.run_core import (
    _execute,
    build_optimization,
    strict_constraint_evaluation,
)
from copa.metrics import (
    constraint_aware_ndcg_at_k,
    hypervolume_shared,
    loss_waterfall,
    ndcg_at_k,
    recall_at_k,
    spacing_shared,
)

from .artifacts import PrecomputedCandidateStore, sha256_file
from .candidate_analysis import (
    candidate_quality_metrics,
    controlled_hit_pairs,
    controlled_hit_users,
    intervene_candidate_pool,
    intervene_positive_pool,
    oracle_candidate_pool,
)


FORMAL_END_TO_END_METHODS = (
    "unconstrained_relevance",
    "item_filtered_relevance",
    "feasible_relevance",
    "feasible_weighted_ga",
    "copa",
)
OPTIMIZER_KERNEL_VERSION = 2
TASK_SIGNATURE_SCHEMA = 5
LOSS_DECOMPOSITION_KEYS = (
    "candidate_recall",
    "target_in_feasible_domain",
    "retrieval_loss",
    "constraint_filter_loss",
)
OPPORTUNITY_METADATA_KEYS = (
    "positive_count",
    "item_feasible_positive_count",
    "model_covered_positive_count",
    "retrieved_positive_count",
    "opportunity_positive_count",
    "item_feasible_positive_ids",
    "request_eligible",
    "opportunity_eligible",
    "opportunity_status",
    "opportunity_solver_seconds",
    "item_constraint_loss",
    "model_catalog_loss",
    "retrieval_loss",
    "slate_constraint_loss",
    "target_in_feasible_domain",
    "constraint_filter_loss",
)


@dataclass(frozen=True)
class RetrievalProtocolContext:
    split: pd.DataFrame
    item_catalog: pd.DataFrame
    popularity: Mapping[str, float]
    histories: Mapping[str, frozenset[str]]
    budgets: Mapping[str, float]
    targets: Mapping[str, tuple[str, ...]]
    train_catalog: frozenset[str]
    train_median_prices: Mapping[str, float]
    slate_policy: Mapping[str, Any]
    catalog_size: int

    @classmethod
    def load(
        cls,
        split_path: Path | str,
        item_catalog_path: Path | str,
        *,
        slate_policy: Mapping[str, Any] | None = None,
    ) -> "RetrievalProtocolContext":
        split = pd.read_parquet(split_path)
        required_split = {"user_id", "item_id", "timestamp", "split"}
        if missing := required_split - set(split.columns):
            raise ValueError(f"Protocol split missing columns: {sorted(missing)}")
        split = split.copy()
        split["user_id"] = split["user_id"].astype(str)
        split["item_id"] = split["item_id"].astype(str)
        if not set(split["split"]).issubset({"train", "valid", "test"}):
            raise ValueError("Protocol split contains an unknown partition")
        test = split[split["split"] == "test"]
        test_counts = test.groupby("user_id")["item_id"].size()
        valid_counts = split[split["split"] == "valid"].groupby("user_id")["item_id"].size()
        if not (test_counts == 2).all() or not (valid_counts == 1).all():
            raise ValueError(
                "Temporal protocol requires one validation and two test positives per user"
            )
        targets = {
            str(user_id): tuple(group.sort_values(
                ["timestamp", "item_id"], kind="mergesort"
            )["item_id"].astype(str))
            for user_id, group in test.groupby("user_id", sort=True)
        }

        visible = split[split["split"] == "train"]
        histories = {
            str(user_id): frozenset(map(str, values))
            for user_id, values in visible.groupby("user_id")["item_id"]
        }
        counts = visible["item_id"].value_counts().astype(float)
        log_counts = np.log1p(counts)
        maximum = float(log_counts.max()) if len(log_counts) else 1.0
        popularity = {
            str(item_id): float(value / max(maximum, 1e-12))
            for item_id, value in log_counts.items()
        }

        catalog = pd.read_parquet(item_catalog_path).copy()
        if "item_id" not in catalog or "price_filled" not in catalog:
            raise ValueError("Item catalog requires item_id and price_filled")
        catalog["item_id"] = catalog["item_id"].astype(str)
        catalog = catalog.drop_duplicates("item_id", keep="last")
        prices = pd.to_numeric(catalog["price_filled"], errors="coerce")
        global_price = float(prices.median())
        if not np.isfinite(global_price):
            raise ValueError("Item catalog has no finite price")
        catalog["price_filled"] = prices.fillna(global_price)
        visible_prices = visible[["user_id", "item_id"]].merge(
            catalog[["item_id", "price_filled"]], on="item_id", how="left"
        )
        budgets: Dict[str, float] = {}
        train_median_prices: Dict[str, float] = {}
        for user_id, group in visible_prices.groupby("user_id"):
            user_prices = pd.to_numeric(group["price_filled"], errors="coerce").dropna()
            center = float(user_prices.median()) if len(user_prices) else global_price
            train_median_prices[str(user_id)] = center
            budgets[str(user_id)] = 1.2 * center
        train_catalog = frozenset(visible["item_id"].astype(str))
        return cls(
            split=split,
            item_catalog=catalog,
            popularity=popularity,
            histories=histories,
            budgets=budgets,
            targets=targets,
            train_catalog=train_catalog,
            train_median_prices=train_median_prices,
            slate_policy=dict(slate_policy or {}),
            catalog_size=len(train_catalog),
        )

    def constraints(self, user_id: str) -> list[ConstraintSpec]:
        user_id = str(user_id)
        return [
            ConstraintSpec(
                "budget_high",
                "numeric",
                "price_filled",
                "<=",
                float(self.budgets[user_id]),
                "Budget estimated only from frozen train history.",
            ),
            ConstraintSpec(
                "seen_items",
                "exclusion",
                "item_id",
                "not_in",
                sorted(self.histories[user_id]),
                "Exclude frozen train interactions; validation is not request input.",
            ),
        ]

    def slate_constraints(self, user_id: str, requested_k: int) -> list[SlateConstraintSpec]:
        user_id = str(user_id)
        specs: list[SlateConstraintSpec] = []
        if "total_budget_alpha" in self.slate_policy:
            alpha = float(self.slate_policy["total_budget_alpha"])
            specs.append(
                SlateConstraintSpec(
                    "slate_total_budget",
                    "aggregate_sum",
                    "price_filled",
                    "<=",
                    alpha * int(requested_k) * self.train_median_prices[user_id],
                    description="Total price calibrated from frozen train median.",
                )
            )
        if "brand_cap" in self.slate_policy:
            specs.append(
                SlateConstraintSpec(
                    "slate_brand_cap",
                    "per_group_count",
                    str(self.slate_policy.get("brand_attribute", "brand_id")),
                    "<=",
                    int(self.slate_policy["brand_cap"]),
                    description="Maximum items from any one brand.",
                )
            )
        if "category_distinct_min" in self.slate_policy:
            specs.append(
                SlateConstraintSpec(
                    "slate_category_coverage",
                    "distinct_count",
                    str(self.slate_policy.get("category_attribute", "category")),
                    ">=",
                    int(self.slate_policy["category_distinct_min"]),
                    description="Minimum distinct category coverage.",
                )
            )
        return specs

    def user_case(
        self,
        user_id: str,
        candidates: Sequence[CandidateRecord],
        *,
        condition: str,
        requested_k: int = 10,
    ) -> UserCase:
        user_id = str(user_id)
        return UserCase(
            user_id=user_id,
            candidates=list(candidates),
            constraints=self.constraints(user_id),
            relevant_items=list(self.targets[user_id]),
            context={
                "source": "precomputed_retrieval_artifact",
                "condition": condition,
                "budget_high": float(self.budgets[user_id]),
            },
            slate_constraints=self.slate_constraints(user_id, requested_k),
        )


def _validate_protocol_boundary(
    store: PrecomputedCandidateStore, context: RetrievalProtocolContext
) -> None:
    for user_id in store.user_ids:
        if user_id not in context.targets:
            raise ValueError(f"Artifact user {user_id} is absent from the protocol test split")
        target_rows = store.target_rows(user_id)
        artifact_targets = tuple(
            str(row["target_item_id"])
            for row in sorted(target_rows, key=lambda row: int(row["target_order"]))
        )
        if artifact_targets != context.targets[user_id]:
            raise ValueError(f"Artifact target disagrees with protocol split for {user_id}")
        store.load(
            user_id,
            len(store.ranked_frame(user_id)),
            seen_items=context.histories[user_id],
        )


def _opportunity_state(
    *,
    store: PrecomputedCandidateStore,
    context: RetrievalProtocolContext,
    user_id: str,
    candidates: Sequence[CandidateRecord],
    requested_k: int,
    solver_time_limit_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Compute P/I/M/C/U and the exact candidate-stage loss waterfall."""

    positives = set(context.targets[user_id])
    item_registry = ConstraintRegistry()
    item_specs = context.constraints(user_id)
    catalog = context.item_catalog.set_index("item_id", drop=False)
    item_feasible: set[str] = set()
    for item_id in positives:
        if item_id not in catalog.index:
            raise ValueError(f"Test positive {item_id} is missing from the item catalog")
        row = catalog.loc[item_id].to_dict()
        if all(item_registry.evaluate(spec, row).satisfied for spec in item_specs):
            item_feasible.add(item_id)

    target_rows = store.target_rows(user_id)
    covered_ids = {
        str(row["target_item_id"])
        for row in target_rows
        if bool(row["target_model_covered"])
    }
    model_covered = item_feasible & covered_ids
    candidate_ids = {str(candidate.item_id) for candidate in candidates}
    retrieved = model_covered & candidate_ids
    feasible_candidates = [
        candidate
        for candidate in candidates
        if all(item_registry.evaluate(spec, candidate.to_row()).satisfied for spec in item_specs)
    ]
    opportunity_status = "optimal"
    opportunity_items: list[str] = []
    solver_seconds = 0.0
    if len(feasible_candidates) < requested_k:
        opportunity_status = "item_pool_shortage"
    else:
        frame = pd.DataFrame([candidate.to_row() for candidate in feasible_candidates])
        slate_specs = context.slate_constraints(user_id, requested_k)
        if not slate_specs:
            opportunity_items = sorted(
                frame["item_id"].astype(str),
                key=lambda item_id: (item_id not in retrieved, item_id),
            )[:requested_k]
        else:
            solved = SlateConstraintRegistry().solve(
                frame,
                slate_specs,
                requested_k,
                objective_scores={item_id: 1.0 for item_id in retrieved},
                time_limit_seconds=float(solver_time_limit_seconds),
            )
            opportunity_status = solved.status
            solver_seconds = float(solved.runtime_seconds)
            if solved.status == "optimal":
                opportunity_items = list(solved.item_ids)
    opportunity = len(retrieved & set(opportunity_items))
    candidate_waterfall = loss_waterfall(
        positives=len(positives),
        item_feasible=len(item_feasible),
        model_covered=len(model_covered),
        retrieved=len(retrieved),
        opportunity=opportunity,
        hits=opportunity,
    )
    return {
        "positive_count": len(positives),
        "item_feasible_positive_count": len(item_feasible),
        "model_covered_positive_count": len(model_covered),
        "retrieved_positive_count": len(retrieved),
        "opportunity_positive_count": opportunity,
        "item_feasible_positive_ids": sorted(item_feasible),
        "opportunity_status": opportunity_status,
        "opportunity_solver_seconds": solver_seconds,
        "request_eligible": float(bool(item_feasible)),
        "opportunity_eligible": float(opportunity > 0),
        **{
            key: value
            for key, value in candidate_waterfall.items()
            if key != "ranking_selection_loss"
        },
        # Backward-compatible names used by pre-v2 checkpoints.
        "target_in_feasible_domain": opportunity / max(1, len(positives)),
        "constraint_filter_loss": candidate_waterfall["item_constraint_loss"],
    }


def _quality_row(
    *,
    store: PrecomputedCandidateStore,
    context: RetrievalProtocolContext,
    user_id: str,
    candidates: Sequence[CandidateRecord],
    condition: str,
    requested_k: int,
    opportunity_solver_time_limit_seconds: float = 5.0,
) -> Dict[str, Any]:
    targets = store.target_rows(user_id)
    quality = candidate_quality_metrics(
        candidates,
        context.targets[user_id],
        context.constraints(user_id),
        requested_k=requested_k,
        target_model_covered=[
            bool(row["target_model_covered"])
            for row in sorted(targets, key=lambda row: int(row["target_order"]))
        ],
    )
    opportunity = _opportunity_state(
        store=store,
        context=context,
        user_id=user_id,
        candidates=candidates,
        requested_k=10,
        solver_time_limit_seconds=opportunity_solver_time_limit_seconds,
    )
    covered_ranks = [
        int(row["target_full_rank"])
        for row in targets
        if bool(row["target_model_covered"])
    ]
    return {
        "dataset": store.manifest.dataset if store.manifest else "unknown",
        "retriever": store.manifest.retriever if store.manifest else "unknown",
        "backend": store.manifest.backend if store.manifest else "unknown",
        "model_seed": store.manifest.model_seed if store.manifest else -1,
        "user_id": user_id,
        "condition": condition,
        "candidate_k": int(requested_k),
        "catalog_size": context.catalog_size,
        "target_full_rank": min(covered_ranks) if covered_ranks else np.nan,
        "target_full_ranks": json.dumps(covered_ranks),
        **quality,
        **opportunity,
    }


def _run_methods(
    case: UserCase,
    *,
    condition: str,
    retrieval_metadata: Mapping[str, Any],
    optimization_config: Mapping[str, Any],
    seeds: Sequence[int],
    methods: Sequence[str],
    trace_dir: Path,
    weights: Sequence[float],
    hv_sample_power: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    rows: list[Dict[str, Any]] = []
    fronts: list[Dict[str, Any]] = []
    for seed in seeds:
        optimization = build_optimization(optimization_config, int(seed))
        for method in methods:
            safe_user = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in case.user_id
            )
            trace_path = (
                trace_dir
                / condition
                / f"core_retrieval_{condition}_{method}_s{optimization.seed}_{safe_user}.jsonl"
            )
            trace_path.unlink(missing_ok=True)
            started = perf_counter()
            result = _execute(
                case,
                f"retrieval_{condition}",
                method,
                optimization,
                trace_dir / condition,
                weights,
            )
            runtime = perf_counter() - started
            hit_count = len(
                {str(item_id) for item_id in result.item_ids}
                & {str(item_id) for item_id in case.relevant_items}
            )
            recommendation_hit = float(bool(hit_count))
            positive_count = int(
                retrieval_metadata.get("positive_count", len(case.relevant_items))
            )
            if "opportunity_positive_count" in retrieval_metadata:
                item_feasible_count = int(
                    retrieval_metadata["item_feasible_positive_count"]
                )
                model_covered_count = int(
                    retrieval_metadata["model_covered_positive_count"]
                )
                retrieved_count = int(retrieval_metadata["retrieved_positive_count"])
                opportunity = int(retrieval_metadata["opportunity_positive_count"])
            else:
                # Read-only compatibility for v1 single-positive checkpoints.
                retrieved_count = int(
                    round(
                        float(retrieval_metadata.get("candidate_recall", 0.0))
                        * positive_count
                    )
                )
                opportunity = int(
                    round(
                        float(
                            retrieval_metadata.get("target_in_feasible_domain", 0.0)
                        )
                        * positive_count
                    )
                )
                model_covered_count = max(retrieved_count, opportunity)
                item_feasible_count = model_covered_count
            waterfall = loss_waterfall(
                positives=positive_count,
                item_feasible=item_feasible_count,
                model_covered=model_covered_count,
                retrieved=retrieved_count,
                opportunity=opportunity,
                hits=hit_count,
            )
            strict = strict_constraint_evaluation(case, result.item_ids)
            violations = strict.pop("strict_violations")
            strict = {
                key: value
                for key, value in strict.items()
                if key
                not in {
                    "candidate_recall",
                    "target_in_feasible_domain",
                    "recommendation_hit",
                    "retrieval_loss",
                    "constraint_filter_loss",
                    "ranking_loss",
                }
            }
            all_constraint_ids = [
                constraint.id
                for constraint in (*case.constraints, *case.slate_constraints)
            ]
            violation_counts = {
                constraint_id: sum(
                    violation["constraint_id"] == constraint_id
                    for violation in violations
                )
                for constraint_id in all_constraint_ids
            }
            violation_magnitudes = {
                constraint_id: max(
                    (
                        float(violation.get("violation_magnitude", 1.0))
                        for violation in violations
                        if violation["constraint_id"] == constraint_id
                    ),
                    default=0.0,
                )
                for constraint_id in all_constraint_ids
            }
            item_feasible_ids = retrieval_metadata.get(
                "item_feasible_positive_ids", case.relevant_items
            )
            first_positive = case.relevant_items[:1]
            rows.append(
                {
                    **dict(retrieval_metadata),
                    "user_id": case.user_id,
                    "condition": condition,
                    "method": method,
                    "optimizer_seed": int(seed),
                    "candidate_count": len(case.candidates),
                    "requested_k": optimization.top_k,
                    "actual_k": len(result.item_ids),
                    "shortage": float(len(result.item_ids) < optimization.top_k),
                    "result_status": result.status,
                    "delivered_slate": float(bool(result.item_ids)),
                    "delivered_slate_verifier_pass": (
                        float(result.verification.feasible)
                        if result.item_ids
                        else None
                    ),
                    "verified_feasible": float(result.verification.feasible),
                    "verifier_violation_count": len(result.verification.violations),
                    "recall_at_10": recall_at_k(result.item_ids, case.relevant_items, 10),
                    "ndcg_at_10": ndcg_at_k(result.item_ids, case.relevant_items, 10),
                    # Paper ablation only: reproduce the legacy protocol that
                    # retained the first test positive and silently discarded
                    # the second.  The multi-positive columns above remain the
                    # authoritative metrics.
                    "legacy_single_positive_recall_at_10": recall_at_k(
                        result.item_ids, first_positive, 10
                    ),
                    "legacy_single_positive_ndcg_at_10": ndcg_at_k(
                        result.item_ids, first_positive, 10
                    ),
                    "recommendation_hit": recommendation_hit,
                    "hit_count": hit_count,
                    "feasible_recall_at_10": recall_at_k(
                        result.item_ids, item_feasible_ids, 10
                    ),
                    "feasible_ndcg_at_10": ndcg_at_k(
                        result.item_ids, item_feasible_ids, 10
                    ),
                    "opportunity_recall": (
                        hit_count / opportunity if opportunity else None
                    ),
                    "constraint_aware_ndcg_at_10": constraint_aware_ndcg_at_k(
                        result.item_ids, case.relevant_items, 10, opportunity
                    ),
                    "ranking_loss": waterfall["ranking_selection_loss"],
                    **waterfall,
                    "runtime_seconds": runtime,
                    "optimizer_runtime_seconds": float(
                        result.diagnostics.get("runtime_seconds", runtime)
                    ),
                    "preflight_solver_seconds": float(
                        result.diagnostics.get("preflight_runtime_seconds", 0.0)
                    ),
                    "repair_fallbacks": int(
                        result.diagnostics.get("repair_fallbacks", 0)
                    ),
                    "initialization_fallbacks": int(
                        result.diagnostics.get("initialization_fallbacks", 0)
                    ),
                    "unique_population_rate": float(
                        result.diagnostics.get("unique_population_rate", 1.0)
                    ),
                    "evaluations": int(result.diagnostics.get("evaluations", 1)),
                    "optimizer_kernel_version": int(
                        result.diagnostics.get(
                            "optimizer_kernel_version",
                            optimization.optimizer_kernel_version,
                        )
                    ),
                    "optimizer_compile_seconds": float(
                        result.diagnostics.get("compile_seconds", 0.0)
                    ),
                    "objective_evaluation_seconds": float(
                        result.diagnostics.get("objective_evaluation_seconds", 0.0)
                    ),
                    "sorting_seconds": float(
                        result.diagnostics.get("sorting_seconds", 0.0)
                    ),
                    "constraint_check_seconds": float(
                        result.diagnostics.get("constraint_check_seconds", 0.0)
                    ),
                    "constraint_checks": int(
                        result.diagnostics.get("constraint_checks", 0)
                    ),
                    "swap_attempts": int(
                        result.diagnostics.get("swap_attempts", 0)
                    ),
                    "accepted_swaps": int(
                        result.diagnostics.get("accepted_swaps", 0)
                    ),
                    "variation_fallbacks": int(
                        result.diagnostics.get("variation_fallbacks", 0)
                    ),
                    "shared_hypervolume": hypervolume_shared(
                        result.pareto_front,
                        lower=[0.0, 0.0, 0.0],
                        upper=[1.0, 1.0, 1.0],
                        sample_power=hv_sample_power,
                        seed=int(seed),
                    ),
                    "shared_spacing": spacing_shared(
                        result.pareto_front,
                        lower=[0.0, 0.0, 0.0],
                        upper=[1.0, 1.0, 1.0],
                    ),
                    **{f"objective_{key}": value for key, value in result.objective_values.items()},
                    **strict,
                    **{
                        f"violation_{constraint_id}_count": count
                        for constraint_id, count in violation_counts.items()
                    },
                    **{
                        f"violation_{constraint_id}_rate": count
                        / max(1, len(result.item_ids))
                        for constraint_id, count in violation_counts.items()
                    },
                    **{
                        f"violation_{constraint_id}_magnitude": magnitude
                        for constraint_id, magnitude in violation_magnitudes.items()
                    },
                }
            )
            fronts.append(
                {
                    **dict(retrieval_metadata),
                    "user_id": case.user_id,
                    "condition": condition,
                    "method": method,
                    "optimizer_seed": int(seed),
                    "selected_items": result.item_ids,
                    "result_status": result.status,
                    "strict_violations": violations,
                    "front": [solution.to_dict() for solution in result.pareto_front],
                }
            )
    return rows, fronts


def _run_case_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    rows, fronts = _run_methods(
        task["case"],
        condition=str(task["condition"]),
        retrieval_metadata=task["metadata"],
        optimization_config=task["optimization_config"],
        seeds=task["optimizer_seeds"],
        methods=task["methods"],
        trace_dir=Path(task["trace_dir"]),
        weights=task["weights"],
        hv_sample_power=int(task["hv_sample_power"]),
    )
    return {"signature": task["signature"], "rows": rows, "fronts": fronts}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cached_task_matches(
    cached: Mapping[str, Any],
    *,
    signature: str,
    legacy_signature: str,
    retrieval_metadata: Mapping[str, Any],
) -> bool:
    """Validate a current checkpoint or safely migrate the pre-v2 signature.

    Signature v1 already bound the manifest, optimizer, methods, and HV
    settings, but omitted the intervention seed and levels.  Its per-row loss
    decomposition is sufficient to prove that every current user/condition has
    the same target hit/feasibility state; given the bound manifest, that state
    deterministically identifies the real/oracle/controlled candidate pool.
    """

    observed_signature = cached.get("signature")
    rows = cached.get("rows")
    fronts = cached.get("fronts")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(fronts, list)
        or not fronts
    ):
        return False
    if observed_signature == signature:
        return True
    if observed_signature != legacy_signature:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        for key in LOSS_DECOMPOSITION_KEYS:
            if key not in row or key not in retrieval_metadata:
                return False
            try:
                matches = np.isclose(
                    float(row[key]),
                    float(retrieval_metadata[key]),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            except (TypeError, ValueError):
                return False
            if not matches:
                return False
    return True


def run_artifact_evaluation(
    manifest_path: Path | str,
    split_path: Path | str,
    item_catalog_path: Path | str,
    output_dir: Path | str,
    *,
    candidate_ks: Sequence[int] = (50, 100, 200, 500),
    end_to_end_candidate_k: int = 100,
    optimizer_seeds: Sequence[int] = (42, 43, 44),
    population_size: int = 100,
    generations: int = 50,
    methods: Sequence[str] = FORMAL_END_TO_END_METHODS,
    controlled_levels: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    intervention_seed: int = 42,
    run_interventions: bool = False,
    run_end_to_end: bool = True,
    run_real_end_to_end: bool = True,
    hv_sample_power: int = 12,
    workers: int = 1,
    resume: bool = False,
    slate_policy: Mapping[str, Any] | None = None,
    evaluation_users: Iterable[str] | None = None,
    use_milp_seed: bool = True,
    use_slate_feasible_operators: bool = True,
    slate_preflight_time_limit_seconds: float = 2.0,
    opportunity_solver_time_limit_seconds: float = 5.0,
    optimizer_time_limit_seconds: float = 120.0,
    optimizer_kernel_version: int = OPTIMIZER_KERNEL_VERSION,
    calibration_sha256: str = "",
) -> Dict[str, Any]:
    """Evaluate real candidate K and optional target-coverage interventions."""

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if slate_preflight_time_limit_seconds <= 0 or opportunity_solver_time_limit_seconds <= 0:
        raise ValueError("Slate solver time limits must be positive")
    if optimizer_time_limit_seconds <= 0:
        raise ValueError("optimizer_time_limit_seconds must be positive")
    if optimizer_kernel_version != OPTIMIZER_KERNEL_VERSION:
        raise ValueError(
            f"optimizer_kernel_version must be {OPTIMIZER_KERNEL_VERSION}"
        )
    unknown_methods = set(methods) - set(FORMAL_END_TO_END_METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported end-to-end methods: {sorted(unknown_methods)}")
    if run_end_to_end and int(end_to_end_candidate_k) not in {
        int(value) for value in candidate_ks
    }:
        raise ValueError("end-to-end candidate K must also appear in candidate_ks")
    if run_end_to_end and not run_real_end_to_end and not run_interventions:
        raise ValueError(
            "Disabling real end-to-end evaluation requires intervention conditions"
        )
    context = RetrievalProtocolContext.load(
        split_path, item_catalog_path, slate_policy=slate_policy
    )
    store = PrecomputedCandidateStore(
        manifest_path.parent / json.loads(manifest_path.read_text(encoding="utf-8"))["candidate_file"],
        manifest_path=manifest_path,
        item_catalog=context.item_catalog,
        popularity=context.popularity,
    )
    for source_key, source_path in (
        ("protocol_split", Path(split_path)),
        ("items", Path(item_catalog_path)),
    ):
        expected_hash = store.manifest.source_hashes.get(source_key)
        if expected_hash and sha256_file(source_path) != expected_hash:
            raise ValueError(
                f"Evaluation {source_key} SHA-256 does not match the candidate manifest"
            )
    _validate_protocol_boundary(store, context)
    requested_users = (
        {str(value) for value in evaluation_users}
        if evaluation_users is not None
        else set(store.user_ids)
    )
    unknown_users = requested_users - set(store.user_ids)
    if unknown_users:
        raise ValueError(f"Evaluation users are absent from the artifact: {sorted(unknown_users)[:5]}")
    active_user_ids = sorted(requested_users)
    maximum_available_k = int(store.manifest.candidate_k if store.manifest else max(candidate_ks))
    quality_rows: list[Dict[str, Any]] = []
    cases: list[tuple[str, UserCase, Dict[str, Any]]] = []
    coverage_items: Dict[tuple[str, int], set[str]] = {}

    for candidate_k in sorted({int(value) for value in candidate_ks}):
        if candidate_k <= 0:
            raise ValueError("candidate K values must be positive")
        for user_id in active_user_ids:
            candidates = store.load(
                user_id,
                min(candidate_k, maximum_available_k),
                seen_items=context.histories[user_id],
            )
            quality_row = _quality_row(
                store=store,
                context=context,
                user_id=user_id,
                candidates=candidates,
                condition="real",
                requested_k=candidate_k,
                opportunity_solver_time_limit_seconds=(
                    opportunity_solver_time_limit_seconds
                ),
            )
            quality_rows.append(quality_row)
            coverage_items.setdefault(("real", candidate_k), set()).update(
                candidate.item_id for candidate in candidates
            )
            if candidate_k == end_to_end_candidate_k and run_real_end_to_end:
                metadata = {
                    "dataset": store.manifest.dataset,
                    "retriever": store.manifest.retriever,
                    "backend": store.manifest.backend,
                    "model_seed": store.manifest.model_seed,
                    "candidate_k": candidate_k,
                    **{
                        key: quality_row[key]
                        for key in OPPORTUNITY_METADATA_KEYS
                    },
                }
                cases.append(
                    (
                        "real",
                        context.user_case(
                            user_id, candidates, condition="real", requested_k=10
                        ),
                        metadata,
                    )
                )

    if run_interventions:
        target_records_by_user = {
            user_id: store.scored_target_records(user_id) for user_id in active_user_ids
        }
        eligible_pairs = {
            (user_id, str(target.item_id))
            for user_id, targets in target_records_by_user.items()
            for target in targets
        }
        if not eligible_pairs:
            raise RuntimeError("No model-covered test targets are eligible for interventions")
        hit_sets = {
            float(level): controlled_hit_pairs(
                eligible_pairs, float(level), intervention_seed
            )
            for level in controlled_levels
        }
        for user_id in active_user_ids:
            reservoir = store.load(
                user_id,
                maximum_available_k,
                seen_items=context.histories[user_id],
            )
            targets = target_records_by_user[user_id]
            oracle = intervene_positive_pool(
                reservoir,
                targets,
                end_to_end_candidate_k,
                included_target_ids=[target.item_id for target in targets],
                seen_items=context.histories[user_id],
            )
            condition_pools = [("oracle", oracle)]
            for level, hit_pairs in hit_sets.items():
                condition_pools.append(
                    (
                        f"controlled_{int(round(level * 100)):03d}",
                        intervene_positive_pool(
                            reservoir,
                            targets,
                            end_to_end_candidate_k,
                            included_target_ids=[
                                item_id
                                for pair_user, item_id in hit_pairs
                                if pair_user == user_id
                            ],
                            seen_items=context.histories[user_id],
                        ),
                    )
                )
            for condition, pool in condition_pools:
                quality_row = _quality_row(
                    store=store,
                    context=context,
                    user_id=user_id,
                    candidates=pool,
                    condition=condition,
                    requested_k=end_to_end_candidate_k,
                    opportunity_solver_time_limit_seconds=(
                        opportunity_solver_time_limit_seconds
                    ),
                )
                quality_rows.append(quality_row)
                coverage_items.setdefault(
                    (condition, end_to_end_candidate_k), set()
                ).update(candidate.item_id for candidate in pool)
                metadata = {
                    "dataset": store.manifest.dataset,
                    "retriever": store.manifest.retriever,
                    "backend": store.manifest.backend,
                    "model_seed": store.manifest.model_seed,
                    "candidate_k": end_to_end_candidate_k,
                    **{
                        key: quality_row[key]
                        for key in OPPORTUNITY_METADATA_KEYS
                    },
                }
                cases.append(
                    (
                        condition,
                        context.user_case(
                            user_id, pool, condition=condition, requested_k=10
                        ),
                        metadata,
                    )
                )

    quality_frame = pd.DataFrame(quality_rows)
    quality_frame.to_csv(output_dir / "candidate_metrics.csv", index=False)
    numeric_quality = [
        column
        for column in quality_frame.select_dtypes(include=[np.number]).columns
        if column != "candidate_k"
    ]
    candidate_summary = quality_frame.groupby(
        ["dataset", "retriever", "condition", "candidate_k"], as_index=False
    )[numeric_quality].mean()
    coverage_rows = []
    for (condition, candidate_k), item_ids in coverage_items.items():
        coverage_rows.append(
            {
                "condition": condition,
                "candidate_k": int(candidate_k),
                "catalog_coverage": len(item_ids) / max(1, context.catalog_size),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    candidate_summary = candidate_summary.merge(
        coverage, on=["condition", "candidate_k"], how="left"
    )
    candidate_summary.to_csv(output_dir / "candidate_summary.csv", index=False)

    e2e_rows: list[Dict[str, Any]] = []
    fronts: list[Dict[str, Any]] = []
    if run_end_to_end:
        optimization_config = {
            "optimization": {
                "top_k": 10,
                "population_size": int(population_size),
                "generations": int(generations),
                "crossover_rate": 0.9,
                "mutation_rate": 0.15,
                "tournament_size": 2,
                "use_milp_seed": bool(use_milp_seed),
                "use_slate_feasible_operators": bool(
                    use_slate_feasible_operators
                ),
                "slate_solver_time_limit_seconds": float(
                    slate_preflight_time_limit_seconds
                ),
                "optimizer_time_limit_seconds": float(
                    optimizer_time_limit_seconds
                ),
                "optimizer_kernel_version": int(optimizer_kernel_version),
            }
        }
        legacy_signature_payload = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "candidate_k": end_to_end_candidate_k,
            "optimizer_seeds": list(map(int, optimizer_seeds)),
            "population_size": population_size,
            "generations": generations,
            "methods": list(methods),
            "hv_sample_power": hv_sample_power,
            "use_milp_seed": bool(use_milp_seed),
            "use_slate_feasible_operators": bool(
                use_slate_feasible_operators
            ),
            "slate_preflight_time_limit_seconds": float(
                slate_preflight_time_limit_seconds
            ),
            "opportunity_solver_time_limit_seconds": float(
                opportunity_solver_time_limit_seconds
            ),
        }
        legacy_signature = hashlib.sha256(
            json.dumps(legacy_signature_payload, sort_keys=True).encode()
        ).hexdigest()
        signature_payload = {
            "signature_schema": TASK_SIGNATURE_SCHEMA,
            **legacy_signature_payload,
            "controlled_levels": sorted(map(float, controlled_levels)),
            "intervention_seed": int(intervention_seed),
            "run_interventions": bool(run_interventions),
            "run_real_end_to_end": bool(run_real_end_to_end),
            "slate_policy": dict(slate_policy or {}),
            "use_milp_seed": bool(use_milp_seed),
            "use_slate_feasible_operators": bool(
                use_slate_feasible_operators
            ),
            "optimizer_time_limit_seconds": float(
                optimizer_time_limit_seconds
            ),
            "optimizer_kernel_version": int(optimizer_kernel_version),
            "calibration_sha256": str(calibration_sha256),
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode()
        ).hexdigest()
        checkpoint_dir = output_dir / "task_results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        cached_task_count = 0
        for condition, case, metadata in cases:
            for optimizer_seed in map(int, optimizer_seeds):
                for method in methods:
                    atomic_signature = hashlib.sha256(
                        json.dumps(
                            {
                                "base": signature,
                                "condition": condition,
                                "user_id": case.user_id,
                                "method": method,
                                "optimizer_seed": optimizer_seed,
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                    task_id = hashlib.sha256(
                        (
                            f"{condition}:{case.user_id}:{method}:"
                            f"{optimizer_seed}:k{optimizer_kernel_version}"
                        ).encode()
                    ).hexdigest()[:24]
                    checkpoint_path = checkpoint_dir / f"{task_id}.json"
                    if resume and checkpoint_path.exists():
                        try:
                            cached = json.loads(
                                checkpoint_path.read_text(encoding="utf-8")
                            )
                            if _cached_task_matches(
                                cached,
                                signature=atomic_signature,
                                legacy_signature="",
                                retrieval_metadata=metadata,
                            ):
                                e2e_rows.extend(cached["rows"])
                                fronts.extend(cached["fronts"])
                                cached_task_count += 1
                                continue
                        except (OSError, json.JSONDecodeError, KeyError):
                            pass
                    tasks.append(
                        {
                            "condition": condition,
                            "case": case,
                            "metadata": metadata,
                            "optimization_config": optimization_config,
                            "optimizer_seeds": [optimizer_seed],
                            "methods": [method],
                            "trace_dir": str(output_dir / "traces"),
                            "weights": [1 / 3, 1 / 3, 1 / 3],
                            "hv_sample_power": int(hv_sample_power),
                            "signature": atomic_signature,
                            "checkpoint_path": str(checkpoint_path),
                            "method": method,
                            "optimizer_seed": optimizer_seed,
                            "user_id": case.user_id,
                        }
                    )
        progress_path = output_dir / "progress.json"
        progress_started = perf_counter()
        completed_task_count = cached_task_count
        failed_tasks: list[Dict[str, Any]] = []
        future_handles: list[Any] = []
        sequential_running = 0

        def write_progress() -> None:
            elapsed = perf_counter() - progress_started
            newly_completed = completed_task_count - cached_task_count
            remaining = len(tasks) - newly_completed - len(failed_tasks)
            rate = newly_completed / elapsed if elapsed > 0 else 0.0
            if future_handles:
                running = sum(
                    future.running() and not future.done()
                    for future in future_handles
                )
                pending = sum(
                    not future.running() and not future.done()
                    for future in future_handles
                )
            else:
                running = sequential_running
                pending = max(0, remaining - running)
            _atomic_json(
                progress_path,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "optimizer_kernel_version": optimizer_kernel_version,
                    "total_tasks": cached_task_count + len(tasks),
                    "cached_tasks": cached_task_count,
                    "completed_tasks": completed_task_count,
                    "failed_tasks": len(failed_tasks),
                    "running_tasks": int(running),
                    "pending_tasks": int(pending),
                    "running_or_pending_tasks": max(0, remaining),
                    "elapsed_seconds": elapsed,
                    "throughput_tasks_per_second": rate,
                    "estimated_remaining_seconds": (
                        remaining / rate if rate > 0 else None
                    ),
                    "failures": failed_tasks[-20:],
                },
            )

        write_progress()
        def record_completed(payload: Mapping[str, Any], task: Mapping[str, Any]) -> None:
            nonlocal completed_task_count
            _atomic_json(Path(task["checkpoint_path"]), payload)
            e2e_rows.extend(payload["rows"])
            fronts.extend(payload["fronts"])
            completed_task_count += 1
            write_progress()

        if workers <= 1:
            for task in tasks:
                sequential_running = 1
                write_progress()
                try:
                    record_completed(_run_case_task(task), task)
                except Exception as exc:
                    failed_tasks.append(
                        {
                            "user_id": task["user_id"],
                            "condition": task["condition"],
                            "method": task["method"],
                            "optimizer_seed": task["optimizer_seed"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    sequential_running = 0
                    write_progress()
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {executor.submit(_run_case_task, task): task for task in tasks}
                future_handles.extend(futures)
                write_progress()
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record_completed(future.result(), task)
                    except Exception as exc:
                        failed_tasks.append(
                            {
                                "user_id": task["user_id"],
                                "condition": task["condition"],
                                "method": task["method"],
                                "optimizer_seed": task["optimizer_seed"],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        write_progress()
        if failed_tasks:
            raise RuntimeError(
                f"{len(failed_tasks)} atomic optimizer tasks failed; see {progress_path}"
            )
        e2e_frame = pd.DataFrame(e2e_rows)
        e2e_frame.to_csv(output_dir / "end_to_end_metrics.csv", index=False)
        numeric = e2e_frame.select_dtypes(include=[np.number]).columns.tolist()
        e2e_summary = e2e_frame.groupby(
            ["dataset", "retriever", "condition", "method"]
        )[numeric].agg(["mean", "std"])
        e2e_summary.columns = [
            f"{column}_{statistic}" for column, statistic in e2e_summary.columns
        ]
        e2e_summary = e2e_summary.reset_index()
        group_columns = ["dataset", "retriever", "condition", "method"]
        executable_hv = (
            e2e_frame[e2e_frame["opportunity_status"] == "optimal"]
            .groupby(group_columns)["shared_hypervolume"]
            .mean()
            .rename("shared_hypervolume_executable_mean")
            .reset_index()
        )
        e2e_summary = e2e_summary.merge(executable_hv, on=group_columns, how="left")
        e2e_summary.to_csv(output_dir / "end_to_end_summary.csv", index=False)
        latency_summary = (
            e2e_frame.groupby(group_columns)["runtime_seconds"]
            .agg(
                latency_p50=lambda values: float(values.quantile(0.50)),
                latency_p95=lambda values: float(values.quantile(0.95)),
                latency_mean="mean",
            )
            .reset_index()
        )
        latency_summary.to_csv(output_dir / "latency_summary.csv", index=False)
        paired = _paired_user_inference(e2e_frame, bootstrap_samples=10_000, seed=42)
        paired.to_csv(output_dir / "paired_user_inference.csv", index=False)
        with (output_dir / "pareto_fronts.jsonl").open("w", encoding="utf-8") as handle:
            for record in fronts:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "manifest": str(manifest_path),
        "dataset": store.manifest.dataset,
        "retriever": store.manifest.retriever,
        "users": len(active_user_ids),
        "artifact_users": len(store.user_ids),
        "candidate_metric_rows": len(quality_frame),
        "end_to_end_rows": len(e2e_rows),
        "candidate_ks": sorted({int(value) for value in candidate_ks}),
        "optimizer_seeds": list(map(int, optimizer_seeds)),
        "population_size": int(population_size),
        "generations": int(generations),
        "methods": list(methods),
        "controlled_levels": list(map(float, controlled_levels)),
        "intervention_seed": int(intervention_seed),
        "hv_sample_power": int(hv_sample_power),
        "run_interventions": bool(run_interventions),
        "run_end_to_end": bool(run_end_to_end),
        "run_real_end_to_end": bool(run_real_end_to_end),
        "workers": int(workers),
        "resume": bool(resume),
        "slate_policy": dict(slate_policy or {}),
        "use_milp_seed": bool(use_milp_seed),
        "use_slate_feasible_operators": bool(use_slate_feasible_operators),
        "slate_preflight_time_limit_seconds": float(
            slate_preflight_time_limit_seconds
        ),
        "opportunity_solver_time_limit_seconds": float(
            opportunity_solver_time_limit_seconds
        ),
        "optimizer_time_limit_seconds": float(optimizer_time_limit_seconds),
        "optimizer_kernel_version": int(optimizer_kernel_version),
        "calibration_sha256": str(calibration_sha256),
    }
    (output_dir / "evaluation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _paired_user_inference(
    frame: pd.DataFrame, *, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    """Aggregate optimizer seeds per user, then run paired user inference."""

    if frame.empty or "copa" not in set(frame["method"]):
        return pd.DataFrame()
    metrics = [
        column
        for column in (
            "recall_at_10",
            "ndcg_at_10",
            "feasible_recall_at_10",
            "constraint_aware_ndcg_at_10",
            "opportunity_recall",
            "shared_hypervolume",
            "runtime_seconds",
        )
        if column in frame
    ]
    group_keys = ["dataset", "retriever", "condition"]
    user_aggregated = (
        frame.groupby([*group_keys, "method", "user_id"], as_index=False)[metrics]
        .mean()
    )
    rng = np.random.default_rng(seed)
    records: list[Dict[str, Any]] = []
    for group_values, group in user_aggregated.groupby(group_keys, sort=True):
        copa = group[group["method"] == "copa"].set_index("user_id")
        for baseline in sorted(set(group["method"]) - {"copa"}):
            other = group[group["method"] == baseline].set_index("user_id")
            common = sorted(set(copa.index) & set(other.index))
            for metric in metrics:
                differences = (
                    copa.loc[common, metric].to_numpy(dtype=float)
                    - other.loc[common, metric].to_numpy(dtype=float)
                )
                differences = differences[np.isfinite(differences)]
                if not len(differences):
                    continue
                samples = rng.choice(
                    differences,
                    size=(int(bootstrap_samples), len(differences)),
                    replace=True,
                ).mean(axis=1)
                signs = rng.choice(
                    np.asarray([-1.0, 1.0]),
                    size=(int(bootstrap_samples), len(differences)),
                    replace=True,
                )
                null_means = (signs * differences).mean(axis=1)
                observed = float(differences.mean())
                p_value = float(
                    (1 + np.sum(np.abs(null_means) >= abs(observed)))
                    / (bootstrap_samples + 1)
                )
                records.append(
                    {
                        **dict(zip(group_keys, group_values)),
                        "method": "copa",
                        "baseline": baseline,
                        "metric": metric,
                        "users": len(differences),
                        "mean_paired_difference": observed,
                        "ci95_lower": float(np.quantile(samples, 0.025)),
                        "ci95_upper": float(np.quantile(samples, 0.975)),
                        "paired_randomization_p": p_value,
                        "bootstrap_samples": int(bootstrap_samples),
                        "inference_seed": int(seed),
                    }
                )
    output = pd.DataFrame(records)
    if output.empty:
        return output
    order = output["paired_randomization_p"].sort_values().index.tolist()
    adjusted: Dict[int, float] = {}
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(output.loc[index, "paired_randomization_p"]))
        running = max(running, value)
        adjusted[index] = running
    output["holm_adjusted_p"] = pd.Series(adjusted)
    return output
