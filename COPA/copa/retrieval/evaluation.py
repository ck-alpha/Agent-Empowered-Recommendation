"""Leakage-safe recall and COPA evaluation over versioned candidate artifacts."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from copa.core import CandidateRecord, ConstraintSpec
from copa.data import UserCase
from copa.experiments.run_core import (
    _execute,
    build_optimization,
    strict_constraint_evaluation,
)
from copa.metrics import hypervolume_shared, ndcg_at_k, recall_at_k, spacing_shared

from .artifacts import PrecomputedCandidateStore, sha256_file
from .candidate_analysis import (
    candidate_quality_metrics,
    controlled_hit_users,
    intervene_candidate_pool,
    oracle_candidate_pool,
)


FORMAL_END_TO_END_METHODS = (
    "feasible_relevance",
    "feasible_weighted_ga",
    "copa",
)
LOSS_DECOMPOSITION_KEYS = (
    "candidate_recall",
    "target_in_feasible_domain",
    "retrieval_loss",
    "constraint_filter_loss",
)


@dataclass(frozen=True)
class RetrievalProtocolContext:
    split: pd.DataFrame
    item_catalog: pd.DataFrame
    popularity: Mapping[str, float]
    histories: Mapping[str, frozenset[str]]
    budgets: Mapping[str, float]
    targets: Mapping[str, str]
    catalog_size: int

    @classmethod
    def load(
        cls, split_path: Path | str, item_catalog_path: Path | str
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
        if test["user_id"].duplicated().any():
            raise ValueError("Temporal protocol must contain one test target per user")
        targets = test.set_index("user_id")["item_id"].to_dict()

        visible = split[split["split"].isin(["train", "valid"])]
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
        for user_id, group in visible_prices.groupby("user_id"):
            user_prices = pd.to_numeric(group["price_filled"], errors="coerce").dropna()
            center = float(user_prices.median()) if len(user_prices) else global_price
            budgets[str(user_id)] = 1.2 * center
        return cls(
            split=split,
            item_catalog=catalog,
            popularity=popularity,
            histories=histories,
            budgets=budgets,
            targets={str(key): str(value) for key, value in targets.items()},
            catalog_size=int(split["item_id"].nunique()),
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
                "Budget estimated only from train+validation history.",
            ),
            ConstraintSpec(
                "seen_items",
                "exclusion",
                "item_id",
                "not_in",
                sorted(self.histories[user_id]),
                "Exclude all interactions visible before the test event.",
            ),
        ]

    def user_case(
        self,
        user_id: str,
        candidates: Sequence[CandidateRecord],
        *,
        condition: str,
    ) -> UserCase:
        user_id = str(user_id)
        return UserCase(
            user_id=user_id,
            candidates=list(candidates),
            constraints=self.constraints(user_id),
            relevant_items=[self.targets[user_id]],
            context={
                "source": "precomputed_retrieval_artifact",
                "condition": condition,
                "budget_high": float(self.budgets[user_id]),
            },
        )


def _validate_protocol_boundary(
    store: PrecomputedCandidateStore, context: RetrievalProtocolContext
) -> None:
    for user_id in store.user_ids:
        if user_id not in context.targets:
            raise ValueError(f"Artifact user {user_id} is absent from the protocol test split")
        target_row = store.target_row(user_id)
        if str(target_row["target_item_id"]) != context.targets[user_id]:
            raise ValueError(f"Artifact target disagrees with protocol split for {user_id}")
        store.load(
            user_id,
            len(store.ranked_frame(user_id)),
            seen_items=context.histories[user_id],
        )


def _quality_row(
    *,
    store: PrecomputedCandidateStore,
    context: RetrievalProtocolContext,
    user_id: str,
    candidates: Sequence[CandidateRecord],
    condition: str,
    requested_k: int,
) -> Dict[str, Any]:
    target = store.target_row(user_id)
    quality = candidate_quality_metrics(
        candidates,
        [context.targets[user_id]],
        context.constraints(user_id),
        requested_k=requested_k,
        target_model_covered=bool(target["target_model_covered"]),
    )
    return {
        "dataset": store.manifest.dataset if store.manifest else "unknown",
        "retriever": store.manifest.retriever if store.manifest else "unknown",
        "backend": store.manifest.backend if store.manifest else "unknown",
        "model_seed": store.manifest.model_seed if store.manifest else -1,
        "user_id": user_id,
        "condition": condition,
        "candidate_k": int(requested_k),
        "catalog_size": context.catalog_size,
        "target_full_rank": int(target["target_full_rank"]),
        **quality,
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
            recommendation_hit = float(
                bool(
                    {str(item_id) for item_id in result.item_ids}
                    & {str(item_id) for item_id in case.relevant_items}
                )
            )
            strict = strict_constraint_evaluation(case, result.item_ids)
            violations = strict.pop("strict_violations")
            violation_counts = {
                constraint.id: sum(
                    violation["constraint_id"] == constraint.id
                    for violation in violations
                )
                for constraint in case.constraints
            }
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
                    "verified_feasible": float(result.verification.feasible),
                    "verifier_violation_count": len(result.verification.violations),
                    "recall_at_10": recall_at_k(result.item_ids, case.relevant_items, 10),
                    "ndcg_at_10": ndcg_at_k(result.item_ids, case.relevant_items, 10),
                    "recommendation_hit": recommendation_hit,
                    "ranking_loss": float(
                        float(retrieval_metadata.get("target_in_feasible_domain", 0.0))
                        > 0.0
                        and not bool(recommendation_hit)
                    ),
                    "runtime_seconds": runtime,
                    "optimizer_runtime_seconds": float(
                        result.diagnostics.get("runtime_seconds", runtime)
                    ),
                    "evaluations": int(result.diagnostics.get("evaluations", 1)),
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
) -> Dict[str, Any]:
    """Evaluate real candidate K and optional target-coverage interventions."""

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 0:
        raise ValueError("workers must be positive")
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
    context = RetrievalProtocolContext.load(split_path, item_catalog_path)
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
    maximum_available_k = int(store.manifest.candidate_k if store.manifest else max(candidate_ks))
    quality_rows: list[Dict[str, Any]] = []
    cases: list[tuple[str, UserCase, Dict[str, Any]]] = []
    coverage_items: Dict[tuple[str, int], set[str]] = {}

    for candidate_k in sorted({int(value) for value in candidate_ks}):
        if candidate_k <= 0:
            raise ValueError("candidate K values must be positive")
        for user_id in store.user_ids:
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
                        for key in LOSS_DECOMPOSITION_KEYS
                    },
                }
                cases.append(
                    (
                        "real",
                        context.user_case(user_id, candidates, condition="real"),
                        metadata,
                    )
                )

    if run_interventions:
        eligible = [
            user_id
            for user_id in store.user_ids
            if bool(store.target_row(user_id)["target_model_covered"])
        ]
        if not eligible:
            raise RuntimeError("No model-covered test targets are eligible for interventions")
        hit_sets = {
            float(level): controlled_hit_users(eligible, float(level), intervention_seed)
            for level in controlled_levels
        }
        for user_id in eligible:
            reservoir = store.load(
                user_id,
                maximum_available_k,
                seen_items=context.histories[user_id],
            )
            target = store.target_record(user_id)
            oracle = oracle_candidate_pool(
                reservoir,
                target,
                end_to_end_candidate_k,
                seen_items=context.histories[user_id],
            )
            condition_pools = [("oracle", oracle)]
            for level, hit_users in hit_sets.items():
                condition_pools.append(
                    (
                        f"controlled_{int(round(level * 100)):03d}",
                        intervene_candidate_pool(
                            reservoir,
                            target,
                            end_to_end_candidate_k,
                            include_target=user_id in hit_users,
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
                        for key in LOSS_DECOMPOSITION_KEYS
                    },
                }
                cases.append(
                    (
                        condition,
                        context.user_case(user_id, pool, condition=condition),
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
        }
        legacy_signature = hashlib.sha256(
            json.dumps(legacy_signature_payload, sort_keys=True).encode()
        ).hexdigest()
        signature_payload = {
            "signature_schema": 2,
            **legacy_signature_payload,
            "controlled_levels": sorted(map(float, controlled_levels)),
            "intervention_seed": int(intervention_seed),
            "run_interventions": bool(run_interventions),
            "run_real_end_to_end": bool(run_real_end_to_end),
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode()
        ).hexdigest()
        checkpoint_dir = output_dir / "task_results"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for condition, case, metadata in cases:
            task_id = hashlib.sha256(
                f"{condition}:{case.user_id}".encode()
            ).hexdigest()[:20]
            checkpoint_path = checkpoint_dir / f"{task_id}.json"
            if resume and checkpoint_path.exists():
                try:
                    cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                    if _cached_task_matches(
                        cached,
                        signature=signature,
                        legacy_signature=legacy_signature,
                        retrieval_metadata=metadata,
                    ):
                        if cached.get("signature") != signature:
                            cached["signature"] = signature
                            _atomic_json(checkpoint_path, cached)
                        e2e_rows.extend(cached["rows"])
                        fronts.extend(cached["fronts"])
                        continue
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
            tasks.append(
                {
                    "condition": condition,
                    "case": case,
                    "metadata": metadata,
                    "optimization_config": optimization_config,
                    "optimizer_seeds": list(map(int, optimizer_seeds)),
                    "methods": list(methods),
                    "trace_dir": str(output_dir / "traces"),
                    "weights": [1 / 3, 1 / 3, 1 / 3],
                    "hv_sample_power": int(hv_sample_power),
                    "signature": signature,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
        def record_completed(payload: Mapping[str, Any], task: Mapping[str, Any]) -> None:
            _atomic_json(Path(task["checkpoint_path"]), payload)
            e2e_rows.extend(payload["rows"])
            fronts.extend(payload["fronts"])

        if workers <= 1:
            for task in tasks:
                record_completed(_run_case_task(task), task)
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = {executor.submit(_run_case_task, task): task for task in tasks}
                for future in as_completed(futures):
                    record_completed(future.result(), futures[future])
        e2e_frame = pd.DataFrame(e2e_rows)
        e2e_frame.to_csv(output_dir / "end_to_end_metrics.csv", index=False)
        numeric = e2e_frame.select_dtypes(include=[np.number]).columns.tolist()
        e2e_summary = e2e_frame.groupby(
            ["dataset", "retriever", "condition", "method"]
        )[numeric].agg(["mean", "std"])
        e2e_summary.columns = [
            f"{column}_{statistic}" for column, statistic in e2e_summary.columns
        ]
        e2e_summary.reset_index().to_csv(output_dir / "end_to_end_summary.csv", index=False)
        with (output_dir / "pareto_fronts.jsonl").open("w", encoding="utf-8") as handle:
            for record in fronts:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    result = {
        "manifest": str(manifest_path),
        "dataset": store.manifest.dataset,
        "retriever": store.manifest.retriever,
        "users": len(store.user_ids),
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
    }
    (output_dir / "evaluation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
