"""Research-grade core Phase-1 experiment matrix for COPA."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import yaml

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry
from copa.core import ObjectiveSpec, OptimizationConfig, RecommendationRequest
from copa.data import AllBeautyAdapter, UserCase, build_synthetic_case
from copa.metrics import (
    front_metrics,
    hypervolume_shared,
    ndcg_at_k,
    recall_at_k,
    spacing_shared,
)
from copa.optimization import WeightedGeneticOptimizer
from copa.session import COPAExecutionSession


CORE_METHODS = {
    "A": ["unconstrained_relevance", "feasible_relevance"],
    "B": ["unconstrained_relevance", "unconstrained_weighted_ga", "unconstrained_copa"],
    "C": ["feasible_relevance", "feasible_weighted_ga", "copa"],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COPA core Phase-1 matrix")
    parser.add_argument("--config", default="COPA/configs/core_experiment.yaml")
    parser.add_argument("--suite", choices=["core"], default="core")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cohort-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "dataset" not in payload or "optimization" not in payload:
        raise ValueError("core config requires dataset and optimization sections")
    return payload


def objective_specs() -> list[ObjectiveSpec]:
    return [
        ObjectiveSpec("relevance", "maximize", "candidate"),
        ObjectiveSpec("diversity", "maximize", "slate", {"attribute": "brand_id"}),
        ObjectiveSpec("novelty", "maximize", "candidate", {"attribute": "popularity"}),
    ]


def build_optimization(config: Mapping[str, Any], seed: int) -> OptimizationConfig:
    payload = config["optimization"]
    return OptimizationConfig(
        top_k=int(payload.get("top_k", 10)),
        population_size=int(payload.get("population_size", 100)),
        generations=int(payload.get("generations", 50)),
        crossover_rate=float(payload.get("crossover_rate", 0.9)),
        mutation_rate=float(payload.get("mutation_rate", 0.15)),
        tournament_size=int(payload.get("tournament_size", 2)),
        seed=int(seed),
        selection_strategy="compromise",
        slate_solver_time_limit_seconds=float(
            payload.get("slate_solver_time_limit_seconds", 2.0)
        ),
        slate_repair_attempts=int(payload.get("slate_repair_attempts", 20)),
        use_milp_seed=bool(payload.get("use_milp_seed", True)),
        use_slate_feasible_operators=bool(
            payload.get("use_slate_feasible_operators", True)
        ),
        optimizer_time_limit_seconds=float(
            payload.get("optimizer_time_limit_seconds", 120.0)
        ),
        optimizer_kernel_version=int(payload.get("optimizer_kernel_version", 2)),
    )


def build_cohort(config: Mapping[str, Any], cohort_seed: int) -> list[UserCase]:
    dataset = config["dataset"]
    dataset_type = str(dataset.get("type", "beauty"))
    if dataset_type == "synthetic":
        return [
            build_synthetic_case(
                seed=cohort_seed,
                candidate_count=int(dataset.get("candidate_count", 30)),
            )
        ]
    if dataset_type != "beauty":
        raise ValueError(f"unsupported core dataset: {dataset_type}")
    adapter = AllBeautyAdapter(
        dataset.get("processed_dir", "data/processed"),
        dataset.get("prefix", "beauty_scenario1"),
    )
    cases = list(
        adapter.iter_user_cases(
            num_users=int(dataset.get("num_users", 100)),
            candidate_k=int(dataset.get("candidate_k", 100)),
            seed=cohort_seed,
            min_history=int(dataset.get("min_history", 2)),
        )
    )
    if len(cases) != int(dataset.get("num_users", 100)):
        raise RuntimeError(
            f"expected {dataset.get('num_users', 100)} cohort users, got {len(cases)}"
        )
    return cases


def _uses_constraints(method: str) -> bool:
    return (
        method.startswith("feasible_")
        or method == "copa"
        or method == "item_filtered_relevance"
    )


def _uses_slate_constraints(method: str) -> bool:
    return method.startswith("feasible_") or method == "copa"


def _strategy(method: str) -> str:
    if method.endswith("relevance"):
        return "feasible_topk"
    if method.endswith("weighted_ga"):
        return "fixed_weight_ga"
    return "pareto"


def _execute(
    case: UserCase,
    experiment: str,
    method: str,
    optimization: OptimizationConfig,
    trace_dir: Path,
    weights: Sequence[float],
):
    request = RecommendationRequest(
        case.user_id,
        case.candidates,
        case.constraints if _uses_constraints(method) else [],
        objective_specs(),
        optimization,
        case.context,
        case.slate_constraints if _uses_slate_constraints(method) else (),
    )
    run_id = f"core_{experiment}_{method}_s{optimization.seed}"
    session = COPAExecutionSession(request, trace_dir=trace_dir, run_id=run_id)
    strategy = _strategy(method)
    if strategy == "fixed_weight_ga":
        session.compute_objectives()
        if not session.preflight():
            session._empty_selection()
            session.verify()
            return session.result()
        try:
            selected, front, diagnostics = WeightedGeneticOptimizer(
                session.objectives, session.slate_constraints
            ).optimize(
                session.bus,
                request.objectives,
                request.optimization,
                request.context,
                weights=weights,
                slate_specs=request.slate_constraints,
                feasible_seed=(
                    session._preflight_result.item_ids
                    if session._preflight_result is not None
                    else None
                ),
            )
        except (RuntimeError, ValueError) as exc:
            session.status = "optimizer_failed"
            session.diagnostics["optimizer_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            session._empty_selection()
            session.verify()
            return session.result()
        session.selected = selected
        session.pareto_front = front
        session.diagnostics = {"strategy": strategy, **diagnostics}
        session.verify()
        return session.result()
    return session.execute(strategy)


def strict_constraint_evaluation(case: UserCase, selected_ids: Sequence[str]) -> Dict[str, Any]:
    registry = ConstraintRegistry()
    rows = {candidate.item_id: candidate.to_row() for candidate in case.candidates}
    strict_feasible_ids = []
    for candidate in case.candidates:
        if all(registry.evaluate(spec, rows[candidate.item_id]).satisfied for spec in case.constraints):
            strict_feasible_ids.append(candidate.item_id)

    violations = []
    for item_id in selected_ids:
        if item_id not in rows:
            violations.append({"item_id": item_id, "constraint_id": "unknown_item"})
            continue
        for spec in case.constraints:
            evaluated = registry.evaluate(spec, rows[item_id])
            if not evaluated.satisfied:
                violations.append(
                    {
                        "item_id": item_id,
                        "constraint_id": spec.id,
                        "attribute": spec.attribute,
                        "actual": evaluated.actual,
                    }
                )
    slate_registry = SlateConstraintRegistry()
    slate_frame = pd.DataFrame([candidate.to_row() for candidate in case.candidates])
    slate_violations = []
    if len(selected_ids) == len(set(selected_ids)) and all(
        item_id in rows for item_id in selected_ids
    ):
        for spec, evaluated in slate_registry.evaluate_all(
            case.slate_constraints, selected_ids, slate_frame
        ):
            if not evaluated.satisfied:
                slate_violations.append(
                    {
                        "constraint_id": spec.id,
                        "actual": evaluated.actual,
                        "violation_magnitude": evaluated.violation_magnitude,
                        "scope": "slate",
                    }
                )
    violations.extend(slate_violations)
    relevant = set(map(str, case.relevant_items))
    candidate_ids = set(rows)
    feasible_ids = set(strict_feasible_ids)
    target_in_candidates = bool(relevant & candidate_ids)
    target_in_feasible = bool(relevant & feasible_ids)
    hit = bool(relevant & set(map(str, selected_ids)))
    denominator = max(
        1,
        len(selected_ids) * len(case.constraints) + len(case.slate_constraints),
    )
    return {
        "strict_constraint_satisfaction_rate": float(not violations),
        "strict_violation_rate": len(violations) / denominator,
        "strict_violation_count": len(violations),
        "strict_feasible_candidate_count": len(strict_feasible_ids),
        "strict_candidate_filter_rate": 1.0 - len(strict_feasible_ids) / max(1, len(case.candidates)),
        "candidate_recall": float(target_in_candidates),
        "target_in_feasible_domain": float(target_in_feasible),
        "recommendation_hit": float(hit),
        "retrieval_loss": float(not target_in_candidates),
        "constraint_filter_loss": float(target_in_candidates and not target_in_feasible),
        "ranking_loss": float(target_in_feasible and not hit),
        "strict_violations": violations,
        "strict_item_violation_count": len(violations) - len(slate_violations),
        "strict_slate_violation_count": len(slate_violations),
    }


def _run_user_seed(task: Dict[str, Any]) -> Dict[str, Any]:
    case: UserCase = task["case"]
    seed = int(task["seed"])
    config = task["config"]
    output_dir = Path(task["output_dir"])
    optimization = build_optimization(config, seed)
    trace_dir = output_dir / "traces"
    weights = config.get("objectives", {}).get(
        "weighted_ga_weights", [1 / 3, 1 / 3, 1 / 3]
    )
    metrics_config = config.get("metrics", {})
    hv_power = int(metrics_config.get("hypervolume_sample_power", 14))
    bounds = metrics_config.get("hypervolume_bounds", {})
    lower = bounds.get("lower", [0.0, 0.0, 0.0])
    upper = bounds.get("upper", [1.0, 1.0, 1.0])
    rows = []
    fronts = []
    for experiment, methods in CORE_METHODS.items():
        for method in methods:
            tracemalloc.start()
            started = perf_counter()
            result = _execute(
                case, experiment, method, optimization, trace_dir, weights
            )
            wall_seconds = perf_counter() - started
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            strict = strict_constraint_evaluation(case, result.item_ids)
            if not result.verification.feasible:
                raise RuntimeError(
                    f"training verifier failed for {experiment}/{method}/{seed}/{case.user_id}"
                )
            legacy = front_metrics(result.pareto_front, sample_power=hv_power, seed=seed)
            row = {
                "experiment": experiment,
                "method": method,
                "dataset": str(config["dataset"].get("type", "beauty")),
                "cohort_seed": int(task["cohort_seed"]),
                "seed": seed,
                "user_id": case.user_id,
                "training_constraint_count": len(case.constraints) if _uses_constraints(method) else 0,
                "strict_evaluation_constraint_count": len(case.constraints),
                "requested_k": optimization.top_k,
                "actual_k": len(result.item_ids),
                "actual_k_ratio": len(result.item_ids) / optimization.top_k,
                "shortage": float(len(result.item_ids) < optimization.top_k),
                "recall_at_k": recall_at_k(result.item_ids, case.relevant_items, optimization.top_k),
                "ndcg_at_k": ndcg_at_k(result.item_ids, case.relevant_items, optimization.top_k),
                "verified_feasible": float(result.verification.feasible),
                "candidate_count": len(case.candidates),
                "runtime_seconds": wall_seconds,
                "optimizer_runtime_seconds": float(result.diagnostics.get("runtime_seconds", wall_seconds)),
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "evaluations": int(result.diagnostics.get("evaluations", 1)),
                "selection_strategy": str(result.diagnostics.get("selection_strategy", _strategy(method))),
                "trace_path": str(result.trace_path) if result.trace_path else "",
                **{f"objective_{key}": value for key, value in result.objective_values.items()},
                **legacy,
                "shared_hypervolume": hypervolume_shared(
                    result.pareto_front,
                    lower=lower,
                    upper=upper,
                    sample_power=hv_power,
                    seed=seed,
                ),
                "shared_spacing": spacing_shared(
                    result.pareto_front, lower=lower, upper=upper
                ),
                **{key: value for key, value in strict.items() if key != "strict_violations"},
            }
            rows.append(row)
            fronts.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "seed": seed,
                    "user_id": case.user_id,
                    "selected_items": result.item_ids,
                    "strict_violations": strict["strict_violations"],
                    "front": [solution.to_dict() for solution in result.pareto_front],
                }
            )
    return {"rows": rows, "fronts": fronts}


def _signature(config: Mapping[str, Any], cohort_seed: int) -> str:
    encoded = json.dumps(
        {"config": config, "cohort_seed": cohort_seed, "methods": CORE_METHODS},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _collect_generations(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in rows.to_dict("records"):
        trace_path = Path(row["trace_path"])
        if not trace_path.exists():
            continue
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("operation") != "generation":
                continue
            summary = event.get("input_summary", {})
            output = {
                "experiment": row["experiment"],
                "method": row["method"],
                "seed": row["seed"],
                "user_id": row["user_id"],
                "generation": summary.get("generation"),
                "pareto_size": summary.get("pareto_size"),
                "evaluations": summary.get("evaluations"),
                "feasible_candidates": summary.get("feasible_candidates"),
                "generation_duration_ms": event.get("duration_ms"),
                "best_scalar_fitness": summary.get("best_scalar_fitness"),
            }
            for objective, values in summary.get("objective_summary", {}).items():
                for statistic, value in values.items():
                    output[f"{objective}_{statistic}"] = value
            records.append(output)
    return pd.DataFrame(records)


def run_core(
    config: Dict[str, Any],
    output_dir: Path,
    *,
    workers: int,
    cohort_seed: int,
    resume: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "task_results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    signature = _signature(config, cohort_seed)
    cohort = build_cohort(config, cohort_seed)
    seeds = [int(value) for value in config.get("seeds", [42, 43, 44])]
    tasks = []
    for user_index, case in enumerate(cohort):
        for seed in seeds:
            path = checkpoint_dir / f"u{user_index:04d}_s{seed}.json"
            if resume and path.exists():
                try:
                    if json.loads(path.read_text(encoding="utf-8")).get("signature") == signature:
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            tasks.append(
                {
                    "user_index": user_index,
                    "case": case,
                    "seed": seed,
                    "config": config,
                    "cohort_seed": cohort_seed,
                    "output_dir": str(output_dir),
                    "checkpoint_path": str(path),
                }
            )
    total = len(cohort) * len(seeds)
    completed = total - len(tasks)
    print(json.dumps({"stage": "phase1", "completed_tasks": completed, "total_tasks": total}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_user_seed, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            payload = future.result()
            payload["signature"] = signature
            _atomic_json(Path(task["checkpoint_path"]), payload)
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": "phase1",
                        "completed_tasks": completed,
                        "total_tasks": total,
                        "user_index": task["user_index"],
                        "seed": task["seed"],
                    }
                ),
                flush=True,
            )

    all_rows = []
    all_fronts = []
    for path in sorted(checkpoint_dir.glob("u*_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            continue
        all_rows.extend(payload["rows"])
        all_fronts.extend(payload["fronts"])
    frame = pd.DataFrame(all_rows).sort_values(
        ["experiment", "method", "seed", "user_id"]
    )
    expected_rows = len(cohort) * len(seeds) * sum(map(len, CORE_METHODS.values()))
    keys = ["experiment", "method", "seed", "user_id"]
    if len(frame) != expected_rows or frame.duplicated(keys).any():
        raise RuntimeError(
            f"core row integrity failed: expected {expected_rows}, got {len(frame)}"
        )
    required = [
        "recall_at_k",
        "ndcg_at_k",
        "strict_constraint_satisfaction_rate",
        "strict_violation_rate",
        "objective_relevance",
        "objective_diversity",
        "objective_novelty",
    ]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise RuntimeError("core metrics contain non-finite required values")
    frame.to_csv(output_dir / "per_user_metrics.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    summary = frame.groupby(["experiment", "method"], as_index=False)[numeric].mean()
    summary.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "pareto_fronts.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_fronts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    generations = _collect_generations(frame)
    generations.to_csv(output_dir / "generation_metrics.csv", index=False)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "core",
        "signature": signature,
        "cohort_seed": cohort_seed,
        "optimization_seeds": seeds,
        "users": len(cohort),
        "record_count": len(frame),
        "generation_record_count": len(generations),
        "methods": CORE_METHODS,
    }
    _atomic_json(output_dir / "results.json", result)
    return result


def environment_payload() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    workers = int(args.workers or config.get("workers", 4))
    cohort_seed = int(
        args.cohort_seed if args.cohort_seed is not None else config.get("cohort_seed", 42)
    )
    output_dir = Path(args.output_dir or config.get("output_dir", "COPA/results/core"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _atomic_json(output_dir / "environment.json", environment_payload())
    result = run_core(
        config,
        output_dir,
        workers=workers,
        cohort_seed=cohort_seed,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
