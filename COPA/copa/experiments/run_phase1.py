"""Unified CLI for COPA Phase-1 experiments A, B, and C."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tracemalloc
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import yaml

from copa.core import ObjectiveSpec, OptimizationConfig, RecommendationRequest
from copa.data import AllBeautyAdapter, UserCase, build_synthetic_case
from copa.metrics import front_metrics, ndcg_at_k, recall_at_k
from copa.pipeline import COPAPipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic COPA Phase-1 experiments.")
    parser.add_argument("--config", required=True, help="YAML experiment configuration.")
    parser.add_argument("--experiment", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--output-dir", default=None, help="Override the configured output directory.")
    return parser.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "dataset" not in payload or "optimization" not in payload:
        raise ValueError("Configuration requires dataset and optimization sections")
    return payload


def build_optimization(payload: Mapping[str, Any], seed: int) -> OptimizationConfig:
    fields = {
        "top_k": int(payload.get("top_k", 10)),
        "population_size": int(payload.get("population_size", 100)),
        "generations": int(payload.get("generations", 50)),
        "crossover_rate": float(payload.get("crossover_rate", 0.9)),
        "mutation_rate": float(payload.get("mutation_rate", 0.15)),
        "tournament_size": int(payload.get("tournament_size", 2)),
        "seed": int(seed),
        "selection_strategy": str(payload.get("selection_strategy", "compromise")),
        "objective_weights": payload.get("objective_weights"),
    }
    return OptimizationConfig(**fields)


def objective_specs(include_fairness: bool = False) -> List[ObjectiveSpec]:
    specs = [
        ObjectiveSpec("relevance", "maximize", "candidate"),
        ObjectiveSpec("diversity", "maximize", "slate", {"attribute": "brand_id"}),
        ObjectiveSpec("novelty", "maximize", "candidate", {"attribute": "popularity"}),
    ]
    if include_fairness:
        specs.append(ObjectiveSpec("fairness", "maximize", "slate", {"attribute": "group"}))
    return specs


def iter_cases(config: Mapping[str, Any], seed: int) -> Iterable[UserCase]:
    dataset = config["dataset"]
    dataset_type = str(dataset.get("type", "synthetic"))
    if dataset_type == "synthetic":
        yield build_synthetic_case(seed=seed, candidate_count=int(dataset.get("candidate_count", 30)))
        return
    if dataset_type == "beauty":
        adapter = AllBeautyAdapter(dataset.get("processed_dir", "data/processed"), dataset.get("prefix", "beauty_scenario1"))
        yield from adapter.iter_user_cases(
            num_users=int(dataset.get("num_users", 100)),
            candidate_k=int(dataset.get("candidate_k", 100)),
            seed=seed,
            min_history=int(dataset.get("min_history", 2)),
        )
        return
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _method_requests(
    experiment: str,
    case: UserCase,
    optimization: OptimizationConfig,
    include_fairness: bool,
) -> List[tuple[str, RecommendationRequest]]:
    all_objectives = objective_specs(include_fairness)
    if experiment == "A":
        hard_config = replace(
            optimization,
            population_size=max(2, min(optimization.population_size, 10)),
            generations=0,
            selection_strategy="weighted",
            objective_weights=[1.0],
        )
        return [
            (
                "hard_topk",
                RecommendationRequest(
                    case.user_id,
                    case.candidates,
                    case.constraints,
                    [ObjectiveSpec("relevance", "maximize", "candidate")],
                    hard_config,
                    case.context,
                    case.slate_constraints,
                ),
            )
        ]
    if experiment == "B":
        relevance_config = replace(
            optimization,
            population_size=max(2, min(optimization.population_size, 10)),
            generations=0,
            selection_strategy="weighted",
            objective_weights=[1.0] + [0.0] * (len(all_objectives) - 1),
        )
        return [
            (
                "relevance_topk",
                RecommendationRequest(case.user_id, case.candidates, [], all_objectives, relevance_config, case.context, ()),
            ),
            (
                "pareto_nsga2",
                RecommendationRequest(case.user_id, case.candidates, [], all_objectives, optimization, case.context, ()),
            ),
        ]
    if experiment == "C":
        return [
            (
                "joint_nsga2",
                RecommendationRequest(
                    case.user_id,
                    case.candidates,
                    case.constraints,
                    all_objectives,
                    optimization,
                    case.context,
                    case.slate_constraints,
                ),
            )
        ]
    raise ValueError(f"Unsupported experiment: {experiment}")


def run_experiments(
    config: Mapping[str, Any], experiments: Sequence[str], output_dir: Path
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config.get("seeds", [42, 43, 44])]
    include_fairness = bool(config.get("objectives", {}).get("include_fairness", config["dataset"].get("type") == "synthetic"))
    hv_power = int(config.get("metrics", {}).get("hypervolume_sample_power", 14))
    records: List[Dict[str, Any]] = []
    fronts: List[Dict[str, Any]] = []
    for seed in seeds:
        optimization = build_optimization(config["optimization"], seed)
        for case in iter_cases(config, seed):
            for experiment in experiments:
                for method, request in _method_requests(experiment, case, optimization, include_fairness):
                    run_id = f"phase1_{experiment}_{method}_s{seed}"
                    pipeline = COPAPipeline(trace_dir=trace_dir, run_id=run_id)
                    tracemalloc.start()
                    started = perf_counter()
                    result = pipeline.run(request)
                    runtime_seconds = perf_counter() - started
                    _, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    hard_violations = [
                        violation
                        for violation in result.verification.violations
                        if violation.get("code") in {"hard_constraint_violation", "inactive_candidate", "unknown_item"}
                    ]
                    denominator = max(1, len(result.item_ids) * len(request.constraints))
                    metric_row: Dict[str, Any] = {
                        "experiment": experiment,
                        "method": method,
                        "dataset": str(config["dataset"].get("type", "synthetic")),
                        "seed": seed,
                        "user_id": case.user_id,
                        "requested_k": request.optimization.top_k,
                        "actual_k": len(result.item_ids),
                        "recall_at_k": recall_at_k(result.item_ids, case.relevant_items, request.optimization.top_k),
                        "ndcg_at_k": ndcg_at_k(result.item_ids, case.relevant_items, request.optimization.top_k),
                        "constraint_satisfaction_rate": float(not hard_violations),
                        "violation_rate": len(hard_violations) / denominator,
                        "verified_feasible": float(result.verification.feasible),
                        "candidate_count": result.diagnostics["candidate_count"],
                        "feasible_candidate_count": result.diagnostics["feasible_candidate_count"],
                        "candidate_filter_rate": 1.0 - result.diagnostics["feasible_candidate_count"] / max(1, result.diagnostics["candidate_count"]),
                        "runtime_seconds": runtime_seconds,
                        "peak_memory_mb": peak_memory / (1024 * 1024),
                        **{f"objective_{name}": value for name, value in result.objective_values.items()},
                        **front_metrics(result.pareto_front, sample_power=hv_power, seed=seed),
                    }
                    records.append(metric_row)
                    fronts.append(
                        {
                            "experiment": experiment,
                            "method": method,
                            "seed": seed,
                            "user_id": case.user_id,
                            "selected_items": result.item_ids,
                            "verification": result.verification.to_dict(),
                            "front": [solution.to_dict() for solution in result.pareto_front],
                        }
                    )
    records_frame = pd.DataFrame(records)
    if records_frame.empty:
        raise RuntimeError("No experiment records were produced")
    numeric_columns = records_frame.select_dtypes(include=[np.number]).columns.tolist()
    summary = records_frame.groupby(["experiment", "method"], as_index=False)[numeric_columns].mean()
    records_frame.to_csv(output_dir / "per_user_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "pareto_fronts.json").open("w", encoding="utf-8") as handle:
        json.dump(fronts, handle, ensure_ascii=False, indent=2)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiments": list(experiments),
        "record_count": len(records),
        "summary": summary.to_dict("records"),
        "artifacts": {
            "per_user_metrics": "per_user_metrics.csv",
            "summary": "summary.csv",
            "pareto_fronts": "pareto_fronts.json",
            "traces": "traces/",
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def environment_payload() -> Dict[str, str]:
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
    experiments = ["A", "B", "C"] if args.experiment == "all" else [args.experiment]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or config.get("output_dir", f"COPA/results/phase1_{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
    with (output_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment_payload(), handle, indent=2)
    payload = run_experiments(config, experiments, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
