import json

import numpy as np
import pandas as pd

from copa import ObjectiveSpec, OptimizationConfig
from copa.core import CandidateStateBus
from copa.data import build_synthetic_case
from copa.experiments.analyze_core import analyze, phase1_statistics, phase3_aggregate
from copa.experiments.run_core import run_core, strict_constraint_evaluation
from copa.metrics import hypervolume_shared
from copa.objectives import ObjectiveRegistry
from copa.optimization import WeightedGeneticOptimizer


def _synthetic_config():
    return {
        "dataset": {"type": "synthetic", "candidate_count": 20},
        "cohort_seed": 42,
        "seeds": [42],
        "workers": 1,
        "optimization": {
            "top_k": 5,
            "population_size": 8,
            "generations": 1,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "tournament_size": 2,
        },
        "objectives": {"weighted_ga_weights": [1 / 3, 1 / 3, 1 / 3]},
        "metrics": {
            "hypervolume_sample_power": 6,
            "hypervolume_bounds": {"lower": [0, 0, 0], "upper": [1, 1, 1]},
        },
    }


def test_weighted_ga_is_deterministic_and_shared_hv_is_bounded():
    case = build_synthetic_case(seed=42, candidate_count=20)
    specs = [
        ObjectiveSpec("relevance", scope="candidate"),
        ObjectiveSpec("diversity", params={"attribute": "brand_id"}),
        ObjectiveSpec("novelty", scope="candidate", params={"attribute": "popularity"}),
    ]
    outputs = []
    for _ in range(2):
        bus = CandidateStateBus()
        bus.initialize(case.candidates)
        registry = ObjectiveRegistry()
        registry.annotate_candidates(bus, specs, case.context)
        selected, front, diagnostics = WeightedGeneticOptimizer(registry).optimize(
            bus,
            specs,
            OptimizationConfig(top_k=5, population_size=10, generations=2, seed=42),
            case.context,
        )
        outputs.append((selected.item_ids, [entry.to_dict() for entry in front]))
        assert diagnostics["selection_strategy"] == "fixed_weight_ga"
        assert 0 <= hypervolume_shared(front, sample_power=6, seed=42) <= 1
    assert outputs[0] == outputs[1]


def test_strict_evaluation_catches_unconstrained_violation():
    case = build_synthetic_case(seed=42, candidate_count=20)
    strict = strict_constraint_evaluation(case, ["item_000", "item_001"])
    assert strict["strict_constraint_satisfaction_rate"] == 0
    assert strict["strict_violation_count"] >= 2
    assert strict["candidate_recall"] == 1


def test_core_runner_resume_and_analysis_schema(tmp_path):
    output = tmp_path / "phase1"
    config = _synthetic_config()
    first = run_core(config, output, workers=1, cohort_seed=42, resume=False)
    second = run_core(config, output, workers=1, cohort_seed=42, resume=True)
    assert first["record_count"] == second["record_count"] == 8
    frame = pd.read_csv(output / "per_user_metrics.csv")
    assert len(frame) == 8
    assert not frame.duplicated(["experiment", "method", "seed", "user_id"]).any()
    assert np.isfinite(frame[["recall_at_k", "shared_hypervolume"]].to_numpy()).all()
    root = tmp_path
    summary = analyze(root, bootstrap_samples=100, allow_partial=True)
    assert summary["phase1_records"] == 8
    assert summary["figure_png_count"] == summary["figure_pdf_count"]
    assert (root / "REPORT_ZH.md").exists()


def test_statistics_and_mcnemar_outputs_have_adjusted_p_values():
    rows = []
    for user in range(4):
        for method, value in [("feasible_relevance", 1.0), ("unconstrained_relevance", 0.0)]:
            rows.append(
                {
                    "experiment": "A",
                    "method": method,
                    "user_id": f"u{user}",
                    "recall_at_k": value,
                    "ndcg_at_k": value,
                    "objective_diversity": value,
                    "objective_novelty": value,
                    "strict_constraint_satisfaction_rate": value,
                    "strict_violation_rate": 1 - value,
                    "shared_hypervolume": value,
                }
            )
    aggregate, tests = phase1_statistics(pd.DataFrame(rows), 50)
    assert len(aggregate)
    assert "p_holm" in tests

    mode_rows = []
    for repeat in range(1, 4):
        for case_id in ["a", "b"]:
            for mode in ["phase2_fixed", "planner_only", "planner_repair"]:
                mode_rows.append(
                    {
                        "repeat": repeat,
                        "id": case_id,
                        "mode": mode,
                        "status": "success" if mode == "planner_repair" else "failed",
                        "verified_feasible": mode == "planner_repair",
                        "recall_at_k": 0.0,
                        "ndcg_at_k": 0.0,
                        "diversity": 0.0,
                        "novelty": 0.0,
                        "tool_call_count": 0,
                        "repair_count": 0,
                        "llm_call_count": 1,
                        "prompt_tokens": 1,
                        "output_tokens": 1,
                        "latency_seconds": 1.0,
                    }
                )
    _, consistency, mode_tests = phase3_aggregate(pd.DataFrame(mode_rows))
    assert consistency["status_repeat_consistent"].all()
    assert "p_holm" in mode_tests
