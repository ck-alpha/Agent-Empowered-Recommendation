import json
from pathlib import Path

import pytest

from copa import COPAPipeline, ObjectiveSpec, OptimizationConfig, RecommendationRequest
from copa.data import AllBeautyAdapter, build_synthetic_case
from copa.experiments.run_phase1 import main


def test_pipeline_end_to_end_writes_trace(tmp_path):
    case = build_synthetic_case(seed=42, candidate_count=20)
    request = RecommendationRequest(
        case.user_id,
        case.candidates,
        case.constraints,
        [
            ObjectiveSpec("relevance"),
            ObjectiveSpec("diversity", params={"attribute": "brand_id"}),
            ObjectiveSpec("novelty", params={"attribute": "popularity"}),
        ],
        OptimizationConfig(top_k=5, population_size=12, generations=2, seed=42),
        case.context,
    )
    result = COPAPipeline(trace_dir=tmp_path, run_id="e2e").run(request)
    assert len(result.item_ids) == 5
    assert result.verification.feasible
    assert result.trace_path and result.trace_path.exists()
    modules = {json.loads(line)["module"] for line in result.trace_path.read_text().splitlines()}
    assert {"CandidateStateBus", "HardConstraintModule", "SoftObjectiveModule", "ParetoOptimizationModule", "Verifier"} <= modules


def test_cli_runs_all_three_experiments(tmp_path):
    config = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    if not config.exists():
        pytest.skip("external experiment configuration is not included in the source-only repository")
    assert main(["--config", str(config), "--experiment", "all", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "results.json").exists()
    records = __import__("pandas").read_csv(tmp_path / "per_user_metrics.csv")
    assert set(records["experiment"]) == {"A", "B", "C"}
    assert set(records[records["experiment"] == "B"]["method"]) == {"relevance_topk", "pareto_nsga2"}


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "beauty_scenario1_items.parquet").exists(),
    reason="All Beauty processed parquet files are not available",
)
def test_all_beauty_adapter_real_data_smoke():
    processed = Path(__file__).resolve().parents[2] / "data" / "processed"
    adapter = AllBeautyAdapter(processed)
    cases = list(adapter.iter_user_cases(num_users=1, candidate_k=30, seed=42))
    assert len(cases) == 1
    case = cases[0]
    assert len(case.candidates) == 30
    assert case.relevant_items
    request = RecommendationRequest(
        case.user_id,
        case.candidates,
        case.constraints,
        [
            ObjectiveSpec("relevance"),
            ObjectiveSpec("diversity", params={"attribute": "brand_id"}),
            ObjectiveSpec("novelty", params={"attribute": "popularity"}),
        ],
        OptimizationConfig(top_k=10, population_size=8, generations=1, seed=42),
        case.context,
    )
    result = COPAPipeline(run_id="beauty_test").run(request)
    assert result.verification.feasible
    assert len(result.item_ids) == 10
