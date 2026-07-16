"""Minimal typed-API example for COPA Phase 1."""

from pathlib import Path

from copa import COPAPipeline, ObjectiveSpec, OptimizationConfig, RecommendationRequest
from copa.data import build_synthetic_case


case = build_synthetic_case(seed=42)
request = RecommendationRequest(
    user_id=case.user_id,
    candidates=case.candidates,
    constraints=case.constraints,
    objectives=[
        ObjectiveSpec("relevance", scope="candidate"),
        ObjectiveSpec("diversity", params={"attribute": "brand_id"}),
        ObjectiveSpec("novelty", scope="candidate", params={"attribute": "popularity"}),
        ObjectiveSpec("fairness", params={"attribute": "group"}),
    ],
    optimization=OptimizationConfig(top_k=5, population_size=30, generations=5, seed=42),
    context=case.context,
)
result = COPAPipeline(trace_dir=Path("COPA/logs"), run_id="synthetic_example").run(request)
print(result.to_dict())
