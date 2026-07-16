"""Natural-language COPA example using the local Qwen2.5:14B model."""

from pathlib import Path

from copa import (
    ConstraintCompiler,
    NaturalLanguageCOPAPipeline,
    NaturalLanguageRecommendationRequest,
    OptimizationConfig,
)
from copa.data import build_synthetic_case
from copa.phase2 import CompilerConfig


case = build_synthetic_case(seed=42)
compiler = ConstraintCompiler(config=CompilerConfig(audit_dir=Path("COPA/logs/phase2")))
request = NaturalLanguageRecommendationRequest(
    user_id=case.user_id,
    text="推荐价格不超过60美元且有货的商品，最好品牌多样一些，给我5个",
    candidates=case.candidates,
    domain="synthetic",
    base_constraints=[],
    optimization=OptimizationConfig(top_k=10, population_size=30, generations=5, seed=42),
    context=case.context,
)
result = NaturalLanguageCOPAPipeline(
    compiler,
    trace_dir=Path("COPA/logs/phase2"),
    run_id="phase2_example",
).run(request)
print(result.to_dict())
