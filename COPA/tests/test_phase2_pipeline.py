import json

from copa import OptimizationConfig
from copa.data import build_synthetic_case
from copa.phase2 import (
    CompilerConfig,
    ConstraintCompiler,
    NaturalLanguageCOPAPipeline,
    NaturalLanguageRecommendationRequest,
)
from copa.phase2.ollama import OllamaResponse


class StaticClient:
    def __init__(self, payload):
        self.payload = json.dumps(payload)

    def generate_structured(self, *, system, prompt, schema):
        return OllamaResponse(self.payload, {})


def base_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "hard_constraints": [],
        "soft_objectives": [],
        "top_k": 5,
        "unresolved_requirements": [],
    }
    payload.update(overrides)
    return payload


def test_natural_language_pipeline_executes_phase1_with_multiple_diversity_objectives(tmp_path):
    case = build_synthetic_case(seed=42)
    payload = base_payload(
        hard_constraints=[{"attribute": "price", "operator": "<=", "value": 60, "currency": "USD"}],
        soft_objectives=[
            {"objective": "brand_diversity", "direction": "maximize"},
            {"objective": "category_diversity", "direction": "maximize"},
        ],
    )
    compiler = ConstraintCompiler(
        StaticClient(payload),
        config=CompilerConfig(audit_dir=tmp_path / "compiler"),
    )
    request = NaturalLanguageRecommendationRequest(
        user_id=case.user_id,
        text="60美元以内，品牌和类别都尽量多样，给我5个",
        candidates=case.candidates,
        domain="synthetic",
        optimization=OptimizationConfig(top_k=10, population_size=12, generations=2, seed=42),
        context=case.context,
    )
    result = NaturalLanguageCOPAPipeline(compiler, trace_dir=tmp_path, run_id="nl_e2e").run(request)
    assert result.compile_result.status == "success"
    assert result.recommendation is not None
    assert result.recommendation.verification.feasible
    assert len(result.recommendation.item_ids) == 5
    assert set(result.recommendation.objective_values) == {"relevance", "brand_diversity", "category_diversity"}


def test_clarification_never_executes_recommendation():
    case = build_synthetic_case()
    payload = base_payload(
        unresolved_requirements=[
            {"text": "便宜一点", "reason": "missing threshold", "clarification_question": "What maximum price?"}
        ]
    )
    compiler = ConstraintCompiler(StaticClient(payload))
    request = NaturalLanguageRecommendationRequest(
        case.user_id,
        "推荐便宜一点的",
        case.candidates,
        optimization=OptimizationConfig(top_k=5, population_size=8, generations=1),
    )
    result = NaturalLanguageCOPAPipeline(compiler).run(request)
    assert result.compile_result.status == "clarification_required"
    assert result.recommendation is None


def test_system_constraints_are_merged_with_user_constraints():
    case = build_synthetic_case()
    payload = base_payload(
        hard_constraints=[{"attribute": "category", "operator": "==", "value": "A", "currency": None}]
    )
    compiler = ConstraintCompiler(StaticClient(payload))
    request = NaturalLanguageRecommendationRequest(
        case.user_id,
        "只要A类",
        case.candidates,
        base_constraints=case.constraints[:1],
        optimization=OptimizationConfig(top_k=5, population_size=8, generations=1),
    )
    result = NaturalLanguageCOPAPipeline(compiler).run(request)
    assert result.compile_result.plan is not None
    provenance = [entry.provenance for entry in result.compile_result.plan.constraints]
    assert provenance == ["system", "user"]
