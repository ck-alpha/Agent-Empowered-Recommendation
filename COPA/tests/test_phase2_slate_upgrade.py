import json
from pathlib import Path

from copa import OptimizationConfig
from copa.data import build_synthetic_case
from copa.phase2 import (
    CompilerConfig,
    ConstraintCompiler,
    NaturalLanguageCOPAPipeline,
    NaturalLanguageRecommendationRequest,
    SemanticCompiler,
    ConstraintIR,
    synthetic_domain_schema,
)
from copa.phase2.ollama import OllamaResponse
from copa.phase2.evaluation.compiler_eval import (
    _expected_constraints,
    _feasible_slates,
    load_gold_cases,
)


class StaticClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_structured(self, *, system, prompt, schema):
        return OllamaResponse(json.dumps(self.payload), {})


def slate_payload(constraints, *, top_k=5):
    return {
        "schema_version": "1.1",
        "hard_constraints": constraints,
        "soft_objectives": [],
        "top_k": top_k,
        "unresolved_requirements": [],
    }


def test_ir_v10_promotes_missing_scope_to_item():
    ir = ConstraintIR.model_validate(
        {
            "schema_version": "1.0",
            "hard_constraints": [
                {"attribute": "price", "operator": "<=", "value": 50}
            ],
            "soft_objectives": [],
            "top_k": 5,
            "unresolved_requirements": [],
        }
    )
    assert ir.hard_constraints[0].scope == "item"
    assert ir.hard_constraints[0].aggregation is None


def test_compiler_v4_executes_all_four_slate_kinds(tmp_path):
    case = build_synthetic_case(candidate_count=30)
    payload = slate_payload(
        [
            {
                "scope": "slate",
                "aggregation": "aggregate_sum",
                "attribute": "price",
                "operator": "<=",
                "value": 350,
                "currency": "USD",
            },
            {
                "scope": "slate",
                "aggregation": "distinct_count",
                "attribute": "category",
                "operator": ">=",
                "value": 2,
            },
            {
                "scope": "slate",
                "aggregation": "per_group_count",
                "attribute": "brand",
                "operator": "<=",
                "value": 2,
            },
            {
                "scope": "slate",
                "aggregation": "group_count",
                "attribute": "category",
                "operator": ">=",
                "value": 2,
                "target_values": ["A", "B"],
            },
        ]
    )
    compiler = ConstraintCompiler(
        StaticClient(payload),
        config=CompilerConfig(
            prompt_version="constraint_compiler_v4", audit_dir=tmp_path
        ),
    )
    result = compiler.compile("列表级约束", synthetic_domain_schema(), case.candidates)
    assert result.status == "success"
    assert [entry.spec.type for entry in result.plan.slate_constraints] == [
        "aggregate_sum",
        "distinct_count",
        "per_group_count",
        "group_count",
    ]

    recommendation = NaturalLanguageCOPAPipeline(compiler).run(
        NaturalLanguageRecommendationRequest(
            user_id=case.user_id,
            text="列表级约束",
            candidates=case.candidates,
            optimization=OptimizationConfig(
                top_k=5, population_size=10, generations=1, seed=42
            ),
        )
    ).recommendation
    assert recommendation is not None
    assert recommendation.status == "success"
    assert recommendation.verification.feasible
    assert len(recommendation.verification.checked_slate_constraints) == 4


def test_compiler_preflight_blocks_proven_slate_infeasibility():
    case = build_synthetic_case(candidate_count=30)
    payload = slate_payload(
        [
            {
                "scope": "slate",
                "aggregation": "distinct_count",
                "attribute": "category",
                "operator": ">=",
                "value": 4,
            }
        ]
    )
    result = ConstraintCompiler(StaticClient(payload)).compile(
        "至少四个类别", synthetic_domain_schema(), case.candidates
    )
    assert result.status == "clarification_required"
    assert "slate_constraints_infeasible" in {issue.code for issue in result.issues}


def test_all_32_bilingual_slate_gold_cases_have_executable_gold_semantics():
    root = Path(__file__).resolve().parents[1]
    gold = load_gold_cases(root / "evaluation" / "constraint_compiler_gold.jsonl")
    slate_cases = [case for case in gold if str(case["id"]).startswith("slate_")]
    assert len(slate_cases) == 32
    assert {case["language"] for case in slate_cases} == {"zh", "en"}
    fixture = build_synthetic_case(candidate_count=30)
    domain = synthetic_domain_schema()
    for case in slate_cases:
        if case["expected_status"] != "success":
            continue
        item_specs, slate_specs = _expected_constraints(
            case, domain, fixture.candidates
        )
        if slate_specs:
            _feasible_slates(
                fixture.candidates[:12],
                item_specs,
                slate_specs,
                int(case["top_k"]),
            )
