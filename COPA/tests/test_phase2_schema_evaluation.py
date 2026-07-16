import json

import pytest

from copa.constraints import ConstraintRegistry
from copa.core import CandidateRecord
from copa.data import build_synthetic_case
from copa.phase2 import (
    AttributeCapability,
    CompileResult,
    ConstraintIR,
    DomainSchema,
    DomainSchemaRegistry,
    SemanticCompiler,
    get_domain_schema,
)
from copa.phase2.evaluation import evaluate_compiler, load_gold_cases


def test_domain_schema_registry_supports_aliases_and_rejects_collisions():
    registry = DomainSchemaRegistry()
    schema = DomainSchema(
        "custom",
        [
            AttributeCapability(
                "score",
                "quality_score",
                "numeric",
                (">=",),
                description="Auditable quality score.",
            )
        ],
        [],
    )
    registry.register("custom", schema, aliases=("demo_custom",))
    assert registry.get("demo-custom") is schema
    assert registry.names() == ("custom",)
    with pytest.raises(KeyError):
        registry.register("custom", DomainSchema("other", [], []))


def test_mind_schema_executes_topic_entity_and_freshness_constraints():
    domain = get_domain_schema("news")
    candidates = [
        CandidateRecord(
            "n1",
            1.0,
            {
                "category": "sports",
                "subcategory": "football",
                "entity_ids": ["Q1", "Q2"],
                "age_hours": 4.0,
                "popularity": 0.8,
            },
            "mind_test",
        ),
        CandidateRecord(
            "n2",
            0.5,
            {
                "category": "finance",
                "subcategory": "markets",
                "entity_ids": ["Q3"],
                "age_hours": 36.0,
                "popularity": 0.2,
            },
            "mind_test",
        ),
    ]
    ir = ConstraintIR.model_validate(
        {
            "schema_version": "1.0",
            "hard_constraints": [
                {"attribute": "topic", "operator": "==", "value": "sports"},
                {"attribute": "entity", "operator": "contains_any", "value": ["Q2"]},
                {"attribute": "freshness", "operator": "<=", "value": 12},
            ],
            "soft_objectives": [{"objective": "topic_diversity", "direction": "maximize"}],
            "top_k": 1,
            "unresolved_requirements": [],
        }
    )
    plan = SemanticCompiler().compile(ir, domain, candidates)
    assert not [issue for issue in plan.issues if issue.severity == "blocking"]
    assert [spec.type for spec in plan.executable_constraints] == [
        "categorical",
        "set_membership",
        "numeric",
    ]
    registry = ConstraintRegistry()
    feasible = [
        candidate.item_id
        for candidate in candidates
        if all(registry.evaluate(spec, candidate.to_row()).satisfied for spec in plan.executable_constraints)
    ]
    assert feasible == ["n1"]


def test_mind_freshness_without_adapter_metadata_is_blocking():
    candidates = [CandidateRecord("n1", 1.0, {"category": "sports", "entity_ids": ["Q1"]})]
    ir = ConstraintIR.model_validate(
        {
            "schema_version": "1.0",
            "hard_constraints": [
                {"attribute": "freshness", "operator": "<=", "value": 12},
            ],
            "soft_objectives": [],
            "top_k": 1,
            "unresolved_requirements": [],
        }
    )
    plan = SemanticCompiler().compile(ir, get_domain_schema("mind"), candidates)
    assert "attribute_has_no_values" in {issue.code for issue in plan.issues}


class _ExactCompiler:
    def __init__(self, cases, domain, candidates):
        self.cases = {case["text"]: case for case in cases}
        self.domain = domain
        self.candidates = candidates

    def compile(self, text, domain, candidates):
        case = self.cases[text]
        ir = ConstraintIR.model_validate(
            {
                "schema_version": "1.0",
                "hard_constraints": case["constraints"],
                "soft_objectives": [
                    {"objective": name, "direction": "maximize"}
                    for name in case["objectives"]
                    if name != "relevance"
                ],
                "top_k": case["top_k"],
                "unresolved_requirements": [],
            }
        )
        plan = SemanticCompiler().compile(ir, domain, candidates)
        return CompileResult("success", ir=ir, plan=plan, attempts=1, latency_seconds=0.01)


def test_compiler_evaluation_reports_parsing_execution_and_scenarios(tmp_path):
    case = build_synthetic_case(candidate_count=30)
    domain = get_domain_schema("synthetic")
    gold = [
        {
            "id": "exact_seen",
            "language": "en",
            "scenario": "seen",
            "text": "price cap",
            "expected_status": "success",
            "constraints": [{"attribute": "price", "operator": "<=", "value": 50.0}],
            "objectives": ["relevance"],
            "top_k": 10,
        },
        {
            "id": "exact_composed",
            "language": "zh",
            "scenario": "compositional",
            "text": "价格和类别",
            "expected_status": "success",
            "constraints": [
                {"attribute": "price", "operator": "<=", "value": 60.0},
                {"attribute": "category", "operator": "==", "value": "A"},
            ],
            "objectives": ["relevance"],
            "top_k": 5,
        },
    ]
    summary = evaluate_compiler(
        _ExactCompiler(gold, domain, case.candidates),
        gold,
        domain,
        case.candidates,
        tmp_path,
    )
    assert summary["parsing_accuracy"] == 1.0
    assert summary["constraint_execution_accuracy"] == 1.0
    assert summary["candidate_feasibility_f1"] == 1.0
    assert summary["scenario_metrics"]["seen"]["parsing_accuracy"] == 1.0
    assert summary["scenario_metrics"]["compositional"]["constraint_execution_accuracy"] == 1.0


def test_gold_loader_requires_valid_scenario(tmp_path):
    path = tmp_path / "gold.jsonl"
    payload = {
        "id": "bad",
        "language": "en",
        "scenario": "other",
        "text": "x",
        "expected_status": "success",
        "constraints": [],
        "objectives": ["relevance"],
        "top_k": 10,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported scenario"):
        load_gold_cases(path)
