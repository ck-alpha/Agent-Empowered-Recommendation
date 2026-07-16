import json
from pathlib import Path

import pytest

from copa.data import build_synthetic_case
from copa.phase2 import CompilerConfig, ConstraintCompiler, synthetic_domain_schema
from copa.phase2.ollama import OllamaResponse, OllamaTransportError


def ir_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "hard_constraints": [],
        "soft_objectives": [],
        "top_k": None,
        "unresolved_requirements": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_structured(self, *, system, prompt, schema):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return OllamaResponse(output, {"prompt_eval_count": 10, "eval_count": 5})


def test_compiler_maps_ir_adds_relevance_and_writes_private_audit(tmp_path):
    case = build_synthetic_case()
    client = FakeClient(
        [
            ir_payload(
                hard_constraints=[{"attribute": "价格", "operator": "<=", "value": 60, "currency": "USD"}],
                soft_objectives=[{"objective": "brand_diversity", "direction": "maximize"}],
                top_k=5,
            )
        ]
    )
    secret_text = "推荐60美元以内的商品 SECRET-CONTENT"
    result = ConstraintCompiler(client, config=CompilerConfig(audit_dir=tmp_path)).compile(
        secret_text, synthetic_domain_schema(), case.candidates
    )
    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.top_k == 5
    assert [(entry.spec.attribute, entry.spec.operator, entry.spec.value) for entry in result.plan.constraints] == [
        ("price", "<=", 60.0)
    ]
    assert [entry.spec.name for entry in result.plan.objectives] == ["relevance", "brand_diversity"]
    assert result.plan.objectives[0].provenance == "system_default"
    audit = result.audit_path.read_text(encoding="utf-8")
    assert "SECRET-CONTENT" not in audit
    assert "request_sha256" in audit


def test_compiler_repairs_invalid_json_once():
    case = build_synthetic_case()
    client = FakeClient(["not json", ir_payload(top_k=5)])
    result = ConstraintCompiler(
        client,
        config=CompilerConfig(max_attempts=2, retry_backoff_seconds=0),
    ).compile("给我5个商品", synthetic_domain_schema(), case.candidates)
    assert result.status == "success"
    assert result.attempts == 2
    assert len(client.calls) == 2
    assert "repair_request" in client.calls[1]["prompt"]


def test_compiler_fails_closed_after_transport_errors():
    case = build_synthetic_case()
    client = FakeClient([OllamaTransportError("down"), OllamaTransportError("still down")])
    result = ConstraintCompiler(
        client,
        config=CompilerConfig(max_attempts=2, retry_backoff_seconds=0),
    ).compile("推荐商品", synthetic_domain_schema(), case.candidates)
    assert result.status == "failed"
    assert result.plan is None
    assert result.attempts == 2


def test_unknown_attribute_and_unresolved_requirement_require_clarification():
    case = build_synthetic_case()
    payload = ir_payload(
        hard_constraints=[{"attribute": "color", "operator": "==", "value": "red", "currency": None}],
        unresolved_requirements=[
            {"text": "高质量", "reason": "quality is undefined", "clarification_question": "Which quality metric?"}
        ],
    )
    result = ConstraintCompiler(FakeClient([payload])).compile(
        "只要红色高质量商品", synthetic_domain_schema(), case.candidates
    )
    assert result.status == "clarification_required"
    assert result.plan is not None
    assert {issue.code for issue in result.issues} == {"unknown_attribute", "unresolved_requirement"}


def test_conflicts_and_unsupported_currency_are_blocking():
    case = build_synthetic_case()
    payload = ir_payload(
        hard_constraints=[
            {"attribute": "price", "operator": ">", "value": 80, "currency": "USD"},
            {"attribute": "price", "operator": "<", "value": 20, "currency": "USD"},
            {"attribute": "price", "operator": "<=", "value": 100, "currency": "EUR"},
        ]
    )
    result = ConstraintCompiler(FakeClient([payload])).compile(
        "conflicting request", synthetic_domain_schema(), case.candidates
    )
    assert result.status == "clarification_required"
    assert {issue.code for issue in result.issues} >= {"numeric_conflict", "unsupported_currency"}


@pytest.mark.parametrize(
    "constraints,expected_code",
    [
        (
            [
                {"attribute": "price", "operator": ">", "value": 20, "currency": "USD"},
                {"attribute": "price", "operator": "<=", "value": 20, "currency": "USD"},
            ],
            "numeric_conflict",
        ),
        (
            [
                {"attribute": "category", "operator": "==", "value": "A", "currency": None},
                {"attribute": "category", "operator": "not_in", "value": ["A"], "currency": None},
            ],
            "categorical_conflict",
        ),
    ],
)
def test_boundary_and_set_conflicts_are_detected(constraints, expected_code):
    case = build_synthetic_case()
    result = ConstraintCompiler(FakeClient([ir_payload(hard_constraints=constraints)])).compile(
        "conflict", synthetic_domain_schema(), case.candidates
    )
    assert result.status == "clarification_required"
    assert expected_code in {issue.code for issue in result.issues}


def test_gold_corpus_has_sixty_bilingual_cases():
    from copa.phase2.evaluation import load_gold_cases

    path = Path(__file__).resolve().parents[1] / "evaluation" / "constraint_compiler_gold.jsonl"
    cases = load_gold_cases(path)
    assert len(cases) == 60
    assert {case["language"] for case in cases} == {"zh", "en"}
