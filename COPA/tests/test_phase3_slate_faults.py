import pytest

from copa import (
    COPAExecutionSession,
    ObjectiveSpec,
    OptimizationConfig,
    RecommendationRequest,
    SlateConstraintSpec,
)
from copa.data import build_synthetic_case
from copa.phase3.models import AgentExecutionPlan
from copa.phase3.repair import RepairPolicy
from copa.phase3.tools import AgentPlanExecutor


def plan():
    return AgentExecutionPlan.model_validate(
        {
            "plan_version": "1.0",
            "strategy": "feasible_topk",
            "steps": [
                {"tool": "apply_constraints"},
                {"tool": "compute_objectives"},
                {"tool": "select_feasible_topk"},
                {"tool": "verify"},
            ],
            "assumptions": [],
            "unresolved_requirements": [],
        }
    )


@pytest.mark.parametrize(
    "fault,spec",
    [
        (
            "total_price_over",
            SlateConstraintSpec(
                "budget", "aggregate_sum", "price", "<=", 200.0
            ),
        ),
        (
            "brand_cap_over",
            SlateConstraintSpec(
                "brand_cap", "per_group_count", "brand_id", "<=", 2
            ),
        ),
        (
            "coverage_under",
            SlateConstraintSpec(
                "coverage", "distinct_count", "category", ">=", 3
            ),
        ),
        (
            "forged_optimizer_feasible",
            SlateConstraintSpec(
                "budget", "aggregate_sum", "price", "<=", 200.0
            ),
        ),
    ],
)
def test_equal_slate_faults_are_independently_detected(fault, spec):
    case = build_synthetic_case(candidate_count=30)
    request = RecommendationRequest(
        user_id=case.user_id,
        candidates=case.candidates,
        constraints=[],
        objectives=[ObjectiveSpec("relevance")],
        optimization=OptimizationConfig(
            top_k=5, population_size=8, generations=1, seed=42
        ),
        slate_constraints=[spec],
    )
    session = COPAExecutionSession(request)
    execution = AgentPlanExecutor().execute(
        plan(), session, fault={"type": fault}
    )
    report = execution.session.verification
    assert not report.feasible
    assert "slate_constraint_violation" in {
        violation["code"] for violation in report.violations
    }


def test_repair_policy_never_relaxes_proven_infeasible_constraints():
    policy = RepairPolicy()
    with pytest.raises(ValueError):
        from copa.phase3.models import RepairDecision

        policy.validate(
            RepairDecision(
                decision_version="1.0",
                action="reexecute_selection",
                reason_code="forbidden_relaxation",
            ),
            ["proven_infeasible"],
        )
