from copa import (
    CandidateRecord,
    CandidateStateBus,
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationConfig,
)
from copa.constraints import ConstraintRegistry
from copa.core.verifier import DeterministicVerifier
from copa.optimization import ParetoOptimizer, dominates


def make_bus(count=12):
    bus = CandidateStateBus()
    bus.initialize(
        [
            CandidateRecord(
                f"i{index}",
                1 - index / count,
                {"brand_id": f"b{index % 4}", "popularity": index / count, "price": index},
            )
            for index in range(count)
        ]
    )
    return bus


def test_nsga2_is_reproducible_unique_and_returns_nondominated_front():
    specs = [
        ObjectiveSpec("relevance"),
        ObjectiveSpec("diversity", params={"attribute": "brand_id"}),
        ObjectiveSpec("novelty", params={"attribute": "popularity"}),
    ]
    config = OptimizationConfig(top_k=5, population_size=20, generations=4, seed=7)
    selected_a, front_a, _ = ParetoOptimizer().optimize(make_bus(), specs, config)
    selected_b, front_b, _ = ParetoOptimizer().optimize(make_bus(), specs, config)
    assert selected_a.item_ids == selected_b.item_ids
    assert len(selected_a.item_ids) == len(set(selected_a.item_ids)) == 5
    assert [solution.item_ids for solution in front_a] == [solution.item_ids for solution in front_b]
    for left in front_a:
        for right in front_a:
            if left is not right:
                assert not dominates(left, right)


def test_verifier_reports_shortage_and_rechecks_hard_constraints():
    bus = make_bus(3)
    spec = ConstraintSpec("cheap", "numeric", "price", "<=", 0)
    ConstraintRegistry().apply(bus, [spec])
    report = DeterministicVerifier().verify(["i0"], bus, [spec], requested_k=2, objective_values={"relevance": 1.0})
    assert report.feasible is False
    assert report.violations[0]["code"] == "insufficient_candidates"

    violating = DeterministicVerifier().verify(["i1"], bus, [spec], requested_k=1)
    codes = {violation["code"] for violation in violating.violations}
    assert "inactive_candidate" in codes
    assert "hard_constraint_violation" in codes
