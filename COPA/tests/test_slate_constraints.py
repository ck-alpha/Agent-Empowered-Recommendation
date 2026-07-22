from itertools import combinations

import pandas as pd
import pytest

from copa import (
    CandidateRecord,
    ObjectiveSpec,
    OptimizationConfig,
    RecommendationRequest,
    SlateConstraintRegistry,
    SlateConstraintSpec,
)
from copa.constraints import ConstraintRegistry
from copa.core import CandidateStateBus
from copa.core.verifier import DeterministicVerifier
from copa.objectives import ObjectiveRegistry
from copa.pipeline import COPAPipeline


def candidates():
    return [
        CandidateRecord("a", 1.0, {"price": 8.0, "brand": "x", "brand_id": "x", "category": "c1", "popularity": 0.9}),
        CandidateRecord("b", 0.9, {"price": 9.0, "brand": "x", "brand_id": "x", "category": "c1", "popularity": 0.8}),
        CandidateRecord("c", 0.8, {"price": 5.0, "brand": "y", "brand_id": "y", "category": "c2", "popularity": 0.7}),
        CandidateRecord("d", 0.7, {"price": 4.0, "brand": "z", "brand_id": "z", "category": "c3", "popularity": 0.6}),
        CandidateRecord("e", 0.6, {"price": 3.0, "brand": "w", "brand_id": "w", "category": "c2", "popularity": 0.5}),
    ]


def frame():
    return pd.DataFrame([item.to_row() for item in candidates()])


def specs():
    return [
        SlateConstraintSpec("budget", "aggregate_sum", "price", "<=", 20.0),
        SlateConstraintSpec("brands", "distinct_count", "brand", ">=", 2),
        SlateConstraintSpec("brand_cap", "per_group_count", "brand", "<=", 2),
        SlateConstraintSpec(
            "category_coverage", "group_count", "category", ">=", 1, ("c2",)
        ),
    ]


def test_all_slate_constraint_types_and_exact_solver_match_enumeration():
    registry = SlateConstraintRegistry()
    feasible = {
        tuple(ids)
        for ids in combinations([item.item_id for item in candidates()], 3)
        if registry.is_feasible(specs(), ids, frame())
    }
    result = registry.solve(
        frame(),
        specs(),
        3,
        objective_scores={item.item_id: item.base_score for item in candidates()},
    )
    assert result.status == "optimal"
    assert tuple(sorted(result.item_ids)) in {tuple(sorted(value)) for value in feasible}
    optimum = max(
        sum(next(item.base_score for item in candidates() if item.item_id == item_id) for item_id in ids)
        for ids in feasible
    )
    assert result.objective_value == pytest.approx(optimum)


def test_missing_metadata_and_impossible_slate_fail_closed():
    registry = SlateConstraintRegistry()
    with pytest.raises(KeyError, match="Missing candidate attribute"):
        registry.validate(
            [SlateConstraintSpec("missing", "aggregate_sum", "duration", "<=", 3)],
            frame(),
        )
    impossible = [SlateConstraintSpec("too_many", "distinct_count", "brand", ">=", 5)]
    assert registry.solve(frame(), impossible, 3).status == "infeasible"
    with pytest.raises(ValueError, match="must be integers"):
        registry.validate(
            [SlateConstraintSpec("fractional", "distinct_count", "brand", ">=", 2.5)],
            frame(),
        )


def test_optimizer_and_independent_verifier_enforce_slate_constraints():
    request = RecommendationRequest(
        "u",
        candidates(),
        [],
        [ObjectiveSpec("relevance"), ObjectiveSpec("diversity", params={"attribute": "brand"})],
        OptimizationConfig(top_k=3, population_size=12, generations=3, seed=9),
        {},
        specs(),
    )
    first = COPAPipeline().run(request)
    second = COPAPipeline().run(request)
    assert first.status == "success"
    assert first.item_ids == second.item_ids
    assert first.verification.feasible is True
    assert SlateConstraintRegistry().is_feasible(specs(), first.item_ids, frame())

    bus = CandidateStateBus()
    bus.initialize(candidates())
    ConstraintRegistry().apply(bus, [])
    report = DeterministicVerifier().verify(
        ["a", "b", "c"], bus, [], 3, slate_specs=specs()
    )
    assert report.feasible is False
    assert "slate_constraint_violation" in {
        violation["code"] for violation in report.violations
    }


def test_preflight_returns_no_public_slate_for_proven_infeasible_request():
    request = RecommendationRequest(
        "u",
        candidates(),
        [],
        [ObjectiveSpec("relevance")],
        OptimizationConfig(top_k=3, population_size=4, generations=0),
        {},
        [SlateConstraintSpec("impossible", "distinct_count", "brand", ">=", 5)],
    )
    result = COPAPipeline().run(request)
    assert result.status == "proven_infeasible"
    assert result.item_ids == []
    assert result.verification.feasible is False


def test_operator_ablation_still_excludes_infeasible_population_members():
    spec = SlateConstraintSpec(
        "brand_cap", "per_group_count", "brand", "<=", 2
    )
    request = RecommendationRequest(
        "u",
        candidates(),
        [],
        [ObjectiveSpec("relevance")],
        OptimizationConfig(
            top_k=3,
            population_size=8,
            generations=2,
            seed=42,
            use_milp_seed=False,
            use_slate_feasible_operators=False,
        ),
        {},
        [spec],
    )
    result = COPAPipeline().run(request)
    assert result.status == "success"
    assert all(
        SlateConstraintRegistry().is_feasible(
            [spec], solution.item_ids, frame()
        )
        for solution in result.pareto_front
    )
    assert result.diagnostics["use_milp_seed"] is False


def test_per_group_minimum_applies_only_to_represented_groups():
    registry = SlateConstraintRegistry()
    spec = SlateConstraintSpec(
        "brand_min", "per_group_count", "brand", ">=", 2
    )
    result = registry.solve(
        frame(),
        [spec],
        2,
        objective_scores={item_id: 1.0 for item_id in ("a", "b")},
    )
    assert result.status == "optimal"
    assert registry.evaluate(spec, result.item_ids, frame()).satisfied
    selected = frame().set_index("item_id").loc[result.item_ids]
    brands = selected["metadata"].map(lambda metadata: metadata["brand"])
    assert all(count >= 2 for count in brands.value_counts())


def test_compiled_constraints_match_reference_for_every_slate_and_swap():
    registry = SlateConstraintRegistry()
    candidate_frame = frame()
    compiled = registry.compile(specs(), candidate_frame)
    all_ids = [item.item_id for item in candidates()]
    for item_ids in combinations(all_ids, 3):
        reference = registry.evaluate_all(specs(), item_ids, candidate_frame)
        optimized = compiled.evaluate_all(item_ids)
        assert [value.satisfied for _, value in optimized] == [
            value.satisfied for _, value in reference
        ]
        assert [value.violation_magnitude for _, value in optimized] == pytest.approx(
            [value.violation_magnitude for _, value in reference], abs=1e-15
        )
        assert compiled.is_feasible(item_ids) == registry.is_feasible(
            specs(), item_ids, candidate_frame
        )
        assert compiled.total_violation(item_ids) == pytest.approx(
            registry.total_violation(specs(), item_ids, candidate_frame), abs=1e-15
        )

    initial = ["a", "c", "e"]
    initial_state = compiled.build_state(initial)
    swapped = compiled.try_swap(initial_state, 0, "d")
    expected = ["d", "c", "e"]
    assert (swapped is not None) == registry.is_feasible(
        specs(), expected, candidate_frame
    )
    if swapped is not None:
        assert compiled.item_ids_for_state(swapped) == expected


def test_compiled_builtin_objectives_are_numerically_identical():
    candidate_frame = frame()
    objective_specs = [
        ObjectiveSpec("relevance"),
        ObjectiveSpec("diversity", params={"attribute": "brand"}),
        ObjectiveSpec("novelty", params={"attribute": "price"}),
        ObjectiveSpec("fairness", params={"attribute": "category"}),
    ]
    registry = ObjectiveRegistry()
    compiled = registry.compile(candidate_frame, objective_specs)
    for item_ids in combinations([item.item_id for item in candidates()], 3):
        assert compiled.evaluate(item_ids) == pytest.approx(
            registry.evaluate(item_ids, candidate_frame, objective_specs), abs=1e-15
        )


@pytest.mark.parametrize("method", ["copa", "feasible_weighted_ga"])
def test_optimizer_deadline_fails_closed_without_delivering_a_slate(method, tmp_path):
    from copa.experiments.run_core import _execute

    request = RecommendationRequest(
        "timeout-user",
        candidates(),
        [],
        [ObjectiveSpec("relevance"), ObjectiveSpec("diversity", params={"attribute": "brand"})],
        OptimizationConfig(
            top_k=3,
            population_size=20,
            generations=20,
            seed=42,
            optimizer_time_limit_seconds=1e-12,
        ),
        {},
        specs(),
    )
    case = type(
        "Case",
        (),
        {
            "user_id": request.user_id,
            "candidates": request.candidates,
            "constraints": request.constraints,
            "slate_constraints": request.slate_constraints,
            "context": request.context,
            "relevant_items": [],
        },
    )()
    result = _execute(
        case,
        "timeout",
        method,
        request.optimization,
        tmp_path,
        (1 / 3, 1 / 3, 1 / 3),
    )
    assert result.status == "optimizer_failed"
    assert result.item_ids == []
    assert "exceeded" in result.diagnostics["optimizer_error"]
