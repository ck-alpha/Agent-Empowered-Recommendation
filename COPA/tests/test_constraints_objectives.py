import math

import pytest

from copa import CandidateRecord, CandidateStateBus, ConstraintSpec, ObjectiveSpec
from copa.constraints import ConstraintRegistry
from copa.objectives import ObjectiveRegistry


def make_bus():
    bus = CandidateStateBus()
    bus.initialize(
        [
            CandidateRecord("a", 1.0, {"price": 10, "category": "x", "brand_id": "p", "popularity": 0.8, "group": "g1", "available": True}),
            CandidateRecord("b", 0.5, {"price": 30, "category": "y", "brand_id": "q", "popularity": 0.2, "group": "g2", "available": False}),
            CandidateRecord("c", 0.2, {"price": 50, "category": "x", "brand_id": "p", "popularity": 0.0, "group": "g2", "available": True}),
        ]
    )
    return bus


def test_constraint_registry_applies_all_builtin_types():
    bus = make_bus()
    registry = ConstraintRegistry()
    specs = [
        ConstraintSpec("price", "numeric", "price", "between", [0, 40]),
        ConstraintSpec("category", "categorical", "category", "in", ["x", "y"]),
        ConstraintSpec("excluded", "exclusion", "item_id", "not_in", ["never"]),
        ConstraintSpec("available", "boolean", "available", "==", True),
    ]
    registry.apply(bus, specs)
    assert bus.query(feasible_only=True)["item_id"].tolist() == ["a"]
    state_b = bus.query().set_index("item_id").loc["b", "hard_state"]
    assert state_b["feasible"] is False
    assert state_b["violations"][0]["constraint_id"] == "available"


def test_invalid_constraint_does_not_create_version():
    bus = make_bus()
    with pytest.raises(KeyError, match="Missing candidate attribute"):
        ConstraintRegistry().apply(bus, [ConstraintSpec("missing", "numeric", "unknown", "<=", 2)])
    assert bus.version == 0


def test_objectives_have_hand_computable_values_and_candidate_annotations():
    bus = make_bus()
    registry = ObjectiveRegistry()
    specs = [
        ObjectiveSpec("relevance", scope="candidate"),
        ObjectiveSpec("diversity", params={"attribute": "brand_id"}),
        ObjectiveSpec("novelty", scope="candidate", params={"attribute": "popularity"}),
        ObjectiveSpec("fairness", params={"attribute": "group", "target_distribution": {"g1": 0.5, "g2": 0.5}}),
    ]
    values = registry.evaluate(["a", "b"], bus.query(), specs)
    weights = [1 / math.log2(2), 1 / math.log2(3)]
    assert values["relevance"] == pytest.approx((1.0 * weights[0] + 0.5 * weights[1]) / sum(weights))
    assert values["diversity"] == 1.0
    assert values["novelty"] == pytest.approx(0.5)
    assert values["fairness"] == 1.0

    registry.annotate_candidates(bus, specs)
    annotations = bus.query().set_index("item_id").loc["a", "soft_objectives"]
    assert set(annotations) == {"relevance", "novelty"}
