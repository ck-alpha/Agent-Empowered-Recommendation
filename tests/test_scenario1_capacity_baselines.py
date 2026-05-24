from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import (  # noqa: E402
    EcommerceInProcessingAgent,
    EcommerceInProcessingConfig,
    EcommercePostProcessingAgent,
    EcommercePostProcessingConfig,
)
from constraints import EcommerceConstraintConfig, EcommerceConstraintHandler  # noqa: E402


def test_capacity_diagnostics_counts_overflow() -> None:
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.0, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "A", "base_score": 0.9, "inventory_initial": 1},
        ]
    )

    handler = EcommerceConstraintHandler(EcommerceConstraintConfig())
    diagnostics = handler.capacity_diagnostics(recs)

    assert diagnostics["capacity_satisfied"] is False
    assert diagnostics["capacity_violation_total"] == 1.0
    assert diagnostics["over_capacity_item_count"] == 1
    assert diagnostics["max_capacity_overflow"] == 1.0


def test_postprocessing_repairs_over_capacity_with_candidate_replacement() -> None:
    candidates = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.0, "inventory_initial": 1},
            {"user_id": "u1", "item_id": "B", "base_score": 0.8, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "A", "base_score": 0.9, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "C", "base_score": 0.7, "inventory_initial": 1},
        ]
    )
    agent = EcommercePostProcessingAgent(EcommercePostProcessingConfig(top_k=1))

    result = agent.recommend_batch(candidates, user_ids=["u1", "u2"], top_k=1)
    recs = result["recommendations"]
    diagnostics = result["diagnostics"]

    assert diagnostics["final_constraints"]["capacity_satisfied"] is True
    assert diagnostics["num_swaps"] == 1
    assert len(recs) == 2
    assert recs["item_id"].value_counts().max() == 1


def test_postprocessing_shortens_list_when_no_capacity_replacement_exists() -> None:
    candidates = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.0, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "A", "base_score": 0.9, "inventory_initial": 1},
        ]
    )
    agent = EcommercePostProcessingAgent(EcommercePostProcessingConfig(top_k=1))

    result = agent.recommend_batch(candidates, user_ids=["u1", "u2"], top_k=1)

    assert result["diagnostics"]["final_constraints"]["capacity_satisfied"] is True
    assert result["diagnostics"]["candidate_shortage_rate"] == 0.5
    assert len(result["recommendations"]) == 1


def test_inprocessing_is_penalty_based_and_can_violate_capacity() -> None:
    candidates = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.0, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "A", "base_score": 0.9, "inventory_initial": 1},
        ]
    )
    agent = EcommerceInProcessingAgent(EcommerceInProcessingConfig(top_k=1, random_seed=42))

    result = agent.recommend_batch(candidates, user_ids=["u1", "u2"], top_k=1)

    assert result["diagnostics"]["final_constraints"]["capacity_satisfied"] is False
    assert result["diagnostics"]["final_constraints"]["capacity_violation_total"] == 1.0
    assert result["diagnostics"]["fully_repaired"] is False
    assert result["diagnostics"]["candidate_shortage_rate"] == 0.0
    assert len(result["recommendations"]) == 2


def test_inprocessing_dual_price_can_reduce_capacity_conflict_without_repair() -> None:
    candidates = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.00, "inventory_initial": 1},
            {"user_id": "u1", "item_id": "B", "base_score": 0.99, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "A", "base_score": 0.98, "inventory_initial": 1},
            {"user_id": "u2", "item_id": "C", "base_score": 0.10, "inventory_initial": 1},
        ]
    )
    post_agent = EcommercePostProcessingAgent(EcommercePostProcessingConfig(top_k=1))
    in_agent = EcommerceInProcessingAgent(EcommerceInProcessingConfig(top_k=1, random_seed=42))

    post_result = post_agent.recommend_batch(candidates, user_ids=["u1", "u2"], top_k=1)
    in_result = in_agent.recommend_batch(candidates, user_ids=["u1", "u2"], top_k=1)
    post_pairs = set(zip(post_result["recommendations"]["user_id"], post_result["recommendations"]["item_id"]))
    in_pairs = set(zip(in_result["recommendations"]["user_id"], in_result["recommendations"]["item_id"]))

    assert post_pairs == {("u1", "A"), ("u2", "C")}
    assert in_pairs == {("u1", "B"), ("u2", "A")}
    assert in_result["diagnostics"]["final_constraints"]["capacity_satisfied"] is True
    assert in_result["diagnostics"]["dual_nonzero_price_count"] == 1
