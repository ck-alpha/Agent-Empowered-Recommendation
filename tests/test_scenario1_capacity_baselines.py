from pathlib import Path
import sys
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import (  # noqa: E402
    EcommerceInProcessingAgent,
    EcommerceInProcessingConfig,
    EcommerceOnlineGreedyAgent,
    EcommerceOnlineGreedyConfig,
    EcommercePostProcessingAgent,
    EcommercePostProcessingConfig,
)
from constraints import EcommerceConstraintConfig, EcommerceConstraintHandler  # noqa: E402
from run_scenario1_baselines import (  # noqa: E402
    annotate_expected_consumption,
    apply_dynamic_inventory_protocol,
    build_id_mappings,
    build_raw_topk_recommendations,
    iterative_k_core_filter,
    make_data_stat,
    sample_test_users,
    temporal_train_test_split,
)


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


def test_capacity_diagnostics_supports_expected_consumption() -> None:
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "inventory_capacity": 1.0, "expected_consumption": 0.6},
            {"user_id": "u2", "item_id": "A", "inventory_capacity": 1.0, "expected_consumption": 0.7},
        ]
    )

    handler = EcommerceConstraintHandler(
        EcommerceConstraintConfig(capacity_col="inventory_capacity", consumption_col="expected_consumption")
    )
    diagnostics = handler.evaluate_all(recs)

    assert diagnostics["capacity_satisfied"] is False
    assert abs(diagnostics["capacity_violation_total"] - 0.3) < 1e-9
    assert diagnostics["consumption_mode"] == "expected"
    assert abs(diagnostics["total_consumption"] - 1.3) < 1e-9


def test_online_greedy_respects_dynamic_expected_inventory() -> None:
    candidates = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.0, "inventory_capacity": 1.0, "expected_consumption": 0.7},
            {"user_id": "u1", "item_id": "B", "base_score": 0.8, "inventory_capacity": 1.0, "expected_consumption": 0.5},
            {"user_id": "u2", "item_id": "A", "base_score": 1.0, "inventory_capacity": 1.0, "expected_consumption": 0.7},
            {"user_id": "u2", "item_id": "C", "base_score": 0.7, "inventory_capacity": 1.0, "expected_consumption": 0.4},
        ]
    )
    agent = EcommerceOnlineGreedyAgent(
        EcommerceOnlineGreedyConfig(top_k=1, capacity_col="inventory_capacity", consumption_col="expected_consumption")
    )

    result = agent.recommend_batch(candidates, user_ids=["u1", "u2"], top_k=1)
    recs = result["recommendations"]

    assert recs["item_id"].tolist() == ["A", "C"]
    assert result["diagnostics"]["final_constraints"]["capacity_satisfied"] is True


def test_dynamic_inventory_protocol_pressure_ordering() -> None:
    candidates = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "base_score": 1.0, "inventory_initial": 5, "popularity": 1.0},
            {"user_id": "u1", "item_id": "B", "base_score": 0.8, "inventory_initial": 5, "popularity": 0.5},
            {"user_id": "u2", "item_id": "A", "base_score": 1.0, "inventory_initial": 5, "popularity": 1.0},
            {"user_id": "u2", "item_id": "C", "base_score": 0.7, "inventory_initial": 5, "popularity": 0.2},
        ]
    )
    annotated = annotate_expected_consumption(candidates, top_k=1, expected_orders_per_user=1.0)
    raw = build_raw_topk_recommendations(annotated, top_k=1)

    abundant, abundant_summary, _ = apply_dynamic_inventory_protocol(
        annotated,
        raw,
        mechanism="demand_aligned",
        pressure_level="abundant",
        seed=42,
    )
    scarce, scarce_summary, _ = apply_dynamic_inventory_protocol(
        annotated,
        raw,
        mechanism="demand_aligned",
        pressure_level="scarce",
        seed=42,
    )

    assert abundant_summary["inventory_total"] >= scarce_summary["inventory_total"]
    assert scarce_summary["realized_pressure"] >= abundant_summary["realized_pressure"]
    assert "inventory_capacity" in abundant.columns
    assert "inventory_capacity" in scarce.columns


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


def test_iterative_k_core_filters_users_and_items_until_stable() -> None:
    rows = []
    timestamp = 0
    core_users = [f"u{i}" for i in range(6)]
    core_items = [f"i{i}" for i in range(6)]
    for user_id in core_users:
        for item_id in core_items:
            rows.append({"user_id": user_id, "item_id": item_id, "timestamp": timestamp})
            timestamp += 1

    for item_id in core_items[:5] + ["rare"]:
        rows.append({"user_id": "u_bridge", "item_id": item_id, "timestamp": timestamp})
        timestamp += 1

    interactions = pd.DataFrame(rows)

    filtered, stats = iterative_k_core_filter(
        interactions,
        min_user_interactions=6,
        min_item_interactions=6,
    )

    assert set(filtered["user_id"]) == set(core_users)
    assert "rare" not in set(filtered["item_id"])
    assert int(filtered["user_id"].value_counts().min()) >= 6
    assert int(filtered["item_id"].value_counts().min()) >= 6
    assert [row["step"] for row in stats][-1] == "final_filtered"
    assert len([row for row in stats if str(row["step"]).startswith("kcore_iter_")]) >= 2


def test_sample_test_users_zero_keeps_all_eligible_users() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "timestamp": 1},
            {"user_id": "u1", "item_id": "B", "timestamp": 2},
            {"user_id": "u2", "item_id": "A", "timestamp": 1},
            {"user_id": "u2", "item_id": "C", "timestamp": 2},
            {"user_id": "u3", "item_id": "D", "timestamp": 1},
            {"user_id": "u3", "item_id": "A", "timestamp": 2},
        ]
    )
    train_df, test_df = temporal_train_test_split(interactions, min_user_interactions=2)
    mappings = build_id_mappings(train_df)

    sampled = sample_test_users(test_df, mappings, test_users=0, seed=42)

    assert len(sampled) == len(test_df)
    assert sampled["user_id"].tolist() == test_df["user_id"].tolist()


def test_data_stats_records_split_coverage_and_sampling_rate() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "A", "timestamp": 1},
            {"user_id": "u1", "item_id": "B", "timestamp": 2},
            {"user_id": "u2", "item_id": "A", "timestamp": 1},
            {"user_id": "u2", "item_id": "C", "timestamp": 2},
        ]
    )
    train_df, test_df = temporal_train_test_split(interactions, min_user_interactions=2)

    stat = make_data_stat(
        "train_test_split",
        interactions,
        min_user_interactions=2,
        min_item_interactions=1,
        train_df=train_df,
        test_df=test_df,
        sampled_test=test_df,
    )

    assert stat["train_interactions"] == 2
    assert stat["test_users"] == 2
    assert stat["evaluated_users"] == 2
    assert stat["test_user_sampling_rate"] == 1.0
    assert stat["test_only_item_count"] == 2
    assert stat["cold_start_unrecallable_rate"] == 1.0
    assert stat["train_item_coverage_rate"] == 0.0
