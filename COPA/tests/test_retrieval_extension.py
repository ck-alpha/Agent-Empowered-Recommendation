import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copa import OptimizationConfig
from copa.data import UserCase
from copa.experiments.run_core import _execute
from copa.experiments.run_retrieval_extension import _load_if_matching
from copa.retrieval import (
    CANDIDATE_COLUMNS,
    TARGET_COLUMNS,
    CandidateArtifactManifest,
    PrecomputedCandidateStore,
    candidate_quality_metrics,
    controlled_hit_users,
    evaluate_temporal_popularity,
    intervene_candidate_pool,
    oracle_candidate_pool,
    sha256_file,
    validate_artifact_alignment,
)
from copa.retrieval.recbole_backend import (
    RecBoleRunSpec,
    _install_train_catalog_evaluation_mask,
    _truncate_precomputed_sequences,
    _validate_id_round_trip,
    _recbole_config,
    ensure_recbole_numpy2_compatibility,
    prepare_recbole_dataset,
    recbole_run_signature,
    training_visible_statistics,
)
from copa.retrieval.calibration import (
    _select_monotone_policies,
    calibrate_slate_policies,
)
from copa.retrieval.evaluation import _cached_task_matches, _run_methods


def _write_artifact(root: Path, *, retriever: str = "bpr") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    for user_id in ("u0", "u1"):
        for rank in range(1, 13):
            candidate_rows.append(
                {
                    "user_id": user_id,
                    "item_id": f"i{rank:02d}",
                    "raw_score": float(13 - rank),
                    "base_score": float((13 - rank) / 12),
                    "retrieval_rank": rank,
                    "retriever": retriever,
                    "backend": "recbole-1.2.1",
                    "model_seed": 42,
                }
            )
    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    targets = pd.DataFrame(
        [
            {
                "user_id": "u0",
                "target_item_id": "i03",
                "target_raw_score": 10.0,
                "target_base_score": 10 / 12,
                "target_full_rank": 3,
                "target_model_covered": True,
            },
            {
                "user_id": "u1",
                "target_item_id": "i20",
                "target_raw_score": 0.5,
                "target_base_score": 0.05,
                "target_full_rank": 20,
                "target_model_covered": True,
            },
        ]
    )
    candidate_path = root / "candidates.parquet"
    target_path = root / "targets.parquet"
    candidates.to_parquet(candidate_path, index=False)
    targets.to_parquet(target_path, index=False)
    manifest = CandidateArtifactManifest(
        dataset="fixture",
        protocol="temporal_leave_two_out",
        retriever=retriever,
        backend="recbole-1.2.1",
        model_seed=42,
        candidate_k=12,
        candidate_file=candidate_path.name,
        target_file=target_path.name,
        source_hashes={"protocol_split": "same", "interactions": "same"},
        model_config={"model": retriever},
        environment={"recbole": "1.2.1"},
        split_statistics={"mapped_users": 2, "mapped_items": 20},
        candidate_sha256=sha256_file(candidate_path),
        target_sha256=sha256_file(target_path),
        schema_version="1.0",
    )
    manifest_path = root / "manifest.json"
    manifest.write(manifest_path)
    return manifest_path


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [f"i{index:02d}" for index in range(1, 21)],
            "price_filled": [10.0] * 20,
            "brand_id": [f"b{index % 4}" for index in range(1, 21)],
        }
    )


def _write_v2_artifact(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(
        [
            {
                "user_id": "u0",
                "item_id": f"i{rank:02d}",
                "raw_score": float(6 - rank),
                "base_score": float((6 - rank) / 5),
                "retrieval_rank": rank,
                "retriever": "sasrec",
                "backend": "recbole-1.2.1",
                "model_seed": 42,
            }
            for rank in range(1, 6)
        ],
        columns=CANDIDATE_COLUMNS,
    )
    targets = pd.DataFrame(
        [
            {
                "user_id": "u0",
                "target_item_id": "i02",
                "target_order": 1,
                "target_timestamp": 100.0,
                "target_raw_score": 4.0,
                "target_base_score": 0.8,
                "target_full_rank": 2,
                "target_model_covered": True,
            },
            {
                "user_id": "u0",
                "target_item_id": "i20",
                "target_order": 2,
                "target_timestamp": 101.0,
                "target_raw_score": None,
                "target_base_score": None,
                "target_full_rank": None,
                "target_model_covered": False,
            },
        ],
        columns=TARGET_COLUMNS,
    )
    candidate_path = root / "candidates.parquet"
    target_path = root / "targets.parquet"
    candidates.to_parquet(candidate_path, index=False)
    targets.to_parquet(target_path, index=False)
    manifest = CandidateArtifactManifest(
        dataset="fixture_v2",
        protocol="frozen_train_two_test_v2",
        retriever="sasrec",
        backend="recbole-1.2.1",
        model_seed=42,
        candidate_k=5,
        candidate_file=candidate_path.name,
        target_file=target_path.name,
        source_hashes={"protocol_split": "fixture"},
        model_config={},
        environment={},
        split_statistics={"mapped_users": 1, "mapped_items": 20},
        candidate_sha256=sha256_file(candidate_path),
        target_sha256=sha256_file(target_path),
    )
    path = root / "manifest.json"
    manifest.write(path)
    return path


def test_temporal_split_deduplicates_and_leaves_last_two_out(tmp_path):
    rows = []
    for user in ("u0", "u1"):
        for step in range(5):
            rows.append({"user_id": user, "item_id": f"{user}_i{step}", "timestamp": step})
    rows.extend(
        [
            {"user_id": "u0", "item_id": "u0_i0", "timestamp": -1},
            {"user_id": "u1", "item_id": "u1_i0", "timestamp": -1},
        ]
    )
    interactions = tmp_path / "interactions.parquet"
    pd.DataFrame(rows).to_parquet(interactions, index=False)
    prepared = prepare_recbole_dataset(
        interactions, tmp_path / "atomic", dataset_name="fixture", k_core=1
    )
    split = pd.read_parquet(prepared.split_path)
    assert len(split) == 10
    for _, group in split.sort_values("timestamp").groupby("user_id"):
        assert group.iloc[-3]["split"] == "valid"
        assert list(group.iloc[-2:]["split"]) == ["test", "test"]
        assert set(group.iloc[:-3]["split"]) == {"train"}
    assert prepared.statistics["deduplicated_rows"] == 10
    dataset_dir = prepared.atomic_path.parent
    general_counts = {
        split_name: len(
            pd.read_csv(dataset_dir / f"fixture.{split_name}.inter", sep="\t")
        )
        for split_name in ("train", "valid", "test")
    }
    assert general_counts == {"train": 4, "valid": 2, "test": 4}
    sequential = {
        split_name: pd.read_csv(
            dataset_dir / f"fixture.sasrec_{split_name}.inter", sep="\t"
        )
        for split_name in ("train", "valid", "test")
    }
    assert {key: len(value) for key, value in sequential.items()} == {
        "train": 2,
        "valid": 2,
        "test": 4,
    }
    for _, group in sequential["test"].groupby("user_id:token"):
        assert group["item_id_list:token_seq"].nunique() == 1


def test_request_statistics_use_train_only(tmp_path):
    rows = [
        {"user_id": "u0", "item_id": f"i{step}", "timestamp": step}
        for step in range(5)
    ]
    interactions = tmp_path / "interactions.parquet"
    items = tmp_path / "items.parquet"
    pd.DataFrame(rows).to_parquet(interactions, index=False)
    pd.DataFrame(
        {
            "item_id": [f"i{step}" for step in range(5)],
            "price_filled": [10.0, 20.0, 10_000.0, 20_000.0, 30_000.0],
        }
    ).to_parquet(items, index=False)
    prepared = prepare_recbole_dataset(
        interactions, tmp_path / "atomic", dataset_name="fixture", k_core=1
    )
    popularity, histories, budgets = training_visible_statistics(prepared, items)
    assert histories["u0"] == {"i0", "i1"}
    assert set(popularity) == {"i0", "i1"}
    assert budgets["u0"] == pytest.approx(18.0)


def test_checkpoint_selection_masks_non_train_catalog_items():
    import torch

    class Dataset:
        iid_field = "item_id"
        item_num = 4
        field2token_id = {
            "item_id": {"[PAD]": 0, "train_a": 1, "valid_only": 2, "train_b": 3}
        }

    class Trainer:
        device = torch.device("cpu")

        def _full_sort_batch_eval(self, batched_data):
            return "interaction", torch.tensor([[0.0, 3.0, 9.0, 2.0]]), "u", "i"

    trainer = Trainer()
    _install_train_catalog_evaluation_mask(
        trainer, Dataset(), {"train_a", "train_b"}
    )
    _, scores, _, _ = trainer._full_sort_batch_eval(None)
    assert torch.isneginf(scores[0, 0])
    assert torch.isneginf(scores[0, 2])
    assert scores[0, 1].item() == 3.0
    assert scores[0, 3].item() == 2.0


def test_explicit_sasrec_histories_are_recently_truncated_to_trial_maxlen():
    class Dataset:
        iid_field = "item_id"
        item_id_list_field = "item_id_list"
        item_list_length_field = "item_length"
        field2id_token = {
            "item_id": np.asarray(["[PAD]", "a"]),
            "item_id_list": np.asarray(["[PAD]", "a"]),
        }
        inter_feat = pd.DataFrame(
            {
                "item_id_list": [np.asarray([1, 2]), np.arange(1, 8)],
                "item_length": [2, 7],
            }
        )

    dataset = Dataset()
    statistics = _truncate_precomputed_sequences(dataset, 4)
    assert statistics == {
        "sasrec_sequence_length_before_truncation": 7,
        "sasrec_sequence_length_after_truncation": 4,
    }
    assert dataset.inter_feat.iloc[1]["item_id_list"].tolist() == [4, 5, 6, 7]


def test_validation_slate_calibration_fails_closed_when_three_tiers_are_unreachable(
    tmp_path,
):
    users = ("u0", "u1", "u2")
    candidates = pd.DataFrame(
        [
            {
                "user_id": user,
                "item_id": f"i{item}",
                "retrieval_rank": item + 1,
            }
            for user in users
            for item in range(6)
        ]
    )
    split = pd.DataFrame(
        [
            {"user_id": user, "item_id": item_id, "timestamp": index, "split": "train"}
            for user in users
            for index, item_id in enumerate(("h0", "h1"))
        ]
    )
    items = pd.DataFrame(
        {
            "item_id": ["h0", "h1", *[f"i{item}" for item in range(6)]],
            "price_filled": [10.0] * 8,
            "brand_id": ["history", "history", "a", "b", "c", "a", "b", "c"],
            "main_category": ["Beauty"] * 8,
        }
    )
    candidate_path = tmp_path / "validation_candidates.parquet"
    split_path = tmp_path / "split.parquet"
    item_path = tmp_path / "items.parquet"
    candidates.to_parquet(candidate_path, index=False)
    split.to_parquet(split_path, index=False)
    items.to_parquet(item_path, index=False)
    calibrated = calibrate_slate_policies(
        candidate_path,
        split_path,
        item_path,
        tmp_path / "calibration",
        dataset_name="beauty_5core",
        candidate_k=6,
        top_k=3,
        alpha_grid=(0.8, 1.0, 1.5),
        brand_cap_grid=(1, 2),
        category_grid=(2,),
    )
    assert calibrated["status"] == "failed"
    assert calibrated["policies"] == {}
    assert "target intervals" in calibrated["failure_reason"]
    assert (tmp_path / "calibration" / "slate_calibration_search.csv").exists()


def test_calibration_policy_selection_requires_distinct_monotone_target_hits():
    search = pd.DataFrame(
        [
            {
                "total_budget_alpha": 1.25,
                "brand_cap": 4,
                "full_slate_feasible_rate": 0.94,
                "solver_unknown_rate": 0.0,
            },
            {
                "total_budget_alpha": 0.85,
                "brand_cap": 2,
                "full_slate_feasible_rate": 0.78,
                "solver_unknown_rate": 0.0,
            },
            {
                "total_budget_alpha": 0.55,
                "brand_cap": 1,
                "full_slate_feasible_rate": 0.55,
                "solver_unknown_rate": 0.0,
            },
        ]
    )
    policies, error = _select_monotone_policies(search, electronics=False)
    assert error is None
    assert list(policies) == ["loose", "medium", "tight"]
    assert len(
        {
            (policy["total_budget_alpha"], policy["brand_cap"])
            for policy in policies.values()
        }
    ) == 3
    assert all(policy["within_target_interval"] for policy in policies.values())


def test_resume_rejects_a_different_export_user_cohort(tmp_path):
    rows = [
        {"user_id": user, "item_id": f"{user}_i{step}", "timestamp": step}
        for user in ("u0", "u1")
        for step in range(5)
    ]
    interactions = tmp_path / "interactions.parquet"
    pd.DataFrame(rows).to_parquet(interactions, index=False)
    prepared = prepare_recbole_dataset(
        interactions, tmp_path / "atomic", dataset_name="fixture", k_core=1
    )
    spec = RecBoleRunSpec(model="BPR", retriever="bpr", seed=42)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({"user_id": ["u0", "u1"]}).to_parquet(
        run_dir / "targets.parquet", index=False
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        '{"target_file": "targets.parquet"}\n', encoding="utf-8"
    )
    result_path = run_dir / "run_result.json"
    result_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "run_signature": recbole_run_signature(
                    prepared, spec, validation_only=False
                ),
            }
        ),
        encoding="utf-8",
    )

    assert _load_if_matching(
        result_path,
        prepared,
        spec,
        validation_only=False,
        resume=True,
        export_users={"u0", "u1"},
    ) is not None
    assert _load_if_matching(
        result_path,
        prepared,
        spec,
        validation_only=False,
        resume=True,
        export_users={"u0"},
    ) is None


@pytest.mark.parametrize(
    "model,retriever",
    [("BPR", "bpr"), ("ItemKNN", "itemknn"), ("SASRec", "sasrec")],
)
def test_recbole_models_consume_explicit_frozen_benchmarks(
    tmp_path, model, retriever
):
    rows = [
        {"user_id": user, "item_id": f"i{(user_index + step) % 8}", "timestamp": step}
        for user_index, user in enumerate(("u0", "u1", "u2"))
        for step in range(5)
    ]
    interactions = tmp_path / "interactions.parquet"
    pd.DataFrame(rows).to_parquet(interactions, index=False)
    prepared = prepare_recbole_dataset(
        interactions, tmp_path / "atomic", dataset_name="fixture", k_core=1
    )
    spec = RecBoleRunSpec(
        model=model,
        retriever=retriever,
        epochs=1,
        model_parameters=(
            {
                "n_layers": 1,
                "n_heads": 1,
                "hidden_size": 8,
                "inner_size": 16,
                "MAX_ITEM_LIST_LENGTH": 10,
            }
            if model == "SASRec"
            else {}
        ),
    )
    ensure_recbole_numpy2_compatibility()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    config_dict = _recbole_config(prepared, spec, tmp_path / retriever)
    config_dict["use_gpu"] = False
    config = Config(model=model, dataset=prepared.name, config_dict=config_dict)
    dataset = create_dataset(config)
    if model == "SASRec":
        assert dataset.field2id_token[dataset.iid_field] is dataset.field2id_token[
            dataset.item_id_list_field
        ]
        assert dataset.field2token_id[dataset.iid_field] is dataset.field2token_id[
            dataset.item_id_list_field
        ]
    train, valid, test = data_preparation(config, dataset)
    assert len(train.dataset) > 0
    assert len(valid.dataset) == 3
    assert len(test.dataset) in {3, 6}


def test_temporal_popularity_uses_only_history_visible_at_each_stage(tmp_path):
    split = pd.DataFrame(
        [
            {"user_id": "u0", "item_id": "a", "split": "train"},
            {"user_id": "u0", "item_id": "b", "split": "valid"},
            {"user_id": "u0", "item_id": "c", "split": "test"},
            {"user_id": "u0", "item_id": "x", "split": "test"},
            {"user_id": "u1", "item_id": "a", "split": "train"},
            {"user_id": "u1", "item_id": "c", "split": "valid"},
            {"user_id": "u1", "item_id": "d", "split": "test"},
            {"user_id": "u1", "item_id": "y", "split": "test"},
            {"user_id": "u2", "item_id": "b", "split": "train"},
            {"user_id": "u2", "item_id": "c", "split": "valid"},
            {"user_id": "u2", "item_id": "e", "split": "test"},
            {"user_id": "u2", "item_id": "z", "split": "test"},
        ]
    )
    split_path = tmp_path / "split.parquet"
    split.to_parquet(split_path, index=False)
    result = evaluate_temporal_popularity(split_path, candidate_ks=(1, 2))

    test_rows = result[result["stage"] == "test"].set_index("candidate_k")
    assert test_rows.loc[1, "candidate_recall"] == 0.0
    assert test_rows.loc[2, "candidate_recall"] == 0.0
    assert test_rows.loc[1, "model_target_coverage"] == 0.0


def test_artifact_contract_target_mapping_seen_exclusion_and_hash(tmp_path):
    manifest_path = _write_artifact(tmp_path / "bpr")
    store = PrecomputedCandidateStore(
        manifest_path.parent / "candidates.parquet",
        manifest_path=manifest_path,
        item_catalog=_catalog(),
        popularity={"i01": 0.4},
    )
    records = store.load("u0", 10)
    assert len(records) == 10
    assert records[0].metadata["popularity"] == pytest.approx(0.4)
    assert records[1].metadata["popularity"] == 0.0
    assert store.target_record("u1").item_id == "i20"
    assert store.target_record("u1").base_score == pytest.approx(0.05)
    with pytest.raises(ValueError, match="seen items"):
        store.load("u0", 10, seen_items={"i01"})

    inconsistent_target_manifest = _write_artifact(tmp_path / "target_mismatch")
    inconsistent_target_path = inconsistent_target_manifest.parent / "targets.parquet"
    inconsistent_targets = pd.read_parquet(inconsistent_target_path)
    inconsistent_targets.loc[0, "target_base_score"] = 0.1
    inconsistent_targets.to_parquet(inconsistent_target_path, index=False)
    with pytest.raises(ValueError, match="Target score disagrees"):
        PrecomputedCandidateStore(
            inconsistent_target_manifest.parent / "candidates.parquet",
            manifest_path=inconsistent_target_manifest,
            verify_hashes=False,
        )

    candidate_path = manifest_path.parent / "candidates.parquet"
    frame = pd.read_parquet(candidate_path)
    frame.assign(untrusted_extra=1).to_parquet(candidate_path, index=False)
    with pytest.raises(ValueError, match="schema/order"):
        PrecomputedCandidateStore(
            candidate_path, manifest_path=manifest_path, verify_hashes=False
        )

    inconsistent = frame.copy()
    inconsistent.loc[0, "raw_score"] = -1.0
    inconsistent.to_parquet(candidate_path, index=False)
    with pytest.raises(ValueError, match="inconsistent with raw_score"):
        PrecomputedCandidateStore(
            candidate_path, manifest_path=manifest_path, verify_hashes=False
        )

    frame.loc[0, "base_score"] = 0.0
    frame.to_parquet(candidate_path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        PrecomputedCandidateStore(
            candidate_path, manifest_path=manifest_path
        )


def test_artifact_v2_multi_target_and_cold_target_null_contract(tmp_path):
    manifest_path = _write_v2_artifact(tmp_path / "v2")
    store = PrecomputedCandidateStore(
        manifest_path.parent / "candidates.parquet",
        manifest_path=manifest_path,
        item_catalog=_catalog(),
    )
    rows = store.target_rows("u0")
    assert [row["target_order"] for row in rows] == [1, 2]
    assert len(store.scored_target_records("u0")) == 1
    with pytest.raises(ValueError, match="2 positives"):
        store.target_record("u0")

    targets = pd.read_parquet(manifest_path.parent / "targets.parquet")
    targets.loc[1, "target_raw_score"] = 0.1
    targets.to_parquet(manifest_path.parent / "targets.parquet", index=False)
    with pytest.raises(ValueError, match="Model-uncovered targets"):
        PrecomputedCandidateStore(
            manifest_path.parent / "candidates.parquet",
            manifest_path=manifest_path,
            verify_hashes=False,
        )


def test_oracle_and_controlled_recall_are_exact_without_score_boost(tmp_path):
    manifest_path = _write_artifact(tmp_path / "bpr")
    store = PrecomputedCandidateStore(
        manifest_path.parent / "candidates.parquet",
        manifest_path=manifest_path,
        item_catalog=_catalog(),
        popularity={},
    )
    reservoir = store.load("u1", 12)
    target = store.target_record("u1")
    oracle = oracle_candidate_pool(reservoir, target, 10)
    inserted = next(candidate for candidate in oracle if candidate.item_id == "i20")
    assert inserted.base_score == target.base_score == pytest.approx(0.05)
    assert len(oracle) == 10
    miss = intervene_candidate_pool(reservoir, target, 10, include_target=False)
    assert len(miss) == 10 and "i20" not in {candidate.item_id for candidate in miss}

    users = [f"u{index:03d}" for index in range(100)]
    for level, expected in ((0.1, 10), (0.3, 30), (0.5, 50), (0.7, 70), (0.9, 90), (1.0, 100)):
        selected = controlled_hit_users(users, level, seed=42)
        assert len(selected) == expected
        assert selected == controlled_hit_users(reversed(users), level, seed=42)


def test_artifact_to_candidate_to_copa_and_verifier(tmp_path):
    manifest_path = _write_artifact(tmp_path / "bpr")
    store = PrecomputedCandidateStore(
        manifest_path.parent / "candidates.parquet",
        manifest_path=manifest_path,
        item_catalog=_catalog(),
        popularity={f"i{index:02d}": index / 20 for index in range(1, 21)},
    )
    candidates = store.load("u0", 12)
    case = UserCase(
        user_id="u0",
        candidates=candidates,
        constraints=[],
        relevant_items=["i03"],
        context={"source": "artifact-test"},
    )
    result = _execute(
        case,
        "retrieval_test",
        "copa",
        OptimizationConfig(top_k=10, population_size=12, generations=2, seed=42),
        tmp_path / "traces",
        [1 / 3, 1 / 3, 1 / 3],
    )
    assert result.verification.feasible
    assert len(result.item_ids) == 10
    metrics = candidate_quality_metrics(candidates, ["i03"], requested_k=12)
    assert metrics["candidate_recall"] == 1.0
    assert math.isfinite(metrics["candidate_ndcg"])

    loss_case = UserCase(
        user_id="u0",
        candidates=candidates,
        constraints=[],
        relevant_items=["i12"],
        context={"source": "loss-decomposition-test"},
    )
    rows, _ = _run_methods(
        loss_case,
        condition="real",
        retrieval_metadata={
            "dataset": "fixture",
            "retriever": "bpr",
            "backend": "recbole-1.2.1",
            "model_seed": 42,
            "candidate_k": 12,
            "candidate_recall": 1.0,
            "target_in_feasible_domain": 1.0,
            "retrieval_loss": 0.0,
            "constraint_filter_loss": 0.0,
        },
        optimization_config={
            "optimization": {
                "top_k": 10,
                "population_size": 12,
                "generations": 2,
                "crossover_rate": 0.9,
                "mutation_rate": 0.15,
                "tournament_size": 2,
            }
        },
        seeds=(42,),
        methods=("feasible_relevance",),
        trace_dir=tmp_path / "loss_traces",
        weights=(1 / 3, 1 / 3, 1 / 3),
        hv_sample_power=6,
    )
    assert rows[0]["recommendation_hit"] == 0.0
    assert rows[0]["retrieval_loss"] == 0.0
    assert rows[0]["constraint_filter_loss"] == 0.0
    assert rows[0]["ranking_loss"] == 1.0

    legacy_cache = {
        "signature": "v1",
        "rows": [dict(rows[0])],
        "fronts": [{"user_id": "u0"}],
    }
    metadata = {
        key: rows[0][key]
        for key in (
            "candidate_recall",
            "target_in_feasible_domain",
            "retrieval_loss",
            "constraint_filter_loss",
        )
    }
    assert _cached_task_matches(
        legacy_cache,
        signature="v2",
        legacy_signature="v1",
        retrieval_metadata=metadata,
    )
    changed = dict(metadata, candidate_recall=0.0)
    assert not _cached_task_matches(
        legacy_cache,
        signature="v2",
        legacy_signature="v1",
        retrieval_metadata=changed,
    )


def test_artifact_alignment_and_recbole_id_round_trip(tmp_path):
    first = _write_artifact(tmp_path / "bpr", retriever="bpr")
    second = _write_artifact(tmp_path / "itemknn", retriever="itemknn")
    aligned = validate_artifact_alignment([first, second])
    assert aligned["users"] == 2
    assert aligned["mapped_items"] == 20

    class FakeDataset:
        uid_field = "user_id"
        iid_field = "item_id"
        field2id_token = {
            "user_id": np.array(["[PAD]", "raw-u"]),
            "item_id": np.array(["[PAD]", "raw-i"]),
        }
        field2token_id = {
            "user_id": {"[PAD]": 0, "raw-u": 1},
            "item_id": {"[PAD]": 0, "raw-i": 1},
        }

        def id2token(self, field, internal):
            return self.field2id_token[field][internal]

    assert _validate_id_round_trip(FakeDataset()) == {
        "mapped_users": 1,
        "mapped_items": 1,
    }
