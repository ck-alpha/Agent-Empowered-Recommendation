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
    _validate_id_round_trip,
    prepare_recbole_dataset,
    recbole_run_signature,
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
        assert group.iloc[-2]["split"] == "valid"
        assert group.iloc[-1]["split"] == "test"
        assert set(group.iloc[:-2]["split"]) == {"train"}
    assert prepared.statistics["deduplicated_rows"] == 10


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


def test_temporal_popularity_uses_only_history_visible_at_each_stage(tmp_path):
    split = pd.DataFrame(
        [
            {"user_id": "u0", "item_id": "a", "split": "train"},
            {"user_id": "u0", "item_id": "b", "split": "valid"},
            {"user_id": "u0", "item_id": "c", "split": "test"},
            {"user_id": "u1", "item_id": "a", "split": "train"},
            {"user_id": "u1", "item_id": "c", "split": "valid"},
            {"user_id": "u1", "item_id": "d", "split": "test"},
            {"user_id": "u2", "item_id": "b", "split": "train"},
            {"user_id": "u2", "item_id": "c", "split": "valid"},
            {"user_id": "u2", "item_id": "e", "split": "test"},
        ]
    )
    split_path = tmp_path / "split.parquet"
    split.to_parquet(split_path, index=False)
    result = evaluate_temporal_popularity(split_path, candidate_ks=(1, 2))

    test_rows = result[result["stage"] == "test"].set_index("candidate_k")
    assert test_rows.loc[1, "candidate_recall"] == pytest.approx(1 / 3)
    assert test_rows.loc[2, "candidate_recall"] == pytest.approx(2 / 3)
    assert test_rows.loc[1, "model_target_coverage"] == pytest.approx(1 / 3)


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
