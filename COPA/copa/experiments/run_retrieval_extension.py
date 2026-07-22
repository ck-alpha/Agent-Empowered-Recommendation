"""Prepare, tune, and export mature RecBole retrieval candidates for COPA."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from copa.retrieval.recbole_backend import (
    RecBoleRunSpec,
    load_prepared_dataset,
    prepare_recbole_dataset,
    recbole_run_signature,
    train_recbole_and_export,
)
from copa.retrieval.evaluation import (
    FORMAL_END_TO_END_METHODS,
    OPTIMIZER_KERNEL_VERSION,
    _paired_user_inference,
    run_artifact_evaluation,
)
from copa.retrieval.popularity_baseline import evaluate_temporal_popularity
from copa.retrieval.artifacts import (
    CandidateArtifactManifest,
    sha256_file,
    validate_artifact_alignment,
)
from copa.retrieval.calibration import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_BRAND_CAP_GRID,
    DEFAULT_CATEGORY_GRID,
    attach_calibration_hash,
    calibrate_slate_policies,
)


FORMAL_MODELS = {"bpr": "BPR", "itemknn": "ItemKNN", "sasrec": "SASRec"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COPA mature retrieval extension")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--interactions", required=True)
    prepare.add_argument("--items", default=None)
    prepare.add_argument("--dataset-name", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--k-core", type=int, default=5)

    train = subparsers.add_parser("train")
    train.add_argument("--data-root", required=True)
    train.add_argument("--dataset-name", required=True)
    train.add_argument("--source-interactions", required=True)
    train.add_argument("--source-items", default=None)
    train.add_argument("--model", choices=sorted(FORMAL_MODELS), required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--stopping-step", type=int, default=10)
    train.add_argument("--candidate-k", type=int, default=500)
    train.add_argument("--train-batch-size", type=int, default=2048)
    train.add_argument("--eval-batch-size", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--model-parameters", default="{}")
    train.add_argument("--export-users", default=None, help="Optional text file with one user ID per line")
    train.add_argument(
        "--export-validation-candidates",
        action="store_true",
        help="Export the frozen-train validation query pool for calibration",
    )

    suite = subparsers.add_parser("suite")
    suite.add_argument("--config", required=True)
    suite.add_argument("--output-dir", required=True)
    suite.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--items", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--candidate-ks", default="50,100,200,500")
    evaluate.add_argument("--end-to-end-candidate-k", type=int, default=100)
    evaluate.add_argument("--optimizer-seeds", default="42,43,44")
    evaluate.add_argument("--population-size", type=int, default=100)
    evaluate.add_argument("--generations", type=int, default=50)
    evaluate.add_argument("--methods", default=",".join(FORMAL_END_TO_END_METHODS))
    evaluate.add_argument("--controlled-levels", default="0.1,0.3,0.5,0.7,0.9,1.0")
    evaluate.add_argument("--interventions", action="store_true")
    evaluate.add_argument("--recall-only", action="store_true")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--slate-preflight-time-limit-seconds", type=float, default=2.0)
    evaluate.add_argument("--opportunity-solver-time-limit-seconds", type=float, default=5.0)
    evaluate.add_argument("--optimizer-time-limit-seconds", type=float, default=120.0)
    evaluate.add_argument("--optimizer-kernel-version", type=int, default=OPTIMIZER_KERNEL_VERSION)
    evaluate.add_argument("--calibration-sha256", default="")

    evaluate_suite = subparsers.add_parser("evaluate-suite")
    evaluate_suite.add_argument("--suite-dir", required=True)
    evaluate_suite.add_argument("--workers", type=int, default=4)
    evaluate_suite.add_argument("--resume", action="store_true")
    evaluate_suite.add_argument("--skip-interventions", action="store_true")
    evaluate_suite.add_argument("--evaluation-output-dir", default=None)
    evaluate_suite.add_argument("--recalibrate", action="store_true")
    evaluate_suite.add_argument("--calibration-only", action="store_true")
    evaluate_suite.add_argument("--recall-only", action="store_true")
    evaluate_suite.add_argument("--skip-recall", action="store_true")

    pilot = subparsers.add_parser("performance-pilot")
    pilot.add_argument("--suite-dir", required=True)
    pilot.add_argument("--evaluation-output-dir", required=True)
    pilot.add_argument("--users-per-dataset", type=int, default=8)
    pilot.add_argument("--workers", type=int, default=12)
    pilot.add_argument("--resume", action="store_true")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _spec(
    model_key: str,
    payload: Mapping[str, Any],
    *,
    seed: int,
    defaults: Mapping[str, Any],
) -> RecBoleRunSpec:
    parameters = dict(payload.get("parameters", {}))
    return RecBoleRunSpec(
        model=FORMAL_MODELS[model_key],
        retriever=model_key,
        seed=int(seed),
        epochs=int(payload.get("epochs", defaults.get("epochs", 100))),
        stopping_step=int(payload.get("stopping_step", defaults.get("stopping_step", 10))),
        max_candidate_k=int(defaults.get("max_candidate_k", 500)),
        train_batch_size=int(payload.get("train_batch_size", defaults.get("train_batch_size", 2048))),
        eval_batch_size=int(payload.get("eval_batch_size", defaults.get("eval_batch_size", 4096))),
        learning_rate=float(payload.get("learning_rate", defaults.get("learning_rate", 1e-3))),
        model_parameters=parameters,
    )


def _trial_payload(
    model_key: str,
    model_config: Mapping[str, Any],
    trial: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {key: value for key, value in model_config.items() if key not in {"trials", "parameters"}}
    payload.update({key: value for key, value in trial.items() if key != "parameters"})
    parameters = dict(model_config.get("parameters", {}))
    parameters.update(dict(trial.get("parameters", {})))
    override = dict(dataset_config.get("model_overrides", {}).get(model_key, {}))
    parameters.update(dict(override.pop("parameters", {})))
    payload.update(override)
    payload["parameters"] = parameters
    return payload


def _load_if_matching(
    result_path: Path,
    prepared,
    spec: RecBoleRunSpec,
    *,
    validation_only: bool,
    resume: bool,
    export_users: set[str] | None = None,
    export_validation_candidates: bool = False,
) -> Dict[str, Any] | None:
    if not resume or not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = recbole_run_signature(
        prepared,
        spec,
        validation_only=validation_only,
        export_validation_candidates=export_validation_candidates,
    )
    if result.get("run_signature") != expected:
        return None
    if validation_only or export_users is None:
        return result

    # The export cohort does not affect model training, so it deliberately is not
    # part of the model run signature. It does affect the candidate artifact,
    # however, and must be checked independently before a final run is resumed.
    manifest_value = result.get("manifest")
    if not manifest_value:
        return None
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = result_path.parent / manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target_path = manifest_path.parent / str(manifest["target_file"])
        artifact_users = set(
            pd.read_parquet(target_path, columns=["user_id"])["user_id"].astype(str)
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return result if artifact_users == {str(value) for value in export_users} else None


def _load_export_users(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _validation_selection_key(result: Mapping[str, Any]) -> tuple[float, float, float]:
    """Validation NDCG primary, with Hit/HR and MRR as declared tie-breakers."""

    validation = dict(result.get("best_valid_result", {}))
    ndcg = float(result["best_valid_score"])
    hit_values = [float(value) for key, value in validation.items() if str(key).startswith("hit@")]
    mrr_values = [float(value) for key, value in validation.items() if str(key).startswith("mrr@")]
    return ndcg, max(hit_values, default=float("-inf")), max(mrr_values, default=float("-inf"))


def _ensure_checkpoint_hash(result: Mapping[str, Any]) -> None:
    """Backfill checkpoint hashes when resuming artifacts made by an older run."""

    checkpoint = Path(str(result["checkpoint"]))
    manifest_path = Path(str(result["manifest"]))
    manifest = CandidateArtifactManifest.read(manifest_path)
    expected = sha256_file(checkpoint)
    if manifest.source_hashes.get("checkpoint") == expected:
        return
    hashes = dict(manifest.source_hashes)
    hashes["checkpoint"] = expected
    replace(manifest, source_hashes=hashes).write(manifest_path)


def _write_smoke_tables(output_dir: Path) -> tuple[Path, Path]:
    rows = []
    item_count = 24
    for user_index in range(12):
        for step in range(8):
            rows.append(
                {
                    "user_id": f"u{user_index:02d}",
                    "item_id": f"i{(user_index * 3 + step * 2) % item_count:03d}",
                    "timestamp": user_index * 100 + step,
                }
            )
    interactions = output_dir / "smoke_interactions.parquet"
    items = output_dir / "smoke_items.parquet"
    pd.DataFrame(rows).to_parquet(interactions, index=False)
    pd.DataFrame(
        {
            "item_id": [f"i{index:03d}" for index in range(item_count)],
            "price_filled": np.linspace(5.0, 50.0, item_count),
            "brand_id": [f"brand_{index % 5}" for index in range(item_count)],
        }
    ).to_parquet(items, index=False)
    return interactions, items


def run_smoke(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions, items = _write_smoke_tables(output_dir)
    prepared = prepare_recbole_dataset(
        interactions,
        output_dir / "atomic",
        dataset_name="copa_retrieval_smoke",
        k_core=1,
        items_path=items,
        protocol="smoke_to_ls",
    )
    test_users = sorted(
        pd.read_parquet(prepared.split_path)
        .query("split == 'test'")["user_id"]
        .astype(str)
        .unique()
    )[:4]
    specs = {
        "bpr": RecBoleRunSpec(
            "BPR", "bpr", epochs=2, stopping_step=2, max_candidate_k=20,
            train_batch_size=64, eval_batch_size=1024, model_parameters={"embedding_size": 16},
        ),
        "itemknn": RecBoleRunSpec(
            "ItemKNN", "itemknn", epochs=1, stopping_step=1, max_candidate_k=20,
            train_batch_size=64, eval_batch_size=1024, model_parameters={"k": 5, "shrink": 0.0},
        ),
        "sasrec": RecBoleRunSpec(
            "SASRec", "sasrec", epochs=2, stopping_step=2, max_candidate_k=20,
            train_batch_size=32, eval_batch_size=32,
            model_parameters={
                "n_layers": 1, "n_heads": 1, "hidden_size": 16, "inner_size": 32,
                "hidden_dropout_prob": 0.2, "attn_dropout_prob": 0.2, "loss_type": "CE",
                "MAX_ITEM_LIST_LENGTH": 10,
            },
        ),
    }
    results: Dict[str, Any] = {}
    for key, spec in specs.items():
        results[key] = train_recbole_and_export(
            prepared,
            spec,
            output_dir / key,
            export_users=test_users,
            export_validation_candidates=(key == "sasrec"),
        )
    determinism: Dict[str, Any] = {}
    for key, spec in specs.items():
        repeat_dir = output_dir / f"{key}_same_seed_repeat"
        repeat = train_recbole_and_export(
            prepared,
            spec,
            repeat_dir,
            export_users=test_users,
            export_validation_candidates=(key == "sasrec"),
        )
        for filename in ("candidates.parquet", "targets.parquet"):
            first_frame = pd.read_parquet(output_dir / key / filename)
            repeat_frame = pd.read_parquet(repeat_dir / filename)
            pd.testing.assert_frame_equal(first_frame, repeat_frame, check_exact=True)
        determinism[key] = {
            "model": spec.model,
            "seed": spec.seed,
            "candidate_sha256": sha256_file(output_dir / key / "candidates.parquet"),
            "repeat_candidate_sha256": sha256_file(repeat_dir / "candidates.parquet"),
            "target_sha256": sha256_file(output_dir / key / "targets.parquet"),
            "repeat_target_sha256": sha256_file(repeat_dir / "targets.parquet"),
            "exact_frames_equal": True,
            "repeat_run": repeat,
        }
    results["same_seed_determinism"] = determinism
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def run_suite(config_path: Path, output_dir: Path, resume: bool) -> Dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    defaults = config.get("defaults", {})
    tuning_seed = int(config.get("tuning_seed", 42))
    final_seeds = [int(value) for value in config.get("final_seeds", [42, 43, 44])]
    export_cohort_size = int(config.get("export_cohort_size", 0))
    summary_rows = []
    popularity_frames = []
    selected: Dict[str, Any] = {}
    for dataset_config in config.get("datasets", []):
        dataset_name = str(dataset_config["name"])
        dataset_dir = output_dir / dataset_name
        protocol = str(dataset_config.get("protocol", "deduplicated_5core_to_ls"))
        k_core = int(dataset_config.get("k_core", 5))
        prepared = None
        if resume:
            try:
                cached = load_prepared_dataset(
                    dataset_dir / "atomic",
                    dataset_name,
                    source_interactions=dataset_config["interactions"],
                    source_items=dataset_config.get("items"),
                    protocol=protocol,
                )
                if (
                    cached.statistics.get("protocol") == protocol
                    and int(cached.statistics.get("k_core", -1)) == k_core
                    and cached.statistics.get("source_interactions_sha256")
                    == sha256_file(dataset_config["interactions"])
                    and (
                        not dataset_config.get("items")
                        or cached.statistics.get("source_items_sha256")
                        == sha256_file(dataset_config["items"])
                    )
                ):
                    prepared = cached
            except FileNotFoundError:
                pass
        if prepared is None:
            prepared = prepare_recbole_dataset(
                dataset_config["interactions"],
                dataset_dir / "atomic",
                dataset_name=dataset_name,
                k_core=k_core,
                items_path=dataset_config.get("items"),
                protocol=protocol,
            )
        all_test_users = sorted(
            pd.read_parquet(prepared.split_path)
            .query("split == 'test'")["user_id"]
            .astype(str)
            .unique()
        )
        rng = np.random.default_rng(int(config.get("cohort_seed", 42)))
        test_users = list(all_test_users)
        if export_cohort_size > 0 and len(test_users) > export_cohort_size:
            test_users = sorted(
                rng.choice(test_users, size=export_cohort_size, replace=False).tolist()
            )
        downstream_size = int(
            dataset_config.get("downstream_cohort_size", len(all_test_users))
        )
        downstream_users = list(all_test_users)
        if downstream_size > 0 and len(downstream_users) > downstream_size:
            downstream_users = sorted(
                rng.choice(
                    downstream_users, size=downstream_size, replace=False
                ).tolist()
            )
        (dataset_dir / "export_users.txt").write_text(
            "\n".join(test_users) + "\n", encoding="utf-8"
        )
        (dataset_dir / "downstream_users.txt").write_text(
            "\n".join(downstream_users) + "\n", encoding="utf-8"
        )
        popularity_ks = (
            (10, 50, 100, 200, 500)
            if int(prepared.statistics.get("items", 0)) >= 500
            else (10, 50, 100, 200)
        )
        popularity = evaluate_temporal_popularity(
            prepared.split_path, candidate_ks=popularity_ks
        )
        popularity.insert(0, "dataset", dataset_name)
        popularity.to_csv(dataset_dir / "popularity_full_sort.csv", index=False)
        popularity_frames.append(popularity)
        selected[dataset_name] = {}
        final_manifests: Dict[int, list[str]] = {seed: [] for seed in final_seeds}
        for model_key, model_config in config.get("models", {}).items():
            if model_key not in FORMAL_MODELS:
                raise ValueError(f"Unsupported formal retrieval model: {model_key}")
            trials = list(model_config.get("trials", [{}]))
            trial_results = []
            for trial_index, trial in enumerate(trials):
                trial_dir = dataset_dir / "tuning" / model_key / f"trial_{trial_index:03d}"
                result_path = trial_dir / "run_result.json"
                effective_trial = _trial_payload(
                    model_key, model_config, trial, dataset_config
                )
                trial_spec = _spec(
                    model_key, effective_trial, seed=tuning_seed, defaults=defaults
                )
                result = _load_if_matching(
                    result_path,
                    prepared,
                    trial_spec,
                    validation_only=True,
                    resume=resume,
                )
                if result is None:
                    result = train_recbole_and_export(
                        prepared,
                        trial_spec,
                        trial_dir,
                        export_users=test_users,
                        validation_only=True,
                    )
                result["trial_index"] = trial_index
                result["trial"] = trial
                trial_results.append(result)
            best = max(trial_results, key=_validation_selection_key)
            selected[dataset_name][model_key] = {
                "trial_index": best["trial_index"],
                "trial": best["trial"],
                "best_valid_score": best["best_valid_score"],
                "selection_key": list(_validation_selection_key(best)),
                "selection_rule": "validation NDCG, then Hit/HR, then MRR; test is never read",
            }
            for seed in final_seeds:
                final_dir = dataset_dir / "final" / model_key / f"seed_{seed}"
                result_path = final_dir / "run_result.json"
                effective_best = _trial_payload(
                    model_key, model_config, best["trial"], dataset_config
                )
                final_spec = _spec(
                    model_key, effective_best, seed=seed, defaults=defaults
                )
                export_validation = model_key == "sasrec" and seed == 42
                result = _load_if_matching(
                    result_path,
                    prepared,
                    final_spec,
                    validation_only=False,
                    resume=resume,
                    export_users=set(test_users),
                    export_validation_candidates=export_validation,
                )
                if result is None:
                    result = train_recbole_and_export(
                        prepared,
                        final_spec,
                        final_dir,
                        export_users=test_users,
                        export_validation_candidates=export_validation,
                    )
                _ensure_checkpoint_hash(result)
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_key,
                        "seed": seed,
                        "best_valid_score": result["best_valid_score"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "manifest": result["manifest"],
                        **{f"test_{key}": value for key, value in result["test_result"].items()},
                    }
                )
                final_manifests[seed].append(result["manifest"])
        selected[dataset_name]["alignment"] = {
            str(seed): (
                validate_artifact_alignment(paths)
                if len(paths) >= 2
                else {"artifacts": len(paths), "validated": False}
            )
            for seed, paths in final_manifests.items()
        }
        calibration_config = dict(config.get("slate_calibration", {}))
        if bool(calibration_config.get("enabled", True)):
            validation_candidates = (
                dataset_dir
                / "final"
                / "sasrec"
                / "seed_42"
                / "validation_candidates"
                / "candidates.parquet"
            )
            if not validation_candidates.exists():
                raise FileNotFoundError(
                    f"SASRec seed-42 validation candidates are required: {validation_candidates}"
                )
            calibration = calibrate_slate_policies(
                validation_candidates,
                prepared.split_path,
                dataset_config["items"],
                dataset_dir / "slate_calibration",
                dataset_name=dataset_name,
                user_ids=downstream_users,
                candidate_k=int(calibration_config.get("candidate_k", 100)),
                top_k=int(calibration_config.get("top_k", 10)),
                solver_time_limit_seconds=float(
                    calibration_config.get("solver_time_limit_seconds", 2.0)
                ),
                alpha_grid=tuple(
                    map(
                        float,
                        calibration_config.get(
                            "alpha_grid", [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
                        ),
                    )
                ),
                brand_cap_grid=tuple(
                    map(
                        int,
                        calibration_config.get("brand_cap_grid", [1, 2, 3, 4]),
                    )
                ),
                category_grid=tuple(
                    map(
                        int,
                        calibration_config.get(
                            "electronics_category_distinct_grid", [2, 3, 4, 5]
                        ),
                    )
                ),
            )
            attach_calibration_hash(
                [path for paths in final_manifests.values() for path in paths],
                calibration["calibration_sha256"],
            )
            selected[dataset_name]["slate_calibration"] = calibration
    retrieval_frame = pd.DataFrame(summary_rows)
    retrieval_frame.to_csv(output_dir / "retrieval_summary.csv", index=False)
    numeric_columns = retrieval_frame.select_dtypes(include=[np.number]).columns.tolist()
    stability = retrieval_frame.groupby(["dataset", "model"])[numeric_columns].agg(
        ["mean", "std"]
    )
    stability.columns = [
        f"{column}_{statistic}" for column, statistic in stability.columns
    ]
    stability.reset_index().to_csv(
        output_dir / "retrieval_stability_summary.csv", index=False
    )
    popularity_summary = pd.concat(popularity_frames, ignore_index=True)
    popularity_summary.to_csv(output_dir / "popularity_full_sort.csv", index=False)
    result = {
        "selected": selected,
        "runs": summary_rows,
        "popularity_baseline": popularity_summary.to_dict("records"),
    }
    (output_dir / "suite_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_suite(
    suite_dir: Path,
    *,
    workers: int,
    resume: bool,
    run_interventions: bool,
    evaluation_output_dir: Path | None = None,
    recalibrate: bool = False,
    calibration_only: bool = False,
    recall_only: bool = False,
    skip_recall: bool = False,
) -> Dict[str, Any]:
    """Run the declared recall, end-to-end, and selected-retriever sensitivity matrix."""

    if recall_only and skip_recall:
        raise ValueError("recall_only and skip_recall are mutually exclusive")
    suite_dir = suite_dir.resolve()
    suite_result_path = suite_dir / "suite_result.json"
    config_path = suite_dir / "config_snapshot.yaml"
    if not suite_result_path.exists() or not config_path.exists():
        raise FileNotFoundError("Completed suite_result.json and config_snapshot.yaml are required")
    suite_result = json.loads(suite_result_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    evaluation_config = dict(config.get("evaluation", {}))
    solver_settings = {
        "slate_preflight_time_limit_seconds": float(
            evaluation_config.get("slate_preflight_time_limit_seconds", 2.0)
        ),
        "opportunity_solver_time_limit_seconds": float(
            evaluation_config.get("opportunity_solver_time_limit_seconds", 5.0)
        ),
    }
    result_root = (
        evaluation_output_dir.resolve()
        if evaluation_output_dir is not None
        else suite_dir
    )
    result_root.mkdir(parents=True, exist_ok=True)
    evaluation_root = result_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    calibration_root = result_root / "calibration"
    protocol_manifest_path = result_root / "evaluation_protocol_manifest.json"
    protocol_datasets: list[Dict[str, Any]] = []
    all_results = []
    output_dirs: list[tuple[str, Path]] = []

    # Calibration is a suite-level gate: complete both datasets before any
    # formal optimizer task is submitted.  This prevents a partial matrix from
    # being mistaken for an admissible run if the second dataset cannot form
    # three distinct monotone strength tiers.
    calibrations: Dict[str, Dict[str, Any]] = {}
    calibration_config = dict(config.get("slate_calibration", {}))
    for dataset_config in config.get("datasets", []):
        dataset_name = str(dataset_config["name"])
        if not recalibrate:
            frozen_path = calibration_root / dataset_name / "slate_calibration.json"
            calibrations[dataset_name] = (
                json.loads(frozen_path.read_text(encoding="utf-8"))
                if frozen_path.exists()
                else dict(
                    suite_result["selected"][dataset_name].get(
                        "slate_calibration", {}
                    )
                )
            )
            continue
        split_path = (
            suite_dir
            / dataset_name
            / "atomic"
            / dataset_name
            / f"{dataset_name}_split.parquet"
        )
        downstream_users = _load_export_users(
            str(suite_dir / dataset_name / "downstream_users.txt")
        )
        calibration = calibrate_slate_policies(
            suite_dir
            / dataset_name
            / "final"
            / "sasrec"
            / "seed_42"
            / "validation_candidates"
            / "candidates.parquet",
            split_path,
            Path(dataset_config["items"]).resolve(),
            calibration_root / dataset_name,
            dataset_name=dataset_name,
            user_ids=downstream_users,
            candidate_k=int(calibration_config.get("candidate_k", 100)),
            top_k=int(calibration_config.get("top_k", 10)),
            # The source suite snapshot belongs to protocol v1 and contains
            # the old seven-point grid.  Kernel-v2 recalibration deliberately
            # binds the expanded protocol constants instead of inheriting that
            # stale experiment setting.
            alpha_grid=DEFAULT_ALPHA_GRID,
            brand_cap_grid=DEFAULT_BRAND_CAP_GRID,
            category_grid=DEFAULT_CATEGORY_GRID,
            solver_time_limit_seconds=float(
                calibration_config.get("solver_time_limit_seconds", 2.0)
            ),
        )
        calibrations[dataset_name] = calibration
        if calibration.get("status") != "success":
            protocol_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "status": "calibration_failed",
                        "source_suite_dir": str(suite_dir),
                        "result_root": str(result_root),
                        "optimizer_kernel_version": OPTIMIZER_KERNEL_VERSION,
                        "failed_dataset": dataset_name,
                        "calibrations": calibrations,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Calibration failed closed for {dataset_name}: "
                f"{calibration.get('failure_reason')}"
            )

    for dataset_name, calibration in calibrations.items():
        policies = dict(calibration.get("policies", {}))
        if calibration.get("status") != "success" or set(policies) != {
            "loose",
            "medium",
            "tight",
        }:
            raise RuntimeError(
                f"Formal evaluation requires a successful v2 calibration for {dataset_name}"
            )
        policy_keys = ["total_budget_alpha", "brand_cap"]
        if "category_distinct_min" in policies["medium"]:
            policy_keys.append("category_distinct_min")
        if len(
            {
                tuple(policy.get(key) for key in policy_keys)
                for policy in policies.values()
            }
        ) != 3 or not all(
            bool(policy.get("within_target_interval"))
            for policy in policies.values()
        ):
            raise RuntimeError(
                f"Calibration tiers for {dataset_name} are not distinct target hits"
            )

    if calibration_only:
        dataset_bindings = []
        for dataset_config in config.get("datasets", []):
            dataset_name = str(dataset_config["name"])
            cohort_path = suite_dir / dataset_name / "downstream_users.txt"
            dataset_bindings.append(
                {
                    "dataset": dataset_name,
                    "calibration_sha256": str(
                        calibrations[dataset_name].get("calibration_sha256", "")
                    ),
                    "cohort_sha256": sha256_file(cohort_path),
                    "candidate_manifest_sha256": {
                        f"{model_key}_seed_{seed}": sha256_file(
                            suite_dir
                            / dataset_name
                            / "final"
                            / model_key
                            / f"seed_{seed}"
                            / "manifest.json"
                        )
                        for model_key in FORMAL_MODELS
                        for seed in (42, 43, 44)
                    },
                }
            )
        payload = {
            "schema_version": "1.0",
            "status": "calibration_complete",
            "source_suite_dir": str(suite_dir),
            "result_root": str(result_root),
            "optimizer_kernel_version": OPTIMIZER_KERNEL_VERSION,
            "datasets": dataset_bindings,
        }
        protocol_manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return payload

    for dataset_config in config.get("datasets", []):
        dataset_name = str(dataset_config["name"])
        items_path = Path(dataset_config["items"]).resolve()
        split_path = (
            suite_dir
            / dataset_name
            / "atomic"
            / dataset_name
            / f"{dataset_name}_split.parquet"
        )
        protocol_catalog_size = int(
            pd.read_parquet(split_path, columns=["item_id"])["item_id"].nunique()
        )
        candidate_ks = (
            (50, 100, 200, 500)
            if protocol_catalog_size >= 500
            else (50, 100, 200)
        )
        model_entries = {
            key: value
            for key, value in suite_result["selected"][dataset_name].items()
            if key in FORMAL_MODELS
        }
        downstream_users = _load_export_users(
            str(suite_dir / dataset_name / "downstream_users.txt")
        )
        calibration = calibrations[dataset_name]
        slate_policies = dict(
            calibration.get(
                "policies", dataset_config.get("frozen_slate_policies", {})
            )
        )
        calibration_sha256 = str(calibration.get("calibration_sha256", ""))
        dataset_solver_settings = {
            **solver_settings,
            "optimizer_time_limit_seconds": float(
                evaluation_config.get("optimizer_time_limit_seconds", 120.0)
            ),
            "optimizer_kernel_version": int(
                evaluation_config.get(
                    "optimizer_kernel_version", OPTIMIZER_KERNEL_VERSION
                )
            ),
            "calibration_sha256": calibration_sha256,
        }
        cohort_path = suite_dir / dataset_name / "downstream_users.txt"
        protocol_datasets.append(
            {
                "dataset": dataset_name,
                "calibration_sha256": calibration_sha256,
                "cohort_sha256": sha256_file(cohort_path),
                "candidate_manifest_sha256": {
                    f"{model_key}_seed_{seed}": sha256_file(
                        suite_dir
                        / dataset_name
                        / "final"
                        / model_key
                        / f"seed_{seed}"
                        / "manifest.json"
                    )
                    for model_key in FORMAL_MODELS
                    for seed in (42, 43, 44)
                },
            }
        )
        protocol_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "running",
                    "source_suite_dir": str(suite_dir),
                    "result_root": str(result_root),
                    "optimizer_kernel_version": OPTIMIZER_KERNEL_VERSION,
                    "datasets": protocol_datasets,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        medium_slate_policy = dict(slate_policies.get("medium", {}))
        for model_key in FORMAL_MODELS:
            if not skip_recall:
                for model_seed in (42, 43, 44):
                    manifest = (
                        suite_dir
                        / dataset_name
                        / "final"
                        / model_key
                        / f"seed_{model_seed}"
                        / "manifest.json"
                    )
                    recall_dir = (
                        evaluation_root
                        / "recall"
                        / dataset_name
                        / model_key
                        / f"seed_{model_seed}"
                    )
                    result = run_artifact_evaluation(
                        manifest,
                        split_path,
                        items_path,
                        recall_dir,
                        candidate_ks=candidate_ks,
                        run_end_to_end=False,
                        workers=workers,
                        resume=resume,
                        **dataset_solver_settings,
                    )
                    all_results.append({"stage": "recall", **result})
                    output_dirs.append(("recall", recall_dir))
            else:
                for model_seed in (42, 43, 44):
                    recall_dir = (
                        evaluation_root
                        / "recall"
                        / dataset_name
                        / model_key
                        / f"seed_{model_seed}"
                    )
                    result_path = recall_dir / "evaluation_result.json"
                    if not result_path.exists():
                        raise FileNotFoundError(
                            "skip_recall requires a completed recall audit: "
                            f"{result_path}"
                        )
                    all_results.append(
                        {
                            "stage": "recall",
                            **json.loads(result_path.read_text(encoding="utf-8")),
                        }
                    )
                    output_dirs.append(("recall", recall_dir))

            if recall_only:
                continue

            manifest = (
                suite_dir
                / dataset_name
                / "final"
                / model_key
                / "seed_42"
                / "manifest.json"
            )
            e2e_dir = evaluation_root / "end_to_end" / dataset_name / model_key
            result = run_artifact_evaluation(
                manifest,
                split_path,
                items_path,
                e2e_dir,
                candidate_ks=candidate_ks,
                end_to_end_candidate_k=100,
                optimizer_seeds=(42, 43, 44),
                population_size=100,
                generations=50,
                run_end_to_end=True,
                workers=workers,
                resume=resume,
                evaluation_users=downstream_users,
                slate_policy=medium_slate_policy,
                **dataset_solver_settings,
            )
            all_results.append({"stage": "end_to_end", **result})
            output_dirs.append(("end_to_end", e2e_dir))

        if recall_only:
            continue

        best_model = max(
            model_entries,
            key=lambda key: tuple(model_entries[key]["selection_key"]),
        )
        best_manifest = (
            suite_dir
            / dataset_name
            / "final"
            / best_model
            / "seed_42"
            / "manifest.json"
        )
        for strength in ("loose", "medium", "tight"):
            if strength not in slate_policies:
                continue
            strength_dir = (
                evaluation_root
                / "constraint_strength"
                / dataset_name
                / best_model
                / strength
            )
            result = run_artifact_evaluation(
                best_manifest,
                split_path,
                items_path,
                strength_dir,
                candidate_ks=(100,),
                end_to_end_candidate_k=100,
                optimizer_seeds=(42, 43, 44),
                population_size=100,
                generations=50,
                methods=("feasible_relevance", "feasible_weighted_ga", "copa"),
                workers=workers,
                resume=resume,
                evaluation_users=downstream_users,
                slate_policy=dict(slate_policies[strength]),
                **dataset_solver_settings,
            )
            all_results.append(
                {
                    "stage": "constraint_strength",
                    "strength": strength,
                    "validation_selected_model": best_model,
                    **result,
                }
            )
            output_dirs.append((f"constraint_strength_{strength}", strength_dir))

        if run_interventions:
            sensitivity_dir = evaluation_root / "sensitivity" / dataset_name / best_model
            result = run_artifact_evaluation(
                best_manifest,
                split_path,
                items_path,
                sensitivity_dir,
                candidate_ks=(100,),
                end_to_end_candidate_k=100,
                optimizer_seeds=(42, 43, 44),
                population_size=100,
                generations=50,
                run_interventions=True,
                run_end_to_end=True,
                run_real_end_to_end=False,
                methods=("copa",),
                workers=workers,
                resume=resume,
                evaluation_users=downstream_users,
                slate_policy=medium_slate_policy,
                **dataset_solver_settings,
            )
            all_results.append(
                {"stage": "sensitivity", "validation_selected_model": best_model, **result}
            )
            output_dirs.append(("sensitivity", sensitivity_dir))

        # Core ablations on exactly the same best-retriever users, candidates,
        # item constraints, and optimizer seeds.  The item-only run removes
        # only slate constraints; the operator run retains the hard constraints
        # but disables MILP population seeding and feasible variation operators.
        item_only_dir = (
            evaluation_root / "ablation" / dataset_name / best_model / "item_only"
        )
        item_only = run_artifact_evaluation(
            best_manifest,
            split_path,
            items_path,
            item_only_dir,
            candidate_ks=(100,),
            end_to_end_candidate_k=100,
            optimizer_seeds=(42, 43, 44),
            population_size=100,
            generations=50,
            methods=("copa",),
            workers=workers,
            resume=resume,
            evaluation_users=downstream_users,
            slate_policy={},
            **dataset_solver_settings,
        )
        all_results.append(
            {"stage": "ablation_item_only", "validation_selected_model": best_model, **item_only}
        )
        output_dirs.append(("ablation_item_only", item_only_dir))

        no_operators_dir = (
            evaluation_root
            / "ablation"
            / dataset_name
            / best_model
            / "no_milp_seed_or_feasible_operators"
        )
        no_operators = run_artifact_evaluation(
            best_manifest,
            split_path,
            items_path,
            no_operators_dir,
            candidate_ks=(100,),
            end_to_end_candidate_k=100,
            optimizer_seeds=(42, 43, 44),
            population_size=100,
            generations=50,
            methods=("copa",),
            workers=workers,
            resume=resume,
            evaluation_users=downstream_users,
            slate_policy=medium_slate_policy,
            use_milp_seed=False,
            use_slate_feasible_operators=False,
            **dataset_solver_settings,
        )
        all_results.append(
            {
                "stage": "ablation_no_milp_seed_or_feasible_operators",
                "validation_selected_model": best_model,
                **no_operators,
            }
        )
        output_dirs.append(
            ("ablation_no_milp_seed_or_feasible_operators", no_operators_dir)
        )

    candidate_frames = []
    e2e_frames = []
    for stage, path in output_dirs:
        if (path / "candidate_metrics.csv").exists():
            frame = pd.read_csv(path / "candidate_metrics.csv")
            frame.insert(0, "evaluation_stage", stage)
            candidate_frames.append(frame)
        if (path / "end_to_end_metrics.csv").exists():
            frame = pd.read_csv(path / "end_to_end_metrics.csv")
            frame.insert(0, "evaluation_stage", stage)
            e2e_frames.append(frame)
    if candidate_frames:
        all_candidates = pd.concat(candidate_frames, ignore_index=True)
        all_candidates.to_csv(
            evaluation_root / "all_candidate_metrics.csv", index=False
        )
        candidate_numeric = all_candidates.select_dtypes(include=[np.number]).columns.tolist()
        candidate_group_keys = [
            "evaluation_stage",
            "dataset",
            "retriever",
            "model_seed",
            "condition",
            "candidate_k",
        ]
        candidate_numeric = [
            column for column in candidate_numeric if column not in candidate_group_keys
        ]
        candidate_summary = all_candidates.groupby(candidate_group_keys)[
            candidate_numeric
        ].agg(["mean", "std"])
        candidate_summary.columns = [
            f"{column}_{statistic}"
            for column, statistic in candidate_summary.columns
        ]
        candidate_summary.reset_index().to_csv(
            evaluation_root / "all_candidate_summary.csv", index=False
        )
    if e2e_frames:
        all_e2e = pd.concat(e2e_frames, ignore_index=True)
        all_e2e.to_csv(
            evaluation_root / "all_end_to_end_metrics.csv", index=False
        )
        e2e_numeric = all_e2e.select_dtypes(include=[np.number]).columns.tolist()
        e2e_group_keys = [
            "evaluation_stage", "dataset", "retriever", "condition", "method"
        ]
        e2e_summary = all_e2e.groupby(e2e_group_keys)[e2e_numeric].agg(
            ["mean", "std"]
        )
        e2e_summary.columns = [
            f"{column}_{statistic}" for column, statistic in e2e_summary.columns
        ]
        e2e_summary.reset_index().to_csv(
            evaluation_root / "all_end_to_end_summary.csv", index=False
        )
        ablation = all_e2e[
            all_e2e["evaluation_stage"].isin(
                {
                    "end_to_end",
                    "ablation_item_only",
                    "ablation_no_milp_seed_or_feasible_operators",
                }
            )
            & (all_e2e["method"] == "copa")
        ].copy()
        ablation.loc[
            ablation["evaluation_stage"] == "ablation_item_only", "method"
        ] = "copa_item_only"
        ablation.loc[
            ablation["evaluation_stage"]
            == "ablation_no_milp_seed_or_feasible_operators",
            "method",
        ] = "copa_no_milp_seed_or_feasible_operators"
        if not ablation.empty:
            ablation_inference = _paired_user_inference(
                ablation, bootstrap_samples=10_000, seed=42
            )
            ablation_inference.to_csv(
                evaluation_root / "ablation_paired_user_inference.csv",
                index=False,
            )
    payload = {
        "suite_dir": str(suite_dir),
        "result_root": str(result_root),
        "workers": int(workers),
        "resume": bool(resume),
        "run_interventions": bool(run_interventions),
        "recall_only": bool(recall_only),
        "skip_recall": bool(skip_recall),
        "runs": all_results,
    }
    (evaluation_root / "evaluation_suite_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_manifest = json.loads(
        protocol_manifest_path.read_text(encoding="utf-8")
    )
    protocol_manifest["status"] = "complete"
    protocol_manifest_path.write_text(
        json.dumps(
            protocol_manifest, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def run_performance_pilot(
    suite_dir: Path,
    result_root: Path,
    *,
    users_per_dataset: int = 8,
    workers: int = 12,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run the declared full-budget GA/COPA gate on fixed cohort prefixes."""

    if users_per_dataset <= 0:
        raise ValueError("users_per_dataset must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    suite_dir = suite_dir.resolve()
    result_root = result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    suite_result = json.loads(
        (suite_dir / "suite_result.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (suite_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
    ) or {}
    evaluation_config = dict(config.get("evaluation", {}))
    pilot_root = result_root / "performance_pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)

    memory_stop = threading.Event()
    memory_stats = {"peak_bytes": 0}

    def monitor_memory() -> None:
        import psutil

        process = psutil.Process()
        while not memory_stop.wait(0.05):
            processes = [process, *process.children(recursive=True)]
            rss = 0
            seen: set[int] = set()
            for current in processes:
                if current.pid in seen:
                    continue
                seen.add(current.pid)
                try:
                    rss += int(current.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            memory_stats["peak_bytes"] = max(memory_stats["peak_bytes"], rss)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    started = perf_counter()
    runs: list[Dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    try:
        for dataset_config in config.get("datasets", []):
            dataset_name = str(dataset_config["name"])
            calibration_path = (
                result_root / "calibration" / dataset_name / "slate_calibration.json"
            )
            if not calibration_path.exists():
                raise FileNotFoundError(
                    f"Performance pilot requires frozen calibration: {calibration_path}"
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration.get("status") != "success":
                raise RuntimeError(
                    f"Performance pilot rejected failed calibration for {dataset_name}"
                )
            policies = dict(calibration.get("policies", {}))
            if set(policies) != {"loose", "medium", "tight"}:
                raise RuntimeError(
                    f"Calibration for {dataset_name} does not contain all three tiers"
                )
            model_entries = {
                key: value
                for key, value in suite_result["selected"][dataset_name].items()
                if key in FORMAL_MODELS
            }
            best_model = max(
                model_entries,
                key=lambda key: tuple(model_entries[key]["selection_key"]),
            )
            cohort = sorted(
                _load_export_users(
                    str(suite_dir / dataset_name / "downstream_users.txt")
                )
                or set()
            )
            pilot_users = cohort[: int(users_per_dataset)]
            if len(pilot_users) != int(users_per_dataset):
                raise RuntimeError(
                    f"{dataset_name} has only {len(pilot_users)} pilot users"
                )
            output_dir = pilot_root / dataset_name / best_model
            result = run_artifact_evaluation(
                suite_dir
                / dataset_name
                / "final"
                / best_model
                / "seed_42"
                / "manifest.json",
                suite_dir
                / dataset_name
                / "atomic"
                / dataset_name
                / f"{dataset_name}_split.parquet",
                Path(dataset_config["items"]).resolve(),
                output_dir,
                candidate_ks=(100,),
                end_to_end_candidate_k=100,
                optimizer_seeds=(42, 43, 44),
                population_size=100,
                generations=50,
                methods=("feasible_weighted_ga", "copa"),
                run_end_to_end=True,
                workers=workers,
                resume=resume,
                evaluation_users=pilot_users,
                slate_policy=dict(policies["medium"]),
                slate_preflight_time_limit_seconds=float(
                    evaluation_config.get(
                        "slate_preflight_time_limit_seconds", 2.0
                    )
                ),
                opportunity_solver_time_limit_seconds=float(
                    evaluation_config.get(
                        "opportunity_solver_time_limit_seconds", 5.0
                    )
                ),
                optimizer_time_limit_seconds=float(
                    evaluation_config.get("optimizer_time_limit_seconds", 120.0)
                ),
                optimizer_kernel_version=int(
                    evaluation_config.get(
                        "optimizer_kernel_version", OPTIMIZER_KERNEL_VERSION
                    )
                ),
                calibration_sha256=str(calibration["calibration_sha256"]),
            )
            runs.append(
                {
                    "dataset": dataset_name,
                    "retriever": best_model,
                    "users": pilot_users,
                    **result,
                }
            )
            frame = pd.read_csv(output_dir / "end_to_end_metrics.csv")
            frame.insert(0, "pilot_dataset", dataset_name)
            metric_frames.append(frame)
    finally:
        elapsed = perf_counter() - started
        memory_stop.set()
        monitor.join(timeout=2.0)

    metrics = pd.concat(metric_frames, ignore_index=True)
    successful = metrics[metrics["result_status"] == "success"]
    latency: Dict[str, Dict[str, float]] = {}
    thresholds = {"feasible_weighted_ga": 10.0, "copa": 20.0}
    latency_pass = True
    for method, threshold in thresholds.items():
        values = successful.loc[
            successful["method"] == method, "runtime_seconds"
        ].astype(float)
        p95 = float(values.quantile(0.95)) if len(values) else float("inf")
        maximum = float(values.max()) if len(values) else float("inf")
        passed = p95 <= threshold and maximum <= 60.0
        latency[method] = {
            "successful_tasks": int(len(values)),
            "p95_seconds": p95,
            "maximum_seconds": maximum,
            "p95_limit_seconds": threshold,
            "single_task_limit_seconds": 60.0,
            "passed": passed,
        }
        latency_pass = latency_pass and passed

    unexplained_statuses = {"optimizer_failed", "verification_failed", "solver_unknown"}
    unexplained = metrics[metrics["result_status"].isin(unexplained_statuses)]
    delivered = metrics[metrics["delivered_slate"] == 1.0]
    verifier_pass = bool(
        delivered.empty
        or (delivered["delivered_slate_verifier_pass"] == 1.0).all()
    )
    peak_gib = memory_stats["peak_bytes"] / (1024**3)
    memory_pass = peak_gib < 24.0
    cohort_tasks = sum(
        # 45 medium-main + 27 strength + 6 ablation + 21
        # controlled/oracle atomic method-seed tasks per user.
        int(dataset_config.get("downstream_cohort_size", 0)) * 99
        for dataset_config in config.get("datasets", [])
    )
    throughput = len(metrics) / max(elapsed, 1e-12)
    estimated_hours = cohort_tasks / max(throughput, 1e-12) / 3600.0
    estimate_pass = estimated_hours <= 48.0
    gate_passed = bool(
        latency_pass
        and memory_pass
        and estimate_pass
        and unexplained.empty
        and verifier_pass
    )
    payload = {
        "status": "passed" if gate_passed else "failed",
        "optimizer_kernel_version": OPTIMIZER_KERNEL_VERSION,
        "workers": int(workers),
        "users_per_dataset": int(users_per_dataset),
        "population_size": 100,
        "generations": 50,
        "optimizer_seeds": [42, 43, 44],
        "elapsed_seconds": elapsed,
        "completed_rows": int(len(metrics)),
        "throughput_tasks_per_second": throughput,
        "latency": latency,
        "peak_process_tree_memory_gib": peak_gib,
        "memory_limit_gib": 24.0,
        "memory_passed": memory_pass,
        "estimated_formal_optimizer_tasks": cohort_tasks,
        "estimated_formal_hours": estimated_hours,
        "estimated_formal_hours_limit": 48.0,
        "estimate_passed": estimate_pass,
        "unexplained_failure_rows": unexplained.to_dict("records"),
        "delivered_slate_verifier_passed": verifier_pass,
        "runs": runs,
    }
    (pilot_root / "performance_gate_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if not gate_passed:
        raise RuntimeError(
            "Performance pilot failed; formal evaluation is blocked. See "
            f"{pilot_root / 'performance_gate_result.json'}"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        prepared = prepare_recbole_dataset(
            args.interactions,
            args.output_root,
            dataset_name=args.dataset_name,
            k_core=args.k_core,
            items_path=args.items,
        )
        print(json.dumps(dict(prepared.statistics), ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        prepared = load_prepared_dataset(
            args.data_root,
            args.dataset_name,
            source_interactions=args.source_interactions,
            source_items=args.source_items,
        )
        spec = RecBoleRunSpec(
            model=FORMAL_MODELS[args.model],
            retriever=args.model,
            seed=args.seed,
            epochs=args.epochs,
            stopping_step=args.stopping_step,
            max_candidate_k=args.candidate_k,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            model_parameters=json.loads(args.model_parameters),
        )
        result = train_recbole_and_export(
            prepared,
            spec,
            args.output_dir,
            export_users=_load_export_users(args.export_users),
            export_validation_candidates=args.export_validation_candidates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "suite":
        result = run_suite(Path(args.config), Path(args.output_dir), args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        parse_csv = lambda value, cast: [cast(item) for item in value.split(",") if item]
        result = run_artifact_evaluation(
            args.manifest,
            args.split,
            args.items,
            args.output_dir,
            candidate_ks=parse_csv(args.candidate_ks, int),
            end_to_end_candidate_k=args.end_to_end_candidate_k,
            optimizer_seeds=parse_csv(args.optimizer_seeds, int),
            population_size=args.population_size,
            generations=args.generations,
            methods=parse_csv(args.methods, str),
            controlled_levels=parse_csv(args.controlled_levels, float),
            run_interventions=args.interventions,
            run_end_to_end=not args.recall_only,
            workers=args.workers,
            resume=args.resume,
            slate_preflight_time_limit_seconds=(
                args.slate_preflight_time_limit_seconds
            ),
            opportunity_solver_time_limit_seconds=(
                args.opportunity_solver_time_limit_seconds
            ),
            optimizer_time_limit_seconds=args.optimizer_time_limit_seconds,
            optimizer_kernel_version=args.optimizer_kernel_version,
            calibration_sha256=args.calibration_sha256,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate-suite":
        result = evaluate_suite(
            Path(args.suite_dir),
            workers=args.workers,
            resume=args.resume,
            run_interventions=not args.skip_interventions,
            evaluation_output_dir=(
                Path(args.evaluation_output_dir)
                if args.evaluation_output_dir
                else None
            ),
            recalibrate=args.recalibrate,
            calibration_only=args.calibration_only,
            recall_only=args.recall_only,
            skip_recall=args.skip_recall,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "performance-pilot":
        result = run_performance_pilot(
            Path(args.suite_dir),
            Path(args.evaluation_output_dir),
            users_per_dataset=args.users_per_dataset,
            workers=args.workers,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = run_smoke(Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
