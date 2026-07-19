"""Prepare, tune, and export mature RecBole retrieval candidates for COPA."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
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
    run_artifact_evaluation,
)
from copa.retrieval.popularity_baseline import evaluate_temporal_popularity
from copa.retrieval.artifacts import (
    CandidateArtifactManifest,
    sha256_file,
    validate_artifact_alignment,
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

    evaluate_suite = subparsers.add_parser("evaluate-suite")
    evaluate_suite.add_argument("--suite-dir", required=True)
    evaluate_suite.add_argument("--workers", type=int, default=4)
    evaluate_suite.add_argument("--resume", action="store_true")
    evaluate_suite.add_argument("--skip-interventions", action="store_true")

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
) -> Dict[str, Any] | None:
    if not resume or not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = recbole_run_signature(
        prepared, spec, validation_only=validation_only
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
            prepared, spec, output_dir / key, export_users=test_users
        )
    determinism: Dict[str, Any] = {}
    for key, spec in specs.items():
        repeat_dir = output_dir / f"{key}_same_seed_repeat"
        repeat = train_recbole_and_export(
            prepared, spec, repeat_dir, export_users=test_users
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
    cohort_size = int(config.get("export_cohort_size", 100))
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
        test_users = sorted(
            pd.read_parquet(prepared.split_path)
            .query("split == 'test'")["user_id"]
            .astype(str)
            .unique()
        )
        rng = np.random.default_rng(int(config.get("cohort_seed", 42)))
        if cohort_size > 0 and len(test_users) > cohort_size:
            test_users = sorted(rng.choice(test_users, size=cohort_size, replace=False).tolist())
        (dataset_dir / "export_users.txt").write_text("\n".join(test_users) + "\n", encoding="utf-8")
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
                result = _load_if_matching(
                    result_path,
                    prepared,
                    final_spec,
                    validation_only=False,
                    resume=resume,
                    export_users=set(test_users),
                )
                if result is None:
                    result = train_recbole_and_export(
                        prepared,
                        final_spec,
                        final_dir,
                        export_users=test_users,
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
    suite_dir: Path, *, workers: int, resume: bool, run_interventions: bool
) -> Dict[str, Any]:
    """Run the declared recall, end-to-end, and selected-retriever sensitivity matrix."""

    suite_dir = suite_dir.resolve()
    suite_result_path = suite_dir / "suite_result.json"
    config_path = suite_dir / "config_snapshot.yaml"
    if not suite_result_path.exists() or not config_path.exists():
        raise FileNotFoundError("Completed suite_result.json and config_snapshot.yaml are required")
    suite_result = json.loads(suite_result_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    evaluation_root = suite_dir / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    output_dirs: list[tuple[str, Path]] = []
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
        for model_key in FORMAL_MODELS:
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
                )
                all_results.append({"stage": "recall", **result})
                output_dirs.append(("recall", recall_dir))

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
            )
            all_results.append({"stage": "end_to_end", **result})
            output_dirs.append(("end_to_end", e2e_dir))

        if run_interventions:
            best_model = max(
                model_entries,
                key=lambda key: tuple(model_entries[key]["selection_key"]),
            )
            manifest = (
                suite_dir
                / dataset_name
                / "final"
                / best_model
                / "seed_42"
                / "manifest.json"
            )
            sensitivity_dir = evaluation_root / "sensitivity" / dataset_name / best_model
            result = run_artifact_evaluation(
                manifest,
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
            )
            all_results.append(
                {"stage": "sensitivity", "validation_selected_model": best_model, **result}
            )
            output_dirs.append(("sensitivity", sensitivity_dir))

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
    payload = {
        "suite_dir": str(suite_dir),
        "workers": int(workers),
        "resume": bool(resume),
        "run_interventions": bool(run_interventions),
        "runs": all_results,
    }
    (evaluation_root / "evaluation_suite_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate-suite":
        result = evaluate_suite(
            Path(args.suite_dir),
            workers=args.workers,
            resume=args.resume,
            run_interventions=not args.skip_interventions,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = run_smoke(Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
