"""Offline RecBole training and full-sort candidate export.

This module is optional and is never imported by the deterministic COPA
pipeline.  Its only durable output is the artifact contract in ``artifacts``.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import CandidateArtifactManifest, sha256_file


def ensure_recbole_numpy2_compatibility() -> list[str]:
    """Provide removed scalar aliases expected by RecBole 1.2.1.

    RecBole's own compatibility hook maps legacy aliases (``np.float`` etc.)
    through names such as ``np.float_`` that NumPy 2 removed.  Keeping this
    shim at the optional backend boundary avoids downgrading COPA's numerical
    stack or modifying site-packages.
    """

    aliases = {
        "bool_": np.bool_,
        "int_": np.int64,
        "float_": np.float64,
        "complex_": np.complex128,
        "object_": np.object_,
        "str_": np.str_,
        "unicode_": np.str_,
    }
    installed = []
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
            installed.append(name)
    return installed


def _reset_recbole_root_logger() -> None:
    """Prevent RecBole's process-global handlers from leaking across run dirs."""

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)


@dataclass(frozen=True)
class PreparedRecBoleDataset:
    name: str
    protocol: str
    data_root: Path
    atomic_path: Path
    split_path: Path
    source_interactions: Path
    source_items: Optional[Path]
    statistics: Mapping[str, Any]


@dataclass(frozen=True)
class RecBoleRunSpec:
    model: str
    retriever: str
    seed: int = 42
    epochs: int = 100
    stopping_step: int = 10
    max_candidate_k: int = 500
    train_batch_size: int = 2048
    eval_batch_size: int = 4096
    learning_rate: float = 1e-3
    model_parameters: Mapping[str, Any] = field(default_factory=dict)


def recbole_run_signature(
    prepared: PreparedRecBoleDataset,
    spec: RecBoleRunSpec,
    *,
    validation_only: bool,
) -> str:
    """Fingerprint all settings that can change a trained/evaluated run."""

    payload = {
        "dataset": prepared.name,
        "protocol": prepared.protocol,
        "atomic_sha256": sha256_file(prepared.atomic_path),
        "split_sha256": sha256_file(prepared.split_path),
        "model": spec.model,
        "retriever": spec.retriever,
        "seed": spec.seed,
        "epochs": spec.epochs,
        "stopping_step": spec.stopping_step,
        "max_candidate_k": spec.max_candidate_k,
        "train_batch_size": spec.train_batch_size,
        "eval_batch_size": spec.eval_batch_size,
        "learning_rate": spec.learning_rate,
        "model_parameters": dict(spec.model_parameters),
        "validation_only": bool(validation_only),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_prepared_dataset(
    data_root: Path | str,
    dataset_name: str,
    *,
    source_interactions: Path | str,
    source_items: Path | str | None = None,
    protocol: str = "deduplicated_5core_to_ls",
) -> PreparedRecBoleDataset:
    data_root = Path(data_root).resolve()
    dataset_dir = data_root / dataset_name
    atomic_path = dataset_dir / f"{dataset_name}.inter"
    split_path = dataset_dir / f"{dataset_name}_split.parquet"
    stats_path = dataset_dir / f"{dataset_name}_split_statistics.json"
    for path in (atomic_path, split_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"Prepared RecBole dataset file not found: {path}")
    return PreparedRecBoleDataset(
        name=dataset_name,
        protocol=protocol,
        data_root=data_root,
        atomic_path=atomic_path,
        split_path=split_path,
        source_interactions=Path(source_interactions).resolve(),
        source_items=Path(source_items).resolve() if source_items else None,
        statistics=json.loads(stats_path.read_text(encoding="utf-8")),
    )


def _iterative_k_core(frame: pd.DataFrame, k: int) -> tuple[pd.DataFrame, int]:
    if k <= 1:
        return frame.copy(), 0
    current = frame.copy()
    iterations = 0
    while True:
        iterations += 1
        before = len(current)
        user_counts = current["user_id"].value_counts()
        item_counts = current["item_id"].value_counts()
        current = current[
            current["user_id"].isin(user_counts[user_counts >= k].index)
            & current["item_id"].isin(item_counts[item_counts >= k].index)
        ].copy()
        if current.empty:
            raise ValueError(f"{k}-core filtering removed all interactions")
        if len(current) == before:
            return current.reset_index(drop=True), iterations


def prepare_recbole_dataset(
    interactions_path: Path | str,
    output_root: Path | str,
    *,
    dataset_name: str,
    protocol: str = "deduplicated_5core_to_ls",
    k_core: int = 5,
    items_path: Path | str | None = None,
) -> PreparedRecBoleDataset:
    """Create a stable RecBole atomic file and an independently auditable split."""

    interactions_path = Path(interactions_path).resolve()
    items = Path(items_path).resolve() if items_path else None
    output_root = Path(output_root).resolve()
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    atomic_path = dataset_dir / f"{dataset_name}.inter"
    split_path = dataset_dir / f"{dataset_name}_split.parquet"
    stats_path = dataset_dir / f"{dataset_name}_split_statistics.json"

    frame = pd.read_parquet(
        interactions_path, columns=["user_id", "item_id", "timestamp"]
    )
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    if frame[["user_id", "item_id", "timestamp"]].isna().any().any():
        raise ValueError("Interactions contain null identifiers or timestamps")
    loaded_rows = len(frame)
    frame = frame.sort_values(["user_id", "item_id", "timestamp"], kind="mergesort")
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="last")
    deduplicated_rows = len(frame)
    frame, iterations = _iterative_k_core(frame, int(k_core))
    frame = frame.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")

    counts = frame["user_id"].value_counts()
    if (counts < 3).any():
        raise ValueError("Temporal train/validation/test requires at least three items per user")
    frame["split"] = "train"
    test_index = frame.groupby("user_id", sort=False).tail(1).index
    remaining = frame.drop(index=test_index)
    valid_index = remaining.groupby("user_id", sort=False).tail(1).index
    frame.loc[valid_index, "split"] = "valid"
    frame.loc[test_index, "split"] = "test"
    frame.to_parquet(split_path, index=False)

    atomic = frame[["user_id", "item_id", "timestamp"]].copy()
    atomic.columns = ["user_id:token", "item_id:token", "timestamp:float"]
    atomic.to_csv(atomic_path, sep="\t", index=False)

    train = frame[frame["split"] == "train"]
    valid = frame[frame["split"] == "valid"]
    test = frame[frame["split"] == "test"]
    train_items = set(train["item_id"])
    statistics = {
        "protocol": protocol,
        "k_core": int(k_core),
        "k_core_iterations": int(iterations),
        "loaded_rows": int(loaded_rows),
        "deduplicated_rows": int(deduplicated_rows),
        "filtered_rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "items": int(frame["item_id"].nunique()),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_items": int(train["item_id"].nunique()),
        "test_target_train_catalog_coverage": float(test["item_id"].isin(train_items).mean()),
        "minimum_user_interactions": int(counts.min()),
        "median_user_interactions": float(counts.median()),
        "source_interactions_sha256": sha256_file(interactions_path),
    }
    if items is not None:
        statistics["source_items_sha256"] = sha256_file(items)
    stats_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedRecBoleDataset(
        name=dataset_name,
        protocol=protocol,
        data_root=output_root,
        atomic_path=atomic_path,
        split_path=split_path,
        source_interactions=interactions_path,
        source_items=items,
        statistics=statistics,
    )


def training_visible_statistics(
    prepared: PreparedRecBoleDataset,
    items_path: Path | str,
) -> tuple[Dict[str, float], Dict[str, set[str]], Dict[str, float]]:
    """Compute test-time popularity, histories, and budgets without test labels."""

    split = pd.read_parquet(prepared.split_path)
    visible = split[split["split"].isin(["train", "valid"])].copy()
    counts = visible["item_id"].value_counts().astype(float)
    popularity = np.log1p(counts)
    denominator = float(popularity.max()) if len(popularity) else 1.0
    popularity_map = (popularity / max(denominator, 1e-12)).to_dict()
    histories = (
        visible.groupby("user_id")["item_id"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    items = pd.read_parquet(items_path, columns=["item_id", "price_filled"])
    items["item_id"] = items["item_id"].astype(str)
    merged = visible[["user_id", "item_id"]].merge(items, on="item_id", how="left")
    global_price = float(pd.to_numeric(items["price_filled"], errors="coerce").median())
    budgets: Dict[str, float] = {}
    for user_id, group in merged.groupby("user_id"):
        prices = pd.to_numeric(group["price_filled"], errors="coerce").dropna()
        center = float(prices.median()) if len(prices) else global_price
        budgets[str(user_id)] = 1.2 * center
    return (
        {str(key): float(value) for key, value in popularity_map.items()},
        {str(key): value for key, value in histories.items()},
        budgets,
    )


def _recbole_config(
    prepared: PreparedRecBoleDataset,
    spec: RecBoleRunSpec,
    output_dir: Path,
) -> Dict[str, Any]:
    sequential = spec.model.lower() == "sasrec"
    catalog_items = max(1, int(prepared.statistics.get("items", 100)))
    evaluation_topk = sorted({min(value, catalog_items) for value in (10, 50, 100)})
    validation_k = max(evaluation_topk)
    config: Dict[str, Any] = {
        "data_path": str(prepared.data_root),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "mode": "full",
        },
        "metrics": ["Recall", "NDCG", "MRR", "Hit"],
        "topk": evaluation_topk,
        "valid_metric": f"NDCG@{validation_k}",
        "epochs": int(spec.epochs),
        "stopping_step": int(spec.stopping_step),
        "eval_step": 1,
        "train_batch_size": int(spec.train_batch_size),
        "eval_batch_size": int(spec.eval_batch_size),
        "learning_rate": float(spec.learning_rate),
        "seed": int(spec.seed),
        "reproducibility": True,
        "use_gpu": True,
        "gpu_id": 0,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "log_wandb": False,
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "single_spec": True,
    }
    if sequential:
        loss_type = str(spec.model_parameters.get("loss_type", "CE")).upper()
        config.update(
            {
                "MAX_ITEM_LIST_LENGTH": int(spec.model_parameters.get("MAX_ITEM_LIST_LENGTH", 50)),
                "ITEM_LIST_LENGTH_FIELD": "item_length",
                "LIST_SUFFIX": "_list",
                "train_neg_sample_args": (
                    None
                    if loss_type == "CE"
                    else {
                        "distribution": "uniform",
                        "sample_num": 1,
                        "alpha": 1.0,
                        "dynamic": False,
                        "candidate_num": 0,
                    }
                ),
            }
        )
    else:
        config["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
    config.update(dict(spec.model_parameters))
    return config


def _normalize_scores(scores: "Any") -> tuple["Any", "Any", "Any"]:
    import torch

    finite = torch.isfinite(scores)
    safe = torch.where(finite, scores, torch.zeros_like(scores))
    minimum = torch.where(finite, scores, torch.inf).amin(dim=1, keepdim=True)
    maximum = torch.where(finite, scores, -torch.inf).amax(dim=1, keepdim=True)
    span = maximum - minimum
    normalized = torch.where(span > 0, (safe - minimum) / span, torch.zeros_like(safe))
    normalized = torch.where(finite, normalized, torch.full_like(normalized, -torch.inf))
    return normalized, minimum, maximum


def _validate_id_round_trip(dataset: "Any") -> Dict[str, int]:
    """Audit RecBole's raw-token/internal-id mapping for both entity fields."""

    audited: Dict[str, int] = {}
    for label, field in (("users", dataset.uid_field), ("items", dataset.iid_field)):
        tokens = np.asarray(dataset.field2id_token[field]).astype(str)
        internal = np.arange(len(tokens), dtype=np.int64)
        round_trip_tokens = np.asarray(dataset.id2token(field, internal)).astype(str)
        if not np.array_equal(tokens, round_trip_tokens):
            raise RuntimeError(f"RecBole {label} internal-to-token mapping is not bijective")
        token_to_id = dataset.field2token_id[field]
        for index, token in enumerate(tokens):
            if int(token_to_id[token]) != index:
                raise RuntimeError(f"RecBole {label} token-to-internal mapping is not bijective")
        audited[f"mapped_{label}"] = int(len(tokens) - 1)  # exclude padding id 0
    return audited


def _load_self_produced_checkpoint(
    trainer: "Any", model: "Any", output_dir: Path
) -> Path:
    """Load only the checkpoint created by the current local training run.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    RecBole 1.2.1 checkpoints also contain its Config object, so the upstream
    ``Trainer.evaluate(load_best_model=True)`` path no longer loads them.  The
    broader pickle loader is safe here only because the path is constrained to
    this run's checkpoint directory and the file was just written by RecBole.
    """

    import torch

    checkpoint_root = (output_dir / "checkpoints").resolve()
    checkpoint_path = Path(trainer.saved_model_file).resolve()
    if not checkpoint_path.is_relative_to(checkpoint_root):
        raise RuntimeError(
            f"Refusing to unpickle checkpoint outside this run: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=trainer.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    trainer.model = model
    return checkpoint_path


def _export_candidates(
    trainer: "Any",
    model: "Any",
    dataset: "Any",
    test_data: "Any",
    *,
    output_dir: Path,
    spec: RecBoleRunSpec,
    export_users: Optional[set[str]],
    train_item_ids: set[str],
    expected_targets: Mapping[str, str],
    visible_histories: Mapping[str, set[str]],
) -> tuple[Path, Path, Dict[str, Any]]:
    import torch

    model.eval()
    trainer.model = model
    trainer.tot_item_num = test_data._dataset.item_num
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    item_tokens = np.asarray(dataset.field2id_token[iid_field]).astype(str)
    candidate_rows: list[Dict[str, Any]] = []
    target_rows: list[Dict[str, Any]] = []
    exported_users = 0
    with torch.no_grad():
        for batch in test_data:
            interaction, _, positive_u, positive_i = batch
            all_user_internal = interaction[uid_field].detach().cpu().numpy().astype(int)
            all_user_tokens = np.asarray(
                dataset.id2token(uid_field, all_user_internal)
            ).astype(str)
            selected_positions = [
                index
                for index, user_id in enumerate(all_user_tokens)
                if export_users is None or str(user_id) in export_users
            ]
            if not selected_positions:
                continue
            positive_u_np = positive_u.detach().cpu().numpy().astype(int)
            positive_i_np = positive_i.detach().cpu().numpy().astype(int)
            target_internal_ids = []
            for original_position in selected_positions:
                target_indices = positive_i_np[positive_u_np == original_position]
                user_id = str(all_user_tokens[original_position])
                if len(target_indices) != 1:
                    raise RuntimeError(
                        f"Expected one test target for user {user_id}, got {len(target_indices)}"
                    )
                target_internal = int(target_indices[0])
                target_id = str(item_tokens[target_internal])
                if expected_targets.get(user_id) != target_id:
                    raise RuntimeError(
                        f"RecBole target/split disagreement for {user_id}: "
                        f"{target_id} != {expected_targets.get(user_id)}"
                    )
                target_internal_ids.append(target_internal)

            selected_interaction = interaction[selected_positions].to(trainer.device)
            try:
                scores = model.full_sort_predict(selected_interaction)
            except NotImplementedError as exc:
                raise RuntimeError(
                    f"Formal retriever {spec.model} must implement full_sort_predict"
                ) from exc
            scores = scores.view(len(selected_positions), trainer.tot_item_num)
            scores[:, 0] = -torch.inf
            user_tokens = all_user_tokens[selected_positions]
            # RecBole's general and sequential full-sort loaders do not expose
            # identical history masks. Apply the independently audited protocol
            # mask here so all three formal backends have exactly the same
            # candidate eligibility rule.
            scores = scores.clone()
            token_to_internal = dataset.field2token_id[iid_field]
            for row_index, user_id in enumerate(user_tokens):
                seen_internal = [
                    int(token_to_internal[item_id])
                    for item_id in visible_histories.get(str(user_id), set())
                    if item_id in token_to_internal
                ]
                if seen_internal:
                    scores[row_index, seen_internal] = -torch.inf
            normalized, _, _ = _normalize_scores(scores)
            for row_index, (user_id, target_internal) in enumerate(
                zip(user_tokens, target_internal_ids)
            ):
                target_id = str(item_tokens[target_internal])
                row_scores = scores[row_index]
                row_base = normalized[row_index]
                finite_count = int(torch.isfinite(row_scores).sum().item())
                top_count = min(int(spec.max_candidate_k), finite_count)
                order = torch.argsort(row_scores, descending=True, stable=True)[:top_count]
                top_indices = order.detach().cpu().numpy().astype(int)
                top_item_ids = {str(item_tokens[index]) for index in top_indices}
                seen_overlap = top_item_ids & visible_histories.get(str(user_id), set())
                if seen_overlap:
                    raise RuntimeError(
                        f"Full-sort export contains seen items for {user_id}: "
                        f"{sorted(seen_overlap)[:5]}"
                    )
                raw_values = row_scores[order].detach().cpu().numpy().astype(float)
                base_values = row_base[order].detach().cpu().numpy().astype(float)
                target_score = float(row_scores[target_internal].detach().cpu())
                target_base = float(row_base[target_internal].detach().cpu())
                if not np.isfinite(target_score):
                    raise RuntimeError(f"Test target was masked for user {user_id}")
                greater = int((row_scores > row_scores[target_internal]).sum().item())
                equal_before = int(
                    (
                        (row_scores[:target_internal] == row_scores[target_internal])
                        & torch.isfinite(row_scores[:target_internal])
                    ).sum().item()
                )
                target_rank = 1 + greater + equal_before
                for rank, (internal_id, raw_score, base_score) in enumerate(
                    zip(top_indices, raw_values, base_values), start=1
                ):
                    candidate_rows.append(
                        {
                            "user_id": str(user_id),
                            "item_id": str(item_tokens[internal_id]),
                            "raw_score": float(raw_score),
                            "base_score": float(base_score),
                            "retrieval_rank": int(rank),
                            "retriever": spec.retriever,
                            "backend": f"recbole-{__import__('recbole').__version__}",
                            "model_seed": int(spec.seed),
                        }
                    )
                target_rows.append(
                    {
                        "user_id": str(user_id),
                        "target_item_id": target_id,
                        "target_raw_score": target_score,
                        "target_base_score": target_base,
                        "target_full_rank": int(target_rank),
                        "target_model_covered": bool(target_id in train_item_ids),
                    }
                )
                exported_users += 1
    candidates = pd.DataFrame(candidate_rows)
    targets = pd.DataFrame(target_rows)
    if candidates.empty or targets.empty:
        raise RuntimeError("Full-sort export produced no candidates")
    candidate_path = output_dir / "candidates.parquet"
    target_path = output_dir / "targets.parquet"
    candidates.to_parquet(candidate_path, index=False)
    targets.to_parquet(target_path, index=False)
    return candidate_path, target_path, {
        "exported_users": int(exported_users),
        "candidate_rows": int(len(candidates)),
        "target_model_coverage": float(targets["target_model_covered"].mean()),
    }


def _train_recbole_and_export_in_workdir(
    prepared: PreparedRecBoleDataset,
    spec: RecBoleRunSpec,
    output_dir: Path | str,
    *,
    export_users: Optional[Iterable[str]] = None,
    validation_only: bool = False,
) -> Dict[str, Any]:
    """Train one mature RecBole model, evaluate it, and export ranked candidates."""

    numpy_compatibility_aliases = ensure_recbole_numpy2_compatibility()
    run_signature = recbole_run_signature(
        prepared, spec, validation_only=validation_only
    )
    import recbole
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_dict = _recbole_config(prepared, spec, output_dir)
    config = Config(model=spec.model, dataset=prepared.name, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    _reset_recbole_root_logger()
    init_logger(config)
    dataset = create_dataset(config)
    mapping_statistics = _validate_id_round_trip(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    started = perf_counter()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=True, show_progress=False
    )
    checkpoint_path = _load_self_produced_checkpoint(trainer, model, output_dir)
    environment = {
        "python": platform.python_version(),
        "recbole": recbole.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numpy2_compatibility_aliases": numpy_compatibility_aliases,
        "trusted_local_checkpoint_weights_only": False,
    }
    if validation_only:
        elapsed = perf_counter() - started
        result = {
            "dataset": prepared.name,
            "model": spec.model,
            "retriever": spec.retriever,
            "seed": spec.seed,
            "best_valid_score": float(best_valid_score),
            "best_valid_result": dict(best_valid_result),
            "test_result": {},
            "elapsed_seconds": float(elapsed),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest": None,
            "exported_users": 0,
            "candidate_rows": 0,
            "target_model_coverage": None,
            "validation_only": True,
            "environment": environment,
            "run_signature": run_signature,
        }
        (output_dir / "run_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trainer.tensorboard.close()
        return result

    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    elapsed = perf_counter() - started

    split = pd.read_parquet(prepared.split_path)
    split["user_id"] = split["user_id"].astype(str)
    split["item_id"] = split["item_id"].astype(str)
    train_item_ids = set(split.loc[split["split"] == "train", "item_id"].astype(str))
    expected_targets = (
        split.loc[split["split"] == "test", ["user_id", "item_id"]]
        .set_index("user_id")["item_id"]
        .to_dict()
    )
    visible_histories = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in split[split["split"].isin(["train", "valid"])].groupby("user_id")
    }
    candidate_path, target_path, export_stats = _export_candidates(
        trainer,
        model,
        dataset,
        test_data,
        output_dir=output_dir,
        spec=spec,
        export_users={str(value) for value in export_users} if export_users is not None else None,
        train_item_ids=train_item_ids,
        expected_targets=expected_targets,
        visible_histories=visible_histories,
    )
    source_hashes = {
        "interactions": sha256_file(prepared.source_interactions),
        "recbole_atomic": sha256_file(prepared.atomic_path),
        "protocol_split": sha256_file(prepared.split_path),
    }
    if prepared.source_items is not None:
        source_hashes["items"] = sha256_file(prepared.source_items)
    source_hashes["checkpoint"] = sha256_file(checkpoint_path)
    manifest = CandidateArtifactManifest(
        dataset=prepared.name,
        protocol=prepared.protocol,
        retriever=spec.retriever,
        backend=f"recbole-{recbole.__version__}",
        model_seed=spec.seed,
        candidate_k=spec.max_candidate_k,
        candidate_file=candidate_path.name,
        target_file=target_path.name,
        source_hashes=source_hashes,
        model_config={
            "model": spec.model,
            "epochs": spec.epochs,
            "stopping_step": spec.stopping_step,
            "train_batch_size": spec.train_batch_size,
            "eval_batch_size": spec.eval_batch_size,
            "learning_rate": spec.learning_rate,
            **dict(spec.model_parameters),
        },
        environment=environment,
        split_statistics={
            **dict(prepared.statistics),
            **mapping_statistics,
            **export_stats,
        },
        candidate_sha256=sha256_file(candidate_path),
        target_sha256=sha256_file(target_path),
        notes=(
            "Formal backend; legacy custom PyTorch BPR is excluded.",
            "Scores are full-sort model outputs; masked history items are never exported.",
        ),
    )
    manifest_path = output_dir / "manifest.json"
    manifest.write(manifest_path)
    result = {
        "dataset": prepared.name,
        "model": spec.model,
        "retriever": spec.retriever,
        "seed": spec.seed,
        "best_valid_score": float(best_valid_score),
        "best_valid_result": dict(best_valid_result),
        "test_result": dict(test_result),
        "elapsed_seconds": float(elapsed),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "validation_only": False,
        "run_signature": run_signature,
        **export_stats,
    }
    (output_dir / "run_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer.tensorboard.close()
    return result


def train_recbole_and_export(
    prepared: PreparedRecBoleDataset,
    spec: RecBoleRunSpec,
    output_dir: Path | str,
    *,
    export_users: Optional[Iterable[str]] = None,
    validation_only: bool = False,
) -> Dict[str, Any]:
    """Train/export while confining RecBole's hard-coded logs to the run."""

    resolved_output = Path(output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    try:
        os.chdir(resolved_output)
        return _train_recbole_and_export_in_workdir(
            prepared,
            spec,
            resolved_output,
            export_users=export_users,
            validation_only=validation_only,
        )
    finally:
        os.chdir(original_directory)
