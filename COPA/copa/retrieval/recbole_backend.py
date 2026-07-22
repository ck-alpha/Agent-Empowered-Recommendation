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
    export_validation_candidates: bool = False,
) -> str:
    """Fingerprint all settings that can change a trained/evaluated run."""

    benchmark_hashes = {
        path.name: sha256_file(path)
        for path in _benchmark_paths(prepared, sequential=spec.model.lower() == "sasrec")
    }
    payload = {
        "dataset": prepared.name,
        "protocol": prepared.protocol,
        "atomic_sha256": sha256_file(prepared.atomic_path),
        "split_sha256": sha256_file(prepared.split_path),
        "benchmark_sha256": benchmark_hashes,
        "protocol_component_hashes": _protocol_component_hashes(prepared),
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
        "export_validation_candidates": bool(export_validation_candidates),
    }
    if spec.model.lower() == "sasrec":
        # Explicit token_seq benchmarks need a shared target/history item-ID
        # vocabulary. This marker invalidates artifacts produced before that
        # mapping invariant was enforced without retraining other backends.
        payload["sequential_item_id_alias"] = "item_id_list"
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_prepared_dataset(
    data_root: Path | str,
    dataset_name: str,
    *,
    source_interactions: Path | str,
    source_items: Path | str | None = None,
    protocol: str = "deduplicated_5core_frozen_train_leave_one_valid_two_test_v2",
) -> PreparedRecBoleDataset:
    data_root = Path(data_root).resolve()
    dataset_dir = data_root / dataset_name
    atomic_path = dataset_dir / f"{dataset_name}.inter"
    split_path = dataset_dir / f"{dataset_name}_split.parquet"
    stats_path = dataset_dir / f"{dataset_name}_split_statistics.json"
    benchmark_paths = [
        dataset_dir / f"{dataset_name}.{suffix}.inter"
        for suffix in (
            "train",
            "valid",
            "test",
            "sasrec_train",
            "sasrec_valid",
            "sasrec_test",
        )
    ]
    for path in (atomic_path, split_path, stats_path, *benchmark_paths):
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
    protocol: str = "deduplicated_5core_frozen_train_leave_one_valid_two_test_v2",
    k_core: int = 5,
    items_path: Path | str | None = None,
) -> PreparedRecBoleDataset:
    """Create frozen-train train/valid/two-positive-test benchmarks.

    The canonical atomic file is retained for auditing. RecBole consumes the
    explicit benchmark files so no backend can silently reinterpret the split.
    """

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
    if (counts < 5).any():
        raise ValueError(
            "Frozen train/validation/two-positive-test requires at least five items per user"
        )
    frame["split"] = "train"
    test_index = frame.groupby("user_id", sort=False).tail(2).index
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
    for split_name, split_frame in (("train", train), ("valid", valid), ("test", test)):
        benchmark = split_frame[["user_id", "item_id", "timestamp"]].copy()
        benchmark.columns = ["user_id:token", "item_id:token", "timestamp:float"]
        benchmark.to_csv(
            dataset_dir / f"{dataset_name}.{split_name}.inter", sep="\t", index=False
        )

    sequential_frames = _build_sequential_benchmarks(frame)
    for split_name, benchmark in sequential_frames.items():
        benchmark.to_csv(
            dataset_dir / f"{dataset_name}.sasrec_{split_name}.inter",
            sep="\t",
            index=False,
        )

    train_items = set(train["item_id"])
    test_positive_counts = test.groupby("user_id")["item_id"].size()
    train_history_counts = train.groupby("user_id")["item_id"].size()
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
        "test_positives_per_user_min": int(test_positive_counts.min()),
        "test_positives_per_user_max": int(test_positive_counts.max()),
        "train_history_per_user_min": int(train_history_counts.min()),
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


def _build_sequential_benchmarks(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build SASRec samples with one frozen train history for valid and test."""

    columns = [
        "user_id:token",
        "item_id_list:token_seq",
        "item_id:token",
        "timestamp:float",
    ]
    rows: Dict[str, list[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for user_id, group in frame.groupby("user_id", sort=True):
        ordered = group.sort_values(["timestamp", "item_id"], kind="mergesort")
        train = ordered[ordered["split"] == "train"]
        valid = ordered[ordered["split"] == "valid"]
        test = ordered[ordered["split"] == "test"]
        train_items = train["item_id"].astype(str).tolist()
        if len(train_items) < 2 or len(valid) != 1 or len(test) != 2:
            raise ValueError(f"Invalid frozen temporal split for user {user_id}")
        if any(any(character.isspace() for character in item) for item in train_items):
            raise ValueError("RecBole token_seq cannot encode item IDs containing whitespace")

        for position in range(1, len(train_items)):
            target = train.iloc[position]
            rows["train"].append(
                {
                    columns[0]: str(user_id),
                    columns[1]: " ".join(train_items[:position]),
                    columns[2]: train_items[position],
                    columns[3]: float(target["timestamp"]),
                }
            )
        frozen_history = " ".join(train_items)
        for split_name, targets in (("valid", valid), ("test", test)):
            for _, target in targets.iterrows():
                rows[split_name].append(
                    {
                        columns[0]: str(user_id),
                        columns[1]: frozen_history,
                        columns[2]: str(target["item_id"]),
                        columns[3]: float(target["timestamp"]),
                    }
                )
    return {name: pd.DataFrame(values, columns=columns) for name, values in rows.items()}


def _benchmark_paths(
    prepared: PreparedRecBoleDataset, *, sequential: bool
) -> tuple[Path, Path, Path]:
    prefix = "sasrec_" if sequential else ""
    dataset_dir = prepared.data_root / prepared.name
    return tuple(
        dataset_dir / f"{prepared.name}.{prefix}{split_name}.inter"
        for split_name in ("train", "valid", "test")
    )


def _stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _protocol_component_hashes(prepared: PreparedRecBoleDataset) -> Dict[str, str]:
    split = pd.read_parquet(
        prepared.split_path, columns=["user_id", "item_id", "timestamp", "split"]
    )
    split["user_id"] = split["user_id"].astype(str)
    split["item_id"] = split["item_id"].astype(str)
    train = split[split["split"] == "train"].sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    )
    test = split[split["split"] == "test"].sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    )
    cutoffs = (
        train.groupby("user_id", sort=True)["timestamp"].max().astype(float).to_dict()
    )
    return {
        "split_policy": _stable_payload_hash(
            {
                "protocol": prepared.protocol,
                "train": "all_before_last_three",
                "validation": "third_from_last",
                "test": "last_two",
                "request_statistics": "train_only",
            }
        ),
        "positive_set": _stable_payload_hash(
            test[["user_id", "item_id", "timestamp"]].to_dict("records")
        ),
        "train_catalog": _stable_payload_hash(sorted(set(train["item_id"]))),
        "request_cutoff": _stable_payload_hash(cutoffs),
    }


def training_visible_statistics(
    prepared: PreparedRecBoleDataset,
    items_path: Path | str,
) -> tuple[Dict[str, float], Dict[str, set[str]], Dict[str, float]]:
    """Compute test-time popularity, histories, and budgets without test labels."""

    split = pd.read_parquet(prepared.split_path)
    visible = split[split["split"] == "train"].copy()
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
    catalog_items = max(1, int(prepared.statistics.get("train_items", 100)))
    evaluation_topk = sorted({min(value, catalog_items) for value in (10, 50, 100)})
    validation_k = max(evaluation_topk)
    config: Dict[str, Any] = {
        "data_path": str(prepared.data_root),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {
            "inter": (
                ["user_id", "item_id_list", "item_id", "timestamp"]
                if sequential
                else ["user_id", "item_id", "timestamp"]
            )
        },
        "benchmark_filename": (
            ["sasrec_train", "sasrec_valid", "sasrec_test"]
            if sequential
            else ["train", "valid", "test"]
        ),
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "split": None,
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
                "alias_of_item_id": ["item_id_list"],
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


def _install_train_catalog_evaluation_mask(
    trainer: "Any",
    dataset: "Any",
    train_item_ids: set[str],
) -> None:
    """Mask non-train items during RecBole validation/checkpoint selection."""

    import types
    import torch

    token_to_internal = dataset.field2token_id[dataset.iid_field]
    eligible = torch.zeros(dataset.item_num, dtype=torch.bool, device=trainer.device)
    internal_ids = sorted(
        int(token_to_internal[item_id])
        for item_id in train_item_ids
        if item_id in token_to_internal
    )
    if not internal_ids:
        raise RuntimeError("Training catalog is empty in RecBole ID space")
    eligible[internal_ids] = True
    original = trainer._full_sort_batch_eval

    def masked_full_sort_batch_eval(self: "Any", batched_data: "Any"):
        interaction, scores, positive_u, positive_i = original(batched_data)
        scores[:, ~eligible.to(scores.device)] = -torch.inf
        return interaction, scores, positive_u, positive_i

    trainer._full_sort_batch_eval = types.MethodType(
        masked_full_sort_batch_eval, trainer
    )


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


def _truncate_precomputed_sequences(
    dataset: "Any", maximum_length: int
) -> Dict[str, int]:
    """Keep the most recent tokens in explicit SASRec benchmark histories."""

    if maximum_length <= 0:
        raise ValueError("MAX_ITEM_LIST_LENGTH must be positive")
    sequence_field = dataset.item_id_list_field
    length_field = dataset.item_list_length_field
    if not np.array_equal(
        dataset.field2id_token[sequence_field],
        dataset.field2id_token[dataset.iid_field],
    ):
        raise RuntimeError(
            "SASRec history and target fields do not share one item-ID vocabulary"
        )
    before = int(dataset.inter_feat[length_field].max())
    dataset.inter_feat[sequence_field] = dataset.inter_feat[sequence_field].map(
        lambda values: np.asarray(values)[-int(maximum_length) :].copy()
    )
    dataset.inter_feat[length_field] = dataset.inter_feat[sequence_field].map(len)
    after = int(dataset.inter_feat[length_field].max())
    if after > maximum_length:
        raise RuntimeError("Explicit sequential benchmark truncation failed")
    return {
        "sasrec_sequence_length_before_truncation": before,
        "sasrec_sequence_length_after_truncation": after,
    }


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
    expected_targets: Mapping[str, Sequence[Mapping[str, Any]]],
    visible_histories: Mapping[str, set[str]],
) -> tuple[Path, Path, Dict[str, Any]]:
    import torch

    model.eval()
    trainer.model = model
    trainer.tot_item_num = test_data._dataset.item_num
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    item_tokens = np.asarray(dataset.field2id_token[iid_field]).astype(str)
    token_to_internal = dataset.field2token_id[iid_field]
    catalog_internal = {
        int(token_to_internal[item_id])
        for item_id in train_item_ids
        if item_id in token_to_internal
    }
    score_rows: Dict[str, Any] = {}
    observed_targets: Dict[str, set[str]] = {}
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
            for original_position in selected_positions:
                user_id = str(all_user_tokens[original_position])
                observed_targets.setdefault(user_id, set()).update(
                    str(item_tokens[int(internal_id)])
                    for internal_id in positive_i_np[positive_u_np == original_position]
                )

            selected_interaction = interaction[selected_positions].to(trainer.device)
            try:
                scores = model.full_sort_predict(selected_interaction)
            except NotImplementedError as exc:
                raise RuntimeError(
                    f"Formal retriever {spec.model} must implement full_sort_predict"
                ) from exc
            scores = scores.view(len(selected_positions), trainer.tot_item_num)
            eligible = torch.zeros(
                trainer.tot_item_num, dtype=torch.bool, device=scores.device
            )
            if catalog_internal:
                eligible[list(sorted(catalog_internal))] = True
            scores[:, ~eligible] = -torch.inf
            user_tokens = all_user_tokens[selected_positions]
            # RecBole's general and sequential full-sort loaders do not expose
            # identical history masks. Apply the independently audited protocol
            # mask here so all three formal backends have exactly the same
            # candidate eligibility rule.
            scores = scores.clone()
            for row_index, user_id in enumerate(user_tokens):
                seen_internal = [
                    int(token_to_internal[item_id])
                    for item_id in visible_histories.get(str(user_id), set())
                    if item_id in token_to_internal
                ]
                if seen_internal:
                    scores[row_index, seen_internal] = -torch.inf
            for row_index, user_id in enumerate(user_tokens):
                user_id = str(user_id)
                row_scores = scores[row_index].detach().cpu()
                if user_id in score_rows:
                    previous = score_rows[user_id]
                    if not torch.equal(torch.isfinite(previous), torch.isfinite(row_scores)) or not torch.allclose(
                        previous[torch.isfinite(previous)],
                        row_scores[torch.isfinite(row_scores)],
                        rtol=1e-6,
                        atol=1e-7,
                    ):
                        raise RuntimeError(
                            f"Test queries for {user_id} do not share one frozen history"
                        )
                else:
                    score_rows[user_id] = row_scores

    expected_users = {
        str(user_id)
        for user_id in expected_targets
        if export_users is None or str(user_id) in export_users
    }
    if set(score_rows) != expected_users:
        raise RuntimeError(
            "RecBole test-query users disagree with the protocol split: "
            f"missing={sorted(expected_users - set(score_rows))[:5]}, "
            f"extra={sorted(set(score_rows) - expected_users)[:5]}"
        )

    candidate_rows: list[Dict[str, Any]] = []
    target_rows: list[Dict[str, Any]] = []
    ranked_by_user: Dict[str, list[str]] = {}
    for user_id in sorted(score_rows):
        expected_ids = {
            str(target["target_item_id"]) for target in expected_targets[user_id]
        }
        if observed_targets.get(user_id, set()) != expected_ids:
            raise RuntimeError(
                f"RecBole target/split disagreement for {user_id}: "
                f"{sorted(observed_targets.get(user_id, set()))} != {sorted(expected_ids)}"
            )
        row_scores = score_rows[user_id]
        normalized, _, _ = _normalize_scores(row_scores.view(1, -1))
        row_base = normalized[0]
        finite_count = int(torch.isfinite(row_scores).sum().item())
        top_count = min(int(spec.max_candidate_k), finite_count)
        if top_count <= 0:
            raise RuntimeError(f"No train-catalog candidates remain for user {user_id}")
        order = torch.argsort(row_scores, descending=True, stable=True)[:top_count]
        top_indices = order.detach().cpu().numpy().astype(int)
        ranked_by_user[user_id] = [str(item_tokens[index]) for index in top_indices]
        seen_overlap = set(ranked_by_user[user_id]) & visible_histories.get(user_id, set())
        if seen_overlap:
            raise RuntimeError(
                f"Full-sort export contains seen items for {user_id}: "
                f"{sorted(seen_overlap)[:5]}"
            )
        raw_values = row_scores[order].detach().cpu().numpy().astype(float)
        base_values = row_base[order].detach().cpu().numpy().astype(float)
        for rank, (internal_id, raw_score, base_score) in enumerate(
            zip(top_indices, raw_values, base_values), start=1
        ):
            candidate_rows.append(
                {
                    "user_id": user_id,
                    "item_id": str(item_tokens[internal_id]),
                    "raw_score": float(raw_score),
                    "base_score": float(base_score),
                    "retrieval_rank": int(rank),
                    "retriever": spec.retriever,
                    "backend": f"recbole-{__import__('recbole').__version__}",
                    "model_seed": int(spec.seed),
                }
            )
        for target in expected_targets[user_id]:
            target_id = str(target["target_item_id"])
            covered = target_id in train_item_ids and target_id in token_to_internal
            target_score: Optional[float] = None
            target_base: Optional[float] = None
            target_rank: Optional[int] = None
            if covered:
                target_internal = int(token_to_internal[target_id])
                target_score = float(row_scores[target_internal])
                target_base = float(row_base[target_internal])
                if not np.isfinite(target_score) or not np.isfinite(target_base):
                    raise RuntimeError(f"Covered test target was masked for user {user_id}")
                greater = int((row_scores > row_scores[target_internal]).sum().item())
                equal_before = int(
                    (
                        (row_scores[:target_internal] == row_scores[target_internal])
                        & torch.isfinite(row_scores[:target_internal])
                    ).sum().item()
                )
                target_rank = 1 + greater + equal_before
            target_rows.append(
                {
                    "user_id": user_id,
                    "target_item_id": target_id,
                    "target_order": int(target["target_order"]),
                    "target_timestamp": float(target["target_timestamp"]),
                    "target_raw_score": target_score,
                    "target_base_score": target_base,
                    "target_full_rank": target_rank,
                    "target_model_covered": bool(covered),
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    targets = pd.DataFrame(target_rows)
    if candidates.empty or targets.empty:
        raise RuntimeError("Full-sort export produced no candidates")
    candidate_path = output_dir / "candidates.parquet"
    target_path = output_dir / "targets.parquet"
    candidates.to_parquet(candidate_path, index=False)
    targets.to_parquet(target_path, index=False)
    full_sort_metrics = _multi_positive_full_sort_metrics(
        ranked_by_user, expected_targets, top_ks=(10, 50, 100)
    )
    return candidate_path, target_path, {
        "exported_users": int(len(score_rows)),
        "candidate_rows": int(len(candidates)),
        "target_model_coverage": float(targets["target_model_covered"].mean()),
        "test_positive_pairs": int(len(targets)),
        **full_sort_metrics,
    }


def _multi_positive_full_sort_metrics(
    ranked_by_user: Mapping[str, Sequence[str]],
    expected_targets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    top_ks: Sequence[int],
) -> Dict[str, float]:
    """Evaluate all models against the same two-positive frozen query."""

    metrics: Dict[str, float] = {}
    for top_k in top_ks:
        recalls: list[float] = []
        ndcgs: list[float] = []
        for user_id, ranking in ranked_by_user.items():
            positives = {
                str(target["target_item_id"]) for target in expected_targets[user_id]
            }
            prefix = list(ranking[: int(top_k)])
            recalls.append(len(positives & set(prefix)) / len(positives))
            dcg = sum(
                1.0 / np.log2(index + 2.0)
                for index, item_id in enumerate(prefix)
                if item_id in positives
            )
            ideal = sum(
                1.0 / np.log2(index + 2.0)
                for index in range(min(len(positives), int(top_k)))
            )
            ndcgs.append(dcg / ideal if ideal else 0.0)
        metrics[f"recall@{int(top_k)}"] = float(np.mean(recalls))
        metrics[f"ndcg@{int(top_k)}"] = float(np.mean(ndcgs))
    return metrics


def _train_recbole_and_export_in_workdir(
    prepared: PreparedRecBoleDataset,
    spec: RecBoleRunSpec,
    output_dir: Path | str,
    *,
    export_users: Optional[Iterable[str]] = None,
    validation_only: bool = False,
    export_validation_candidates: bool = False,
) -> Dict[str, Any]:
    """Train one mature RecBole model, evaluate it, and export ranked candidates."""

    numpy_compatibility_aliases = ensure_recbole_numpy2_compatibility()
    run_signature = recbole_run_signature(
        prepared,
        spec,
        validation_only=validation_only,
        export_validation_candidates=export_validation_candidates,
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
    if spec.model.lower() == "sasrec":
        mapping_statistics.update(
            _truncate_precomputed_sequences(
                dataset,
                int(spec.model_parameters.get("MAX_ITEM_LIST_LENGTH", 50)),
            )
        )
    train_data, valid_data, test_data = data_preparation(config, dataset)
    split = pd.read_parquet(prepared.split_path)
    split["user_id"] = split["user_id"].astype(str)
    split["item_id"] = split["item_id"].astype(str)
    train_item_ids = set(
        split.loc[split["split"] == "train", "item_id"].astype(str)
    )
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    _install_train_catalog_evaluation_mask(trainer, dataset, train_item_ids)
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

    test_targets = split.loc[
        split["split"] == "test", ["user_id", "item_id", "timestamp"]
    ].sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    test_targets["target_order"] = test_targets.groupby("user_id").cumcount() + 1
    expected_targets = {
        str(user_id): [
            {
                "target_item_id": str(row["item_id"]),
                "target_timestamp": float(row["timestamp"]),
                "target_order": int(row["target_order"]),
            }
            for row in group.to_dict("records")
        ]
        for user_id, group in test_targets.groupby("user_id", sort=True)
    }
    visible_histories = {
        str(user_id): set(group["item_id"].astype(str))
        for user_id, group in split[split["split"] == "train"].groupby("user_id")
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
    validation_export: Dict[str, Any] = {}
    if export_validation_candidates:
        validation_targets = split.loc[
            split["split"] == "valid", ["user_id", "item_id", "timestamp"]
        ].sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
        expected_validation_targets = {
            str(row["user_id"]): [
                {
                    "target_item_id": str(row["item_id"]),
                    "target_timestamp": float(row["timestamp"]),
                    "target_order": 1,
                }
            ]
            for row in validation_targets.to_dict("records")
        }
        validation_dir = output_dir / "validation_candidates"
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_candidate_path, validation_target_path, validation_stats = (
            _export_candidates(
                trainer,
                model,
                dataset,
                valid_data,
                output_dir=validation_dir,
                spec=spec,
                export_users=(
                    {str(value) for value in export_users}
                    if export_users is not None
                    else None
                ),
                train_item_ids=train_item_ids,
                expected_targets=expected_validation_targets,
                visible_histories=visible_histories,
            )
        )
        validation_export = {
            "validation_candidate_file": str(validation_candidate_path),
            "validation_target_file": str(validation_target_path),
            "validation_candidate_sha256": sha256_file(validation_candidate_path),
            "validation_target_sha256": sha256_file(validation_target_path),
            **{
                f"validation_{key}": value for key, value in validation_stats.items()
            },
        }
    elapsed = perf_counter() - started
    test_result = {
        key: value
        for key, value in export_stats.items()
        if key.startswith("recall@") or key.startswith("ndcg@")
    }
    protocol_hashes = _protocol_component_hashes(prepared)
    source_hashes = {
        "interactions": sha256_file(prepared.source_interactions),
        "recbole_atomic": sha256_file(prepared.atomic_path),
        "protocol_split": sha256_file(prepared.split_path),
        **protocol_hashes,
        "constraint_calibration": _stable_payload_hash(
            spec.model_parameters.get("constraint_calibration", "not_calibrated")
        ),
    }
    if validation_export:
        source_hashes["validation_candidates"] = str(
            validation_export["validation_candidate_sha256"]
        )
        source_hashes["validation_targets"] = str(
            validation_export["validation_target_sha256"]
        )
    for benchmark_path in _benchmark_paths(
        prepared, sequential=spec.model.lower() == "sasrec"
    ):
        source_hashes[f"benchmark_{benchmark_path.stem.split('.')[-1]}"] = sha256_file(
            benchmark_path
        )
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
            "Scores are full-sort model outputs over the frozen train catalog.",
            "Checkpoint selection masks validation/test-only catalog items.",
            "Validation and both test positives are excluded from request construction.",
            "SASRec and general recommenders use the same multi-positive full-sort evaluator.",
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
        **validation_export,
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
    export_validation_candidates: bool = False,
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
            export_validation_candidates=export_validation_candidates,
        )
    finally:
        os.chdir(original_directory)
