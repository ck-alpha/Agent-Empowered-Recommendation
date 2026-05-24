"""
Scenario-1 BPR diagnostics and PyTorch tuning runner.

This script is intentionally separate from run_scenario1_baselines.py. It
answers a narrower question: why does BPR fail to provide useful personalized
recall on the current scenario-1 datasets?

Outputs include:
- filter_stats.csv: dedup/k-core/split data funnel statistics
- epoch_metrics.csv: per-epoch train loss, sampled train AUC, validation metrics
- config_summary.csv: best validation metrics per BPR config
- final_test_summary.csv: final test comparison against BPR, popularity, and Item-KNN
- diagnostics.json: machine-readable run metadata
- analysis.md: compact human-readable conclusion
- several PNG figures for training and data diagnostics
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

# Allow running as: python src/run_scenario1_bpr_diagnostics.py
sys.path.insert(0, os.path.dirname(__file__))

from run_scenario1_baselines import (  # noqa: E402
    IdMappings,
    build_id_mappings,
    build_user_item_matrix,
    load_scenario1_tables,
    train_item_knn_model,
)


LOGGER = logging.getLogger(__name__)
EPS = 1e-12
DEFAULT_OUTPUT_DIR = "results/scenario1_bpr_diagnostics/electronics_sample1m"


@dataclass(frozen=True)
class TorchBPRConfig:
    config_id: str
    optimizer: str
    embedding_size: int
    learning_rate: float
    regularization: float
    batch_size: int
    negative_sampler: str
    epochs: int
    stage: str


@dataclass
class MetricSummary:
    num_users: int
    covered_users: int
    recall_count: float
    hr_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    covered_hr_at_k: float
    covered_ndcg_at_k: float
    covered_mrr_at_k: float
    ground_truth_coverage_rate: float


class TorchBPR(torch.nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_size: int):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users, embedding_size)
        self.item_factors = torch.nn.Embedding(num_items, embedding_size)
        torch.nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, mean=0.0, std=0.01)

    def pair_scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_factors(users) * self.item_factors(items)).sum(dim=1)


def parse_csv_values(text: str, cast: Callable[[str], Any]) -> List[Any]:
    return [cast(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and tune scenario-1 BPR recall.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_prefix", default="electronics_scenario1_sample1m")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k_core", type=int, default=5)
    parser.add_argument("--recall_k", type=int, default=200)
    parser.add_argument("--eval_users", type=int, default=1000, help="Sampled validation/test users. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU fallback when CUDA is unavailable.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny diagnostic path for validation.")

    parser.add_argument("--optimizers", default="adam")
    parser.add_argument("--embedding_sizes", default="64,128")
    parser.add_argument("--learning_rates", default="0.0005,0.001")
    parser.add_argument("--regularizations", default="0.00001,0.0001")
    parser.add_argument("--negative_samplers", default="uniform,popularity,mixed")
    parser.add_argument("--batch_sizes", default="8192")
    parser.add_argument("--stage1_epochs", type=int, default=30)
    parser.add_argument("--stage2_epochs", type=int, default=150)
    parser.add_argument("--top_configs", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--max_stage1_configs", type=int, default=0, help="Optional cap for smoke/debug runs.")

    parser.add_argument("--item_knn_neighbors", type=int, default=100)
    parser.add_argument("--item_knn_weighting", choices=["bm25", "cosine", "tfidf"], default="bm25")
    parser.add_argument("--skip_comparisons", action="store_true")
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.stage1_epochs = min(args.stage1_epochs, 2)
    args.stage2_epochs = min(args.stage2_epochs, 2)
    args.top_configs = 1
    args.max_stage1_configs = 1
    args.eval_users = min(args.eval_users if args.eval_users > 0 else 100, 100)
    args.output_dir = args.output_dir or "/tmp/scenario1_bpr_diagnostics_smoke"
    return args


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")
        LOGGER.warning("CUDA unavailable; falling back to CPU because --allow_cpu was set.")
        return torch.device("cpu")
    return torch.device(args.device)


def density(rows: int, users: int, items: int) -> float:
    denom = max(1, users * items)
    return float(rows / denom)


def interaction_quantiles(counts: pd.Series, prefix: str) -> Dict[str, float]:
    if counts.empty:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    q = counts.astype(float).quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return {
        f"{prefix}_min": float(q.loc[0.0]),
        f"{prefix}_p25": float(q.loc[0.25]),
        f"{prefix}_median": float(q.loc[0.5]),
        f"{prefix}_p75": float(q.loc[0.75]),
        f"{prefix}_p90": float(q.loc[0.9]),
        f"{prefix}_p99": float(q.loc[0.99]),
        f"{prefix}_max": float(q.loc[1.0]),
    }


def make_filter_stat(
    step: str,
    interactions: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    train_items: Optional[set[str]] = None,
) -> Dict[str, Any]:
    rows = int(len(interactions))
    users = int(interactions["user_id"].nunique()) if rows else 0
    items = int(interactions["item_id"].nunique()) if rows else 0
    duplicate_rows = int(interactions.duplicated(["user_id", "item_id"]).sum()) if rows else 0
    user_counts = interactions.groupby("user_id").size() if rows else pd.Series(dtype=int)
    item_counts = interactions.groupby("item_id").size() if rows else pd.Series(dtype=int)
    out: Dict[str, Any] = {
        "step": step,
        "rows": rows,
        "users": users,
        "items": items,
        "density": density(rows, users, items),
        "duplicate_user_item_rows": duplicate_rows,
        "val_ground_truth_coverage_rate": np.nan,
        "test_ground_truth_coverage_rate": np.nan,
    }
    out.update(interaction_quantiles(user_counts, "user_interactions"))
    out.update(interaction_quantiles(item_counts, "item_interactions"))
    if train_items is not None and val_df is not None and len(val_df):
        out["val_ground_truth_coverage_rate"] = float(val_df["item_id"].astype(str).isin(train_items).mean())
    if train_items is not None and test_df is not None and len(test_df):
        out["test_ground_truth_coverage_rate"] = float(test_df["item_id"].astype(str).isin(train_items).mean())
    return out


def deduplicate_latest(interactions: pd.DataFrame) -> pd.DataFrame:
    ordered = interactions.sort_values(["user_id", "item_id", "timestamp"]).copy()
    return ordered.drop_duplicates(["user_id", "item_id"], keep="last").reset_index(drop=True)


def iterative_k_core(
    interactions: pd.DataFrame,
    k: int,
    stats: List[Dict[str, Any]],
) -> pd.DataFrame:
    if k <= 1:
        stats.append(make_filter_stat("kcore_skipped", interactions))
        return interactions.copy().reset_index(drop=True)

    current = interactions.copy()
    iteration = 0
    while True:
        iteration += 1
        before = len(current)
        user_counts = current["user_id"].value_counts()
        item_counts = current["item_id"].value_counts()
        keep_users = set(user_counts[user_counts >= k].index.astype(str))
        keep_items = set(item_counts[item_counts >= k].index.astype(str))
        current = current[
            current["user_id"].isin(keep_users) & current["item_id"].isin(keep_items)
        ].copy()
        stats.append(make_filter_stat(f"kcore_{k}_iter_{iteration}", current))
        if len(current) == before:
            break
        if current.empty:
            raise ValueError(f"k-core filtering with k={k} removed all interactions.")
    return current.reset_index(drop=True)


def temporal_train_val_test_split(interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = interactions["user_id"].value_counts()
    eligible_users = counts[counts >= 3].index
    eligible = interactions[interactions["user_id"].isin(eligible_users)].copy()
    eligible = eligible.sort_values(["user_id", "timestamp", "item_id"])
    if eligible.empty:
        raise ValueError("No users with at least 3 interactions for train/validation/test split.")

    test_idx = eligible.groupby("user_id", sort=False).tail(1).index
    test = eligible.loc[test_idx, ["user_id", "item_id", "timestamp"]].copy()
    remain = eligible.drop(index=test_idx)
    val_idx = remain.groupby("user_id", sort=False).tail(1).index
    val = remain.loc[val_idx, ["user_id", "item_id", "timestamp"]].copy()
    train = remain.drop(index=val_idx).copy()
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_internal_eval_frame(eval_df: pd.DataFrame, mappings: IdMappings, sample_users: int, seed: int) -> pd.DataFrame:
    eligible = eval_df[eval_df["user_id"].astype(str).isin(mappings.user_id_to_idx)].copy()
    if sample_users and sample_users > 0 and len(eligible) > sample_users:
        eligible = eligible.sample(n=sample_users, random_state=seed)
    eligible["user_idx"] = eligible["user_id"].map(mappings.user_id_to_idx).astype(np.int64)
    eligible["item_idx"] = eligible["item_id"].map(mappings.item_id_to_idx)
    eligible["item_idx"] = eligible["item_idx"].fillna(-1).astype(np.int64)
    return eligible.reset_index(drop=True)


def train_history_sets(train_df: pd.DataFrame, mappings: IdMappings) -> List[set[int]]:
    sets: List[set[int]] = [set() for _ in range(len(mappings.user_id_to_idx))]
    mapped = train_df[["user_id", "item_id"]].copy()
    mapped["user_idx"] = mapped["user_id"].map(mappings.user_id_to_idx)
    mapped["item_idx"] = mapped["item_id"].map(mappings.item_id_to_idx)
    mapped = mapped.dropna(subset=["user_idx", "item_idx"])
    for row in mapped.itertuples(index=False):
        sets[int(row.user_idx)].add(int(row.item_idx))
    return sets


def train_pairs_array(train_df: pd.DataFrame, mappings: IdMappings) -> np.ndarray:
    mapped = train_df[["user_id", "item_id"]].copy()
    mapped["user_idx"] = mapped["user_id"].map(mappings.user_id_to_idx)
    mapped["item_idx"] = mapped["item_id"].map(mappings.item_id_to_idx)
    mapped = mapped.dropna(subset=["user_idx", "item_idx"])
    return mapped[["user_idx", "item_idx"]].astype(np.int64).to_numpy()


def popularity_distribution(train_pairs: np.ndarray, num_items: int) -> np.ndarray:
    counts = np.bincount(train_pairs[:, 1], minlength=num_items).astype(np.float64)
    weights = np.power(counts + 1.0, 0.75)
    return weights / max(EPS, weights.sum())


def sample_negatives(
    users: np.ndarray,
    num_items: int,
    user_pos_sets: Sequence[set[int]],
    rng: np.random.Generator,
    sampler: str,
    pop_probs: np.ndarray,
) -> Tuple[np.ndarray, float]:
    size = len(users)

    def draw(n: int) -> np.ndarray:
        if sampler == "uniform":
            return rng.integers(0, num_items, size=n, dtype=np.int64)
        if sampler == "popularity":
            return rng.choice(num_items, size=n, replace=True, p=pop_probs).astype(np.int64)
        if sampler == "mixed":
            mask = rng.random(n) < 0.5
            out = rng.integers(0, num_items, size=n, dtype=np.int64)
            if mask.any():
                out[mask] = rng.choice(num_items, size=int(mask.sum()), replace=True, p=pop_probs).astype(np.int64)
            return out
        raise ValueError(f"Unsupported negative sampler: {sampler}")

    negatives = draw(size)
    collisions = 0
    for _ in range(20):
        bad = np.fromiter(
            (int(item) in user_pos_sets[int(user)] for user, item in zip(users, negatives)),
            dtype=bool,
            count=size,
        )
        bad_count = int(bad.sum())
        collisions += bad_count
        if bad_count == 0:
            break
        negatives[bad] = draw(bad_count)
    ratio = float(collisions / max(1, size))
    return negatives.astype(np.int64), ratio


def topk_metrics(recommended: Sequence[int], ground_truth: int, k: int) -> Tuple[float, float, float, Optional[int]]:
    if ground_truth < 0:
        return 0.0, 0.0, 0.0, None
    top_items = list(recommended[:k])
    try:
        rank_idx = top_items.index(int(ground_truth))
    except ValueError:
        return 0.0, 0.0, 0.0, None
    rank = rank_idx + 1
    return 1.0, float(1.0 / math.log2(rank_idx + 2)), float(1.0 / rank), rank


def summarize_eval(hits: List[float], ndcgs: List[float], mrrs: List[float], counts: List[int], covered: List[bool]) -> MetricSummary:
    covered_idx = [idx for idx, flag in enumerate(covered) if flag]
    return MetricSummary(
        num_users=len(hits),
        covered_users=len(covered_idx),
        recall_count=float(np.mean(counts)) if counts else 0.0,
        hr_at_k=float(np.mean(hits)) if hits else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        mrr_at_k=float(np.mean(mrrs)) if mrrs else 0.0,
        covered_hr_at_k=float(np.mean([hits[i] for i in covered_idx])) if covered_idx else 0.0,
        covered_ndcg_at_k=float(np.mean([ndcgs[i] for i in covered_idx])) if covered_idx else 0.0,
        covered_mrr_at_k=float(np.mean([mrrs[i] for i in covered_idx])) if covered_idx else 0.0,
        ground_truth_coverage_rate=float(len(covered_idx) / max(1, len(hits))),
    )


@torch.no_grad()
def evaluate_torch_bpr(
    model: TorchBPR,
    eval_frame: pd.DataFrame,
    user_pos_sets: Sequence[set[int]],
    recall_k: int,
    device: torch.device,
    batch_users: int = 128,
) -> MetricSummary:
    model.eval()
    if eval_frame.empty:
        return summarize_eval([], [], [], [], [])
    item_matrix = model.item_factors.weight.detach()
    hits: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []
    counts: List[int] = []
    covered: List[bool] = []
    user_indices = eval_frame["user_idx"].to_numpy(dtype=np.int64)
    gt_indices = eval_frame["item_idx"].to_numpy(dtype=np.int64)
    top_k = min(recall_k, item_matrix.shape[0])

    for start in range(0, len(eval_frame), batch_users):
        end = min(start + batch_users, len(eval_frame))
        batch_user_np = user_indices[start:end]
        users = torch.as_tensor(batch_user_np, dtype=torch.long, device=device)
        scores = model.user_factors(users) @ item_matrix.T
        for local_idx, user_idx in enumerate(batch_user_np):
            positives = user_pos_sets[int(user_idx)]
            if positives:
                pos_tensor = torch.as_tensor(list(positives), dtype=torch.long, device=device)
                scores[local_idx, pos_tensor] = -torch.inf
        top_items = torch.topk(scores, k=top_k, dim=1).indices.detach().cpu().numpy()
        for row_idx, recs in enumerate(top_items):
            gt = int(gt_indices[start + row_idx])
            hit, ndcg, mrr, _ = topk_metrics(recs.tolist(), gt, recall_k)
            hits.append(hit)
            ndcgs.append(ndcg)
            mrrs.append(mrr)
            counts.append(len(recs))
            covered.append(gt >= 0)
    return summarize_eval(hits, ndcgs, mrrs, counts, covered)


def train_torch_bpr(
    config: TorchBPRConfig,
    train_pairs: np.ndarray,
    num_users: int,
    num_items: int,
    user_pos_sets: Sequence[set[int]],
    pop_probs: np.ndarray,
    validation_frame: Optional[pd.DataFrame],
    recall_k: int,
    device: torch.device,
    seed: int,
    early_stopping_patience: int,
) -> Tuple[TorchBPR, List[Dict[str, Any]], Dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TorchBPR(num_users, num_items, config.embedding_size).to(device)
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = -1.0
    best_epoch = -1
    no_improve = 0
    epoch_rows: List[Dict[str, Any]] = []
    started_at = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = rng.permutation(len(train_pairs))
        total_loss = 0.0
        total_auc_correct = 0
        total_pairs = 0
        collision_ratios: List[float] = []
        epoch_started = time.time()

        for start in range(0, len(train_pairs), config.batch_size):
            batch_idx = perm[start : start + config.batch_size]
            batch = train_pairs[batch_idx]
            users_np = batch[:, 0]
            pos_np = batch[:, 1]
            neg_np, collision_ratio = sample_negatives(
                users_np,
                num_items,
                user_pos_sets,
                rng,
                config.negative_sampler,
                pop_probs,
            )
            collision_ratios.append(collision_ratio)

            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(pos_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(neg_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            user_emb = model.user_factors(users)
            pos_emb = model.item_factors(positives)
            neg_emb = model.item_factors(negatives)
            pos_scores = (user_emb * pos_emb).sum(dim=1)
            neg_scores = (user_emb * neg_emb).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            if config.regularization > 0:
                reg = (
                    user_emb.pow(2).sum(dim=1)
                    + pos_emb.pow(2).sum(dim=1)
                    + neg_emb.pow(2).sum(dim=1)
                ).mean()
                loss = loss + config.regularization * reg
            loss.backward()
            optimizer.step()

            batch_size = len(batch)
            total_loss += float(loss.item()) * batch_size
            total_auc_correct += int((pos_scores.detach() > neg_scores.detach()).sum().item())
            total_pairs += batch_size

        validation = None
        if validation_frame is not None:
            validation = evaluate_torch_bpr(model, validation_frame, user_pos_sets, recall_k, device)
            monitor = validation.covered_ndcg_at_k
            if monitor > best_metric:
                best_metric = monitor
                best_epoch = epoch
                no_improve = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                no_improve += 1

        row = {
            "config_id": config.config_id,
            "stage": config.stage,
            "epoch": epoch,
            "optimizer": config.optimizer,
            "embedding_size": config.embedding_size,
            "learning_rate": config.learning_rate,
            "regularization": config.regularization,
            "batch_size": config.batch_size,
            "negative_sampler": config.negative_sampler,
            "train_loss": float(total_loss / max(1, total_pairs)),
            "sampled_train_auc": float(total_auc_correct / max(1, total_pairs)),
            "negative_collision_ratio": float(np.mean(collision_ratios)) if collision_ratios else 0.0,
            "epoch_seconds": float(time.time() - epoch_started),
            "validation_hr_at_200": validation.hr_at_k if validation else np.nan,
            "validation_ndcg_at_200": validation.ndcg_at_k if validation else np.nan,
            "validation_mrr_at_200": validation.mrr_at_k if validation else np.nan,
            "validation_covered_hr_at_200": validation.covered_hr_at_k if validation else np.nan,
            "validation_covered_ndcg_at_200": validation.covered_ndcg_at_k if validation else np.nan,
            "validation_covered_mrr_at_200": validation.covered_mrr_at_k if validation else np.nan,
            "validation_coverage_rate": validation.ground_truth_coverage_rate if validation else np.nan,
        }
        epoch_rows.append(row)
        LOGGER.info(
            "[%s] epoch=%s loss=%.5f train_auc=%.4f val_cov_ndcg=%.5f",
            config.config_id,
            epoch,
            row["train_loss"],
            row["sampled_train_auc"],
            row["validation_covered_ndcg_at_200"],
        )

        if validation_frame is not None and early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            LOGGER.info("[%s] early stopping at epoch %s.", config.config_id, epoch)
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    else:
        best_epoch = len(epoch_rows)
        best_metric = float(epoch_rows[-1].get("validation_covered_ndcg_at_200", 0.0)) if epoch_rows else 0.0

    summary = {
        **asdict(config),
        "completed_epochs": len(epoch_rows),
        "best_epoch": int(best_epoch),
        "best_validation_covered_ndcg_at_200": float(best_metric),
        "fit_seconds": float(time.time() - started_at),
    }
    if epoch_rows:
        last = epoch_rows[-1]
        summary.update(
            {
                "last_train_loss": float(last["train_loss"]),
                "last_sampled_train_auc": float(last["sampled_train_auc"]),
                "last_validation_covered_hr_at_200": float(last["validation_covered_hr_at_200"]),
                "last_validation_covered_ndcg_at_200": float(last["validation_covered_ndcg_at_200"]),
                "last_validation_covered_mrr_at_200": float(last["validation_covered_mrr_at_200"]),
            }
        )
    return model, epoch_rows, summary


def build_stage1_configs(args: argparse.Namespace) -> List[TorchBPRConfig]:
    optimizers = parse_csv_values(args.optimizers, str)
    embeddings = parse_csv_values(args.embedding_sizes, int)
    lrs = parse_csv_values(args.learning_rates, float)
    regs = parse_csv_values(args.regularizations, float)
    samplers = parse_csv_values(args.negative_samplers, str)
    batch_sizes = parse_csv_values(args.batch_sizes, int)
    configs: List[TorchBPRConfig] = []
    for idx, values in enumerate(itertools.product(optimizers, embeddings, lrs, regs, samplers, batch_sizes), start=1):
        opt, emb, lr, reg, sampler, batch = values
        configs.append(
            TorchBPRConfig(
                config_id=f"s1_{idx:03d}",
                optimizer=opt,
                embedding_size=emb,
                learning_rate=lr,
                regularization=reg,
                batch_size=batch,
                negative_sampler=sampler,
                epochs=args.stage1_epochs,
                stage="stage1",
            )
        )
    if args.max_stage1_configs and args.max_stage1_configs > 0:
        configs = configs[: args.max_stage1_configs]
    return configs


def build_stage2_configs(stage1_summary: pd.DataFrame, args: argparse.Namespace) -> List[TorchBPRConfig]:
    if stage1_summary.empty:
        return []
    top = stage1_summary.sort_values(
        ["best_validation_covered_ndcg_at_200", "last_validation_covered_ndcg_at_200"],
        ascending=False,
    ).head(args.top_configs)
    configs: List[TorchBPRConfig] = []
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        configs.append(
            TorchBPRConfig(
                config_id=f"s2_{idx:03d}_{row.config_id}",
                optimizer=str(row.optimizer),
                embedding_size=int(row.embedding_size),
                learning_rate=float(row.learning_rate),
                regularization=float(row.regularization),
                batch_size=int(row.batch_size),
                negative_sampler=str(row.negative_sampler),
                epochs=args.stage2_epochs,
                stage="stage2",
            )
        )
    return configs


def evaluate_popularity(
    train_pairs: np.ndarray,
    eval_frame: pd.DataFrame,
    user_pos_sets: Sequence[set[int]],
    num_items: int,
    recall_k: int,
) -> MetricSummary:
    counts = np.bincount(train_pairs[:, 1], minlength=num_items)
    order = np.argsort(-counts, kind="mergesort")
    hits: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []
    recall_counts: List[int] = []
    covered: List[bool] = []
    for row in eval_frame.itertuples(index=False):
        user_idx = int(row.user_idx)
        gt = int(row.item_idx)
        positives = user_pos_sets[user_idx]
        recs: List[int] = []
        for item_idx in order:
            item_idx_int = int(item_idx)
            if item_idx_int in positives:
                continue
            recs.append(item_idx_int)
            if len(recs) >= recall_k:
                break
        hit, ndcg, mrr, _ = topk_metrics(recs, gt, recall_k)
        hits.append(hit)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        recall_counts.append(len(recs))
        covered.append(gt >= 0)
    return summarize_eval(hits, ndcgs, mrrs, recall_counts, covered)


def evaluate_implicit_like(
    model: Any,
    eval_frame: pd.DataFrame,
    user_item_matrix: csr_matrix,
    recall_k: int,
) -> MetricSummary:
    hits: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []
    recall_counts: List[int] = []
    covered: List[bool] = []
    for row in eval_frame.itertuples(index=False):
        user_idx = int(row.user_idx)
        gt = int(row.item_idx)
        if model is None:
            recs = []
        else:
            item_indices, _ = model.recommend(
                user_idx,
                user_item_matrix[user_idx],
                N=recall_k,
                filter_already_liked_items=True,
            )
            recs = np.asarray(item_indices).reshape(-1).astype(int).tolist()
        hit, ndcg, mrr, _ = topk_metrics(recs, gt, recall_k)
        hits.append(hit)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        recall_counts.append(len(recs))
        covered.append(gt >= 0)
    return summarize_eval(hits, ndcgs, mrrs, recall_counts, covered)


def metric_summary_row(method: str, summary: MetricSummary) -> Dict[str, Any]:
    return {"method": method, **asdict(summary)}


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300})


def plot_filter_funnel(filter_df: pd.DataFrame, out_dir: Path) -> None:
    if filter_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=filter_df, x="step", y="rows", ax=ax, color="#4c78a8")
    ax.set_title("Scenario-1 Data Filtering Funnel")
    ax.set_xlabel("Step")
    ax.set_ylabel("Interactions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "filter_funnel.png")
    plt.close(fig)


def plot_interaction_distribution(interactions: pd.DataFrame, out_dir: Path) -> None:
    def draw_count_frequency(ax: plt.Axes, counts: pd.Series, title: str, color: str, entity: str) -> None:
        frequency = counts.value_counts().sort_index()
        x = frequency.index.to_numpy(dtype=float)
        y = frequency.to_numpy(dtype=float)
        widths = np.maximum(0.8, x * 0.08)
        ax.bar(x, y, width=widths, color=color, alpha=0.82, edgecolor="#2f2f2f", linewidth=0.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(max(1.0, float(x.min()) * 0.8), float(x.max()) * 1.2)
        ax.set_ylim(0.8, float(y.max()) * 1.25)
        ax.set_title(title)
        ax.set_xlabel(f"Interactions per {entity}")
        ax.set_ylabel(f"Number of {entity}s")
        summary = (
            f"n={counts.size:,}\n"
            f"median={counts.median():.0f}\n"
            f"p90={counts.quantile(0.90):.0f}\n"
            f"p99={counts.quantile(0.99):.0f}\n"
            f"max={counts.max():.0f}"
        )
        ax.text(
            0.98,
            0.97,
            summary,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.92},
        )

    user_counts = interactions.groupby("user_id").size()
    item_counts = interactions.groupby("item_id").size()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    draw_count_frequency(axes[0], user_counts, "User Interaction Count Frequency", "#59a14f", "user")
    draw_count_frequency(axes[1], item_counts, "Item Interaction Count Frequency", "#f28e2b", "item")
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_distributions.png")
    plt.close(fig)


def plot_epoch_curves(epoch_df: pd.DataFrame, out_dir: Path) -> None:
    if epoch_df.empty:
        return
    for metric, filename, title in [
        ("train_loss", "training_loss_curve.png", "Training BPR Loss"),
        ("sampled_train_auc", "training_auc_curve.png", "Sampled Train AUC"),
        ("validation_covered_ndcg_at_200", "validation_ndcg_curve.png", "Validation Covered NDCG@200"),
        ("validation_covered_hr_at_200", "validation_hr_curve.png", "Validation Covered HR@200"),
        ("validation_covered_mrr_at_200", "validation_mrr_curve.png", "Validation Covered MRR@200"),
    ]:
        if metric not in epoch_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.lineplot(data=epoch_df, x="epoch", y=metric, hue="config_id", style="stage", ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)


def plot_config_diagnostics(config_df: pd.DataFrame, out_dir: Path) -> None:
    if config_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sampler_df = (
        config_df.groupby("negative_sampler", as_index=False)["best_validation_covered_ndcg_at_200"]
        .max()
        .sort_values("best_validation_covered_ndcg_at_200", ascending=False)
    )
    sns.barplot(data=sampler_df, x="negative_sampler", y="best_validation_covered_ndcg_at_200", ax=ax)
    ax.set_title("Best Validation Covered NDCG@200 by Negative Sampler")
    ax.set_xlabel("Negative sampler")
    ax.set_ylabel("Best covered NDCG@200")
    fig.tight_layout()
    fig.savefig(out_dir / "negative_sampler_comparison.png")
    plt.close(fig)

    heat = config_df.pivot_table(
        index="learning_rate",
        columns="regularization",
        values="best_validation_covered_ndcg_at_200",
        aggfunc="max",
    )
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        sns.heatmap(heat, annot=True, fmt=".4f", cmap="viridis", ax=ax)
        ax.set_title("Best Validation Covered NDCG@200 by LR / Regularization")
        fig.tight_layout()
        fig.savefig(out_dir / "lr_regularization_heatmap.png")
        plt.close(fig)


def plot_final_comparison(final_df: pd.DataFrame, out_dir: Path) -> None:
    if final_df.empty:
        return
    metrics = ["covered_hr_at_k", "covered_ndcg_at_k", "covered_mrr_at_k"]
    plot_df = final_df.melt(id_vars=["method"], value_vars=metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=plot_df, x="method", y="score", hue="metric", ax=ax)
    ax.set_title("Final Test Comparison on Train-Item-Covered Users")
    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "final_test_method_comparison.png")
    plt.close(fig)


def write_analysis(
    out_dir: Path,
    filter_df: pd.DataFrame,
    config_df: pd.DataFrame,
    final_df: pd.DataFrame,
    best_config: Mapping[str, Any],
) -> None:
    lines = ["# Scenario-1 BPR Diagnostics", ""]
    if not filter_df.empty:
        last = filter_df.iloc[-1]
        lines.extend(
            [
                "## Data Funnel",
                f"- Final filtered rows/users/items: {int(last['rows'])} / {int(last['users'])} / {int(last['items'])}",
                f"- Final density: {float(last['density']):.8f}",
                f"- Validation coverage: {float(last['val_ground_truth_coverage_rate']):.4f}",
                f"- Test coverage: {float(last['test_ground_truth_coverage_rate']):.4f}",
                "",
            ]
        )
    if best_config:
        lines.extend(
            [
                "## Best BPR Config",
                f"- Config: `{best_config.get('config_id')}`",
                f"- Optimizer: `{best_config.get('optimizer')}`",
                f"- Embedding size: `{best_config.get('embedding_size')}`",
                f"- Learning rate: `{best_config.get('learning_rate')}`",
                f"- Regularization: `{best_config.get('regularization')}`",
                f"- Negative sampler: `{best_config.get('negative_sampler')}`",
                f"- Best epoch: `{best_config.get('best_epoch')}`",
                f"- Best validation covered NDCG@200: `{best_config.get('best_validation_covered_ndcg_at_200'):.6f}`",
                "",
            ]
        )
    if not final_df.empty:
        best_method = final_df.sort_values("covered_ndcg_at_k", ascending=False).iloc[0]
        bpr_row = final_df.loc[final_df["method"] == "bpr"]
        lines.extend(["## Final Test Conclusion"])
        lines.append(
            f"- Best covered-test method by NDCG@K: `{best_method.method}` "
            f"with NDCG={float(best_method.covered_ndcg_at_k):.6f}."
        )
        if not bpr_row.empty:
            bpr_ndcg = float(bpr_row.iloc[0]["covered_ndcg_at_k"])
            pop_row = final_df.loc[final_df["method"] == "popularity"]
            if not pop_row.empty and bpr_ndcg < float(pop_row.iloc[0]["covered_ndcg_at_k"]):
                lines.append(
                    "- BPR remains below popularity on covered-test NDCG; this points to "
                    "data/target mismatch, overly easy negatives, or popularity-dominant demand."
                )
            else:
                lines.append("- BPR beats or matches popularity on covered-test NDCG.")
        lines.append("")
    lines.append("See CSV files and PNG figures in this directory for detailed diagnostics.")
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = apply_smoke_overrides(parse_args())
    device = resolve_device(args)
    LOGGER.info("Using torch=%s cuda=%s device=%s", torch.__version__, torch.version.cuda, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    _, _, interactions = load_scenario1_tables(args.processed_dir, args.output_prefix)
    stats: List[Dict[str, Any]] = [make_filter_stat("loaded", interactions)]
    deduped = deduplicate_latest(interactions)
    stats.append(make_filter_stat("deduplicated_latest", deduped))
    filtered = iterative_k_core(deduped, args.k_core, stats)
    train_df, val_df, test_df = temporal_train_val_test_split(filtered)
    train_items = set(train_df["item_id"].astype(str))
    stats.append(make_filter_stat("train_val_test_split", filtered, val_df, test_df, train_items))
    filter_df = pd.DataFrame(stats)
    filter_df.to_csv(out_dir / "filter_stats.csv", index=False)
    plot_filter_funnel(filter_df, out_dir)
    plot_interaction_distribution(filtered, out_dir)

    mappings = build_id_mappings(train_df)
    train_matrix = build_user_item_matrix(train_df, mappings)
    train_pairs = train_pairs_array(train_df, mappings)
    user_pos = train_history_sets(train_df, mappings)
    pop_probs = popularity_distribution(train_pairs, len(mappings.item_id_to_idx))
    val_eval = build_internal_eval_frame(val_df, mappings, args.eval_users, args.seed)

    LOGGER.info(
        "Filtered split: train=%s val=%s test=%s mapped_users=%s mapped_items=%s eval_val=%s",
        len(train_df),
        len(val_df),
        len(test_df),
        len(mappings.user_id_to_idx),
        len(mappings.item_id_to_idx),
        len(val_eval),
    )

    all_epoch_rows: List[Dict[str, Any]] = []
    all_config_rows: List[Dict[str, Any]] = []

    stage1_configs = build_stage1_configs(args)
    for config in stage1_configs:
        _, rows, summary = train_torch_bpr(
            config,
            train_pairs,
            len(mappings.user_id_to_idx),
            len(mappings.item_id_to_idx),
            user_pos,
            pop_probs,
            val_eval,
            args.recall_k,
            device,
            args.seed,
            early_stopping_patience=0,
        )
        all_epoch_rows.extend(rows)
        all_config_rows.append(summary)

    stage1_summary_df = pd.DataFrame(all_config_rows)
    stage2_configs = build_stage2_configs(stage1_summary_df, args)
    best_stage2_model: Optional[TorchBPR] = None
    best_stage2_summary: Dict[str, Any] = {}
    best_stage2_score = -1.0
    for config in stage2_configs:
        model, rows, summary = train_torch_bpr(
            config,
            train_pairs,
            len(mappings.user_id_to_idx),
            len(mappings.item_id_to_idx),
            user_pos,
            pop_probs,
            val_eval,
            args.recall_k,
            device,
            args.seed,
            early_stopping_patience=args.early_stopping_patience,
        )
        all_epoch_rows.extend(rows)
        all_config_rows.append(summary)
        score = float(summary.get("best_validation_covered_ndcg_at_200", 0.0))
        if score > best_stage2_score:
            best_stage2_model = model
            best_stage2_summary = summary
            best_stage2_score = score

    epoch_df = pd.DataFrame(all_epoch_rows)
    config_df = pd.DataFrame(all_config_rows)
    epoch_df.to_csv(out_dir / "epoch_metrics.csv", index=False)
    config_df.to_csv(out_dir / "config_summary.csv", index=False)
    plot_epoch_curves(epoch_df, out_dir)
    plot_config_diagnostics(config_df, out_dir)

    if best_stage2_model is None:
        raise RuntimeError("No BPR model was trained.")

    final_train_df = pd.concat([train_df, val_df], ignore_index=True)
    final_mappings = build_id_mappings(final_train_df)
    final_train_matrix = build_user_item_matrix(final_train_df, final_mappings)
    final_train_pairs = train_pairs_array(final_train_df, final_mappings)
    final_user_pos = train_history_sets(final_train_df, final_mappings)
    final_pop_probs = popularity_distribution(final_train_pairs, len(final_mappings.item_id_to_idx))
    final_test_eval = build_internal_eval_frame(test_df, final_mappings, args.eval_users, args.seed)
    best_epochs = max(1, int(best_stage2_summary.get("best_epoch", args.stage2_epochs)))
    best_final_config = TorchBPRConfig(
        config_id="final_bpr",
        optimizer=str(best_stage2_summary["optimizer"]),
        embedding_size=int(best_stage2_summary["embedding_size"]),
        learning_rate=float(best_stage2_summary["learning_rate"]),
        regularization=float(best_stage2_summary["regularization"]),
        batch_size=int(best_stage2_summary["batch_size"]),
        negative_sampler=str(best_stage2_summary["negative_sampler"]),
        epochs=best_epochs,
        stage="final",
    )
    final_model, final_rows, final_summary = train_torch_bpr(
        best_final_config,
        final_train_pairs,
        len(final_mappings.user_id_to_idx),
        len(final_mappings.item_id_to_idx),
        final_user_pos,
        final_pop_probs,
        validation_frame=None,
        recall_k=args.recall_k,
        device=device,
        seed=args.seed,
        early_stopping_patience=0,
    )
    if final_rows:
        final_epoch_df = pd.DataFrame(final_rows)
        final_epoch_df.to_csv(out_dir / "final_bpr_epoch_metrics.csv", index=False)

    final_rows_summary = [
        metric_summary_row("bpr", evaluate_torch_bpr(final_model, final_test_eval, final_user_pos, args.recall_k, device)),
        metric_summary_row(
            "popularity",
            evaluate_popularity(final_train_pairs, final_test_eval, final_user_pos, len(final_mappings.item_id_to_idx), args.recall_k),
        ),
    ]

    comparison_diagnostics: Dict[str, Any] = {}
    if not args.skip_comparisons:
        LOGGER.info("Training comparison Item-KNN on train+validation.")
        item_knn, item_knn_diag = train_item_knn_model(
            final_train_matrix,
            neighbors=args.item_knn_neighbors,
            weighting=args.item_knn_weighting,
            show_progress=False,
        )
        final_rows_summary.append(metric_summary_row("item_knn", evaluate_implicit_like(item_knn, final_test_eval, final_train_matrix, args.recall_k)))
        comparison_diagnostics = {"item_knn": item_knn_diag}

    final_df = pd.DataFrame(final_rows_summary)
    final_df.to_csv(out_dir / "final_test_summary.csv", index=False)
    plot_final_comparison(final_df, out_dir)

    diagnostics = {
        "config": vars(args),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "processed_dir": args.processed_dir,
            "output_prefix": args.output_prefix,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "final_train_rows": int(len(final_train_df)),
            "mapped_users": int(len(mappings.user_id_to_idx)),
            "mapped_items": int(len(mappings.item_id_to_idx)),
        },
        "best_stage2_config": best_stage2_summary,
        "final_bpr_config": final_summary,
        "comparison_diagnostics": comparison_diagnostics,
    }
    save_json(out_dir / "diagnostics.json", diagnostics)
    write_analysis(out_dir, filter_df, config_df, final_df, best_stage2_summary)
    LOGGER.info("Saved BPR diagnostics to %s", out_dir)


if __name__ == "__main__":
    main()
