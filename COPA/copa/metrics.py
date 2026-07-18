"""Offline recommendation and multi-objective metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.stats import qmc

from copa.core import SlateSolution


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(map(str, relevant))
    if not relevant_set:
        return 0.0
    return len(set(map(str, recommended[:k])) & relevant_set) / len(relevant_set)


def ndcg_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(map(str, relevant))
    if not relevant_set:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item_id in enumerate(recommended[:k]) if str(item_id) in relevant_set)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant_set))))
    return float(dcg / ideal) if ideal else 0.0


def normalize_front(front: Sequence[SlateSolution]) -> np.ndarray:
    if not front:
        return np.empty((0, 0), dtype=float)
    values = np.asarray([solution.maximization_values for solution in front], dtype=float)
    # Built-in objectives are bounded to [0, 1]. Minimized objectives are normalized
    # against the observed front because their natural scale is plugin-defined.
    if np.all((values >= 0.0) & (values <= 1.0)):
        return values
    lower, upper = values.min(axis=0), values.max(axis=0)
    ranges = upper - lower
    return np.divide(values - lower, ranges, out=np.ones_like(values), where=ranges > 0)


def hypervolume_sobol(front: Sequence[SlateSolution], sample_power: int = 14, seed: int = 42) -> float:
    normalized = normalize_front(front)
    if normalized.size == 0:
        return 0.0
    samples = qmc.Sobol(d=normalized.shape[1], scramble=True, seed=seed).random_base2(sample_power)
    dominated = np.any(np.all(normalized[:, None, :] >= samples[None, :, :], axis=2), axis=0)
    return float(np.mean(dominated))


def hypervolume_shared(
    front: Sequence[SlateSolution],
    *,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    sample_power: int = 14,
    seed: int = 42,
) -> float:
    """Estimate HV on explicit shared maximization bounds.

    Unlike ``hypervolume_sobol``, this function never derives bounds from one
    method's observed front, so values are comparable across methods.
    """
    if not front:
        return 0.0
    values = np.asarray([solution.maximization_values for solution in front], dtype=float)
    dimensions = values.shape[1]
    low = np.asarray(lower if lower is not None else np.zeros(dimensions), dtype=float)
    high = np.asarray(upper if upper is not None else np.ones(dimensions), dtype=float)
    if low.shape != (dimensions,) or high.shape != (dimensions,):
        raise ValueError("shared hypervolume bounds must match objective dimensions")
    if np.any(high <= low):
        raise ValueError("shared hypervolume upper bounds must exceed lower bounds")
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    samples = qmc.Sobol(d=dimensions, scramble=True, seed=seed).random_base2(sample_power)
    dominated = np.any(
        np.all(normalized[:, None, :] >= samples[None, :, :], axis=2), axis=0
    )
    return float(np.mean(dominated))


def spacing_shared(
    front: Sequence[SlateSolution],
    *,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
) -> float:
    """Nearest-neighbour spacing on explicit shared bounds."""
    if len(front) < 2:
        return 0.0
    values = np.asarray([solution.maximization_values for solution in front], dtype=float)
    dimensions = values.shape[1]
    low = np.asarray(lower if lower is not None else np.zeros(dimensions), dtype=float)
    high = np.asarray(upper if upper is not None else np.ones(dimensions), dtype=float)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    minimum_distances = []
    for index, point in enumerate(normalized):
        others = np.delete(normalized, index, axis=0)
        minimum_distances.append(float(np.min(np.linalg.norm(others - point, axis=1))))
    return float(np.std(minimum_distances))


def spacing(front: Sequence[SlateSolution]) -> float:
    values = normalize_front(front)
    if len(values) < 2:
        return 0.0
    minimum_distances = []
    for index, point in enumerate(values):
        others = np.delete(values, index, axis=0)
        minimum_distances.append(float(np.min(np.linalg.norm(others - point, axis=1))))
    return float(np.std(minimum_distances))


def front_metrics(front: Sequence[SlateSolution], sample_power: int = 14, seed: int = 42) -> Dict[str, float]:
    return {
        "pareto_front_size": float(len(front)),
        "hypervolume": hypervolume_sobol(front, sample_power=sample_power, seed=seed),
        "spacing": spacing(front),
        "hypervolume_sample_count": float(2**sample_power),
        "hypervolume_seed": float(seed),
    }
