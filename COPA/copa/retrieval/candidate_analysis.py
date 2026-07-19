"""Candidate-quality diagnostics and target-coverage interventions."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from copa.constraints import ConstraintRegistry
from copa.core import CandidateRecord, ConstraintSpec


def _stable_key(user_id: str, seed: int) -> str:
    return hashlib.sha256(f"candidate-coverage:{seed}:{user_id}".encode()).hexdigest()


def controlled_hit_users(user_ids: Iterable[str], level: float, seed: int) -> set[str]:
    """Choose an exact, deterministic cohort fraction for target inclusion."""

    if not 0.0 <= float(level) <= 1.0:
        raise ValueError("level must lie in [0, 1]")
    ordered = sorted({str(value) for value in user_ids}, key=lambda value: (_stable_key(value, seed), value))
    hit_count = int(round(float(level) * len(ordered)))
    return set(ordered[:hit_count])


def _ranked(records: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    return sorted(
        records,
        key=lambda item: (
            -float(item.base_score),
            int(item.metadata.get("retrieval_rank", 2**63 - 1)),
            str(item.item_id),
        ),
    )


def oracle_candidate_pool(
    candidates: Sequence[CandidateRecord],
    target: CandidateRecord,
    candidate_k: int,
    *,
    seen_items: Iterable[str] = (),
) -> list[CandidateRecord]:
    """Inject a normally scored target while retaining real hard negatives."""

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    seen = {str(value) for value in seen_items}
    if str(target.item_id) in seen:
        raise ValueError("Oracle target cannot be a seen item")
    if not math.isfinite(float(target.base_score)):
        raise ValueError("Oracle target score must be finite")
    unique: Dict[str, CandidateRecord] = {}
    for candidate in _ranked(candidates):
        if candidate.item_id in seen:
            raise ValueError(f"Candidate pool contains seen item {candidate.item_id}")
        unique.setdefault(str(candidate.item_id), candidate)
    unique[str(target.item_id)] = target
    ranked = _ranked(list(unique.values()))
    if len(ranked) > candidate_k:
        selected = ranked[:candidate_k]
        if str(target.item_id) not in {item.item_id for item in selected}:
            selected[-1] = target
            selected = _ranked(selected)
        ranked = selected
    if len(ranked) != min(candidate_k, len(unique)):
        raise RuntimeError("Oracle intervention changed candidate cardinality unexpectedly")
    return ranked


def intervene_candidate_pool(
    ranked_candidates: Sequence[CandidateRecord],
    target: CandidateRecord,
    candidate_k: int,
    *,
    include_target: bool,
    seen_items: Iterable[str] = (),
) -> list[CandidateRecord]:
    """Create one fixed-size hit/miss condition from a larger ranked reservoir."""

    if include_target:
        return oracle_candidate_pool(
            ranked_candidates, target, candidate_k, seen_items=seen_items
        )
    target_id = str(target.item_id)
    filtered = [item for item in _ranked(ranked_candidates) if str(item.item_id) != target_id]
    selected = filtered[:candidate_k]
    if len(selected) < candidate_k:
        raise ValueError("Candidate reservoir cannot fill a target-miss intervention")
    return selected


def candidate_quality_metrics(
    candidates: Sequence[CandidateRecord],
    relevant_items: Iterable[str],
    constraints: Sequence[ConstraintSpec] = (),
    *,
    requested_k: int | None = None,
    target_model_covered: bool = True,
) -> Dict[str, Any]:
    relevant = {str(value) for value in relevant_items}
    ranked = _ranked(candidates)
    candidate_ids = [str(item.item_id) for item in ranked]
    hits = relevant & set(candidate_ids)
    target_rank = min((candidate_ids.index(item_id) + 1 for item_id in hits), default=None)
    registry = ConstraintRegistry()
    feasible_ids = {
        item.item_id
        for item in ranked
        if all(registry.evaluate(spec, item.to_row()).satisfied for spec in constraints)
    }
    feasible_hits = relevant & feasible_ids
    denominator = max(1, len(relevant))
    return {
        "candidate_size": len(ranked),
        "requested_candidate_size": int(requested_k or len(ranked)),
        "candidate_fill_rate": len(ranked) / max(1, int(requested_k or len(ranked))),
        "candidate_recall": len(hits) / denominator,
        "candidate_hit": float(bool(hits)),
        "candidate_ndcg": 0.0 if target_rank is None else 1.0 / math.log2(target_rank + 1),
        "candidate_mrr": 0.0 if target_rank is None else 1.0 / target_rank,
        "target_rank_before_copa": target_rank,
        "target_model_covered": float(bool(target_model_covered)),
        "conditional_candidate_recall": (
            len(hits) / denominator if target_model_covered else float("nan")
        ),
        "feasible_candidate_count": len(feasible_ids),
        "target_in_feasible_domain": len(feasible_hits) / denominator,
        "retrieval_loss": float(not bool(hits)),
        "constraint_filter_loss": float(bool(hits) and not bool(feasible_hits)),
    }
