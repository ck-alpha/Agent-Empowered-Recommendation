"""Candidate-quality diagnostics and target-coverage interventions."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from copa.constraints import ConstraintRegistry
from copa.core import CandidateRecord, ConstraintSpec


def _stable_key(user_id: str, seed: int) -> str:
    return hashlib.sha256(f"candidate-coverage:{seed}:{user_id}".encode()).hexdigest()


def _stable_pair_key(pair: tuple[str, str], seed: int) -> str:
    return hashlib.sha256(
        f"candidate-positive-coverage:{seed}:{pair[0]}:{pair[1]}".encode()
    ).hexdigest()


def controlled_hit_users(user_ids: Iterable[str], level: float, seed: int) -> set[str]:
    """Choose an exact, deterministic cohort fraction for target inclusion."""

    if not 0.0 <= float(level) <= 1.0:
        raise ValueError("level must lie in [0, 1]")
    ordered = sorted({str(value) for value in user_ids}, key=lambda value: (_stable_key(value, seed), value))
    hit_count = int(round(float(level) * len(ordered)))
    return set(ordered[:hit_count])


def controlled_hit_pairs(
    pairs: Iterable[tuple[str, str]], level: float, seed: int
) -> set[tuple[str, str]]:
    """Choose an exact deterministic fraction of user-positive pairs."""

    if not 0.0 <= float(level) <= 1.0:
        raise ValueError("level must lie in [0, 1]")
    normalized = {(str(user_id), str(item_id)) for user_id, item_id in pairs}
    ordered = sorted(
        normalized, key=lambda pair: (_stable_pair_key(pair, seed), pair)
    )
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


def intervene_positive_pool(
    ranked_candidates: Sequence[CandidateRecord],
    targets: Sequence[CandidateRecord],
    candidate_k: int,
    *,
    included_target_ids: Iterable[str],
    seen_items: Iterable[str] = (),
) -> list[CandidateRecord]:
    """Create a fixed-K pool with exact inclusion for multiple scored positives."""

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    seen = {str(value) for value in seen_items}
    target_by_id = {str(target.item_id): target for target in targets}
    included = {str(value) for value in included_target_ids}
    if not included.issubset(target_by_id):
        raise ValueError("Included target IDs must have audited model scores")
    if included & seen:
        raise ValueError("Controlled targets cannot be seen items")
    base = [
        candidate
        for candidate in _ranked(ranked_candidates)
        if str(candidate.item_id) not in target_by_id
    ]
    if len(base) + len(included) < candidate_k:
        raise ValueError("Candidate reservoir cannot fill a controlled intervention")
    selected = base[:candidate_k]
    for target_id in sorted(included):
        target = target_by_id[target_id]
        if not math.isfinite(float(target.base_score)):
            raise ValueError("Controlled target score must be finite")
        if len(selected) >= candidate_k:
            selected.pop()
        selected.append(target)
        selected = _ranked(selected)
    if len(selected) != candidate_k:
        raise RuntimeError("Controlled intervention changed candidate cardinality")
    selected_ids = {str(candidate.item_id) for candidate in selected}
    if (selected_ids & set(target_by_id)) != included:
        raise RuntimeError("Controlled intervention failed exact pair inclusion")
    return selected


def candidate_quality_metrics(
    candidates: Sequence[CandidateRecord],
    relevant_items: Iterable[str],
    constraints: Sequence[ConstraintSpec] = (),
    *,
    requested_k: int | None = None,
    target_model_covered: bool | Iterable[bool] = True,
) -> Dict[str, Any]:
    relevant = {str(value) for value in relevant_items}
    ranked = _ranked(candidates)
    candidate_ids = [str(item.item_id) for item in ranked]
    hits = relevant & set(candidate_ids)
    hit_ranks = sorted(candidate_ids.index(item_id) + 1 for item_id in hits)
    target_rank = hit_ranks[0] if hit_ranks else None
    registry = ConstraintRegistry()
    feasible_ids = {
        item.item_id
        for item in ranked
        if all(registry.evaluate(spec, item.to_row()).satisfied for spec in constraints)
    }
    feasible_hits = relevant & feasible_ids
    denominator = max(1, len(relevant))
    if isinstance(target_model_covered, bool):
        covered_count = len(relevant) if target_model_covered else 0
    else:
        coverage_values = list(target_model_covered)
        if len(coverage_values) != len(relevant):
            raise ValueError("Target coverage flags must align with relevant_items")
        covered_count = sum(bool(value) for value in coverage_values)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hit_ranks)
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), len(candidate_ids)) + 1)
    )
    return {
        "candidate_size": len(ranked),
        "requested_candidate_size": int(requested_k or len(ranked)),
        "candidate_fill_rate": len(ranked) / max(1, int(requested_k or len(ranked))),
        "candidate_recall": len(hits) / denominator,
        "candidate_hit": float(bool(hits)),
        "candidate_ndcg": dcg / ideal if ideal else 0.0,
        "candidate_mrr": 0.0 if target_rank is None else 1.0 / target_rank,
        "target_rank_before_copa": target_rank,
        "target_model_covered": covered_count / denominator,
        "conditional_candidate_recall": (
            len(hits) / covered_count if covered_count else float("nan")
        ),
        "feasible_candidate_count": len(feasible_ids),
        "target_in_feasible_domain": len(feasible_hits) / denominator,
        "retrieval_loss": (covered_count - len(hits)) / denominator,
        "constraint_filter_loss": (len(hits) - len(feasible_hits)) / denominator,
    }
