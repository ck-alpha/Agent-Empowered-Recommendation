"""Deterministic synthetic fixture used by examples and exact tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from copa.core import CandidateRecord, ConstraintSpec


@dataclass(frozen=True)
class UserCase:
    user_id: str
    candidates: Sequence[CandidateRecord]
    constraints: Sequence[ConstraintSpec]
    relevant_items: Sequence[str]
    context: Dict[str, object]


def build_synthetic_case(seed: int = 42, candidate_count: int = 30) -> UserCase:
    if candidate_count < 10:
        raise ValueError("Synthetic fixture requires at least 10 candidates")
    rng = np.random.default_rng(seed)
    candidates: List[CandidateRecord] = []
    groups = ["A", "B", "C"]
    for index in range(candidate_count):
        group = groups[index % len(groups)]
        brand = "blocked" if index in {1, 11} else f"brand_{index % 6}"
        metadata = {
            "price": float(10 + (index * 7) % 90),
            "category": group,
            "brand_id": brand,
            "group": group,
            "popularity": float(index / max(candidate_count - 1, 1)),
            "availability": index not in {2, 12},
        }
        # Small seeded jitter prevents broad ties while retaining deterministic ordering.
        base_score = float(np.clip(1.0 - index / candidate_count + rng.uniform(-0.01, 0.01), 0.0, 1.0))
        candidates.append(CandidateRecord(f"item_{index:03d}", base_score, metadata, "synthetic"))
    relevant = ["item_003"]
    constraints = [
        ConstraintSpec("price_limit", "numeric", "price", "<=", 60.0),
        ConstraintSpec("allowed_categories", "categorical", "category", "in", ["A", "B", "C"]),
        ConstraintSpec("blocked_brand", "categorical", "brand_id", "not_in", ["blocked"]),
        ConstraintSpec("available", "boolean", "availability", "==", True),
        ConstraintSpec("seen_items", "exclusion", "item_id", "not_in", ["item_000"]),
    ]
    return UserCase(
        user_id="synthetic_user",
        candidates=candidates,
        constraints=constraints,
        relevant_items=relevant,
        context={"source": "synthetic", "seed": seed},
    )
