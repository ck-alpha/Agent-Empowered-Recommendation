"""Typed public contracts for COPA Phase 1."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence


Direction = Literal["maximize", "minimize"]
ObjectiveScope = Literal["candidate", "slate"]
RecommendationStatus = Literal[
    "success",
    "proven_infeasible",
    "solver_unknown",
    "optimizer_failed",
    "verification_failed",
]


@dataclass(frozen=True)
class CandidateRecord:
    item_id: str
    base_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def to_row(self) -> Dict[str, Any]:
        return {
            "item_id": str(self.item_id),
            "base_score": float(self.base_score),
            "metadata": dict(self.metadata),
            "hard_state": {"feasible": True, "violations": []},
            "soft_objectives": {},
            "active": True,
            "source": self.source,
            "version": 0,
        }


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    type: str
    attribute: str = "item_id"
    operator: str = "=="
    value: Any = None
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstraintSpec":
        return cls(**dict(payload))


@dataclass(frozen=True)
class SlateConstraintSpec:
    """Typed hard constraint evaluated over a complete unordered Top-K slate."""

    id: str
    type: Literal[
        "aggregate_sum", "distinct_count", "per_group_count", "group_count"
    ]
    attribute: str
    operator: str
    value: Any
    target_values: Sequence[Any] = field(default_factory=tuple)
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SlateConstraintSpec":
        values = dict(payload)
        if "target_values" in values:
            values["target_values"] = tuple(values["target_values"] or ())
        return cls(**values)


@dataclass(frozen=True)
class SlateConstraintEvaluation:
    satisfied: bool
    actual: Any
    violation_magnitude: float
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: Direction = "maximize"
    scope: ObjectiveScope = "slate"
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        return cls(**dict(payload))


@dataclass(frozen=True)
class OptimizationConfig:
    top_k: int = 10
    population_size: int = 100
    generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    tournament_size: int = 2
    seed: int = 42
    selection_strategy: Literal["compromise", "weighted"] = "compromise"
    objective_weights: Optional[Sequence[float]] = None
    slate_solver_time_limit_seconds: float = 2.0
    slate_repair_attempts: int = 20
    use_milp_seed: bool = True
    use_slate_feasible_operators: bool = True
    optimizer_time_limit_seconds: float = 120.0
    optimizer_kernel_version: int = 2


@dataclass
class SlateSolution:
    item_ids: List[str]
    objective_values: Dict[str, float] = field(default_factory=dict)
    maximization_values: List[float] = field(default_factory=list)
    rank: int = 0
    crowding_distance: float = 0.0
    constraint_feasible: bool = True
    constraint_violation: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not math.isfinite(float(self.crowding_distance)):
            payload["crowding_distance"] = None
        return payload


@dataclass(frozen=True)
class VerificationReport:
    feasible: bool
    violations: List[Dict[str, Any]]
    requested_k: int
    actual_k: int
    checked_constraints: List[str]
    checked_item_constraints: List[str] = field(default_factory=list)
    checked_slate_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationRequest:
    user_id: str
    candidates: Sequence[CandidateRecord]
    constraints: Sequence[ConstraintSpec]
    objectives: Sequence[ObjectiveSpec]
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    context: Mapping[str, Any] = field(default_factory=dict)
    slate_constraints: Sequence[SlateConstraintSpec] = field(default_factory=tuple)


@dataclass
class RecommendationResult:
    user_id: str
    item_ids: List[str]
    objective_values: Dict[str, float]
    pareto_front: List[SlateSolution]
    verification: VerificationReport
    bus_version: int
    trace_path: Optional[Path]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    status: RecommendationStatus = "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "item_ids": self.item_ids,
            "objective_values": self.objective_values,
            "pareto_front": [solution.to_dict() for solution in self.pareto_front],
            "verification": self.verification.to_dict(),
            "bus_version": self.bus_version,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "diagnostics": self.diagnostics,
            "status": self.status,
        }
