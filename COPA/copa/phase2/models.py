"""Strict public contracts for the COPA Phase-2 constraint compiler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copa.core import (
    CandidateRecord,
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationConfig,
    RecommendationResult,
)


ConstraintOperator = Literal[
    "<", "<=", ">", ">=", "==", "!=", "between", "in", "not_in",
    "contains_any", "contains_all", "not_contains",
]
CompileStatus = Literal["success", "clarification_required", "failed"]
IssueSeverity = Literal["warning", "blocking"]
Provenance = Literal["user", "system", "system_default"]
IRScalar = Union[str, int, float, bool]
IRValue = Union[IRScalar, List[IRScalar]]


def _ensure_finite(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("IR numeric values must be finite")
        return value
    if isinstance(value, list):
        return [_ensure_finite(item) for item in value]
    raise TypeError("IR values must be JSON scalars or arrays")


class HardConstraintIR(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    attribute: str = Field(min_length=1)
    operator: ConstraintOperator
    value: IRValue
    currency: Optional[str] = None

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: Any) -> Any:
        return _ensure_finite(value)


class SoftObjectiveIR(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    objective: str = Field(min_length=1)
    direction: Literal["maximize", "minimize"] = "maximize"


class UnresolvedRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    clarification_question: str = Field(min_length=1)


class ConstraintIR(BaseModel):
    """The only model-generated object accepted by the compiler."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    hard_constraints: List[HardConstraintIR]
    soft_objectives: List[SoftObjectiveIR]
    top_k: Optional[int]
    unresolved_requirements: List[UnresolvedRequirement]


class CompileIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: IssueSeverity
    message: str
    field: Optional[str] = None
    clarification_question: Optional[str] = None


@dataclass(frozen=True)
class CompiledConstraint:
    spec: ConstraintSpec
    provenance: Provenance

    def to_dict(self) -> Dict[str, Any]:
        return {"spec": asdict(self.spec), "provenance": self.provenance}


@dataclass(frozen=True)
class CompiledObjective:
    spec: ObjectiveSpec
    provenance: Provenance

    def to_dict(self) -> Dict[str, Any]:
        return {"spec": asdict(self.spec), "provenance": self.provenance}


@dataclass
class CompiledPlan:
    constraints: List[CompiledConstraint]
    objectives: List[CompiledObjective]
    top_k: int
    assumptions: List[str] = field(default_factory=list)
    issues: List[CompileIssue] = field(default_factory=list)

    @property
    def executable_constraints(self) -> List[ConstraintSpec]:
        return [entry.spec for entry in self.constraints]

    @property
    def executable_objectives(self) -> List[ObjectiveSpec]:
        return [entry.spec for entry in self.objectives]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": [entry.to_dict() for entry in self.constraints],
            "objectives": [entry.to_dict() for entry in self.objectives],
            "top_k": self.top_k,
            "assumptions": self.assumptions,
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }


@dataclass
class CompileResult:
    status: CompileStatus
    ir: Optional[ConstraintIR] = None
    plan: Optional[CompiledPlan] = None
    issues: List[CompileIssue] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    latency_seconds: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
    audit_path: Optional[Path] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success" and self.plan is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "ir": self.ir.model_dump(mode="json") if self.ir else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
            "errors": self.errors,
            "attempts": self.attempts,
            "latency_seconds": self.latency_seconds,
            "usage": self.usage,
            "audit_path": str(self.audit_path) if self.audit_path else None,
        }


@dataclass(frozen=True)
class NaturalLanguageRecommendationRequest:
    user_id: str
    text: str
    candidates: Sequence[CandidateRecord]
    domain: str = "synthetic"
    base_constraints: Sequence[ConstraintSpec] = field(default_factory=list)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class NaturalLanguageRecommendationResult:
    user_id: str
    compile_result: CompileResult
    recommendation: Optional[RecommendationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "compile_result": self.compile_result.to_dict(),
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
        }
