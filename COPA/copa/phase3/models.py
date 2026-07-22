"""Strict public contracts for COPA Phase 3 agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copa.core import (
    CandidateRecord,
    ConstraintSpec,
    OptimizationConfig,
    RecommendationResult,
    SlateConstraintSpec,
)


AgentStatus = Literal["success", "clarification_required", "failed"]
Strategy = Literal["feasible_topk", "pareto"]
ToolName = Literal[
    "apply_constraints",
    "compute_objectives",
    "select_feasible_topk",
    "optimize_pareto",
    "verify",
]
RepairAction = Literal[
    "recompute_objectives",
    "reexecute_selection",
    "replan",
    "request_clarification",
    "abort",
]


class ToolStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool: ToolName


class AgentExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    plan_version: Literal["1.0"]
    strategy: Strategy
    steps: List[ToolStep] = Field(min_length=1, max_length=5)
    assumptions: List[str] = Field(default_factory=list)
    unresolved_requirements: List[str] = Field(default_factory=list)


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_version: Literal["1.0"]
    action: RepairAction
    reason_code: str = Field(min_length=1)
    clarification_question: Optional[str] = None

    @model_validator(mode="after")
    def require_question(self) -> "RepairDecision":
        if self.action == "request_clarification" and not self.clarification_question:
            raise ValueError("request_clarification requires clarification_question")
        if self.action != "request_clarification" and self.clarification_question is not None:
            raise ValueError("clarification_question is only allowed for request_clarification")
        return self


@dataclass(frozen=True)
class AgentRecommendationRequest:
    user_id: str
    text: str
    candidates: Sequence[CandidateRecord]
    domain: str = "synthetic"
    base_constraints: Sequence[ConstraintSpec] = field(default_factory=list)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    context: Mapping[str, Any] = field(default_factory=dict)
    base_slate_constraints: Sequence[SlateConstraintSpec] = field(default_factory=list)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "text": self.text,
            "candidates": [
                {
                    "item_id": candidate.item_id,
                    "base_score": candidate.base_score,
                    "metadata": dict(candidate.metadata),
                    "source": candidate.source,
                }
                for candidate in self.candidates
            ],
            "domain": self.domain,
            "base_constraints": [asdict(spec) for spec in self.base_constraints],
            "base_slate_constraints": [
                asdict(spec) for spec in self.base_slate_constraints
            ],
            "optimization": asdict(self.optimization),
            "context": dict(self.context),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "AgentRecommendationRequest":
        return cls(
            user_id=str(payload["user_id"]),
            text=str(payload["text"]),
            candidates=[CandidateRecord(**dict(item)) for item in payload["candidates"]],
            domain=str(payload.get("domain", "synthetic")),
            base_constraints=[ConstraintSpec.from_dict(item) for item in payload.get("base_constraints", [])],
            optimization=OptimizationConfig(**dict(payload.get("optimization", {}))),
            context=dict(payload.get("context", {})),
            base_slate_constraints=[
                SlateConstraintSpec.from_dict(item)
                for item in payload.get("base_slate_constraints", [])
            ],
        )


@dataclass
class AgentRecommendationResult:
    status: AgentStatus
    thread_id: str
    compile_result: Optional[Dict[str, Any]] = None
    execution_plan: Optional[AgentExecutionPlan] = None
    recommendation: Optional[RecommendationResult] = None
    clarification: Optional[Dict[str, Any]] = None
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    agent_trace_path: Optional[Path] = None
    candidate_trace_paths: List[Path] = field(default_factory=list)
    checkpoint_path: Optional[Path] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "thread_id": self.thread_id,
            "compile_result": self.compile_result,
            "execution_plan": self.execution_plan.model_dump(mode="json") if self.execution_plan else None,
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "clarification": self.clarification,
            "repair_history": self.repair_history,
            "agent_trace_path": str(self.agent_trace_path) if self.agent_trace_path else None,
            "candidate_trace_paths": [str(path) for path in self.candidate_trace_paths],
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "diagnostics": self.diagnostics,
            "errors": self.errors,
        }


@dataclass
class StructuredAgentCallResult:
    status: Literal["success", "failed"]
    payload: Optional[BaseModel] = None
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    latency_seconds: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)
