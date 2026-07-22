"""Deterministic COPA Phase-1 orchestration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry
from copa.core import RecommendationRequest, RecommendationResult
from copa.objectives import ObjectiveRegistry
from copa.session import COPAExecutionSession


class COPAPipeline:
    def __init__(
        self,
        *,
        constraints: Optional[ConstraintRegistry] = None,
        slate_constraints: Optional[SlateConstraintRegistry] = None,
        objectives: Optional[ObjectiveRegistry] = None,
        trace_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.constraints = constraints or ConstraintRegistry()
        self.slate_constraints = slate_constraints or SlateConstraintRegistry()
        self.objectives = objectives or ObjectiveRegistry()
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.run_id = run_id or uuid4().hex

    def run(self, request: RecommendationRequest) -> RecommendationResult:
        return COPAExecutionSession(
            request,
            constraints=self.constraints,
            slate_constraints=self.slate_constraints,
            objectives=self.objectives,
            trace_dir=self.trace_dir,
            run_id=self.run_id,
        ).execute("pareto")
