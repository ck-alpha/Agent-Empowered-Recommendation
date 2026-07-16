"""Deterministic COPA Phase-1 orchestration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from copa.constraints import ConstraintRegistry
from copa.core import (
    CandidateStateBus,
    CandidateTracker,
    RecommendationRequest,
    RecommendationResult,
)
from copa.core.verifier import DeterministicVerifier
from copa.objectives import ObjectiveRegistry
from copa.optimization import ParetoOptimizer


class COPAPipeline:
    def __init__(
        self,
        *,
        constraints: Optional[ConstraintRegistry] = None,
        objectives: Optional[ObjectiveRegistry] = None,
        trace_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.constraints = constraints or ConstraintRegistry()
        self.objectives = objectives or ObjectiveRegistry()
        self.optimizer = ParetoOptimizer(self.objectives)
        self.verifier = DeterministicVerifier(self.constraints)
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.run_id = run_id or uuid4().hex

    def run(self, request: RecommendationRequest) -> RecommendationResult:
        trace_path = None
        if self.trace_dir:
            safe_user = "".join(character if character.isalnum() or character in "-_" else "_" for character in request.user_id)
            trace_path = self.trace_dir / f"{self.run_id}_{safe_user}.jsonl"
        tracker = CandidateTracker(self.run_id, request.user_id, trace_path)
        bus = CandidateStateBus(tracker)
        bus.initialize(request.candidates, source=str(request.context.get("source", "request")))
        if request.constraints:
            self.constraints.apply(bus, request.constraints)
        self.objectives.annotate_candidates(bus, request.objectives, request.context)
        selected, pareto_front, diagnostics = self.optimizer.optimize(
            bus, request.objectives, request.optimization, request.context
        )
        verification = self.verifier.verify(
            selected.item_ids,
            bus,
            request.constraints,
            request.optimization.top_k,
            selected.objective_values,
        )
        diagnostics["candidate_count"] = len(request.candidates)
        diagnostics["feasible_candidate_count"] = len(bus.query(feasible_only=True))
        diagnostics["trace_event_count"] = len(tracker.events)
        return RecommendationResult(
            user_id=request.user_id,
            item_ids=selected.item_ids,
            objective_values=selected.objective_values,
            pareto_front=pareto_front,
            verification=verification,
            bus_version=bus.version,
            trace_path=trace_path,
            diagnostics=diagnostics,
        )
