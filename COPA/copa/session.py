"""Step-wise deterministic COPA execution shared by Phase 1 and Agent workflows."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

from copa.constraints import ConstraintRegistry
from copa.core import (
    CandidateStateBus,
    CandidateTracker,
    RecommendationRequest,
    RecommendationResult,
    SlateSolution,
    VerificationReport,
)
from copa.core.verifier import DeterministicVerifier
from copa.objectives import ObjectiveRegistry
from copa.optimization import ParetoOptimizer


class COPAExecutionSession:
    """One deterministic request execution with explicit, ordered tool methods."""

    def __init__(
        self,
        request: RecommendationRequest,
        *,
        constraints: Optional[ConstraintRegistry] = None,
        objectives: Optional[ObjectiveRegistry] = None,
        trace_dir: Optional[Path] = None,
        run_id: str = "copa_session",
    ) -> None:
        self.request = request
        self.constraints = constraints or ConstraintRegistry()
        self.objectives = objectives or ObjectiveRegistry()
        self.optimizer = ParetoOptimizer(self.objectives)
        self.verifier = DeterministicVerifier(self.constraints)
        self.run_id = run_id
        safe_user = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in request.user_id
        )
        self.trace_path = Path(trace_dir) / f"{run_id}_{safe_user}.jsonl" if trace_dir else None
        self.tracker = CandidateTracker(run_id, request.user_id, self.trace_path)
        self.bus = CandidateStateBus(self.tracker)
        self.selected: Optional[SlateSolution] = None
        self.pareto_front: list[SlateSolution] = []
        self.verification: Optional[VerificationReport] = None
        self.diagnostics: dict[str, object] = {}
        self._initialized = False
        self._constraints_applied = False
        self._objectives_computed = False

    def initialize_candidates(self) -> int:
        if self._initialized:
            return self.bus.version
        version = self.bus.initialize(
            self.request.candidates,
            source=str(self.request.context.get("source", "request")),
        )
        self._initialized = True
        return version

    def apply_constraints(self) -> int:
        self.initialize_candidates()
        if self._constraints_applied:
            return self.bus.version
        if self.request.constraints:
            version = self.constraints.apply(self.bus, self.request.constraints)
        else:
            version = self.bus.version
        self._constraints_applied = True
        return version

    def compute_objectives(self) -> int:
        self.apply_constraints()
        if self._objectives_computed:
            return self.bus.version
        version = self.objectives.annotate_candidates(
            self.bus,
            self.request.objectives,
            self.request.context,
        )
        self._objectives_computed = True
        return version

    def select_feasible_topk(self) -> SlateSolution:
        started = perf_counter()
        self.compute_objectives()
        frame = self.bus.query(feasible_only=True)
        ordered = frame.sort_values(
            ["base_score", "item_id"], ascending=[False, True]
        )["item_id"].astype(str).tolist()
        item_ids = ordered[: self.request.optimization.top_k]
        values = self.objectives.evaluate(
            item_ids,
            frame,
            self.request.objectives,
            self.request.context,
        )
        self.selected = SlateSolution(
            item_ids,
            values,
            self.objectives.to_maximization(values, self.request.objectives),
        )
        self.pareto_front = [self.selected]
        self.diagnostics = {
            "strategy": "feasible_topk",
            "evaluations": 1,
            "pareto_size": 1,
            "slate_size": len(item_ids),
            "available_feasible_candidates": len(frame),
            "runtime_seconds": perf_counter() - started,
        }
        self.tracker.record(
            module="DeterministicSelectionModule",
            operation="select_feasible_topk",
            status="success",
            before_version=self.bus.version,
            after_version=self.bus.version,
            before_candidates=len(frame),
            after_candidates=len(item_ids),
            duration_ms=(perf_counter() - started) * 1000,
            seed=self.request.optimization.seed,
            input_summary={"requested_k": self.request.optimization.top_k},
        )
        return self.selected

    def optimize_pareto(self) -> SlateSolution:
        self.compute_objectives()
        selected, front, diagnostics = self.optimizer.optimize(
            self.bus,
            self.request.objectives,
            self.request.optimization,
            self.request.context,
        )
        self.selected = selected
        self.pareto_front = front
        self.diagnostics = {"strategy": "pareto", **diagnostics}
        return selected

    def verify(
        self,
        *,
        item_ids: Optional[Sequence[str]] = None,
        objective_values: Optional[dict[str, float]] = None,
    ) -> VerificationReport:
        if self.selected is None and item_ids is None:
            raise RuntimeError("Selection must run before verification")
        selected_ids = list(item_ids) if item_ids is not None else list(self.selected.item_ids)
        values = objective_values if objective_values is not None else (
            dict(self.selected.objective_values) if self.selected else {}
        )
        self.verification = self.verifier.verify(
            selected_ids,
            self.bus,
            self.request.constraints,
            self.request.optimization.top_k,
            values,
        )
        return self.verification

    def result(self) -> RecommendationResult:
        if self.selected is None or self.verification is None:
            raise RuntimeError("Selection and verification must complete before building a result")
        self.diagnostics["candidate_count"] = len(self.request.candidates)
        self.diagnostics["feasible_candidate_count"] = len(self.bus.query(feasible_only=True))
        self.diagnostics["trace_event_count"] = len(self.tracker.events)
        return RecommendationResult(
            user_id=self.request.user_id,
            item_ids=list(self.selected.item_ids),
            objective_values=dict(self.selected.objective_values),
            pareto_front=list(self.pareto_front),
            verification=self.verification,
            bus_version=self.bus.version,
            trace_path=self.trace_path,
            diagnostics=dict(self.diagnostics),
        )

    def execute(self, strategy: str = "pareto") -> RecommendationResult:
        if strategy == "pareto":
            self.optimize_pareto()
        elif strategy == "feasible_topk":
            self.select_feasible_topk()
        else:
            raise ValueError(f"Unknown execution strategy: {strategy}")
        self.verify()
        return self.result()
