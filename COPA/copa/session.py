"""Step-wise deterministic COPA execution shared by Phase 1 and Agent workflows."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry, SlateSolveResult
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
        slate_constraints: Optional[SlateConstraintRegistry] = None,
        objectives: Optional[ObjectiveRegistry] = None,
        trace_dir: Optional[Path] = None,
        run_id: str = "copa_session",
    ) -> None:
        self.request = request
        self.constraints = constraints or ConstraintRegistry()
        self.slate_constraints = slate_constraints or SlateConstraintRegistry()
        self.objectives = objectives or ObjectiveRegistry()
        self.optimizer = ParetoOptimizer(self.objectives, self.slate_constraints)
        self.verifier = DeterministicVerifier(self.constraints, self.slate_constraints)
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
        self.status = "success"
        self._preflight_result: Optional[SlateSolveResult] = None
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
        self.slate_constraints.validate(
            self.request.slate_constraints, self.bus.query(feasible_only=True)
        )
        self._constraints_applied = True
        return version

    def preflight(self) -> bool:
        """Prove full-K slate feasibility before selection or optimization."""

        self.apply_constraints()
        frame = self.bus.query(feasible_only=True)
        requested_k = self.request.optimization.top_k
        if requested_k <= 0:
            raise ValueError("top_k must be positive")
        if len(frame) < requested_k:
            self.status = "proven_infeasible"
            self._preflight_result = SlateSolveResult(
                "infeasible",
                message=f"Only {len(frame)} item-feasible candidates are available for K={requested_k}",
            )
        elif self.request.slate_constraints:
            scores = {
                str(row["item_id"]): float(row["base_score"])
                for row in frame.to_dict("records")
            }
            self._preflight_result = self.slate_constraints.solve(
                frame,
                self.request.slate_constraints,
                requested_k,
                objective_scores=scores,
                time_limit_seconds=self.request.optimization.slate_solver_time_limit_seconds,
            )
            if self._preflight_result.status == "infeasible":
                self.status = "proven_infeasible"
            elif self._preflight_result.status != "optimal":
                self.status = "solver_unknown"
        else:
            self._preflight_result = SlateSolveResult(
                "optimal",
                frame.sort_values(["base_score", "item_id"], ascending=[False, True])
                ["item_id"]
                .astype(str)
                .head(requested_k)
                .tolist(),
            )
        result = self._preflight_result
        self.diagnostics.update(
            {
                "preflight_status": result.status,
                "preflight_runtime_seconds": result.runtime_seconds,
                "preflight_message": result.message,
                "preflight_mip_gap": result.mip_gap,
                "preflight_mip_node_count": result.mip_node_count,
            }
        )
        return self.status == "success"

    def _empty_selection(self) -> SlateSolution:
        frame = self.bus.query(feasible_only=True)
        values = self.objectives.evaluate(
            [], frame, self.request.objectives, self.request.context
        )
        self.selected = SlateSolution(
            [], values, self.objectives.to_maximization(values, self.request.objectives),
            constraint_feasible=False,
            constraint_violation=float("inf"),
        )
        self.pareto_front = []
        return self.selected

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
        if not self.preflight():
            return self._empty_selection()
        frame = self.bus.query(feasible_only=True)
        item_ids = list(self._preflight_result.item_ids if self._preflight_result else [])
        rows = frame.set_index(frame["item_id"].astype(str), drop=False)
        item_ids.sort(key=lambda item_id: (-float(rows.loc[item_id, "base_score"]), item_id))
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
            **self.diagnostics,
            "strategy": "feasible_topk",
            "evaluations": 1,
            "pareto_size": 1,
            "slate_size": len(item_ids),
            "available_feasible_candidates": len(frame),
            "runtime_seconds": perf_counter() - started,
            "slate_constraint_count": len(self.request.slate_constraints),
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
        if not self.preflight():
            return self._empty_selection()
        try:
            selected, front, diagnostics = self.optimizer.optimize(
                self.bus,
                self.request.objectives,
                self.request.optimization,
                self.request.context,
                slate_specs=self.request.slate_constraints,
                feasible_seed=(
                    self._preflight_result.item_ids if self._preflight_result else None
                ),
            )
        except (RuntimeError, ValueError) as exc:
            self.status = "optimizer_failed"
            self.diagnostics["optimizer_error"] = f"{type(exc).__name__}: {exc}"
            return self._empty_selection()
        self.selected = selected
        self.pareto_front = front
        self.diagnostics = {**self.diagnostics, "strategy": "pareto", **diagnostics}
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
            self.request.slate_constraints,
            self.status,
        )
        return self.verification

    def result(self) -> RecommendationResult:
        if self.selected is None or self.verification is None:
            raise RuntimeError("Selection and verification must complete before building a result")
        self.diagnostics["candidate_count"] = len(self.request.candidates)
        self.diagnostics["feasible_candidate_count"] = len(self.bus.query(feasible_only=True))
        self.diagnostics["trace_event_count"] = len(self.tracker.events)
        final_status = self.status
        if final_status == "success" and not self.verification.feasible:
            final_status = "verification_failed"
        delivered = list(self.selected.item_ids) if final_status == "success" else []
        if final_status != "success":
            self.diagnostics["attempted_item_ids"] = list(self.selected.item_ids)
        return RecommendationResult(
            user_id=self.request.user_id,
            item_ids=delivered,
            objective_values=dict(self.selected.objective_values),
            pareto_front=list(self.pareto_front),
            verification=self.verification,
            bus_version=self.bus.version,
            trace_path=self.trace_path,
            diagnostics=dict(self.diagnostics),
            status=final_status,
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
