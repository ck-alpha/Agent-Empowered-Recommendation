"""Independent deterministic verification of a final recommendation slate."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, List, Mapping

import numpy as np

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry

from .bus import CandidateStateBus
from .models import ConstraintSpec, SlateConstraintSpec, VerificationReport


class DeterministicVerifier:
    def __init__(
        self,
        constraints: ConstraintRegistry | None = None,
        slate_constraints: SlateConstraintRegistry | None = None,
    ):
        self.constraints = constraints or ConstraintRegistry()
        self.slate_constraints = slate_constraints or SlateConstraintRegistry()

    def verify(
        self,
        item_ids: List[str],
        bus: CandidateStateBus,
        specs: Iterable[ConstraintSpec],
        requested_k: int,
        objective_values: Mapping[str, float] | None = None,
        slate_specs: Iterable[SlateConstraintSpec] = (),
        execution_status: str = "success",
    ) -> VerificationReport:
        started = perf_counter()
        specs = list(specs)
        slate_specs = list(slate_specs)
        frame = bus.query()
        rows_by_id = {str(row["item_id"]): row for row in frame.to_dict("records")}
        violations: List[dict[str, Any]] = []
        if execution_status != "success":
            violations.append(
                {
                    "code": (
                        "optimizer_failure"
                        if execution_status == "optimizer_failed"
                        else execution_status
                    )
                }
            )
        if len(item_ids) != requested_k:
            violations.append(
                {
                    "code": "insufficient_candidates" if len(item_ids) < requested_k else "unexpected_list_length",
                    "expected": requested_k,
                    "actual": len(item_ids),
                }
            )
        if len(set(item_ids)) != len(item_ids):
            violations.append({"code": "duplicate_items"})
        for item_id in item_ids:
            row = rows_by_id.get(str(item_id))
            if row is None:
                violations.append({"code": "unknown_item", "item_id": item_id})
                continue
            if not bool(row["active"]):
                violations.append({"code": "inactive_candidate", "item_id": item_id})
            for spec in specs:
                result = self.constraints.evaluate(spec, row)
                if not result.satisfied:
                    violations.append(
                        {
                            "code": "hard_constraint_violation",
                            "item_id": item_id,
                            "constraint_id": spec.id,
                            "actual": result.actual,
                            "expected": spec.value,
                        }
                    )
        for name, value in (objective_values or {}).items():
            if not np.isfinite(float(value)):
                violations.append({"code": "non_finite_objective", "objective": name, "actual": value})
        if (
            len(item_ids) == requested_k
            and len(set(item_ids)) == len(item_ids)
            and all(str(item_id) in rows_by_id for item_id in item_ids)
        ):
            try:
                slate_results = self.slate_constraints.evaluate_all(
                    slate_specs, item_ids, frame
                )
            except Exception as exc:
                violations.append(
                    {
                        "code": "slate_constraint_evaluation_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                for spec, result in slate_results:
                    if not result.satisfied:
                        violations.append(
                            {
                                "code": "slate_constraint_violation",
                                "constraint_id": spec.id,
                                "actual": result.actual,
                                "expected": spec.value,
                                "operator": spec.operator,
                                "violation_magnitude": result.violation_magnitude,
                                "details": dict(result.details),
                            }
                        )
        report = VerificationReport(
            feasible=not violations,
            violations=violations,
            requested_k=requested_k,
            actual_k=len(item_ids),
            checked_constraints=[spec.id for spec in specs] + [spec.id for spec in slate_specs],
            checked_item_constraints=[spec.id for spec in specs],
            checked_slate_constraints=[spec.id for spec in slate_specs],
        )
        if bus.tracker:
            bus.tracker.record(
                module="Verifier",
                operation="verify",
                status="success" if report.feasible else "violation",
                before_version=bus.version,
                after_version=bus.version,
                before_candidates=len(frame),
                after_candidates=len(frame),
                duration_ms=(perf_counter() - started) * 1000,
                input_summary={"requested_k": requested_k, "actual_k": len(item_ids), "violation_count": len(violations)},
            )
        return report
