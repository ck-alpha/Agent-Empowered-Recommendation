"""Independent deterministic verification of a final recommendation slate."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, List, Mapping

import numpy as np

from copa.constraints import ConstraintRegistry

from .bus import CandidateStateBus
from .models import ConstraintSpec, VerificationReport


class DeterministicVerifier:
    def __init__(self, constraints: ConstraintRegistry | None = None):
        self.constraints = constraints or ConstraintRegistry()

    def verify(
        self,
        item_ids: List[str],
        bus: CandidateStateBus,
        specs: Iterable[ConstraintSpec],
        requested_k: int,
        objective_values: Mapping[str, float] | None = None,
    ) -> VerificationReport:
        started = perf_counter()
        specs = list(specs)
        frame = bus.query()
        rows_by_id = {str(row["item_id"]): row for row in frame.to_dict("records")}
        violations: List[dict[str, Any]] = []
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
        report = VerificationReport(
            feasible=not violations,
            violations=violations,
            requested_k=requested_k,
            actual_k=len(item_ids),
            checked_constraints=[spec.id for spec in specs],
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
