"""Allow-listed deterministic tools for Agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from copa.constraints import SlateConstraintRegistry, resolve_attribute
from copa.session import COPAExecutionSession

from .models import AgentExecutionPlan


ToolHandler = Callable[[COPAExecutionSession], Any]


class AgentToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolHandler] = {}
        self.register("apply_constraints", lambda session: session.apply_constraints())
        self.register("compute_objectives", lambda session: session.compute_objectives())
        self.register("select_feasible_topk", lambda session: session.select_feasible_topk())
        self.register("optimize_pareto", lambda session: session.optimize_pareto())
        self.register("verify", lambda session: session.verify())

    def register(self, name: str, handler: ToolHandler, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name in self._tools and not replace:
            raise KeyError(f"Agent tool already registered: {name}")
        self._tools[name] = handler

    def execute(self, name: str, session: COPAExecutionSession) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown Agent tool: {name}")
        return self._tools[name](session)


@dataclass
class ToolExecutionOutput:
    session: COPAExecutionSession
    tool_summaries: list[Dict[str, Any]]


class AgentPlanExecutor:
    def __init__(self, registry: Optional[AgentToolRegistry] = None):
        self.registry = registry or AgentToolRegistry()

    def execute(
        self,
        plan: AgentExecutionPlan,
        session: COPAExecutionSession,
        *,
        fault: Optional[Mapping[str, Any]] = None,
    ) -> ToolExecutionOutput:
        summaries: list[Dict[str, Any]] = []
        for sequence, step in enumerate(plan.steps):
            if step.tool == "verify" and fault:
                report = self._verify_with_fault(session, fault)
                result: Any = report
            else:
                result = self.registry.execute(step.tool, session)
            summaries.append(self._summarize(step.tool, sequence, session, result))
        return ToolExecutionOutput(session, summaries)

    @staticmethod
    def _verify_with_fault(session: COPAExecutionSession, fault: Mapping[str, Any]):
        if session.selected is None:
            raise RuntimeError("Cannot inject verifier fault before selection")
        kind = str(fault.get("type", ""))
        item_ids = list(session.selected.item_ids)
        objective_values = dict(session.selected.objective_values)
        if kind == "duplicate" and item_ids:
            item_ids[-1] = item_ids[0]
        elif kind == "unknown" and item_ids:
            item_ids[-1] = "__fault_unknown_item__"
        elif kind == "shortage" and item_ids:
            item_ids = item_ids[:-1]
        elif kind == "non_finite" and objective_values:
            objective_values[next(iter(objective_values))] = float("nan")
        elif kind in {
            "total_price_over",
            "brand_cap_over",
            "coverage_under",
            "forged_optimizer_feasible",
        }:
            requested_type = {
                "total_price_over": "aggregate_sum",
                "brand_cap_over": "per_group_count",
                "coverage_under": "distinct_count",
                "forged_optimizer_feasible": None,
            }[kind]
            item_ids = AgentPlanExecutor._violating_slate(
                session, requested_type=requested_type
            )
            if kind == "forged_optimizer_feasible":
                session.selected.constraint_feasible = True
                session.selected.constraint_violation = 0.0
        elif kind:
            raise ValueError(f"Unknown deterministic fault type: {kind}")
        session.selected.item_ids = item_ids
        session.selected.objective_values = objective_values
        return session.verify(item_ids=item_ids, objective_values=objective_values)

    @staticmethod
    def _violating_slate(
        session: COPAExecutionSession, *, requested_type: Optional[str]
    ) -> list[str]:
        specs = [
            spec
            for spec in session.request.slate_constraints
            if requested_type is None or spec.type == requested_type
        ]
        if not specs:
            raise ValueError(
                f"Fault requires a {requested_type or 'slate'} hard constraint"
            )
        spec = specs[0]
        frame = session.bus.query(feasible_only=True)
        requested_k = session.request.optimization.top_k
        rows = frame.to_dict("records")
        if len(rows) < requested_k:
            raise ValueError("Fault fixture has fewer than K item-feasible candidates")

        reverse_bound = spec.operator in {"<", "<="}
        if spec.type == "aggregate_sum":
            ordered = sorted(
                rows,
                key=lambda row: (
                    float(resolve_attribute(row, spec.attribute)), str(row["item_id"])
                ),
                reverse=reverse_bound,
            )
        else:
            groups: Dict[Any, list[Mapping[str, Any]]] = {}
            for row in rows:
                groups.setdefault(resolve_attribute(row, spec.attribute), []).append(row)
            for values in groups.values():
                values.sort(key=lambda row: str(row["item_id"]))
            if spec.type == "distinct_count":
                if spec.operator in {">", ">="}:
                    ordered = [
                        row
                        for _, values in sorted(
                            groups.items(), key=lambda pair: (-len(pair[1]), str(pair[0]))
                        )
                        for row in values
                    ]
                else:
                    ordered = []
                    maximum = max(len(values) for values in groups.values())
                    for position in range(maximum):
                        for key in sorted(groups, key=str):
                            if position < len(groups[key]):
                                ordered.append(groups[key][position])
            elif spec.type == "per_group_count":
                ordered = [
                    row
                    for _, values in sorted(
                        groups.items(), key=lambda pair: (-len(pair[1]), str(pair[0]))
                    )
                    for row in values
                ]
            else:
                targets = set(spec.target_values)
                target_rows = [
                    row
                    for row in rows
                    if resolve_attribute(row, spec.attribute) in targets
                ]
                other_rows = [row for row in rows if row not in target_rows]
                ordered = (
                    target_rows + other_rows
                    if spec.operator in {"<", "<="}
                    else other_rows + target_rows
                )
        item_ids = [str(row["item_id"]) for row in ordered[:requested_k]]
        evaluated = SlateConstraintRegistry().evaluate(spec, item_ids, frame)
        if evaluated.satisfied:
            raise ValueError(
                f"Fault fixture cannot produce a violation for slate constraint {spec.id}"
            )
        return item_ids

    @staticmethod
    def _summarize(
        tool: str,
        sequence: int,
        session: COPAExecutionSession,
        result: Any,
    ) -> Dict[str, Any]:
        summary = {
            "tool": tool,
            "sequence": sequence,
            "bus_version": session.bus.version,
            "feasible_candidates": len(session.bus.query(feasible_only=True)),
        }
        if tool in {"select_feasible_topk", "optimize_pareto"} and session.selected:
            summary["selected_count"] = len(session.selected.item_ids)
        if tool == "verify":
            summary["feasible"] = bool(result.feasible)
            summary["violation_codes"] = [item["code"] for item in result.violations]
        return summary
