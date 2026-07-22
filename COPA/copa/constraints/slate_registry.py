"""Deterministic slate-level hard constraints and exact MILP feasibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from copa.core import SlateConstraintEvaluation, SlateConstraintSpec

from .registry import resolve_attribute


COUNT_TYPES = {"distinct_count", "per_group_count", "group_count"}
OPERATORS = {"<", "<=", ">", ">=", "==", "between"}


@dataclass(frozen=True)
class SlateSolveResult:
    status: str
    item_ids: list[str] = field(default_factory=list)
    objective_value: float | None = None
    runtime_seconds: float = 0.0
    solver_status: int | None = None
    message: str = ""
    mip_gap: float | None = None
    mip_node_count: int | None = None


class SlateConstraintRegistry:
    """Evaluates complete slates and linearizes the built-in v1 constraint set."""

    TYPES = {"aggregate_sum", "distinct_count", "per_group_count", "group_count"}

    def compile(
        self, specs: Iterable[SlateConstraintSpec], frame: pd.DataFrame
    ) -> "CompiledSlateConstraintSet":
        """Validate once and build the array-backed optimizer hot path."""

        return CompiledSlateConstraintSet(self, self.validate(specs, frame), frame)

    def validate(
        self, specs: Iterable[SlateConstraintSpec], frame: pd.DataFrame
    ) -> list[SlateConstraintSpec]:
        normalized = list(specs)
        if len({spec.id for spec in normalized}) != len(normalized):
            raise ValueError("Slate constraint ids must be unique")
        rows = frame.to_dict("records")
        for spec in normalized:
            if not spec.id:
                raise ValueError("Slate constraint id cannot be empty")
            if spec.type not in self.TYPES:
                raise KeyError(f"Unknown slate constraint type: {spec.type}")
            if spec.operator not in OPERATORS:
                raise ValueError(f"Unsupported slate constraint operator: {spec.operator}")
            if spec.type == "group_count" and not tuple(spec.target_values):
                raise ValueError("group_count requires non-empty target_values")
            if spec.type != "group_count" and tuple(spec.target_values):
                raise ValueError(f"{spec.type} does not accept target_values")
            self._numeric_bounds(spec)
            for row in rows:
                actual = resolve_attribute(row, spec.attribute)
                if spec.type == "aggregate_sum":
                    number = float(actual)
                    if not np.isfinite(number):
                        raise ValueError(
                            f"Slate attribute {spec.attribute} contains a non-finite value"
                        )
                elif self._group_key(actual) is None:
                    raise ValueError(
                        f"Slate grouping attribute {spec.attribute} contains a null value"
                    )
        return normalized

    def evaluate(
        self,
        spec: SlateConstraintSpec,
        item_ids: Sequence[str],
        frame: pd.DataFrame,
    ) -> SlateConstraintEvaluation:
        self.validate([spec], frame)
        return self._evaluate_validated(spec, item_ids, frame)

    def _evaluate_validated(
        self,
        spec: SlateConstraintSpec,
        item_ids: Sequence[str],
        frame: pd.DataFrame,
    ) -> SlateConstraintEvaluation:
        selected = self._selected(item_ids, frame)
        if spec.type == "aggregate_sum":
            actual: Any = float(
                sum(float(resolve_attribute(row, spec.attribute)) for row in selected)
            )
            satisfied, magnitude = self._compare(actual, spec)
            details: Dict[str, Any] = {"selected_count": len(selected)}
        elif spec.type == "distinct_count":
            values = [self._group_key(resolve_attribute(row, spec.attribute)) for row in selected]
            actual = len(set(values))
            satisfied, magnitude = self._compare(float(actual), spec, scale=max(1, len(selected)))
            details = {"distinct_values": sorted(set(values), key=str)}
        elif spec.type == "group_count":
            targets = {self._group_key(value) for value in spec.target_values}
            actual = sum(
                self._group_key(resolve_attribute(row, spec.attribute)) in targets
                for row in selected
            )
            satisfied, magnitude = self._compare(float(actual), spec, scale=max(1, len(selected)))
            details = {"target_values": sorted(targets, key=str)}
        else:
            counts: Dict[Any, int] = {}
            for row in selected:
                value = self._group_key(resolve_attribute(row, spec.attribute))
                counts[value] = counts.get(value, 0) + 1
            evaluations = [
                self._compare(float(count), spec, scale=max(1, len(selected)))
                for count in counts.values()
            ]
            satisfied = all(value[0] for value in evaluations)
            magnitude = max((value[1] for value in evaluations), default=0.0)
            actual = {str(key): value for key, value in sorted(counts.items(), key=lambda x: str(x[0]))}
            details = {"group_count": len(counts)}
        return SlateConstraintEvaluation(bool(satisfied), actual, float(magnitude), details)

    def evaluate_all(
        self,
        specs: Iterable[SlateConstraintSpec],
        item_ids: Sequence[str],
        frame: pd.DataFrame,
    ) -> list[tuple[SlateConstraintSpec, SlateConstraintEvaluation]]:
        specs = self.validate(specs, frame)
        return [
            (spec, self._evaluate_validated(spec, item_ids, frame)) for spec in specs
        ]

    def is_feasible(
        self,
        specs: Iterable[SlateConstraintSpec],
        item_ids: Sequence[str],
        frame: pd.DataFrame,
    ) -> bool:
        return all(result.satisfied for _, result in self.evaluate_all(specs, item_ids, frame))

    def total_violation(
        self,
        specs: Iterable[SlateConstraintSpec],
        item_ids: Sequence[str],
        frame: pd.DataFrame,
    ) -> float:
        return float(
            sum(result.violation_magnitude for _, result in self.evaluate_all(specs, item_ids, frame))
        )

    def solve(
        self,
        frame: pd.DataFrame,
        specs: Iterable[SlateConstraintSpec],
        requested_k: int,
        *,
        objective_scores: Mapping[str, float] | None = None,
        time_limit_seconds: float = 2.0,
    ) -> SlateSolveResult:
        """Find an exact feasible K-set while maximizing optional additive scores."""

        started = perf_counter()
        specs = self.validate(specs, frame)
        if requested_k <= 0:
            raise ValueError("requested_k must be positive")
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        ordered = frame.copy()
        ordered["item_id"] = ordered["item_id"].astype(str)
        ordered = ordered.sort_values("item_id", kind="mergesort").reset_index(drop=True)
        item_ids = ordered["item_id"].tolist()
        n_items = len(item_ids)
        if n_items < requested_k:
            return SlateSolveResult(
                "infeasible",
                runtime_seconds=perf_counter() - started,
                message=f"Only {n_items} candidates are available for K={requested_k}",
            )

        distinct_layout: Dict[str, tuple[int, list[Any], Dict[Any, list[int]]]] = {}
        per_group_layout: Dict[str, tuple[int, list[Any], Dict[Any, list[int]]]] = {}
        variable_count = n_items
        for spec in specs:
            if spec.type not in {"distinct_count", "per_group_count"}:
                continue
            groups: Dict[Any, list[int]] = {}
            for index, row in enumerate(ordered.to_dict("records")):
                key = self._group_key(resolve_attribute(row, spec.attribute))
                groups.setdefault(key, []).append(index)
            values = sorted(groups, key=str)
            layout = (variable_count, values, groups)
            if spec.type == "distinct_count":
                distinct_layout[spec.id] = layout
            else:
                per_group_layout[spec.id] = layout
            variable_count += len(values)

        rows: list[Dict[int, float]] = []
        lower: list[float] = []
        upper: list[float] = []

        def add(coefficients: Mapping[int, float], low: float, high: float) -> None:
            rows.append(dict(coefficients))
            lower.append(float(low))
            upper.append(float(high))

        add({index: 1.0 for index in range(n_items)}, requested_k, requested_k)
        records = ordered.to_dict("records")
        for spec in specs:
            if spec.type == "aggregate_sum":
                coefficients = {
                    index: float(resolve_attribute(row, spec.attribute))
                    for index, row in enumerate(records)
                }
                low, high = self._linear_bounds(spec, count=False)
                add(coefficients, low, high)
            elif spec.type == "group_count":
                targets = {self._group_key(value) for value in spec.target_values}
                coefficients = {
                    index: float(self._group_key(resolve_attribute(row, spec.attribute)) in targets)
                    for index, row in enumerate(records)
                }
                low, high = self._linear_bounds(spec, count=True)
                add(coefficients, low, high)
            elif spec.type == "per_group_count":
                offset, values, groups = per_group_layout[spec.id]
                low, high = self._linear_bounds(spec, count=True)
                for group_position, value in enumerate(values):
                    y_index = offset + group_position
                    indices = groups[value]
                    # Bounds apply to groups represented in the selected slate.
                    # Candidate groups with y=0 are intentionally exempt from
                    # positive minima, matching the deterministic evaluator.
                    selected_count = {index: 1.0 for index in indices}
                    link_upper = dict(selected_count)
                    link_upper[y_index] = -float(requested_k)
                    add(link_upper, -np.inf, 0.0)
                    link_lower = {index: -1.0 for index in indices}
                    link_lower[y_index] = 1.0
                    add(link_lower, -np.inf, 0.0)
                    if np.isfinite(low):
                        lower_coefficients = dict(selected_count)
                        lower_coefficients[y_index] = -float(low)
                        add(lower_coefficients, 0.0, np.inf)
                    if np.isfinite(high):
                        upper_coefficients = dict(selected_count)
                        upper_coefficients[y_index] = -float(high)
                        add(upper_coefficients, -np.inf, 0.0)
            elif spec.type == "distinct_count":
                offset, values, groups = distinct_layout[spec.id]
                for group_position, value in enumerate(values):
                    y_index = offset + group_position
                    indices = groups[value]
                    # y == 1 iff any item from this group is selected.
                    coefficients = {index: 1.0 for index in indices}
                    coefficients[y_index] = -float(requested_k)
                    add(coefficients, -np.inf, 0.0)
                    coefficients = {index: -1.0 for index in indices}
                    coefficients[y_index] = 1.0
                    add(coefficients, -np.inf, 0.0)
                low, high = self._linear_bounds(spec, count=True)
                add(
                    {offset + position: 1.0 for position in range(len(values))},
                    low,
                    high,
                )

        matrix = np.zeros((len(rows), variable_count), dtype=float)
        for row_index, coefficients in enumerate(rows):
            for column, value in coefficients.items():
                matrix[row_index, column] = value
        scores = objective_scores or {}
        objective = np.zeros(variable_count, dtype=float)
        for index, item_id in enumerate(item_ids):
            objective[index] = -float(scores.get(item_id, 0.0)) + index * 1e-12
        result = milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
            options={
                "time_limit": float(time_limit_seconds),
                "mip_rel_gap": 0.0,
                "presolve": True,
                "disp": False,
            },
        )
        runtime = perf_counter() - started
        common = {
            "runtime_seconds": runtime,
            "solver_status": int(result.status),
            "message": str(result.message),
            "mip_gap": (
                float(result.mip_gap)
                if getattr(result, "mip_gap", None) is not None
                else None
            ),
            "mip_node_count": (
                int(result.mip_node_count)
                if getattr(result, "mip_node_count", None) is not None
                else None
            ),
        }
        if int(result.status) == 2:
            return SlateSolveResult("infeasible", **common)
        if int(result.status) != 0 or result.x is None:
            return SlateSolveResult("unknown", **common)
        selected = [item_ids[index] for index, value in enumerate(result.x[:n_items]) if value > 0.5]
        if len(selected) != requested_k or not self.is_feasible(specs, selected, ordered):
            return SlateSolveResult(
                "unknown",
                runtime_seconds=runtime,
                solver_status=int(result.status),
                message="MILP returned a solution that failed deterministic re-verification",
            )
        objective_value = float(sum(float(scores.get(item_id, 0.0)) for item_id in selected))
        return SlateSolveResult("optimal", selected, objective_value, **common)

    @staticmethod
    def _selected(item_ids: Sequence[str], frame: pd.DataFrame) -> list[Mapping[str, Any]]:
        if len(set(map(str, item_ids))) != len(item_ids):
            raise ValueError("Slate contains duplicate item ids")
        rows = {str(row["item_id"]): row for row in frame.to_dict("records")}
        unknown = [str(item_id) for item_id in item_ids if str(item_id) not in rows]
        if unknown:
            raise KeyError(f"Unknown slate item ids: {unknown[:5]}")
        return [rows[str(item_id)] for item_id in item_ids]

    @staticmethod
    def _group_key(value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _numeric_bounds(spec: SlateConstraintSpec) -> tuple[float, float]:
        if spec.operator == "between":
            if not isinstance(spec.value, (list, tuple)) or len(spec.value) != 2:
                raise TypeError("between expects a two-element [lower, upper] value")
            low, high = float(spec.value[0]), float(spec.value[1])
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("Invalid finite between bounds")
            if spec.type in COUNT_TYPES and (
                not low.is_integer() or not high.is_integer() or low < 0
            ):
                raise ValueError("Count constraint bounds must be non-negative integers")
            return low, high
        try:
            target = float(spec.value)
        except (TypeError, ValueError) as exc:
            raise TypeError("Slate constraint value must be numeric") from exc
        if not np.isfinite(target):
            raise ValueError("Slate constraint value must be finite")
        if spec.type in COUNT_TYPES and target < 0:
            raise ValueError("Count constraint values cannot be negative")
        if spec.type in COUNT_TYPES and not target.is_integer():
            raise ValueError("Count constraint values must be integers")
        return target, target

    @classmethod
    def _linear_bounds(
        cls, spec: SlateConstraintSpec, *, count: bool
    ) -> tuple[float, float]:
        low, high = cls._numeric_bounds(spec)
        if spec.operator == "between":
            return low, high
        target = low
        if spec.operator == "<=":
            return -np.inf, target
        if spec.operator == "<":
            return -np.inf, (np.ceil(target) - 1 if count else np.nextafter(target, -np.inf))
        if spec.operator == ">=":
            return target, np.inf
        if spec.operator == ">":
            return (np.floor(target) + 1 if count else np.nextafter(target, np.inf)), np.inf
        if spec.operator == "==":
            return target, target
        raise ValueError(f"Unsupported slate constraint operator: {spec.operator}")

    @classmethod
    def _compare(
        cls, actual: float, spec: SlateConstraintSpec, *, scale: float | None = None
    ) -> tuple[bool, float]:
        low, high = cls._numeric_bounds(spec)
        if spec.operator == "between":
            satisfied = low <= actual <= high
            violation = max(low - actual, actual - high, 0.0)
            denominator = scale or max(abs(low), abs(high), 1.0)
        else:
            target = low
            operations = {
                "<": actual < target,
                "<=": actual <= target,
                ">": actual > target,
                ">=": actual >= target,
                "==": actual == target,
            }
            satisfied = bool(operations[spec.operator])
            if satisfied:
                violation = 0.0
            elif spec.operator in {"<", "<="}:
                violation = max(actual - target, np.finfo(float).eps)
            elif spec.operator in {">", ">="}:
                violation = max(target - actual, np.finfo(float).eps)
            else:
                violation = abs(actual - target)
            denominator = scale or max(abs(target), 1.0)
        return bool(satisfied), float(violation / denominator)


@dataclass(frozen=True)
class _CompiledConstraint:
    spec: SlateConstraintSpec
    kind: str
    values: np.ndarray
    group_count: int = 0


@dataclass
class CompiledSlateState:
    """Incremental statistics for one feasible unordered slate."""

    selected_indices: list[int]
    selected_set: set[int]
    statistics: list[Any]


class CompiledSlateConstraintSet:
    """Array-backed equivalent of :class:`SlateConstraintRegistry`.

    The public registry remains the independent reference implementation. This
    object is request-local and is used only by the evolutionary hot path.
    """

    def __init__(
        self,
        registry: SlateConstraintRegistry,
        specs: Sequence[SlateConstraintSpec],
        frame: pd.DataFrame,
    ) -> None:
        self.registry = registry
        self.specs = tuple(specs)
        records = frame.to_dict("records")
        self.item_ids = tuple(str(row["item_id"]) for row in records)
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("Compiled slate candidates require unique item ids")
        self.index_by_item_id = {
            item_id: index for index, item_id in enumerate(self.item_ids)
        }
        compiled: list[_CompiledConstraint] = []
        for spec in self.specs:
            raw_values = [resolve_attribute(row, spec.attribute) for row in records]
            if spec.type == "aggregate_sum":
                compiled.append(
                    _CompiledConstraint(
                        spec,
                        spec.type,
                        np.asarray(raw_values, dtype=float),
                    )
                )
                continue
            group_values = [registry._group_key(value) for value in raw_values]
            if spec.type == "group_count":
                targets = {
                    registry._group_key(value) for value in spec.target_values
                }
                compiled.append(
                    _CompiledConstraint(
                        spec,
                        spec.type,
                        np.asarray([value in targets for value in group_values], dtype=np.int8),
                    )
                )
                continue
            ordered_groups = sorted(set(group_values), key=str)
            group_index = {
                value: index for index, value in enumerate(ordered_groups)
            }
            compiled.append(
                _CompiledConstraint(
                    spec,
                    spec.type,
                    np.asarray([group_index[value] for value in group_values], dtype=np.int32),
                    len(ordered_groups),
                )
            )
        self._compiled = tuple(compiled)

    def _indices(self, item_ids: Sequence[str]) -> list[int]:
        normalized = [str(item_id) for item_id in item_ids]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Slate contains duplicate item ids")
        unknown = [
            item_id for item_id in normalized if item_id not in self.index_by_item_id
        ]
        if unknown:
            raise KeyError(f"Unknown slate item ids: {unknown[:5]}")
        return [self.index_by_item_id[item_id] for item_id in normalized]

    def build_state(self, item_ids: Sequence[str]) -> CompiledSlateState:
        indices = self._indices(item_ids)
        statistics: list[Any] = []
        selected = np.asarray(indices, dtype=np.int64)
        for constraint in self._compiled:
            if constraint.kind == "aggregate_sum":
                statistics.append(float(constraint.values[selected].sum()))
            elif constraint.kind == "group_count":
                statistics.append(int(constraint.values[selected].sum()))
            else:
                statistics.append(
                    np.bincount(
                        constraint.values[selected], minlength=constraint.group_count
                    ).astype(np.int16, copy=False)
                )
        return CompiledSlateState(indices, set(indices), statistics)

    def _evaluation_from_statistic(
        self, constraint: _CompiledConstraint, statistic: Any, slate_size: int
    ) -> SlateConstraintEvaluation:
        if constraint.kind == "aggregate_sum":
            actual: Any = float(statistic)
            satisfied, magnitude = self.registry._compare(actual, constraint.spec)
            details: Dict[str, Any] = {"selected_count": slate_size}
        elif constraint.kind == "distinct_count":
            actual = int(np.count_nonzero(statistic))
            satisfied, magnitude = self.registry._compare(
                float(actual), constraint.spec, scale=max(1, slate_size)
            )
            details = {"distinct_count": actual}
        elif constraint.kind == "group_count":
            actual = int(statistic)
            satisfied, magnitude = self.registry._compare(
                float(actual), constraint.spec, scale=max(1, slate_size)
            )
            details = {"target_count": actual}
        else:
            represented = np.asarray(statistic)[np.asarray(statistic) > 0]
            comparisons = [
                self.registry._compare(
                    float(value), constraint.spec, scale=max(1, slate_size)
                )
                for value in represented
            ]
            satisfied = all(value[0] for value in comparisons)
            magnitude = max((value[1] for value in comparisons), default=0.0)
            actual = represented.astype(int).tolist()
            details = {"group_count": int(len(represented))}
        return SlateConstraintEvaluation(
            bool(satisfied), actual, float(magnitude), details
        )

    def evaluate_state(
        self, state: CompiledSlateState
    ) -> list[tuple[SlateConstraintSpec, SlateConstraintEvaluation]]:
        slate_size = len(state.selected_indices)
        return [
            (
                constraint.spec,
                self._evaluation_from_statistic(
                    constraint, state.statistics[index], slate_size
                ),
            )
            for index, constraint in enumerate(self._compiled)
        ]

    def evaluate_all(
        self, item_ids: Sequence[str]
    ) -> list[tuple[SlateConstraintSpec, SlateConstraintEvaluation]]:
        return self.evaluate_state(self.build_state(item_ids))

    def is_feasible(self, item_ids: Sequence[str]) -> bool:
        return self.state_is_feasible(self.build_state(item_ids))

    def state_is_feasible(self, state: CompiledSlateState) -> bool:
        return all(result.satisfied for _, result in self.evaluate_state(state))

    def total_violation(self, item_ids: Sequence[str]) -> float:
        return self.state_total_violation(self.build_state(item_ids))

    def state_total_violation(self, state: CompiledSlateState) -> float:
        return float(
            sum(
                result.violation_magnitude
                for _, result in self.evaluate_state(state)
            )
        )

    def try_swap(
        self,
        state: CompiledSlateState,
        position: int,
        incoming_item_id: str,
    ) -> CompiledSlateState | None:
        """Return the incrementally updated state iff the swap is feasible."""

        incoming = self.index_by_item_id.get(str(incoming_item_id))
        if incoming is None:
            raise KeyError(f"Unknown slate item id: {incoming_item_id}")
        if position < 0 or position >= len(state.selected_indices):
            raise IndexError("Slate swap position is out of range")
        outgoing = state.selected_indices[position]
        if incoming == outgoing:
            return state
        if incoming in state.selected_set:
            return None
        updated_statistics: list[Any] = []
        for constraint, statistic in zip(self._compiled, state.statistics):
            if constraint.kind in {"aggregate_sum", "group_count"}:
                updated_statistics.append(
                    statistic
                    - constraint.values[outgoing].item()
                    + constraint.values[incoming].item()
                )
            else:
                counts = np.asarray(statistic).copy()
                counts[int(constraint.values[outgoing])] -= 1
                counts[int(constraint.values[incoming])] += 1
                updated_statistics.append(counts)
        selected_indices = state.selected_indices.copy()
        selected_indices[position] = incoming
        selected_set = state.selected_set.copy()
        selected_set.remove(outgoing)
        selected_set.add(incoming)
        updated = CompiledSlateState(
            selected_indices, selected_set, updated_statistics
        )
        return updated if self.state_is_feasible(updated) else None

    def item_ids_for_state(self, state: CompiledSlateState) -> list[str]:
        return [self.item_ids[index] for index in state.selected_indices]
