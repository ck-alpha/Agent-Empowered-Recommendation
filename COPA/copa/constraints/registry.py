"""Extensible deterministic hard-constraint registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from copa.core import CandidateStateBus, ConstraintSpec


ConstraintEvaluator = Callable[[Any, str, Any], bool]


@dataclass(frozen=True)
class ConstraintEvaluation:
    satisfied: bool
    actual: Any


def resolve_attribute(row: Mapping[str, Any], attribute: str) -> Any:
    """Resolve a direct candidate field or a dotted path inside metadata."""
    if attribute in row and attribute != "metadata":
        return row[attribute]
    current: Any = row.get("metadata", {})
    path = attribute.split(".")
    if path and path[0] == "metadata":
        path = path[1:]
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing candidate attribute: {attribute}")
        current = current[part]
    return current


class ConstraintRegistry:
    """Maps constraint types to deterministic evaluators and applies them atomically."""

    NUMERIC_OPERATORS = {"<", "<=", ">", ">=", "==", "!=", "between"}
    CATEGORICAL_OPERATORS = {"==", "!=", "in", "not_in"}

    def __init__(self) -> None:
        self._evaluators: Dict[str, ConstraintEvaluator] = {}
        self.register("numeric", self._numeric)
        self.register("categorical", self._categorical)
        self.register("exclusion", self._exclusion)
        self.register("boolean", self._boolean)
        self.register("set_membership", self._set_membership)

    def register(self, name: str, evaluator: ConstraintEvaluator, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("Constraint type name cannot be empty")
        if name in self._evaluators and not replace:
            raise KeyError(f"Constraint type already registered: {name}")
        self._evaluators[name] = evaluator

    def validate(self, specs: Iterable[ConstraintSpec], frame: pd.DataFrame) -> List[ConstraintSpec]:
        normalized = list(specs)
        if len({spec.id for spec in normalized}) != len(normalized):
            raise ValueError("Constraint ids must be unique")
        rows = frame.to_dict("records")
        for spec in normalized:
            if spec.type not in self._evaluators:
                raise KeyError(f"Unknown constraint type: {spec.type}")
            if spec.type == "numeric" and spec.operator not in self.NUMERIC_OPERATORS:
                raise ValueError(f"Unsupported numeric operator: {spec.operator}")
            if spec.type == "categorical" and spec.operator not in self.CATEGORICAL_OPERATORS:
                raise ValueError(f"Unsupported categorical operator: {spec.operator}")
            for row in rows:
                actual = resolve_attribute(row, spec.attribute)
                self._evaluators[spec.type](actual, spec.operator, spec.value)
        return normalized

    def evaluate(self, spec: ConstraintSpec, row: Mapping[str, Any]) -> ConstraintEvaluation:
        if spec.type not in self._evaluators:
            raise KeyError(f"Unknown constraint type: {spec.type}")
        actual = resolve_attribute(row, spec.attribute)
        return ConstraintEvaluation(
            satisfied=bool(self._evaluators[spec.type](actual, spec.operator, spec.value)),
            actual=actual,
        )

    def apply(self, bus: CandidateStateBus, specs: Iterable[ConstraintSpec]) -> int:
        specs = self.validate(specs, bus.query())

        def updater(frame: pd.DataFrame) -> pd.DataFrame:
            output = frame.copy(deep=True)
            hard_states: List[Dict[str, Any]] = []
            active: List[bool] = []
            for row in output.to_dict("records"):
                violations: List[Dict[str, Any]] = []
                for spec in specs:
                    result = self.evaluate(spec, row)
                    if not result.satisfied:
                        violations.append(
                            {
                                "constraint_id": spec.id,
                                "attribute": spec.attribute,
                                "operator": spec.operator,
                                "expected": spec.value,
                                "actual": result.actual,
                            }
                        )
                feasible = not violations
                hard_states.append({"feasible": feasible, "violations": violations})
                active.append(feasible)
            output["hard_state"] = hard_states
            output["active"] = active
            return output

        return bus.update(
            updater,
            module="HardConstraintModule",
            operation="apply_constraints",
            input_summary={
                "constraint_ids": [spec.id for spec in specs],
                "constraint_count": len(specs),
            },
        )

    @staticmethod
    def _numeric(actual: Any, operator: str, expected: Any) -> bool:
        try:
            actual_number = float(actual)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Numeric constraint received non-numeric value: {actual!r}") from exc
        if not np.isfinite(actual_number):
            raise ValueError(f"Numeric constraint received non-finite value: {actual!r}")
        if operator == "between":
            if not isinstance(expected, (list, tuple)) or len(expected) != 2:
                raise TypeError("between expects a two-element [lower, upper] value")
            lower, upper = float(expected[0]), float(expected[1])
            if lower > upper:
                raise ValueError("between lower bound cannot exceed upper bound")
            return lower <= actual_number <= upper
        try:
            target = float(expected)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Numeric constraint expected value must be numeric: {expected!r}") from exc
        operations: Dict[str, Callable[[float, float], bool]] = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        if operator not in operations:
            raise ValueError(f"Unsupported numeric operator: {operator}")
        return bool(operations[operator](actual_number, target))

    @staticmethod
    def _categorical(actual: Any, operator: str, expected: Any) -> bool:
        if operator == "==":
            return actual == expected
        if operator == "!=":
            return actual != expected
        if operator in {"in", "not_in"}:
            if isinstance(expected, (str, bytes)) or not isinstance(expected, (list, tuple, set, frozenset)):
                raise TypeError(f"{operator} expects a collection value")
            contained = actual in expected
            return contained if operator == "in" else not contained
        raise ValueError(f"Unsupported categorical operator: {operator}")

    @staticmethod
    def _exclusion(actual: Any, operator: str, expected: Any) -> bool:
        del operator
        if isinstance(expected, (str, bytes)) or not isinstance(expected, (list, tuple, set, frozenset)):
            raise TypeError("exclusion expects a collection of forbidden values")
        return actual not in expected

    @staticmethod
    def _boolean(actual: Any, operator: str, expected: Any) -> bool:
        if not isinstance(actual, (bool, np.bool_)):
            raise TypeError(f"Boolean constraint received non-boolean value: {actual!r}")
        if not isinstance(expected, (bool, np.bool_)):
            raise TypeError(f"Boolean constraint expected value must be boolean: {expected!r}")
        if operator not in {"==", "!="}:
            raise ValueError(f"Unsupported boolean operator: {operator}")
        return bool(actual == expected) if operator == "==" else bool(actual != expected)

    @staticmethod
    def _set_membership(actual: Any, operator: str, expected: Any) -> bool:
        if isinstance(actual, (str, bytes)) or not isinstance(actual, (list, tuple, set, frozenset)):
            raise TypeError("set_membership requires a collection-valued candidate attribute")
        expected_values = expected if isinstance(expected, (list, tuple, set, frozenset)) else [expected]
        if not expected_values:
            raise ValueError("set_membership expected values cannot be empty")
        actual_set, expected_set = set(actual), set(expected_values)
        if operator == "contains_any":
            return bool(actual_set & expected_set)
        if operator == "contains_all":
            return expected_set <= actual_set
        if operator == "not_contains":
            return not bool(actual_set & expected_set)
        raise ValueError(f"Unsupported set_membership operator: {operator}")
