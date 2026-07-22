"""Deterministic semantic validation and IR-to-Phase-1 conversion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from copa.core import CandidateRecord, ConstraintSpec, ObjectiveSpec, SlateConstraintSpec

from .domain import AttributeCapability, DomainSchema
from .models import (
    CompileIssue,
    CompiledConstraint,
    CompiledObjective,
    CompiledPlan,
    CompiledSlateConstraint,
    ConstraintIR,
)


class SemanticCompiler:
    def __init__(self, *, default_top_k: int = 10, min_top_k: int = 1, max_top_k: int = 50):
        self.default_top_k = default_top_k
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

    def compile(
        self,
        ir: ConstraintIR,
        domain: DomainSchema,
        candidates: Sequence[CandidateRecord],
        base_constraints: Sequence[ConstraintSpec] = (),
        base_slate_constraints: Sequence[SlateConstraintSpec] = (),
    ) -> CompiledPlan:
        issues: List[CompileIssue] = []
        assumptions: List[str] = []
        constraints: List[CompiledConstraint] = [
            CompiledConstraint(spec, "system") for spec in base_constraints
        ]
        slate_constraints: List[CompiledSlateConstraint] = [
            CompiledSlateConstraint(spec, "system") for spec in base_slate_constraints
        ]
        for unresolved in ir.unresolved_requirements:
            issues.append(
                CompileIssue(
                    code="unresolved_requirement",
                    severity="blocking",
                    message=unresolved.reason,
                    field=unresolved.text,
                    clarification_question=unresolved.clarification_question,
                )
            )

        for index, raw in enumerate(ir.hard_constraints, start=1):
            capability = domain.resolve_attribute(raw.attribute)
            if capability is None:
                issues.append(self._blocking("unknown_attribute", f"Unsupported attribute: {raw.attribute}", raw.attribute))
                continue
            if not domain.candidate_values(candidates, capability):
                issues.append(
                    self._blocking(
                        "attribute_has_no_values",
                        f"No candidate provides executable attribute {capability.executable_attribute}",
                        capability.logical_name,
                    )
                )
                continue
            if capability.logical_name == "price" and raw.currency:
                normalized_currency = raw.currency.strip().upper().replace("$", "USD")
                if normalized_currency not in {domain.currency, "US DOLLAR", "US DOLLARS"}:
                    issues.append(self._blocking("unsupported_currency", f"Cannot convert {raw.currency} to {domain.currency} without an exchange-rate policy", raw.attribute))
                    continue
            elif capability.logical_name == "price" and raw.currency is None:
                assumptions.append(f"price currency defaulted to domain currency {domain.currency}")
            if raw.scope == "slate":
                aggregation = str(raw.aggregation)
                if aggregation not in capability.slate_aggregations:
                    issues.append(
                        self._blocking(
                            "unsupported_slate_aggregation",
                            f"{aggregation} is not executable for {capability.logical_name}",
                            raw.attribute,
                        )
                    )
                    continue
                if raw.operator not in {"<", "<=", ">", ">=", "==", "between"}:
                    issues.append(
                        self._blocking(
                            "invalid_slate_operator",
                            f"Operator {raw.operator} cannot bound a slate aggregation",
                            raw.attribute,
                        )
                    )
                    continue
                normalized_value, value_issue = self._normalize_slate_bound(
                    raw.value, raw.operator, capability
                )
                if value_issue:
                    issues.append(value_issue)
                    continue
                target_values: list[Any] = []
                for target in raw.target_values:
                    resolved, target_issue = self._resolve_catalog_value(
                        target, capability, domain, candidates
                    )
                    if target_issue:
                        issues.append(target_issue)
                        break
                    target_values.append(resolved)
                else:
                    slate_constraints.append(
                        CompiledSlateConstraint(
                            SlateConstraintSpec(
                                f"user_slate_hc_{index:03d}",
                                aggregation,
                                capability.executable_attribute,
                                raw.operator,
                                normalized_value,
                                tuple(target_values),
                            ),
                            "user",
                        )
                    )
                continue
            if raw.operator not in capability.operators:
                issues.append(self._blocking("invalid_operator", f"Operator {raw.operator} is not valid for {capability.logical_name}", raw.attribute))
                continue
            normalized_value, value_issue = self._normalize_value(raw.value, raw.operator, capability, domain, candidates)
            if value_issue:
                issues.append(value_issue)
                continue
            spec = self._to_constraint_spec(index, capability, raw.operator, normalized_value)
            constraints.append(CompiledConstraint(spec, "user"))

        objectives = [
            CompiledObjective(ObjectiveSpec("relevance", "maximize", "candidate"), "system_default")
        ]
        for raw in ir.soft_objectives:
            capability = domain.resolve_objective(raw.objective)
            if capability is None:
                issues.append(self._blocking("unknown_objective", f"Unsupported objective: {raw.objective}", raw.objective))
                continue
            params: Dict[str, Any] = {}
            scope = "candidate" if capability.executable_name in {"relevance", "novelty"} else "slate"
            if "logical_attribute" in capability.params:
                attribute = domain.resolve_attribute(str(capability.params["logical_attribute"]))
                if attribute is None:
                    issues.append(self._blocking("objective_attribute_unavailable", f"Objective {raw.objective} is unavailable in domain {domain.name}", raw.objective))
                    continue
                if not domain.candidate_values(candidates, attribute):
                    issues.append(self._blocking("objective_attribute_has_no_values", f"No candidate provides {attribute.executable_attribute} for objective {raw.objective}", raw.objective))
                    continue
                params["attribute"] = attribute.executable_attribute
                params["registry_name"] = capability.executable_name
            elif capability.executable_name == "novelty":
                popularity = domain.resolve_attribute("popularity")
                if popularity and domain.candidate_values(candidates, popularity):
                    params["attribute"] = popularity.executable_attribute
                else:
                    issues.append(self._blocking("objective_attribute_has_no_values", f"No candidate provides popularity for objective {raw.objective}", raw.objective))
                    continue
            elif capability.executable_name == "fairness":
                group = domain.resolve_attribute("group") or domain.resolve_attribute("category")
                if not group:
                    issues.append(self._blocking("fairness_group_unavailable", "No auditable fairness group is configured for this domain", raw.objective))
                    continue
                if not domain.candidate_values(candidates, group):
                    issues.append(self._blocking("objective_attribute_has_no_values", f"No candidate provides {group.executable_attribute} for objective {raw.objective}", raw.objective))
                    continue
                params["attribute"] = group.executable_attribute
            objective = CompiledObjective(
                ObjectiveSpec(
                    capability.ir_name if capability.executable_name == "diversity" else capability.executable_name,
                    raw.direction,
                    scope,
                    params,
                ),
                "user",
            )
            if not self._objective_duplicate(objectives, objective):
                objectives.append(objective)

        if ir.top_k is None:
            top_k = self.default_top_k
            assumptions.append(f"top_k defaulted to {top_k}")
        elif not self.min_top_k <= ir.top_k <= self.max_top_k:
            top_k = ir.top_k
            issues.append(self._blocking("top_k_out_of_range", f"top_k must be between {self.min_top_k} and {self.max_top_k}", "top_k"))
        else:
            top_k = ir.top_k

        constraints, duplicate_issues = self._deduplicate_constraints(constraints)
        issues.extend(duplicate_issues)
        issues.extend(self._detect_conflicts(constraints))
        return CompiledPlan(
            constraints=constraints,
            objectives=objectives,
            top_k=top_k,
            slate_constraints=slate_constraints,
            assumptions=assumptions,
            issues=issues,
        )

    @staticmethod
    def _blocking(code: str, message: str, field: Optional[str] = None) -> CompileIssue:
        return CompileIssue(
            code=code,
            severity="blocking",
            message=message,
            field=field,
            clarification_question=f"Please clarify the requirement for {field or code}.",
        )

    def _normalize_value(
        self,
        value: Any,
        operator: str,
        capability: AttributeCapability,
        domain: DomainSchema,
        candidates: Sequence[CandidateRecord],
    ) -> Tuple[Any, Optional[CompileIssue]]:
        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                return value, self._blocking("invalid_between_value", "between requires exactly two bounds", capability.logical_name)
            try:
                lower, upper = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return value, self._blocking("invalid_numeric_value", "Numeric bounds are required", capability.logical_name)
            if lower > upper:
                return value, self._blocking("reversed_range", "Lower bound exceeds upper bound", capability.logical_name)
            return [lower, upper], None
        if capability.kind == "numeric":
            try:
                return float(value), None
            except (TypeError, ValueError):
                return value, self._blocking("invalid_numeric_value", f"A numeric value is required for {capability.logical_name}", capability.logical_name)
        if capability.kind == "boolean":
            if isinstance(value, bool):
                return value, None
            return value, self._blocking("invalid_boolean_value", f"A boolean value is required for {capability.logical_name}", capability.logical_name)
        if capability.kind == "set":
            raw_values = value if isinstance(value, list) else [value]
            if not raw_values:
                return value, self._blocking("invalid_collection_value", f"{operator} requires a non-empty value array", capability.logical_name)
            normalized: List[Any] = []
            for item in raw_values:
                resolved, issue = self._resolve_catalog_value(item, capability, domain, candidates)
                if issue:
                    return value, issue
                normalized.append(resolved)
            return normalized, None
        if operator in {"in", "not_in"}:
            if not isinstance(value, list) or not value:
                return value, self._blocking("invalid_collection_value", f"{operator} requires a non-empty array", capability.logical_name)
            normalized: List[Any] = []
            for item in value:
                resolved, issue = self._resolve_catalog_value(item, capability, domain, candidates)
                if issue:
                    return value, issue
                normalized.append(resolved)
            return normalized, None
        return self._resolve_catalog_value(value, capability, domain, candidates)

    def _normalize_slate_bound(
        self, value: Any, operator: str, capability: AttributeCapability
    ) -> Tuple[Any, Optional[CompileIssue]]:
        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                return value, self._blocking(
                    "invalid_between_value",
                    "between requires exactly two numeric bounds",
                    capability.logical_name,
                )
            try:
                lower, upper = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return value, self._blocking(
                    "invalid_numeric_value", "Slate bounds must be numeric", capability.logical_name
                )
            if lower > upper:
                return value, self._blocking(
                    "reversed_range", "Lower bound exceeds upper bound", capability.logical_name
                )
            return [lower, upper], None
        try:
            return float(value), None
        except (TypeError, ValueError):
            return value, self._blocking(
                "invalid_numeric_value", "Slate bounds must be numeric", capability.logical_name
            )

    def _resolve_catalog_value(
        self,
        value: Any,
        capability: AttributeCapability,
        domain: DomainSchema,
        candidates: Sequence[CandidateRecord],
    ) -> Tuple[Any, Optional[CompileIssue]]:
        alias_key = str(value).strip().casefold()
        if alias_key in capability.value_aliases:
            value = capability.value_aliases[alias_key]
        available = domain.candidate_values(candidates, capability)
        if not available:
            return value, self._blocking("attribute_has_no_values", f"No candidate has attribute {capability.logical_name}", capability.logical_name)
        exact = [candidate_value for candidate_value in available if candidate_value == value]
        if exact:
            return exact[0], None
        folded = [candidate_value for candidate_value in available if str(candidate_value).strip().casefold() == str(value).strip().casefold()]
        if len(folded) == 1:
            return folded[0], None
        return value, self._blocking("unknown_catalog_value", f"Value {value!r} is not present for {capability.logical_name} in the candidate set", capability.logical_name)

    @staticmethod
    def _to_constraint_spec(index: int, capability: AttributeCapability, operator: str, value: Any) -> ConstraintSpec:
        if capability.special_transform == "positive_inventory":
            truth = bool(value)
            if operator == "!=":
                truth = not truth
            return ConstraintSpec(
                f"user_hc_{index:03d}", "numeric", capability.executable_attribute,
                ">" if truth else "<=", 0,
            )
        if capability.executor_type:
            constraint_type = capability.executor_type
        elif capability.kind == "numeric":
            constraint_type = "numeric"
        elif capability.kind == "boolean":
            constraint_type = "boolean"
        elif capability.kind == "identifier" and operator == "not_in":
            constraint_type = "exclusion"
        else:
            constraint_type = "categorical"
        return ConstraintSpec(f"user_hc_{index:03d}", constraint_type, capability.executable_attribute, operator, value)

    @staticmethod
    def _objective_duplicate(existing: Sequence[CompiledObjective], candidate: CompiledObjective) -> bool:
        return any(
            item.spec.name == candidate.spec.name
            and item.spec.direction == candidate.spec.direction
            and dict(item.spec.params) == dict(candidate.spec.params)
            for item in existing
        )

    @staticmethod
    def _constraint_key(entry: CompiledConstraint) -> Tuple[Any, ...]:
        value = entry.spec.value
        if isinstance(value, list):
            value = tuple(value)
        return (entry.spec.type, entry.spec.attribute, entry.spec.operator, value)

    def _deduplicate_constraints(self, entries: Sequence[CompiledConstraint]) -> Tuple[List[CompiledConstraint], List[CompileIssue]]:
        output: List[CompiledConstraint] = []
        seen = set()
        issues: List[CompileIssue] = []
        for entry in entries:
            key = self._constraint_key(entry)
            if key in seen:
                issues.append(CompileIssue(code="duplicate_constraint", severity="warning", message=f"Duplicate constraint removed: {entry.spec.attribute}"))
                continue
            seen.add(key)
            output.append(entry)
        return output, issues

    def _detect_conflicts(self, entries: Sequence[CompiledConstraint]) -> List[CompileIssue]:
        issues: List[CompileIssue] = []
        numeric = defaultdict(list)
        categorical = defaultdict(list)
        for entry in entries:
            spec = entry.spec
            if spec.type == "numeric":
                numeric[spec.attribute].append(spec)
            elif spec.type in {"categorical", "exclusion"}:
                categorical[spec.attribute].append(spec)
        for attribute, specs in categorical.items():
            required = {str(spec.value) for spec in specs if spec.operator == "=="}
            allowed_sets = [set(map(str, spec.value)) for spec in specs if spec.operator == "in"]
            forbidden = {str(spec.value) for spec in specs if spec.operator == "!="}
            for spec in specs:
                if spec.operator == "not_in":
                    forbidden.update(map(str, spec.value))
            if len(required) > 1:
                issues.append(self._blocking("categorical_conflict", f"Mutually exclusive values for {attribute}: {sorted(required)}", attribute))
                continue
            allowed = set.intersection(*allowed_sets) if allowed_sets else None
            if allowed_sets and not allowed:
                issues.append(self._blocking("categorical_conflict", f"Allowed value sets for {attribute} have no intersection", attribute))
                continue
            if required:
                required_value = next(iter(required))
                if required_value in forbidden or (allowed is not None and required_value not in allowed):
                    issues.append(self._blocking("categorical_conflict", f"Required value {required_value!r} is excluded for {attribute}", attribute))
            elif allowed is not None and not (allowed - forbidden):
                issues.append(self._blocking("categorical_conflict", f"All allowed values are excluded for {attribute}", attribute))
        for attribute, specs in numeric.items():
            lower, upper = float("-inf"), float("inf")
            lower_inclusive = upper_inclusive = True
            excluded = set()
            for spec in specs:
                value = spec.value
                if spec.operator == "between":
                    candidate_lower, candidate_upper = float(value[0]), float(value[1])
                    if candidate_lower > lower:
                        lower, lower_inclusive = candidate_lower, True
                    if candidate_upper < upper:
                        upper, upper_inclusive = candidate_upper, True
                elif spec.operator in {">", ">="}:
                    candidate = float(value)
                    inclusive = spec.operator == ">="
                    if candidate > lower:
                        lower, lower_inclusive = candidate, inclusive
                    elif candidate == lower:
                        lower_inclusive = lower_inclusive and inclusive
                elif spec.operator in {"<", "<="}:
                    candidate = float(value)
                    inclusive = spec.operator == "<="
                    if candidate < upper:
                        upper, upper_inclusive = candidate, inclusive
                    elif candidate == upper:
                        upper_inclusive = upper_inclusive and inclusive
                elif spec.operator == "==":
                    candidate = float(value)
                    if candidate > lower:
                        lower, lower_inclusive = candidate, True
                    if candidate < upper:
                        upper, upper_inclusive = candidate, True
                elif spec.operator == "!=":
                    excluded.add(float(value))
            point_excluded = lower == upper and lower in excluded
            if lower > upper or (lower == upper and (not lower_inclusive or not upper_inclusive or point_excluded)):
                issues.append(self._blocking("numeric_conflict", f"Contradictory bounds for {attribute}", attribute))
        return issues
