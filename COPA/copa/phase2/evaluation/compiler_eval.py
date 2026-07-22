"""Gold-set evaluation for constraint compilation quality and latency."""

from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry
from copa.core import CandidateRecord

from ..compiler import ConstraintCompiler
from ..domain import DomainSchema
from ..models import ConstraintIR
from ..semantic import SemanticCompiler


SCENARIOS = {"seen", "compositional", "unseen"}


def load_gold_cases(path: Path | str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid gold JSONL at line {line_number}: {exc}") from exc
            required = {"id", "language", "scenario", "text", "expected_status", "constraints", "objectives", "top_k"}
            missing = required - set(case)
            if missing:
                raise ValueError(f"Gold case {case.get('id', line_number)} missing fields: {sorted(missing)}")
            if case["scenario"] not in SCENARIOS:
                raise ValueError(
                    f"Gold case {case.get('id', line_number)} has unsupported scenario: {case['scenario']!r}"
                )
            cases.append(case)
    return cases


def _freeze_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _constraint_set(constraints: Iterable[Mapping[str, Any]]) -> set[Tuple[str, ...]]:
    return {
        (
            str(item.get("scope", "item")),
            str(item.get("aggregation")),
            str(item["attribute"]),
            str(item["operator"]),
            _freeze_value(item.get("value")),
            _freeze_value(item.get("target_values", [])),
        )
        for item in constraints
    }


def _prf(tp: int, predicted: int, expected: int) -> Tuple[float, float, float]:
    precision = tp / predicted if predicted else float(expected == 0)
    recall = tp / expected if expected else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _feasible_ids(
    candidates: Sequence[CandidateRecord],
    constraints: Sequence[Any],
) -> set[str]:
    registry = ConstraintRegistry()
    output: set[str] = set()
    for candidate in candidates:
        row = candidate.to_row()
        if all(registry.evaluate(spec, row).satisfied for spec in constraints):
            output.add(str(candidate.item_id))
    return output


def _expected_constraints(
    case: Mapping[str, Any],
    domain: DomainSchema,
    candidates: Sequence[CandidateRecord],
) -> tuple[Sequence[Any], Sequence[Any]]:
    ir = ConstraintIR.model_validate(
        {
            "schema_version": str(case.get("schema_version", "1.0")),
            "hard_constraints": list(case["constraints"]),
            "soft_objectives": [],
            "top_k": case["top_k"],
            "unresolved_requirements": [],
        }
    )
    plan = SemanticCompiler().compile(ir, domain, candidates)
    blocking = [issue for issue in plan.issues if issue.severity == "blocking"]
    if blocking:
        raise ValueError(
            f"Executable gold case {case['id']} has invalid expected constraints: "
            f"{[issue.code for issue in blocking]}"
        )
    return plan.executable_constraints, plan.executable_slate_constraints


def _feasible_slates(
    candidates: Sequence[CandidateRecord],
    constraints: Sequence[Any],
    slate_constraints: Sequence[Any],
    top_k: int,
) -> set[tuple[str, ...]]:
    registry = ConstraintRegistry()
    item_feasible = [
        candidate
        for candidate in candidates
        if all(
            registry.evaluate(spec, candidate.to_row()).satisfied
            for spec in constraints
        )
    ]
    frame = pd.DataFrame([candidate.to_row() for candidate in item_feasible])
    slate_registry = SlateConstraintRegistry()
    return {
        tuple(candidate.item_id for candidate in combination)
        for combination in itertools.combinations(item_feasible, int(top_k))
        if slate_registry.is_feasible(
            slate_constraints,
            [candidate.item_id for candidate in combination],
            frame,
        )
    }


def _scenario_summary(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for scenario, group in frame.groupby("scenario", sort=True):
        executable = group["execution_exact"].dropna()
        output[str(scenario)] = {
            "case_count": int(len(group)),
            "parsing_accuracy": float(group["parsing_exact"].mean()),
            "schema_valid_rate": float(group["schema_valid"].mean()),
            "status_accuracy": float(group["status_correct"].mean()),
            "hard_exact_match_rate": float(group["hard_exact"].mean()),
            "objective_exact_match_rate": float(group["objective_exact"].mean()),
            "top_k_accuracy": float(group["top_k_correct"].mean()),
            "constraint_execution_accuracy": float(executable.mean()) if len(executable) else None,
        }
    return output


def evaluate_compiler(
    compiler: ConstraintCompiler,
    cases: Sequence[Mapping[str, Any]],
    domain: DomainSchema,
    candidates: Sequence[CandidateRecord],
    output_dir: Path | str,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("Compiler evaluation requires at least one gold case")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    hard_tp = hard_predicted = hard_expected = 0
    objective_tp = objective_predicted = objective_expected = 0
    execution_tp = execution_predicted = execution_expected = 0
    for case in cases:
        result = compiler.compile(str(case["text"]), domain, candidates)
        predicted_constraints: List[Dict[str, Any]] = []
        predicted_objectives: List[str] = []
        predicted_top_k = None
        if result.plan:
            predicted_constraints = [
                {
                    "scope": "item",
                    "attribute": entry.spec.attribute,
                    "operator": entry.spec.operator,
                    "value": entry.spec.value,
                }
                for entry in result.plan.constraints
                if entry.provenance == "user"
            ]
            predicted_constraints.extend(
                {
                    "scope": "slate",
                    "aggregation": entry.spec.type,
                    "attribute": entry.spec.attribute,
                    "operator": entry.spec.operator,
                    "value": entry.spec.value,
                    "target_values": list(entry.spec.target_values),
                }
                for entry in result.plan.slate_constraints
                if entry.provenance == "user"
            )
            predicted_objectives = [entry.spec.name for entry in result.plan.objectives]
            predicted_top_k = result.plan.top_k
        expected_specs: Sequence[Any] | None = None
        expected_slate_specs: Sequence[Any] | None = None
        if case["expected_status"] == "success":
            expected_specs, expected_slate_specs = _expected_constraints(
                case, domain, candidates
            )
            expected_constraints = [
                {
                    "scope": "item",
                    "attribute": spec.attribute,
                    "operator": spec.operator,
                    "value": spec.value,
                }
                for spec in expected_specs
            ]
            expected_constraints.extend(
                {
                    "scope": "slate",
                    "aggregation": spec.type,
                    "attribute": spec.attribute,
                    "operator": spec.operator,
                    "value": spec.value,
                    "target_values": list(spec.target_values),
                }
                for spec in expected_slate_specs
            )
        else:
            expected_constraints = list(case["constraints"])
        expected_objectives = list(case["objectives"])
        predicted_hard_set = _constraint_set(predicted_constraints)
        expected_hard_set = _constraint_set(expected_constraints)
        predicted_objective_set = set(map(str, predicted_objectives))
        expected_objective_set = set(map(str, expected_objectives))
        hard_tp += len(predicted_hard_set & expected_hard_set)
        hard_predicted += len(predicted_hard_set)
        hard_expected += len(expected_hard_set)
        objective_tp += len(predicted_objective_set & expected_objective_set)
        objective_predicted += len(predicted_objective_set)
        objective_expected += len(expected_objective_set)
        expected_clarification = case["expected_status"] == "clarification_required"
        predicted_clarification = result.status == "clarification_required"
        parsing_exact = bool(
            result.ir is not None
            and result.status == case["expected_status"]
            and predicted_hard_set == expected_hard_set
            and predicted_objective_set == expected_objective_set
            and predicted_top_k == case["top_k"]
        )
        execution_exact = None
        expected_feasible_count = predicted_feasible_count = None
        if case["expected_status"] == "success":
            assert expected_specs is not None and expected_slate_specs is not None
            expected_feasible = _feasible_ids(candidates, expected_specs)
            predicted_feasible = (
                _feasible_ids(candidates, result.plan.executable_constraints)
                if result.succeeded and result.plan is not None
                else set()
            )
            predicted_slate_specs = (
                result.plan.executable_slate_constraints
                if result.succeeded and result.plan is not None
                else []
            )
            if expected_slate_specs or predicted_slate_specs:
                fixture = list(candidates[:12])
                expected_slates = _feasible_slates(
                    fixture,
                    expected_specs,
                    expected_slate_specs,
                    int(case["top_k"]),
                )
                predicted_slates = _feasible_slates(
                    fixture,
                    result.plan.executable_constraints if result.plan else [],
                    predicted_slate_specs,
                    int(case["top_k"]),
                )
                execution_exact = predicted_slates == expected_slates
            else:
                execution_exact = predicted_feasible == expected_feasible
            expected_feasible_count = len(expected_feasible)
            predicted_feasible_count = len(predicted_feasible)
            execution_tp += len(predicted_feasible & expected_feasible)
            execution_predicted += len(predicted_feasible)
            execution_expected += len(expected_feasible)
        records.append(
            {
                "id": case["id"],
                "language": case["language"],
                "scenario": case["scenario"],
                "status": result.status,
                "expected_status": case["expected_status"],
                "status_correct": result.status == case["expected_status"],
                "schema_valid": result.ir is not None,
                "executable": result.status == "success",
                "top_k_correct": predicted_top_k == case["top_k"],
                "hard_exact": predicted_hard_set == expected_hard_set,
                "objective_exact": predicted_objective_set == expected_objective_set,
                "parsing_exact": parsing_exact,
                "execution_exact": execution_exact,
                "expected_feasible_count": expected_feasible_count,
                "predicted_feasible_count": predicted_feasible_count,
                "expected_clarification": expected_clarification,
                "predicted_clarification": predicted_clarification,
                "attempts": result.attempts,
                "latency_seconds": result.latency_seconds,
                "prompt_tokens": int(result.usage.get("prompt_eval_count", 0)),
                "output_tokens": int(result.usage.get("eval_count", 0)),
                "issue_codes": ",".join(issue.code for issue in result.issues),
                "predicted_constraints": json.dumps(predicted_constraints, ensure_ascii=False),
                "predicted_objectives": json.dumps(predicted_objectives, ensure_ascii=False),
            }
        )
    frame = pd.DataFrame(records)
    hard_precision, hard_recall, hard_f1 = _prf(hard_tp, hard_predicted, hard_expected)
    objective_precision, objective_recall, objective_f1 = _prf(objective_tp, objective_predicted, objective_expected)
    clarification_tp = int(((frame["expected_clarification"]) & (frame["predicted_clarification"])).sum())
    clarification_precision, clarification_recall, clarification_f1 = _prf(
        clarification_tp,
        int(frame["predicted_clarification"].sum()),
        int(frame["expected_clarification"].sum()),
    )
    execution_precision, execution_recall, execution_f1 = _prf(
        execution_tp, execution_predicted, execution_expected
    )
    executable_rows = frame["execution_exact"].dropna()
    summary = {
        "case_count": len(frame),
        "parsing_accuracy": float(frame["parsing_exact"].mean()),
        "schema_valid_rate": float(frame["schema_valid"].mean()),
        "semantic_executable_rate": float(frame["executable"].mean()),
        "status_accuracy": float(frame["status_correct"].mean()),
        "hard_constraint_precision": hard_precision,
        "hard_constraint_recall": hard_recall,
        "hard_constraint_f1": hard_f1,
        "objective_precision": objective_precision,
        "objective_recall": objective_recall,
        "objective_f1": objective_f1,
        "top_k_accuracy": float(frame["top_k_correct"].mean()),
        "clarification_precision": clarification_precision,
        "clarification_recall": clarification_recall,
        "clarification_f1": clarification_f1,
        "constraint_execution_accuracy": float(executable_rows.mean()) if len(executable_rows) else None,
        "candidate_feasibility_precision": execution_precision,
        "candidate_feasibility_recall": execution_recall,
        "candidate_feasibility_f1": execution_f1,
        "retry_rate": float((frame["attempts"] > 1).mean()),
        "latency_p50_seconds": float(frame["latency_seconds"].quantile(0.5)),
        "latency_p95_seconds": float(frame["latency_seconds"].quantile(0.95)),
        "prompt_tokens": int(frame["prompt_tokens"].sum()),
        "output_tokens": int(frame["output_tokens"].sum()),
        "mean_prompt_tokens": float(frame["prompt_tokens"].mean()),
        "mean_output_tokens": float(frame["output_tokens"].mean()),
        "scenario_metrics": _scenario_summary(frame),
    }
    frame.to_csv(output_dir / "compiler_case_results.csv", index=False)
    with (output_dir / "compiler_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
