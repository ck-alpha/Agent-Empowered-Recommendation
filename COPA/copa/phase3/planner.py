"""Structured Qwen planner and deterministic plan policy gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, Mapping, Optional, Sequence

from pydantic import ValidationError

from copa.phase2.ollama import (
    OllamaConfig,
    OllamaStructuredClient,
    OllamaTransportError,
    StructuredLLMClient,
)

from .models import AgentExecutionPlan, StructuredAgentCallResult


@dataclass(frozen=True)
class PlannerConfig:
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.5
    prompt_version: str = "planner_v1"


class PlanPolicy:
    """Rejects LLM plans that violate deterministic execution invariants."""

    @staticmethod
    def expected_strategy(objective_names: Sequence[str]) -> str:
        return "feasible_topk" if set(objective_names) <= {"relevance"} else "pareto"

    def validate(self, plan: AgentExecutionPlan, objective_names: Sequence[str]) -> AgentExecutionPlan:
        if plan.unresolved_requirements:
            raise ValueError(f"Planner left unresolved requirements: {plan.unresolved_requirements}")
        expected_strategy = self.expected_strategy(objective_names)
        if plan.strategy != expected_strategy:
            raise ValueError(
                f"Strategy must be {expected_strategy} for objectives {list(objective_names)}"
            )
        selection = "select_feasible_topk" if expected_strategy == "feasible_topk" else "optimize_pareto"
        expected_steps = ["apply_constraints", "compute_objectives", selection, "verify"]
        actual_steps = [step.tool for step in plan.steps]
        if actual_steps != expected_steps:
            raise ValueError(f"Tool sequence must be exactly {expected_steps}; received {actual_steps}")
        return plan


class PlannerAgent:
    system_prompt = "You are a constrained recommendation execution planner. Return structured JSON only."

    def __init__(
        self,
        client: Optional[StructuredLLMClient] = None,
        *,
        ollama_config: Optional[OllamaConfig] = None,
        config: Optional[PlannerConfig] = None,
        policy: Optional[PlanPolicy] = None,
    ) -> None:
        self.client = client or OllamaStructuredClient(ollama_config)
        self.config = config or PlannerConfig()
        self.policy = policy or PlanPolicy()
        path = Path(__file__).with_name("prompts") / f"{self.config.prompt_version}.txt"
        self.template = path.read_text(encoding="utf-8")

    def plan(
        self,
        compiled_plan: Mapping[str, Any],
        candidate_statistics: Mapping[str, Any],
    ) -> StructuredAgentCallResult:
        started = perf_counter()
        objective_names = [str(entry["spec"]["name"]) for entry in compiled_plan.get("objectives", [])]
        issues = list(compiled_plan.get("issues", []))
        summary = {
            "constraint_count": len(compiled_plan.get("constraints", [])),
            "objective_names": objective_names,
            "top_k": compiled_plan.get("top_k"),
            "blocking_issue_codes": [
                issue.get("code") for issue in issues if issue.get("severity") == "blocking"
            ],
            "warning_issue_codes": [
                issue.get("code") for issue in issues if issue.get("severity") == "warning"
            ],
            "executable": not any(issue.get("severity") == "blocking" for issue in issues),
        }
        schema = AgentExecutionPlan.model_json_schema()
        errors: list[str] = []
        usage: Dict[str, Any] = {}
        previous_output = ""
        for attempt in range(1, self.config.max_attempts + 1):
            repair_context = ""
            if errors:
                repair_context = (
                    "<validation_repair>\n"
                    f"Previous output: {previous_output[:3000]}\n"
                    f"Validation error: {errors[-1][:2000]}\n"
                    "Correct the plan without changing the compiled requirements.\n"
                    "</validation_repair>"
                )
            prompt = self.template.format(
                plan_summary=json.dumps(summary, ensure_ascii=False, sort_keys=True),
                candidate_statistics=json.dumps(dict(candidate_statistics), ensure_ascii=False, sort_keys=True),
                json_schema=json.dumps(schema, ensure_ascii=False, sort_keys=True),
                repair_context=repair_context,
            )
            try:
                response = self.client.generate_structured(
                    system=self.system_prompt,
                    prompt=prompt,
                    schema=schema,
                )
                previous_output = response.content
                usage = _merge_usage(usage, response.usage)
                plan = AgentExecutionPlan.model_validate_json(response.content)
                self.policy.validate(plan, objective_names)
            except (OllamaTransportError, ValidationError, ValueError, TypeError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.config.max_attempts:
                    sleep(self.config.retry_backoff_seconds)
                    continue
                return StructuredAgentCallResult(
                    "failed", errors=errors, attempts=attempt,
                    latency_seconds=perf_counter() - started, usage=usage,
                )
            return StructuredAgentCallResult(
                "success", payload=plan, errors=errors, attempts=attempt,
                latency_seconds=perf_counter() - started, usage=usage,
            )
        raise AssertionError("unreachable planner state")


def _merge_usage(existing: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    output = dict(existing)
    for key, value in dict(current).items():
        if isinstance(value, (int, float)) and isinstance(output.get(key, 0), (int, float)):
            output[key] = output.get(key, 0) + value
        else:
            output[key] = value
    return output
