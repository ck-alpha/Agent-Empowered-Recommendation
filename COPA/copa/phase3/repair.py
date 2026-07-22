"""Constrained verifier-failure repair decisions."""

from __future__ import annotations

import json
from collections import Counter
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

from .models import RepairDecision, StructuredAgentCallResult
from .planner import _merge_usage


@dataclass(frozen=True)
class RepairConfig:
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.5
    prompt_version: str = "repair_v1"


class RepairPolicy:
    def validate(self, decision: RepairDecision, violation_codes: Sequence[str]) -> RepairDecision:
        codes = set(violation_codes)
        if codes & {"insufficient_candidates", "proven_infeasible"}:
            allowed = {"request_clarification", "abort"}
        elif "solver_unknown" in codes:
            allowed = {"replan", "abort"}
        elif "non_finite_objective" in codes:
            allowed = {"recompute_objectives", "abort"}
        elif codes & {
            "duplicate_items", "unknown_item", "inactive_candidate",
            "hard_constraint_violation", "unexpected_list_length",
            "slate_constraint_violation", "optimizer_failure",
        }:
            allowed = {"reexecute_selection", "replan", "abort"}
        else:
            allowed = {"abort"}
        if decision.action not in allowed:
            raise ValueError(
                f"Repair action {decision.action} is not allowed for violations {sorted(codes)}; "
                f"allowed={sorted(allowed)}"
            )
        return decision


class RepairAgent:
    system_prompt = "You are a constrained repair decision agent. Return structured JSON only."

    def __init__(
        self,
        client: Optional[StructuredLLMClient] = None,
        *,
        ollama_config: Optional[OllamaConfig] = None,
        config: Optional[RepairConfig] = None,
        policy: Optional[RepairPolicy] = None,
    ) -> None:
        self.client = client or OllamaStructuredClient(ollama_config)
        self.config = config or RepairConfig()
        self.policy = policy or RepairPolicy()
        path = Path(__file__).with_name("prompts") / f"{self.config.prompt_version}.txt"
        self.template = path.read_text(encoding="utf-8")

    def decide(
        self,
        violations: Sequence[Mapping[str, Any]],
        repair_summary: Mapping[str, Any],
    ) -> StructuredAgentCallResult:
        started = perf_counter()
        violation_codes = [str(item.get("code", "unknown")) for item in violations]
        verification_summary = {
            "violation_counts": dict(Counter(violation_codes)),
            "violation_count": len(violations),
        }
        schema = RepairDecision.model_json_schema()
        errors: list[str] = []
        usage: Dict[str, Any] = {}
        previous_output = ""
        for attempt in range(1, self.config.max_attempts + 1):
            validation_context = ""
            if errors:
                validation_context = (
                    "<validation_repair>\n"
                    f"Previous output: {previous_output[:3000]}\n"
                    f"Validation error: {errors[-1][:2000]}\n"
                    "Choose an allowed action without changing any constraint.\n"
                    "</validation_repair>"
                )
            prompt = self.template.format(
                verification_summary=json.dumps(verification_summary, ensure_ascii=False, sort_keys=True),
                repair_summary=json.dumps(dict(repair_summary), ensure_ascii=False, sort_keys=True),
                json_schema=json.dumps(schema, ensure_ascii=False, sort_keys=True),
                validation_context=validation_context,
            )
            try:
                response = self.client.generate_structured(
                    system=self.system_prompt,
                    prompt=prompt,
                    schema=schema,
                )
                previous_output = response.content
                usage = _merge_usage(usage, response.usage)
                decision = RepairDecision.model_validate_json(response.content)
                self.policy.validate(decision, violation_codes)
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
                "success", payload=decision, errors=errors, attempts=attempt,
                latency_seconds=perf_counter() - started, usage=usage,
            )
        raise AssertionError("unreachable repair state")
