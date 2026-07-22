"""Qwen-backed, fail-closed natural-language constraint compiler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from pydantic import ValidationError

from copa.constraints import ConstraintRegistry, SlateConstraintRegistry
from copa.core import (
    CandidateRecord,
    CandidateStateBus,
    ConstraintSpec,
    SlateConstraintSpec,
)

from .audit import CompilerAuditLogger
from .domain import DomainSchema
from .models import CompileIssue, CompileResult, ConstraintIR
from .ollama import OllamaConfig, OllamaResponse, OllamaStructuredClient, OllamaTransportError, StructuredLLMClient
from .prompting import ConstraintCompilerPrompt
from .semantic import SemanticCompiler


@dataclass(frozen=True)
class CompilerConfig:
    max_attempts: int = 2
    retry_backoff_seconds: float = 0.5
    audit_dir: Optional[Path] = None
    prompt_version: str = "constraint_compiler_v4"


class ConstraintCompiler:
    def __init__(
        self,
        client: Optional[StructuredLLMClient] = None,
        *,
        ollama_config: Optional[OllamaConfig] = None,
        config: Optional[CompilerConfig] = None,
        semantic: Optional[SemanticCompiler] = None,
        prompt: Optional[ConstraintCompilerPrompt] = None,
    ) -> None:
        self.client = client or OllamaStructuredClient(ollama_config)
        self.config = config or CompilerConfig()
        self.semantic = semantic or SemanticCompiler()
        self.prompt = prompt or ConstraintCompilerPrompt(version=self.config.prompt_version)
        if self.config.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def compile(
        self,
        text: str,
        domain_schema: DomainSchema,
        candidates: Sequence[CandidateRecord],
        base_constraints: Sequence[ConstraintSpec] = (),
        clarification_answers: Sequence[str] = (),
        base_slate_constraints: Sequence[SlateConstraintSpec] = (),
    ) -> CompileResult:
        started = perf_counter()
        request_id = uuid4().hex
        audit_text = text if isinstance(text, str) else ""
        if clarification_answers:
            audit_text += "\n" + "\n".join(str(answer) for answer in clarification_answers)
        request_hash = hashlib.sha256(audit_text.encode("utf-8")).hexdigest()
        audit_path = self.config.audit_dir / f"compile_{request_id}.jsonl" if self.config.audit_dir else None
        audit = CompilerAuditLogger(audit_path)
        if not isinstance(text, str) or not text.strip():
            issue = CompileIssue(code="empty_request", severity="blocking", message="Natural-language request cannot be empty")
            result = CompileResult("failed", issues=[issue], errors=[issue.message], audit_path=audit_path)
            self._audit(audit, request_id, request_hash, audit_text, domain_schema, result)
            return result
        if not candidates:
            issue = CompileIssue(code="empty_candidates", severity="blocking", message="Candidate collection cannot be empty")
            result = CompileResult("failed", issues=[issue], errors=[issue.message], audit_path=audit_path)
            self._audit(audit, request_id, request_hash, audit_text, domain_schema, result)
            return result

        schema = ConstraintIR.model_json_schema()
        domain_payload = domain_schema.prompt_summary(candidates)
        previous_output: Optional[str] = None
        validation_error: Optional[str] = None
        errors: List[str] = []
        usage: Dict[str, Any] = {}
        for attempt in range(1, self.config.max_attempts + 1):
            prompt = self.prompt.render(
                user_request=text.strip(),
                domain_schema=domain_payload,
                json_schema=schema,
                previous_output=previous_output,
                validation_error=validation_error,
                clarification_answers=clarification_answers,
            )
            attempt_started = perf_counter()
            try:
                response = self.client.generate_structured(
                    system=self.prompt.system_prompt,
                    prompt=prompt,
                    schema=schema,
                )
                previous_output = response.content
                usage = self._merge_usage(usage, response.usage)
                ir = ConstraintIR.model_validate_json(response.content)
            except (OllamaTransportError, ValidationError, ValueError, TypeError) as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                errors.append(validation_error)
                audit.record(
                    {
                        "event": "attempt",
                        "request_id": request_id,
                        "request_sha256": request_hash,
                        "request_length": len(audit_text),
                        "domain": domain_schema.name,
                        "prompt_version": self.prompt.version,
                        "attempt": attempt,
                        "status": "failed",
                        "error": validation_error[:2000],
                        "latency_seconds": perf_counter() - attempt_started,
                    }
                )
                if attempt < self.config.max_attempts:
                    sleep(self.config.retry_backoff_seconds)
                    continue
                result = CompileResult(
                    "failed",
                    errors=errors,
                    attempts=attempt,
                    latency_seconds=perf_counter() - started,
                    usage=usage,
                    audit_path=audit_path,
                )
                self._audit(audit, request_id, request_hash, audit_text, domain_schema, result)
                return result

            plan = self.semantic.compile(
                ir,
                domain_schema,
                candidates,
                base_constraints,
                base_slate_constraints,
            )
            self._preflight(plan, candidates)
            blocking = [issue for issue in plan.issues if issue.severity == "blocking"]
            status = "clarification_required" if blocking else "success"
            result = CompileResult(
                status,
                ir=ir,
                plan=plan,
                issues=plan.issues,
                errors=errors,
                attempts=attempt,
                latency_seconds=perf_counter() - started,
                usage=usage,
                audit_path=audit_path,
            )
            self._audit(audit, request_id, request_hash, audit_text, domain_schema, result)
            return result

        raise AssertionError("unreachable compiler state")

    @staticmethod
    def _preflight(plan: Any, candidates: Sequence[CandidateRecord]) -> None:
        if any(issue.severity == "blocking" for issue in plan.issues):
            return
        bus = CandidateStateBus()
        bus.initialize(candidates, source="compiler_preflight")
        try:
            if plan.executable_constraints:
                ConstraintRegistry().apply(bus, plan.executable_constraints)
        except (KeyError, TypeError, ValueError) as exc:
            plan.issues.append(
                CompileIssue(
                    code="invalid_candidate_metadata",
                    severity="blocking",
                    message=f"Candidate metadata cannot execute the compiled constraints: {type(exc).__name__}: {exc}",
                    clarification_question="Please provide candidates with metadata matching the selected domain schema.",
                )
            )
            return
        feasible_frame = bus.query(feasible_only=True)
        feasible = len(feasible_frame)
        if feasible == 0:
            plan.issues.append(
                CompileIssue(
                    code="zero_feasible_candidates",
                    severity="blocking",
                    message="The compiled hard constraints leave no feasible candidate",
                    clarification_question="Would you like to relax one of the hard constraints?",
                )
            )
        elif feasible < plan.top_k:
            plan.issues.append(
                CompileIssue(
                    code="candidate_shortage",
                    severity="blocking",
                    message=f"Only {feasible} candidates satisfy the constraints for requested top_k={plan.top_k}",
                    clarification_question="Please provide more candidates or revise the hard requirements.",
                )
            )
        elif plan.executable_slate_constraints:
            try:
                solved = SlateConstraintRegistry().solve(
                    feasible_frame,
                    plan.executable_slate_constraints,
                    plan.top_k,
                    time_limit_seconds=2.0,
                )
            except (KeyError, TypeError, ValueError) as exc:
                plan.issues.append(
                    CompileIssue(
                        code="invalid_slate_constraint_metadata",
                        severity="blocking",
                        message=f"Slate constraints cannot execute: {type(exc).__name__}: {exc}",
                        clarification_question="Please use a slate requirement supported by the available metadata.",
                    )
                )
                return
            if solved.status == "infeasible":
                plan.issues.append(
                    CompileIssue(
                        code="slate_constraints_infeasible",
                        severity="blocking",
                        message="No full-K slate satisfies all compiled hard constraints.",
                        clarification_question="Would you like to revise one of the slate requirements?",
                    )
                )
            elif solved.status != "optimal":
                plan.issues.append(
                    CompileIssue(
                        code="slate_solver_unknown",
                        severity="blocking",
                        message="Slate feasibility could not be established within the solver limit.",
                        clarification_question="Please retry or provide a smaller candidate set.",
                    )
                )

    @staticmethod
    def _merge_usage(existing: Dict[str, Any], current: Any) -> Dict[str, Any]:
        output = dict(existing)
        for key, value in dict(current).items():
            if isinstance(value, (int, float)) and isinstance(output.get(key, 0), (int, float)):
                output[key] = output.get(key, 0) + value
            else:
                output[key] = value
        return output

    def _audit(
        self,
        audit: CompilerAuditLogger,
        request_id: str,
        request_hash: str,
        text: Any,
        domain: DomainSchema,
        result: CompileResult,
    ) -> None:
        audit.record(
            {
                "event": "compile_complete",
                "request_id": request_id,
                "request_sha256": request_hash,
                "request_length": len(text) if isinstance(text, str) else 0,
                "domain": domain.name,
                "prompt_version": self.prompt.version,
                "model": getattr(getattr(self.client, "config", None), "model", "injected_client"),
                "status": result.status,
                "attempts": result.attempts,
                "issue_codes": [issue.code for issue in result.issues],
                "latency_seconds": result.latency_seconds,
                "usage": result.usage,
            }
        )
