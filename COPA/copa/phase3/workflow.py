"""Checkpointed LangGraph orchestration for COPA Phase 3."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from copa.core import (
    ConstraintSpec,
    ObjectiveSpec,
    RecommendationRequest,
    RecommendationResult,
    SlateConstraintSpec,
    SlateSolution,
    VerificationReport,
)
from copa.phase2 import ConstraintCompiler, get_domain_schema
from copa.session import COPAExecutionSession

from .models import (
    AgentExecutionPlan,
    AgentRecommendationRequest,
    AgentRecommendationResult,
    RepairDecision,
)
from .planner import PlanPolicy, PlannerAgent
from .repair import RepairAgent
from .tools import AgentPlanExecutor
from .tracker import AgentExecutionTracker


class AgentGraphState(TypedDict, total=False):
    thread_id: str
    request: Dict[str, Any]
    status: str
    compile_result: Dict[str, Any]
    compiled_plan: Dict[str, Any]
    execution_plan: Dict[str, Any]
    recommendation: Dict[str, Any]
    clarification: Dict[str, Any]
    clarification_answers: list[str]
    repair_history: list[Dict[str, Any]]
    repair_count: int
    last_violation_signature: str
    repair_route: str
    errors: list[str]
    diagnostics: Dict[str, Any]
    candidate_trace_paths: list[str]


@dataclass(frozen=True)
class AgentGraphConfig:
    max_repairs: int = 2
    checkpoint_path: Path = Path("COPA/checkpoints/phase3.sqlite")
    trace_dir: Path = Path("COPA/logs/phase3")
    candidate_trace_dir: Path = Path("COPA/logs/phase3/candidates")
    enable_fault_injection: bool = False


class AgentCOPAPipeline:
    def __init__(
        self,
        compiler: ConstraintCompiler,
        planner: PlannerAgent,
        repair_agent: RepairAgent,
        *,
        config: Optional[AgentGraphConfig] = None,
        executor: Optional[AgentPlanExecutor] = None,
        checkpointer: Any = None,
    ) -> None:
        self.compiler = compiler
        self.planner = planner
        self.repair_agent = repair_agent
        self.config = config or AgentGraphConfig()
        self.executor = executor or AgentPlanExecutor()
        self._connection: Optional[sqlite3.Connection] = None
        if checkpointer is None:
            os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
            self.config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.config.checkpoint_path,
                check_same_thread=False,
            )
            checkpointer = SqliteSaver(self._connection)
            self.config.checkpoint_path.chmod(0o600)
        self.checkpointer = checkpointer
        self.graph = self._build_graph().compile(
            checkpointer=self.checkpointer,
            name="copa_phase3_agent",
        )

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentGraphState)
        graph.add_node("compile_request", self._compile_request)
        graph.add_node("await_clarification", self._await_clarification)
        graph.add_node("plan_execution", self._plan_execution)
        graph.add_node("validate_plan", self._validate_plan)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("verify_result", self._verify_result)
        graph.add_node("decide_repair", self._decide_repair)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "compile_request")
        graph.add_conditional_edges(
            "compile_request",
            self._route_after_compile,
            {
                "clarification": "await_clarification",
                "plan": "plan_execution",
                "finalize": "finalize",
            },
        )
        graph.add_edge("await_clarification", "compile_request")
        graph.add_conditional_edges(
            "plan_execution",
            lambda state: "validate" if state.get("status") != "failed" else "finalize",
            {"validate": "validate_plan", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "validate_plan",
            lambda state: "execute" if state.get("status") != "failed" else "finalize",
            {"execute": "execute_tools", "finalize": "finalize"},
        )
        graph.add_edge("execute_tools", "verify_result")
        graph.add_conditional_edges(
            "verify_result",
            lambda state: "finalize" if state.get("status") == "success" else "repair",
            {"finalize": "finalize", "repair": "decide_repair"},
        )
        graph.add_conditional_edges(
            "decide_repair",
            lambda state: state.get("repair_route", "finalize"),
            {
                "execute": "execute_tools",
                "plan": "plan_execution",
                "clarification": "await_clarification",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph

    def run(
        self,
        request: AgentRecommendationRequest,
        thread_id: Optional[str] = None,
    ) -> AgentRecommendationResult:
        thread_id = thread_id or uuid4().hex
        config = self._thread_config(thread_id)
        existing = self.graph.get_state(config)
        if existing.values:
            raise ValueError(f"Agent thread already exists: {thread_id}")
        initial: AgentGraphState = {
            "thread_id": thread_id,
            "request": request.to_state_dict(),
            "status": "running",
            "clarification_answers": [],
            "repair_history": [],
            "repair_count": 0,
            "errors": [],
            "diagnostics": {
                "request_sha256": hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
                "candidate_count": len(request.candidates),
                "system_constraints_sha256": _constraint_fingerprint(
                    _scoped_constraint_payload(request.to_state_dict())
                ),
            },
            "candidate_trace_paths": [],
        }
        self.graph.invoke(initial, config=config)
        return self._result_from_thread(thread_id)

    def resume(self, thread_id: str, answer: str) -> AgentRecommendationResult:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Clarification answer cannot be empty")
        config = self._thread_config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"Unknown Agent thread: {thread_id}")
        if not snapshot.next or "await_clarification" not in snapshot.next:
            raise ValueError(f"Agent thread is not awaiting clarification: {thread_id}")
        self.graph.invoke(Command(resume={"answer": answer.strip()}), config=config)
        return self._result_from_thread(thread_id)

    def get_state(self, thread_id: str) -> Dict[str, Any]:
        snapshot = self.graph.get_state(self._thread_config(thread_id))
        if not snapshot.values:
            raise KeyError(f"Unknown Agent thread: {thread_id}")
        return dict(snapshot.values)

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def cleanup(self, older_than_days: int) -> list[str]:
        if older_than_days < 0:
            raise ValueError("older_than_days cannot be negative")
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        newest: Dict[str, float] = {}
        for item in self.checkpointer.list(None):
            thread_id = str(item.config.get("configurable", {}).get("thread_id", ""))
            timestamp = str(item.checkpoint.get("ts", ""))
            if not thread_id or not timestamp:
                continue
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
            newest[thread_id] = max(newest.get(thread_id, 0.0), parsed)
        deleted = sorted(thread_id for thread_id, timestamp in newest.items() if timestamp < cutoff)
        for thread_id in deleted:
            self.delete_thread(thread_id)
        return deleted

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _compile_request(self, state: AgentGraphState) -> Dict[str, Any]:
        started = perf_counter()
        request = AgentRecommendationRequest.from_state_dict(state["request"])
        answers = list(state.get("clarification_answers", []))
        text = request.text
        result = self.compiler.compile(
            text,
            get_domain_schema(request.domain),
            request.candidates,
            request.base_constraints,
            clarification_answers=answers,
            base_slate_constraints=request.base_slate_constraints,
        )
        compiled_payload = result.to_dict()
        update: Dict[str, Any] = {
            "compile_result": compiled_payload,
            "status": result.status,
        }
        if result.plan:
            previous_compile = state.get("compile_result", {})
            current = result.plan.to_dict()
            update["compiled_plan"] = current
            expected_system_fingerprint = _constraint_fingerprint(
                _scoped_constraint_payload(request.to_state_dict())
            )
            actual_system_fingerprint = _constraint_fingerprint(
                [
                    {"scope": scope, **dict(entry["spec"])}
                    for scope, entries in (
                        ("item", current.get("constraints", [])),
                        ("slate", current.get("slate_constraints", [])),
                    )
                    for entry in entries
                    if entry.get("provenance") == "system"
                ]
            )
            if actual_system_fingerprint != expected_system_fingerprint:
                update["status"] = "failed"
                update["errors"] = [
                    *state.get("errors", []),
                    "system_constraint_integrity_violation",
                ]
            if previous_compile and answers:
                diagnostics = dict(state.get("diagnostics", {}))
                diagnostics.setdefault("user_confirmed_constraint_changes", []).append(
                    _ir_constraint_diff(previous_compile, compiled_payload)
                )
                update["diagnostics"] = diagnostics
        if result.status == "clarification_required":
            questions = [
                issue.clarification_question for issue in result.issues
                if issue.severity == "blocking" and issue.clarification_question
            ]
            update["clarification"] = {
                "source": "compiler",
                "questions": questions,
                "issue_codes": [issue.code for issue in result.issues],
            }
        self._tracker(state).record(
            node="compile_request", operation="compile", status=result.status,
            attempt=len(answers), duration_ms=(perf_counter() - started) * 1000,
            input_summary={
                "request_length": len(text),
                "clarification_count": len(answers),
                "clarification_length": sum(len(answer) for answer in answers),
                "candidate_count": len(request.candidates),
            },
            output_summary={"issue_codes": [issue.code for issue in result.issues]},
        )
        return update

    @staticmethod
    def _route_after_compile(state: AgentGraphState) -> str:
        if state.get("status") == "success":
            return "plan"
        if state.get("status") == "clarification_required":
            return "clarification"
        return "finalize"

    def _await_clarification(self, state: AgentGraphState) -> Dict[str, Any]:
        payload = {
            "thread_id": state["thread_id"],
            **dict(state.get("clarification", {})),
        }
        response = interrupt(payload)
        answer = response.get("answer") if isinstance(response, Mapping) else response
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Clarification resume payload must contain a non-empty answer")
        answers = [*state.get("clarification_answers", []), answer.strip()]
        self._tracker(state).record(
            node="await_clarification", operation="resume", status="success",
            attempt=len(answers), input_summary={"answer_length": len(answer.strip())},
        )
        return {
            "clarification_answers": answers,
            "clarification": {},
            "status": "running",
        }

    def _plan_execution(self, state: AgentGraphState) -> Dict[str, Any]:
        started = perf_counter()
        request = AgentRecommendationRequest.from_state_dict(state["request"])
        plan_result = self.planner.plan(
            state["compiled_plan"],
            {
                "candidate_count": len(request.candidates),
                "repair_count": state.get("repair_count", 0),
            },
        )
        diagnostics = _append_llm_diagnostics(
            state.get("diagnostics", {}), "planner", plan_result
        )
        update: Dict[str, Any] = {"diagnostics": diagnostics}
        if plan_result.status == "failed" or plan_result.payload is None:
            update.update({
                "status": "failed",
                "errors": [*state.get("errors", []), *plan_result.errors],
            })
        else:
            update.update({
                "status": "running",
                "execution_plan": plan_result.payload.model_dump(mode="json"),
            })
        self._tracker(state).record(
            node="plan_execution", operation="plan", status=plan_result.status,
            attempt=state.get("repair_count", 0), duration_ms=(perf_counter() - started) * 1000,
            output_summary={"attempts": plan_result.attempts},
            error="; ".join(plan_result.errors) if plan_result.errors else None,
        )
        return update

    def _validate_plan(self, state: AgentGraphState) -> Dict[str, Any]:
        try:
            plan = AgentExecutionPlan.model_validate(state["execution_plan"])
            objective_names = [
                str(entry["spec"]["name"])
                for entry in state["compiled_plan"].get("objectives", [])
            ]
            PlanPolicy().validate(plan, objective_names)
        except (KeyError, ValueError, TypeError) as exc:
            self._tracker(state).record(
                node="validate_plan", operation="policy_gate", status="failed",
                attempt=state.get("repair_count", 0), error=f"{type(exc).__name__}: {exc}",
            )
            return {
                "status": "failed",
                "errors": [*state.get("errors", []), f"{type(exc).__name__}: {exc}"],
            }
        self._tracker(state).record(
            node="validate_plan", operation="policy_gate", status="success",
            attempt=state.get("repair_count", 0),
            output_summary={"strategy": plan.strategy},
        )
        return {"status": "running"}

    def _execute_tools(self, state: AgentGraphState) -> Dict[str, Any]:
        started = perf_counter()
        request = AgentRecommendationRequest.from_state_dict(state["request"])
        compiled = state["compiled_plan"]
        optimization = replace(request.optimization, top_k=int(compiled["top_k"]))
        phase1_request = RecommendationRequest(
            user_id=request.user_id,
            candidates=request.candidates,
            constraints=[ConstraintSpec.from_dict(entry["spec"]) for entry in compiled["constraints"]],
            objectives=[ObjectiveSpec.from_dict(entry["spec"]) for entry in compiled["objectives"]],
            optimization=optimization,
            context={
                **dict(request.context),
                "phase3_thread_id": state["thread_id"],
                "repair_count": state.get("repair_count", 0),
            },
            slate_constraints=[
                SlateConstraintSpec.from_dict(entry["spec"])
                for entry in compiled.get("slate_constraints", [])
            ],
        )
        attempt = int(state.get("repair_count", 0))
        run_id = f"phase3_{state['thread_id']}_attempt{attempt}"
        session = COPAExecutionSession(
            phase1_request,
            trace_dir=self.config.candidate_trace_dir / state["thread_id"],
            run_id=run_id,
        )
        plan = AgentExecutionPlan.model_validate(state["execution_plan"])
        fault = None
        if self.config.enable_fault_injection and attempt == 0:
            value = request.context.get("phase3_fault")
            fault = dict(value) if isinstance(value, Mapping) else None
        try:
            execution = self.executor.execute(plan, session, fault=fault)
            recommendation = execution.session.result()
        except Exception as exc:
            self._tracker(state).record(
                node="execute_tools", operation="execute", status="failed",
                attempt=attempt, duration_ms=(perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {
                "status": "failed",
                "errors": [*state.get("errors", []), f"{type(exc).__name__}: {exc}"],
                "recommendation": {},
            }
        candidate_paths = list(state.get("candidate_trace_paths", []))
        if recommendation.trace_path:
            candidate_paths.append(str(recommendation.trace_path))
        self._tracker(state).record(
            node="execute_tools", operation="execute", status="success",
            attempt=attempt, duration_ms=(perf_counter() - started) * 1000,
            input_summary={"strategy": plan.strategy, "tool_count": len(plan.steps)},
            output_summary={
                "tool_summaries": execution.tool_summaries,
                "verified": recommendation.verification.feasible,
            },
        )
        diagnostics = dict(state.get("diagnostics", {}))
        diagnostics.setdefault("tool_executions", []).append(execution.tool_summaries)
        return {
            "status": "running",
            "recommendation": recommendation.to_dict(),
            "candidate_trace_paths": candidate_paths,
            "diagnostics": diagnostics,
        }

    def _verify_result(self, state: AgentGraphState) -> Dict[str, Any]:
        recommendation = state.get("recommendation", {})
        verification = recommendation.get("verification", {})
        feasible = bool(verification.get("feasible", False))
        violations = list(verification.get("violations", []))
        self._tracker(state).record(
            node="verify_result", operation="route_verification",
            status="success" if feasible else "violation",
            attempt=state.get("repair_count", 0),
            output_summary={
                "violation_codes": [item.get("code") for item in violations],
                "violation_count": len(violations),
            },
        )
        return {"status": "success" if feasible else "verification_failed"}

    def _decide_repair(self, state: AgentGraphState) -> Dict[str, Any]:
        repair_count = int(state.get("repair_count", 0))
        violations = list(state.get("recommendation", {}).get("verification", {}).get("violations", []))
        signature = _violation_signature(violations)
        if repair_count >= self.config.max_repairs:
            return self._repair_terminal(state, "repair_exhausted", signature)
        if repair_count > 0 and signature == state.get("last_violation_signature"):
            return self._repair_terminal(state, "repeated_violation", signature)
        result = self.repair_agent.decide(
            violations,
            {
                "repair_count": repair_count,
                "max_repairs": self.config.max_repairs,
                "same_as_previous": signature == state.get("last_violation_signature"),
                "candidate_count": state.get("diagnostics", {}).get("candidate_count"),
            },
        )
        diagnostics = _append_llm_diagnostics(state.get("diagnostics", {}), "repair", result)
        if result.status == "failed" or result.payload is None:
            return {
                "status": "failed",
                "repair_route": "finalize",
                "last_violation_signature": signature,
                "diagnostics": diagnostics,
                "errors": [*state.get("errors", []), *result.errors],
            }
        decision = RepairDecision.model_validate(result.payload)
        history = [
            *state.get("repair_history", []),
            {
                "attempt": repair_count + 1,
                "decision": decision.model_dump(mode="json"),
                "violation_signature": signature,
            },
        ]
        route = {
            "recompute_objectives": "execute",
            "reexecute_selection": "execute",
            "replan": "plan",
            "request_clarification": "clarification",
            "abort": "finalize",
        }[decision.action]
        status = "failed" if decision.action == "abort" else (
            "clarification_required" if decision.action == "request_clarification" else "running"
        )
        update: Dict[str, Any] = {
            "status": status,
            "repair_route": route,
            "repair_count": repair_count + 1,
            "repair_history": history,
            "last_violation_signature": signature,
            "diagnostics": diagnostics,
        }
        if decision.action == "request_clarification":
            update["clarification"] = {
                "source": "repair",
                "questions": [decision.clarification_question],
                "issue_codes": sorted({str(item.get("code")) for item in violations}),
            }
        self._tracker(state).record(
            node="decide_repair", operation="repair_decision", status=status,
            attempt=repair_count + 1,
            input_summary={"violation_signature": signature},
            output_summary={"action": decision.action, "route": route},
        )
        return update

    def _repair_terminal(
        self, state: AgentGraphState, code: str, signature: str
    ) -> Dict[str, Any]:
        self._tracker(state).record(
            node="decide_repair", operation="repair_guard", status="failed",
            attempt=state.get("repair_count", 0),
            input_summary={"violation_signature": signature}, error=code,
        )
        return {
            "status": "failed",
            "repair_route": "finalize",
            "last_violation_signature": signature,
            "errors": [*state.get("errors", []), code],
        }

    def _finalize(self, state: AgentGraphState) -> Dict[str, Any]:
        status = state.get("status", "failed")
        if status not in {"success", "clarification_required"}:
            status = "failed"
        self._tracker(state).record(
            node="finalize", operation="finalize", status=status,
            attempt=state.get("repair_count", 0),
            output_summary={"repair_count": state.get("repair_count", 0)},
        )
        return {"status": status}

    def _result_from_thread(self, thread_id: str) -> AgentRecommendationResult:
        snapshot = self.graph.get_state(self._thread_config(thread_id))
        state = dict(snapshot.values)
        interrupts = getattr(snapshot, "interrupts", ())
        clarification = state.get("clarification")
        status = str(state.get("status", "failed"))
        if snapshot.next and "await_clarification" in snapshot.next:
            status = "clarification_required"
            if interrupts:
                clarification = dict(interrupts[0].value)
        recommendation = _recommendation_from_dict(state.get("recommendation"))
        if status != "success":
            recommendation = None
        return AgentRecommendationResult(
            status=status if status in {"success", "clarification_required", "failed"} else "failed",
            thread_id=thread_id,
            compile_result=state.get("compile_result"),
            execution_plan=(
                AgentExecutionPlan.model_validate(state["execution_plan"])
                if state.get("execution_plan") else None
            ),
            recommendation=recommendation,
            clarification=clarification or None,
            repair_history=list(state.get("repair_history", [])),
            agent_trace_path=self._trace_path(thread_id),
            candidate_trace_paths=[Path(path) for path in state.get("candidate_trace_paths", [])],
            checkpoint_path=self.config.checkpoint_path,
            diagnostics=dict(state.get("diagnostics", {})),
            errors=list(state.get("errors", [])),
        )

    @staticmethod
    def _thread_config(thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _trace_path(self, thread_id: str) -> Path:
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in thread_id)
        return self.config.trace_dir / f"agent_{safe}.jsonl"

    def _tracker(self, state: AgentGraphState) -> AgentExecutionTracker:
        return AgentExecutionTracker(state["thread_id"], self._trace_path(state["thread_id"]))


def _violation_signature(violations: list[Mapping[str, Any]]) -> str:
    counts = Counter(str(item.get("code", "unknown")) for item in violations)
    return hashlib.sha256(json.dumps(counts, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _ir_constraint_diff(previous: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare the user's semantic IR before and after an explicit clarification.

    Compiled constraints are intentionally deduplicated against system constraints.
    Comparing the IR preserves the user's confirmed change even when the executable
    plan correctly keeps only the system-owned copy.
    """

    def hard_constraints(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        ir = payload.get("ir") or {}
        return [dict(item) for item in ir.get("hard_constraints", [])]

    before, after = hard_constraints(previous), hard_constraints(current)
    return {"before": before, "after": after, "actor": "user_clarification"}


def _constraint_fingerprint(specs: Any) -> str:
    canonical = sorted(
        (dict(spec) for spec in specs),
        key=lambda spec: json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str),
    )
    payload = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scoped_constraint_payload(request_state: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        {"scope": scope, **dict(spec)}
        for scope, key in (
            ("item", "base_constraints"),
            ("slate", "base_slate_constraints"),
        )
        for spec in request_state.get(key, [])
    ]


def _append_llm_diagnostics(
    diagnostics: Mapping[str, Any], component: str, result: Any
) -> Dict[str, Any]:
    output = dict(diagnostics)
    output.setdefault("llm_calls", []).append(
        {
            "component": component,
            "status": result.status,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
            "usage": dict(result.usage),
        }
    )
    return output


def _recommendation_from_dict(payload: Optional[Mapping[str, Any]]) -> Optional[RecommendationResult]:
    if not payload:
        return None
    front = []
    for raw in payload.get("pareto_front", []):
        item = dict(raw)
        if item.get("crowding_distance") is None:
            item["crowding_distance"] = 0.0
        front.append(SlateSolution(**item))
    return RecommendationResult(
        user_id=str(payload["user_id"]),
        item_ids=list(payload.get("item_ids", [])),
        objective_values=dict(payload.get("objective_values", {})),
        pareto_front=front,
        verification=VerificationReport(**dict(payload["verification"])),
        bus_version=int(payload.get("bus_version", 0)),
        trace_path=Path(payload["trace_path"]) if payload.get("trace_path") else None,
        diagnostics=dict(payload.get("diagnostics", {})),
        status=str(payload.get("status", "success")),
    )
