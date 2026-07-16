import json
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from copa import OptimizationConfig
from copa.data import build_synthetic_case
from copa.phase2 import CompilerConfig, ConstraintCompiler
from copa.phase2.ollama import OllamaResponse
from copa.phase3 import (
    AgentCOPAPipeline,
    AgentGraphConfig,
    AgentRecommendationRequest,
    PlannerAgent,
    RepairAgent,
)
from copa.phase3.tools import AgentPlanExecutor

from test_phase3_agents import SequenceClient, planner_payload


def compiler_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "hard_constraints": [],
        "soft_objectives": [],
        "top_k": 5,
        "unresolved_requirements": [],
    }
    payload.update(overrides)
    return payload


def repair_payload(action="reexecute_selection"):
    question = "Which user constraint would you like to relax?" if action == "request_clarification" else None
    return {
        "decision_version": "1.0",
        "action": action,
        "reason_code": "test_repair",
        "clarification_question": question,
    }


def make_request(context=None):
    case = build_synthetic_case()
    return AgentRecommendationRequest(
        case.user_id,
        "推荐5个最相关的商品",
        case.candidates,
        optimization=OptimizationConfig(top_k=5, population_size=8, generations=1, seed=42),
        context={**case.context, **(context or {})},
    )


def make_pipeline(
    tmp_path,
    compiler_outputs,
    planner_outputs,
    repair_outputs,
    *,
    memory=True,
    faults=False,
    executor=None,
):
    compiler = ConstraintCompiler(
        SequenceClient(compiler_outputs),
        config=CompilerConfig(
            max_attempts=2,
            retry_backoff_seconds=0,
            audit_dir=tmp_path / "compiler",
            prompt_version="constraint_compiler_v3",
        ),
    )
    pipeline = AgentCOPAPipeline(
        compiler,
        PlannerAgent(SequenceClient(planner_outputs)),
        RepairAgent(SequenceClient(repair_outputs)),
        config=AgentGraphConfig(
            max_repairs=2,
            checkpoint_path=tmp_path / "checkpoints.sqlite",
            trace_dir=tmp_path / "agent",
            candidate_trace_dir=tmp_path / "candidate",
            enable_fault_injection=faults,
        ),
        executor=executor,
        checkpointer=MemorySaver() if memory else None,
    )
    return pipeline


def test_agent_workflow_success(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        [compiler_payload()],
        [planner_payload()],
        [repair_payload()],
    )
    result = pipeline.run(make_request(), "success-thread")
    assert result.status == "success"
    assert result.recommendation is not None
    assert result.recommendation.verification.feasible
    assert result.execution_plan.strategy == "feasible_topk"
    assert result.agent_trace_path.exists()


def test_agent_repairs_one_shot_duplicate_fault(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        [compiler_payload()],
        [planner_payload()],
        [repair_payload("reexecute_selection")],
        faults=True,
    )
    result = pipeline.run(
        make_request({"phase3_fault": {"type": "duplicate"}}),
        "repair-thread",
    )
    assert result.status == "success"
    assert result.recommendation.verification.feasible
    assert len(result.repair_history) == 1
    assert result.repair_history[0]["decision"]["action"] == "reexecute_selection"


def test_compiler_clarification_interrupt_and_resume(tmp_path):
    unresolved = compiler_payload(
        unresolved_requirements=[{
            "text": "便宜一点",
            "reason": "missing threshold",
            "clarification_question": "What maximum price?",
        }]
    )
    resolved = compiler_payload(
        hard_constraints=[{
            "attribute": "price", "operator": "<=", "value": 60, "currency": "USD"
        }]
    )
    pipeline = make_pipeline(
        tmp_path,
        [unresolved, resolved],
        [planner_payload()],
        [repair_payload()],
    )
    paused = pipeline.run(make_request(), "clarify-thread")
    assert paused.status == "clarification_required"
    assert "What maximum price?" in paused.clarification["questions"]
    resumed = pipeline.resume("clarify-thread", "最高60美元")
    assert resumed.status == "success"
    assert resumed.recommendation.verification.feasible
    changes = resumed.diagnostics["user_confirmed_constraint_changes"]
    assert changes[-1]["before"] == []
    assert changes[-1]["after"][0]["attribute"] == "price"
    assert changes[-1]["after"][0]["value"] == 60
    resumed_prompt = pipeline.compiler.client.calls[-1]["prompt"]
    assert '<phase3_user_clarifications>' in resumed_prompt
    assert '["最高60美元"]' in resumed_prompt


def test_fault_context_is_ignored_unless_explicitly_enabled(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        [compiler_payload()],
        [planner_payload()],
        [repair_payload()],
        faults=False,
    )
    secret_text = "推荐5个最相关的商品-privacy-marker"
    request = make_request({"phase3_fault": {"type": "duplicate"}})
    request = type(request)(
        request.user_id,
        secret_text,
        request.candidates,
        request.domain,
        request.base_constraints,
        request.optimization,
        request.context,
    )
    result = pipeline.run(request, "fault-disabled-thread")
    assert result.status == "success"
    assert result.repair_history == []
    trace = result.agent_trace_path.read_text(encoding="utf-8")
    assert secret_text not in trace
    event_ids = [json.loads(line)["event_id"] for line in trace.splitlines()]
    assert len(event_ids) == len(set(event_ids))


def test_system_constraints_remain_system_owned_and_fingerprinted(tmp_path):
    case = build_synthetic_case()
    request = replace(make_request(), base_constraints=case.constraints)
    pipeline = make_pipeline(
        tmp_path,
        [compiler_payload()],
        [planner_payload()],
        [repair_payload()],
    )
    result = pipeline.run(request, "system-constraint-thread")
    assert result.status == "success"
    constraints = result.compile_result["plan"]["constraints"]
    system_constraints = [entry for entry in constraints if entry["provenance"] == "system"]
    assert [entry["spec"]["id"] for entry in system_constraints] == [
        spec.id for spec in case.constraints
    ]
    assert len(result.diagnostics["system_constraints_sha256"]) == 64


class AlwaysDuplicateExecutor(AgentPlanExecutor):
    def execute(self, plan, session, *, fault=None):
        return super().execute(plan, session, fault={"type": "duplicate"})


def test_repeated_violation_signature_stops_repair_loop(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        [compiler_payload()],
        [planner_payload()],
        [repair_payload("reexecute_selection")],
        executor=AlwaysDuplicateExecutor(),
    )
    result = pipeline.run(make_request(), "repeat-violation-thread")
    assert result.status == "failed"
    assert len(result.repair_history) == 1
    assert "repeated_violation" in result.errors


def test_sqlite_checkpoint_resumes_across_pipeline_instances(tmp_path):
    unresolved = compiler_payload(
        unresolved_requirements=[{
            "text": "便宜",
            "reason": "missing threshold",
            "clarification_question": "What maximum price?",
        }]
    )
    first = make_pipeline(
        tmp_path,
        [unresolved],
        [planner_payload()],
        [repair_payload()],
        memory=False,
    )
    paused = first.run(make_request(), "sqlite-thread")
    assert paused.status == "clarification_required"
    first.close()

    second = make_pipeline(
        tmp_path,
        [compiler_payload()],
        [planner_payload()],
        [repair_payload()],
        memory=False,
    )
    resumed = second.resume("sqlite-thread", "最高60美元")
    assert resumed.status == "success"
    assert (tmp_path / "checkpoints.sqlite").stat().st_mode & 0o777 == 0o600
    second.delete_thread("sqlite-thread")
    second.close()


def test_thread_isolation_delete_and_age_cleanup(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        [compiler_payload(), compiler_payload()],
        [planner_payload(), planner_payload()],
        [repair_payload()],
    )
    pipeline.run(make_request(), "isolated-a")
    pipeline.run(make_request(), "isolated-b")
    pipeline.delete_thread("isolated-a")
    with pytest.raises(KeyError):
        pipeline.get_state("isolated-a")
    assert pipeline.get_state("isolated-b")["status"] == "success"
    assert "isolated-b" in pipeline.cleanup(0)
    with pytest.raises(KeyError):
        pipeline.get_state("isolated-b")
