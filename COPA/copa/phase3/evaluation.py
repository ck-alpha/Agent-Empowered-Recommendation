"""Phase 3 Agent workflow evaluation and scenario reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from copa.core import OptimizationConfig, RecommendationRequest
from copa.data import build_synthetic_case
from copa.metrics import ndcg_at_k, recall_at_k
from copa.phase2 import NaturalLanguageCOPAPipeline, NaturalLanguageRecommendationRequest
from copa.session import COPAExecutionSession

from .models import AgentExecutionPlan, AgentRecommendationRequest
from .tools import AgentPlanExecutor
from .workflow import AgentCOPAPipeline


AGENT_SCENARIOS = {"seen", "compositional", "repair", "unseen"}


def load_agent_gold(path: Path | str) -> list[Dict[str, Any]]:
    cases: list[Dict[str, Any]] = []
    required = {
        "id", "language", "scenario", "text", "expected_status",
        "expected_strategy", "fault", "expected_repair_action",
    }
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = required - set(case)
            if missing:
                raise ValueError(f"Agent gold line {line_number} missing fields: {sorted(missing)}")
            if case["scenario"] not in AGENT_SCENARIOS:
                raise ValueError(f"Agent gold case {case['id']} has invalid scenario {case['scenario']!r}")
            cases.append(case)
    return cases


def evaluate_agent(
    pipeline: AgentCOPAPipeline,
    cases: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    *,
    seed: int = 42,
    candidate_count: int = 30,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("Agent evaluation requires at least one case")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_synthetic_case(seed=seed, candidate_count=candidate_count)
    records: list[Dict[str, Any]] = []
    for case in cases:
        context = {**fixture.context, "evaluation_case_id": case["id"]}
        if case.get("fault"):
            context["phase3_fault"] = {"type": case["fault"]}
        request = AgentRecommendationRequest(
            user_id=f"agent_eval_{case['id']}",
            text=str(case["text"]),
            candidates=fixture.candidates,
            domain="synthetic",
            optimization=OptimizationConfig(
                top_k=10,
                population_size=12,
                generations=2,
                seed=seed,
            ),
            context=context,
        )
        started = perf_counter()
        errors: list[str] = []
        thread_id = f"agent-gold-{case['id']}-{seed}"
        try:
            _delete_if_present(pipeline, thread_id)
            result = pipeline.run(request, thread_id=thread_id)
        except Exception as exc:
            result = None
            errors.append(f"{type(exc).__name__}: {exc}")
        latency = perf_counter() - started
        strategy = result.execution_plan.strategy if result and result.execution_plan else None
        repair_actions = [
            entry["decision"]["action"] for entry in (result.repair_history if result else [])
        ]
        expected_repair = case.get("expected_repair_action")
        status = result.status if result else "failed"
        recommendation = result.recommendation if result else None
        effectiveness = _effectiveness(
            recommendation.item_ids if recommendation else [],
            fixture.candidates,
            fixture.relevant_items,
        )
        diagnostics = result.diagnostics if result else {}
        llm_calls = diagnostics.get("llm_calls", [])
        records.append(
            {
                "id": case["id"],
                "language": case["language"],
                "scenario": case["scenario"],
                "status": status,
                "expected_status": case["expected_status"],
                "status_correct": status == case["expected_status"],
                "strategy": strategy,
                "expected_strategy": case.get("expected_strategy"),
                "strategy_correct": strategy == case.get("expected_strategy"),
                "verified_feasible": bool(recommendation and recommendation.verification.feasible),
                **effectiveness,
                "repair_triggered": bool(repair_actions),
                "repair_actions": json.dumps(repair_actions, ensure_ascii=False),
                "expected_repair_action": expected_repair,
                "repair_action_correct": (
                    expected_repair in repair_actions if expected_repair else not repair_actions
                ),
                "repair_success": bool(expected_repair and status == case["expected_status"]),
                "repair_count": len(repair_actions),
                "tool_call_count": sum(len(batch) for batch in diagnostics.get("tool_executions", [])),
                "llm_call_count": len(llm_calls),
                "prompt_tokens": sum(int(call.get("usage", {}).get("prompt_eval_count", 0)) for call in llm_calls),
                "output_tokens": sum(int(call.get("usage", {}).get("eval_count", 0)) for call in llm_calls),
                "latency_seconds": latency,
                "errors": "; ".join([*(result.errors if result else []), *errors]),
            }
        )
    frame = pd.DataFrame(records)
    scenario_metrics = {
        str(scenario): {
            "case_count": int(len(group)),
            "status_accuracy": float(group["status_correct"].mean()),
            "strategy_accuracy": float(group["strategy_correct"].mean()),
            "repair_action_accuracy": float(group["repair_action_correct"].mean()),
            "verified_feasible_rate": float(group["verified_feasible"].mean()),
        }
        for scenario, group in frame.groupby("scenario", sort=True)
    }
    repair_rows = frame[frame["expected_repair_action"].notna()]
    success_rows = frame[frame["status"] == "success"]
    summary = {
        "case_count": int(len(frame)),
        "agent_status_accuracy": float(frame["status_correct"].mean()),
        "planner_strategy_accuracy": float(frame["strategy_correct"].mean()),
        "verified_feasible_rate": float(frame["verified_feasible"].mean()),
        "mean_recall_at_k": float(success_rows["recall_at_k"].mean()) if len(success_rows) else None,
        "mean_ndcg_at_k": float(success_rows["ndcg_at_k"].mean()) if len(success_rows) else None,
        "mean_diversity": float(success_rows["diversity"].mean()) if len(success_rows) else None,
        "mean_novelty": float(success_rows["novelty"].mean()) if len(success_rows) else None,
        "repair_trigger_rate": float(frame["repair_triggered"].mean()),
        "repair_action_accuracy": float(frame["repair_action_correct"].mean()),
        "repair_success_rate": float(repair_rows["repair_success"].mean()) if len(repair_rows) else None,
        "mean_tool_calls": float(frame["tool_call_count"].mean()),
        "mean_llm_calls": float(frame["llm_call_count"].mean()),
        "latency_p50_seconds": float(frame["latency_seconds"].quantile(0.5)),
        "latency_p95_seconds": float(frame["latency_seconds"].quantile(0.95)),
        "prompt_tokens": int(frame["prompt_tokens"].sum()),
        "output_tokens": int(frame["output_tokens"].sum()),
        "scenario_metrics": scenario_metrics,
    }
    frame.to_csv(output_dir / "agent_case_results.csv", index=False)
    (output_dir / "agent_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_mode_comparison(
    phase2_pipeline: NaturalLanguageCOPAPipeline,
    planner_only_pipeline: AgentCOPAPipeline,
    cases: Sequence[Mapping[str, Any]],
    full_results_path: Path | str,
    output_dir: Path | str,
    *,
    seed: int = 42,
    candidate_count: int = 30,
) -> Dict[str, Any]:
    """Compare fixed Phase 2, Planner-only, and full Planner+Repair modes."""

    fixture = build_synthetic_case(seed=seed, candidate_count=candidate_count)
    rows: list[Dict[str, Any]] = []
    full = pd.read_csv(full_results_path)
    for record in full.to_dict("records"):
        rows.append({
            "id": record["id"],
            "scenario": record["scenario"],
            "mode": "planner_repair",
            "status": record["status"],
            "verified_feasible": _as_bool(record["verified_feasible"]),
            "recall_at_k": float(record["recall_at_k"]),
            "ndcg_at_k": float(record["ndcg_at_k"]),
            "diversity": float(record["diversity"]),
            "novelty": float(record["novelty"]),
            "tool_call_count": int(record["tool_call_count"]),
            "repair_count": int(record["repair_count"]),
            "llm_call_count": int(record["llm_call_count"]),
            "prompt_tokens": int(record["prompt_tokens"]),
            "output_tokens": int(record["output_tokens"]),
            "latency_seconds": float(record["latency_seconds"]),
        })
    for case in cases:
        base_context = {**fixture.context, "evaluation_case_id": case["id"]}
        phase2_request = NaturalLanguageRecommendationRequest(
            user_id=f"phase2_compare_{case['id']}",
            text=str(case["text"]),
            candidates=fixture.candidates,
            domain="synthetic",
            optimization=OptimizationConfig(top_k=10, population_size=12, generations=2, seed=seed),
            context=base_context,
        )
        started = perf_counter()
        phase2_result = phase2_pipeline.run(phase2_request)
        latency = perf_counter() - started
        phase2_recommendation = phase2_result.recommendation
        if case.get("fault") and phase2_result.compile_result.succeeded:
            compiled = phase2_result.compile_result.plan
            strategy = "pareto" if len(compiled.executable_objectives) > 1 else "feasible_topk"
            selection = "optimize_pareto" if strategy == "pareto" else "select_feasible_topk"
            fixed_plan = AgentExecutionPlan.model_validate(
                {
                    "plan_version": "1.0",
                    "strategy": strategy,
                    "steps": [
                        {"tool": "apply_constraints"},
                        {"tool": "compute_objectives"},
                        {"tool": selection},
                        {"tool": "verify"},
                    ],
                    "assumptions": [],
                    "unresolved_requirements": [],
                }
            )
            fixed_session = COPAExecutionSession(
                RecommendationRequest(
                    user_id=phase2_request.user_id,
                    candidates=phase2_request.candidates,
                    constraints=compiled.executable_constraints,
                    objectives=compiled.executable_objectives,
                    optimization=replace(
                        phase2_request.optimization, top_k=compiled.top_k
                    ),
                    context=phase2_request.context,
                    slate_constraints=compiled.executable_slate_constraints,
                )
            )
            AgentPlanExecutor().execute(
                fixed_plan,
                fixed_session,
                fault={"type": case["fault"]},
            )
            phase2_recommendation = fixed_session.result()
        rows.append(_comparison_row(
            case,
            "phase2_fixed",
            (
                phase2_recommendation.status
                if phase2_recommendation is not None
                else phase2_result.compile_result.status
            ),
            phase2_recommendation,
            fixture,
            latency,
            tool_call_count=0,
            repair_count=0,
            llm_call_count=int(phase2_result.compile_result.attempts),
            usage=phase2_result.compile_result.usage,
        ))

        planner_context = dict(base_context)
        if case.get("fault"):
            planner_context["phase3_fault"] = {"type": case["fault"]}
        planner_request = AgentRecommendationRequest(
            user_id=f"planner_only_{case['id']}",
            text=str(case["text"]),
            candidates=fixture.candidates,
            domain="synthetic",
            optimization=OptimizationConfig(top_k=10, population_size=12, generations=2, seed=seed),
            context=planner_context,
        )
        started = perf_counter()
        planner_thread_id = f"agent-gold-planner-only-{case['id']}-{seed}"
        _delete_if_present(planner_only_pipeline, planner_thread_id)
        planner_result = planner_only_pipeline.run(
            planner_request,
            thread_id=planner_thread_id,
        )
        planner_diagnostics = planner_result.diagnostics
        planner_calls = planner_diagnostics.get("llm_calls", [])
        rows.append(_comparison_row(
            case,
            "planner_only",
            planner_result.status,
            planner_result.recommendation,
            fixture,
            perf_counter() - started,
            tool_call_count=sum(
                len(batch) for batch in planner_diagnostics.get("tool_executions", [])
            ),
            repair_count=len(planner_result.repair_history),
            llm_call_count=len(planner_calls),
            usage={
                "prompt_eval_count": sum(
                    int(call.get("usage", {}).get("prompt_eval_count", 0))
                    for call in planner_calls
                ),
                "eval_count": sum(
                    int(call.get("usage", {}).get("eval_count", 0))
                    for call in planner_calls
                ),
            },
        ))
    frame = pd.DataFrame(rows)
    summary: Dict[str, Any] = {"case_count": len(cases), "modes": {}}
    for mode, group in frame.groupby("mode", sort=True):
        successful = group[group["status"] == "success"]
        summary["modes"][str(mode)] = {
            "completion_rate": float((group["status"] == "success").mean()),
            "verified_feasible_rate": float(group["verified_feasible"].mean()),
            "mean_recall_at_k": float(successful["recall_at_k"].mean()) if len(successful) else None,
            "mean_ndcg_at_k": float(successful["ndcg_at_k"].mean()) if len(successful) else None,
            "mean_diversity": float(successful["diversity"].mean()) if len(successful) else None,
            "mean_novelty": float(successful["novelty"].mean()) if len(successful) else None,
            "mean_tool_calls": float(group["tool_call_count"].mean()),
            "mean_repair_loops": float(group["repair_count"].mean()),
            "mean_llm_calls": float(group["llm_call_count"].mean()),
            "prompt_tokens": int(group["prompt_tokens"].sum()),
            "output_tokens": int(group["output_tokens"].sum()),
            "latency_p50_seconds": float(group["latency_seconds"].quantile(0.5)),
            "latency_p95_seconds": float(group["latency_seconds"].quantile(0.95)),
        }
    output_dir = Path(output_dir)
    frame.to_csv(output_dir / "mode_comparison.csv", index=False)
    (output_dir / "mode_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _comparison_row(
    case,
    mode,
    status,
    recommendation,
    fixture,
    latency,
    *,
    tool_call_count,
    repair_count,
    llm_call_count,
    usage,
):
    values = _effectiveness(
        recommendation.item_ids if recommendation else [],
        fixture.candidates,
        fixture.relevant_items,
    )
    return {
        "id": case["id"],
        "scenario": case["scenario"],
        "mode": mode,
        "status": status,
        "verified_feasible": bool(recommendation and recommendation.verification.feasible),
        **values,
        "tool_call_count": int(tool_call_count),
        "repair_count": int(repair_count),
        "llm_call_count": int(llm_call_count),
        "prompt_tokens": int(usage.get("prompt_eval_count", 0)),
        "output_tokens": int(usage.get("eval_count", 0)),
        "latency_seconds": latency,
    }


def _delete_if_present(pipeline: AgentCOPAPipeline, thread_id: str) -> None:
    try:
        pipeline.get_state(thread_id)
    except KeyError:
        return
    pipeline.delete_thread(thread_id)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return bool(value)


def _effectiveness(item_ids, candidates, relevant_items):
    metadata = {candidate.item_id: candidate.metadata for candidate in candidates}
    selected = [metadata[item_id] for item_id in item_ids if item_id in metadata]
    if len(selected) < 2:
        diversity = 0.0
    else:
        pairs = 0
        different = 0
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                pairs += 1
                different += selected[left].get("brand_id") != selected[right].get("brand_id")
        diversity = different / pairs if pairs else 0.0
    novelty = (
        sum(1.0 - float(item.get("popularity", 0.5)) for item in selected) / len(selected)
        if selected else 0.0
    )
    return {
        "recall_at_k": recall_at_k(item_ids, relevant_items, len(item_ids)),
        "ndcg_at_k": ndcg_at_k(item_ids, relevant_items, len(item_ids)),
        "diversity": float(diversity),
        "novelty": float(novelty),
    }
