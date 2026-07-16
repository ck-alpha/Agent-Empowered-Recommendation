"""Unified CLI for COPA Phase 3 Agent workflows."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Sequence
from uuid import uuid4

from copa.core import OptimizationConfig
from copa.phase2.cli import build_compiler, load_case, load_config
from copa.phase2.ollama import OllamaConfig
from copa.phase2.pipeline import NaturalLanguageCOPAPipeline

from .evaluation import evaluate_agent, evaluate_mode_comparison, load_agent_gold
from .models import AgentRecommendationRequest
from .planner import PlannerAgent, PlannerConfig
from .repair import RepairAgent, RepairConfig
from .workflow import AgentCOPAPipeline, AgentGraphConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COPA Phase-3 checkpointed Agent workflow")
    parser.add_argument("--config", default="COPA/configs/phase3_agent.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", default=argparse.SUPPRESS)
    run.add_argument("--text", required=True)
    run.add_argument("--thread-id", default=None)
    run.add_argument("--domain", choices=["synthetic", "beauty"], default="synthetic")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", default=None)
    resume = commands.add_parser("resume")
    resume.add_argument("--config", default=argparse.SUPPRESS)
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--answer", required=True)
    resume.add_argument("--output", default=None)
    status = commands.add_parser("status")
    status.add_argument("--config", default=argparse.SUPPRESS)
    status.add_argument("--thread-id", required=True)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--config", default=argparse.SUPPRESS)
    target = cleanup.add_mutually_exclusive_group(required=True)
    target.add_argument("--thread-id")
    target.add_argument("--older-than-days", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", default=argparse.SUPPRESS)
    evaluate.add_argument("--gold", default="COPA/evaluation/agent_workflow_gold.jsonl")
    evaluate.add_argument("--output-dir", default="COPA/results/phase3_agent_eval")
    evaluate.add_argument("--case-ids", default=None)
    evaluate.add_argument("--skip-comparison", action="store_true", help="Run only Planner+Repair mode for a faster live smoke.")
    return parser.parse_args(argv)


def _ollama_config(config: MappingLike) -> OllamaConfig:
    llm = config.get("ollama", {})
    return OllamaConfig(
        base_url=str(llm.get("base_url", "http://127.0.0.1:11434")),
        model=str(llm.get("model", "qwen2.5:14b")),
        connect_timeout_seconds=float(llm.get("connect_timeout_seconds", 5)),
        read_timeout_seconds=float(llm.get("read_timeout_seconds", 180)),
        temperature=float(llm.get("temperature", 0)),
        seed=int(llm.get("seed", 42)),
        num_ctx=int(llm.get("num_ctx", 4096)),
        num_predict=int(llm.get("num_predict", 1024)),
        keep_alive=str(llm.get("keep_alive", "5m")),
    )


MappingLike = Dict[str, Any]


def build_agent_pipeline(config: MappingLike, *, evaluation: bool = False) -> AgentCOPAPipeline:
    paths = config.get("paths", {})
    agent = config.get("agent", {})
    planner_config = config.get("planner", {})
    repair_config = config.get("repair", {})
    ollama = _ollama_config(config)
    trace_dir = Path(paths.get("trace_dir", "COPA/logs/phase3"))
    compiler = build_compiler(config, trace_dir / "compiler")
    return AgentCOPAPipeline(
        compiler,
        PlannerAgent(
            ollama_config=ollama,
            config=PlannerConfig(
                max_attempts=int(planner_config.get("max_attempts", 2)),
                retry_backoff_seconds=float(planner_config.get("retry_backoff_seconds", 0.5)),
                prompt_version=str(planner_config.get("prompt_version", "planner_v1")),
            ),
        ),
        RepairAgent(
            ollama_config=ollama,
            config=RepairConfig(
                max_attempts=int(repair_config.get("max_attempts", 2)),
                retry_backoff_seconds=float(repair_config.get("retry_backoff_seconds", 0.5)),
                prompt_version=str(repair_config.get("prompt_version", "repair_v1")),
            ),
        ),
        config=AgentGraphConfig(
            max_repairs=int(agent.get("max_repairs", 2)),
            checkpoint_path=Path(paths.get("checkpoint", "COPA/checkpoints/phase3.sqlite")),
            trace_dir=trace_dir,
            candidate_trace_dir=Path(paths.get("candidate_trace_dir", "COPA/logs/phase3/candidates")),
            enable_fault_injection=bool(evaluation and config.get("evaluation", {}).get("enable_fault_injection", True)),
        ),
    )


def _write(payload: Dict[str, Any], output: str | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    pipeline = build_agent_pipeline(config, evaluation=args.command == "evaluate")
    try:
        if args.command == "run":
            case = load_case(args.domain, config, args.seed)
            optimization = config.get("optimization", {})
            request = AgentRecommendationRequest(
                case.user_id,
                args.text,
                case.candidates,
                domain=args.domain,
                base_constraints=case.constraints,
                optimization=OptimizationConfig(
                    top_k=int(optimization.get("top_k", 10)),
                    population_size=int(optimization.get("population_size", 30)),
                    generations=int(optimization.get("generations", 10)),
                    seed=args.seed,
                ),
                context=case.context,
            )
            result = pipeline.run(request, args.thread_id or uuid4().hex)
            _write(result.to_dict(), args.output)
            return 0 if result.status == "success" else (3 if result.status == "clarification_required" else 2)
        if args.command == "resume":
            result = pipeline.resume(args.thread_id, args.answer)
            _write(result.to_dict(), args.output)
            return 0 if result.status == "success" else (3 if result.status == "clarification_required" else 2)
        if args.command == "status":
            state = pipeline.get_state(args.thread_id)
            _write({
                "thread_id": args.thread_id,
                "status": state.get("status"),
                "clarification": state.get("clarification"),
                "repair_count": state.get("repair_count", 0),
                "has_plan": bool(state.get("execution_plan")),
                "has_recommendation": bool(state.get("recommendation")),
                "errors": state.get("errors", []),
            })
            return 0
        if args.command == "cleanup":
            if args.thread_id:
                pipeline.delete_thread(args.thread_id)
                deleted = [args.thread_id]
            else:
                deleted = pipeline.cleanup(args.older_than_days)
            _write({"deleted_threads": deleted, "count": len(deleted)})
            return 0
        if args.command == "evaluate":
            cases = load_agent_gold(args.gold)
            if args.case_ids:
                selected = {value.strip() for value in args.case_ids.split(",") if value.strip()}
                cases = [case for case in cases if case["id"] in selected]
                missing = selected - {case["id"] for case in cases}
                if missing:
                    raise ValueError(f"Unknown Agent gold case IDs: {sorted(missing)}")
            evaluation = config.get("evaluation", {})
            summary = evaluate_agent(
                pipeline,
                cases,
                args.output_dir,
                seed=int(evaluation.get("seed", 42)),
                candidate_count=int(config.get("dataset", {}).get("candidate_count", 30)),
            )
            if bool(evaluation.get("compare_modes", True)) and not args.skip_comparison:
                comparison_config = copy.deepcopy(config)
                comparison_config.setdefault("agent", {})["max_repairs"] = 0
                comparison_paths = comparison_config.setdefault("paths", {})
                comparison_paths["checkpoint"] = str(
                    Path(comparison_paths.get("checkpoint", "COPA/checkpoints/phase3.sqlite"))
                    .with_name("phase3_planner_only.sqlite")
                )
                comparison_paths["trace_dir"] = str(Path(args.output_dir) / "planner_only_traces")
                comparison_paths["candidate_trace_dir"] = str(Path(args.output_dir) / "planner_only_candidates")
                planner_only = build_agent_pipeline(comparison_config, evaluation=True)
                phase2_compiler = build_compiler(config, Path(args.output_dir) / "phase2_compiler")
                phase2_pipeline = NaturalLanguageCOPAPipeline(
                    phase2_compiler,
                    trace_dir=Path(args.output_dir) / "phase2_traces",
                    run_id="phase3_comparison_phase2",
                )
                try:
                    summary["mode_comparison"] = evaluate_mode_comparison(
                        phase2_pipeline,
                        planner_only,
                        cases,
                        Path(args.output_dir) / "agent_case_results.csv",
                        args.output_dir,
                        seed=int(evaluation.get("seed", 42)),
                        candidate_count=int(config.get("dataset", {}).get("candidate_count", 30)),
                    )
                finally:
                    planner_only.close()
            _write(summary, str(Path(args.output_dir) / "summary_print.json"))
            return 0
        raise AssertionError("unreachable command")
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
