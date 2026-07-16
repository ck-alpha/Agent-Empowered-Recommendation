"""CLI for compiling, recommending, and evaluating COPA Phase 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from copa.core import OptimizationConfig
from copa.data import AllBeautyAdapter, build_synthetic_case

from .compiler import CompilerConfig, ConstraintCompiler
from .domain import get_domain_schema
from .evaluation import SCENARIOS, evaluate_compiler, load_gold_cases
from .models import NaturalLanguageRecommendationRequest
from .ollama import OllamaConfig
from .pipeline import NaturalLanguageCOPAPipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COPA Phase-2 Qwen constraint compiler")
    parser.add_argument("--config", default="COPA/configs/phase2_qwen.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="Compile natural language into Constraint IR")
    compile_parser.add_argument("--text", required=True)
    compile_parser.add_argument("--domain", choices=["synthetic", "beauty"], default="synthetic")
    compile_parser.add_argument("--output", default=None)
    recommend_parser = subparsers.add_parser("recommend", help="Compile and execute deterministic COPA")
    recommend_parser.add_argument("--text", required=True)
    recommend_parser.add_argument("--domain", choices=["synthetic", "beauty"], default="synthetic")
    recommend_parser.add_argument("--seed", type=int, default=42)
    recommend_parser.add_argument("--output", default=None)
    evaluate_parser = subparsers.add_parser("evaluate", help="Run the bilingual compiler gold set")
    evaluate_parser.add_argument("--gold", default="COPA/evaluation/constraint_compiler_gold.jsonl")
    evaluate_parser.add_argument("--output-dir", default="COPA/results/phase2_compiler_eval")
    evaluate_parser.add_argument("--case-ids", default=None, help="Optional comma-separated gold case IDs for live smoke runs.")
    evaluate_parser.add_argument("--scenarios", default=None, help="Optional comma-separated seen,compositional,unseen filter.")
    return parser.parse_args(argv)


def load_config(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_compiler(config: Dict[str, Any], audit_dir: Path | None = None) -> ConstraintCompiler:
    llm = config.get("ollama", {})
    compiler = config.get("compiler", {})
    return ConstraintCompiler(
        ollama_config=OllamaConfig(
            base_url=str(llm.get("base_url", "http://127.0.0.1:11434")),
            model=str(llm.get("model", "qwen2.5:14b")),
            connect_timeout_seconds=float(llm.get("connect_timeout_seconds", 5)),
            read_timeout_seconds=float(llm.get("read_timeout_seconds", 180)),
            temperature=float(llm.get("temperature", 0)),
            seed=int(llm.get("seed", 42)),
            num_ctx=int(llm.get("num_ctx", 4096)),
            num_predict=int(llm.get("num_predict", 1024)),
            keep_alive=str(llm.get("keep_alive", "5m")),
        ),
        config=CompilerConfig(
            max_attempts=int(compiler.get("max_attempts", 2)),
            retry_backoff_seconds=float(compiler.get("retry_backoff_seconds", 0.5)),
            audit_dir=audit_dir,
            prompt_version=str(compiler.get("prompt_version", "constraint_compiler_v2")),
        ),
    )


def load_case(domain: str, config: Dict[str, Any], seed: int = 42):
    if domain == "synthetic":
        return build_synthetic_case(seed=seed, candidate_count=int(config.get("dataset", {}).get("candidate_count", 30)))
    dataset = config.get("dataset", {})
    adapter = AllBeautyAdapter(dataset.get("processed_dir", "data/processed"), dataset.get("prefix", "beauty_scenario1"))
    return next(adapter.iter_user_cases(num_users=1, candidate_k=int(dataset.get("candidate_k", 100)), seed=seed))


def _write(payload: Dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.command == "compile":
        case = load_case(args.domain, config)
        compiler = build_compiler(config, Path(config.get("audit_dir", "COPA/logs/phase2")))
        result = compiler.compile(args.text, get_domain_schema(args.domain), case.candidates)
        _write(result.to_dict(), args.output)
        return 0 if result.status != "failed" else 2
    if args.command == "recommend":
        case = load_case(args.domain, config, args.seed)
        audit_dir = Path(config.get("audit_dir", "COPA/logs/phase2"))
        compiler = build_compiler(config, audit_dir)
        optimization_payload = config.get("optimization", {})
        optimization = OptimizationConfig(
            top_k=int(optimization_payload.get("top_k", 10)),
            population_size=int(optimization_payload.get("population_size", 30)),
            generations=int(optimization_payload.get("generations", 10)),
            seed=args.seed,
        )
        request = NaturalLanguageRecommendationRequest(
            user_id=case.user_id,
            text=args.text,
            candidates=case.candidates,
            domain=args.domain,
            base_constraints=case.constraints,
            optimization=optimization,
            context=case.context,
        )
        result = NaturalLanguageCOPAPipeline(
            compiler,
            trace_dir=audit_dir,
            run_id=f"phase2_{args.domain}_{args.seed}",
        ).run(request)
        _write(result.to_dict(), args.output)
        return 0 if result.compile_result.status != "failed" else 2
    if args.command == "evaluate":
        output_dir = Path(args.output_dir)
        evaluation_config = config.get("evaluation", {})
        case = build_synthetic_case(
            seed=int(evaluation_config.get("candidate_seed", 42)),
            candidate_count=int(config.get("dataset", {}).get("candidate_count", 30)),
        )
        compiler = build_compiler(config, output_dir / "audits")
        gold_cases = load_gold_cases(args.gold)
        if args.case_ids:
            selected_ids = {case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()}
            gold_cases = [gold_case for gold_case in gold_cases if gold_case["id"] in selected_ids]
            missing = selected_ids - {gold_case["id"] for gold_case in gold_cases}
            if missing:
                raise ValueError(f"Unknown gold case IDs: {sorted(missing)}")
        if args.scenarios:
            selected_scenarios = {value.strip() for value in args.scenarios.split(",") if value.strip()}
            invalid = selected_scenarios - SCENARIOS
            if invalid:
                raise ValueError(f"Unknown evaluation scenarios: {sorted(invalid)}")
            gold_cases = [gold_case for gold_case in gold_cases if gold_case["scenario"] in selected_scenarios]
        summary = evaluate_compiler(
            compiler,
            gold_cases,
            get_domain_schema("synthetic"),
            case.candidates,
            output_dir,
        )
        _write(summary, str(output_dir / "summary_print.json"))
        return 0
    raise AssertionError("unreachable CLI command")


if __name__ == "__main__":
    raise SystemExit(main())
