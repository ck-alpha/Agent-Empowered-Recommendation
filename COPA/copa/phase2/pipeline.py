"""End-to-end natural-language entrypoint over the unchanged Phase-1 pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Optional
from uuid import uuid4

from copa.core import RecommendationRequest
from copa.pipeline import COPAPipeline

from .compiler import ConstraintCompiler
from .domain import DomainSchema, get_domain_schema
from .models import (
    NaturalLanguageRecommendationRequest,
    NaturalLanguageRecommendationResult,
)


class NaturalLanguageCOPAPipeline:
    def __init__(
        self,
        compiler: ConstraintCompiler,
        *,
        domains: Optional[Mapping[str, DomainSchema]] = None,
        trace_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.compiler = compiler
        self.domains = dict(domains or {})
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.run_id = run_id or uuid4().hex

    def run(self, request: NaturalLanguageRecommendationRequest) -> NaturalLanguageRecommendationResult:
        domain = self.domains.get(request.domain) or get_domain_schema(request.domain)
        compile_result = self.compiler.compile(
            request.text,
            domain,
            request.candidates,
            request.base_constraints,
            base_slate_constraints=request.base_slate_constraints,
        )
        if not compile_result.succeeded:
            return NaturalLanguageRecommendationResult(request.user_id, compile_result, None)
        assert compile_result.plan is not None
        optimization = replace(request.optimization, top_k=compile_result.plan.top_k)
        context = {
            **dict(request.context),
            "compiler_status": compile_result.status,
            "compiler_prompt_version": self.compiler.prompt.version,
            "compiler_assumptions": compile_result.plan.assumptions,
        }
        phase1_request = RecommendationRequest(
            user_id=request.user_id,
            candidates=request.candidates,
            constraints=compile_result.plan.executable_constraints,
            objectives=compile_result.plan.executable_objectives,
            optimization=optimization,
            context=context,
            slate_constraints=compile_result.plan.executable_slate_constraints,
        )
        recommendation_trace = self.trace_dir / "recommendation" if self.trace_dir else None
        recommendation = COPAPipeline(
            trace_dir=recommendation_trace,
            run_id=f"{self.run_id}_phase1",
        ).run(phase1_request)
        return NaturalLanguageRecommendationResult(request.user_id, compile_result, recommendation)
