"""COPA: Constraint-Oriented Pareto Optimization Agent for Recommendation."""

from .constraints import ConstraintRegistry
from .core import (
    CandidateRecord,
    CandidateStateBus,
    CandidateTracker,
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationConfig,
    RecommendationRequest,
    RecommendationResult,
    VerificationReport,
)
from .core.verifier import DeterministicVerifier
from .objectives import ObjectiveRegistry
from .optimization import ParetoOptimizer
from .pipeline import COPAPipeline
from .session import COPAExecutionSession
from .phase2 import (
    CompileResult,
    ConstraintCompiler,
    ConstraintIR,
    DomainSchemaRegistry,
    NaturalLanguageCOPAPipeline,
    NaturalLanguageRecommendationRequest,
    NaturalLanguageRecommendationResult,
    OllamaConfig,
)
from .phase3 import (
    AgentCOPAPipeline,
    AgentExecutionPlan,
    AgentGraphConfig,
    AgentRecommendationRequest,
    AgentRecommendationResult,
    AgentToolRegistry,
    PlannerAgent,
    RepairAgent,
    RepairDecision,
)

__version__ = "0.3.0"

__all__ = [
    "COPAPipeline",
    "COPAExecutionSession",
    "CandidateRecord",
    "CandidateStateBus",
    "CandidateTracker",
    "AgentCOPAPipeline",
    "AgentExecutionPlan",
    "AgentGraphConfig",
    "AgentRecommendationRequest",
    "AgentRecommendationResult",
    "AgentToolRegistry",
    "CompileResult",
    "ConstraintCompiler",
    "ConstraintIR",
    "DomainSchemaRegistry",
    "ConstraintRegistry",
    "ConstraintSpec",
    "DeterministicVerifier",
    "ObjectiveRegistry",
    "ObjectiveSpec",
    "NaturalLanguageCOPAPipeline",
    "NaturalLanguageRecommendationRequest",
    "NaturalLanguageRecommendationResult",
    "OllamaConfig",
    "OptimizationConfig",
    "ParetoOptimizer",
    "PlannerAgent",
    "RecommendationRequest",
    "RecommendationResult",
    "RepairAgent",
    "RepairDecision",
    "VerificationReport",
]
