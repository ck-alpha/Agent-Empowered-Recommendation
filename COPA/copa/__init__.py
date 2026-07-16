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

__version__ = "0.2.0"

__all__ = [
    "COPAPipeline",
    "CandidateRecord",
    "CandidateStateBus",
    "CandidateTracker",
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
    "RecommendationRequest",
    "RecommendationResult",
    "VerificationReport",
]
