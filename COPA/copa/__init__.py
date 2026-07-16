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

__version__ = "0.1.0"

__all__ = [
    "COPAPipeline",
    "CandidateRecord",
    "CandidateStateBus",
    "CandidateTracker",
    "ConstraintRegistry",
    "ConstraintSpec",
    "DeterministicVerifier",
    "ObjectiveRegistry",
    "ObjectiveSpec",
    "OptimizationConfig",
    "ParetoOptimizer",
    "RecommendationRequest",
    "RecommendationResult",
    "VerificationReport",
]
