"""Optional offline-retrieval boundary for COPA experiments.

The deterministic COPA pipeline never imports RecBole.  RecBole is confined to
``recbole_backend`` and communicates with COPA through versioned parquet
artifacts loaded by :class:`PrecomputedCandidateStore`.
"""

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    CANDIDATE_COLUMNS,
    TARGET_COLUMNS,
    CandidateArtifactManifest,
    PrecomputedCandidateStore,
    sha256_file,
    validate_artifact_alignment,
)
from .candidate_analysis import (
    candidate_quality_metrics,
    controlled_hit_users,
    intervene_candidate_pool,
    oracle_candidate_pool,
)
from .evaluation import (
    FORMAL_END_TO_END_METHODS,
    RetrievalProtocolContext,
    run_artifact_evaluation,
)
from .popularity_baseline import evaluate_temporal_popularity

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CANDIDATE_COLUMNS",
    "TARGET_COLUMNS",
    "CandidateArtifactManifest",
    "PrecomputedCandidateStore",
    "FORMAL_END_TO_END_METHODS",
    "RetrievalProtocolContext",
    "candidate_quality_metrics",
    "controlled_hit_users",
    "evaluate_temporal_popularity",
    "intervene_candidate_pool",
    "oracle_candidate_pool",
    "run_artifact_evaluation",
    "sha256_file",
    "validate_artifact_alignment",
]
