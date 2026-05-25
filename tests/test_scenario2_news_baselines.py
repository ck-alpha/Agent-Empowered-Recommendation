from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import (  # noqa: E402
    NewsInProcessingAgent,
    NewsInProcessingConfig,
    NewsPostProcessingAgent,
    NewsPostProcessingConfig,
)
from constraints import NewsConstraintConfig, NewsConstraintHandler  # noqa: E402


def test_topic_entropy_penalty_and_alm_are_natural_log() -> None:
    slate = pd.DataFrame(
        [
            {"news_id": "a1", "category": "a"},
            {"news_id": "a2", "category": "a"},
            {"news_id": "b1", "category": "b"},
            {"news_id": "b2", "category": "b"},
        ]
    )
    handler = NewsConstraintHandler(
        NewsConstraintConfig(target_topic_entropy=1.0, lambda_diversity=2.0, rho_diversity=4.0)
    )

    diagnostics = handler.evaluate_all(slate)
    expected_entropy = float(np.log(2.0))
    expected_penalty = 1.0 - expected_entropy
    expected_alm = 2.0 * expected_penalty + 0.5 * 4.0 * expected_penalty**2

    assert diagnostics["topic_entropy"] == expected_entropy
    assert diagnostics["topic_coverage"] == 2
    assert diagnostics["diversity_penalty"] == expected_penalty
    assert diagnostics["augmented_lagrangian_penalty"] == expected_alm
    assert diagnostics["entropy_target_satisfied"] is False


def test_news_handler_requires_only_topic_column_for_diagnostics() -> None:
    slate = pd.DataFrame(
        [
            {"news_id": "a1", "category": "a", "base_score": 1.0},
            {"news_id": "b1", "category": "b", "base_score": 0.9},
        ]
    )
    handler = NewsConstraintHandler(NewsConstraintConfig(target_topic_entropy=0.5))

    diagnostics = handler.evaluate_all(slate)

    assert diagnostics["topic_entropy"] > 0.0
    assert sorted(diagnostics) == [
        "augmented_lagrangian_penalty",
        "diversity_penalty",
        "entropy_target_satisfied",
        "target_topic_entropy",
        "topic_coverage",
        "topic_distribution",
        "topic_entropy",
    ]


def test_postprocessing_greedy_uses_entropy_gain_to_introduce_topics() -> None:
    candidates = pd.DataFrame(
        [
            {"news_id": "a1", "category": "a", "base_score": 1.00},
            {"news_id": "a2", "category": "a", "base_score": 0.99},
            {"news_id": "a3", "category": "a", "base_score": 0.98},
            {"news_id": "b1", "category": "b", "base_score": 0.20},
            {"news_id": "c1", "category": "c", "base_score": 0.19},
        ]
    )
    agent = NewsPostProcessingAgent(
        NewsPostProcessingConfig(top_k=3, target_topic_entropy=1.1, lambda_diversity=5.0, rho_diversity=1.0)
    )

    result = agent.recommend(user_id="u1", candidate_items=candidates)
    recs = result["recommendations"]

    assert recs["category"].nunique() >= 2
    assert result["diagnostics"]["num_swaps"] > 0
    assert result["diagnostics"]["final_constraints"]["topic_entropy"] > 0.0


def test_inprocessing_exact_slate_level_optimization_selects_global_topic_counts() -> None:
    candidates = pd.DataFrame(
        [
            {"news_id": "a1", "category": "a", "base_score": 1.00},
            {"news_id": "a2", "category": "a", "base_score": 0.99},
            {"news_id": "a3", "category": "a", "base_score": 0.98},
            {"news_id": "a4", "category": "a", "base_score": 0.97},
            {"news_id": "b1", "category": "b", "base_score": 0.50},
            {"news_id": "b2", "category": "b", "base_score": 0.49},
        ]
    )
    agent = NewsInProcessingAgent(
        NewsInProcessingConfig(top_k=4, target_topic_entropy=float(np.log(2.0)), lambda_diversity=10.0)
    )

    result = agent.recommend(user_id="u1", candidate_items=candidates)
    item_ids = result["item_ids"]

    assert item_ids == ["a1", "a2", "b1", "b2"]
    assert result["diagnostics"]["best_topic_counts"] == {"a": 2, "b": 2}
    assert result["diagnostics"]["final_constraints"]["entropy_target_satisfied"] is True


def test_news_exports_are_available_from_public_packages() -> None:
    assert NewsConstraintHandler is not None
    assert NewsConstraintConfig is not None
    assert NewsPostProcessingAgent is not None
    assert NewsPostProcessingConfig is not None
    assert NewsInProcessingAgent is not None
    assert NewsInProcessingConfig is not None
