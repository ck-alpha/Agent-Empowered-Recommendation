"""
Agent module for DualAgent-Rec.
"""

from .base_agent import BaseAgent, Individual
from .exploitation_agent import ExploitationAgent
from .exploration_agent import ExplorationAgent
from .baseline_agents import (
    EcommerceInProcessingAgent,
    EcommerceInProcessingConfig,
    EcommercePostProcessingAgent,
    EcommercePostProcessingConfig,
    GreedyRerankConfig,
    GreedyRerankingBaseline,
    InProcessingConfig,
    WeightedSumInProcessingBaseline,
)
from .news_baseline_agents import (
    NewsInProcessingAgent,
    NewsInProcessingConfig,
    NewsPostProcessingAgent,
    NewsPostProcessingConfig,
)

__all__ = [
    'BaseAgent',
    'Individual',
    'ExploitationAgent',
    'ExplorationAgent',
    'EcommerceInProcessingAgent',
    'EcommerceInProcessingConfig',
    'EcommercePostProcessingAgent',
    'EcommercePostProcessingConfig',
    'InProcessingConfig',
    'WeightedSumInProcessingBaseline',
    'GreedyRerankConfig',
    'GreedyRerankingBaseline',
    'NewsInProcessingAgent',
    'NewsInProcessingConfig',
    'NewsPostProcessingAgent',
    'NewsPostProcessingConfig',
]
