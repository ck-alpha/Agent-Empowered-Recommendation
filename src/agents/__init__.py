"""
Agent module for DualAgent-Rec.
"""

from .base_agent import BaseAgent, Individual
from .exploitation_agent import ExploitationAgent
from .exploration_agent import ExplorationAgent
from .baseline_agents import (
    EcommercePostProcessingAgent,
    EcommercePostProcessingConfig,
    GreedyRerankConfig,
    GreedyRerankingBaseline,
    InProcessingConfig,
    WeightedSumInProcessingBaseline,
)

__all__ = [
    'BaseAgent',
    'Individual',
    'ExploitationAgent',
    'ExplorationAgent',
    'EcommercePostProcessingAgent',
    'EcommercePostProcessingConfig',
    'InProcessingConfig',
    'WeightedSumInProcessingBaseline',
    'GreedyRerankConfig',
    'GreedyRerankingBaseline',
]
