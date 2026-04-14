"""
Agent module for DualAgent-Rec.
"""

from .base_agent import BaseAgent, Individual
from .exploitation_agent import ExploitationAgent
from .exploration_agent import ExplorationAgent
from .baseline_agents import (
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
    'InProcessingConfig',
    'WeightedSumInProcessingBaseline',
    'GreedyRerankConfig',
    'GreedyRerankingBaseline',
]
