"""
Agent module for DualAgent-Rec.
"""

from .base_agent import BaseAgent, Individual
from .exploitation_agent import ExploitationAgent
from .exploration_agent import ExplorationAgent
from .baseline_agents import (
    EcommerceDualAgentAgent,
    EcommerceDualAgentConfig,
    EcommerceInProcessingAgent,
    EcommerceInProcessingConfig,
    EcommerceOnlineGreedyAgent,
    EcommerceOnlineGreedyConfig,
    EcommercePostProcessingAgent,
    EcommercePostProcessingConfig,
    GreedyRerankConfig,
    GreedyRerankingBaseline,
    InProcessingConfig,
    WeightedSumInProcessingBaseline,
)
from .news_baseline_agents import (
    NewsDualAgentAgent,
    NewsDualAgentConfig,
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
    'EcommerceDualAgentAgent',
    'EcommerceDualAgentConfig',
    'EcommerceInProcessingAgent',
    'EcommerceInProcessingConfig',
    'EcommerceOnlineGreedyAgent',
    'EcommerceOnlineGreedyConfig',
    'EcommercePostProcessingAgent',
    'EcommercePostProcessingConfig',
    'InProcessingConfig',
    'WeightedSumInProcessingBaseline',
    'GreedyRerankConfig',
    'GreedyRerankingBaseline',
    'NewsDualAgentAgent',
    'NewsDualAgentConfig',
    'NewsInProcessingAgent',
    'NewsInProcessingConfig',
    'NewsPostProcessingAgent',
    'NewsPostProcessingConfig',
]
