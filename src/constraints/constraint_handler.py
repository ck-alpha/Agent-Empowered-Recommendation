"""
Constraint Handler for DualAgent-Rec.
Handles fairness, seller coverage, and new item exposure constraints.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class ConstraintConfig:
    """Configuration for constraints."""
    fairness_threshold: float = 0.7  # Category exposure fairness (Gini coefficient)
    seller_coverage_threshold: float = 0.3  # Minimum proportion of unique sellers
    new_item_threshold: float = 0.1  # Minimum proportion of new items
    epsilon_initial: float = 1.0  # Initial constraint relaxation
    epsilon_decay: float = 0.95  # Decay rate per generation


class ConstraintHandler:
    """
    Handles constraint calculations and adaptive relaxation.

    Constraints:
    - g1: Category fairness (Gini coefficient of category distribution)
    - g2: Seller coverage (proportion of unique sellers)
    - g3: New item exposure (proportion of items < 30 days old)
    """

    def __init__(self, config: Optional[ConstraintConfig] = None):
        """
        Initialize constraint handler.

        Args:
            config: Constraint configuration
        """
        self.config = config or ConstraintConfig()
        self.epsilon = self.config.epsilon_initial
        self.generation = 0

        # Track initial violation for self-calibrating epsilon
        self.initial_violation = None

    def calculate_violations(
        self,
        recommended_items: List[str],
        item_features: Dict[str, Dict]
    ) -> List[float]:
        """
        Calculate constraint violations.

        Returns:
            List of violations [fairness_violation, seller_violation, new_item_violation]
            Positive values indicate constraint violation.
        """
        violations = []

        # g1: 类别公平性约束（Gini）
        fairness_violation = self._calculate_fairness_violation(
            recommended_items, item_features
        )
        violations.append(fairness_violation)

        # g2: 卖家覆盖率约束
        seller_violation = self._calculate_seller_coverage_violation(
            recommended_items, item_features
        )
        violations.append(seller_violation)

        # g3: 新品曝光约束
        new_item_violation = self._calculate_new_item_violation(
            recommended_items, item_features
        )
        violations.append(new_item_violation)

        return violations

    def _calculate_fairness_violation(
        self,
        items: List[str],
        item_features: Dict[str, Dict]
    ) -> float:
        """
        Calculate category fairness violation using Gini coefficient.

        Lower Gini = more equal distribution (fair)
        Target: Gini < (1 - fairness_threshold)
        """
        if not items:
            return 1.0

        # Get category distribution (support both 'category' and 'main_category')
        categories = []
        for item_id in items:
            item_info = item_features.get(item_id, {})
            cat = item_info.get('category') or item_info.get('main_category', 'Unknown')
            categories.append(cat)
        category_counts = Counter(categories)

        if len(category_counts) <= 1:
            return 0.0  # Only one category, maximally concentrated but not unfair

        # Calculate Gini coefficient
        counts = np.array(list(category_counts.values()), dtype=float)
        n = len(counts)
        counts_sorted = np.sort(counts)
        cumsum = np.cumsum(counts_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1) * counts_sorted))) / (n * np.sum(counts)) - (n + 1) / n

        # 约束形态：gini <= (1 - threshold)。
        # 通过 epsilon 放宽目标阈值，形成“先宽后严”的可行域收缩过程。
        target_gini = 1 - self.config.fairness_threshold
        relaxed_target = target_gini + self.epsilon * (1 - target_gini)

        violation = gini - relaxed_target
        return max(0, violation)

    def _calculate_seller_coverage_violation(
        self,
        items: List[str],
        item_features: Dict[str, Dict]
    ) -> float:
        """
        Calculate seller coverage violation.

        Target: unique_sellers / total_items >= threshold
        """
        if not items:
            return self.config.seller_coverage_threshold

        sellers = set()
        for item_id in items:
            seller = item_features.get(item_id, {}).get('seller_id', item_id)
            sellers.add(seller)

        coverage = len(sellers) / len(items)

        # 覆盖率约束：coverage >= threshold。
        # 早期 epsilon 较大时，阈值被放松，帮助种群先探索可行边界附近区域。
        relaxed_threshold = self.config.seller_coverage_threshold * (1 - self.epsilon)

        violation = relaxed_threshold - coverage
        return max(0, violation)

    def _calculate_new_item_violation(
        self,
        items: List[str],
        item_features: Dict[str, Dict]
    ) -> float:
        """
        Calculate new item exposure violation.

        Target: proportion of new items >= threshold
        """
        if not items:
            return self.config.new_item_threshold

        new_items = 0
        for item_id in items:
            is_new = item_features.get(item_id, {}).get('is_new', False)
            if is_new:
                new_items += 1

        new_ratio = new_items / len(items)

        # 新品约束：new_ratio >= threshold，同样采用 epsilon 松弛。
        relaxed_threshold = self.config.new_item_threshold * (1 - self.epsilon)

        violation = relaxed_threshold - new_ratio
        return max(0, violation)

    def update_epsilon(self, feasibility_rate: float = None) -> None:
        """
        Update epsilon for adaptive constraint relaxation.

        Args:
            feasibility_rate: Current proportion of feasible solutions
        """
        self.generation += 1

        # 基础衰减：epsilon 随代数递减，逐步收紧约束。
        self.epsilon = max(0.0, self.epsilon * self.config.epsilon_decay)

        # 自适应校正：可行率过低则短暂回调 epsilon，避免陷入“无可行解”死区。
        if feasibility_rate is not None:
            if feasibility_rate < 0.1:
                # Too few feasible solutions, relax constraints
                self.epsilon = min(1.0, self.epsilon * 1.1)
            elif feasibility_rate > 0.9:
                # Most solutions feasible, tighten constraints
                self.epsilon = max(0.0, self.epsilon * 0.9)

    def calibrate_epsilon(self, initial_violations: List[float]) -> None:
        """
        Self-calibrating epsilon based on initial constraint violations.

        Formula: cp = (-log(VAR0) - 6) / log(0.5)
        """
        if self.initial_violation is None:
            self.initial_violation = max(initial_violations) if initial_violations else 1.0

            if self.initial_violation > 0:
                # 自校准思想：根据初始违反程度估计更合适的衰减速率，减少手工调参成本。
                import math
                try:
                    cp = (-math.log(self.initial_violation) - 6) / math.log(0.5)
                    cp = max(0.8, min(0.99, cp))  # Bound between 0.8 and 0.99
                    self.config.epsilon_decay = cp
                except (ValueError, ZeroDivisionError):
                    pass  # Keep default decay

    def get_relaxed_thresholds(self) -> Dict[str, float]:
        """Get current relaxed constraint thresholds."""
        return {
            'fairness': self.config.fairness_threshold * (1 - self.epsilon),
            'seller_coverage': self.config.seller_coverage_threshold * (1 - self.epsilon),
            'new_item': self.config.new_item_threshold * (1 - self.epsilon),
            'epsilon': self.epsilon,
        }


class AdaptiveConstraintHandler(ConstraintHandler):
    """
    Adaptive constraint handler with dynamic threshold adjustment
    based on optimization progress and LLM guidance.
    """

    def __init__(
        self,
        config: Optional[ConstraintConfig] = None,
        adaptation_rate: float = 0.1
    ):
        super().__init__(config)
        self.adaptation_rate = adaptation_rate
        self.history: List[Dict[str, float]] = []

    def adapt_thresholds(
        self,
        current_performance: Dict[str, float],
        llm_suggestion: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Adapt constraint thresholds based on performance and LLM suggestion.

        Args:
            current_performance: Current objective scores
            llm_suggestion: Optional LLM-suggested threshold adjustments
        """
        self.history.append(current_performance)

        if len(self.history) < 5:
            return  # Need enough history

        # 当近期性能几乎无改进时，轻微放宽阈值以帮助跳出停滞区。
        recent_performance = [h.get('avg_score', 0) for h in self.history[-5:]]
        if max(recent_performance) - min(recent_performance) < 0.01:
            # Likely stuck, relax constraints slightly
            self.config.fairness_threshold *= (1 - self.adaptation_rate)
            self.config.seller_coverage_threshold *= (1 - self.adaptation_rate)

        # 若外部（如协调器/LLM）给出阈值调节建议，则在此统一落地。
        if llm_suggestion:
            if 'fairness_adjustment' in llm_suggestion:
                self.config.fairness_threshold *= (1 + llm_suggestion['fairness_adjustment'])
            if 'seller_coverage_adjustment' in llm_suggestion:
                self.config.seller_coverage_threshold *= (1 + llm_suggestion['seller_coverage_adjustment'])
            if 'new_item_adjustment' in llm_suggestion:
                self.config.new_item_threshold *= (1 + llm_suggestion['new_item_adjustment'])

        # Ensure thresholds stay in valid range
        self.config.fairness_threshold = max(0.3, min(0.95, self.config.fairness_threshold))
        self.config.seller_coverage_threshold = max(0.1, min(0.8, self.config.seller_coverage_threshold))
        self.config.new_item_threshold = max(0.05, min(0.5, self.config.new_item_threshold))
