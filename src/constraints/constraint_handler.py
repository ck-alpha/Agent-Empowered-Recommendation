"""
Constraint Handler for DualAgent-Rec.
Handles fairness, seller coverage, and new item exposure constraints.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Mapping, Union
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


@dataclass
class EcommerceConstraintConfig:
    """场景一电商库存容量硬约束配置。"""

    capacity_col: str = "inventory_initial"


class EcommerceConstraintHandler:
    """
    场景一（电商环境）库存容量硬约束处理器。

    推荐矩阵上的唯一约束是全局库存容量：
        sum_u 1(item_i in R_u) <= inventory_initial_i

    输入可以是 pandas DataFrame，也可以是记录字典列表；内部统一转成
    DataFrame 后计算 item-level exposure 与 capacity overflow。
    """

    def __init__(self, config: Optional[EcommerceConstraintConfig] = None):
        self.config = config or EcommerceConstraintConfig()

    @staticmethod
    def _to_dataframe(candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """统一候选集输入格式，并 copy，避免修改调用方对象。"""
        if isinstance(candidate_items, pd.DataFrame):
            return candidate_items.copy()
        if isinstance(candidate_items, list):
            return pd.DataFrame(candidate_items).copy()
        raise TypeError("candidate_items must be a pandas DataFrame or a list of record dictionaries.")

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: List[str], method_name: str) -> None:
        """检查必要字段，字段缺失时给出可定位的错误信息。"""
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{method_name} requires columns: {missing}")

    def capacity_diagnostics(
        self,
        recommendations: Union[pd.DataFrame, List[Dict[str, Any]]],
        capacity_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        评估全局库存容量约束。

        capacity_satisfaction_rate 以被曝光商品为分母，衡量有多少被曝光商品
        没有超过库存容量。未曝光商品不进入该分母。
        """
        df = self._to_dataframe(recommendations)
        if df.empty:
            return {
                "capacity_satisfied": True,
                "capacity_violation_total": 0.0,
                "over_capacity_item_count": 0,
                "max_capacity_overflow": 0.0,
                "capacity_satisfaction_rate": 1.0,
                "mean_item_utilization": 0.0,
                "exposure_by_item": pd.DataFrame(
                    columns=["item_id", "capacity_max", "exposure_count", "overflow"]
                ),
            }

        col = self.config.capacity_col if capacity_col is None else str(capacity_col)
        self._require_columns(df, ["item_id", col], "capacity_diagnostics")

        work = df[["item_id", col]].copy()
        work["item_id"] = work["item_id"].astype(str)
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)

        exposure = work["item_id"].value_counts().rename("exposure_count").astype(float)
        capacity = (
            work.drop_duplicates("item_id", keep="first")
            .set_index("item_id")[col]
            .rename("capacity_max")
            .astype(float)
        )
        joined = capacity.to_frame().join(exposure, how="right")
        joined["capacity_max"] = joined["capacity_max"].fillna(0.0).clip(lower=0.0)
        joined["exposure_count"] = joined["exposure_count"].fillna(0.0)
        joined["overflow"] = (joined["exposure_count"] - joined["capacity_max"]).clip(lower=0.0)

        safe_capacity = joined["capacity_max"].replace(0.0, np.nan)
        utilization = (joined["exposure_count"] / safe_capacity).replace([np.inf, -np.inf], np.nan)
        finite_utilization = utilization.dropna()

        violation_total = float(joined["overflow"].sum())
        over_count = int((joined["overflow"] > 0.0).sum())
        exposed_count = max(1, len(joined))
        satisfaction_rate = float((joined["overflow"] <= 0.0).sum() / exposed_count)

        return {
            "capacity_satisfied": bool(violation_total <= 0.0),
            "capacity_violation_total": violation_total,
            "over_capacity_item_count": over_count,
            "max_capacity_overflow": float(joined["overflow"].max()) if len(joined) else 0.0,
            "capacity_satisfaction_rate": satisfaction_rate,
            "mean_item_utilization": float(finite_utilization.mean()) if not finite_utilization.empty else 0.0,
            "exposure_by_item": joined.reset_index()[["item_id", "capacity_max", "exposure_count", "overflow"]],
        }

    def evaluate_all(
        self,
        recommendations: Union[pd.DataFrame, List[Dict[str, Any]]],
        capacity_col: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        统筹评估入口：一次性返回库存容量硬约束检查结果。

        **_ 用于兼容旧调用签名；场景一新实验不再评估预算、新品、
        供应商或 stockout 机会约束。
        """
        diagnostics = self.capacity_diagnostics(recommendations, capacity_col=capacity_col)
        compact = dict(diagnostics)
        compact.pop("exposure_by_item", None)
        compact["all_hard_constraints_satisfied"] = bool(compact.get("capacity_satisfied", False))
        return compact
