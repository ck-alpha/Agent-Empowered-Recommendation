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
    """场景一电商软硬约束配置。"""
    alpha_inv: float = 0.05  # 库存机会约束允许的最大缺货风险
    required_new_count: int = 2  # 推荐列表中的新品数量底线
    min_sellers: int = 3  # 推荐列表中最少不同供应商数量
    target_entropy_threshold: float = 1.5  # 品牌曝光熵最低目标
    lambda_budget: float = 1.0  # 预算软约束拉格朗日乘子
    lambda_entropy: float = 1.0  # 熵软约束拉格朗日乘子
    rho_budget: float = 1.0  # 预算软约束二次惩罚系数
    rho_entropy: float = 1.0  # 熵软约束二次惩罚系数


class EcommerceConstraintHandler:
    """
    场景一（电商环境）软硬约束处理器。

    该类独立于现有 ConstraintHandler，不改变当前 DualAgent-Rec 的实验口径。
    输入候选集可以是 pandas DataFrame，也可以是记录字典列表；内部统一转成
    DataFrame 后用 pandas/numpy 向量化计算。
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

    @staticmethod
    def _resolve_budget_info(user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]]) -> Optional[Dict[str, float]]:
        """标准化用户预算输入；None 表示本次不评估预算软约束。"""
        if user_budget_info is None:
            return None
        if isinstance(user_budget_info, pd.Series):
            data = user_budget_info.to_dict()
        elif isinstance(user_budget_info, Mapping):
            data = dict(user_budget_info)
        else:
            raise TypeError("user_budget_info must be a dict, pandas Series, or None.")

        missing = [key for key in ("target_budget", "budget_tolerance") if key not in data]
        if missing:
            raise ValueError(f"user_budget_info missing required keys: {missing}")

        target_budget = float(data["target_budget"])
        budget_tolerance = float(data["budget_tolerance"])
        if not np.isfinite(target_budget) or not np.isfinite(budget_tolerance):
            raise ValueError("target_budget and budget_tolerance must be finite numbers.")
        return {
            "target_budget": target_budget,
            "budget_tolerance": max(0.0, budget_tolerance),
        }

    def check_inventory(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        alpha_inv: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        动态库存机会约束：P(未来有货) >= 1 - alpha_inv。

        程序化转化为：保留 stockout_risk <= alpha_inv 的商品。
        """
        df = self._to_dataframe(candidate_items)
        if df.empty:
            return df

        self._require_columns(df, ["stockout_risk"], "check_inventory")
        threshold = self.config.alpha_inv if alpha_inv is None else float(alpha_inv)
        risks = pd.to_numeric(df["stockout_risk"], errors="coerce")
        return df.loc[risks <= threshold].copy()

    def check_new_item_floor(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        required_new_count: Optional[int] = None,
    ) -> bool:
        """
        新品扶持底线：sum(I(item in new_items)) >= required_new_count。
        """
        df = self._to_dataframe(candidate_items)
        if df.empty:
            return False

        self._require_columns(df, ["is_new"], "check_new_item_floor")
        required = self.config.required_new_count if required_new_count is None else int(required_new_count)
        new_count = int(df["is_new"].fillna(False).astype(bool).sum())
        return new_count >= required

    def check_seller_diversity(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        min_sellers: Optional[int] = None,
    ) -> bool:
        """
        多供应商反垄断约束：|{seller_id(item)}| >= min_sellers。
        """
        df = self._to_dataframe(candidate_items)
        if df.empty:
            return False

        self._require_columns(df, ["seller_id"], "check_seller_diversity")
        required = self.config.min_sellers if min_sellers is None else int(min_sellers)
        seller_count = int(df["seller_id"].dropna().nunique())
        return seller_count >= required

    def calc_budget_penalty(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]],
    ) -> float:
        """
        价格与预算偏离惩罚：
        mean(max(0, abs(price_filled - target_budget) - budget_tolerance))。
        """
        budget_info = self._resolve_budget_info(user_budget_info)
        if budget_info is None:
            return 0.0

        df = self._to_dataframe(candidate_items)
        if df.empty:
            return 0.0

        self._require_columns(df, ["price_filled"], "calc_budget_penalty")
        prices = pd.to_numeric(df["price_filled"], errors="coerce").to_numpy(dtype=float)
        valid_prices = prices[np.isfinite(prices)]
        if valid_prices.size == 0:
            return 0.0

        deviations = np.abs(valid_prices - budget_info["target_budget"])
        penalties = np.maximum(0.0, deviations - budget_info["budget_tolerance"])
        return float(np.mean(penalties))

    def calc_entropy_penalty(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        target_entropy_threshold: Optional[float] = None,
    ) -> float:
        """
        供给侧曝光信息熵惩罚：
        max(0, H_target - (-sum(q_v * log(q_v))))。
        """
        df = self._to_dataframe(candidate_items)
        if df.empty:
            return 0.0

        self._require_columns(df, ["brand_id"], "calc_entropy_penalty")
        threshold = (
            self.config.target_entropy_threshold
            if target_entropy_threshold is None
            else float(target_entropy_threshold)
        )

        brand_probs = df["brand_id"].dropna().value_counts(normalize=True).to_numpy(dtype=float)
        if brand_probs.size == 0:
            entropy = 0.0
        else:
            # value_counts 不会产生 0 概率；仍保留过滤以对应公式里的 0 log 0 处理。
            brand_probs = brand_probs[brand_probs > 0]
            entropy = float(-np.sum(brand_probs * np.log(brand_probs)))
        return float(max(0.0, threshold - entropy))

    def calc_augmented_lagrangian_penalty(
        self,
        budget_penalty: float,
        entropy_penalty: float,
        lambda_budget: Optional[float] = None,
        lambda_entropy: Optional[float] = None,
        rho_budget: Optional[float] = None,
        rho_entropy: Optional[float] = None,
    ) -> float:
        """
        增广拉格朗日软约束总惩罚：
        lambda * phi + rho / 2 * phi^2。
        """
        lb = self.config.lambda_budget if lambda_budget is None else float(lambda_budget)
        le = self.config.lambda_entropy if lambda_entropy is None else float(lambda_entropy)
        rb = self.config.rho_budget if rho_budget is None else float(rho_budget)
        re = self.config.rho_entropy if rho_entropy is None else float(rho_entropy)

        budget_phi = max(0.0, float(budget_penalty))
        entropy_phi = max(0.0, float(entropy_penalty))
        total = (
            lb * budget_phi
            + 0.5 * rb * budget_phi ** 2
            + le * entropy_phi
            + 0.5 * re * entropy_phi ** 2
        )
        return float(total)

    def evaluate_all(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]] = None,
        alpha_inv: Optional[float] = None,
        required_new_count: Optional[int] = None,
        min_sellers: Optional[int] = None,
        target_entropy_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        统筹评估入口：一次性返回所有硬约束检查结果和软约束惩罚。
        """
        df = self._to_dataframe(candidate_items)
        total_count = len(df)

        inventory_feasible_items = self.check_inventory(df, alpha_inv=alpha_inv)
        inventory_feasible_count = len(inventory_feasible_items)
        inventory_pass_rate = inventory_feasible_count / total_count if total_count else 0.0

        new_item_satisfied = self.check_new_item_floor(
            df,
            required_new_count=required_new_count,
        )
        seller_satisfied = self.check_seller_diversity(
            df,
            min_sellers=min_sellers,
        )

        new_item_count = (
            int(df["is_new"].fillna(False).astype(bool).sum())
            if "is_new" in df.columns and total_count
            else 0
        )
        seller_count = (
            int(df["seller_id"].dropna().nunique())
            if "seller_id" in df.columns and total_count
            else 0
        )

        budget_evaluated = user_budget_info is not None
        budget_penalty = self.calc_budget_penalty(df, user_budget_info)
        entropy_penalty = self.calc_entropy_penalty(
            df,
            target_entropy_threshold=target_entropy_threshold,
        )
        augmented_penalty = self.calc_augmented_lagrangian_penalty(
            budget_penalty=budget_penalty,
            entropy_penalty=entropy_penalty,
        )

        inventory_satisfied = total_count > 0 and inventory_feasible_count == total_count
        all_hard_satisfied = bool(
            inventory_satisfied and new_item_satisfied and seller_satisfied
        )

        return {
            "inventory_feasible_items": inventory_feasible_items,
            "inventory_feasible_count": inventory_feasible_count,
            "inventory_pass_rate": float(inventory_pass_rate),
            "inventory_satisfied": bool(inventory_satisfied),
            "new_item_count": new_item_count,
            "new_item_floor_satisfied": bool(new_item_satisfied),
            "seller_count": seller_count,
            "seller_diversity_satisfied": bool(seller_satisfied),
            "budget_evaluated": bool(budget_evaluated),
            "budget_penalty": float(budget_penalty),
            "entropy_penalty": float(entropy_penalty),
            "augmented_lagrangian_penalty": float(augmented_penalty),
            "all_hard_constraints_satisfied": all_hard_satisfied,
        }
