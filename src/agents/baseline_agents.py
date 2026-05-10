"""
Traditional baselines for DualAgent-Rec comparisons.

Includes:
1) In-processing weighted-sum baseline with soft penalty.
2) Post-processing greedy re-ranking baseline.
"""

from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base_agent import Individual
from constraints import (
    ConstraintConfig,
    ConstraintHandler,
    EcommerceConstraintConfig,
    EcommerceConstraintHandler,
)
from evaluation import MultiObjectiveMetrics, ObjectivesCalculator, RecommendationMetrics


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class InProcessingConfig:
    """Configuration for weighted-sum in-processing baseline."""

    population_size: int = 100
    max_generations: int = 50
    recommendation_size: int = 10
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    fairness_threshold: float = 0.6
    seller_coverage_threshold: float = 0.2
    new_item_threshold: float = 0.1


class WeightedSumInProcessingBaseline:
    """
    Traditional single-scalar in-processing baseline.

    Score = (w1*f1 + w2*f2 + w3*f3) - lambda*(g1 + g2 + g3)
    """

    def __init__(self, config: Optional[InProcessingConfig] = None):
        self.config = config or InProcessingConfig()
        self.objectives_calculator = ObjectivesCalculator()
        self.constraint_handler = ConstraintHandler(
            ConstraintConfig(
                fairness_threshold=self.config.fairness_threshold,
                seller_coverage_threshold=self.config.seller_coverage_threshold,
                new_item_threshold=self.config.new_item_threshold,
            )
        )

    def _initialize_population(self, candidate_items: List[str]) -> List[Individual]:
        """Initialize random recommendation lists."""
        population: List[Individual] = []
        k = min(self.config.recommendation_size, len(candidate_items))
        if k == 0:
            return population

        for _ in range(self.config.population_size):
            items = random.sample(candidate_items, k)
            population.append(Individual(item_ids=items))
        return population

    def _scalar_score(self, individual: Individual, weights: Tuple[float, float, float], penalty_lambda: float) -> float:
        """Compute scalar fitness for weighted-sum optimization."""
        objective_term = (
            weights[0] * individual.scores[0]
            + weights[1] * individual.scores[1]
            + weights[2] * individual.scores[2]
        )
        penalty_term = penalty_lambda * float(np.sum(np.maximum(0.0, individual.constraint_violations)))
        return float(objective_term - penalty_term)

    def _evaluate_population(
        self,
        population: List[Individual],
        user_history: List[Dict[str, Any]],
        item_features: Dict[str, Dict[str, Any]],
        weights: Tuple[float, float, float],
        penalty_lambda: float,
    ) -> List[Tuple[Individual, float]]:
        """Evaluate objective/constraint values and scalar score."""
        scored: List[Tuple[Individual, float]] = []
        for individual in population:
            scores = self.objectives_calculator.calculate(individual.item_ids, user_history, item_features)
            violations = self.constraint_handler.calculate_violations(individual.item_ids, item_features)
            individual.scores = np.array(scores, dtype=float)
            individual.constraint_violations = np.array(violations, dtype=float)
            scalar = self._scalar_score(individual, weights, penalty_lambda)
            individual.fitness = scalar
            scored.append((individual, scalar))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _mutate(self, items: List[str], candidate_items: List[str]) -> List[str]:
        """Random replacement mutation with de-duplication."""
        mutated = items.copy()
        for i in range(len(mutated)):
            if random.random() < self.config.mutation_rate:
                available = [it for it in candidate_items if it not in mutated]
                if available:
                    mutated[i] = random.choice(available)
        deduped = list(dict.fromkeys(mutated))
        while len(deduped) < self.config.recommendation_size:
            available = [it for it in candidate_items if it not in deduped]
            if not available:
                break
            deduped.append(random.choice(available))
        return deduped[: self.config.recommendation_size]

    def _reproduce(self, elite: List[Individual], candidate_items: List[str], population_size: int) -> List[Individual]:
        """Crossover + mutation to generate next generation."""
        if not elite:
            return self._initialize_population(candidate_items)

        next_population: List[Individual] = [Individual(item_ids=ind.item_ids.copy()) for ind in elite]
        k = self.config.recommendation_size
        while len(next_population) < population_size:
            p1, p2 = random.sample(elite, 2) if len(elite) >= 2 else (elite[0], elite[0])
            child_items: List[str] = []
            for idx in range(k):
                if random.random() < self.config.crossover_rate:
                    source = p1 if random.random() < 0.5 else p2
                    if idx < len(source.item_ids):
                        child_items.append(source.item_ids[idx])
                    else:
                        child_items.append(random.choice(candidate_items))
                else:
                    child_items.append(random.choice(candidate_items))

            child_items = list(dict.fromkeys(child_items))
            while len(child_items) < k:
                available = [it for it in candidate_items if it not in child_items]
                if not available:
                    break
                child_items.append(random.choice(available))

            child_items = self._mutate(child_items[:k], candidate_items)
            next_population.append(Individual(item_ids=child_items))
        return next_population[:population_size]

    def optimize(
        self,
        candidate_items: List[str],
        user_history: List[Dict[str, Any]],
        test_ground_truth: List[Dict[str, Any]],
        item_features: Dict[str, Dict[str, Any]],
        weights: Tuple[float, float, float],
        penalty_lambda: float,
    ) -> Dict[str, Any]:
        """Run weighted-sum optimization and return aligned metric dict."""
        if not candidate_items:
            return {}

        population = self._initialize_population(candidate_items)
        if not population:
            return {}

        best_individual: Optional[Individual] = None
        best_fitness = float("-inf")

        for _ in range(self.config.max_generations):
            scored = self._evaluate_population(
                population=population,
                user_history=user_history,
                item_features=item_features,
                weights=weights,
                penalty_lambda=penalty_lambda,
            )
            if scored and scored[0][1] > best_fitness:
                best_individual = Individual(item_ids=scored[0][0].item_ids.copy())
                best_individual.scores = scored[0][0].scores.copy()
                best_individual.constraint_violations = scored[0][0].constraint_violations.copy()
                best_individual.fitness = scored[0][0].fitness
                best_fitness = scored[0][1]

            top_k = max(2, self.config.population_size // 5)
            elite = [pair[0] for pair in scored[:top_k]]
            population = self._reproduce(elite, candidate_items, self.config.population_size)

            # [修复 In-processing 集成]
            # Keep the same violation scale by reusing the existing constraint handler
            # with generation-wise epsilon updates.
            feasible_rate = float(np.mean([1.0 if pair[0].is_feasible else 0.0 for pair in scored])) if scored else 0.0
            self.constraint_handler.update_epsilon(feasible_rate)

        if best_individual is None:
            return {}

        offline = RecommendationMetrics.evaluate_with_ground_truth(
            recommended=best_individual.item_ids,
            test_ground_truth=test_ground_truth,
            k=10,
        )

        return {
            "hypervolume": MultiObjectiveMetrics.hypervolume([best_individual.scores], np.array([1.0, 1.0, 1.0])),
            "spacing": 0.0,
            "pareto_size": 1,
            "avg_accuracy": float(best_individual.scores[0]),
            "avg_diversity": float(best_individual.scores[1]),
            "avg_novelty": float(best_individual.scores[2]),
            "real_ndcg@10": offline.get("ndcg@k", 0.0),
            "real_hr@10": offline.get("hr@k", 0.0),
            "feasibility_rate": 1.0 if best_individual.is_feasible else 0.0,
            "coordinator_summary": "baseline_inprocessing_weighted_sum",
            "total_generations": self.config.max_generations,
            "weighted_sum_score": float(best_individual.fitness),
            "weights": {"w1": weights[0], "w2": weights[1], "w3": weights[2]},
            "penalty_lambda": float(penalty_lambda),
            "recommended_items": best_individual.item_ids.copy(),
        }


@dataclass
class GreedyRerankConfig:
    """Configuration for post-processing greedy reranking baseline."""

    candidate_pool_size: int = 50
    recommendation_size: int = 10
    fairness_threshold: float = 0.6
    seller_coverage_threshold: float = 0.2
    new_item_threshold: float = 0.1


class GreedyRerankingBaseline:
    """
    Post-processing baseline:
    1) Build Top-N by f1 proxy relevance.
    2) Initialize Top-K.
    3) Iteratively replace the lowest-f1 item to reduce total violations.
    """

    def __init__(self, config: Optional[GreedyRerankConfig] = None):
        self.config = config or GreedyRerankConfig()
        self.objectives_calculator = ObjectivesCalculator()
        self.constraint_handler = ConstraintHandler(
            ConstraintConfig(
                fairness_threshold=self.config.fairness_threshold,
                seller_coverage_threshold=self.config.seller_coverage_threshold,
                new_item_threshold=self.config.new_item_threshold,
            )
        )
        # Greedy repair stage targets strict final constraint satisfaction.
        self.constraint_handler.epsilon = 0.0

    def _item_relevance(
        self,
        item_id: str,
        user_history: List[Dict[str, Any]],
        item_features: Dict[str, Dict[str, Any]],
    ) -> float:
        """Approximate per-item f1 using relevance objective on a singleton list."""
        relevance = self.objectives_calculator.calculate([item_id], user_history, item_features)[0]
        return float(relevance)

    def _evaluate_list(
        self,
        items: List[str],
        user_history: List[Dict[str, Any]],
        test_ground_truth: List[Dict[str, Any]],
        item_features: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate list-level objective and constraint metrics."""
        scores = self.objectives_calculator.calculate(items, user_history, item_features)
        violations = self.constraint_handler.calculate_violations(items, item_features)
        offline = RecommendationMetrics.evaluate_with_ground_truth(
            recommended=items,
            test_ground_truth=test_ground_truth,
            k=10,
        )
        total_violation = float(np.sum(np.maximum(0.0, np.array(violations, dtype=float))))
        return {
            "items": items,
            "scores": scores,
            "violations": violations,
            "total_violation": total_violation,
            "real_ndcg@10": offline.get("ndcg@k", 0.0),
            "real_hr@10": offline.get("hr@k", 0.0),
            "is_feasible": total_violation <= 0.0,
        }

    def optimize(
        self,
        candidate_items: List[str],
        user_history: List[Dict[str, Any]],
        test_ground_truth: List[Dict[str, Any]],
        item_features: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run greedy reranking and return aligned metric dict."""
        if not candidate_items:
            return {}

        relevance_by_item = {
            item_id: self._item_relevance(item_id, user_history, item_features)
            for item_id in candidate_items
        }
        ranked = sorted(candidate_items, key=lambda x: relevance_by_item.get(x, 0.0), reverse=True)

        n = min(self.config.candidate_pool_size, len(ranked))
        k = min(self.config.recommendation_size, n)
        candidate_pool = ranked[:n]
        current_topk = candidate_pool[:k]
        backup_pool = candidate_pool[k:]

        initial_eval = self._evaluate_list(
            items=current_topk.copy(),
            user_history=user_history,
            test_ground_truth=test_ground_truth,
            item_features=item_features,
        )

        iterations = 0
        while backup_pool:
            current_eval = self._evaluate_list(
                items=current_topk,
                user_history=user_history,
                test_ground_truth=test_ground_truth,
                item_features=item_features,
            )
            if current_eval["is_feasible"]:
                break

            # Remove the lowest-f1 item in current top-k
            remove_idx = min(range(len(current_topk)), key=lambda idx: relevance_by_item.get(current_topk[idx], 0.0))
            removed_item = current_topk[remove_idx]

            best_candidate_idx = -1
            best_candidate_eval: Optional[Dict[str, Any]] = None
            best_improvement = float("-inf")

            for cand_idx, candidate in enumerate(backup_pool):
                trial = current_topk.copy()
                trial[remove_idx] = candidate
                trial_eval = self._evaluate_list(
                    items=trial,
                    user_history=user_history,
                    test_ground_truth=test_ground_truth,
                    item_features=item_features,
                )
                improvement = current_eval["total_violation"] - trial_eval["total_violation"]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate_idx = cand_idx
                    best_candidate_eval = trial_eval
                elif (
                    improvement == best_improvement
                    and best_candidate_eval is not None
                    and relevance_by_item.get(candidate, 0.0)
                    > relevance_by_item.get(backup_pool[best_candidate_idx], 0.0)
                ):
                    # Tie break by higher f1 (as required)
                    best_candidate_idx = cand_idx
                    best_candidate_eval = trial_eval

            # No further violation reduction => exhausted useful optimization
            if best_candidate_idx < 0 or best_candidate_eval is None or best_improvement <= 0:
                break

            selected = backup_pool.pop(best_candidate_idx)
            current_topk[remove_idx] = selected
            backup_pool.append(removed_item)
            iterations += 1

        final_eval = self._evaluate_list(
            items=current_topk,
            user_history=user_history,
            test_ground_truth=test_ground_truth,
            item_features=item_features,
        )

        final_scores = np.array(final_eval["scores"], dtype=float)
        return {
            "hypervolume": MultiObjectiveMetrics.hypervolume([final_scores], np.array([1.0, 1.0, 1.0])),
            "spacing": 0.0,
            "pareto_size": 1,
            "avg_accuracy": float(final_scores[0]),
            "avg_diversity": float(final_scores[1]),
            "avg_novelty": float(final_scores[2]),
            "real_ndcg@10": final_eval["real_ndcg@10"],
            "real_hr@10": final_eval["real_hr@10"],
            "feasibility_rate": 1.0 if final_eval["is_feasible"] else 0.0,
            "coordinator_summary": "baseline_postprocessing_greedy_reranking",
            "total_generations": iterations,
            "initial_real_ndcg@10": initial_eval["real_ndcg@10"],
            "initial_real_hr@10": initial_eval["real_hr@10"],
            "initial_total_violation": float(initial_eval["total_violation"]),
            "final_total_violation": float(final_eval["total_violation"]),
            "candidate_pool_size": int(n),
            "recommendation_size": int(k),
            "num_rerank_iterations": int(iterations),
            "recommended_items": current_topk.copy(),
        }


@dataclass
class EcommercePostProcessingConfig:
    """Configuration for scenario-1 e-commerce post-processing baseline."""

    top_k: int = 10
    alpha_inv: float = 0.05
    required_new_count: int = 2
    min_sellers: int = 3
    target_entropy_threshold: float = 1.5
    lambda_budget: float = 1.0
    rho_budget: float = 1.0
    base_score_col: str = "base_score"


class EcommercePostProcessingAgent:
    """
    场景一电商后处理基线。

    策略顺序：
    1) 库存机会约束作为绝对硬红线，先过滤高缺货风险商品。
    2) 用 item-level 预算偏离构造 ALM 惩罚，对传统召回分数做重排。
    3) 对 Top-K 列表做新品底线和供应商多样性的贪心替换修复。
    """

    def __init__(self, config: Optional[EcommercePostProcessingConfig] = None):
        self.config = config or EcommercePostProcessingConfig()
        self.constraint_handler = EcommerceConstraintHandler(
            EcommerceConstraintConfig(
                alpha_inv=self.config.alpha_inv,
                required_new_count=self.config.required_new_count,
                min_sellers=self.config.min_sellers,
                target_entropy_threshold=self.config.target_entropy_threshold,
                lambda_budget=self.config.lambda_budget,
                rho_budget=self.config.rho_budget,
            )
        )

    @staticmethod
    def _to_dataframe(candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """统一候选输入格式，copy 后处理，避免污染调用方数据。"""
        if isinstance(candidate_items, pd.DataFrame):
            return candidate_items.copy()
        if isinstance(candidate_items, list):
            return pd.DataFrame(candidate_items).copy()
        raise TypeError("candidate_items must be a pandas DataFrame or a list of dictionaries.")

    @staticmethod
    def _resolve_budget_info(user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]]) -> Optional[Dict[str, float]]:
        """标准化用户预算；None 表示本轮不做预算惩罚。"""
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

    def _prepare_candidates(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """准备候选集：补 base_score、计算 item-level ALM 预算惩罚和 final_score。"""
        df = self._to_dataframe(candidate_items)
        diagnostics = {
            "base_score_missing": False,
            "budget_evaluated": user_budget_info is not None,
        }

        if "item_id" not in df.columns:
            raise ValueError("candidate_items requires an item_id column.")

        score_col = self.config.base_score_col
        if score_col not in df.columns:
            df[score_col] = 0.0
            diagnostics["base_score_missing"] = True

        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)
        budget_info = self._resolve_budget_info(user_budget_info)

        if budget_info is None or df.empty:
            df["item_budget_penalty"] = 0.0
            df["item_budget_alm_penalty"] = 0.0
        else:
            if "price_filled" not in df.columns:
                raise ValueError("candidate_items requires a price_filled column when user_budget_info is provided.")
            prices = pd.to_numeric(df["price_filled"], errors="coerce")
            deviations = (prices - budget_info["target_budget"]).abs()
            item_penalty = (deviations - budget_info["budget_tolerance"]).clip(lower=0.0).fillna(0.0)
            df["item_budget_penalty"] = item_penalty.astype(float)
            # 增广拉格朗日项：lambda * phi + rho/2 * phi^2。这里按商品逐项扣分，排序才会真的变化。
            df["item_budget_alm_penalty"] = (
                self.config.lambda_budget * df["item_budget_penalty"]
                + 0.5 * self.config.rho_budget * df["item_budget_penalty"] ** 2
            ).astype(float)

        df["final_score"] = df[score_col] - df["item_budget_alm_penalty"]
        return df, diagnostics

    @staticmethod
    def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
        """按后处理得分降序排序，并用 item_id 稳定打破平分。"""
        if df.empty:
            return df.copy()
        return df.sort_values(["final_score", "item_id"], ascending=[False, True]).reset_index(drop=True)

    @staticmethod
    def _lowest_score_index(df: pd.DataFrame) -> Optional[int]:
        """返回 final_score 最低项的行位置。"""
        if df.empty:
            return None
        return int(df["final_score"].astype(float).idxmin())

    def _candidate_pool(self, ranked: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
        """返回尚未进入当前推荐列表的候选池。"""
        selected_ids = set(current["item_id"].tolist()) if "item_id" in current.columns else set()
        return ranked.loc[~ranked["item_id"].isin(selected_ids)].copy()

    @staticmethod
    def _compact_constraint_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """压缩 Handler 约束结果，避免 diagnostics 内嵌 DataFrame 影响日志/JSON 追溯。"""
        compact = dict(result)
        compact.pop("inventory_feasible_items", None)
        return compact

    def _swap_rows(
        self,
        current: pd.DataFrame,
        replacement: pd.Series,
        remove_idx: int,
    ) -> pd.DataFrame:
        """用 replacement 替换 current 中指定行，保持 DataFrame 结构稳定。"""
        updated = current.copy()
        updated.loc[remove_idx, :] = replacement
        return self._sort_candidates(updated)

    def _repair_new_items(
        self,
        current: pd.DataFrame,
        ranked: pd.DataFrame,
        swap_log: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """新品底线贪心修复：用高分新品替换低分非新品。"""
        max_iterations = max(0, len(ranked) - len(current))
        iterations = 0

        while (
            len(current) > 0
            and not self.constraint_handler.check_new_item_floor(current, self.config.required_new_count)
            and iterations <= max_iterations
        ):
            before_count = int(current["is_new"].fillna(False).astype(bool).sum())
            pool = self._candidate_pool(ranked, current)
            replacements = pool.loc[pool["is_new"].fillna(False).astype(bool)]
            if replacements.empty:
                break

            removable = current.loc[~current["is_new"].fillna(False).astype(bool)]
            if removable.empty:
                break

            remove_idx = self._lowest_score_index(removable)
            if remove_idx is None:
                break

            replacement = replacements.iloc[0]
            removed = current.loc[remove_idx].copy()
            current = self._swap_rows(current, replacement, remove_idx)
            after_count = int(current["is_new"].fillna(False).astype(bool).sum())
            swap_log.append(
                {
                    "reason": "new_item_floor",
                    "removed_item_id": removed.get("item_id"),
                    "added_item_id": replacement.get("item_id"),
                    "before_count": before_count,
                    "after_count": after_count,
                }
            )
            iterations += 1

        return current

    def _repair_seller_diversity(
        self,
        current: pd.DataFrame,
        ranked: pd.DataFrame,
        swap_log: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """供应商多样性贪心修复：优先引入当前列表未覆盖的新 seller。"""
        max_iterations = max(0, len(ranked) - len(current))
        iterations = 0

        while (
            len(current) > 0
            and not self.constraint_handler.check_seller_diversity(current, self.config.min_sellers)
            and iterations <= max_iterations
        ):
            before_count = int(current["seller_id"].dropna().nunique())
            current_sellers = set(current["seller_id"].dropna().tolist())
            pool = self._candidate_pool(ranked, current)
            replacements = pool.loc[~pool["seller_id"].isin(current_sellers)]
            if replacements.empty:
                break

            seller_counts = current["seller_id"].value_counts(dropna=True)
            duplicate_sellers = set(seller_counts[seller_counts > 1].index.tolist())
            removable = current.loc[current["seller_id"].isin(duplicate_sellers)]
            if removable.empty:
                removable = current

            remove_idx = self._lowest_score_index(removable)
            if remove_idx is None:
                break

            replacement = replacements.iloc[0]
            removed = current.loc[remove_idx].copy()
            current = self._swap_rows(current, replacement, remove_idx)
            after_count = int(current["seller_id"].dropna().nunique())
            swap_log.append(
                {
                    "reason": "seller_diversity",
                    "removed_item_id": removed.get("item_id"),
                    "added_item_id": replacement.get("item_id"),
                    "before_count": before_count,
                    "after_count": after_count,
                }
            )
            iterations += 1

        return current

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        执行基于 ALM 重排与贪心替换的后处理推荐。

        返回 recommendations DataFrame、item_ids，以及可追溯 diagnostics。
        """
        requested_k = self.config.top_k if top_k is None else int(top_k)
        requested_k = max(0, requested_k)

        prepared, prep_diag = self._prepare_candidates(candidate_items, user_budget_info)
        total_candidates = len(prepared)
        diagnostics: Dict[str, Any] = {
            "user_id": user_id,
            "requested_top_k": requested_k,
            "input_candidate_count": total_candidates,
            "swap_log": [],
            **prep_diag,
        }

        if requested_k == 0 or prepared.empty:
            empty = prepared.head(0).copy()
            diagnostics.update(
                {
                    "inventory_feasible_count": 0,
                    "inventory_filtered_count": 0,
                    "candidate_shortage": requested_k > 0,
                    "initial_constraints": {},
                    "final_constraints": {},
                    "fully_repaired": False,
                }
            )
            return {"recommendations": empty, "item_ids": [], "diagnostics": diagnostics}

        # Step 1: 库存机会约束是绝对硬红线，不满足就直接剔除。
        inventory_feasible = self.constraint_handler.check_inventory(
            prepared,
            alpha_inv=self.config.alpha_inv,
        )
        diagnostics["inventory_feasible_count"] = int(len(inventory_feasible))
        diagnostics["inventory_filtered_count"] = int(total_candidates - len(inventory_feasible))

        if inventory_feasible.empty:
            diagnostics.update(
                {
                    "candidate_shortage": True,
                    "initial_constraints": {},
                    "final_constraints": {},
                    "fully_repaired": False,
                }
            )
            return {
                "recommendations": inventory_feasible.copy(),
                "item_ids": [],
                "diagnostics": diagnostics,
            }

        # Step 2: 用 item-level 预算 ALM 惩罚重排。熵是列表级性质，留到诊断和修复后评估。
        ranked = self._sort_candidates(inventory_feasible)
        k = min(requested_k, len(ranked))
        current = ranked.head(k).copy()
        diagnostics["candidate_shortage"] = bool(len(ranked) < requested_k)
        diagnostics["effective_top_k"] = int(k)
        initial_constraints = self.constraint_handler.evaluate_all(
            current,
            user_budget_info=user_budget_info,
            alpha_inv=self.config.alpha_inv,
            required_new_count=self.config.required_new_count,
            min_sellers=self.config.min_sellers,
            target_entropy_threshold=self.config.target_entropy_threshold,
        )
        diagnostics["initial_constraints"] = self._compact_constraint_result(initial_constraints)

        # Step 3: 对列表级硬约束做贪心 swap 修复，先补新品，再补供应商多样性。
        swap_log: List[Dict[str, Any]] = diagnostics["swap_log"]
        current = self._repair_new_items(current, ranked, swap_log)
        current = self._repair_seller_diversity(current, ranked, swap_log)
        current = self._sort_candidates(current).head(k).reset_index(drop=True)

        final_constraints = self.constraint_handler.evaluate_all(
            current,
            user_budget_info=user_budget_info,
            alpha_inv=self.config.alpha_inv,
            required_new_count=self.config.required_new_count,
            min_sellers=self.config.min_sellers,
            target_entropy_threshold=self.config.target_entropy_threshold,
        )
        diagnostics["final_constraints"] = self._compact_constraint_result(final_constraints)
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["num_swaps"] = int(len(swap_log))
        diagnostics["final_new_item_count"] = int(final_constraints.get("new_item_count", 0))
        diagnostics["final_seller_count"] = int(final_constraints.get("seller_count", 0))
        diagnostics["final_budget_penalty"] = float(final_constraints.get("budget_penalty", 0.0))
        diagnostics["final_entropy_penalty"] = float(final_constraints.get("entropy_penalty", 0.0))
        diagnostics["final_augmented_lagrangian_penalty"] = float(
            final_constraints.get("augmented_lagrangian_penalty", 0.0)
        )

        return {
            "recommendations": current,
            "item_ids": current["item_id"].tolist(),
            "diagnostics": diagnostics,
        }


@dataclass
class EcommerceInProcessingConfig:
    """Configuration for scenario-1 e-commerce in-processing baseline."""

    top_k: int = 10
    alpha_inv: float = 0.05
    required_new_count: int = 2
    min_sellers: int = 3
    target_entropy_threshold: float = 1.5
    lambda_budget: float = 1.0
    lambda_entropy: float = 1.0
    rho_budget: float = 1.0
    rho_entropy: float = 1.0
    base_score_col: str = "base_score"
    population_size: int = 20
    max_generations: int = 10
    elite_size: int = 5
    mutation_rate: float = 0.25
    hard_violation_weight: float = 10.0
    random_seed: int = RANDOM_SEED


class EcommerceInProcessingAgent:
    """
    场景一电商中处理基线。

    该 baseline 在召回候选集内直接搜索 Top-K 列表，并把场景一硬约束与 ALM
    软约束放进适应度函数；它不调用后处理修复逻辑。
    """

    def __init__(self, config: Optional[EcommerceInProcessingConfig] = None):
        self.config = config or EcommerceInProcessingConfig()
        self.constraint_handler = EcommerceConstraintHandler(
            EcommerceConstraintConfig(
                alpha_inv=self.config.alpha_inv,
                required_new_count=self.config.required_new_count,
                min_sellers=self.config.min_sellers,
                target_entropy_threshold=self.config.target_entropy_threshold,
                lambda_budget=self.config.lambda_budget,
                lambda_entropy=self.config.lambda_entropy,
                rho_budget=self.config.rho_budget,
                rho_entropy=self.config.rho_entropy,
            )
        )

    @staticmethod
    def _to_dataframe(candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(candidate_items, pd.DataFrame):
            return candidate_items.copy()
        if isinstance(candidate_items, list):
            return pd.DataFrame(candidate_items).copy()
        raise TypeError("candidate_items must be a pandas DataFrame or a list of dictionaries.")

    @staticmethod
    def _stable_seed(user_id: str, seed: int) -> int:
        digest = hashlib.sha256(f"ecommerce_inprocessing:{seed}:{user_id}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)

    @staticmethod
    def _compact_constraint_result(result: Dict[str, Any]) -> Dict[str, Any]:
        compact = dict(result)
        compact.pop("inventory_feasible_items", None)
        return compact

    def _prepare_candidates(self, candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        df = self._to_dataframe(candidate_items)
        if "item_id" not in df.columns:
            raise ValueError("candidate_items requires an item_id column.")
        if self.config.base_score_col not in df.columns:
            df[self.config.base_score_col] = 0.0

        df = df.drop_duplicates("item_id", keep="first").copy()
        df["item_id"] = df["item_id"].astype(str)
        df[self.config.base_score_col] = pd.to_numeric(
            df[self.config.base_score_col],
            errors="coerce",
        ).fillna(0.0).astype(float)
        return df

    def _sort_by_base_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return df.sort_values([self.config.base_score_col, "item_id"], ascending=[False, True]).reset_index(drop=True)

    def _position_weighted_utility(self, slate: pd.DataFrame) -> float:
        if slate.empty:
            return 0.0
        scores = pd.to_numeric(slate[self.config.base_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights = 1.0 / np.log2(np.arange(2, len(scores) + 2))
        return float(np.sum(scores * weights) / np.sum(weights))

    def _hard_violation(self, constraints: Mapping[str, Any], slate_size: int) -> float:
        if slate_size <= 0:
            return float(self.config.required_new_count + self.config.min_sellers + 1)
        inventory_gap = 0.0 if constraints.get("inventory_satisfied", False) else 1.0
        new_gap = max(0, self.config.required_new_count - int(constraints.get("new_item_count", 0))) / max(
            1,
            self.config.required_new_count,
        )
        seller_gap = max(0, self.config.min_sellers - int(constraints.get("seller_count", 0))) / max(1, self.config.min_sellers)
        return float(inventory_gap + new_gap + seller_gap)

    def _evaluate_slate(
        self,
        item_ids: List[str],
        candidates_by_id: pd.DataFrame,
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]],
    ) -> Dict[str, Any]:
        slate = candidates_by_id.loc[item_ids].copy().reset_index(drop=True)
        constraints = self.constraint_handler.evaluate_all(
            slate,
            user_budget_info=user_budget_info,
            alpha_inv=self.config.alpha_inv,
            required_new_count=self.config.required_new_count,
            min_sellers=self.config.min_sellers,
            target_entropy_threshold=self.config.target_entropy_threshold,
        )
        utility = self._position_weighted_utility(slate)
        hard_violation = self._hard_violation(constraints, len(slate))
        alm_penalty = float(constraints.get("augmented_lagrangian_penalty", 0.0))
        fitness = utility - alm_penalty - self.config.hard_violation_weight * hard_violation
        feasible = bool(constraints.get("all_hard_constraints_satisfied", False))
        return {
            "item_ids": item_ids,
            "slate": slate,
            "constraints": constraints,
            "utility": float(utility),
            "hard_violation": float(hard_violation),
            "alm_penalty": float(alm_penalty),
            "fitness": float(fitness),
            "feasible": feasible,
        }

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        return list(dict.fromkeys(items))

    def _complete_slate(self, seed_items: List[str], ranked_ids: List[str], k: int) -> List[str]:
        selected = self._dedupe_keep_order(seed_items)[:k]
        selected_set = set(selected)
        for item_id in ranked_ids:
            if item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) >= k:
                break
        return selected[:k]

    def _initial_population(self, ranked: pd.DataFrame, k: int, rng: np.random.Generator) -> List[List[str]]:
        ranked_ids = ranked["item_id"].astype(str).tolist()
        population: List[List[str]] = [ranked_ids[:k]]

        budget_ranked = ranked.copy()
        if "price_filled" in budget_ranked.columns:
            budget_ranked["_price_valid"] = pd.to_numeric(budget_ranked["price_filled"], errors="coerce").notna()
            budget_ranked = budget_ranked.sort_values(
                ["_price_valid", self.config.base_score_col, "item_id"],
                ascending=[False, False, True],
            )
            population.append(self._complete_slate(budget_ranked["item_id"].astype(str).tolist()[:k], ranked_ids, k))

        if "is_new" in ranked.columns:
            new_first = ranked.sort_values(
                ["is_new", self.config.base_score_col, "item_id"],
                ascending=[False, False, True],
            )["item_id"].astype(str).tolist()
            population.append(self._complete_slate(new_first[:k], ranked_ids, k))

        if "seller_id" in ranked.columns:
            seller_seed: List[str] = []
            seen_sellers = set()
            for row in ranked.itertuples(index=False):
                seller = getattr(row, "seller_id", None)
                item_id = str(getattr(row, "item_id"))
                if seller in seen_sellers:
                    continue
                seller_seed.append(item_id)
                seen_sellers.add(seller)
                if len(seller_seed) >= k:
                    break
            population.append(self._complete_slate(seller_seed, ranked_ids, k))

        pool = np.array(ranked_ids, dtype=object)
        while len(population) < self.config.population_size and len(pool) >= k:
            chosen = rng.choice(pool, size=k, replace=False).astype(str).tolist()
            chosen.sort(key=lambda item_id: ranked_ids.index(item_id))
            population.append(chosen)

        return [self._complete_slate(items, ranked_ids, k) for items in population if items]

    def _crossover(
        self,
        parent_a: List[str],
        parent_b: List[str],
        ranked_ids: List[str],
        k: int,
        rng: np.random.Generator,
    ) -> List[str]:
        child: List[str] = []
        for idx in range(k):
            source = parent_a if rng.random() < 0.5 else parent_b
            if idx < len(source):
                child.append(source[idx])
        child = self._complete_slate(child, ranked_ids, k)
        if rng.random() < self.config.mutation_rate:
            replace_idx = int(rng.integers(0, len(child)))
            available = [item_id for item_id in ranked_ids if item_id not in child]
            if available:
                child[replace_idx] = str(rng.choice(np.array(available, dtype=object)))
                child = self._complete_slate(child, ranked_ids, k)
        child.sort(key=lambda item_id: ranked_ids.index(item_id))
        return child[:k]

    @staticmethod
    def _selection_key(evaluated: Mapping[str, Any]) -> Tuple[int, float, float, float]:
        feasible_rank = 1 if evaluated.get("feasible", False) else 0
        return (
            feasible_rank,
            -float(evaluated.get("hard_violation", 0.0)),
            float(evaluated.get("fitness", 0.0)),
            float(evaluated.get("utility", 0.0)),
        )

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        requested_k = self.config.top_k if top_k is None else int(top_k)
        requested_k = max(0, requested_k)
        prepared = self._prepare_candidates(candidate_items)
        diagnostics: Dict[str, Any] = {
            "user_id": user_id,
            "requested_top_k": requested_k,
            "input_candidate_count": int(len(prepared)),
            "budget_evaluated": user_budget_info is not None,
            "num_swaps": 0,
        }

        if requested_k == 0 or prepared.empty:
            empty = prepared.head(0).copy()
            diagnostics.update(
                {
                    "inventory_feasible_count": 0,
                    "inventory_filtered_count": 0,
                    "candidate_shortage": requested_k > 0,
                    "initial_constraints": {},
                    "final_constraints": {},
                    "fully_repaired": False,
                    "search_steps": 0,
                }
            )
            return {"recommendations": empty, "item_ids": [], "diagnostics": diagnostics}

        inventory_feasible = self.constraint_handler.check_inventory(prepared, alpha_inv=self.config.alpha_inv)
        diagnostics["inventory_feasible_count"] = int(len(inventory_feasible))
        diagnostics["inventory_filtered_count"] = int(len(prepared) - len(inventory_feasible))

        if inventory_feasible.empty:
            diagnostics.update(
                {
                    "candidate_shortage": True,
                    "initial_constraints": {},
                    "final_constraints": {},
                    "fully_repaired": False,
                    "search_steps": 0,
                }
            )
            return {"recommendations": inventory_feasible.copy(), "item_ids": [], "diagnostics": diagnostics}

        ranked = self._sort_by_base_score(inventory_feasible)
        k = min(requested_k, len(ranked))
        diagnostics["candidate_shortage"] = bool(len(ranked) < requested_k)
        diagnostics["effective_top_k"] = int(k)

        candidates_by_id = ranked.set_index("item_id", drop=False)
        ranked_ids = ranked["item_id"].astype(str).tolist()
        rng = np.random.default_rng(self._stable_seed(user_id, self.config.random_seed))
        population = self._initial_population(ranked, k, rng)
        if not population:
            population = [ranked_ids[:k]]

        initial_eval = self._evaluate_slate(population[0], candidates_by_id, user_budget_info)
        diagnostics["initial_constraints"] = self._compact_constraint_result(initial_eval["constraints"])

        best_eval = initial_eval
        search_steps = 0
        for _ in range(max(0, self.config.max_generations)):
            evaluated = [
                self._evaluate_slate(self._complete_slate(individual, ranked_ids, k), candidates_by_id, user_budget_info)
                for individual in population
            ]
            evaluated.sort(key=self._selection_key, reverse=True)
            if self._selection_key(evaluated[0]) > self._selection_key(best_eval):
                best_eval = evaluated[0]

            elites = [entry["item_ids"] for entry in evaluated[: max(1, min(self.config.elite_size, len(evaluated)))]]
            next_population = elites.copy()
            while len(next_population) < self.config.population_size:
                if len(elites) >= 2:
                    idx = rng.choice(len(elites), size=2, replace=False)
                    parent_a, parent_b = elites[int(idx[0])], elites[int(idx[1])]
                else:
                    parent_a = parent_b = elites[0]
                next_population.append(self._crossover(parent_a, parent_b, ranked_ids, k, rng))
            population = next_population[: self.config.population_size]
            search_steps += len(evaluated)

        final = best_eval
        final_constraints = final["constraints"]
        diagnostics["final_constraints"] = self._compact_constraint_result(final_constraints)
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["search_steps"] = int(search_steps)
        diagnostics["final_new_item_count"] = int(final_constraints.get("new_item_count", 0))
        diagnostics["final_seller_count"] = int(final_constraints.get("seller_count", 0))
        diagnostics["final_budget_penalty"] = float(final_constraints.get("budget_penalty", 0.0))
        diagnostics["final_entropy_penalty"] = float(final_constraints.get("entropy_penalty", 0.0))
        diagnostics["final_augmented_lagrangian_penalty"] = float(
            final_constraints.get("augmented_lagrangian_penalty", 0.0)
        )
        diagnostics["final_utility"] = float(final.get("utility", 0.0))
        diagnostics["final_hard_violation"] = float(final.get("hard_violation", 0.0))
        diagnostics["final_fitness"] = float(final.get("fitness", 0.0))

        recommendations = final["slate"].copy().reset_index(drop=True)
        return {
            "recommendations": recommendations,
            "item_ids": recommendations["item_id"].astype(str).tolist(),
            "diagnostics": diagnostics,
        }
