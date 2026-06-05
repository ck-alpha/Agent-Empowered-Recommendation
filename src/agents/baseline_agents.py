"""
Traditional baselines for DualAgent-Rec comparisons.

Includes:
1) In-processing weighted-sum baseline with soft penalty.
2) Post-processing greedy re-ranking baseline.
"""

from __future__ import annotations

import random
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
LOGGER = logging.getLogger(__name__)
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
    """Configuration for scenario-1 capacity-only post-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    capacity_col: str = "inventory_initial"
    consumption_col: Optional[str] = None
    max_repair_iterations: int = 10_000
    progress_interval: int = 500


class _EcommerceCapacityBase:
    """Shared utilities for scenario-1 capacity-only baselines."""

    def __init__(self, top_k: int, base_score_col: str, capacity_col: str, consumption_col: Optional[str] = None):
        self.top_k = max(0, int(top_k))
        self.base_score_col = str(base_score_col)
        self.capacity_col = str(capacity_col)
        self.consumption_col = str(consumption_col) if consumption_col else None
        self.constraint_handler = EcommerceConstraintHandler(
            EcommerceConstraintConfig(capacity_col=self.capacity_col, consumption_col=self.consumption_col)
        )

    @staticmethod
    def _to_dataframe(records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, list):
            return pd.DataFrame(records).copy()
        raise TypeError("records must be a pandas DataFrame or a list of dictionaries.")

    def _prepare_candidates(self, candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        df = self._to_dataframe(candidate_items)
        if df.empty:
            columns = ["user_id", "item_id", self.base_score_col, self.capacity_col]
            if self.consumption_col:
                columns.append(self.consumption_col)
            return pd.DataFrame(columns=columns)
        required = ["user_id", "item_id", self.capacity_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"candidate_items requires columns: {missing}")
        if self.base_score_col not in df.columns:
            df[self.base_score_col] = 0.0

        df = df.drop_duplicates(["user_id", "item_id"], keep="first").copy()
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df[self.base_score_col] = pd.to_numeric(df[self.base_score_col], errors="coerce").fillna(0.0).astype(float)
        df[self.capacity_col] = (
            pd.to_numeric(df[self.capacity_col], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
        )
        if self.consumption_col:
            if self.consumption_col not in df.columns:
                df[self.consumption_col] = 1.0
            df[self.consumption_col] = (
                pd.to_numeric(df[self.consumption_col], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .astype(float)
            )
        return df.sort_values(
            ["user_id", self.base_score_col, "item_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    @staticmethod
    def _candidate_order(candidates: pd.DataFrame, user_ids: Optional[Sequence[str]]) -> List[str]:
        if user_ids is not None:
            return [str(user_id) for user_id in user_ids]
        if candidates.empty:
            return []
        return candidates["user_id"].drop_duplicates().astype(str).tolist()

    def _capacity_map(self, candidates: pd.DataFrame) -> Dict[str, float]:
        if candidates.empty:
            return {}
        return (
            candidates.drop_duplicates("item_id", keep="first")
            .set_index("item_id")[self.capacity_col]
            .astype(float)
            .clip(lower=0)
            .to_dict()
        )

    @staticmethod
    def _exposure_counts(recs: pd.DataFrame) -> Dict[str, int]:
        if recs.empty:
            return {}
        return recs["item_id"].astype(str).value_counts().astype(int).to_dict()

    def _rerank(self, recs: pd.DataFrame) -> pd.DataFrame:
        if recs.empty:
            out = recs.copy()
            if "rank" not in out.columns:
                out["rank"] = pd.Series(dtype=int)
            return out
        out = recs.sort_values(
            ["user_id", self.base_score_col, "item_id"],
            ascending=[True, False, True],
        ).copy()
        out["rank"] = out.groupby("user_id").cumcount() + 1
        return out.reset_index(drop=True)

    def _initial_topk(self, candidates: pd.DataFrame, user_ids: Sequence[str]) -> pd.DataFrame:
        if candidates.empty or self.top_k <= 0:
            return candidates.head(0).copy()
        user_set = {str(user_id) for user_id in user_ids}
        source = candidates.loc[candidates["user_id"].isin(user_set)].copy()
        if source.empty:
            return candidates.head(0).copy()
        selected = source.groupby("user_id", sort=False, group_keys=False).head(self.top_k)
        return self._rerank(selected.reset_index(drop=True))

    def _find_replacement(
        self,
        user_id: str,
        candidates: pd.DataFrame,
        current_recs: pd.DataFrame,
        exposure: Mapping[str, int],
        capacity: Mapping[str, int],
        candidate_groups: Optional[Mapping[str, pd.DataFrame]] = None,
    ) -> Optional[pd.Series]:
        if candidate_groups is not None:
            pool = candidate_groups.get(str(user_id), candidates.head(0)).copy()
        else:
            pool = candidates.loc[candidates["user_id"] == str(user_id)].copy()
        if pool.empty:
            return None
        current_items = set(
            current_recs.loc[current_recs["user_id"] == str(user_id), "item_id"].astype(str).tolist()
        )
        pool = pool.loc[~pool["item_id"].astype(str).isin(current_items)].copy()
        if pool.empty:
            return None
        pool["_available_capacity"] = pool["item_id"].map(
            lambda item_id: int(exposure.get(str(item_id), 0)) < int(capacity.get(str(item_id), 0))
        )
        pool = pool.loc[pool["_available_capacity"]]
        if pool.empty:
            return None
        pool = pool.sort_values(
            [self.base_score_col, "item_id"],
            ascending=[False, True],
        )
        return pool.iloc[0].drop(labels=["_available_capacity"], errors="ignore")

    def _utility(self, recs: pd.DataFrame) -> float:
        if recs.empty:
            return 0.0
        ranked = self._rerank(recs)
        scores = pd.to_numeric(ranked[self.base_score_col], errors="coerce").fillna(0.0).astype(float)
        ranks = pd.to_numeric(ranked["rank"], errors="coerce").fillna(1).astype(float)
        weights = 1.0 / np.log2(ranks + 1.0)
        return float(np.sum(scores * weights) / max(float(np.sum(weights)), 1e-9))

    def _candidate_shortage_rate(self, recs: pd.DataFrame, user_ids: Sequence[str]) -> float:
        if not user_ids or self.top_k <= 0:
            return 0.0
        counts = recs["user_id"].astype(str).value_counts().to_dict() if not recs.empty else {}
        shortage = sum(1 for user_id in user_ids if int(counts.get(str(user_id), 0)) < self.top_k)
        return float(shortage / len(user_ids))

    @staticmethod
    def _compact_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
        compact = dict(constraints)
        compact.pop("exposure_by_item", None)
        return compact

    def _empty_result(self, candidates: pd.DataFrame, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        empty = candidates.head(0).copy()
        final_constraints = self.constraint_handler.evaluate_all(empty)
        diagnostics.update(
            {
                "initial_constraints": final_constraints,
                "final_constraints": final_constraints,
                "fully_repaired": True,
                "candidate_shortage_rate": 0.0,
                "final_utility": 0.0,
            }
        )
        return {"recommendations": empty, "diagnostics": diagnostics, "item_ids": []}

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_budget_info: Optional[Union[Mapping[str, Any], pd.Series]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible single-user wrapper; scenario-1 experiments use recommend_batch."""
        del user_budget_info
        df = self._to_dataframe(candidate_items)
        if "user_id" not in df.columns:
            df["user_id"] = str(user_id)
        result = self.recommend_batch(df, user_ids=[str(user_id)], top_k=top_k)
        recs = result.get("recommendations", pd.DataFrame()).copy()
        return {
            "recommendations": recs,
            "item_ids": recs["item_id"].astype(str).tolist() if "item_id" in recs.columns else [],
            "diagnostics": result.get("diagnostics", {}),
        }


class EcommercePostProcessingAgent(_EcommerceCapacityBase):
    """
    场景一电商后处理基线。

    先取每个用户的 raw Top-K，再在全局推荐矩阵上修复超过 inventory_initial
    的 item exposure。替代品只允许来自该用户原始 recall candidates。
    """

    def __init__(self, config: Optional[EcommercePostProcessingConfig] = None):
        self.config = config or EcommercePostProcessingConfig()
        super().__init__(
            self.config.top_k,
            self.config.base_score_col,
            self.config.capacity_col,
            self.config.consumption_col,
        )

    def _selected_state(
        self,
        recs: pd.DataFrame,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, set], Dict[str, set], Dict[str, int]]:
        """Build mutable recommendation state for incremental capacity repair."""
        selected_by_user: Dict[str, List[Dict[str, Any]]] = {}
        selected_items_by_user: Dict[str, set] = {}
        item_to_users: Dict[str, set] = {}
        exposure: Dict[str, int] = {}
        if recs.empty:
            return selected_by_user, selected_items_by_user, item_to_users, exposure

        for user_id, group in recs.groupby("user_id", sort=False):
            user_key = str(user_id)
            ordered = group.sort_values(
                [self.base_score_col, "item_id"],
                ascending=[False, True],
            )
            rows = [row.to_dict() for _, row in ordered.iterrows()]
            selected_by_user[user_key] = rows
            item_ids = {str(row["item_id"]) for row in rows}
            selected_items_by_user[user_key] = item_ids
            for item_id in item_ids:
                exposure[item_id] = int(exposure.get(item_id, 0)) + 1
                item_to_users.setdefault(item_id, set()).add(user_key)
        return selected_by_user, selected_items_by_user, item_to_users, exposure

    def _state_to_dataframe(
        self,
        selected_by_user: Mapping[str, List[Dict[str, Any]]],
        users: Sequence[str],
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Materialize mutable recommendation state back into a ranked DataFrame."""
        columns = list(candidates.columns)
        if "rank" not in columns:
            columns.append("rank")
        rows: List[Dict[str, Any]] = []
        for user_id in users:
            user_key = str(user_id)
            user_rows = sorted(
                selected_by_user.get(user_key, []),
                key=lambda row: (-float(row.get(self.base_score_col, 0.0)), str(row.get("item_id", ""))),
            )
            for rank, row in enumerate(user_rows[: self.top_k], start=1):
                out = dict(row)
                out["user_id"] = user_key
                out["item_id"] = str(out["item_id"])
                out["rank"] = int(rank)
                rows.append(out)
        if not rows:
            out = candidates.head(0).copy()
            out["rank"] = pd.Series(dtype=int)
            return out.reindex(columns=columns)
        return pd.DataFrame(rows).reindex(columns=columns).reset_index(drop=True)

    def _removal_choice(
        self,
        over_item: str,
        selected_by_user: Mapping[str, List[Dict[str, Any]]],
        item_to_users: Mapping[str, set],
    ) -> Optional[Tuple[str, int, Dict[str, Any]]]:
        """Choose the lowest-utility exposure for one over-capacity item."""
        best: Optional[Tuple[Tuple[float, int, str], str, int, Dict[str, Any]]] = None
        for user_id in sorted(item_to_users.get(str(over_item), set())):
            ranked_rows = sorted(
                selected_by_user.get(str(user_id), []),
                key=lambda row: (-float(row.get(self.base_score_col, 0.0)), str(row.get("item_id", ""))),
            )
            for idx, row in enumerate(ranked_rows):
                if str(row.get("item_id")) != str(over_item):
                    continue
                rank = idx + 1
                key = (float(row.get(self.base_score_col, 0.0)), -rank, str(user_id))
                if best is None or key < best[0]:
                    best = (key, str(user_id), idx, row)
                break
        if best is None:
            return None
        _, user_id, idx, row = best
        return user_id, idx, row

    def _replacement_from_group(
        self,
        user_id: str,
        candidate_groups: Mapping[str, pd.DataFrame],
        selected_items: set,
        exposure: Mapping[str, int],
        capacity: Mapping[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Select the best currently feasible replacement for one user."""
        pool = candidate_groups.get(str(user_id))
        if pool is None or pool.empty:
            return None
        for _, row in pool.iterrows():
            item_id = str(row["item_id"])
            if item_id in selected_items:
                continue
            if int(exposure.get(item_id, 0)) >= int(capacity.get(item_id, 0)):
                continue
            replacement = row.to_dict()
            replacement["user_id"] = str(user_id)
            replacement["item_id"] = item_id
            return replacement
        return None

    def recommend_batch(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = max(0, int(top_k))
        candidates = self._prepare_candidates(candidate_items)
        users = self._candidate_order(candidates, user_ids)
        diagnostics: Dict[str, Any] = {
            "input_candidate_count": int(len(candidates)),
            "user_count": int(len(users)),
            "requested_top_k": int(self.top_k),
            "swap_log": [],
            "search_steps": 0,
        }
        if candidates.empty or not users or self.top_k == 0:
            return self._empty_result(candidates, diagnostics)

        recs = self._initial_topk(candidates, users)
        initial_constraints = self.constraint_handler.evaluate_all(recs)
        diagnostics["initial_constraints"] = self._compact_constraints(initial_constraints)

        capacity = self._capacity_map(candidates)
        candidate_groups = {
            str(user_id): group.reset_index(drop=True)
            for user_id, group in candidates.groupby("user_id", sort=False)
        }
        selected_by_user, selected_items_by_user, item_to_users, exposure = self._selected_state(recs)
        iterations = 0
        while iterations < self.config.max_repair_iterations:
            over_items = [item_id for item_id, count in exposure.items() if count > int(capacity.get(item_id, 0))]
            if not over_items:
                break

            repaired_any = False
            for over_item in sorted(over_items):
                while (
                    int(exposure.get(over_item, 0)) > int(capacity.get(over_item, 0))
                    and iterations < self.config.max_repair_iterations
                ):
                    removal = self._removal_choice(over_item, selected_by_user, item_to_users)
                    if removal is None:
                        break
                    user_id, remove_idx, removed = removal
                    user_rows = sorted(
                        selected_by_user.get(user_id, []),
                        key=lambda row: (-float(row.get(self.base_score_col, 0.0)), str(row.get("item_id", ""))),
                    )
                    if remove_idx >= len(user_rows):
                        break

                    removed_item = str(removed["item_id"])
                    user_rows.pop(remove_idx)
                    selected_by_user[user_id] = user_rows
                    selected_items = selected_items_by_user.setdefault(user_id, set())
                    selected_items.discard(removed_item)
                    exposure[removed_item] = max(0, int(exposure.get(removed_item, 0)) - 1)
                    if removed_item in item_to_users:
                        item_to_users[removed_item].discard(user_id)

                    replacement = self._replacement_from_group(
                        user_id=user_id,
                        candidate_groups=candidate_groups,
                        selected_items=selected_items,
                        exposure=exposure,
                        capacity=capacity,
                    )
                    if replacement is not None:
                        added_item = str(replacement["item_id"])
                        user_rows.append(replacement)
                        selected_items.add(added_item)
                        exposure[added_item] = int(exposure.get(added_item, 0)) + 1
                        item_to_users.setdefault(added_item, set()).add(user_id)
                    else:
                        added_item = None
                    selected_by_user[user_id] = sorted(
                        user_rows,
                        key=lambda row: (-float(row.get(self.base_score_col, 0.0)), str(row.get("item_id", ""))),
                    )

                    diagnostics["swap_log"].append(
                        {
                            "reason": "capacity_repair",
                            "user_id": user_id,
                            "removed_item_id": removed_item,
                            "added_item_id": added_item,
                        }
                    )
                    iterations += 1
                    diagnostics["search_steps"] = int(iterations)
                    repaired_any = True
                    if self.config.progress_interval > 0 and iterations % self.config.progress_interval == 0:
                        overflow_total = sum(
                            max(0, int(count) - int(capacity.get(item_id, 0)))
                            for item_id, count in exposure.items()
                        )
                        over_item_count = sum(
                            1 for item_id, count in exposure.items() if int(count) > int(capacity.get(item_id, 0))
                        )
                        LOGGER.info(
                            "Scenario-1 postprocessing repair progress: swaps=%s/%s, overflow=%.4f, over_items=%s",
                            iterations,
                            self.config.max_repair_iterations,
                            float(overflow_total),
                            int(over_item_count),
                        )
            if not repaired_any:
                break

        recs = self._state_to_dataframe(selected_by_user, users, candidates)
        final_constraints = self.constraint_handler.evaluate_all(recs)
        diagnostics["final_constraints"] = self._compact_constraints(final_constraints)
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["num_swaps"] = int(len(diagnostics["swap_log"]))
        diagnostics["candidate_shortage_rate"] = self._candidate_shortage_rate(recs, users)
        diagnostics["final_utility"] = self._utility(recs)
        return {"recommendations": self._rerank(recs), "diagnostics": diagnostics}


@dataclass
class EcommerceOnlineGreedyConfig:
    """Configuration for scenario-1 dynamic inventory online greedy baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    capacity_col: str = "inventory_initial"
    consumption_col: str = "expected_consumption"


class EcommerceOnlineGreedyAgent(_EcommerceCapacityBase):
    """
    Dynamic inventory baseline.

    Users are served in the provided order. For each user, the agent greedily
    takes the highest-scoring candidates whose expected consumption fits the
    current remaining item inventory, then decrements that inventory.
    """

    def __init__(self, config: Optional[EcommerceOnlineGreedyConfig] = None):
        self.config = config or EcommerceOnlineGreedyConfig()
        super().__init__(
            self.config.top_k,
            self.config.base_score_col,
            self.config.capacity_col,
            self.config.consumption_col,
        )

    def recommend_batch(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = max(0, int(top_k))
        candidates = self._prepare_candidates(candidate_items)
        users = self._candidate_order(candidates, user_ids)
        diagnostics: Dict[str, Any] = {
            "input_candidate_count": int(len(candidates)),
            "user_count": int(len(users)),
            "requested_top_k": int(self.top_k),
            "num_swaps": 0,
            "search_steps": 0,
        }
        if candidates.empty or not users or self.top_k == 0:
            return self._empty_result(candidates, diagnostics)

        raw_topk = self._initial_topk(candidates, users)
        diagnostics["initial_constraints"] = self._compact_constraints(self.constraint_handler.evaluate_all(raw_topk))

        remaining = {str(item_id): float(capacity) for item_id, capacity in self._capacity_map(candidates).items()}
        candidate_groups = {
            str(user_id): group.reset_index(drop=True)
            for user_id, group in candidates.groupby("user_id", sort=False)
        }
        rows: List[Dict[str, Any]] = []
        user_remaining_after: Dict[str, float] = {}
        search_steps = 0
        for user_id in users:
            pool = candidate_groups.get(str(user_id))
            if pool is None or pool.empty:
                user_remaining_after[str(user_id)] = float(sum(max(0.0, v) for v in remaining.values()))
                continue
            selected_items: set = set()
            rank = 0
            for _, row in pool.iterrows():
                search_steps += 1
                item_id = str(row["item_id"])
                if item_id in selected_items:
                    continue
                consumption = float(row.get(self.consumption_col or "", 1.0))
                if consumption < 0.0:
                    consumption = 0.0
                if float(remaining.get(item_id, 0.0)) + 1e-9 < consumption:
                    continue
                rank += 1
                out = row.to_dict()
                out["user_id"] = str(user_id)
                out["item_id"] = item_id
                out["rank"] = int(rank)
                rows.append(out)
                selected_items.add(item_id)
                remaining[item_id] = float(remaining.get(item_id, 0.0)) - consumption
                if rank >= self.top_k:
                    break
            user_remaining_after[str(user_id)] = float(sum(max(0.0, v) for v in remaining.values()))

        if rows:
            recs = pd.DataFrame(rows)
        else:
            recs = candidates.head(0).copy()
            recs["rank"] = pd.Series(dtype=int)

        final_constraints = self.constraint_handler.evaluate_all(recs)
        diagnostics["final_constraints"] = self._compact_constraints(final_constraints)
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["search_steps"] = int(search_steps)
        diagnostics["candidate_shortage_rate"] = self._candidate_shortage_rate(recs, users)
        diagnostics["final_utility"] = self._utility(recs)
        diagnostics["remaining_inventory_total"] = float(sum(max(0.0, value) for value in remaining.values()))
        diagnostics["depleted_item_count"] = int(sum(1 for value in remaining.values() if value <= 1e-9))
        diagnostics["remaining_inventory_after_user"] = user_remaining_after
        return {"recommendations": self._rerank(recs), "diagnostics": diagnostics}


@dataclass
class EcommerceDualAgentConfig:
    """Configuration for scenario-1 DualAgent capacity adapter."""

    top_k: int = 10
    base_score_col: str = "base_score"
    capacity_col: str = "inventory_initial"
    consumption_col: Optional[str] = None
    candidate_pool_size: int = 50
    population_size: int = 30
    max_generations: int = 10
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    use_llm: bool = True
    llm_model: str = "qwen2.5:14b"
    llm_update_frequency: int = 10
    random_seed: int = RANDOM_SEED


class _EcommerceDualAgentObjectives:
    """User-local objectives for dynamic inventory-aware DualAgent search."""

    def __init__(
        self,
        score_by_item: Mapping[str, float],
        consumption_by_item: Mapping[str, float],
        remaining_inventory: Mapping[str, float],
        top_k: int,
    ):
        self.score_by_item = {str(key): float(value) for key, value in score_by_item.items()}
        self.consumption_by_item = {str(key): float(value) for key, value in consumption_by_item.items()}
        self.remaining_inventory = {str(key): float(value) for key, value in remaining_inventory.items()}
        self.top_k = max(1, int(top_k))

    @staticmethod
    def _position_weighted_score(scores: Sequence[float]) -> float:
        if not scores:
            return 0.0
        values = np.asarray(scores, dtype=float)
        weights = 1.0 / np.log2(np.arange(2, len(values) + 2, dtype=float))
        return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-9))

    def calculate(
        self,
        recommended_items: List[str],
        user_history: List[Dict],
        item_features: Dict[str, Dict],
    ) -> List[float]:
        del user_history, item_features
        items = [str(item_id) for item_id in recommended_items[: self.top_k]]
        if not items:
            return [0.0, 0.0, 0.0]

        scores = [self.score_by_item.get(item_id, 0.0) for item_id in items]
        utility = self._position_weighted_score(scores)

        safety_scores = []
        for item_id in items:
            consumption = max(0.0, float(self.consumption_by_item.get(item_id, 1.0)))
            remaining = max(0.0, float(self.remaining_inventory.get(item_id, 0.0)))
            if consumption <= 1e-9:
                safety_scores.append(1.0)
            else:
                safety_scores.append(float(min(1.0, remaining / consumption)))
        capacity_safety = float(np.mean(safety_scores)) if safety_scores else 0.0
        fill_ratio = float(min(1.0, len(set(items)) / self.top_k))
        return [utility, capacity_safety, fill_ratio]


class _EcommerceDualAgentConstraint:
    """Hard per-user stockout constraint against current remaining inventory."""

    def __init__(self, consumption_by_item: Mapping[str, float], remaining_inventory: Mapping[str, float]):
        self.consumption_by_item = {str(key): float(value) for key, value in consumption_by_item.items()}
        self.remaining_inventory = {str(key): float(value) for key, value in remaining_inventory.items()}
        self.epsilon = 0.0

    def calculate_violations(
        self,
        recommended_items: List[str],
        item_features: Dict[str, Dict],
    ) -> List[float]:
        del item_features
        demand: Dict[str, float] = {}
        for item_id in recommended_items:
            item_key = str(item_id)
            demand[item_key] = demand.get(item_key, 0.0) + max(
                0.0,
                float(self.consumption_by_item.get(item_key, 1.0)),
            )

        violation = 0.0
        for item_id, consumption in demand.items():
            remaining = max(0.0, float(self.remaining_inventory.get(item_id, 0.0)))
            violation += max(0.0, float(consumption) - remaining)
        return [float(violation), 0.0, 0.0]

    def update_epsilon(self, feasibility_rate: float = None) -> None:
        del feasibility_rate
        self.epsilon = 0.0


class EcommerceDualAgentAgent(_EcommerceCapacityBase):
    """
    场景一 DualAgent 适配器。

    按用户服务顺序运行原 DualAgent 搜索，并用当前 remaining inventory
    评估硬约束。最终输出前会做一次库存可行投影，确保动态 stockout 为 0。
    """

    def __init__(self, config: Optional[EcommerceDualAgentConfig] = None):
        self.config = config or EcommerceDualAgentConfig()
        super().__init__(
            self.config.top_k,
            self.config.base_score_col,
            self.config.capacity_col,
            self.config.consumption_col,
        )

    @staticmethod
    def _position_mismatch_count(left: Sequence[str], right: Sequence[str]) -> int:
        size = max(len(left), len(right))
        total = 0
        for idx in range(size):
            left_value = left[idx] if idx < len(left) else None
            right_value = right[idx] if idx < len(right) else None
            if left_value != right_value:
                total += 1
        return int(total)

    def _consumption(self, row: Mapping[str, Any]) -> float:
        if self.consumption_col:
            return max(0.0, float(row.get(self.consumption_col, 0.0)))
        return 1.0

    def _group_payloads(self, group: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
        score_by_item: Dict[str, float] = {}
        consumption_by_item: Dict[str, float] = {}
        features_by_item: Dict[str, Dict[str, Any]] = {}
        for _, row in group.iterrows():
            item_id = str(row["item_id"])
            record = row.to_dict()
            score_by_item[item_id] = float(record.get(self.base_score_col, 0.0))
            consumption_by_item[item_id] = self._consumption(record)
            features_by_item[item_id] = record
        return score_by_item, consumption_by_item, features_by_item

    def _select_best_item_ids(
        self,
        candidate_ids: List[str],
        score_by_item: Mapping[str, float],
        consumption_by_item: Mapping[str, float],
        remaining: Mapping[str, float],
        user_id: str,
    ) -> Tuple[List[str], int]:
        if not candidate_ids:
            return [], 0

        from dualagent_rec import DualAgentConfig, DualAgentRec

        digest = hashlib.sha256(
            f"ecommerce_dualagent:{self.config.random_seed}:{user_id}".encode("utf-8")
        ).hexdigest()
        seed = int(digest[:16], 16) % (2**32)
        random.seed(seed)
        np.random.seed(seed)

        recommendation_size = min(self.top_k, len(candidate_ids))
        config = DualAgentConfig(
            population_size=max(4, int(self.config.population_size)),
            max_generations=max(1, int(self.config.max_generations)),
            recommendation_size=recommendation_size,
            crossover_rate=float(self.config.crossover_rate),
            mutation_rate=float(self.config.mutation_rate),
            use_llm=bool(self.config.use_llm),
            llm_model=str(self.config.llm_model),
            llm_update_frequency=max(1, int(self.config.llm_update_frequency)),
            save_history=False,
        )
        objectives = _EcommerceDualAgentObjectives(score_by_item, consumption_by_item, remaining, recommendation_size)
        constraints = _EcommerceDualAgentConstraint(consumption_by_item, remaining)
        framework = DualAgentRec(
            config=config,
            objectives_calculator=objectives,
            constraint_handler=constraints,
        )
        best_solutions, _ = framework.optimize(
            candidate_items=candidate_ids,
            user_history=[],
            item_features={item_id: {} for item_id in candidate_ids},
            user_profile={"scenario": "ecommerce_inventory", "user_id": str(user_id)},
        )

        candidates = best_solutions or []
        if not candidates:
            balanced = framework.get_recommendation("balanced")
            candidates = [balanced] if balanced is not None else []
        if not candidates:
            return candidate_ids[:recommendation_size], int(config.population_size * config.max_generations)

        def objective_key(ind: Individual) -> Tuple[float, float, float]:
            violations = constraints.calculate_violations(ind.item_ids, {})
            violation = float(np.sum(np.maximum(0.0, np.asarray(violations, dtype=float))))
            scores = objectives.calculate(ind.item_ids, [], {})
            return (
                -violation,
                float(scores[0]),
                float(np.mean(scores)),
            )

        best = max(candidates, key=objective_key)
        search_steps = int(config.population_size * config.max_generations)
        return [str(item_id) for item_id in best.item_ids[:recommendation_size]], search_steps

    def _project_to_remaining_inventory(
        self,
        selected_ids: Sequence[str],
        full_group: pd.DataFrame,
        remaining: Dict[str, float],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], int]:
        rows: List[Dict[str, Any]] = []
        selected: set = set()
        next_remaining = dict(remaining)
        by_item = {
            str(row["item_id"]): row.to_dict()
            for _, row in full_group.drop_duplicates("item_id", keep="first").iterrows()
        }
        search_steps = 0

        def try_add(record: Mapping[str, Any]) -> bool:
            item_id = str(record["item_id"])
            if item_id in selected:
                return False
            consumption = self._consumption(record)
            if float(next_remaining.get(item_id, 0.0)) + 1e-9 < consumption:
                return False
            out = dict(record)
            out["item_id"] = item_id
            rows.append(out)
            selected.add(item_id)
            next_remaining[item_id] = float(next_remaining.get(item_id, 0.0)) - consumption
            return True

        for item_id in selected_ids:
            record = by_item.get(str(item_id))
            if record is None:
                continue
            search_steps += 1
            try_add(record)
            if len(rows) >= self.top_k:
                break

        if len(rows) < self.top_k:
            for _, row in full_group.iterrows():
                search_steps += 1
                try_add(row.to_dict())
                if len(rows) >= self.top_k:
                    break

        return rows, next_remaining, int(search_steps)

    def recommend_batch(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = max(0, int(top_k))
        candidates = self._prepare_candidates(candidate_items)
        users = self._candidate_order(candidates, user_ids)
        diagnostics: Dict[str, Any] = {
            "input_candidate_count": int(len(candidates)),
            "user_count": int(len(users)),
            "requested_top_k": int(self.top_k),
            "num_swaps": 0,
            "search_steps": 0,
        }
        if candidates.empty or not users or self.top_k == 0:
            return self._empty_result(candidates, diagnostics)

        raw_topk = self._initial_topk(candidates, users)
        diagnostics["initial_constraints"] = self._compact_constraints(self.constraint_handler.evaluate_all(raw_topk))

        remaining = {str(item_id): float(capacity) for item_id, capacity in self._capacity_map(candidates).items()}
        candidate_groups = {
            str(user_id): group.reset_index(drop=True)
            for user_id, group in candidates.groupby("user_id", sort=False)
        }
        rows: List[Dict[str, Any]] = []
        user_remaining_after: Dict[str, float] = {}
        for user_id in users:
            group = candidate_groups.get(str(user_id))
            if group is None or group.empty:
                user_remaining_after[str(user_id)] = float(sum(max(0.0, value) for value in remaining.values()))
                continue

            pool = group.head(max(1, int(self.config.candidate_pool_size))).copy()
            score_by_item, consumption_by_item, _ = self._group_payloads(pool)
            selected_ids, optimize_steps = self._select_best_item_ids(
                candidate_ids=pool["item_id"].astype(str).tolist(),
                score_by_item=score_by_item,
                consumption_by_item=consumption_by_item,
                remaining=remaining,
                user_id=str(user_id),
            )
            projected, remaining, projection_steps = self._project_to_remaining_inventory(
                selected_ids=selected_ids,
                full_group=group,
                remaining=remaining,
            )
            for row in projected:
                out = dict(row)
                out["user_id"] = str(user_id)
                rows.append(out)
            diagnostics["search_steps"] += int(optimize_steps + projection_steps)
            user_remaining_after[str(user_id)] = float(sum(max(0.0, value) for value in remaining.values()))

        if rows:
            recs = pd.DataFrame(rows)
        else:
            recs = candidates.head(0).copy()
        recs = self._rerank(recs)

        raw_lists = {
            str(user_id): group.sort_values("rank")["item_id"].astype(str).tolist()
            for user_id, group in raw_topk.groupby("user_id", sort=False)
        }
        final_lists = {
            str(user_id): group.sort_values("rank")["item_id"].astype(str).tolist()
            for user_id, group in recs.groupby("user_id", sort=False)
        }
        diagnostics["num_swaps"] = int(
            sum(self._position_mismatch_count(raw_lists.get(str(user_id), []), final_lists.get(str(user_id), [])) for user_id in users)
        )

        final_constraints = self.constraint_handler.evaluate_all(recs)
        diagnostics["final_constraints"] = self._compact_constraints(final_constraints)
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["candidate_shortage_rate"] = self._candidate_shortage_rate(recs, users)
        diagnostics["final_utility"] = self._utility(recs)
        diagnostics["remaining_inventory_total"] = float(sum(max(0.0, value) for value in remaining.values()))
        diagnostics["depleted_item_count"] = int(sum(1 for value in remaining.values() if value <= 1e-9))
        diagnostics["remaining_inventory_after_user"] = user_remaining_after
        return {"recommendations": recs, "diagnostics": diagnostics}


@dataclass
class EcommerceInProcessingConfig:
    """Configuration for scenario-1 capacity-only in-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    capacity_col: str = "inventory_initial"
    consumption_col: Optional[str] = None
    random_seed: int = RANDOM_SEED
    dual_iterations: int = 30
    dual_step_size: float = 0.35
    dual_price_weight: float = 0.15
    congestion_penalty_weight: float = 0.05


class EcommerceInProcessingAgent(_EcommerceCapacityBase):
    """
    场景一电商中处理基线。

    该 baseline 不从 infeasible raw Top-K 做修复，也不做最终可行化投影。
    它通过拉格朗日 dual price 和 raw exposure congestion penalty 修改
    候选打分，然后每个用户独立取调整后 Top-K。因此它是标准 penalty-style
    in-processing：能缓解容量冲突，但不保证全局库存容量约束完全满足。
    """

    def __init__(self, config: Optional[EcommerceInProcessingConfig] = None):
        self.config = config or EcommerceInProcessingConfig()
        super().__init__(
            self.config.top_k,
            self.config.base_score_col,
            self.config.capacity_col,
            self.config.consumption_col,
        )

    @staticmethod
    def _stable_seed(user_id: str, seed: int) -> int:
        digest = hashlib.sha256(f"ecommerce_inprocessing_capacity:{seed}:{user_id}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)

    @staticmethod
    def _dcg_weight(rank: int) -> float:
        return float(1.0 / np.log2(max(2, int(rank) + 1)))

    @staticmethod
    def _drop_internal_columns(recs: pd.DataFrame) -> pd.DataFrame:
        if recs.empty:
            return recs.copy()
        internal = [col for col in recs.columns if str(col).startswith("_")]
        return recs.drop(columns=internal, errors="ignore")

    def _learn_dual_prices(
        self,
        groups: Mapping[str, pd.DataFrame],
        capacity: Mapping[str, int],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Estimate item shadow prices from unconstrained top-k demand."""
        prices: Dict[str, float] = {str(item_id): 0.0 for item_id in capacity}
        group_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
        for group in groups.values():
            if group.empty:
                continue
            item_ids = group["item_id"].astype(str).to_numpy(dtype=object)
            scores = pd.to_numeric(group[self.base_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            group_arrays.append((item_ids, scores))

        last_overflow_total = 0.0
        last_over_item_count = 0
        iterations_run = 0

        for iteration in range(max(0, int(self.config.dual_iterations))):
            exposure: Dict[str, int] = {}
            for item_ids, scores in group_arrays:
                if len(item_ids) == 0:
                    continue
                price_values = np.fromiter(
                    (float(prices.get(str(item_id), 0.0)) for item_id in item_ids),
                    dtype=float,
                    count=len(item_ids),
                )
                adjusted = scores - price_values
                chosen_idx = np.argsort(-adjusted, kind="mergesort")[: self.top_k]
                for item_id in item_ids[chosen_idx]:
                    item_key = str(item_id)
                    exposure[item_key] = int(exposure.get(item_key, 0)) + 1

            overflow_total = 0.0
            over_item_count = 0
            step = float(self.config.dual_step_size) / np.sqrt(float(iteration + 1))
            for item_id, count in exposure.items():
                cap = int(capacity.get(str(item_id), 0))
                overflow = max(0, int(count) - cap)
                if overflow <= 0:
                    continue
                over_item_count += 1
                overflow_total += float(overflow)
                prices[str(item_id)] = max(
                    0.0,
                    float(prices.get(str(item_id), 0.0)) + step * float(overflow) / max(1.0, float(cap)),
                )

            last_overflow_total = float(overflow_total)
            last_over_item_count = int(over_item_count)
            iterations_run = iteration + 1
            if overflow_total <= 0.0:
                break

        nonzero_prices = [float(value) for value in prices.values() if float(value) > 0.0]
        diagnostics = {
            "dual_iterations_run": int(iterations_run),
            "dual_unconstrained_overflow_total": float(last_overflow_total),
            "dual_unconstrained_over_item_count": int(last_over_item_count),
            "dual_nonzero_price_count": int(len(nonzero_prices)),
            "dual_max_price": float(max(nonzero_prices) if nonzero_prices else 0.0),
            "dual_mean_positive_price": float(np.mean(nonzero_prices) if nonzero_prices else 0.0),
        }
        return prices, diagnostics

    def _annotate_inprocessing_scores(
        self,
        candidates: pd.DataFrame,
        raw_topk: pd.DataFrame,
        capacity: Mapping[str, int],
        prices: Mapping[str, float],
    ) -> pd.DataFrame:
        """Add dual-price and congestion-penalized in-processing scores."""
        if candidates.empty:
            return candidates.copy()

        annotated = candidates.copy()
        raw_exposure = self._exposure_counts(raw_topk)
        congestion = {
            str(item_id): max(0.0, float(raw_exposure.get(str(item_id), 0) - int(capacity.get(str(item_id), 0))))
            / max(1.0, float(capacity.get(str(item_id), 0)))
            for item_id in set(annotated["item_id"].astype(str).tolist()) | set(capacity.keys())
        }

        annotated["_candidate_rank"] = annotated.groupby("user_id", sort=False).cumcount() + 1
        annotated["_raw_item_congestion"] = annotated["item_id"].map(
            lambda item_id: float(congestion.get(str(item_id), 0.0))
        )
        annotated["_dual_price"] = annotated["item_id"].map(lambda item_id: float(prices.get(str(item_id), 0.0)))
        annotated["_inprocess_score"] = (
            annotated[self.base_score_col].astype(float)
            - float(self.config.dual_price_weight) * annotated["_dual_price"]
            - float(self.config.congestion_penalty_weight) * annotated["_raw_item_congestion"]
        )
        return annotated.sort_values(
            ["user_id", "_inprocess_score", self.base_score_col, "item_id"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

    def _build_penalized_topk(
        self,
        candidates: pd.DataFrame,
        users: Sequence[str],
    ) -> Tuple[pd.DataFrame, int]:
        """Select each user's Top-K by adjusted score without hard capacity projection."""
        if candidates.empty or self.top_k <= 0:
            return candidates.head(0).copy(), 0
        user_set = {str(user_id) for user_id in users}
        source = candidates.loc[candidates["user_id"].isin(user_set)].copy()
        search_steps = int(len(source))
        if source.empty:
            return candidates.head(0).copy(), int(search_steps)
        selected = source.groupby("user_id", sort=False, group_keys=False).head(self.top_k)
        return self._drop_internal_columns(self._rerank(selected.reset_index(drop=True))), int(search_steps)

    def recommend_batch(
        self,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        user_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = max(0, int(top_k))
        candidates = self._prepare_candidates(candidate_items)
        users = self._candidate_order(candidates, user_ids)
        diagnostics: Dict[str, Any] = {
            "input_candidate_count": int(len(candidates)),
            "user_count": int(len(users)),
            "requested_top_k": int(self.top_k),
            "num_swaps": 0,
            "search_steps": 0,
        }
        if candidates.empty or not users or self.top_k == 0:
            return self._empty_result(candidates, diagnostics)

        raw_topk = self._initial_topk(candidates, users)
        diagnostics["initial_constraints"] = self._compact_constraints(self.constraint_handler.evaluate_all(raw_topk))

        capacity = self._capacity_map(candidates)
        groups = {
            str(user_id): group.reset_index(drop=True)
            for user_id, group in candidates.groupby("user_id", sort=False)
        }
        prices, price_diagnostics = self._learn_dual_prices(groups, capacity)
        diagnostics.update(price_diagnostics)

        scored_candidates = self._annotate_inprocessing_scores(candidates, raw_topk, capacity, prices)
        recs, construction_steps = self._build_penalized_topk(scored_candidates, users)

        final_constraints = self.constraint_handler.evaluate_all(recs)
        diagnostics["final_constraints"] = self._compact_constraints(final_constraints)
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["search_steps"] = int(construction_steps)
        diagnostics["construction_search_steps"] = int(construction_steps)
        diagnostics["local_search_steps"] = 0
        diagnostics["local_search_moves"] = 0
        diagnostics["candidate_shortage_rate"] = self._candidate_shortage_rate(recs, users)
        diagnostics["final_utility"] = self._utility(recs)
        return {"recommendations": self._drop_internal_columns(self._rerank(recs)), "diagnostics": diagnostics}
