"""
Traditional baselines for DualAgent-Rec comparisons.

Includes:
1) In-processing weighted-sum baseline with soft penalty.
2) Post-processing greedy re-ranking baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_agent import Individual
from constraints import ConstraintConfig, ConstraintHandler
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
