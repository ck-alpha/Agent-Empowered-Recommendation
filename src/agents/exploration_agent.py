"""
Exploration Agent for DualAgent-Rec.
Focuses on maximizing diversity and discovering novel items.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent, Individual

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ExplorationAgent(BaseAgent):
    """
    Exploration Agent: Focuses on diversity and novelty.

    Uses unconstrained optimization with diversity-based selection.
    Primary objective: Maximize intra-list diversity and coverage.
    """

    def __init__(
        self,
        population_size: int = 100,
        num_objectives: int = 3,
        num_constraints: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,  # Higher mutation for exploration
        diversity_weight: float = 0.6  # Weight for diversity in fitness
    ):
        super().__init__(
            population_size=population_size,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )
        self.diversity_weight = diversity_weight
        self.candidate_items: List[str] = []
        self.k = 10

    def initialize_population(self, candidate_items: List[str], k: int = 10) -> None:
        """
        Initialize population with diverse solutions.
        Explicitly sample from different categories/clusters.
        """
        self.candidate_items = candidate_items
        self.k = k
        self.population = []

        for _ in range(self.population_size):
            # Random diverse sampling
            items = random.sample(candidate_items, min(k, len(candidate_items)))
            self.population.append(Individual(item_ids=items))

        self.generation = 0

    def evaluate_population(
        self,
        user_history: List[Dict],
        item_features: Dict[str, Dict],
        objectives_calculator: Any,
        constraints_handler: Any
    ) -> None:
        """
        Evaluate focusing on diversity objectives.
        Uses relaxed constraint handling to encourage exploration.
        """
        for individual in self.population:
            # 同样评估三目标，但探索侧会把 f2/f3（多样性/新颖性）作为主要驱动力。
            scores = objectives_calculator.calculate(
                recommended_items=individual.item_ids,
                user_history=user_history,
                item_features=item_features
            )
            individual.scores = np.array(scores)

            # 仍然计算约束违反量，但只作为软惩罚，避免探索阶段过早被硬约束“锁死”。
            violations = constraints_handler.calculate_violations(
                recommended_items=individual.item_ids,
                item_features=item_features
            )
            individual.constraint_violations = np.array(violations)

            # 适应度采用 f2/f3 加权，显式鼓励跳出 exploitation 的局部最优盆地。
            diversity_score = individual.scores[1] * self.diversity_weight + \
                             individual.scores[2] * (1 - self.diversity_weight)

            # 软约束惩罚：保留可行性信号，但惩罚系数较小，符合论文“先扩展可行域邻域”思路。
            penalty = 0.1 * individual.total_violation
            individual.fitness = diversity_score - penalty

        # 更新多样性档案，作为探索智能体的“发现库”。
        self._update_diversity_archive()

    def _update_diversity_archive(self) -> None:
        """Update archive keeping diverse solutions."""
        # Sort by diversity fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # 在高适应度基础上，再用重叠率约束过滤，避免档案塌缩为相似解。
        self.archive = []
        for ind in sorted_pop:
            if len(self.archive) >= self.population_size // 2:
                break
            # Check if significantly different from existing archive members
            is_diverse = True
            for arch_ind in self.archive:
                overlap = len(set(ind.item_ids) & set(arch_ind.item_ids)) / self.k
                if overlap > 0.7:  # Too similar
                    is_diverse = False
                    break
            if is_diverse:
                self.archive.append(ind)

    def evolve(self, num_offspring: int) -> List[Individual]:
        """
        Generate offspring using DE/rand/1 inspired strategy.
        Emphasizes exploration and diversity.
        """
        offspring = []

        for _ in range(num_offspring):
            # DE/rand/1：随机选取多个来源，弱化单一精英引导，增强搜索发散性。
            r1, r2, r3 = random.sample(self.population, 3)

            # 按随机组合生成子代，优先覆盖更广决策空间。
            child_items = []
            for i in range(self.k):
                rand_val = random.random()

                if rand_val < 0.33 and i < len(r1.item_ids):
                    item = r1.item_ids[i]
                elif rand_val < 0.66 and i < len(r2.item_ids):
                    item = r2.item_ids[i]
                elif i < len(r3.item_ids):
                    item = r3.item_ids[i]
                else:
                    item = random.choice(self.candidate_items)

                child_items.append(item)

            # Remove duplicates and ensure diversity
            child_items = list(dict.fromkeys(child_items))

            # 补齐阶段继续随机采样，进一步提高候选组合差异性。
            while len(child_items) < self.k:
                available = [it for it in self.candidate_items if it not in child_items]
                if available:
                    child_items.append(random.choice(available))
                else:
                    break

            child = Individual(item_ids=child_items[:self.k])
            # [新增监控指标: source]
            # exploration 分支产生的子代统一标记为 LLM。
            child.source = 'LLM'

            # Higher mutation rate for exploration
            child = self.mutate(child, self.candidate_items)
            child.source = 'LLM'

            offspring.append(child)

        return offspring

    # 修改点：探索智能体覆盖非支配排序逻辑，明确禁用 CDP，可行性不参与支配比较。
    def non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting for exploration branch.
        Uses pure Pareto dominance on objective scores only.
        """
        n = len(population)
        if n == 0:
            return []

        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                # 修改点：use_constraints=False，探索阶段允许跨越不可行边界搜索。
                if population[i].dominates(population[j], use_constraints=False):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i], use_constraints=False):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        first_front_indices = []
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])
                first_front_indices.append(i)

        current_front = 0
        current_indices = first_front_indices
        while current_indices:
            next_front = []
            next_indices = []
            for idx in current_indices:
                for j in dominated_solutions[idx]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(population[j])
                        next_indices.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
            current_indices = next_indices

        return fronts

    # 修改点：显式覆盖环境选择，确保始终走探索侧的纯 Pareto 排序。
    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        """
        Environmental selection using NSGA-II style under pure Pareto ranking.
        """
        fronts = self.non_dominated_sort(combined)

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                self.calculate_crowding_distance(front)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.population_size - len(new_population)
                new_population.extend(front[:remaining])
                break

        return new_population

    def calculate_decision_space_diversity(self) -> float:
        """
        Calculate diversity in decision space (item combinations).
        Used for adaptive resource allocation.
        """
        if len(self.population) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i, ind1 in enumerate(self.population):
            for ind2 in self.population[i + 1:]:
                # Jaccard distance
                set1 = set(ind1.item_ids)
                set2 = set(ind2.item_ids)
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                distance = 1 - (intersection / union if union > 0 else 0)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get exploration-specific metrics."""
        base_metrics = super().get_performance_metrics()

        if self.population:
            # Add exploration-specific metrics
            base_metrics['diversity'] = np.mean([ind.scores[1] for ind in self.population])
            base_metrics['coverage'] = np.mean([ind.scores[2] for ind in self.population])
            base_metrics['decision_space_diversity'] = self.calculate_decision_space_diversity()

        return base_metrics
