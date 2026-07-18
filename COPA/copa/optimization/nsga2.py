"""A transparent, deterministic NSGA-II implementation for Top-K slates."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from copa.core import CandidateStateBus, ObjectiveSpec, OptimizationConfig, SlateSolution
from copa.objectives import ObjectiveRegistry


def dominates(left: SlateSolution, right: SlateSolution) -> bool:
    a = np.asarray(left.maximization_values, dtype=float)
    b = np.asarray(right.maximization_values, dtype=float)
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(population: Sequence[SlateSolution]) -> List[List[SlateSolution]]:
    if not population:
        return []
    domination_counts = [0] * len(population)
    dominates_indices: List[List[int]] = [[] for _ in population]
    fronts: List[List[int]] = [[]]
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if dominates(population[i], population[j]):
                dominates_indices[i].append(j)
                domination_counts[j] += 1
            elif dominates(population[j], population[i]):
                dominates_indices[j].append(i)
                domination_counts[i] += 1
    for index, count in enumerate(domination_counts):
        if count == 0:
            population[index].rank = 0
            fronts[0].append(index)
    level = 0
    while level < len(fronts) and fronts[level]:
        following: List[int] = []
        for index in fronts[level]:
            for dominated_index in dominates_indices[index]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    population[dominated_index].rank = level + 1
                    following.append(dominated_index)
        if following:
            fronts.append(following)
        level += 1
    return [[population[index] for index in front] for front in fronts if front]


def assign_crowding_distance(front: List[SlateSolution]) -> None:
    if not front:
        return
    for solution in front:
        solution.crowding_distance = 0.0
    if len(front) <= 2:
        for solution in front:
            solution.crowding_distance = float("inf")
        return
    dimensions = len(front[0].maximization_values)
    for dimension in range(dimensions):
        ordered = sorted(front, key=lambda solution: solution.maximization_values[dimension])
        ordered[0].crowding_distance = float("inf")
        ordered[-1].crowding_distance = float("inf")
        lower = ordered[0].maximization_values[dimension]
        upper = ordered[-1].maximization_values[dimension]
        if upper == lower:
            continue
        for index in range(1, len(ordered) - 1):
            if np.isfinite(ordered[index].crowding_distance):
                ordered[index].crowding_distance += (
                    ordered[index + 1].maximization_values[dimension]
                    - ordered[index - 1].maximization_values[dimension]
                ) / (upper - lower)


class ParetoOptimizer:
    def __init__(self, objectives: ObjectiveRegistry | None = None):
        self.objectives = objectives or ObjectiveRegistry()

    def optimize(
        self,
        bus: CandidateStateBus,
        specs: Iterable[ObjectiveSpec],
        config: OptimizationConfig,
        context: Mapping[str, Any] | None = None,
    ) -> Tuple[SlateSolution, List[SlateSolution], Dict[str, Any]]:
        started = perf_counter()
        specs = self.objectives.validate(specs)
        context = context or {}
        frame = bus.query(feasible_only=True)
        item_pool = frame["item_id"].astype(str).tolist()
        slate_size = min(config.top_k, len(item_pool))
        if config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if config.generations < 0:
            raise ValueError("generations cannot be negative")
        if slate_size == 0:
            values = self.objectives.evaluate([], frame, specs, context)
            empty = SlateSolution([], values, self.objectives.to_maximization(values, specs))
            return empty, [empty], {"generations": 0, "evaluations": 1, "slate_size": 0, "runtime_seconds": perf_counter() - started}

        rng = np.random.default_rng(config.seed)
        memo: Dict[Tuple[str, ...], SlateSolution] = {}

        def evaluate(item_ids: Sequence[str]) -> SlateSolution:
            key = tuple(str(item_id) for item_id in item_ids)
            if key not in memo:
                values = self.objectives.evaluate(key, frame, specs, context)
                memo[key] = SlateSolution(list(key), values, self.objectives.to_maximization(values, specs))
            cached = memo[key]
            return SlateSolution(
                cached.item_ids.copy(),
                cached.objective_values.copy(),
                cached.maximization_values.copy(),
            )

        base_order = frame.sort_values(["base_score", "item_id"], ascending=[False, True])["item_id"].astype(str).tolist()
        chromosomes: List[List[str]] = [base_order[:slate_size]]
        novelty_order = sorted(
            item_pool,
            key=lambda item_id: self._metadata_number(frame, item_id, "popularity", 0.5),
        )
        chromosomes.append(novelty_order[:slate_size])
        while len(chromosomes) < config.population_size:
            chromosomes.append(rng.choice(item_pool, size=slate_size, replace=False).tolist())
        population = [evaluate(chromosome) for chromosome in chromosomes[: config.population_size]]

        for generation in range(config.generations):
            generation_started = perf_counter()
            fronts = non_dominated_sort(population)
            for front in fronts:
                assign_crowding_distance(front)
            offspring: List[SlateSolution] = []
            while len(offspring) < config.population_size:
                parent_a = self._tournament(population, rng, config.tournament_size)
                parent_b = self._tournament(population, rng, config.tournament_size)
                if rng.random() < config.crossover_rate:
                    child = self._crossover(parent_a.item_ids, parent_b.item_ids, item_pool, slate_size, rng)
                else:
                    child = parent_a.item_ids.copy()
                child = self._mutate(child, item_pool, config.mutation_rate, rng)
                offspring.append(evaluate(child))
            combined = population + offspring
            population = self._environmental_selection(combined, config.population_size)
            if bus.tracker:
                current_front = non_dominated_sort(population)[0]
                objective_names = [spec.name for spec in specs]
                objective_summary = {}
                if current_front:
                    objective_matrix = np.asarray(
                        [solution.maximization_values for solution in current_front],
                        dtype=float,
                    )
                    for index, name in enumerate(objective_names):
                        values = objective_matrix[:, index]
                        objective_summary[name] = {
                            "min": float(np.min(values)),
                            "median": float(np.median(values)),
                            "max": float(np.max(values)),
                        }
                bus.tracker.record(
                    module="ParetoOptimizationModule",
                    operation="generation",
                    status="success",
                    before_version=bus.version,
                    after_version=bus.version,
                    before_candidates=len(item_pool),
                    after_candidates=len(item_pool),
                    duration_ms=(perf_counter() - generation_started) * 1000,
                    seed=config.seed,
                    input_summary={
                        "generation": generation + 1,
                        "pareto_size": len(current_front),
                        "evaluations": len(memo),
                        "feasible_candidates": len(item_pool),
                        "objective_summary": objective_summary,
                    },
                )

        unique: Dict[Tuple[str, ...], SlateSolution] = {}
        for solution in population:
            unique.setdefault(tuple(solution.item_ids), solution)
        pareto_front = non_dominated_sort(list(unique.values()))[0]
        assign_crowding_distance(pareto_front)
        selected = self._select(pareto_front, config)
        diagnostics = {
            "generations": config.generations,
            "evaluations": len(memo),
            "population_size": config.population_size,
            "pareto_size": len(pareto_front),
            "slate_size": slate_size,
            "available_feasible_candidates": len(item_pool),
            "selection_strategy": config.selection_strategy,
            "runtime_seconds": perf_counter() - started,
        }
        return selected, pareto_front, diagnostics

    @staticmethod
    def _metadata_number(frame: pd.DataFrame, item_id: str, attribute: str, default: float) -> float:
        row = frame.loc[frame["item_id"].astype(str) == str(item_id)].iloc[0]
        return float(row["metadata"].get(attribute, default))

    @staticmethod
    def _tournament(
        population: Sequence[SlateSolution], rng: np.random.Generator, tournament_size: int
    ) -> SlateSolution:
        size = min(max(2, tournament_size), len(population))
        candidates = [population[index] for index in rng.choice(len(population), size=size, replace=False)]
        return min(candidates, key=lambda solution: (solution.rank, -solution.crowding_distance))

    @staticmethod
    def _crossover(
        left: Sequence[str], right: Sequence[str], pool: Sequence[str], slate_size: int, rng: np.random.Generator
    ) -> List[str]:
        child: List[str] = []
        for index in range(slate_size):
            value = left[index] if rng.random() < 0.5 else right[index]
            if value not in child:
                child.append(value)
        for value in list(left) + list(right) + rng.permutation(pool).tolist():
            if value not in child:
                child.append(value)
            if len(child) == slate_size:
                break
        return child

    @staticmethod
    def _mutate(
        chromosome: List[str], pool: Sequence[str], mutation_rate: float, rng: np.random.Generator
    ) -> List[str]:
        mutated = chromosome.copy()
        for position in range(len(mutated)):
            if rng.random() < mutation_rate:
                available = [item_id for item_id in pool if item_id not in mutated]
                if available:
                    mutated[position] = str(rng.choice(available))
        if len(mutated) > 1 and rng.random() < mutation_rate:
            left, right = rng.choice(len(mutated), size=2, replace=False)
            mutated[left], mutated[right] = mutated[right], mutated[left]
        return mutated

    @staticmethod
    def _environmental_selection(combined: Sequence[SlateSolution], population_size: int) -> List[SlateSolution]:
        selected: List[SlateSolution] = []
        for front in non_dominated_sort(combined):
            assign_crowding_distance(front)
            if len(selected) + len(front) <= population_size:
                selected.extend(front)
            else:
                selected.extend(sorted(front, key=lambda solution: solution.crowding_distance, reverse=True)[: population_size - len(selected)])
                break
        return selected

    @staticmethod
    def _select(front: Sequence[SlateSolution], config: OptimizationConfig) -> SlateSolution:
        matrix = np.asarray([solution.maximization_values for solution in front], dtype=float)
        lower = matrix.min(axis=0)
        ranges = matrix.max(axis=0) - lower
        normalized = np.divide(matrix - lower, ranges, out=np.ones_like(matrix), where=ranges > 0)
        if config.selection_strategy == "weighted":
            weights = np.asarray(config.objective_weights if config.objective_weights is not None else np.ones(matrix.shape[1]), dtype=float)
            if len(weights) != matrix.shape[1] or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("objective_weights must match objective count and contain positive mass")
            scores = normalized @ (weights / weights.sum())
            index = int(np.argmax(scores))
        elif config.selection_strategy == "compromise":
            index = int(np.argmin(np.linalg.norm(1.0 - normalized, axis=1)))
        else:
            raise ValueError(f"Unsupported selection strategy: {config.selection_strategy}")
        return front[index]
