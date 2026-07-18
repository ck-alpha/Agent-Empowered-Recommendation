"""Scalarized evolutionary baseline with the same search budget as COPA."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from copa.core import CandidateStateBus, ObjectiveSpec, OptimizationConfig, SlateSolution
from copa.objectives import ObjectiveRegistry

from .nsga2 import ParetoOptimizer, assign_crowding_distance, non_dominated_sort


class WeightedGeneticOptimizer:
    """Deterministic fixed-weight GA used as a controlled experiment baseline."""

    def __init__(self, objectives: ObjectiveRegistry | None = None) -> None:
        self.objectives = objectives or ObjectiveRegistry()

    def optimize(
        self,
        bus: CandidateStateBus,
        specs: Iterable[ObjectiveSpec],
        config: OptimizationConfig,
        context: Mapping[str, Any] | None = None,
        *,
        weights: Sequence[float] | None = None,
    ) -> Tuple[SlateSolution, list[SlateSolution], Dict[str, Any]]:
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

        raw_weights = np.asarray(
            weights if weights is not None else np.ones(len(specs)), dtype=float
        )
        if len(raw_weights) != len(specs) or np.any(raw_weights < 0) or raw_weights.sum() <= 0:
            raise ValueError("weights must match objective count and contain positive mass")
        normalized_weights = raw_weights / raw_weights.sum()

        if slate_size == 0:
            values = self.objectives.evaluate([], frame, specs, context)
            empty = SlateSolution([], values, self.objectives.to_maximization(values, specs))
            return empty, [empty], {
                "generations": 0,
                "evaluations": 1,
                "population_size": config.population_size,
                "pareto_size": 1,
                "slate_size": 0,
                "available_feasible_candidates": 0,
                "selection_strategy": "fixed_weight_ga",
                "objective_weights": normalized_weights.tolist(),
                "runtime_seconds": perf_counter() - started,
            }

        rng = np.random.default_rng(config.seed)
        memo: Dict[tuple[str, ...], SlateSolution] = {}

        def evaluate(item_ids: Sequence[str]) -> SlateSolution:
            key = tuple(map(str, item_ids))
            if key not in memo:
                values = self.objectives.evaluate(key, frame, specs, context)
                memo[key] = SlateSolution(
                    list(key), values, self.objectives.to_maximization(values, specs)
                )
            cached = memo[key]
            return SlateSolution(
                cached.item_ids.copy(),
                cached.objective_values.copy(),
                cached.maximization_values.copy(),
            )

        def fitness(solution: SlateSolution) -> float:
            bounded = np.clip(
                np.asarray(solution.maximization_values, dtype=float), 0.0, 1.0
            )
            return float(bounded @ normalized_weights)

        base_order = frame.sort_values(
            ["base_score", "item_id"], ascending=[False, True]
        )["item_id"].astype(str).tolist()
        novelty_order = sorted(
            item_pool,
            key=lambda item_id: ParetoOptimizer._metadata_number(
                frame, item_id, "popularity", 0.5
            ),
        )
        chromosomes: list[list[str]] = [base_order[:slate_size], novelty_order[:slate_size]]
        while len(chromosomes) < config.population_size:
            chromosomes.append(
                rng.choice(item_pool, size=slate_size, replace=False).tolist()
            )
        population = [evaluate(value) for value in chromosomes[: config.population_size]]

        for generation in range(config.generations):
            generation_started = perf_counter()
            offspring: list[SlateSolution] = []
            while len(offspring) < config.population_size:
                parent_a = self._tournament(population, fitness, rng, config.tournament_size)
                parent_b = self._tournament(population, fitness, rng, config.tournament_size)
                if rng.random() < config.crossover_rate:
                    child = ParetoOptimizer._crossover(
                        parent_a.item_ids, parent_b.item_ids, item_pool, slate_size, rng
                    )
                else:
                    child = parent_a.item_ids.copy()
                child = ParetoOptimizer._mutate(child, item_pool, config.mutation_rate, rng)
                offspring.append(evaluate(child))
            population = sorted(
                population + offspring,
                key=lambda solution: (-fitness(solution), tuple(solution.item_ids)),
            )[: config.population_size]
            if bus.tracker:
                current_front = non_dominated_sort(population)[0]
                objective_matrix = np.asarray(
                    [solution.maximization_values for solution in current_front], dtype=float
                )
                summary = {}
                for index, spec in enumerate(specs):
                    values = objective_matrix[:, index]
                    summary[spec.name] = {
                        "min": float(np.min(values)),
                        "median": float(np.median(values)),
                        "max": float(np.max(values)),
                    }
                bus.tracker.record(
                    module="WeightedGeneticOptimizationModule",
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
                        "best_scalar_fitness": fitness(population[0]),
                        "objective_summary": summary,
                    },
                )

        unique: Dict[tuple[str, ...], SlateSolution] = {}
        for solution in population:
            unique.setdefault(tuple(solution.item_ids), solution)
        final_population = list(unique.values())
        pareto_front = non_dominated_sort(final_population)[0]
        assign_crowding_distance(pareto_front)
        selected = min(
            final_population,
            key=lambda solution: (-fitness(solution), tuple(solution.item_ids)),
        )
        diagnostics = {
            "generations": config.generations,
            "evaluations": len(memo),
            "population_size": config.population_size,
            "pareto_size": len(pareto_front),
            "slate_size": slate_size,
            "available_feasible_candidates": len(item_pool),
            "selection_strategy": "fixed_weight_ga",
            "objective_weights": normalized_weights.tolist(),
            "runtime_seconds": perf_counter() - started,
        }
        return selected, pareto_front, diagnostics

    @staticmethod
    def _tournament(population, fitness, rng, tournament_size: int) -> SlateSolution:
        size = min(max(2, tournament_size), len(population))
        candidates = [
            population[index]
            for index in rng.choice(len(population), size=size, replace=False)
        ]
        return min(
            candidates,
            key=lambda solution: (-fitness(solution), tuple(solution.item_ids)),
        )
