"""Scalarized evolutionary baseline with the same search budget as COPA."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from copa.constraints import SlateConstraintRegistry
from copa.core import (
    CandidateStateBus,
    ObjectiveSpec,
    OptimizationConfig,
    SlateConstraintSpec,
    SlateSolution,
)
from copa.objectives import ObjectiveRegistry

from .nsga2 import ParetoOptimizer, assign_crowding_distance, non_dominated_sort
from .feasible_variation import FeasibleVariationKernel, OptimizerTimeoutError


class WeightedGeneticOptimizer:
    """Deterministic fixed-weight GA used as a controlled experiment baseline."""

    def __init__(
        self,
        objectives: ObjectiveRegistry | None = None,
        slate_constraints: SlateConstraintRegistry | None = None,
    ) -> None:
        self.objectives = objectives or ObjectiveRegistry()
        self.slate_constraints = slate_constraints or SlateConstraintRegistry()

    def optimize(
        self,
        bus: CandidateStateBus,
        specs: Iterable[ObjectiveSpec],
        config: OptimizationConfig,
        context: Mapping[str, Any] | None = None,
        *,
        weights: Sequence[float] | None = None,
        slate_specs: Iterable[SlateConstraintSpec] = (),
        feasible_seed: Sequence[str] | None = None,
    ) -> Tuple[SlateSolution, list[SlateSolution], Dict[str, Any]]:
        started = perf_counter()
        specs = self.objectives.validate(specs)
        context = context or {}
        frame = bus.query(feasible_only=True)
        slate_specs = self.slate_constraints.validate(slate_specs, frame)
        item_pool = frame["item_id"].astype(str).tolist()
        slate_size = config.top_k
        if config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if config.generations < 0:
            raise ValueError("generations cannot be negative")
        if config.optimizer_time_limit_seconds <= 0:
            raise ValueError("optimizer_time_limit_seconds must be positive")
        if config.optimizer_kernel_version != 2:
            raise ValueError("Only optimizer_kernel_version=2 is supported")
        if len(item_pool) < slate_size:
            raise ValueError(
                f"Cannot optimize a full K={slate_size} slate from {len(item_pool)} candidates"
            )

        raw_weights = np.asarray(
            weights if weights is not None else np.ones(len(specs)), dtype=float
        )
        if len(raw_weights) != len(specs) or np.any(raw_weights < 0) or raw_weights.sum() <= 0:
            raise ValueError("weights must match objective count and contain positive mass")
        normalized_weights = raw_weights / raw_weights.sum()

        rng = np.random.default_rng(config.seed)
        memo: Dict[tuple[str, ...], SlateSolution] = {}
        deadline = started + float(config.optimizer_time_limit_seconds)
        objective_seconds = 0.0
        sorting_seconds = 0.0

        def check_deadline() -> None:
            if perf_counter() > deadline:
                raise OptimizerTimeoutError(
                    f"Weighted GA exceeded {config.optimizer_time_limit_seconds:.3f}s"
                )

        compile_started = perf_counter()
        compiled_constraints = self.slate_constraints.compile(slate_specs, frame)
        compiled_objectives = self.objectives.compile(frame, specs, context)
        compile_seconds = perf_counter() - compile_started
        variation = FeasibleVariationKernel(
            compiled_constraints, item_pool, rng, check_deadline
        )

        def evaluate(item_ids: Sequence[str]) -> SlateSolution:
            nonlocal objective_seconds
            key = tuple(map(str, item_ids))
            if key not in memo:
                evaluation_started = perf_counter()
                values = compiled_objectives.evaluate(key)
                violation = (
                    compiled_constraints.total_violation(key)
                    if slate_specs
                    else 0.0
                )
                objective_seconds += perf_counter() - evaluation_started
                memo[key] = SlateSolution(
                    list(key), values, self.objectives.to_maximization(values, specs),
                    constraint_feasible=violation <= 0.0,
                    constraint_violation=violation,
                )
            cached = memo[key]
            return SlateSolution(
                cached.item_ids.copy(),
                cached.objective_values.copy(),
                cached.maximization_values.copy(),
                constraint_feasible=cached.constraint_feasible,
                constraint_violation=cached.constraint_violation,
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
        if feasible_seed is not None:
            witness_chromosome = list(map(str, feasible_seed))
        elif slate_specs:
            solved = self.slate_constraints.solve(
                frame,
                slate_specs,
                slate_size,
                objective_scores={
                    str(row["item_id"]): float(row["base_score"])
                    for row in frame.to_dict("records")
                },
                time_limit_seconds=config.slate_solver_time_limit_seconds,
            )
            if solved.status != "optimal":
                raise RuntimeError(f"Slate preflight did not produce a feasible seed: {solved.status}")
            witness_chromosome = solved.item_ids
        else:
            witness_chromosome = base_order[:slate_size]
        seed_chromosome = witness_chromosome.copy()
        initialization_fallbacks = 0
        if slate_specs and not config.use_milp_seed:
            for _ in range(max(1, config.slate_repair_attempts)):
                proposal = rng.choice(
                    item_pool, size=slate_size, replace=False
                ).tolist()
                if compiled_constraints.is_feasible(proposal):
                    seed_chromosome = proposal
                    break
            else:
                initialization_fallbacks += 1
        chromosomes: list[list[str]] = [seed_chromosome]
        if not slate_specs:
            chromosomes.append(novelty_order[:slate_size])
            while len(chromosomes) < config.population_size:
                chromosomes.append(
                    rng.choice(item_pool, size=slate_size, replace=False).tolist()
                )
        elif config.use_slate_feasible_operators:
            current = seed_chromosome.copy()
            while len(chromosomes) < config.population_size:
                current, _ = variation.walk(
                    current, max(config.slate_repair_attempts, slate_size)
                )
                chromosomes.append(current.copy())
        else:
            attempts = 0
            maximum_attempts = max(
                config.population_size * max(1, config.slate_repair_attempts),
                config.population_size,
            )
            while len(chromosomes) < config.population_size and attempts < maximum_attempts:
                attempts += 1
                proposal = rng.choice(
                    item_pool, size=slate_size, replace=False
                ).tolist()
                if compiled_constraints.is_feasible(proposal):
                    chromosomes.append(proposal)
            while len(chromosomes) < config.population_size:
                chromosomes.append(witness_chromosome.copy())
                initialization_fallbacks += 1
        population = [evaluate(value) for value in chromosomes[: config.population_size]]
        repair_fallbacks = 0
        accepted_repairs = 0

        for generation in range(config.generations):
            check_deadline()
            generation_started = perf_counter()
            offspring: list[SlateSolution] = []
            while len(offspring) < config.population_size:
                check_deadline()
                parent_a = self._tournament(population, fitness, rng, config.tournament_size)
                parent_b = self._tournament(population, fitness, rng, config.tournament_size)
                if slate_specs and config.use_slate_feasible_operators:
                    child, repaired = variation.produce(
                        parent_a.item_ids,
                        parent_b.item_ids,
                        crossover_rate=config.crossover_rate,
                        mutation_rate=config.mutation_rate,
                        repair_attempts=config.slate_repair_attempts,
                    )
                    if repaired:
                        accepted_repairs += 1
                    else:
                        repair_fallbacks += 1
                else:
                    if rng.random() < config.crossover_rate:
                        child = ParetoOptimizer._crossover(
                            parent_a.item_ids,
                            parent_b.item_ids,
                            item_pool,
                            slate_size,
                            rng,
                        )
                    else:
                        child = parent_a.item_ids.copy()
                    child = ParetoOptimizer._mutate(
                        child, item_pool, config.mutation_rate, rng
                    )
                    if slate_specs and not compiled_constraints.is_feasible(child):
                        child, repaired = variation.walk(
                            child, config.slate_repair_attempts
                        )
                        if repaired:
                            accepted_repairs += 1
                        else:
                            child = parent_a.item_ids.copy()
                            repair_fallbacks += 1
                if slate_specs and not compiled_constraints.is_feasible(child):
                    child = parent_a.item_ids.copy()
                    if not compiled_constraints.is_feasible(child):
                        raise RuntimeError("Feasible parent invariant was violated")
                    if config.use_slate_feasible_operators:
                        child = parent_a.item_ids.copy()
                        repair_fallbacks += 1
                offspring.append(evaluate(child))
            sorting_started = perf_counter()
            population = sorted(
                population + offspring,
                key=lambda solution: (-fitness(solution), tuple(solution.item_ids)),
            )[: config.population_size]
            sorting_seconds += perf_counter() - sorting_started
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
            "slate_constraint_count": len(slate_specs),
            "repair_fallbacks": repair_fallbacks,
            "accepted_repairs": accepted_repairs,
            "unique_population_rate": len(unique) / max(1, len(population)),
            "initialization_fallbacks": initialization_fallbacks,
            "use_milp_seed": config.use_milp_seed,
            "use_slate_feasible_operators": config.use_slate_feasible_operators,
            "optimizer_kernel_version": config.optimizer_kernel_version,
            "optimizer_time_limit_seconds": config.optimizer_time_limit_seconds,
            "compile_seconds": compile_seconds,
            "objective_evaluation_seconds": objective_seconds,
            "sorting_seconds": sorting_seconds,
            "constraint_check_seconds": variation.diagnostics.elapsed_seconds,
            "constraint_checks": variation.diagnostics.constraint_checks,
            "swap_attempts": variation.diagnostics.swap_attempts,
            "accepted_swaps": variation.diagnostics.accepted_swaps,
            "variation_fallbacks": variation.diagnostics.fallback_count,
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
