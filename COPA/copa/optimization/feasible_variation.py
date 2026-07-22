"""Shared feasible-by-construction variation for the formal GA optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from copa.constraints.slate_registry import (
    CompiledSlateConstraintSet,
    CompiledSlateState,
)


class OptimizerTimeoutError(RuntimeError):
    """Raised at a deterministic optimizer checkpoint when its deadline expires."""


@dataclass
class VariationDiagnostics:
    constraint_checks: int = 0
    swap_attempts: int = 0
    accepted_swaps: int = 0
    fallback_count: int = 0
    elapsed_seconds: float = 0.0


class FeasibleVariationKernel:
    """Generate children using only swaps accepted by compiled constraints."""

    def __init__(
        self,
        compiled: CompiledSlateConstraintSet,
        pool: Sequence[str],
        rng: np.random.Generator,
        check_deadline: Callable[[], None],
    ) -> None:
        self.compiled = compiled
        self.pool = tuple(map(str, pool))
        self.rng = rng
        self.check_deadline = check_deadline
        self.diagnostics = VariationDiagnostics()

    def _available(self, state: CompiledSlateState) -> list[str]:
        return [
            item_id
            for item_id in self.pool
            if self.compiled.index_by_item_id[item_id] not in state.selected_set
        ]

    def _try(
        self, state: CompiledSlateState, position: int, item_id: str
    ) -> CompiledSlateState | None:
        started = perf_counter()
        self.diagnostics.swap_attempts += 1
        self.diagnostics.constraint_checks += 1
        updated = self.compiled.try_swap(state, position, item_id)
        self.diagnostics.elapsed_seconds += perf_counter() - started
        if updated is not None and updated is not state:
            self.diagnostics.accepted_swaps += 1
        return updated

    def walk(self, seed: Sequence[str], attempts: int) -> tuple[list[str], bool]:
        state = self.compiled.build_state(seed)
        changed = False
        for _ in range(max(0, attempts)):
            self.check_deadline()
            available = self._available(state)
            if not available:
                break
            position = int(self.rng.integers(0, len(state.selected_indices)))
            candidate = str(self.rng.choice(available))
            updated = self._try(state, position, candidate)
            if updated is not None:
                changed = changed or updated is not state
                state = updated
        if not changed:
            self.diagnostics.fallback_count += 1
        return self.compiled.item_ids_for_state(state), changed

    def produce(
        self,
        parent_a: Sequence[str],
        parent_b: Sequence[str],
        *,
        crossover_rate: float,
        mutation_rate: float,
        repair_attempts: int,
    ) -> tuple[list[str], bool]:
        """Return a feasible child and whether it differs from parent A."""

        state = self.compiled.build_state(parent_a)
        changed = False
        if self.rng.random() < crossover_rate:
            for position, candidate in enumerate(parent_b):
                self.check_deadline()
                if self.rng.random() >= 0.5:
                    continue
                candidate_index = self.compiled.index_by_item_id[str(candidate)]
                if candidate_index in state.selected_set:
                    continue
                updated = self._try(state, position, str(candidate))
                if updated is not None:
                    state = updated
                    changed = True
        for position in range(len(state.selected_indices)):
            self.check_deadline()
            if self.rng.random() >= mutation_rate:
                continue
            available = self._available(state)
            if not available:
                continue
            updated = self._try(
                state, position, str(self.rng.choice(available))
            )
            if updated is not None:
                state = updated
                changed = True
        if len(state.selected_indices) > 1 and self.rng.random() < mutation_rate:
            left, right = self.rng.choice(
                len(state.selected_indices), size=2, replace=False
            )
            state.selected_indices[int(left)], state.selected_indices[int(right)] = (
                state.selected_indices[int(right)],
                state.selected_indices[int(left)],
            )
            changed = True
        if not changed and repair_attempts > 0:
            child, repaired = self.walk(
                self.compiled.item_ids_for_state(state), repair_attempts
            )
            return child, repaired
        if not changed:
            self.diagnostics.fallback_count += 1
        return self.compiled.item_ids_for_state(state), changed
