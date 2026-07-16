"""Extensible soft-objective evaluation for recommendation slates."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from copa.constraints import resolve_attribute
from copa.core import CandidateStateBus, ObjectiveSpec


ObjectiveEvaluator = Callable[[Sequence[str], pd.DataFrame, ObjectiveSpec, Mapping[str, Any]], float]


class ObjectiveRegistry:
    """Maps objective names to deterministic candidate/slate evaluators."""

    def __init__(self) -> None:
        self._evaluators: Dict[str, ObjectiveEvaluator] = {}
        self.register("relevance", self._relevance)
        self.register("diversity", self._diversity)
        self.register("novelty", self._novelty)
        self.register("fairness", self._fairness)

    def register(self, name: str, evaluator: ObjectiveEvaluator, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("Objective name cannot be empty")
        if name in self._evaluators and not replace:
            raise KeyError(f"Objective already registered: {name}")
        self._evaluators[name] = evaluator

    def validate(self, specs: Iterable[ObjectiveSpec]) -> List[ObjectiveSpec]:
        specs = list(specs)
        if not specs:
            raise ValueError("At least one objective is required")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("Objective names must be unique within one optimization request")
        for spec in specs:
            evaluator_name = self._evaluator_name(spec)
            if evaluator_name not in self._evaluators:
                raise KeyError(f"Unknown objective: {spec.name} (registry evaluator: {evaluator_name})")
            if spec.direction not in {"maximize", "minimize"}:
                raise ValueError(f"Unsupported objective direction: {spec.direction}")
            if spec.scope not in {"candidate", "slate"}:
                raise ValueError(f"Unsupported objective scope: {spec.scope}")
        return specs

    def evaluate(
        self,
        item_ids: Sequence[str],
        frame: pd.DataFrame,
        specs: Iterable[ObjectiveSpec],
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        context = context or {}
        specs = self.validate(specs)
        values: Dict[str, float] = {}
        for spec in specs:
            evaluator_name = self._evaluator_name(spec)
            value = float(self._evaluators[evaluator_name](item_ids, frame, spec, context))
            if not np.isfinite(value):
                raise ValueError(f"Objective {spec.name} produced non-finite value: {value}")
            values[spec.name] = value
        return values

    @staticmethod
    def _evaluator_name(spec: ObjectiveSpec) -> str:
        """Allow uniquely named objective instances to reuse a registered evaluator."""
        return str(spec.params.get("registry_name", spec.name))

    def to_maximization(self, values: Mapping[str, float], specs: Iterable[ObjectiveSpec]) -> List[float]:
        return [float(values[spec.name]) if spec.direction == "maximize" else -float(values[spec.name]) for spec in specs]

    def annotate_candidates(
        self,
        bus: CandidateStateBus,
        specs: Iterable[ObjectiveSpec],
        context: Mapping[str, Any] | None = None,
    ) -> int:
        candidate_specs = [spec for spec in self.validate(specs) if spec.scope == "candidate"]
        context = context or {}

        def updater(frame: pd.DataFrame) -> pd.DataFrame:
            output = frame.copy(deep=True)
            annotations: List[Dict[str, float]] = []
            for item_id in output["item_id"].astype(str):
                annotations.append(self.evaluate([item_id], output, candidate_specs, context) if candidate_specs else {})
            output["soft_objectives"] = annotations
            return output

        return bus.update(
            updater,
            module="SoftObjectiveModule",
            operation="annotate_candidates",
            input_summary={"candidate_objectives": [spec.name for spec in candidate_specs]},
        )

    @staticmethod
    def _selected(item_ids: Sequence[str], frame: pd.DataFrame) -> pd.DataFrame:
        index = frame.set_index(frame["item_id"].astype(str), drop=False)
        unknown = [item_id for item_id in item_ids if str(item_id) not in index.index]
        if unknown:
            raise KeyError(f"Unknown item ids in objective evaluation: {unknown[:5]}")
        return index.loc[[str(item_id) for item_id in item_ids]].reset_index(drop=True)

    def _relevance(
        self, item_ids: Sequence[str], frame: pd.DataFrame, spec: ObjectiveSpec, context: Mapping[str, Any]
    ) -> float:
        del spec, context
        if not item_ids:
            return 0.0
        scores = pd.to_numeric(self._selected(item_ids, frame)["base_score"], errors="raise").to_numpy(float)
        weights = 1.0 / np.log2(np.arange(2, len(scores) + 2))
        return float(np.average(scores, weights=weights))

    def _diversity(
        self, item_ids: Sequence[str], frame: pd.DataFrame, spec: ObjectiveSpec, context: Mapping[str, Any]
    ) -> float:
        if len(item_ids) < 2:
            return 0.0
        embeddings = context.get("item_embeddings", {})
        selected = self._selected(item_ids, frame).to_dict("records")
        attribute = str(spec.params.get("attribute", "brand_id"))
        distances: List[float] = []
        for left, right in combinations(selected, 2):
            left_id, right_id = str(left["item_id"]), str(right["item_id"])
            if left_id in embeddings and right_id in embeddings:
                a, b = np.asarray(embeddings[left_id], dtype=float), np.asarray(embeddings[right_id], dtype=float)
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                similarity = float(np.dot(a, b) / denominator) if denominator else 0.0
                distances.append(float(np.clip(1.0 - similarity, 0.0, 2.0) / 2.0))
            else:
                left_value, right_value = resolve_attribute(left, attribute), resolve_attribute(right, attribute)
                if isinstance(left_value, (list, tuple, set, frozenset)) and isinstance(right_value, (list, tuple, set, frozenset)):
                    left_set, right_set = set(left_value), set(right_value)
                    union = left_set | right_set
                    distances.append(1.0 - len(left_set & right_set) / len(union) if union else 0.0)
                else:
                    distances.append(float(left_value != right_value))
        return float(np.mean(distances))

    def _novelty(
        self, item_ids: Sequence[str], frame: pd.DataFrame, spec: ObjectiveSpec, context: Mapping[str, Any]
    ) -> float:
        del context
        if not item_ids:
            return 0.0
        attribute = str(spec.params.get("attribute", "popularity"))
        selected = self._selected(item_ids, frame).to_dict("records")
        popularity = np.array([float(resolve_attribute(row, attribute)) for row in selected])
        return float(np.mean(1.0 - np.clip(popularity, 0.0, 1.0)))

    def _fairness(
        self, item_ids: Sequence[str], frame: pd.DataFrame, spec: ObjectiveSpec, context: Mapping[str, Any]
    ) -> float:
        del context
        if not item_ids:
            return 0.0
        attribute = str(spec.params.get("attribute", "group"))
        selected_groups = [resolve_attribute(row, attribute) for row in self._selected(item_ids, frame).to_dict("records")]
        observed_counts = Counter(selected_groups)
        target = spec.params.get("target_distribution")
        if target is None:
            catalog_groups = [resolve_attribute(row, attribute) for row in frame.to_dict("records")]
            groups = sorted(set(catalog_groups), key=str)
            target_distribution = {group: 1.0 / len(groups) for group in groups} if groups else {}
        else:
            target_distribution = {key: float(value) for key, value in dict(target).items()}
            total = sum(target_distribution.values())
            if total <= 0:
                raise ValueError("fairness target_distribution must have positive mass")
            target_distribution = {key: value / total for key, value in target_distribution.items()}
        groups = set(target_distribution) | set(observed_counts)
        observed = {group: observed_counts.get(group, 0) / len(selected_groups) for group in groups}
        total_variation = 0.5 * sum(abs(observed.get(group, 0.0) - target_distribution.get(group, 0.0)) for group in groups)
        return float(np.clip(1.0 - total_variation, 0.0, 1.0))
