"""
Scenario-3 news recommendation baselines.

The agents here are intentionally isolated from the existing e-commerce and
recruiting baselines. They operate on one MIND impression at a time, where each
candidate row already contains a base_score from the lightweight supervised
ranker plus news-side features used by the compact constraint protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from constraints.news_constraint_handler import NewsConstraintConfig, NewsConstraintHandler


RANDOM_SEED = 42


@dataclass
class NewsPostProcessingConfig:
    """Configuration for scenario-3 post-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    top_n: int = 5
    max_topic_count: int = 2
    target_topic_entropy: float = 1.1
    target_load: float = 45.0
    load_tolerance: float = 25.0
    lambda_diversity: float = 1.0
    lambda_load: float = 0.02
    rho_diversity: float = 1.0
    rho_load: float = 0.001


@dataclass
class NewsInProcessingConfig:
    """Configuration for scenario-3 in-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    top_n: int = 5
    max_topic_count: int = 2
    target_topic_entropy: float = 1.1
    target_load: float = 45.0
    load_tolerance: float = 25.0
    lambda_diversity: float = 1.0
    lambda_load: float = 0.02
    rho_diversity: float = 1.0
    rho_load: float = 0.001
    population_size: int = 24
    max_generations: int = 12
    elite_size: int = 6
    mutation_rate: float = 0.25
    hard_violation_weight: float = 10.0
    random_seed: int = RANDOM_SEED


def build_news_constraint_handler(
    *,
    top_n: int,
    max_topic_count: int,
    target_topic_entropy: float,
    target_load: float,
    load_tolerance: float,
    lambda_diversity: float,
    lambda_load: float,
    rho_diversity: float,
    rho_load: float,
) -> NewsConstraintHandler:
    return NewsConstraintHandler(
        NewsConstraintConfig(
            top_n=top_n,
            max_topic_count=max_topic_count,
            target_topic_entropy=target_topic_entropy,
            target_load=target_load,
            load_tolerance=load_tolerance,
            lambda_diversity=lambda_diversity,
            lambda_load=lambda_load,
            rho_diversity=rho_diversity,
            rho_load=rho_load,
        )
    )


def hard_constrained_topk(
    candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
    handler: NewsConstraintHandler,
    top_k: int,
    base_score_col: str = "base_score",
) -> pd.DataFrame:
    """Select Top-K after freshness filtering while enforcing Top-N topic cap."""
    df = candidate_items.copy() if isinstance(candidate_items, pd.DataFrame) else pd.DataFrame(candidate_items).copy()
    if df.empty or top_k <= 0:
        return df.head(0).copy()
    required = ["news_id", "category", "word_count", base_score_col]
    handler.require_columns(df, required, "hard_constrained_topk")
    df = handler.add_age_hours(df)
    df[base_score_col] = pd.to_numeric(df[base_score_col], errors="coerce").fillna(0.0).astype(float)
    ranked = handler.filter_freshness(df).sort_values([base_score_col, "news_id"], ascending=[False, True]).reset_index(drop=True)

    selected: List[pd.Series] = []
    selected_ids = set()
    topn_topic_counts: Dict[str, int] = {}
    for _, row in ranked.iterrows():
        news_id = str(row["news_id"])
        if news_id in selected_ids:
            continue
        category = str(row.get("category", "unknown"))
        if (
            len(selected) < handler.config.top_n
            and topn_topic_counts.get(category, 0) + 1 > handler.config.max_topic_count
        ):
            continue
        selected.append(row)
        selected_ids.add(news_id)
        if len(selected) <= handler.config.top_n:
            topn_topic_counts[category] = topn_topic_counts.get(category, 0) + 1
        if len(selected) >= top_k:
            break

    out = pd.DataFrame(selected).copy() if selected else ranked.head(0).copy()
    return _with_rank(out.reset_index(drop=True))


def _would_violate_topn(selected: List[pd.Series], candidate: pd.Series, handler: NewsConstraintHandler) -> bool:
    if len(selected) >= handler.config.top_n:
        return False
    category = str(candidate.get("category", "unknown"))
    count = sum(1 for row in selected[: handler.config.top_n] if str(row.get("category", "unknown")) == category)
    return bool(count + 1 > handler.config.max_topic_count)


def _entropy_from_counts(counts: Mapping[str, int], total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    probs = np.asarray([count / total_count for count in counts.values() if count > 0], dtype=float)
    return float(-np.sum(probs * np.log(probs))) if probs.size else 0.0


def _alm_from_state(
    handler: NewsConstraintHandler,
    category_counts: Mapping[str, int],
    total_count: int,
    total_word_count: float,
) -> float:
    if total_count <= 0:
        diversity_penalty = handler.config.target_topic_entropy
        load_penalty = 0.0
    else:
        entropy = _entropy_from_counts(category_counts, total_count)
        diversity_penalty = max(0.0, handler.config.target_topic_entropy - entropy)
        avg_load = float(total_word_count) / max(1, total_count)
        load_penalty = max(0.0, abs(avg_load - handler.config.target_load) - handler.config.load_tolerance)
    return handler.augmented_lagrangian_penalty(diversity_penalty, load_penalty)


def _with_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["rank"] = pd.Series(dtype=int)
        return out
    out["rank"] = np.arange(1, len(out) + 1)
    return out


class _NewsBase:
    """Shared utilities for scenario-3 news baselines."""

    def __init__(self, top_k: int, base_score_col: str, constraint_handler: NewsConstraintHandler):
        self.top_k = max(0, int(top_k))
        self.base_score_col = base_score_col
        self.constraint_handler = constraint_handler

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
            return df
        required = ["news_id", "category", "word_count", self.base_score_col]
        self.constraint_handler.require_columns(df, required, "_prepare_candidates")
        df = df.drop_duplicates("news_id", keep="first").copy()
        df["news_id"] = df["news_id"].astype(str)
        df[self.base_score_col] = pd.to_numeric(df[self.base_score_col], errors="coerce").fillna(0.0).astype(float)
        df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df = self.constraint_handler.add_age_hours(df)
        return df.sort_values([self.base_score_col, "news_id"], ascending=[False, True]).reset_index(drop=True)

    def _position_weighted_utility(self, slate: pd.DataFrame) -> float:
        if slate.empty:
            return 0.0
        scores = pd.to_numeric(slate[self.base_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights = 1.0 / np.log2(np.arange(2, len(scores) + 2))
        return float(np.sum(scores * weights) / max(np.sum(weights), 1e-9))

    def _evaluate_slate(self, slate: pd.DataFrame) -> Dict[str, Any]:
        slate = _with_rank(slate.reset_index(drop=True))
        constraints = self.constraint_handler.evaluate_all(slate)
        utility = self._position_weighted_utility(slate)
        hard_violation = self.constraint_handler.hard_violation_amount(slate)
        alm = float(constraints.get("augmented_lagrangian_penalty", 0.0))
        return {
            "slate": slate,
            "constraints": constraints,
            "utility": float(utility),
            "hard_violation": float(hard_violation),
            "alm_penalty": alm,
            "feasible": bool(constraints.get("all_hard_constraints_satisfied", False)),
        }


class NewsPostProcessingAgent(_NewsBase):
    """
    Scenario-3 post-processing baseline.

    Steps:
    1) filter stale news by category-aware freshness;
    2) greedily build the list while satisfying Top-N topic concentration;
    3) use incremental ALM soft penalties for topic entropy and reading load.
    """

    def __init__(self, config: Optional[NewsPostProcessingConfig] = None):
        self.config = config or NewsPostProcessingConfig()
        handler = build_news_constraint_handler(
            top_n=self.config.top_n,
            max_topic_count=self.config.max_topic_count,
            target_topic_entropy=self.config.target_topic_entropy,
            target_load=self.config.target_load,
            load_tolerance=self.config.load_tolerance,
            lambda_diversity=self.config.lambda_diversity,
            lambda_load=self.config.lambda_load,
            rho_diversity=self.config.rho_diversity,
            rho_load=self.config.rho_load,
        )
        super().__init__(self.config.top_k, self.config.base_score_col, handler)

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        requested_k = self.top_k if top_k is None else max(0, int(top_k))
        prepared = self._prepare_candidates(candidate_items)
        diagnostics: Dict[str, Any] = {
            "user_id": str(user_id),
            "requested_top_k": int(requested_k),
            "input_candidate_count": int(len(prepared)),
            "num_swaps": 0,
            "search_steps": 0,
        }
        if requested_k == 0 or prepared.empty:
            empty = prepared.head(0).copy()
            diagnostics.update({"candidate_shortage": requested_k > 0, "final_constraints": {}, "fully_repaired": False})
            return {"recommendations": empty, "item_ids": [], "diagnostics": diagnostics}

        freshness_filtered = self.constraint_handler.filter_freshness(prepared)
        ranked = freshness_filtered.sort_values([self.base_score_col, "news_id"], ascending=[False, True]).reset_index(drop=True)
        diagnostics["freshness_filtered_count"] = int(len(prepared) - len(ranked))
        diagnostics["freshness_feasible_count"] = int(len(ranked))
        diagnostics["initial_hard_constraints"] = self.constraint_handler.evaluate_all(
            _with_rank(ranked.head(min(requested_k, len(ranked))).copy())
        )

        selected: List[pd.Series] = []
        selected_ids = set()
        category_counts: Dict[str, int] = {}
        topn_topic_counts: Dict[str, int] = {}
        total_word_count = 0.0
        current_alm = _alm_from_state(self.constraint_handler, category_counts, 0, total_word_count)
        while len(selected) < requested_k:
            best_row: Optional[pd.Series] = None
            best_key: Optional[Tuple[float, float, str]] = None
            for _, row in ranked.iterrows():
                news_id = str(row["news_id"])
                if news_id in selected_ids:
                    continue
                category = str(row.get("category", "unknown"))
                if (
                    len(selected) < self.constraint_handler.config.top_n
                    and topn_topic_counts.get(category, 0) + 1 > self.constraint_handler.config.max_topic_count
                ):
                    continue
                trial_counts = dict(category_counts)
                trial_counts[category] = trial_counts.get(category, 0) + 1
                trial_word_count = total_word_count + float(row.get("word_count", 0.0))
                trial_alm = _alm_from_state(
                    self.constraint_handler,
                    trial_counts,
                    len(selected) + 1,
                    trial_word_count,
                )
                delta_alm = trial_alm - current_alm
                greedy_score = float(row[self.base_score_col]) - delta_alm
                key = (greedy_score, float(row[self.base_score_col]), str(row["news_id"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = row
            if best_row is None:
                break
            selected.append(best_row)
            selected_ids.add(str(best_row["news_id"]))
            selected_category = str(best_row.get("category", "unknown"))
            category_counts[selected_category] = category_counts.get(selected_category, 0) + 1
            if len(selected) <= self.constraint_handler.config.top_n:
                topn_topic_counts[selected_category] = topn_topic_counts.get(selected_category, 0) + 1
            total_word_count += float(best_row.get("word_count", 0.0))
            current_alm = _alm_from_state(self.constraint_handler, category_counts, len(selected), total_word_count)

        recommendations = _with_rank(pd.DataFrame(selected).reset_index(drop=True)) if selected else ranked.head(0).copy()
        hard_reference = hard_constrained_topk(prepared, self.constraint_handler, requested_k, self.base_score_col)
        diagnostics["candidate_shortage"] = bool(len(recommendations) < requested_k)
        diagnostics["final_constraints"] = self.constraint_handler.evaluate_all(recommendations)
        diagnostics["fully_repaired"] = bool(diagnostics["final_constraints"].get("all_hard_constraints_satisfied", False))
        diagnostics["final_utility"] = self._position_weighted_utility(recommendations)
        diagnostics["num_swaps"] = _position_mismatch_count(
            recommendations["news_id"].astype(str).tolist(),
            hard_reference["news_id"].astype(str).tolist(),
        )
        return {
            "recommendations": recommendations,
            "item_ids": recommendations["news_id"].astype(str).tolist(),
            "diagnostics": diagnostics,
        }


def _position_mismatch_count(left: List[str], right: List[str]) -> int:
    size = max(len(left), len(right))
    total = 0
    for idx in range(size):
        l_value = left[idx] if idx < len(left) else None
        r_value = right[idx] if idx < len(right) else None
        if l_value != r_value:
            total += 1
    return total


class NewsInProcessingAgent(_NewsBase):
    """
    Scenario-3 in-processing baseline.

    This baseline searches directly over Top-K slates. It does not call the
    post-processing repair loop; hard violations and ALM soft penalties are
    part of the slate fitness.
    """

    def __init__(self, config: Optional[NewsInProcessingConfig] = None):
        self.config = config or NewsInProcessingConfig()
        handler = build_news_constraint_handler(
            top_n=self.config.top_n,
            max_topic_count=self.config.max_topic_count,
            target_topic_entropy=self.config.target_topic_entropy,
            target_load=self.config.target_load,
            load_tolerance=self.config.load_tolerance,
            lambda_diversity=self.config.lambda_diversity,
            lambda_load=self.config.lambda_load,
            rho_diversity=self.config.rho_diversity,
            rho_load=self.config.rho_load,
        )
        super().__init__(self.config.top_k, self.config.base_score_col, handler)

    @staticmethod
    def _stable_seed(user_id: str, seed: int) -> int:
        digest = hashlib.sha256(f"news_inprocessing:{seed}:{user_id}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        return list(dict.fromkeys(items))

    def _complete_slate(self, seed_ids: List[str], ranked_ids: List[str], k: int) -> List[str]:
        selected = self._dedupe(seed_ids)[:k]
        selected_set = set(selected)
        for news_id in ranked_ids:
            if news_id in selected_set:
                continue
            selected.append(news_id)
            selected_set.add(news_id)
            if len(selected) >= k:
                break
        return selected[:k]

    def _initial_population(self, ranked: pd.DataFrame, k: int, rng: np.random.Generator) -> List[List[str]]:
        ranked_ids = ranked["news_id"].astype(str).tolist()
        population: List[List[str]] = [ranked_ids[:k]]

        fresh = self.constraint_handler.filter_freshness(ranked)
        fresh_ids = fresh.sort_values([self.base_score_col, "news_id"], ascending=[False, True])["news_id"].astype(str).tolist()
        if fresh_ids:
            population.append(self._complete_slate(fresh_ids[:k], ranked_ids, k))

        hard = hard_constrained_topk(ranked, self.constraint_handler, k, self.base_score_col)
        hard_ids = hard["news_id"].astype(str).tolist()
        if hard_ids:
            population.append(self._complete_slate(hard_ids, ranked_ids, k))

        diverse_seed: List[str] = []
        seen_topics = set()
        for row in ranked.itertuples(index=False):
            topic = str(getattr(row, "category"))
            news_id = str(getattr(row, "news_id"))
            if topic in seen_topics:
                continue
            diverse_seed.append(news_id)
            seen_topics.add(topic)
            if len(diverse_seed) >= k:
                break
        if diverse_seed:
            population.append(self._complete_slate(diverse_seed, ranked_ids, k))

        pool = np.array(ranked_ids, dtype=object)
        while len(population) < self.config.population_size and len(pool) >= k:
            chosen = rng.choice(pool, size=k, replace=False).astype(str).tolist()
            chosen.sort(key=lambda item_id: ranked_ids.index(item_id))
            population.append(chosen)

        return [self._complete_slate(items, ranked_ids, k) for items in population if items]

    def _slate_from_ids(self, item_ids: List[str], candidates_by_id: pd.DataFrame) -> pd.DataFrame:
        available_ids = [item_id for item_id in item_ids if item_id in candidates_by_id.index]
        return _with_rank(candidates_by_id.loc[available_ids].copy().reset_index(drop=True))

    def _fitness(self, evaluated: Mapping[str, Any]) -> float:
        constraints = evaluated.get("constraints", {})
        return float(
            evaluated.get("utility", 0.0)
            - float(constraints.get("augmented_lagrangian_penalty", 0.0))
            - self.config.hard_violation_weight * float(evaluated.get("hard_violation", 0.0))
        )

    @staticmethod
    def _selection_key(evaluated: Mapping[str, Any]) -> Tuple[int, float, float, float]:
        return (
            1 if evaluated.get("feasible", False) else 0,
            -float(evaluated.get("hard_violation", 0.0)),
            float(evaluated.get("fitness", 0.0)),
            float(evaluated.get("utility", 0.0)),
        )

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
        if child and rng.random() < self.config.mutation_rate:
            replace_idx = int(rng.integers(0, len(child)))
            available = [item_id for item_id in ranked_ids if item_id not in child]
            if available:
                child[replace_idx] = str(rng.choice(np.array(available, dtype=object)))
                child = self._complete_slate(child, ranked_ids, k)
        child.sort(key=lambda item_id: ranked_ids.index(item_id))
        return child[:k]

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        requested_k = self.top_k if top_k is None else max(0, int(top_k))
        prepared = self._prepare_candidates(candidate_items)
        diagnostics: Dict[str, Any] = {
            "user_id": str(user_id),
            "requested_top_k": int(requested_k),
            "input_candidate_count": int(len(prepared)),
            "num_swaps": 0,
        }
        if requested_k == 0 or prepared.empty:
            empty = prepared.head(0).copy()
            diagnostics.update({"candidate_shortage": requested_k > 0, "search_steps": 0, "final_constraints": {}, "fully_repaired": False})
            return {"recommendations": empty, "item_ids": [], "diagnostics": diagnostics}

        ranked = prepared.sort_values([self.base_score_col, "news_id"], ascending=[False, True]).reset_index(drop=True)
        k = min(requested_k, len(ranked))
        candidates_by_id = ranked.set_index("news_id", drop=False)
        ranked_ids = ranked["news_id"].astype(str).tolist()
        rng = np.random.default_rng(self._stable_seed(user_id, self.config.random_seed))
        population = self._initial_population(ranked, k, rng)
        if not population:
            population = [ranked_ids[:k]]

        best_eval: Optional[Dict[str, Any]] = None
        search_steps = 0
        for _ in range(max(1, self.config.max_generations)):
            evaluated: List[Dict[str, Any]] = []
            for individual in population:
                item_ids = self._complete_slate(individual, ranked_ids, k)
                slate = self._slate_from_ids(item_ids, candidates_by_id)
                entry = self._evaluate_slate(slate)
                entry["item_ids"] = slate["news_id"].astype(str).tolist()
                entry["fitness"] = self._fitness(entry)
                evaluated.append(entry)
            evaluated.sort(key=self._selection_key, reverse=True)
            search_steps += len(evaluated)
            if evaluated and (best_eval is None or self._selection_key(evaluated[0]) > self._selection_key(best_eval)):
                best_eval = evaluated[0]

            elites = [entry["item_ids"] for entry in evaluated[: max(1, min(self.config.elite_size, len(evaluated)))]]
            next_population = elites.copy()
            while len(next_population) < self.config.population_size and elites:
                if len(elites) >= 2:
                    idx = rng.choice(len(elites), size=2, replace=False)
                    parent_a, parent_b = elites[int(idx[0])], elites[int(idx[1])]
                else:
                    parent_a = parent_b = elites[0]
                next_population.append(self._crossover(parent_a, parent_b, ranked_ids, k, rng))
            population = next_population[: self.config.population_size] if next_population else population

        final = best_eval or self._evaluate_slate(self._slate_from_ids(ranked_ids[:k], candidates_by_id))
        recommendations = final["slate"].copy().reset_index(drop=True)
        constraints = final["constraints"]
        diagnostics["candidate_shortage"] = bool(len(recommendations) < requested_k)
        diagnostics["final_constraints"] = constraints
        diagnostics["fully_repaired"] = bool(constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["search_steps"] = int(search_steps)
        diagnostics["final_utility"] = float(final.get("utility", 0.0))
        diagnostics["final_fitness"] = float(final.get("fitness", self._fitness(final)))
        diagnostics["final_hard_violation"] = float(final.get("hard_violation", 0.0))
        return {
            "recommendations": recommendations,
            "item_ids": recommendations["news_id"].astype(str).tolist(),
            "diagnostics": diagnostics,
        }
