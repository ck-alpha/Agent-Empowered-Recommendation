"""
Scenario-2 news recommendation baselines.

Both baselines operate on one MIND impression at a time. Candidate rows must
contain ``news_id``, a topic column (``category`` by default), and a ranker
``base_score``. The only constraint signal is topic entropy.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from constraints.news_constraint_handler import NewsConstraintConfig, NewsConstraintHandler


RANDOM_SEED = 42


@dataclass
class NewsPostProcessingConfig:
    """Configuration for scenario-2 greedy post-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    topic_col: str = "category"
    target_topic_entropy: float = 1.1
    lambda_diversity: float = 1.0
    rho_diversity: float = 1.0


@dataclass
class NewsInProcessingConfig:
    """Configuration for scenario-2 exact slate-level in-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    topic_col: str = "category"
    target_topic_entropy: float = 1.1
    lambda_diversity: float = 1.0
    rho_diversity: float = 1.0
    random_seed: int = RANDOM_SEED


@dataclass
class NewsDualAgentConfig:
    """Configuration for scenario-2 DualAgent news adapter."""

    top_k: int = 10
    base_score_col: str = "base_score"
    topic_col: str = "category"
    target_topic_entropy: float = 1.1
    lambda_diversity: float = 1.0
    rho_diversity: float = 1.0
    population_size: int = 30
    max_generations: int = 10
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    use_llm: bool = True
    llm_model: str = "qwen2.5:14b"
    llm_update_frequency: int = 10
    random_seed: int = RANDOM_SEED


def build_news_constraint_handler(
    *,
    topic_col: str,
    target_topic_entropy: float,
    lambda_diversity: float,
    rho_diversity: float,
) -> NewsConstraintHandler:
    return NewsConstraintHandler(
        NewsConstraintConfig(
            topic_col=topic_col,
            target_topic_entropy=target_topic_entropy,
            lambda_diversity=lambda_diversity,
            rho_diversity=rho_diversity,
        )
    )


def _with_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["rank"] = pd.Series(dtype=int)
        return out
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _position_weights(k: int) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=float)
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=float))


def _position_mismatch_count(left: List[str], right: List[str]) -> int:
    size = max(len(left), len(right))
    total = 0
    for idx in range(size):
        left_value = left[idx] if idx < len(left) else None
        right_value = right[idx] if idx < len(right) else None
        if left_value != right_value:
            total += 1
    return total


def _stable_signature(item_ids: Iterable[str]) -> str:
    return "|".join(str(item_id) for item_id in item_ids)


def raw_topk(
    candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
    top_k: int,
    base_score_col: str = "base_score",
) -> pd.DataFrame:
    """Return the ranker Top-K slate with deterministic tie-breaking."""
    df = candidate_items.copy() if isinstance(candidate_items, pd.DataFrame) else pd.DataFrame(candidate_items).copy()
    if df.empty or top_k <= 0:
        return _with_rank(df.head(0).copy())
    missing = [col for col in ["news_id", base_score_col] if col not in df.columns]
    if missing:
        raise ValueError(f"raw_topk requires columns: {missing}")
    df = df.drop_duplicates("news_id", keep="first").copy()
    df["news_id"] = df["news_id"].astype(str)
    df[base_score_col] = pd.to_numeric(df[base_score_col], errors="coerce").fillna(0.0).astype(float)
    return _with_rank(
        df.sort_values([base_score_col, "news_id"], ascending=[False, True])
        .head(max(0, int(top_k)))
        .reset_index(drop=True)
    )


class _NewsBase:
    """Shared utilities for scenario-2 news baselines."""

    def __init__(self, top_k: int, base_score_col: str, topic_col: str, constraint_handler: NewsConstraintHandler):
        self.top_k = max(0, int(top_k))
        self.base_score_col = str(base_score_col)
        self.topic_col = str(topic_col)
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
        required = ["news_id", self.topic_col, self.base_score_col]
        self.constraint_handler.require_columns(df, required, "_prepare_candidates")
        df = df.drop_duplicates("news_id", keep="first").copy()
        df["news_id"] = df["news_id"].astype(str)
        df[self.topic_col] = df[self.topic_col].fillna("unknown").astype(str)
        df[self.base_score_col] = pd.to_numeric(df[self.base_score_col], errors="coerce").fillna(0.0).astype(float)
        return df.sort_values([self.base_score_col, "news_id"], ascending=[False, True]).reset_index(drop=True)

    def _position_weighted_utility(self, slate: pd.DataFrame) -> float:
        if slate.empty:
            return 0.0
        scores = pd.to_numeric(slate[self.base_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights = _position_weights(len(scores))
        return float(np.sum(scores * weights) / max(float(np.sum(weights)), 1e-9))

    def _constraints(self, slate: pd.DataFrame) -> Dict[str, Any]:
        return self.constraint_handler.evaluate_all(_with_rank(slate.reset_index(drop=True)))

    def _objective(self, slate: pd.DataFrame) -> float:
        constraints = self._constraints(slate)
        return float(
            self._position_weighted_utility(slate)
            - float(constraints.get("augmented_lagrangian_penalty", 0.0))
        )

    def _empty_result(self, prepared: pd.DataFrame, user_id: str, requested_k: int) -> Dict[str, Any]:
        empty = _with_rank(prepared.head(0).copy())
        diagnostics = {
            "user_id": str(user_id),
            "requested_top_k": int(requested_k),
            "input_candidate_count": int(len(prepared)),
            "candidate_shortage": bool(requested_k > 0),
            "initial_constraints": {},
            "final_constraints": self.constraint_handler.evaluate_all(empty),
            "num_swaps": 0,
            "search_steps": 0,
            "final_utility": 0.0,
            "final_objective": 0.0,
        }
        return {"recommendations": empty, "item_ids": [], "diagnostics": diagnostics}


class _NewsDualAgentObjectives:
    """Slate utility and topic-entropy objectives for news DualAgent search."""

    def __init__(
        self,
        records_by_item: Mapping[str, Mapping[str, Any]],
        base_score_col: str,
        topic_col: str,
        constraint_handler: NewsConstraintHandler,
        top_k: int,
    ):
        self.records_by_item = {
            str(item_id): dict(record)
            for item_id, record in records_by_item.items()
        }
        self.base_score_col = str(base_score_col)
        self.topic_col = str(topic_col)
        self.constraint_handler = constraint_handler
        self.top_k = max(1, int(top_k))

    def _ordered_records(self, item_ids: Sequence[str]) -> List[Dict[str, Any]]:
        seen = set()
        records: List[Dict[str, Any]] = []
        for item_id in item_ids:
            item_key = str(item_id)
            if item_key in seen or item_key not in self.records_by_item:
                continue
            seen.add(item_key)
            records.append(dict(self.records_by_item[item_key]))
            if len(records) >= self.top_k:
                break
        records.sort(key=lambda row: (-float(row.get(self.base_score_col, 0.0)), str(row.get("news_id", ""))))
        return records

    @staticmethod
    def _weighted_utility(scores: Sequence[float]) -> float:
        if not scores:
            return 0.0
        values = np.asarray(scores, dtype=float)
        weights = _position_weights(len(values))
        return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-9))

    def calculate(
        self,
        recommended_items: List[str],
        user_history: List[Dict],
        item_features: Dict[str, Dict],
    ) -> List[float]:
        del user_history, item_features
        records = self._ordered_records(recommended_items)
        if not records:
            return [0.0, 0.0, 0.0]

        scores = [float(row.get(self.base_score_col, 0.0)) for row in records]
        utility = self._weighted_utility(scores)
        topics = [str(row.get(self.topic_col, "unknown")) for row in records]
        counts = pd.Series(topics).value_counts().astype(int).to_dict()
        entropy = NewsConstraintHandler.entropy_from_counts(counts, len(records))
        entropy_norm = float(entropy / max(np.log(max(2, len(records))), 1e-9))
        diversity_penalty = self.constraint_handler.diversity_penalty_from_entropy(entropy)
        alm_penalty = self.constraint_handler.augmented_lagrangian_penalty(diversity_penalty)
        penalty_score = float(1.0 / (1.0 + max(0.0, alm_penalty)))
        return [float(utility), entropy_norm, penalty_score]


class _NewsDualAgentSoftConstraint:
    """No hard news constraints; topic diversity is optimized as a soft objective."""

    epsilon = 0.0

    def calculate_violations(
        self,
        recommended_items: List[str],
        item_features: Dict[str, Dict],
    ) -> List[float]:
        del recommended_items, item_features
        return [0.0, 0.0, 0.0]

    def update_epsilon(self, feasibility_rate: float = None) -> None:
        del feasibility_rate
        self.epsilon = 0.0


class NewsDualAgentAgent(_NewsBase):
    """
    场景二 DualAgent 适配器。

    DualAgent 负责从一个 impression 的候选新闻中搜索 slate；最终用
    utility - ALM(topic entropy penalty) 在 Pareto 解中选择输出。
    """

    def __init__(self, config: Optional[NewsDualAgentConfig] = None):
        self.config = config or NewsDualAgentConfig()
        handler = build_news_constraint_handler(
            topic_col=self.config.topic_col,
            target_topic_entropy=self.config.target_topic_entropy,
            lambda_diversity=self.config.lambda_diversity,
            rho_diversity=self.config.rho_diversity,
        )
        super().__init__(self.config.top_k, self.config.base_score_col, self.config.topic_col, handler)

    def _records_by_item(self, prepared: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        return {
            str(row["news_id"]): row.to_dict()
            for _, row in prepared.drop_duplicates("news_id", keep="first").iterrows()
        }

    def _materialize_slate(
        self,
        item_ids: Sequence[str],
        prepared: pd.DataFrame,
        k: int,
    ) -> pd.DataFrame:
        records_by_item = self._records_by_item(prepared)
        rows: List[Dict[str, Any]] = []
        selected = set()
        for item_id in item_ids:
            item_key = str(item_id)
            if item_key in selected or item_key not in records_by_item:
                continue
            rows.append(dict(records_by_item[item_key]))
            selected.add(item_key)
            if len(rows) >= k:
                break
        if len(rows) < k:
            for _, row in prepared.iterrows():
                item_key = str(row["news_id"])
                if item_key in selected:
                    continue
                rows.append(row.to_dict())
                selected.add(item_key)
                if len(rows) >= k:
                    break
        if not rows:
            return _with_rank(prepared.head(0).copy())
        slate = pd.DataFrame(rows).sort_values(
            [self.base_score_col, "news_id"],
            ascending=[False, True],
        )
        return _with_rank(slate.head(k).reset_index(drop=True))

    def _final_objective(self, slate: pd.DataFrame) -> Tuple[float, float, float, str]:
        constraints = self.constraint_handler.evaluate_all(slate)
        utility = self._position_weighted_utility(slate)
        penalty = float(constraints.get("augmented_lagrangian_penalty", 0.0))
        entropy = float(constraints.get("topic_entropy", 0.0))
        signature = _stable_signature(slate["news_id"].astype(str).tolist()) if not slate.empty else ""
        return float(utility - penalty), utility, entropy, signature

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        requested_k = self.top_k if top_k is None else max(0, int(top_k))
        prepared = self._prepare_candidates(candidate_items)
        if requested_k == 0 or prepared.empty:
            return self._empty_result(prepared, user_id, requested_k)

        k = min(requested_k, len(prepared))
        raw_reference = _with_rank(prepared.head(k).copy())
        records_by_item = self._records_by_item(prepared)
        candidate_ids = prepared["news_id"].astype(str).tolist()

        from dualagent_rec import DualAgentConfig, DualAgentRec

        impression_key = str(prepared["impression_id"].iloc[0]) if "impression_id" in prepared.columns else str(user_id)
        digest = hashlib.sha256(
            f"news_dualagent:{self.config.random_seed}:{impression_key}".encode("utf-8")
        ).hexdigest()
        seed = int(digest[:16], 16) % (2**32)
        random.seed(seed)
        np.random.seed(seed)

        config = DualAgentConfig(
            population_size=max(4, int(self.config.population_size)),
            max_generations=max(1, int(self.config.max_generations)),
            recommendation_size=k,
            crossover_rate=float(self.config.crossover_rate),
            mutation_rate=float(self.config.mutation_rate),
            use_llm=bool(self.config.use_llm),
            llm_model=str(self.config.llm_model),
            llm_update_frequency=max(1, int(self.config.llm_update_frequency)),
            save_history=False,
        )
        objectives = _NewsDualAgentObjectives(
            records_by_item=records_by_item,
            base_score_col=self.base_score_col,
            topic_col=self.topic_col,
            constraint_handler=self.constraint_handler,
            top_k=k,
        )
        framework = DualAgentRec(
            config=config,
            objectives_calculator=objectives,
            constraint_handler=_NewsDualAgentSoftConstraint(),
        )
        best_solutions, _ = framework.optimize(
            candidate_items=candidate_ids,
            user_history=[],
            item_features={item_id: {} for item_id in candidate_ids},
            user_profile={"scenario": "news_topic_diversity", "user_id": str(user_id)},
        )

        candidates = best_solutions or []
        if not candidates:
            balanced = framework.get_recommendation("balanced")
            candidates = [balanced] if balanced is not None else []

        best_slate = raw_reference
        best_key = self._final_objective(raw_reference)
        for individual in candidates:
            if individual is None:
                continue
            slate = self._materialize_slate(individual.item_ids, prepared, k)
            key = self._final_objective(slate)
            if key[:3] > best_key[:3] or (key[:3] == best_key[:3] and key[3] < best_key[3]):
                best_key = key
                best_slate = slate

        recommendations = _with_rank(best_slate.reset_index(drop=True))
        item_ids = recommendations["news_id"].astype(str).tolist()
        raw_ids = raw_reference["news_id"].astype(str).tolist()
        constraints = self.constraint_handler.evaluate_all(recommendations)
        utility = self._position_weighted_utility(recommendations)
        diagnostics: Dict[str, Any] = {
            "user_id": str(user_id),
            "requested_top_k": int(requested_k),
            "input_candidate_count": int(len(prepared)),
            "candidate_shortage": bool(len(recommendations) < requested_k),
            "initial_constraints": self.constraint_handler.evaluate_all(raw_reference),
            "final_constraints": constraints,
            "final_utility": float(utility),
            "final_objective": float(utility - float(constraints.get("augmented_lagrangian_penalty", 0.0))),
            "num_swaps": _position_mismatch_count(item_ids, raw_ids),
            "search_steps": int(config.population_size * config.max_generations),
            "pareto_size": int(len(best_solutions)),
        }
        return {"recommendations": recommendations, "item_ids": item_ids, "diagnostics": diagnostics}


class NewsPostProcessingAgent(_NewsBase):
    """
    Greedy post-processing reranker.

    Starting from the ranker candidate order, it constructs the final slate one
    position at a time by maximizing marginal weighted utility minus marginal
    ALM topic-entropy penalty.
    """

    def __init__(self, config: Optional[NewsPostProcessingConfig] = None):
        self.config = config or NewsPostProcessingConfig()
        handler = build_news_constraint_handler(
            topic_col=self.config.topic_col,
            target_topic_entropy=self.config.target_topic_entropy,
            lambda_diversity=self.config.lambda_diversity,
            rho_diversity=self.config.rho_diversity,
        )
        super().__init__(self.config.top_k, self.config.base_score_col, self.config.topic_col, handler)

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        requested_k = self.top_k if top_k is None else max(0, int(top_k))
        prepared = self._prepare_candidates(candidate_items)
        if requested_k == 0 or prepared.empty:
            return self._empty_result(prepared, user_id, requested_k)

        k = min(requested_k, len(prepared))
        raw_reference = _with_rank(prepared.head(k).copy())
        diagnostics: Dict[str, Any] = {
            "user_id": str(user_id),
            "requested_top_k": int(requested_k),
            "input_candidate_count": int(len(prepared)),
            "initial_constraints": self.constraint_handler.evaluate_all(raw_reference),
            "num_swaps": 0,
            "search_steps": 0,
        }

        selected_indices: List[int] = []
        selected_ids = set()
        topic_counts: Dict[str, int] = {}
        weight_norm = max(float(np.sum(_position_weights(k))), 1e-9)
        current_penalty = self.constraint_handler.augmented_lagrangian_penalty(
            self.constraint_handler.config.target_topic_entropy
        )

        for position in range(1, k + 1):
            best_idx: Optional[int] = None
            best_key: Optional[Tuple[float, float, str]] = None
            weight = float(_position_weights(position)[-1] / weight_norm)
            for idx, row in prepared.iterrows():
                news_id = str(row["news_id"])
                if news_id in selected_ids:
                    continue
                topic = str(row[self.topic_col])
                trial_counts = dict(topic_counts)
                trial_counts[topic] = int(trial_counts.get(topic, 0)) + 1
                entropy = NewsConstraintHandler.entropy_from_counts(trial_counts, position)
                diversity_penalty = self.constraint_handler.diversity_penalty_from_entropy(entropy)
                trial_penalty = self.constraint_handler.augmented_lagrangian_penalty(diversity_penalty)
                marginal_penalty = trial_penalty - current_penalty
                marginal_score = float(row[self.base_score_col]) * weight - marginal_penalty
                key = (marginal_score, float(row[self.base_score_col]), news_id)
                diagnostics["search_steps"] += 1
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = int(idx)
            if best_idx is None:
                break
            row = prepared.loc[best_idx]
            selected_indices.append(best_idx)
            selected_ids.add(str(row["news_id"]))
            topic = str(row[self.topic_col])
            topic_counts[topic] = int(topic_counts.get(topic, 0)) + 1
            entropy = NewsConstraintHandler.entropy_from_counts(topic_counts, len(selected_indices))
            current_penalty = self.constraint_handler.augmented_lagrangian_penalty(
                self.constraint_handler.diversity_penalty_from_entropy(entropy)
            )

        recommendations = _with_rank(prepared.loc[selected_indices].reset_index(drop=True)) if selected_indices else prepared.head(0).copy()
        item_ids = recommendations["news_id"].astype(str).tolist() if not recommendations.empty else []
        raw_ids = raw_reference["news_id"].astype(str).tolist()
        diagnostics["candidate_shortage"] = bool(len(recommendations) < requested_k)
        diagnostics["final_constraints"] = self.constraint_handler.evaluate_all(recommendations)
        diagnostics["final_utility"] = self._position_weighted_utility(recommendations)
        diagnostics["final_objective"] = float(
            diagnostics["final_utility"]
            - float(diagnostics["final_constraints"].get("augmented_lagrangian_penalty", 0.0))
        )
        diagnostics["num_swaps"] = _position_mismatch_count(item_ids, raw_ids)
        return {"recommendations": recommendations, "item_ids": item_ids, "diagnostics": diagnostics}


class NewsInProcessingAgent(_NewsBase):
    """
    Exact slate-level in-processing baseline.

    It enumerates topic-count allocations for the whole Top-K slate, evaluates
    the complete list objective, and returns the best slate directly without a
    post-hoc repair step.
    """

    def __init__(self, config: Optional[NewsInProcessingConfig] = None):
        self.config = config or NewsInProcessingConfig()
        handler = build_news_constraint_handler(
            topic_col=self.config.topic_col,
            target_topic_entropy=self.config.target_topic_entropy,
            lambda_diversity=self.config.lambda_diversity,
            rho_diversity=self.config.rho_diversity,
        )
        super().__init__(self.config.top_k, self.config.base_score_col, self.config.topic_col, handler)

    def _topic_groups(self, ranked: pd.DataFrame, k: int) -> Dict[str, pd.DataFrame]:
        groups: Dict[str, pd.DataFrame] = {}
        for topic, group in ranked.groupby(self.topic_col, sort=True):
            groups[str(topic)] = group.sort_values([self.base_score_col, "news_id"], ascending=[False, True]).head(k)
        return groups

    @staticmethod
    def _enumerate_allocations(
        topics: List[str],
        caps: Mapping[str, int],
        k: int,
    ) -> Iterable[Dict[str, int]]:
        allocation: Dict[str, int] = {}

        def backtrack(topic_idx: int, remaining: int) -> Iterable[Dict[str, int]]:
            if topic_idx == len(topics):
                if remaining == 0:
                    yield {topic: count for topic, count in allocation.items() if count > 0}
                return
            topic = topics[topic_idx]
            future_capacity = sum(int(caps[future]) for future in topics[topic_idx + 1 :])
            min_count = max(0, remaining - future_capacity)
            max_count = min(int(caps[topic]), remaining)
            for count in range(min_count, max_count + 1):
                allocation[topic] = int(count)
                yield from backtrack(topic_idx + 1, remaining - count)
            allocation.pop(topic, None)

        yield from backtrack(0, int(k))

    def _slate_from_allocation(self, groups: Mapping[str, pd.DataFrame], allocation: Mapping[str, int]) -> pd.DataFrame:
        pieces = []
        for topic, count in allocation.items():
            if count <= 0:
                continue
            pieces.append(groups[str(topic)].head(int(count)))
        if not pieces:
            return pd.DataFrame()
        return _with_rank(
            pd.concat(pieces, ignore_index=True)
            .sort_values([self.base_score_col, "news_id"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _allocation_objective(self, slate: pd.DataFrame) -> Tuple[float, float, float]:
        constraints = self.constraint_handler.evaluate_all(slate)
        utility = self._position_weighted_utility(slate)
        penalty = float(constraints.get("augmented_lagrangian_penalty", 0.0))
        return float(utility - penalty), float(utility), float(constraints.get("topic_entropy", 0.0))

    def _topic_payloads(self, groups: Mapping[str, pd.DataFrame], k: int) -> Dict[str, Dict[str, Any]]:
        payloads: Dict[str, Dict[str, Any]] = {}
        for topic, group in groups.items():
            trimmed = group.head(k)
            payloads[str(topic)] = {
                "scores": pd.to_numeric(trimmed[self.base_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "ids": trimmed["news_id"].astype(str).tolist(),
            }
        return payloads

    def _allocation_objective_fast(
        self,
        payloads: Mapping[str, Mapping[str, Any]],
        allocation: Mapping[str, int],
        weights: np.ndarray,
        weight_norm: float,
        k: int,
    ) -> Tuple[float, float, float, str]:
        selected: List[Tuple[float, str]] = []
        counts: Dict[str, int] = {}
        for topic, count in allocation.items():
            count = int(count)
            if count <= 0:
                continue
            payload = payloads[str(topic)]
            scores = payload["scores"]
            ids = payload["ids"]
            for idx in range(count):
                selected.append((float(scores[idx]), str(ids[idx])))
            counts[str(topic)] = count
        selected.sort(key=lambda item: (-item[0], item[1]))
        utility = float(sum(score * float(weights[idx]) for idx, (score, _) in enumerate(selected)) / weight_norm)
        entropy = NewsConstraintHandler.entropy_from_counts(counts, k)
        diversity_penalty = self.constraint_handler.diversity_penalty_from_entropy(entropy)
        penalty = self.constraint_handler.augmented_lagrangian_penalty(diversity_penalty)
        signature = _stable_signature(item_id for _, item_id in selected)
        return float(utility - penalty), utility, float(entropy), signature

    def recommend(
        self,
        user_id: str,
        candidate_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        requested_k = self.top_k if top_k is None else max(0, int(top_k))
        prepared = self._prepare_candidates(candidate_items)
        if requested_k == 0 or prepared.empty:
            return self._empty_result(prepared, user_id, requested_k)

        k = min(requested_k, len(prepared))
        raw_reference = _with_rank(prepared.head(k).copy())
        groups = self._topic_groups(prepared, k)
        payloads = self._topic_payloads(groups, k)
        topics = sorted(groups)
        caps = {topic: int(min(k, len(group))) for topic, group in groups.items()}
        weights = _position_weights(k)
        weight_norm = max(float(np.sum(weights)), 1e-9)
        best_slate: Optional[pd.DataFrame] = None
        best_allocation: Dict[str, int] = {}
        best_key: Optional[Tuple[float, float, float, str]] = None
        search_steps = 0

        for allocation in self._enumerate_allocations(topics, caps, k):
            objective, utility, entropy, signature = self._allocation_objective_fast(
                payloads, allocation, weights, weight_norm, k
            )
            key = (objective, utility, entropy, tuple(-ord(ch) for ch in signature))
            search_steps += 1
            if best_key is None or key > best_key:
                best_key = key
                best_allocation = dict(allocation)

        if best_allocation:
            best_slate = self._slate_from_allocation(groups, best_allocation)
        recommendations = best_slate if best_slate is not None else raw_reference
        recommendations = _with_rank(recommendations.reset_index(drop=True))
        item_ids = recommendations["news_id"].astype(str).tolist()
        raw_ids = raw_reference["news_id"].astype(str).tolist()
        constraints = self.constraint_handler.evaluate_all(recommendations)
        utility = self._position_weighted_utility(recommendations)
        diagnostics: Dict[str, Any] = {
            "user_id": str(user_id),
            "requested_top_k": int(requested_k),
            "input_candidate_count": int(len(prepared)),
            "candidate_shortage": bool(len(recommendations) < requested_k),
            "initial_constraints": self.constraint_handler.evaluate_all(raw_reference),
            "final_constraints": constraints,
            "final_utility": float(utility),
            "final_objective": float(utility - float(constraints.get("augmented_lagrangian_penalty", 0.0))),
            "num_swaps": _position_mismatch_count(item_ids, raw_ids),
            "search_steps": int(search_steps),
            "allocations_evaluated": int(search_steps),
            "best_topic_counts": best_allocation,
        }
        return {"recommendations": recommendations, "item_ids": item_ids, "diagnostics": diagnostics}
