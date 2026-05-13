"""
Scenario-2 recruiting baselines.

Both baselines operate on a candidate-job edge table. This is different from
scenario-1 e-commerce, where each user's list can be optimized independently.
Job capacity is a batch-level constraint and must be evaluated over the whole
recommendation matrix.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from constraints import RecruitingConstraintConfig, RecruitingConstraintHandler


RANDOM_SEED = 42
LOGGER = logging.getLogger(__name__)


@dataclass
class RecruitingPostProcessingConfig:
    """Configuration for scenario-2 post-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    tau_hr: float = 0.35
    lambda_reciprocal: float = 1.0
    lambda_mismatch: float = 1.0
    rho_reciprocal: float = 1.0
    rho_mismatch: float = 1.0
    salary_weight: float = 0.5
    skill_weight: float = 0.5
    max_repair_iterations: int = 10_000
    progress_interval: int = 500


@dataclass
class RecruitingInProcessingConfig:
    """Configuration for scenario-2 in-processing baseline."""

    top_k: int = 10
    base_score_col: str = "base_score"
    tau_hr: float = 0.35
    lambda_reciprocal: float = 1.0
    lambda_mismatch: float = 1.0
    rho_reciprocal: float = 1.0
    rho_mismatch: float = 1.0
    salary_weight: float = 0.5
    skill_weight: float = 0.5
    hard_violation_weight: float = 10.0
    max_search_steps: int = 2_000
    progress_interval: int = 250
    random_seed: int = RANDOM_SEED


class _RecruitingBase:
    """Shared utilities for recruiting baselines."""

    def __init__(self, top_k: int, base_score_col: str, constraint_handler: RecruitingConstraintHandler):
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

    def _prepare_pairs(self, candidate_job_pairs: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        pairs = self._to_dataframe(candidate_job_pairs)
        if pairs.empty:
            return pairs
        required = [
            "candidate_id",
            "job_id",
            self.base_score_col,
            "hr_accept_prob",
            "salary_mismatch",
            "skill_mismatch",
            "candidate_exp_years",
            "job_min_exp_years",
            "candidate_english_level_num",
            "job_english_level_num",
        ]
        self.constraint_handler.require_columns(pairs, required, "_prepare_pairs")
        pairs = pairs.drop_duplicates(["candidate_id", "job_id"], keep="first").copy()
        pairs["candidate_id"] = pairs["candidate_id"].astype(str)
        pairs["job_id"] = pairs["job_id"].astype(str)
        for col in [
            self.base_score_col,
            "hr_accept_prob",
            "salary_mismatch",
            "skill_mismatch",
            "candidate_exp_years",
            "job_min_exp_years",
            "candidate_english_level_num",
            "job_english_level_num",
        ]:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce").fillna(0.0).astype(float)
        pairs["final_score"] = self.constraint_handler.final_score(pairs, base_score_col=self.base_score_col)
        return pairs.sort_values(["candidate_id", "final_score", "job_id"], ascending=[True, False, True]).reset_index(drop=True)

    @staticmethod
    def _prepare_jobs(jobs: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        job_df = jobs.copy() if isinstance(jobs, pd.DataFrame) else pd.DataFrame(jobs).copy()
        if job_df.empty:
            return pd.DataFrame(columns=["job_id", "capacity_max"])
        if "job_id" not in job_df.columns:
            raise ValueError("jobs requires a job_id column.")
        if "capacity_max" not in job_df.columns:
            job_df["capacity_max"] = 1
        job_df["job_id"] = job_df["job_id"].astype(str)
        job_df["capacity_max"] = pd.to_numeric(job_df["capacity_max"], errors="coerce").fillna(1).clip(lower=1).astype(int)
        return job_df.drop_duplicates("job_id", keep="first").copy()

    @staticmethod
    def _candidate_order(pairs: pd.DataFrame, candidate_ids: Optional[Sequence[str]]) -> List[str]:
        if candidate_ids is not None:
            return [str(candidate_id) for candidate_id in candidate_ids]
        if pairs.empty:
            return []
        return pairs["candidate_id"].drop_duplicates().astype(str).tolist()

    def _initial_topk(self, qualified_pairs: pd.DataFrame, candidate_ids: Sequence[str]) -> pd.DataFrame:
        if qualified_pairs.empty or self.top_k <= 0:
            return qualified_pairs.head(0).copy()
        selected: List[pd.DataFrame] = []
        for candidate_id in candidate_ids:
            group = qualified_pairs.loc[qualified_pairs["candidate_id"] == str(candidate_id)]
            if group.empty:
                continue
            selected.append(group.sort_values(["final_score", "job_id"], ascending=[False, True]).head(self.top_k))
        if not selected:
            return qualified_pairs.head(0).copy()
        recs = pd.concat(selected, ignore_index=True)
        recs["rank"] = recs.groupby("candidate_id")["final_score"].rank(method="first", ascending=False).astype(int)
        return self._rerank(recs)

    @staticmethod
    def _rerank(recs: pd.DataFrame) -> pd.DataFrame:
        if recs.empty:
            return recs.copy()
        out = recs.sort_values(["candidate_id", "final_score", "job_id"], ascending=[True, False, True]).copy()
        out["rank"] = out.groupby("candidate_id").cumcount() + 1
        return out.reset_index(drop=True)

    @staticmethod
    def _exposure_counts(recs: pd.DataFrame) -> Dict[str, int]:
        if recs.empty:
            return {}
        return recs["job_id"].astype(str).value_counts().astype(int).to_dict()

    @staticmethod
    def _capacity_map(jobs: pd.DataFrame) -> Dict[str, int]:
        return jobs.set_index("job_id")["capacity_max"].astype(int).to_dict()

    def _find_replacement(
        self,
        candidate_id: str,
        qualified_pairs: pd.DataFrame,
        current_recs: pd.DataFrame,
        exposure: Mapping[str, int],
        capacity: Mapping[str, int],
        require_available_capacity: bool = False,
    ) -> Optional[pd.Series]:
        candidate_pool = qualified_pairs.loc[qualified_pairs["candidate_id"] == candidate_id].copy()
        if candidate_pool.empty:
            return None
        current_jobs = set(current_recs.loc[current_recs["candidate_id"] == candidate_id, "job_id"].astype(str))
        candidate_pool = candidate_pool.loc[~candidate_pool["job_id"].astype(str).isin(current_jobs)]
        if candidate_pool.empty:
            return None
        candidate_pool["_available_capacity"] = candidate_pool["job_id"].map(
            lambda job_id: int(exposure.get(str(job_id), 0)) < int(capacity.get(str(job_id), 1))
        )
        if require_available_capacity:
            candidate_pool = candidate_pool.loc[candidate_pool["_available_capacity"]]
            if candidate_pool.empty:
                return None
        candidate_pool = candidate_pool.sort_values(
            ["_available_capacity", "final_score", "job_id"],
            ascending=[False, False, True],
        )
        row = candidate_pool.iloc[0].drop(labels=["_available_capacity"], errors="ignore")
        return row

    def _evaluate(self, recs: pd.DataFrame, jobs: pd.DataFrame) -> Dict[str, Any]:
        constraints = self.constraint_handler.evaluate_all(recs, jobs)
        utility = self._utility(recs)
        return {"constraints": constraints, "utility": utility}

    def _utility(self, recs: pd.DataFrame) -> float:
        if recs.empty:
            return 0.0
        scores = pd.to_numeric(recs[self.base_score_col], errors="coerce").fillna(0.0).astype(float)
        ranks = pd.to_numeric(recs.get("rank", 1), errors="coerce").fillna(1).astype(float)
        weights = 1.0 / np.log2(ranks + 1.0)
        return float(np.sum(scores * weights) / max(np.sum(weights), 1e-9))


class RecruitingPostProcessingAgent(_RecruitingBase):
    """
    Scenario-2 post-processing baseline.

    Steps:
    1) Qualification hard filter.
    2) Per-candidate ALM reranking.
    3) Batch-level capacity repair over selected recommendations.
    """

    def __init__(self, config: Optional[RecruitingPostProcessingConfig] = None):
        self.config = config or RecruitingPostProcessingConfig()
        handler = RecruitingConstraintHandler(
            RecruitingConstraintConfig(
                tau_hr=self.config.tau_hr,
                lambda_reciprocal=self.config.lambda_reciprocal,
                lambda_mismatch=self.config.lambda_mismatch,
                rho_reciprocal=self.config.rho_reciprocal,
                rho_mismatch=self.config.rho_mismatch,
                salary_weight=self.config.salary_weight,
                skill_weight=self.config.skill_weight,
            )
        )
        super().__init__(self.config.top_k, self.config.base_score_col, handler)

    def recommend_batch(
        self,
        candidate_job_pairs: Union[pd.DataFrame, List[Dict[str, Any]]],
        jobs: Union[pd.DataFrame, List[Dict[str, Any]]],
        candidate_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = max(0, int(top_k))
        pairs = self._prepare_pairs(candidate_job_pairs)
        job_df = self._prepare_jobs(jobs)
        candidates = self._candidate_order(pairs, candidate_ids)
        diagnostics: Dict[str, Any] = {
            "input_pair_count": int(len(pairs)),
            "candidate_count": int(len(candidates)),
            "requested_top_k": int(self.top_k),
            "swap_log": [],
        }
        if pairs.empty or self.top_k == 0:
            empty = pairs.head(0).copy()
            diagnostics.update({"qualified_pair_count": 0, "num_swaps": 0, "fully_repaired": False})
            return {"recommendations": empty, "diagnostics": diagnostics}

        qualified = self.constraint_handler.check_qualification(pairs)
        diagnostics["qualified_pair_count"] = int(len(qualified))
        diagnostics["qualification_filter_rate"] = float(len(qualified) / max(1, len(pairs)))
        recs = self._initial_topk(qualified, candidates)
        diagnostics["initial_constraints"] = self.constraint_handler.evaluate_all(recs, job_df)
        initial_constraints = diagnostics["initial_constraints"]
        LOGGER.info(
            "Recruiting postprocessing start: selected=%s, candidates=%s, capacity_overflow=%.4f, over_capacity_jobs=%s",
            len(recs),
            len(candidates),
            float(initial_constraints.get("capacity_violation_total", 0.0)),
            int(initial_constraints.get("over_capacity_job_count", 0)),
        )

        capacity = self._capacity_map(job_df)
        iterations = 0
        while iterations < self.config.max_repair_iterations:
            exposure = self._exposure_counts(recs)
            over_jobs = [job_id for job_id, count in exposure.items() if count > int(capacity.get(job_id, 1))]
            if not over_jobs:
                break

            repaired_any = False
            for over_job in sorted(over_jobs):
                exposure = self._exposure_counts(recs)
                if exposure.get(over_job, 0) <= int(capacity.get(over_job, 1)):
                    continue
                rows = recs.loc[recs["job_id"].astype(str) == over_job].sort_values(
                    ["final_score", "hr_accept_prob", self.base_score_col],
                    ascending=[True, True, True],
                )
                if rows.empty:
                    continue
                remove_idx = int(rows.index[0])
                removed = recs.loc[remove_idx].copy()
                candidate_id = str(removed["candidate_id"])
                recs_without = recs.drop(index=remove_idx).reset_index(drop=True)
                exposure_without = self._exposure_counts(recs_without)
                replacement = self._find_replacement(
                    candidate_id,
                    qualified,
                    recs_without,
                    exposure_without,
                    capacity,
                    require_available_capacity=True,
                )
                if replacement is not None:
                    recs = pd.concat([recs_without, replacement.to_frame().T], ignore_index=True)
                    added_job = str(replacement["job_id"])
                else:
                    recs = recs_without
                    added_job = None
                diagnostics["swap_log"].append(
                    {
                        "reason": "capacity_repair",
                        "candidate_id": candidate_id,
                        "removed_job_id": str(removed["job_id"]),
                        "added_job_id": added_job,
                    }
                )
                recs = self._rerank(recs)
                iterations += 1
                if self.config.progress_interval > 0 and iterations % self.config.progress_interval == 0:
                    progress_constraints = self.constraint_handler.capacity_diagnostics(recs, job_df)
                    LOGGER.info(
                        "Recruiting postprocessing repair progress: swaps=%s/%s, selected=%s, capacity_overflow=%.4f, over_capacity_jobs=%s",
                        iterations,
                        self.config.max_repair_iterations,
                        len(recs),
                        float(progress_constraints.get("capacity_violation_total", 0.0)),
                        int(progress_constraints.get("over_capacity_job_count", 0)),
                    )
                repaired_any = True
                if iterations >= self.config.max_repair_iterations:
                    break
            if not repaired_any:
                break

        final_constraints = self.constraint_handler.evaluate_all(recs, job_df)
        LOGGER.info(
            "Recruiting postprocessing done: swaps=%s, selected=%s, capacity_overflow=%.4f, over_capacity_jobs=%s, fully_repaired=%s",
            iterations,
            len(recs),
            float(final_constraints.get("capacity_violation_total", 0.0)),
            int(final_constraints.get("over_capacity_job_count", 0)),
            bool(final_constraints.get("all_hard_constraints_satisfied", False)),
        )
        diagnostics["final_constraints"] = final_constraints
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["num_swaps"] = int(len(diagnostics["swap_log"]))
        diagnostics["candidate_shortage_rate"] = self._candidate_shortage_rate(recs, candidates)
        diagnostics["final_utility"] = self._utility(recs)
        return {"recommendations": self._rerank(recs), "diagnostics": diagnostics}

    def _candidate_shortage_rate(self, recs: pd.DataFrame, candidates: Sequence[str]) -> float:
        if not candidates:
            return 0.0
        counts = recs["candidate_id"].astype(str).value_counts().to_dict() if not recs.empty else {}
        shortage = sum(1 for candidate_id in candidates if int(counts.get(candidate_id, 0)) < self.top_k)
        return float(shortage / len(candidates))


class RecruitingInProcessingAgent(_RecruitingBase):
    """
    Scenario-2 in-processing baseline.

    This baseline directly searches over the batch-level recommendation matrix.
    The local search objective includes utility, soft ALM penalties, and hard
    penalties for capacity/list-size violations.
    """

    def __init__(self, config: Optional[RecruitingInProcessingConfig] = None):
        self.config = config or RecruitingInProcessingConfig()
        handler = RecruitingConstraintHandler(
            RecruitingConstraintConfig(
                tau_hr=self.config.tau_hr,
                lambda_reciprocal=self.config.lambda_reciprocal,
                lambda_mismatch=self.config.lambda_mismatch,
                rho_reciprocal=self.config.rho_reciprocal,
                rho_mismatch=self.config.rho_mismatch,
                salary_weight=self.config.salary_weight,
                skill_weight=self.config.skill_weight,
            )
        )
        super().__init__(self.config.top_k, self.config.base_score_col, handler)

    @staticmethod
    def _stable_seed(seed: int, namespace: str = "recruiting_inprocessing") -> int:
        digest = hashlib.sha256(f"{namespace}:{seed}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)

    def _fitness(self, recs: pd.DataFrame, jobs: pd.DataFrame, candidates: Sequence[str]) -> float:
        evaluation = self._evaluate(recs, jobs)
        constraints = evaluation["constraints"]
        list_gap = self._list_size_gap(recs, candidates)
        capacity_gap = float(constraints.get("capacity_violation_total", 0.0)) / max(1, len(recs))
        qualification_gap = max(0.0, 1.0 - float(constraints.get("qualification_pass_rate", 0.0)))
        hard_violation = capacity_gap + qualification_gap + list_gap
        return float(
            evaluation["utility"]
            - float(constraints.get("augmented_lagrangian_penalty", 0.0))
            - self.config.hard_violation_weight * hard_violation
        )

    def _list_size_gap(self, recs: pd.DataFrame, candidates: Sequence[str]) -> float:
        if not candidates or self.top_k <= 0:
            return 0.0
        counts = recs["candidate_id"].astype(str).value_counts().to_dict() if not recs.empty else {}
        total_gap = sum(max(0, self.top_k - int(counts.get(candidate_id, 0))) for candidate_id in candidates)
        return float(total_gap / max(1, len(candidates) * self.top_k))

    def recommend_batch(
        self,
        candidate_job_pairs: Union[pd.DataFrame, List[Dict[str, Any]]],
        jobs: Union[pd.DataFrame, List[Dict[str, Any]]],
        candidate_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is not None:
            self.top_k = max(0, int(top_k))
        pairs = self._prepare_pairs(candidate_job_pairs)
        job_df = self._prepare_jobs(jobs)
        candidates = self._candidate_order(pairs, candidate_ids)
        diagnostics: Dict[str, Any] = {
            "input_pair_count": int(len(pairs)),
            "candidate_count": int(len(candidates)),
            "requested_top_k": int(self.top_k),
        }
        if pairs.empty or self.top_k == 0:
            empty = pairs.head(0).copy()
            diagnostics.update({"qualified_pair_count": 0, "search_steps": 0, "fully_repaired": False})
            return {"recommendations": empty, "diagnostics": diagnostics}

        qualified = self.constraint_handler.check_qualification(pairs)
        diagnostics["qualified_pair_count"] = int(len(qualified))
        diagnostics["qualification_filter_rate"] = float(len(qualified) / max(1, len(pairs)))
        recs = self._initial_topk(qualified, candidates)
        best_recs = recs.copy()
        best_fitness = self._fitness(best_recs, job_df, candidates)
        diagnostics["initial_constraints"] = self.constraint_handler.evaluate_all(best_recs, job_df)
        diagnostics["initial_fitness"] = float(best_fitness)
        LOGGER.info(
            "Recruiting inprocessing start: selected=%s, candidates=%s, initial_fitness=%.6f, capacity_overflow=%.4f",
            len(best_recs),
            len(candidates),
            best_fitness,
            float(diagnostics["initial_constraints"].get("capacity_violation_total", 0.0)),
        )

        capacity = self._capacity_map(job_df)
        rng = np.random.default_rng(self._stable_seed(self.config.random_seed))
        search_steps = 0
        while search_steps < self.config.max_search_steps:
            exposure = self._exposure_counts(recs)
            over_jobs = [job_id for job_id, count in exposure.items() if count > int(capacity.get(job_id, 1))]
            candidate_pool = candidates.copy()
            if over_jobs:
                affected = recs.loc[recs["job_id"].isin(over_jobs), "candidate_id"].drop_duplicates().astype(str).tolist()
                candidate_pool = affected or candidate_pool
            if not candidate_pool:
                break

            best_trial: Optional[pd.DataFrame] = None
            best_trial_fitness = best_fitness
            sample_size = min(len(candidate_pool), 25)
            sampled_candidates = rng.choice(np.array(candidate_pool, dtype=object), size=sample_size, replace=False).astype(str).tolist()
            for candidate_id in sampled_candidates:
                current_rows = recs.loc[recs["candidate_id"].astype(str) == candidate_id]
                if current_rows.empty:
                    continue
                removable = current_rows.sort_values(["final_score", "hr_accept_prob"], ascending=[True, True])
                for remove_idx in removable.index[: min(3, len(removable))]:
                    recs_without = recs.drop(index=remove_idx).reset_index(drop=True)
                    exposure_without = self._exposure_counts(recs_without)
                    replacement = self._find_replacement(
                        candidate_id,
                        qualified,
                        recs_without,
                        exposure_without,
                        capacity,
                        require_available_capacity=True,
                    )
                    if replacement is None:
                        trial = recs_without
                    else:
                        trial = pd.concat([recs_without, replacement.to_frame().T], ignore_index=True)
                    trial = self._rerank(trial)
                    trial_fitness = self._fitness(trial, job_df, candidates)
                    search_steps += 1
                    if trial_fitness > best_trial_fitness:
                        best_trial = trial
                        best_trial_fitness = trial_fitness
                    if self.config.progress_interval > 0 and search_steps % self.config.progress_interval == 0:
                        progress_constraints = self.constraint_handler.capacity_diagnostics(best_recs, job_df)
                        LOGGER.info(
                            "Recruiting inprocessing search progress: steps=%s/%s, best_fitness=%.6f, capacity_overflow=%.4f, over_capacity_jobs=%s",
                            search_steps,
                            self.config.max_search_steps,
                            best_fitness,
                            float(progress_constraints.get("capacity_violation_total", 0.0)),
                            int(progress_constraints.get("over_capacity_job_count", 0)),
                        )
                    if search_steps >= self.config.max_search_steps:
                        break
                if search_steps >= self.config.max_search_steps:
                    break

            if best_trial is None:
                break
            recs = best_trial
            if best_trial_fitness > best_fitness:
                best_recs = best_trial.copy()
                best_fitness = best_trial_fitness

        final_constraints = self.constraint_handler.evaluate_all(best_recs, job_df)
        LOGGER.info(
            "Recruiting inprocessing done: steps=%s, final_fitness=%.6f, selected=%s, capacity_overflow=%.4f, over_capacity_jobs=%s, fully_repaired=%s",
            search_steps,
            best_fitness,
            len(best_recs),
            float(final_constraints.get("capacity_violation_total", 0.0)),
            int(final_constraints.get("over_capacity_job_count", 0)),
            bool(final_constraints.get("all_hard_constraints_satisfied", False)),
        )
        diagnostics["final_constraints"] = final_constraints
        diagnostics["fully_repaired"] = bool(final_constraints.get("all_hard_constraints_satisfied", False))
        diagnostics["search_steps"] = int(search_steps)
        diagnostics["final_fitness"] = float(best_fitness)
        diagnostics["final_utility"] = self._utility(best_recs)
        diagnostics["candidate_shortage_rate"] = self._list_size_gap(best_recs, candidates)
        return {"recommendations": self._rerank(best_recs), "diagnostics": diagnostics}
