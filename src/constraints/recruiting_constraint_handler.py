"""
Scenario-2 recruiting constraints.

The recruiting setting is batch-coupled: each candidate receives a Top-K list,
while job capacity is evaluated globally across all candidate lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class RecruitingConstraintConfig:
    """Configuration for scenario-2 recruiting constraints."""

    tau_hr: float = 0.35
    lambda_reciprocal: float = 1.0
    lambda_mismatch: float = 1.0
    rho_reciprocal: float = 1.0
    rho_mismatch: float = 1.0
    salary_weight: float = 0.5
    skill_weight: float = 0.5


class RecruitingConstraintHandler:
    """Constraint and penalty helper for person-job fit baselines."""

    def __init__(self, config: Optional[RecruitingConstraintConfig] = None):
        self.config = config or RecruitingConstraintConfig()

    @staticmethod
    def to_dataframe(records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, list):
            return pd.DataFrame(records).copy()
        raise TypeError("records must be a pandas DataFrame or a list of dictionaries.")

    @staticmethod
    def require_columns(df: pd.DataFrame, columns: Iterable[str], method_name: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{method_name} requires columns: {missing}")

    def check_qualification(self, pairs: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Keep only candidate-job edges satisfying qualification hard constraints."""
        df = self.to_dataframe(pairs)
        if df.empty:
            return df
        self.require_columns(df, ["candidate_exp_years", "job_min_exp_years", "candidate_english_level_num", "job_english_level_num"], "check_qualification")
        candidate_exp = pd.to_numeric(df["candidate_exp_years"], errors="coerce").fillna(0.0)
        job_exp = pd.to_numeric(df["job_min_exp_years"], errors="coerce").fillna(0.0)
        candidate_english = pd.to_numeric(df["candidate_english_level_num"], errors="coerce").fillna(0.0)
        job_english = pd.to_numeric(df["job_english_level_num"], errors="coerce").fillna(0.0)
        mask = (candidate_exp >= job_exp) & (candidate_english >= job_english)
        return df.loc[mask].copy()

    def qualification_pass_rate(self, slate: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        df = self.to_dataframe(slate)
        if df.empty:
            return 0.0
        return float(len(self.check_qualification(df)) / len(df))

    def capacity_diagnostics(
        self,
        recommendations: Union[pd.DataFrame, List[Dict[str, Any]]],
        jobs: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Evaluate global job exposure capacity constraints."""
        recs = self.to_dataframe(recommendations)
        job_df = self.to_dataframe(jobs)
        if recs.empty:
            return {
                "capacity_satisfied": True,
                "capacity_violation_total": 0.0,
                "over_capacity_job_count": 0,
                "max_capacity_overflow": 0.0,
                "mean_job_utilization": 0.0,
                "congestion_rate": 0.0,
                "job_exposure_gini": 0.0,
            }
        self.require_columns(recs, ["job_id"], "capacity_diagnostics")
        self.require_columns(job_df, ["job_id", "capacity_max"], "capacity_diagnostics")

        exposure = recs["job_id"].astype(str).value_counts().rename("exposure_count")
        capacity = job_df[["job_id", "capacity_max"]].copy()
        capacity["job_id"] = capacity["job_id"].astype(str)
        capacity["capacity_max"] = pd.to_numeric(capacity["capacity_max"], errors="coerce").fillna(1.0).clip(lower=1.0)
        joined = capacity.drop_duplicates("job_id").set_index("job_id").join(exposure, how="left")
        joined["exposure_count"] = joined["exposure_count"].fillna(0.0)
        joined["overflow"] = (joined["exposure_count"] - joined["capacity_max"]).clip(lower=0.0)
        exposed = joined.loc[joined["exposure_count"] > 0].copy()
        utilization = (exposed["exposure_count"] / exposed["capacity_max"]).clip(lower=0.0) if not exposed.empty else pd.Series(dtype=float)

        exposure_values = joined["exposure_count"].to_numpy(dtype=float)
        gini = self._gini(exposure_values)
        violation_total = float(joined["overflow"].sum())
        over_jobs = int((joined["overflow"] > 0).sum())
        return {
            "capacity_satisfied": bool(violation_total <= 0.0),
            "capacity_violation_total": violation_total,
            "over_capacity_job_count": over_jobs,
            "max_capacity_overflow": float(joined["overflow"].max()) if len(joined) else 0.0,
            "mean_job_utilization": float(utilization.mean()) if len(utilization) else 0.0,
            "congestion_rate": float(over_jobs / max(1, int((joined["exposure_count"] > 0).sum()))),
            "job_exposure_gini": float(gini),
            "exposure_by_job": exposed.reset_index()[["job_id", "capacity_max", "exposure_count", "overflow"]]
            if not exposed.empty
            else pd.DataFrame(columns=["job_id", "capacity_max", "exposure_count", "overflow"]),
        }

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0 or float(values.sum()) <= 0.0:
            return 0.0
        sorted_values = np.sort(values)
        n = sorted_values.size
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n)

    def reciprocal_penalty(self, slate: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        df = self.to_dataframe(slate)
        if df.empty:
            return 0.0
        self.require_columns(df, ["hr_accept_prob"], "reciprocal_penalty")
        hr_prob = pd.to_numeric(df["hr_accept_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return float(np.mean(np.maximum(0.0, self.config.tau_hr - hr_prob)))

    def mismatch_penalty(self, slate: Union[pd.DataFrame, List[Dict[str, Any]]]) -> float:
        df = self.to_dataframe(slate)
        if df.empty:
            return 0.0
        self.require_columns(df, ["salary_mismatch", "skill_mismatch"], "mismatch_penalty")
        salary = pd.to_numeric(df["salary_mismatch"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        skill = pd.to_numeric(df["skill_mismatch"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        combined = self.config.salary_weight * salary + self.config.skill_weight * skill
        return float(np.mean(combined)) if combined.size else 0.0

    def augmented_lagrangian_penalty(self, reciprocal_penalty: float, mismatch_penalty: float) -> float:
        recip = max(0.0, float(reciprocal_penalty))
        mismatch = max(0.0, float(mismatch_penalty))
        total = (
            self.config.lambda_reciprocal * recip
            + 0.5 * self.config.rho_reciprocal * recip ** 2
            + self.config.lambda_mismatch * mismatch
            + 0.5 * self.config.rho_mismatch * mismatch ** 2
        )
        return float(total)

    def evaluate_all(
        self,
        recommendations: Union[pd.DataFrame, List[Dict[str, Any]]],
        jobs: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        recs = self.to_dataframe(recommendations)
        qualification_rate = self.qualification_pass_rate(recs)
        capacity = self.capacity_diagnostics(recs, jobs)
        compact_capacity = dict(capacity)
        compact_capacity.pop("exposure_by_job", None)
        recip = self.reciprocal_penalty(recs)
        mismatch = self.mismatch_penalty(recs)
        alm = self.augmented_lagrangian_penalty(recip, mismatch)
        all_hard = bool(qualification_rate >= 1.0 and compact_capacity.get("capacity_satisfied", False))
        return {
            "qualification_pass_rate": float(qualification_rate),
            "reciprocal_penalty": float(recip),
            "mismatch_penalty": float(mismatch),
            "augmented_lagrangian_penalty": float(alm),
            "all_hard_constraints_satisfied": all_hard,
            **compact_capacity,
        }

    def final_score(self, pairs: pd.DataFrame, base_score_col: str = "base_score") -> pd.Series:
        """Item-level score used by ranking baselines."""
        if pairs.empty:
            return pd.Series(dtype=float, index=pairs.index)
        self.require_columns(pairs, [base_score_col, "hr_accept_prob", "salary_mismatch", "skill_mismatch"], "final_score")
        base = pd.to_numeric(pairs[base_score_col], errors="coerce").fillna(0.0).astype(float)
        hr_prob = pd.to_numeric(pairs["hr_accept_prob"], errors="coerce").fillna(0.0).astype(float)
        recip = (self.config.tau_hr - hr_prob).clip(lower=0.0)
        salary = pd.to_numeric(pairs["salary_mismatch"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
        skill = pd.to_numeric(pairs["skill_mismatch"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
        mismatch = self.config.salary_weight * salary + self.config.skill_weight * skill
        penalty = (
            self.config.lambda_reciprocal * recip
            + 0.5 * self.config.rho_reciprocal * recip ** 2
            + self.config.lambda_mismatch * mismatch
            + 0.5 * self.config.rho_mismatch * mismatch ** 2
        )
        return (base - penalty).astype(float)

