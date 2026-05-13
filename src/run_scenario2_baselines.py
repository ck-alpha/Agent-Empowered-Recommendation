"""
Scenario-2 offline runner: Djinni recruitment recall -> constrained baselines.

Recommended smoke command:
  python src/run_scenario2_baselines.py --data_mode smoke --baseline_mode both

This script builds a semi-synthetic reciprocal recommendation benchmark from
Djinni JD/CV text. Djinni does not ship observed candidate-job match labels, so
accuracy metrics are reported against a deterministic pseudo-relevance target:
the highest semantic raw-recall job for each candidate before constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from agents import (
    RecruitingInProcessingAgent,
    RecruitingInProcessingConfig,
    RecruitingPostProcessingAgent,
    RecruitingPostProcessingConfig,
)

LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
EPS = 1e-9

SKILL_VOCAB = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "node",
    "django",
    "flask",
    "spring",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "linux",
    "devops",
    "qa",
    "testing",
    "selenium",
    "data",
    "machine learning",
    "ml",
    "nlp",
    "pytorch",
    "tensorflow",
    "scala",
    "go",
    "golang",
    "c++",
    "c#",
    ".net",
    "php",
    "ruby",
    "ios",
    "android",
    "swift",
    "kotlin",
    "product",
    "project manager",
    "scrum",
    "business analyst",
    "designer",
    "figma",
]


@dataclass
class Scenario2EvalRecord:
    method: str
    candidate_id: str
    pseudo_relevant_job_id: str
    recall_count: int
    raw_hit_at_10: float
    raw_ndcg_at_10: float
    final_hit_at_10: float
    final_ndcg_at_10: float
    final_list_size: int
    mean_base_score: float
    mean_hr_accept_prob: float
    reciprocal_penalty: float
    mismatch_penalty: float
    salary_mismatch_penalty: float
    skill_mismatch_penalty: float
    qualification_pass_rate: float
    capacity_satisfied: bool
    fully_repaired: bool
    capacity_violation_total: float
    over_capacity_job_count: int
    max_capacity_overflow: float
    mean_job_utilization: float
    congestion_rate: float
    job_exposure_gini: float
    candidate_shortage_rate: float
    num_swaps: int
    search_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario-2 Djinni recruitment baselines.")
    parser.add_argument("--data_dir", default="data/processed/djinni", help="Directory containing prepared Djinni parquet files.")
    parser.add_argument("--data_mode", choices=["smoke", "full"], default="smoke", help="Prepared data suffix to load.")
    parser.add_argument("--output_json", default="results/scenario2_metrics_djinni_smoke.json", help="Metrics JSON path.")
    parser.add_argument("--baseline_mode", choices=["postprocessing", "inprocessing", "both"], default="both")
    parser.add_argument("--max_candidates", type=int, default=200, help="Number of candidates to evaluate.")
    parser.add_argument("--max_jobs", type=int, default=3000, help="Number of jobs in the recall corpus.")
    parser.add_argument("--recall_k", type=int, default=100, help="Candidate-job recall size before constrained Top-K.")
    parser.add_argument("--top_k", type=int, default=10, help="Final recommendation list length per candidate.")
    parser.add_argument("--tfidf_max_features", type=int, default=30_000, help="Max TF-IDF features.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def resolve_methods(mode: str) -> List[str]:
    return ["postprocessing", "inprocessing"] if mode == "both" else [mode]


def load_prepared_tables(data_dir: str, data_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    jobs_path = base / f"djinni_jobs_english_{data_mode}.parquet"
    candidates_path = base / f"djinni_candidates_english_{data_mode}.parquet"
    missing = [str(path) for path in [jobs_path, candidates_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared Djinni files: {missing}. Run: python src/prepare_scenario2_djinni.py --mode {data_mode}"
        )
    jobs = pd.read_parquet(jobs_path)
    candidates = pd.read_parquet(candidates_path)
    required_jobs = ["job_id", "position", "description", "company_name", "exp_years_raw", "primary_keyword", "english_level"]
    required_candidates = ["candidate_id", "position", "candidate_text", "experience_years", "english_level"]
    missing_jobs = [col for col in required_jobs if col not in jobs.columns]
    missing_candidates = [col for col in required_candidates if col not in candidates.columns]
    if missing_jobs or missing_candidates:
        raise ValueError(f"Prepared data missing columns. jobs={missing_jobs}, candidates={missing_candidates}")
    return jobs, candidates


def stable_hash(text: Any, modulo: Optional[int] = None) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value % modulo if modulo else value


def sample_tables(jobs: pd.DataFrame, candidates: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    jobs = jobs.copy()
    candidates = candidates.copy()
    jobs = jobs.loc[jobs["description"].fillna("").astype(str).str.len() > 0].copy()
    candidates = candidates.loc[candidates["candidate_text"].fillna("").astype(str).str.len() > 0].copy()
    if args.max_jobs > 0 and len(jobs) > args.max_jobs:
        jobs = jobs.sample(n=args.max_jobs, random_state=args.seed).reset_index(drop=True)
    if args.max_candidates > 0 and len(candidates) > args.max_candidates:
        candidates = candidates.sample(n=args.max_candidates, random_state=args.seed).reset_index(drop=True)
    return jobs.reset_index(drop=True), candidates.reset_index(drop=True)


def parse_exp_years(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).lower()
    if any(token in text for token in ["no experience", "without experience", "trainee", "intern"]):
        return 0.0
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return 0.0
    return float(min(numbers))


def english_level_num(value: Any) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    ordered = [
        ("no english", 0.0),
        ("beginner", 1.0),
        ("elementary", 1.0),
        ("pre-intermediate", 2.0),
        ("pre intermediate", 2.0),
        ("intermediate", 3.0),
        ("upper-intermediate", 4.0),
        ("upper intermediate", 4.0),
        ("advanced", 5.0),
        ("fluent", 5.0),
    ]
    for token, level in ordered:
        if token in text:
            return level
    return 2.0


def extract_skills(text: Any) -> set[str]:
    lowered = str(text or "").lower()
    found = {skill for skill in SKILL_VOCAB if skill in lowered}
    return found


def join_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).strip()


def add_job_features(jobs: pd.DataFrame) -> pd.DataFrame:
    jobs = jobs.copy()
    jobs["job_min_exp_years"] = jobs["exp_years_raw"].map(parse_exp_years).astype(float)
    jobs["job_english_level_num"] = jobs["english_level"].map(english_level_num).astype(float)
    jobs["job_text"] = jobs.apply(lambda row: join_text(row["position"], row["primary_keyword"], row["description"]), axis=1)
    jobs["job_skills"] = jobs["job_text"].map(extract_skills)
    jobs["capacity_max"] = jobs["job_id"].map(lambda job_id: 1 + stable_hash(job_id, modulo=3)).astype(int)
    jobs["salary_max"] = jobs.apply(
        lambda row: 1800.0
        + 520.0 * float(row["job_min_exp_years"])
        + 140.0 * float(row["job_english_level_num"])
        + 80.0 * stable_hash(row["primary_keyword"], modulo=8),
        axis=1,
    )
    return jobs


def add_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["candidate_exp_years"] = pd.to_numeric(candidates["experience_years"], errors="coerce").fillna(0.0).clip(lower=0.0)
    candidates["candidate_english_level_num"] = candidates["english_level"].map(english_level_num).astype(float)
    candidates["candidate_full_text"] = candidates.apply(lambda row: join_text(row["position"], row["candidate_text"]), axis=1)
    candidates["candidate_skills"] = candidates["candidate_full_text"].map(extract_skills)
    candidates["expected_salary"] = candidates.apply(
        lambda row: 1500.0
        + 500.0 * float(row["candidate_exp_years"])
        + 120.0 * float(row["candidate_english_level_num"])
        + 50.0 * stable_hash(row["position"], modulo=8),
        axis=1,
    )
    return candidates


def build_candidate_job_pairs(
    jobs: pd.DataFrame,
    candidates: pd.DataFrame,
    recall_k: int,
    tfidf_max_features: int,
) -> pd.DataFrame:
    if jobs.empty or candidates.empty:
        return pd.DataFrame()
    recall_k = min(max(1, recall_k), len(jobs))
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        min_df=2,
        stop_words="english",
        ngram_range=(1, 2),
    )
    LOGGER.info("Fitting TF-IDF: jobs=%s, candidates=%s", len(jobs), len(candidates))
    all_text = pd.concat([jobs["job_text"], candidates["candidate_full_text"]], ignore_index=True)
    vectorizer.fit(all_text)
    job_matrix = vectorizer.transform(jobs["job_text"])
    candidate_matrix = vectorizer.transform(candidates["candidate_full_text"])

    nn = NearestNeighbors(n_neighbors=recall_k, metric="cosine", algorithm="brute")
    nn.fit(job_matrix)
    distances, indices = nn.kneighbors(candidate_matrix, n_neighbors=recall_k, return_distance=True)

    job_records = jobs.reset_index(drop=True).to_dict("records")
    rows: List[Dict[str, Any]] = []
    for cand_idx, candidate in enumerate(tqdm(candidates.itertuples(index=False), total=len(candidates), desc="Building candidate-job pairs")):
        candidate_id = str(getattr(candidate, "candidate_id"))
        candidate_exp = float(getattr(candidate, "candidate_exp_years"))
        candidate_english = float(getattr(candidate, "candidate_english_level_num"))
        expected_salary = float(getattr(candidate, "expected_salary"))
        candidate_skills = set(getattr(candidate, "candidate_skills"))
        for rank_idx, job_idx in enumerate(indices[cand_idx]):
            job = job_records[int(job_idx)]
            semantic_score = float(max(0.0, 1.0 - distances[cand_idx][rank_idx]))
            job_skills = set(job.get("job_skills", set()))
            if job_skills:
                skill_coverage = len(candidate_skills & job_skills) / len(job_skills)
                skill_mismatch = 1.0 - skill_coverage
            else:
                skill_coverage = 0.0
                skill_mismatch = 0.0
            salary_max = float(job.get("salary_max", 0.0))
            salary_mismatch = max(0.0, (expected_salary - salary_max) / max(expected_salary, EPS))
            exp_gap = max(0.0, float(job.get("job_min_exp_years", 0.0)) - candidate_exp)
            english_gap = max(0.0, float(job.get("job_english_level_num", 0.0)) - candidate_english)
            hr_accept_prob = float(
                np.clip(
                    0.15
                    + 0.55 * semantic_score
                    + 0.20 * skill_coverage
                    - 0.06 * exp_gap
                    - 0.04 * english_gap
                    - 0.10 * salary_mismatch,
                    0.0,
                    1.0,
                )
            )
            base_score = float(
                np.clip(
                    0.75 * semantic_score
                    + 0.15 * skill_coverage
                    + 0.10 * min(1.0, candidate_exp / max(float(job.get("job_min_exp_years", 0.0)), 1.0)),
                    0.0,
                    1.0,
                )
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "job_id": str(job["job_id"]),
                    "base_score": base_score,
                    "semantic_score": semantic_score,
                    "hr_accept_prob": hr_accept_prob,
                    "salary_mismatch": salary_mismatch,
                    "skill_mismatch": skill_mismatch,
                    "candidate_exp_years": candidate_exp,
                    "job_min_exp_years": float(job.get("job_min_exp_years", 0.0)),
                    "candidate_english_level_num": candidate_english,
                    "job_english_level_num": float(job.get("job_english_level_num", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def compute_hr_ndcg(recommended: Sequence[str], relevant_job_id: str, k: int) -> Tuple[float, float]:
    top_items = [str(item) for item in recommended[:k]]
    try:
        rank_idx = top_items.index(str(relevant_job_id))
    except ValueError:
        return 0.0, 0.0
    return 1.0, float(1.0 / np.log2(rank_idx + 2))


def pseudo_relevance_by_candidate(pairs: pd.DataFrame) -> Dict[str, str]:
    if pairs.empty:
        return {}
    ranked = pairs.sort_values(["candidate_id", "base_score", "job_id"], ascending=[True, False, True])
    return ranked.drop_duplicates("candidate_id", keep="first").set_index("candidate_id")["job_id"].astype(str).to_dict()


def raw_metrics_by_candidate(pairs: pd.DataFrame, pseudo_relevance: Mapping[str, str], top_k: int) -> Dict[str, Tuple[float, float]]:
    result: Dict[str, Tuple[float, float]] = {}
    for candidate_id, group in pairs.groupby("candidate_id", sort=False):
        ranked_jobs = group.sort_values(["base_score", "job_id"], ascending=[False, True])["job_id"].astype(str).tolist()
        result[str(candidate_id)] = compute_hr_ndcg(ranked_jobs, pseudo_relevance.get(str(candidate_id), ""), top_k)
    return result


def candidate_records(
    recs: pd.DataFrame,
    pairs: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    method: str,
    pseudo_relevance: Mapping[str, str],
    raw_metrics: Mapping[str, Tuple[float, float]],
    candidate_ids: Sequence[str],
    top_k: int,
) -> List[Scenario2EvalRecord]:
    final_constraints = diagnostics.get("final_constraints", {})
    records: List[Scenario2EvalRecord] = []
    recs_by_candidate = {candidate_id: group.copy() for candidate_id, group in recs.groupby("candidate_id", sort=False)} if not recs.empty else {}
    pair_counts = pairs["candidate_id"].astype(str).value_counts().to_dict() if not pairs.empty else {}
    for candidate_id in candidate_ids:
        group = recs_by_candidate.get(str(candidate_id), pd.DataFrame())
        final_jobs = group.sort_values("rank")["job_id"].astype(str).tolist() if not group.empty else []
        relevant = pseudo_relevance.get(str(candidate_id), "")
        final_hr, final_ndcg = compute_hr_ndcg(final_jobs, relevant, top_k)
        raw_hr, raw_ndcg = raw_metrics.get(str(candidate_id), (0.0, 0.0))
        records.append(
            Scenario2EvalRecord(
                method=method,
                candidate_id=str(candidate_id),
                pseudo_relevant_job_id=str(relevant),
                recall_count=int(pair_counts.get(str(candidate_id), 0)),
                raw_hit_at_10=float(raw_hr),
                raw_ndcg_at_10=float(raw_ndcg),
                final_hit_at_10=float(final_hr),
                final_ndcg_at_10=float(final_ndcg),
                final_list_size=int(len(final_jobs)),
                mean_base_score=float(pd.to_numeric(group.get("base_score", pd.Series(dtype=float)), errors="coerce").mean())
                if not group.empty
                else 0.0,
                mean_hr_accept_prob=float(pd.to_numeric(group.get("hr_accept_prob", pd.Series(dtype=float)), errors="coerce").mean())
                if not group.empty
                else 0.0,
                reciprocal_penalty=float(final_constraints.get("reciprocal_penalty", 0.0)),
                mismatch_penalty=float(final_constraints.get("mismatch_penalty", 0.0)),
                salary_mismatch_penalty=float(pd.to_numeric(group.get("salary_mismatch", pd.Series(dtype=float)), errors="coerce").mean())
                if not group.empty
                else 0.0,
                skill_mismatch_penalty=float(pd.to_numeric(group.get("skill_mismatch", pd.Series(dtype=float)), errors="coerce").mean())
                if not group.empty
                else 0.0,
                qualification_pass_rate=float(final_constraints.get("qualification_pass_rate", 0.0)),
                capacity_satisfied=bool(final_constraints.get("capacity_satisfied", False)),
                fully_repaired=bool(diagnostics.get("fully_repaired", False)),
                capacity_violation_total=float(final_constraints.get("capacity_violation_total", 0.0)),
                over_capacity_job_count=int(final_constraints.get("over_capacity_job_count", 0)),
                max_capacity_overflow=float(final_constraints.get("max_capacity_overflow", 0.0)),
                mean_job_utilization=float(final_constraints.get("mean_job_utilization", 0.0)),
                congestion_rate=float(final_constraints.get("congestion_rate", 0.0)),
                job_exposure_gini=float(final_constraints.get("job_exposure_gini", 0.0)),
                candidate_shortage_rate=float(diagnostics.get("candidate_shortage_rate", 0.0)),
                num_swaps=int(diagnostics.get("num_swaps", 0)),
                search_steps=int(diagnostics.get("search_steps", 0)),
            )
        )
    return records


def mean_summary(records: List[Scenario2EvalRecord], method: Optional[str], args: argparse.Namespace) -> Dict[str, Any]:
    if not records:
        return {}
    numeric_keys = [
        "recall_count",
        "raw_hit_at_10",
        "raw_ndcg_at_10",
        "final_hit_at_10",
        "final_ndcg_at_10",
        "final_list_size",
        "mean_base_score",
        "mean_hr_accept_prob",
        "reciprocal_penalty",
        "mismatch_penalty",
        "salary_mismatch_penalty",
        "skill_mismatch_penalty",
        "qualification_pass_rate",
        "capacity_satisfied",
        "fully_repaired",
        "capacity_violation_total",
        "over_capacity_job_count",
        "max_capacity_overflow",
        "mean_job_utilization",
        "congestion_rate",
        "job_exposure_gini",
        "candidate_shortage_rate",
        "num_swaps",
        "search_steps",
    ]
    summary = {key: float(np.mean([getattr(record, key) for record in records])) for key in numeric_keys}
    summary.update(
        {
            "method": method or "mixed",
            "num_candidates": len({record.candidate_id for record in records}),
            "num_records": len(records),
            "data_mode": args.data_mode,
            "recall_k": args.recall_k,
            "top_k": args.top_k,
            "max_candidates": args.max_candidates,
            "max_jobs": args.max_jobs,
        }
    )
    return summary


def save_metrics(output_json: str, args: argparse.Namespace, records: List[Scenario2EvalRecord], summaries_by_method: Dict[str, Dict[str, Any]]) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = mean_summary(records, None, args)
    payload = {
        "config": vars(args),
        "summary": summary,
        "summaries_by_method": summaries_by_method,
        "records": [asdict(record) for record in records],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved scenario-2 metrics JSON: %s", output_path)


def log_report(summaries_by_method: Mapping[str, Dict[str, Any]]) -> None:
    LOGGER.info("\n%s", "=" * 78)
    LOGGER.info("Scenario-2 Djinni Recruitment Baseline Report")
    LOGGER.info("%s", "=" * 78)
    for method, summary in summaries_by_method.items():
        LOGGER.info("[%s] candidates=%s", method, summary.get("num_candidates", 0))
        LOGGER.info("  Final HR@10 / NDCG@10      : %.4f / %.4f", summary.get("final_hit_at_10", 0.0), summary.get("final_ndcg_at_10", 0.0))
        LOGGER.info("  Qualification / Capacity CSR: %.4f / %.4f", summary.get("qualification_pass_rate", 0.0), summary.get("capacity_satisfied", 0.0))
        LOGGER.info("  Congestion rate / Gini      : %.4f / %.4f", summary.get("congestion_rate", 0.0), summary.get("job_exposure_gini", 0.0))
        LOGGER.info("  Reciprocal / mismatch       : %.4f / %.4f", summary.get("reciprocal_penalty", 0.0), summary.get("mismatch_penalty", 0.0))
    LOGGER.info("%s\n", "=" * 78)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    methods = resolve_methods(args.baseline_mode)

    jobs_raw, candidates_raw = load_prepared_tables(args.data_dir, args.data_mode)
    jobs_sampled, candidates_sampled = sample_tables(jobs_raw, candidates_raw, args)
    jobs = add_job_features(jobs_sampled)
    candidates = add_candidate_features(candidates_sampled)
    LOGGER.info("Loaded prepared tables: jobs=%s, candidates=%s", len(jobs), len(candidates))

    pairs = build_candidate_job_pairs(
        jobs=jobs,
        candidates=candidates,
        recall_k=args.recall_k,
        tfidf_max_features=args.tfidf_max_features,
    )
    if pairs.empty:
        raise ValueError("No candidate-job pairs generated.")
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    pseudo_relevance = pseudo_relevance_by_candidate(pairs)
    raw_metrics = raw_metrics_by_candidate(pairs, pseudo_relevance, args.top_k)

    agents = {
        "postprocessing": RecruitingPostProcessingAgent(RecruitingPostProcessingConfig(top_k=args.top_k)),
        "inprocessing": RecruitingInProcessingAgent(RecruitingInProcessingConfig(top_k=args.top_k, random_seed=args.seed)),
    }

    all_records: List[Scenario2EvalRecord] = []
    summaries_by_method: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        LOGGER.info("Running scenario-2 baseline: %s", method)
        method_started_at = time.time()
        result = agents[method].recommend_batch(
            candidate_job_pairs=pairs,
            jobs=jobs[["job_id", "capacity_max"]],
            candidate_ids=candidate_ids,
            top_k=args.top_k,
        )
        recs = result["recommendations"]
        diagnostics = result["diagnostics"]
        method_records = candidate_records(
            recs=recs,
            pairs=pairs,
            diagnostics=diagnostics,
            method=method,
            pseudo_relevance=pseudo_relevance,
            raw_metrics=raw_metrics,
            candidate_ids=candidate_ids,
            top_k=args.top_k,
        )
        summaries_by_method[method] = mean_summary(method_records, method, args)
        all_records.extend(method_records)
        LOGGER.info("Finished scenario-2 baseline %s in %.2f seconds.", method, time.time() - method_started_at)

    save_metrics(args.output_json, args, all_records, summaries_by_method)
    log_report(summaries_by_method)


if __name__ == "__main__":
    main()
