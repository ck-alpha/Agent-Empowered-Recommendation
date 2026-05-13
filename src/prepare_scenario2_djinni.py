"""
Prepare Djinni recruitment data for scenario-2 baselines.

The script downloads the English JD/CV splits from HuggingFace and stores a
stable parquet view under data/processed/djinni. It intentionally does not run
any recommender logic; baseline code should consume the generated parquet files.

Smoke:
  python src/prepare_scenario2_djinni.py --mode smoke

Full:
  python src/prepare_scenario2_djinni.py --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

JOBS_DATASET = "lang-uk/recruitment-dataset-job-descriptions-english"
CANDIDATES_DATASET = "lang-uk/recruitment-dataset-candidate-profiles-english"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Djinni data for scenario-2 recruitment baselines.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke", help="Output size mode.")
    parser.add_argument("--output_dir", default="data/processed/djinni", help="Directory for prepared parquet files.")
    parser.add_argument("--smoke_jobs", type=int, default=10_000, help="Number of JD rows for smoke mode.")
    parser.add_argument("--smoke_candidates", type=int, default=5_000, help="Number of CV rows for smoke mode.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for smoke mode.")
    return parser.parse_args()


def _first_existing(row: pd.Series, columns: Iterable[str], default: Any = None) -> Any:
    for col in columns:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                return value
    return default


def _coalesce_column(df: pd.DataFrame, names: Iterable[str], default: Any = None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    values = []
    for _, row in df.iterrows():
        values.append(_first_existing(row, names, default=default))
    return pd.Series(values, index=df.index)


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def load_hf_dataset(dataset_name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it in the active environment with: pip install datasets"
        ) from exc

    LOGGER.info("Loading HuggingFace dataset: %s", dataset_name)
    ds = load_dataset(dataset_name, split="train")
    return ds.to_pandas()


def normalize_jobs(raw: pd.DataFrame) -> pd.DataFrame:
    jobs = pd.DataFrame()
    jobs["job_id"] = _normalize_text_series(_coalesce_column(raw, ["id", "job_id", "Job ID"]))
    jobs["position"] = _normalize_text_series(_coalesce_column(raw, ["Position", "position", "Title"], ""))
    jobs["description"] = _normalize_text_series(
        _coalesce_column(raw, ["Long Description", "long_description", "Description"], "")
    )
    jobs["company_name"] = _normalize_text_series(_coalesce_column(raw, ["Company Name", "company_name"], ""))
    jobs["exp_years_raw"] = _normalize_text_series(_coalesce_column(raw, ["Exp Years", "exp_years"], ""))
    jobs["primary_keyword"] = _normalize_text_series(_coalesce_column(raw, ["Primary Keyword", "primary_keyword"], ""))
    jobs["english_level"] = _normalize_text_series(_coalesce_column(raw, ["English Level", "english_level"], ""))
    jobs["published"] = _normalize_text_series(_coalesce_column(raw, ["Published", "published"], ""))
    jobs = jobs.dropna(subset=["job_id"]).drop_duplicates("job_id", keep="first").reset_index(drop=True)
    return jobs


def normalize_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.DataFrame()
    candidates["candidate_id"] = _normalize_text_series(_coalesce_column(raw, ["id", "candidate_id", "Candidate ID"]))
    candidates["position"] = _normalize_text_series(_coalesce_column(raw, ["Position", "position", "Title"], ""))
    candidates["candidate_text"] = _normalize_text_series(_coalesce_column(raw, ["CV", "cv", "Candidate Information"], ""))
    candidates["experience_years"] = pd.to_numeric(
        _coalesce_column(raw, ["Experience Years", "experience_years"], 0.0),
        errors="coerce",
    ).fillna(0.0)
    candidates["english_level"] = _normalize_text_series(_coalesce_column(raw, ["English Level", "english_level"], ""))

    optional_mappings = {
        "candidate_information": ["Candidate Information", "candidate_information"],
        "candidate_highlights": ["Candidate Highlights", "candidate_highlights"],
        "job_search_status": ["Job Search Status", "job_search_status"],
        "job_profile_types": ["Job Profile Types", "job_profile_types"],
    }
    for out_col, source_cols in optional_mappings.items():
        if any(col in raw.columns for col in source_cols):
            candidates[out_col] = _normalize_text_series(_coalesce_column(raw, source_cols, ""))

    candidates = candidates.dropna(subset=["candidate_id"]).drop_duplicates("candidate_id", keep="first").reset_index(drop=True)
    return candidates


def sample_if_needed(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def write_manifest(
    path: Path,
    mode: str,
    jobs: pd.DataFrame,
    candidates: pd.DataFrame,
    jobs_path: Path,
    candidates_path: Path,
    raw_job_columns: list[str],
    raw_candidate_columns: list[str],
    args: argparse.Namespace,
) -> None:
    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "source": {
            "jobs_dataset": JOBS_DATASET,
            "candidates_dataset": CANDIDATES_DATASET,
            "jobs_url": f"https://huggingface.co/datasets/{JOBS_DATASET}",
            "candidates_url": f"https://huggingface.co/datasets/{CANDIDATES_DATASET}",
            "license": "mit",
        },
        "config": {
            "smoke_jobs": args.smoke_jobs,
            "smoke_candidates": args.smoke_candidates,
            "seed": args.seed,
        },
        "outputs": {
            "jobs_path": str(jobs_path),
            "candidates_path": str(candidates_path),
            "jobs_rows": int(len(jobs)),
            "candidates_rows": int(len(candidates)),
            "jobs_columns": jobs.columns.tolist(),
            "candidates_columns": candidates.columns.tolist(),
        },
        "raw_columns": {
            "jobs": raw_job_columns,
            "candidates": raw_candidate_columns,
        },
        "quality": {
            "jobs_non_empty_description_rate": float(jobs["description"].ne("").mean()) if len(jobs) else 0.0,
            "candidates_non_empty_text_rate": float(candidates["candidate_text"].ne("").mean()) if len(candidates) else 0.0,
            "jobs_unique_id_rate": float(jobs["job_id"].nunique() / len(jobs)) if len(jobs) else 0.0,
            "candidates_unique_id_rate": float(candidates["candidate_id"].nunique() / len(candidates)) if len(candidates) else 0.0,
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_jobs = load_hf_dataset(JOBS_DATASET)
    raw_candidates = load_hf_dataset(CANDIDATES_DATASET)
    jobs = normalize_jobs(raw_jobs)
    candidates = normalize_candidates(raw_candidates)

    if args.mode == "smoke":
        jobs = sample_if_needed(jobs, args.smoke_jobs, args.seed)
        candidates = sample_if_needed(candidates, args.smoke_candidates, args.seed)

    jobs_path = output_dir / f"djinni_jobs_english_{args.mode}.parquet"
    candidates_path = output_dir / f"djinni_candidates_english_{args.mode}.parquet"
    manifest_path = output_dir / "djinni_prepare_manifest.json"

    jobs.to_parquet(jobs_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    write_manifest(
        path=manifest_path,
        mode=args.mode,
        jobs=jobs,
        candidates=candidates,
        jobs_path=jobs_path,
        candidates_path=candidates_path,
        raw_job_columns=raw_jobs.columns.tolist(),
        raw_candidate_columns=raw_candidates.columns.tolist(),
        args=args,
    )

    LOGGER.info("Saved jobs: %s rows -> %s", len(jobs), jobs_path)
    LOGGER.info("Saved candidates: %s rows -> %s", len(candidates), candidates_path)
    LOGGER.info("Saved manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
