"""Versioned, model-agnostic candidate artifacts for offline retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from copa.core import CandidateRecord


ARTIFACT_SCHEMA_VERSION = "1.0"
CANDIDATE_COLUMNS = (
    "user_id",
    "item_id",
    "raw_score",
    "base_score",
    "retrieval_rank",
    "retriever",
    "backend",
    "model_seed",
)
TARGET_COLUMNS = (
    "user_id",
    "target_item_id",
    "target_raw_score",
    "target_base_score",
    "target_full_rank",
    "target_model_covered",
)


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_alignment(manifest_paths: Iterable[Path | str]) -> Dict[str, Any]:
    """Assert that retrieval backends use one protocol and exported user cohort."""

    paths = [Path(path) for path in manifest_paths]
    if len(paths) < 2:
        raise ValueError("At least two manifests are required for alignment validation")
    manifests = [CandidateArtifactManifest.read(path) for path in paths]
    reference = manifests[0]
    protocol_hash_keys = {"interactions", "items", "recbole_atomic", "protocol_split"}
    reference_protocol_hashes = {
        key: value
        for key, value in reference.source_hashes.items()
        if key in protocol_hash_keys
    }
    invariant = (
        reference.dataset,
        reference.protocol,
        reference_protocol_hashes,
        reference.split_statistics.get("mapped_users"),
        reference.split_statistics.get("mapped_items"),
    )
    cohorts = []
    for path, manifest in zip(paths, manifests):
        observed_protocol_hashes = {
            key: value
            for key, value in manifest.source_hashes.items()
            if key in protocol_hash_keys
        }
        observed = (
            manifest.dataset,
            manifest.protocol,
            observed_protocol_hashes,
            manifest.split_statistics.get("mapped_users"),
            manifest.split_statistics.get("mapped_items"),
        )
        if observed != invariant:
            raise ValueError(f"Artifact protocol/catalog mismatch: {path}")
        candidate_path = path.parent / manifest.candidate_file
        users = frozenset(pd.read_parquet(candidate_path, columns=["user_id"])["user_id"].astype(str))
        cohorts.append(users)
    if any(cohort != cohorts[0] for cohort in cohorts[1:]):
        raise ValueError("Retrieval artifacts use different exported user cohorts")
    return {
        "dataset": reference.dataset,
        "protocol": reference.protocol,
        "artifacts": len(paths),
        "users": len(cohorts[0]),
        "mapped_users": reference.split_statistics.get("mapped_users"),
        "mapped_items": reference.split_statistics.get("mapped_items"),
    }


@dataclass(frozen=True)
class CandidateArtifactManifest:
    dataset: str
    protocol: str
    retriever: str
    backend: str
    model_seed: int
    candidate_k: int
    candidate_file: str
    target_file: str
    source_hashes: Mapping[str, str]
    model_config: Mapping[str, Any]
    environment: Mapping[str, Any]
    split_statistics: Mapping[str, Any]
    candidate_sha256: str = ""
    target_sha256: str = ""
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    notes: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path | str) -> "CandidateArtifactManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported candidate artifact schema: {payload.get('schema_version')!r}"
            )
        return cls(**payload)


class PrecomputedCandidateStore:
    """Validate and expose ranked candidates without importing a model backend."""

    def __init__(
        self,
        candidate_path: Path | str,
        *,
        manifest_path: Path | str | None = None,
        item_catalog: pd.DataFrame | Path | str | None = None,
        popularity: Optional[Mapping[str, float]] = None,
        verify_hashes: bool = True,
    ) -> None:
        self.candidate_path = Path(candidate_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest = (
            CandidateArtifactManifest.read(self.manifest_path)
            if self.manifest_path is not None
            else None
        )
        if self.manifest and verify_hashes and self.manifest.candidate_sha256:
            actual = sha256_file(self.candidate_path)
            if actual != self.manifest.candidate_sha256:
                raise ValueError("Candidate artifact SHA-256 does not match its manifest")

        self.target_path: Optional[Path] = None
        self.targets: Optional[pd.DataFrame] = None
        if self.manifest is not None:
            self.target_path = self.candidate_path.parent / self.manifest.target_file
            if not self.target_path.exists():
                raise FileNotFoundError(f"Target artifact not found: {self.target_path}")
            if verify_hashes and self.manifest.target_sha256:
                actual = sha256_file(self.target_path)
                if actual != self.manifest.target_sha256:
                    raise ValueError("Target artifact SHA-256 does not match its manifest")
            self.targets = pd.read_parquet(self.target_path)
            self._validate_targets(self.targets)
            self.targets["user_id"] = self.targets["user_id"].astype(str)
            self.targets["target_item_id"] = self.targets["target_item_id"].astype(str)
            self.targets = self.targets.set_index("user_id", drop=False)

        self.frame = pd.read_parquet(self.candidate_path)
        self._validate_frame(self.frame)
        self.frame["user_id"] = self.frame["user_id"].astype(str)
        self.frame["item_id"] = self.frame["item_id"].astype(str)
        self.frame = self.frame.sort_values(
            ["user_id", "retrieval_rank", "item_id"], kind="mergesort"
        ).reset_index(drop=True)
        self._groups = {
            user_id: group.reset_index(drop=True)
            for user_id, group in self.frame.groupby("user_id", sort=False)
        }
        self._validate_manifest_metadata()

        if isinstance(item_catalog, (str, Path)):
            catalog = pd.read_parquet(item_catalog)
        elif item_catalog is None:
            catalog = None
        else:
            catalog = item_catalog.copy()
        if catalog is not None:
            if "item_id" not in catalog:
                raise ValueError("Item catalog must contain item_id")
            catalog["item_id"] = catalog["item_id"].astype(str)
            if catalog["item_id"].duplicated().any():
                raise ValueError("Item catalog contains duplicate item_id values")
            self._catalog = catalog.set_index("item_id", drop=False)
        else:
            self._catalog = None
        self._popularity_supplied = popularity is not None
        self._popularity = {str(key): float(value) for key, value in (popularity or {}).items()}

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        observed_columns = tuple(frame.columns)
        if observed_columns != CANDIDATE_COLUMNS:
            raise ValueError(
                "Candidate artifact schema/order mismatch: "
                f"{observed_columns!r} != {CANDIDATE_COLUMNS!r}"
            )
        if frame.empty:
            raise ValueError("Candidate artifact cannot be empty")
        if frame[["user_id", "item_id"]].isna().any().any():
            raise ValueError("Candidate artifact contains null user_id/item_id")
        if frame.duplicated(["user_id", "item_id"]).any():
            raise ValueError("Candidate artifact contains duplicate user/item pairs")
        scores = frame[["raw_score", "base_score"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(scores.to_numpy(dtype=float)).all():
            raise ValueError("Candidate artifact contains non-finite scores")
        base = scores["base_score"].to_numpy(dtype=float)
        if ((base < 0.0) | (base > 1.0)).any():
            raise ValueError("base_score must lie in [0, 1]")
        ranks = pd.to_numeric(frame["retrieval_rank"], errors="coerce")
        if ranks.isna().any() or (ranks < 1).any() or (ranks % 1 != 0).any():
            raise ValueError("retrieval_rank must contain positive integers")
        for user_id, group in frame.assign(_rank=ranks.astype(int)).groupby("user_id"):
            ranked_group = group.sort_values("_rank", kind="mergesort")
            observed = ranked_group["_rank"].tolist()
            if observed != list(range(1, len(group) + 1)):
                raise ValueError(f"Non-contiguous retrieval ranks for user {user_id}")
            raw_scores = pd.to_numeric(ranked_group["raw_score"]).to_numpy(dtype=float)
            if (np.diff(raw_scores) > 0.0).any():
                raise ValueError(
                    f"retrieval_rank is inconsistent with raw_score for user {user_id}"
                )

    @staticmethod
    def _validate_targets(frame: pd.DataFrame) -> None:
        observed_columns = tuple(frame.columns)
        if observed_columns != TARGET_COLUMNS:
            raise ValueError(
                "Target artifact schema/order mismatch: "
                f"{observed_columns!r} != {TARGET_COLUMNS!r}"
            )
        if frame.empty or frame["user_id"].isna().any() or frame["target_item_id"].isna().any():
            raise ValueError("Target artifact contains no rows or null identifiers")
        if frame["user_id"].astype(str).duplicated().any():
            raise ValueError("Target artifact contains duplicate users")
        scores = frame[["target_raw_score", "target_base_score"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(scores.to_numpy(dtype=float)).all():
            raise ValueError("Target artifact contains non-finite scores")
        base = scores["target_base_score"].to_numpy(dtype=float)
        if ((base < 0.0) | (base > 1.0)).any():
            raise ValueError("target_base_score must lie in [0, 1]")
        ranks = pd.to_numeric(frame["target_full_rank"], errors="coerce")
        if ranks.isna().any() or (ranks < 1).any() or (ranks % 1 != 0).any():
            raise ValueError("target_full_rank must contain positive integers")
        coverage = frame["target_model_covered"]
        if not pd.api.types.is_bool_dtype(coverage.dtype):
            raise ValueError("target_model_covered must be a boolean column")

    def _validate_manifest_metadata(self) -> None:
        if self.manifest is None:
            return
        expected = {
            "retriever": str(self.manifest.retriever),
            "backend": str(self.manifest.backend),
            "model_seed": int(self.manifest.model_seed),
        }
        for column, value in expected.items():
            observed = set(self.frame[column].tolist())
            if observed != {value}:
                raise ValueError(
                    f"Candidate {column} does not match manifest: {observed!r} != {value!r}"
                )
        if self.targets is not None and set(self.targets.index) != set(self._groups):
            raise ValueError("Candidate and target artifacts contain different user catalogs")
        if self.targets is not None:
            for user_id, target in self.targets.iterrows():
                group = self._groups[str(user_id)]
                target_id = str(target["target_item_id"])
                matches = group[group["item_id"].astype(str) == target_id]
                expected_in_export = int(target["target_full_rank"]) <= len(group)
                if expected_in_export and len(matches) != 1:
                    raise ValueError(
                        f"Target {target_id} for {user_id} is missing from its declared full rank"
                    )
                if not expected_in_export and not matches.empty:
                    raise ValueError(
                        f"Target {target_id} for {user_id} appears outside its declared full rank"
                    )
                if matches.empty:
                    continue
                candidate = matches.iloc[0]
                if int(candidate["retrieval_rank"]) != int(target["target_full_rank"]):
                    raise ValueError(f"Target rank disagrees with candidates for {user_id}")
                if not np.isclose(
                    float(candidate["raw_score"]),
                    float(target["target_raw_score"]),
                    rtol=0.0,
                    atol=0.0,
                ) or not np.isclose(
                    float(candidate["base_score"]),
                    float(target["target_base_score"]),
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError(
                        f"Target score disagrees with candidates for {user_id}"
                    )

    @property
    def user_ids(self) -> list[str]:
        return sorted(self._groups)

    def ranked_frame(self, user_id: str, candidate_k: Optional[int] = None) -> pd.DataFrame:
        user_id = str(user_id)
        if user_id not in self._groups:
            raise KeyError(f"No candidates for user {user_id}")
        group = self._groups[user_id]
        if candidate_k is not None:
            if candidate_k <= 0:
                raise ValueError("candidate_k must be positive")
            group = group.head(int(candidate_k))
        return group.copy()

    def load(
        self,
        user_id: str,
        candidate_k: int,
        *,
        seen_items: Iterable[str] = (),
    ) -> list[CandidateRecord]:
        group = self.ranked_frame(user_id, candidate_k)
        seen = {str(value) for value in seen_items}
        overlap = seen & set(group["item_id"].astype(str))
        if overlap:
            raise ValueError(f"Candidate artifact contains seen items for {user_id}: {sorted(overlap)[:5]}")
        records: list[CandidateRecord] = []
        for row in group.to_dict("records"):
            item_id = str(row["item_id"])
            if self._catalog is None:
                metadata: Dict[str, Any] = {}
            else:
                if item_id not in self._catalog.index:
                    raise ValueError(f"Candidate {item_id} is missing from item catalog")
                metadata = self._catalog.loc[item_id].drop(labels=["item_id"]).to_dict()
            if self._popularity_supplied:
                metadata["popularity"] = self._popularity.get(item_id, 0.0)
            metadata.update(
                {
                    "retrieval_rank": int(row["retrieval_rank"]),
                    "retrieval_raw_score": float(row["raw_score"]),
                    "retriever": str(row["retriever"]),
                    "retrieval_backend": str(row["backend"]),
                    "retrieval_model_seed": int(row["model_seed"]),
                }
            )
            metadata = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in metadata.items()
            }
            score = float(row["base_score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score for {user_id}/{item_id}")
            records.append(
                CandidateRecord(
                    item_id=item_id,
                    base_score=score,
                    metadata=metadata,
                    source=f"{row['retriever']}:{row['backend']}",
                )
            )
        return records

    def target_row(self, user_id: str) -> Dict[str, Any]:
        if self.targets is None:
            raise RuntimeError("A manifest with a target artifact is required")
        user_id = str(user_id)
        if user_id not in self.targets.index:
            raise KeyError(f"No target for user {user_id}")
        return dict(self.targets.loc[user_id].to_dict())

    def target_record(self, user_id: str) -> CandidateRecord:
        row = self.target_row(user_id)
        item_id = str(row["target_item_id"])
        if self._catalog is None:
            metadata: Dict[str, Any] = {}
        else:
            if item_id not in self._catalog.index:
                raise ValueError(f"Target {item_id} is missing from item catalog")
            metadata = self._catalog.loc[item_id].drop(labels=["item_id"]).to_dict()
        if self._popularity_supplied:
            metadata["popularity"] = self._popularity.get(item_id, 0.0)
        metadata.update(
            {
                "retrieval_rank": int(row["target_full_rank"]),
                "retrieval_raw_score": float(row["target_raw_score"]),
                "retriever": self.manifest.retriever if self.manifest else "unknown",
                "retrieval_backend": self.manifest.backend if self.manifest else "unknown",
                "retrieval_model_seed": self.manifest.model_seed if self.manifest else -1,
                "target_model_covered": bool(row["target_model_covered"]),
            }
        )
        metadata = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in metadata.items()
        }
        return CandidateRecord(
            item_id=item_id,
            base_score=float(row["target_base_score"]),
            metadata=metadata,
            source=(
                f"{self.manifest.retriever}:{self.manifest.backend}:target"
                if self.manifest
                else "precomputed:target"
            ),
        )
