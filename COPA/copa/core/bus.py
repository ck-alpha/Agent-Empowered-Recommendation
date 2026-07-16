"""Versioned in-memory Candidate State Bus."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import pandas as pd

from .models import CandidateRecord
from .tracker import CandidateTracker


CandidateInput = Union[CandidateRecord, Mapping[str, Any]]


@dataclass(frozen=True)
class BusSnapshot:
    version: int
    parent_version: Optional[int]
    operation: str
    frame: pd.DataFrame
    metadata: Mapping[str, Any]


class CandidateStateBus:
    """Shared candidate state with atomic updates and append-only history."""

    REQUIRED_COLUMNS = {
        "item_id",
        "base_score",
        "metadata",
        "hard_state",
        "soft_objectives",
        "active",
        "source",
        "version",
    }

    def __init__(self, tracker: Optional[CandidateTracker] = None):
        self.tracker = tracker
        self._snapshots: List[BusSnapshot] = []

    @property
    def version(self) -> int:
        self._ensure_initialized()
        return self._snapshots[-1].version

    @property
    def history_versions(self) -> List[int]:
        return [snapshot.version for snapshot in self._snapshots]

    def initialize(self, candidates: Iterable[CandidateInput], source: str = "input") -> int:
        started = perf_counter()
        rows: List[Dict[str, Any]] = []
        for candidate in candidates:
            if isinstance(candidate, CandidateRecord):
                row = candidate.to_row()
            else:
                payload = dict(candidate)
                record = CandidateRecord(
                    item_id=str(payload["item_id"]),
                    base_score=float(payload["base_score"]),
                    metadata=payload.get("metadata", {}),
                    source=payload.get("source", source),
                )
                row = record.to_row()
            rows.append(row)
        frame = pd.DataFrame(rows, columns=sorted(self.REQUIRED_COLUMNS))
        self._validate_frame(frame)
        self._snapshots = [
            BusSnapshot(0, None, "initialize", self._clone_frame(frame), {"source": source})
        ]
        self._record(
            module="CandidateStateBus",
            operation="initialize",
            status="success",
            before_version=-1,
            after_version=0,
            before_candidates=0,
            after_candidates=self._active_count(frame),
            duration_ms=(perf_counter() - started) * 1000,
            input_summary={"source": source, "total_candidates": len(frame)},
        )
        return 0

    def query(
        self,
        *,
        active_only: bool = False,
        feasible_only: bool = False,
        version: Optional[int] = None,
    ) -> pd.DataFrame:
        frame = self._clone_frame(self.get_snapshot(version).frame)
        if active_only:
            frame = frame[frame["active"].astype(bool)]
        if feasible_only:
            feasible = frame["hard_state"].map(lambda state: bool(state.get("feasible", False)))
            frame = frame[feasible & frame["active"].astype(bool)]
        return self._clone_frame(frame.reset_index(drop=True))

    def get_snapshot(self, version: Optional[int] = None) -> BusSnapshot:
        self._ensure_initialized()
        target = self.version if version is None else version
        for snapshot in self._snapshots:
            if snapshot.version == target:
                return BusSnapshot(
                    snapshot.version,
                    snapshot.parent_version,
                    snapshot.operation,
                    self._clone_frame(snapshot.frame),
                    dict(snapshot.metadata),
                )
        raise KeyError(f"Unknown Candidate State Bus version: {target}")

    def update(
        self,
        updater: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        module: str,
        operation: str,
        input_summary: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> int:
        self._ensure_initialized()
        started = perf_counter()
        before = self._snapshots[-1]
        working = self._clone_frame(before.frame)
        try:
            updated = updater(working)
            if not isinstance(updated, pd.DataFrame):
                raise TypeError("Candidate State Bus updater must return a pandas DataFrame")
            next_version = before.version + 1
            updated = self._clone_frame(updated)
            updated["version"] = next_version
            self._validate_frame(updated)
        except Exception as exc:
            self._record(
                module=module,
                operation=operation,
                status="failed",
                before_version=before.version,
                after_version=before.version,
                before_candidates=self._active_count(before.frame),
                after_candidates=self._active_count(before.frame),
                duration_ms=(perf_counter() - started) * 1000,
                seed=seed,
                input_summary=dict(input_summary or {}),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._snapshots.append(
            BusSnapshot(next_version, before.version, operation, updated, dict(input_summary or {}))
        )
        self._record(
            module=module,
            operation=operation,
            status="success",
            before_version=before.version,
            after_version=next_version,
            before_candidates=self._active_count(before.frame),
            after_candidates=self._active_count(updated),
            duration_ms=(perf_counter() - started) * 1000,
            seed=seed,
            input_summary=dict(input_summary or {}),
        )
        return next_version

    def rollback(self, target_version: int, reason: str = "manual") -> int:
        target = self.get_snapshot(target_version)
        return self.update(
            lambda _: self._clone_frame(target.frame),
            module="CandidateStateBus",
            operation="rollback",
            input_summary={"target_version": target_version, "reason": reason},
        )

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Candidate frame missing columns: {sorted(missing)}")
        if frame["item_id"].isna().any() or (frame["item_id"].astype(str).str.len() == 0).any():
            raise ValueError("item_id must be non-empty")
        duplicated = frame.loc[frame["item_id"].astype(str).duplicated(), "item_id"].tolist()
        if duplicated:
            raise ValueError(f"Duplicate item_id values: {duplicated[:5]}")
        scores = pd.to_numeric(frame["base_score"], errors="coerce")
        if scores.isna().any() or not pd.Series(scores).map(lambda value: bool(pd.notna(value)) and abs(float(value)) != float("inf")).all():
            raise ValueError("base_score values must be finite numbers")
        if not frame["metadata"].map(lambda value: isinstance(value, Mapping)).all():
            raise TypeError("metadata must be a mapping for every candidate")

    def _ensure_initialized(self) -> None:
        if not self._snapshots:
            raise RuntimeError("Candidate State Bus has not been initialized")

    @staticmethod
    def _active_count(frame: pd.DataFrame) -> int:
        return int(frame["active"].astype(bool).sum()) if not frame.empty else 0

    @staticmethod
    def _clone_frame(frame: pd.DataFrame) -> pd.DataFrame:
        cloned = frame.copy(deep=True)
        for column in cloned.select_dtypes(include=["object"]).columns:
            cloned[column] = cloned[column].map(copy.deepcopy)
        return cloned

    def _record(self, **kwargs: Any) -> None:
        if self.tracker:
            self.tracker.record(**kwargs)
