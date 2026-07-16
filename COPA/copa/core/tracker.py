"""Append-only execution tracing for reproducible COPA runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    user_id: str
    module: str
    operation: str
    status: str
    before_version: int
    after_version: int
    before_candidates: int
    after_candidates: int
    duration_ms: float
    seed: Optional[int] = None
    input_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CandidateTracker:
    """Keeps trace events in memory and optionally mirrors them to JSONL."""

    def __init__(self, run_id: str, user_id: str = "", path: Optional[Path] = None):
        self.run_id = run_id
        self.user_id = user_id
        self.path = Path(path) if path else None
        self.events: List[TraceEvent] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **kwargs: Any) -> TraceEvent:
        payload = {"run_id": self.run_id, "user_id": self.user_id, **kwargs}
        event = TraceEvent(**payload)
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(event), default=_json_default, ensure_ascii=False) + "\n")
        return event

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.events]
