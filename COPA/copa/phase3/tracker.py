"""Privacy-aware, idempotent Agent execution trace."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class AgentExecutionTracker:
    def __init__(self, thread_id: str, path: Optional[Path]):
        self.thread_id = thread_id
        self.path = Path(path) if path else None
        self.events: list[Dict[str, Any]] = []
        self._event_ids: set[str] = set()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "event_id" in event:
                        self._event_ids.add(str(event["event_id"]))

    def record(
        self,
        *,
        node: str,
        operation: str,
        status: str,
        attempt: int = 0,
        sequence: int = 0,
        duration_ms: float = 0.0,
        input_summary: Optional[Mapping[str, Any]] = None,
        output_summary: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        identity = f"{self.thread_id}|{node}|{operation}|{attempt}|{sequence}|{status}"
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        event = {
            "event_id": event_id,
            "thread_id": self.thread_id,
            "node": node,
            "operation": operation,
            "status": status,
            "attempt": attempt,
            "sequence": sequence,
            "duration_ms": duration_ms,
            "input_summary": dict(input_summary or {}),
            "output_summary": dict(output_summary or {}),
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if event_id in self._event_ids:
            return event
        self._event_ids.add(event_id)
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event
