"""Privacy-aware JSONL audit records for compiler calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


class CompilerAuditLogger:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload: Mapping[str, Any]) -> None:
        if not self.path:
            return
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
