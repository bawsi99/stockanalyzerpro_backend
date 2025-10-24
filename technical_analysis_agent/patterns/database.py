import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternRecord:
    symbol: str
    timestamp: str
    features: Dict[str, float]
    outcome: Optional[bool]
    confirmed: bool


class PatternDatabase:
    """Simple JSON-backed store for pattern occurrences and outcomes.

    Adds lightweight schema versioning via an internal _meta block in the JSON file.
    The runtime API filters meta out so existing callers remain unaffected.
    """

    def __init__(self, db_path: str | None = None) -> None:
        # Default to backend/cache directory if available
        default_path = os.path.join(os.path.dirname(__file__), 'cache_pattern_performance.json')
        self.db_path = db_path or default_path
        self._data: Dict[str, List[PatternRecord]] = {}
        self._meta = {"version": "1.1"}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    raw = json.load(f)
                # Support meta block if present
                if isinstance(raw, dict) and "_meta" in raw:
                    try:
                        self._meta = dict(raw.get("_meta") or {})
                    except Exception:
                        self._meta = {"version": "1.1"}
                    payload = {k: v for k, v in raw.items() if k != "_meta"}
                else:
                    payload = raw
                self._data = {
                    k: [PatternRecord(**r) for r in v] for k, v in payload.items()
                }
            else:
                self._data = {}
        except Exception as exc:
            logger.error(f"PatternDatabase load failed: {exc}")
            self._data = {}

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            serializable = {k: [asdict(r) for r in v] for k, v in self._data.items()}
            # Include meta block for versioning; do not expose via get_all
            out = {"_meta": self._meta}
            out.update(serializable)
            with open(self.db_path, 'w') as f:
                json.dump(out, f, indent=2)
        except Exception as exc:
            logger.error(f"PatternDatabase save failed: {exc}")

    def record_pattern(self, pattern_type: str, record: PatternRecord) -> None:
        try:
            arr = self._data.setdefault(pattern_type, [])
            arr.append(record)
            self._save()
        except Exception as exc:
            logger.error(f"record_pattern failed: {exc}")

    def update_outcome(self, pattern_type: str, index: int, outcome: bool) -> None:
        try:
            arr = self._data.get(pattern_type)
            if arr is None or not (0 <= index < len(arr)):
                return
            rec = arr[index]
            rec.outcome = bool(outcome)
            rec.confirmed = True
            self._save()
        except Exception as exc:
            logger.error(f"update_outcome failed: {exc}")

    def get_historical(self, pattern_type: str) -> List[PatternRecord]:
        return [r for r in self._data.get(pattern_type, []) if r.confirmed]

    def get_all(self) -> Dict[str, List[PatternRecord]]:
        # Expose only pattern buckets; meta is internal
        return self._data

    # --- Bulk helpers to reduce disk writes in backtests ---
    def record_patterns_bulk(self, pattern_type: str, records: List[PatternRecord]) -> None:
        try:
            if not records:
                return
            arr = self._data.setdefault(pattern_type, [])
            arr.extend(records)
            self._save()
        except Exception as exc:
            logger.error(f"record_patterns_bulk failed: {exc}")


