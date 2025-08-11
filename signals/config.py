from __future__ import annotations

import json
import os
from typing import Dict

DEFAULT_TF_WEIGHTS = {
    "minute": 0.5,
    "3minute": 0.5,
    "5minute": 0.6,
    "10minute": 0.6,
    "15minute": 0.7,
    "30minute": 0.8,
    "60minute": 0.9,
    "hour": 0.9,
    "day": 1.0,
    "week": 1.2,
    "month": 1.4,
}


def _load_json(config_path: str) -> Dict:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_timeframe_weights(config_path: str = None, regime: str | None = None) -> Dict[str, float]:
    """
    Load timeframe base weights from JSON config if present; else defaults.
    """
    if not config_path:
        # Allow override via env var for calibration runs
        env_path = os.environ.get("SIGNALS_WEIGHTS_CONFIG")
        config_path = env_path or os.path.join(os.path.dirname(__file__), "weights_config.json")

    data = _load_json(config_path) if os.path.exists(config_path) else {}
    # Profiles support: data["profiles"][regime]
    if regime and isinstance(data.get("profiles"), dict):
        prof = data["profiles"].get(regime)
        if isinstance(prof, dict) and prof:
            return {k: float(v) for k, v in prof.items()}

    tfw = data.get("timeframe_weights", {})
    if isinstance(tfw, dict) and tfw:
        return {k: float(v) for k, v in tfw.items()}
    return DEFAULT_TF_WEIGHTS.copy()


def save_timeframe_weights(weights: Dict[str, float], config_path: str = None, regime: str | None = None) -> None:
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "weights_config.json")
    payload = _load_json(config_path) if os.path.exists(config_path) else {}
    if regime:
        payload.setdefault("profiles", {})
        payload["profiles"][regime] = weights
    else:
        payload["timeframe_weights"] = weights
    with open(config_path, "w") as f:
        json.dump(payload, f, indent=2)


