import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def calculate_kelly_fraction(probability: float, win_loss_ratio: float) -> float:
    """Calculate Kelly fraction based on success probability and win/loss ratio.

    Kelly formula (general): f* = (bp - q) / b, where b is win/loss ratio,
    p is probability of win, q=1-p. We clamp invalid inputs gracefully.
    """
    try:
        p = float(probability)
        b = float(win_loss_ratio)
        if not (0.0 <= p <= 1.0) or not np.isfinite(p):
            return 0.0
        if not np.isfinite(b) or b <= 0.0:
            return 0.0
        q = 1.0 - p
        return (b * p - q) / b
    except Exception as exc:
        logger.error(f"calculate_kelly_fraction failed: {exc}")
        return 0.0


def calculate_risk_score(
    probability: float,
    reward_risk_ratio: float,
    volatility: float,
    max_position: float = 0.1,
) -> Dict[str, float]:
    """Compute unified risk score and suggested position size.

    - probability: success probability in [0,1]
    - reward_risk_ratio: potential reward per unit risk (>0)
    - volatility: realized volatility (e.g., std of returns), non-negative
    - max_position: cap on position fraction
    """
    try:
        p = float(np.clip(probability, 0.01, 0.99))
        r = float(reward_risk_ratio)
        if not np.isfinite(r) or r <= 0:
            r = 1.0
        vol = float(volatility)
        if not np.isfinite(vol) or vol < 0:
            vol = 0.02

        # Expected value of a unit bet given reward:risk = r:1
        expected_value = p * r - (1.0 - p)

        # Kelly-based sizing with volatility dampening
        kelly_fraction = calculate_kelly_fraction(p, r)
        # Volatility penalty: in [~0,1]; higher vol => smaller size
        vol_penalty = 1.0 - np.tanh(vol * 3.0)
        position_size = max(0.0, min(float(kelly_fraction) * float(vol_penalty), float(max_position)))

        # Score: 50 neutral, positive EV increases score, volatility penalizes
        base_score = 50.0 + 25.0 * expected_value
        final_score = np.clip(base_score * (0.5 + 0.5 * vol_penalty), 0.0, 100.0)

        return {
            "risk_score": round(float(final_score), 2),
            "position_size": round(float(position_size), 4),
            "expected_value": round(float(expected_value), 4),
            "kelly_fraction": round(float(kelly_fraction), 4),
        }
    except Exception as exc:
        logger.error(f"calculate_risk_score failed: {exc}")
        return {
            "risk_score": 50.0,
            "position_size": 0.05,
            "expected_value": 0.0,
            "kelly_fraction": 0.0,
        }


def extract_reward_risk(pattern: Dict, current_price: float) -> Tuple[float, float]:
    """Derive reward/risk ratio and absolute risk from pattern target/stop.

    Returns (reward_risk_ratio, risk_abs). Falls back to 1.0, 0.0 when unknown.
    """
    try:
        cp = float(current_price)
        target = pattern.get('target_level')
        stop = pattern.get('stop_level')
        if not np.isfinite(cp) or cp <= 0:
            return 1.0, 0.0
        if target is None:
            target = cp * 1.05
        if stop is None:
            stop = cp * 0.95
        target = float(target)
        stop = float(stop)
        reward = abs(target - cp)
        risk = abs(cp - stop)
        if risk <= 1e-6:
            return 1.0, 0.0
        return float(reward / risk), float(risk)
    except Exception as exc:
        logger.error(f"extract_reward_risk failed: {exc}")
        return 1.0, 0.0


