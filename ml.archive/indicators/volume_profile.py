import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_volume_profile(data: pd.DataFrame, bins: int = 24) -> Dict[float, float]:
    """Compute a simple volume profile by binning typical prices and summing volumes.

    Returns a mapping of price-level (bin midpoint) -> aggregated volume.
    """
    try:
        if data is None or data.empty or not set({'high', 'low', 'close', 'volume'}).issubset(data.columns):
            return {}

        df = data.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0

        price_min = float(df['low'].min())
        price_max = float(df['high'].max())
        if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
            return {}

        bins = max(8, int(bins))
        edges = np.linspace(price_min, price_max, bins + 1)
        midpoints = (edges[:-1] + edges[1:]) / 2.0

        # Assign each bar's typical price to a bin
        bin_indices = np.digitize(df['typical_price'].values, edges, right=False) - 1
        # Clip to valid range
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        # Aggregate volume by bin
        vol_by_bin = np.zeros(bins, dtype=float)
        for idx, vol in zip(bin_indices, df['volume'].values):
            if np.isfinite(vol):
                vol_by_bin[idx] += float(vol)

        profile: Dict[float, float] = {}
        for midpoint, volume in zip(midpoints, vol_by_bin):
            profile[float(midpoint)] = float(volume)

        return profile
    except Exception as exc:
        logger.error(f"calculate_volume_profile failed: {exc}")
        return {}


def identify_significant_levels(
    volume_profile: Dict[float, float],
    current_price: float,
    threshold_sigma: float = 1.5,
) -> Tuple[list[float], list[float]]:
    """Identify significant price levels as those whose volume exceeds mean + k*std.

    Returns two sorted lists: (support_levels_desc, resistance_levels_asc)
    """
    try:
        if not volume_profile or not np.isfinite(current_price):
            return [], []

        volumes = np.array(list(volume_profile.values()), dtype=float)
        if volumes.size == 0:
            return [], []

        mean_vol = float(np.mean(volumes))
        std_vol = float(np.std(volumes))
        # Avoid zero std leading to no levels
        if std_vol == 0.0:
            std_vol = 1e-6

        significant_prices: list[float] = []
        for price, vol in volume_profile.items():
            try:
                if float(vol) > (mean_vol + threshold_sigma * std_vol):
                    significant_prices.append(float(price))
            except Exception:
                continue

        if not significant_prices:
            return [], []

        support = sorted([p for p in significant_prices if p < current_price], reverse=True)
        resistance = sorted([p for p in significant_prices if p > current_price])
        return support, resistance
    except Exception as exc:
        logger.error(f"identify_significant_levels failed: {exc}")
        return [], []


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """Compute VWAP across the input series. Returns a pandas Series aligned with input index."""
    try:
        if data is None or data.empty or not set({'high', 'low', 'close', 'volume'}).issubset(data.columns):
            return pd.Series(dtype=float)
        tp = (data['high'] + data['low'] + data['close']) / 3.0
        cum_vp = (tp * data['volume']).cumsum()
        cum_vol = data['volume'].cumsum()
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = cum_vp / cum_vol
        return vwap.fillna(method='ffill').fillna(method='bfill')
    except Exception as exc:
        logger.error(f"calculate_vwap failed: {exc}")
        return pd.Series(dtype=float)


