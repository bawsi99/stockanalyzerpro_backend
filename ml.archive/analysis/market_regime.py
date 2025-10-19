import logging

import numpy as np
import pandas as pd

try:
    # Optional dependency; if missing we'll gracefully degrade
    from sklearn.ensemble import IsolationForest  # type: ignore
    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    HAS_SKLEARN = False


logger = logging.getLogger(__name__)


def detect_market_regime(data: pd.DataFrame) -> str:
    """Detect market regime using ADX (trend strength) and realized volatility.

    Returns one of: 'trending_high_vol', 'trending_low_vol', 'high_volatility',
    'mean_reverting', 'insufficient_data', or 'error'.
    """
    try:
        if data is None or data.empty or not set({'high', 'low', 'close'}).issubset(data.columns):
            return "insufficient_data"

        if len(data) < 40:
            return "insufficient_data"

        df = data.copy()
        df['returns'] = df['close'].pct_change()
        vol = df['returns'].rolling(14, min_periods=14).std()

        # True range and ATR(14)
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean()

        # +DI, -DI and ADX(14)
        plus_dm = (high.diff()).clip(lower=0.0)
        minus_dm = (-low.diff()).clip(lower=0.0)
        plus_di = 100.0 * (plus_dm.rolling(14, min_periods=14).mean() / atr14)
        minus_di = 100.0 * (minus_dm.rolling(14, min_periods=14).mean() / atr14)
        dx = 100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(14, min_periods=14).mean()

        if adx.isna().iloc[-1] or vol.isna().iloc[-1]:
            return "insufficient_data"

        current_adx = float(adx.iloc[-1])
        current_vol = float(vol.iloc[-1])
        vol_mean = float(vol.mean()) if np.isfinite(vol.mean()) else current_vol
        vol_std = float(vol.std()) if np.isfinite(vol.std()) else 0.0

        anomaly_count = 0
        if HAS_SKLEARN:
            features = pd.DataFrame({'volatility': vol, 'returns': df['returns']}).dropna().tail(200)
            if len(features) >= 32:
                try:
                    clf = IsolationForest(contamination=0.05, random_state=42)
                    clf.fit(features)
                    preds = clf.predict(features)
                    anomaly_count = int((preds == -1).sum())
                except Exception:
                    anomaly_count = 0

        # Regime logic
        if current_adx > 25.0:
            if current_vol > vol_mean + vol_std:
                return 'trending_high_vol'
            return 'trending_low_vol'
        else:
            if (vol_std > 0 and current_vol > vol_mean + 1.5 * vol_std) or anomaly_count > 2:
                return 'high_volatility'
            return 'mean_reverting'
    except Exception as exc:
        logger.error(f"detect_market_regime failed: {exc}")
        return "error"


