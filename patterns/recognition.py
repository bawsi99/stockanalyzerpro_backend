import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict


class PatternRecognition:
    """
    Central class for all pattern detection logic (peaks/lows, divergences, double tops/bottoms, triangles, flags, volume anomalies).
    """
    @staticmethod
    def identify_peaks_lows(prices: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Identify local peaks and lows."""
        prices_np = prices.values
        peaks = argrelextrema(prices_np, np.greater, order=order)[0]
        lows = argrelextrema(prices_np, np.less, order=order)[0]
        return peaks, lows

    @staticmethod
    def get_swing_points(prices: pd.Series, order: int = 5) -> Dict[str, np.ndarray]:
        """Return swing highs and lows."""
        highs, lows = PatternRecognition.identify_peaks_lows(prices, order=order)
        return {'swing_highs': highs, 'swing_lows': lows}

    @staticmethod
    def detect_divergence(prices: pd.Series, indicator: pd.Series, order: int = 5) -> List[Tuple[int, int, str]]:
        """Detect bullish/bearish divergence between price and indicator (e.g., RSI)."""
        price_np = prices.values
        indicator_np = indicator.values
        peaks = argrelextrema(price_np, np.greater_equal, order=order)[0]
        lows = argrelextrema(price_np, np.less_equal, order=order)[0]
        divergences = []
        for i in range(1, len(peaks)):
            p1, p2 = peaks[i-1], peaks[i]
            if price_np[p2] > price_np[p1] and indicator_np[p2] < indicator_np[p1]:
                divergences.append((p1, p2, 'bearish'))
        for i in range(1, len(lows)):
            l1, l2 = lows[i-1], lows[i]
            if price_np[l2] < price_np[l1] and indicator_np[l2] > indicator_np[l1]:
                divergences.append((l1, l2, 'bullish'))
        return divergences

    @staticmethod
    def detect_volume_anomalies(volume: pd.Series, threshold: float = 2.0):
        """Detect volume spikes (anomalies) where volume is threshold times above rolling mean."""
        rolling_mean = volume.rolling(window=20, min_periods=1).mean()
        anomalies = volume[volume > threshold * rolling_mean].index.tolist()
        return anomalies

    @staticmethod
    def detect_double_top(prices: pd.Series, threshold: float = 0.02, order: int = 5) -> List[Tuple[int, int]]:
        """Detect double top patterns in price data."""
        peaks, _ = PatternRecognition.identify_peaks_lows(prices, order=order)
        patterns = []
        for i in range(1, len(peaks)):
            price_diff = abs(prices.iloc[peaks[i]] - prices.iloc[peaks[i - 1]])
            if price_diff / prices.iloc[peaks[i]] < threshold:
                patterns.append((peaks[i - 1], peaks[i]))
        return patterns

    @staticmethod
    def detect_double_bottom(prices: pd.Series, threshold: float = 0.02, order: int = 5) -> List[Tuple[int, int]]:
        """Detect double bottom patterns in price data."""
        lows = argrelextrema(prices.values, np.less, order=order)[0]
        patterns = []
        for i in range(1, len(lows)):
            first = lows[i-1]
            second = lows[i]
            if abs(prices.iloc[first] - prices.iloc[second]) / prices.iloc[first] < threshold:
                peak_idx = np.argmax(prices.iloc[first:second+1]) + first
                if peak_idx > first and peak_idx < second:
                    patterns.append((first, second))
        return patterns

    @staticmethod
    def detect_triangle(prices: pd.Series, min_points: int = 5) -> List[List[int]]:
        """
        Detect symmetrical triangles (descending highs + ascending lows).

        Returns: list of lists with indices that belong to each triangle window.
        """
        patterns: List[List[int]] = []

        for start in range(len(prices) - min_points):
            end = start + min_points
            segment = prices.iloc[start:end]

            # Significant local extremes -------------------------------------------------
            order = max(min_points // 12, 4)
            local_highs = argrelextrema(segment.values, np.greater, order=order)[0]
            local_lows  = argrelextrema(segment.values, np.less,    order=order)[0]

            if len(local_highs) < 2 or len(local_lows) < 2:
                continue

            # Use the SAME window for highs & lows ---------------------------------------
            x_hi = start + local_highs
            x_lo = start + local_lows
            y_hi = prices.iloc[x_hi]
            y_lo = prices.iloc[x_lo]

            # Regression lines -----------------------------------------------------------
            slope_hi, _ = np.polyfit(x_hi, y_hi, 1)   # should be negative
            slope_lo, _ = np.polyfit(x_lo, y_lo, 1)   # should be positive

            if slope_hi >= 0 or slope_lo <= 0:
                continue

            # Do the slopes have *similar magnitude*?  -----------------------------------
            mag_hi = abs(slope_hi)
            mag_lo = slope_lo
            rel_diff = abs(mag_hi - mag_lo) / max(mag_hi, mag_lo)

            if rel_diff > 0.35:
                continue

            # Passed all checks → record the whole sub-window (or just [start,end])
            patterns.append(list(range(start, end)))

        return patterns

    @staticmethod
    def detect_flag(
            prices: pd.Series,
            impulse: int = 15,           # bars that define the 'flag pole'
            channel: int = 20,           # length of consolidation window
            pullback_ratio: float = .35  # fraction of the impulse it may retrace
        ) -> List[List[int]]:
        """
        Very simple bullish/bearish flag detector:
        1) looks for an 'impulse' move (straight up or down)
        2) then a sideways / slightly drifting consolidation
        """
        patterns: List[List[int]] = []
        N = len(prices)

        i = impulse
        while i < N - channel:
            # 1) impulse magnitude
            pole_return = (prices.iloc[i] - prices.iloc[i-impulse]) / prices.iloc[i-impulse]

            # bullish pole
            if pole_return > 0.08:       # > +8 % in 'impulse' bars
                seg = prices.iloc[i:i+channel]
                max_pullback = pole_return * pullback_ratio
                retr = (seg.min() - prices.iloc[i]) / prices.iloc[i]

                # sideways channel? (small retracement and std-dev)
                if abs(retr) <= max_pullback and seg.pct_change().std() < 0.02:
                    patterns.append(list(range(i-impulse, i+channel)))
                    i += channel       # skip overlapping windows
                    continue

            # bearish pole
            if pole_return < -0.08:
                seg = prices.iloc[i:i+channel]
                max_pullback = abs(pole_return) * pullback_ratio
                retr = (seg.max() - prices.iloc[i]) / prices.iloc[i]
                if abs(retr) <= max_pullback and seg.pct_change().std() < 0.02:
                    patterns.append(list(range(i-impulse, i+channel)))
                    i += channel
                    continue

            i += 1

        return patterns 