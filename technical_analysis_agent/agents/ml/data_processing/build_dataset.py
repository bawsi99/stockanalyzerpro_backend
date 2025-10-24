#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

EPS = 1e-12


def read_raw_csv(path: str) -> pd.DataFrame:
    # Try common patterns: index in first column; or a timestamp column
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
    except Exception:
        df = pd.read_csv(path)
        # Find a timestamp-like column
        ts_cols = [c for c in df.columns if str(c).lower() in ("datetime", "timestamp", "date")] \
                  or [c for c in df.columns if "time" in str(c).lower() or "date" in str(c).lower()]
        if not ts_cols:
            raise ValueError("Could not find a datetime column and first column is not parseable as datetime.")
        ts = ts_cols[0]
        df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
        df = df.set_index(ts)

    # Normalize index to UTC naive and sort
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()

    # Normalize column names
    rename = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=rename)

    # Map common variants
    col_map = {}
    for want in ["open", "high", "low", "close", "volume"]:
        if want in df.columns:
            continue
        for cand in [want, want.capitalize(), want[0].upper()+want[1:], want[:1]]:
            if cand in df.columns:
                col_map[cand] = want
                break
    if col_map:
        df = df.rename(columns=col_map)

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    diff = series.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    ema_up = ema(up, period)
    ema_down = ema(down, period)
    rs = ema_up / (ema_down + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder's smoothing approximation via EMA
    return tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()


def parkinson_rv(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    # sqrt(1/(4 ln 2)) * std( ln(H/L) )  — use rolling std of log(H/L)
    hl = (high / (low + EPS)).clip(lower=EPS)
    x = np.log(hl)
    coeff = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(coeff) * x.rolling(window, min_periods=window).std()


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low + EPS)
    mfv = mfm * volume
    num = mfv.rolling(window, min_periods=window).sum()
    den = volume.rolling(window, min_periods=window).sum() + EPS
    return num / den


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).fillna(0.0).cumsum()


def rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    # Linear regression slope over window using simple differencing proxy (fast & stable)
    return (series - series.shift(window)) / (window + EPS)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    vol = out["volume"]

    # Basic returns (internal)
    log_close = np.log(close.clip(lower=EPS))
    ret1 = log_close.diff(1)
    ret5 = log_close.diff(5)
    ret10 = log_close.diff(10)

    # Volatility
    # rv_parkinson_20 pruned per de-collinearity plan
    atr14 = atr(high, low, close, period=14)
    out["atr_14_pct"] = atr14 / (close + EPS)
    out["atr_vol_20"] = atr14.rolling(20, min_periods=20).std()
    out["range_pct"] = (high - low) / (close + EPS)

    # Trend / moving average distances & slopes
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    # Drop dist_sma20_pct per de-collinearity plan; keep dist_sma50_pct and slope
    out["dist_sma50_pct"] = (close - sma50) / (sma50 + EPS)

    # RSI (no feature added; removed rsi14_dist_50 per plan)
    # rsi14 = rsi(close, period=14)

    # MACD histogram (kept)
    out["macd_hist"] = macd_hist(close)

    # Bollinger bandwidth (20, 2)
    bb_mid = sma20
    bb_std = close.rolling(20, min_periods=20).std()
    out["bb_bw_20"] = (4.0 * bb_std) / (bb_mid + EPS)  # (upper-lower)/mid = 4*std/mid

    # Volume features
    vol_sma20 = vol.rolling(20, min_periods=20).mean()
    out["vol_ratio_20"] = vol / (vol_sma20 + EPS)
    out["vol_cv_20"] = vol.rolling(20, min_periods=20).std() / (vol_sma20 + EPS)
    out["cmf_20"] = chaikin_money_flow(high, low, close, vol, window=20)

    # Up/Down volume ratio (20)
    up = (close > close.shift(1)).astype(float)
    down = (close < close.shift(1)).astype(float)
    up_vol = (vol * up).rolling(20, min_periods=20).sum()
    down_vol = (vol * down).rolling(20, min_periods=20).sum()
    out["up_down_vol_ratio_20"] = up_vol / (down_vol + EPS)

    # Return–volume correlation (20)
    out["ret_vol_corr_20"] = ret1.rolling(20, min_periods=20).corr(vol)

    # OBV removed (keep up_down_vol_ratio_20 as volume-pressure proxy)
    # obv_series = obv(close, vol)
    # out["obv_slope_20"] = rolling_slope(obv_series, window=20)

    # Price position vs local extremes (Donchian 20)
    roll_max_20 = high.rolling(20, min_periods=20).max()
    roll_min_20 = low.rolling(20, min_periods=20).min()
    out["pct_dist_to_20_high"] = (close - roll_max_20) / (roll_max_20 + EPS)
    # pct_dist_to_20_low pruned per de-collinearity plan
    prior_max_19 = high.rolling(19, min_periods=19).max().shift(1)
    prior_min_19 = low.rolling(19, min_periods=19).min().shift(1)
    out["breakout_up_20"] = (close > prior_max_19).astype(float)
    out["breakout_down_20"] = (close < prior_min_19).astype(float)

    # VWAP (session-based): typical price * volume cumulative per calendar day
    tp = (high + low + close) / 3.0
    dates = pd.to_datetime(out.index.date)
    grouped = dates
    cum_pv = (tp * vol).groupby(grouped).cumsum()
    cum_v = vol.groupby(grouped).cumsum()
    vwap = cum_pv / (cum_v + EPS)
    out["vwap_dist"] = (close - vwap) / (vwap + EPS)
    out["vwap_slope_5"] = rolling_slope(vwap, window=5)

    # Candle shape & patterns
    upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
    lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)
    body = (close - open_).abs()
    out["wick_to_body_ratio"] = (upper_wick + lower_wick) / (body + EPS)
    prev_high, prev_low = high.shift(1), low.shift(1)
    prev_open, prev_close = open_.shift(1), close.shift(1)
    out["inside_bar"] = ((high <= prev_high) & (low >= prev_low)).astype(float)
    out["engulfing"] = ((high >= prev_high) & (low <= prev_low)).astype(float)
    # Replace gap_up/gap_down with signed gap percentage
    out["gap_pct"] = (open_ - prev_close) / (prev_close + EPS)

    # Streaks
    up_move = (close > prev_close).astype(int)
    down_move = (close < prev_close).astype(int)
    def streak(arr):
        s = np.zeros(len(arr), dtype=int)
        for i in range(1, len(arr)):
            s[i] = s[i-1] + 1 if arr[i] == 1 else 0
        return s
    out["up_streak"] = streak(up_move.values)
    out["down_streak"] = streak(down_move.values)
    wick_up_dom = (upper_wick > lower_wick).astype(int)
    wick_down_dom = (lower_wick > upper_wick).astype(int)
    out["wick_up_streak_3"] = pd.Series(streak(wick_up_dom.values), index=out.index).rolling(3).max()
    out["wick_down_streak_3"] = pd.Series(streak(wick_down_dom.values), index=out.index).rolling(3).max()

    # Higher-moment stats on returns
    def _skew(x):
        x = np.asarray(x)
        m = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return np.nan
        return ((x - m)**3).mean() / (sd**3)
    def _kurt(x):
        x = np.asarray(x)
        m = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return np.nan
        return ((x - m)**4).mean() / (sd**4)
    out["ret_skew_20"] = ret1.rolling(20, min_periods=20).apply(_skew, raw=True)
    out["ret_kurt_20"] = ret1.rolling(20, min_periods=20).apply(_kurt, raw=True)
    # ret_skew_60 and ret_kurt_60 pruned due to drift

    # Calendar (cyclical) features
    idx = out.index
    dow = idx.weekday
    out["dow"] = dow
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    hour = idx.hour
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return out


def derive_default_output(input_path: str) -> str:
    # If path matches .../data/raw/symbol=XYZ/timeframe=TF/bars.csv -> .../data/processed/symbol=XYZ/timeframe=TF/features.csv
    base = input_path
    if "/data/raw/" in base and base.endswith("bars.csv"):
        return base.replace("/data/raw/", "/data/processed/").replace("bars.csv", "features.csv")
    # Else append _features.csv next to input
    root, ext = os.path.splitext(input_path)
    return f"{root}_features.csv"


def main():
    parser = argparse.ArgumentParser(description="Build ML features from raw OHLCV CSV")
    parser.add_argument("input_csv", help="Path to raw bars CSV (e.g., backend/agents/ml/data/raw/symbol=RELIANCE/timeframe=15m/bars.csv)")
    parser.add_argument("--output_csv", default=None, help="Optional output CSV path; defaults to derived processed path")
    parser.add_argument("--drop_warmup", action="store_true", help="Drop rows with NaNs from indicator warmup")
    args = parser.parse_args()

    df = read_raw_csv(args.input_csv)
    features = add_features(df)

    out_path = args.output_csv or derive_default_output(args.input_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.drop_warmup:
        features = features.dropna()

    features.to_csv(out_path, index=True)
    print({"wrote": len(features), "output_csv": out_path})


if __name__ == "__main__":
    main()
