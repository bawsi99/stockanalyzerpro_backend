import numpy as np
import json
from typing import Optional

def clean_for_json(obj):
    """Recursively convert objects to JSON-safe types.

    - Converts numpy types to native Python types
    - Converts NaN/Inf floats to None
    - Converts numpy arrays to lists
    - Converts datetime-like objects (e.g., pandas Timestamp) via isoformat
    - Recurses through dicts, lists, and tuples
    """
    # Handle numpy scalar types
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            # Map NaN/Inf to None, otherwise float()
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [clean_for_json(v) for v in obj.tolist()]
    except Exception:
        # If numpy is unavailable or any unexpected error occurs, fall through
        pass

    # Handle native floats (NaN/Inf)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj

    # Datetime-like objects with isoformat
    if hasattr(obj, 'isoformat'):
        try:
            return obj.isoformat()
        except Exception:
            pass

    # Containers
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]

    return obj

def safe_json_dumps(obj, **kwargs):
    """Safely serialize object to JSON string, handling NaN and infinity values."""
    cleaned_obj = clean_for_json(obj)
    return json.dumps(cleaned_obj, **kwargs) 

# -------------------------
# Canonical data utilities
# -------------------------

def normalize_interval(interval: Optional[str]) -> str:
    """Normalize various interval aliases to canonical backend keys.

    Returns one of: '1m','5m','15m','1h','day','week','month'.
    """
    if not interval:
        return 'day'
    i = str(interval).strip().lower()
    mapping = {
        '1min': '1m', 'min': '1m', 'minute': '1m', '1m': '1m',
        '5min': '5m', '5minute': '5m', '5m': '5m',
        '15min': '15m', '15minute': '15m', '15m': '15m',
        '60min': '1h', '1hour': '1h', 'hour': '1h', '1h': '1h', '60m': '1h',
        '1d': 'day', 'day': 'day', 'daily': 'day',
        '1w': 'week', 'week': 'week', 'weekly': 'week',
        '1mo': 'month', '1month': 'month', 'month': 'month', 'monthly': 'month',
    }
    return mapping.get(i, i)


def interval_to_frontend_display(interval: Optional[str]) -> str:
    """Map normalized interval to the frontend's display token.

    Examples:
    - 'day' -> '1D'
    - 'week' -> '1W'
    - 'month' -> '1M'
    - '1h' -> '1H'
    - intraday minutes remain unchanged
    """
    norm = normalize_interval(interval)
    mapping = {
        'day': '1D',
        'week': '1W',
        'month': '1M',
        '1h': '1H',
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
    }
    return mapping.get(norm, str(interval) if interval is not None else '1D')


def ensure_ohlcv_dataframe(df):
    """Ensure a pandas DataFrame has canonical OHLCV columns and proper dtypes.

    - Creates missing columns with zeros
    - Sorts by index if datetime-like
    - Casts numeric columns to float
    """
    if df is None or not hasattr(df, 'columns'):
        return df

    # Standard column names expected downstream
    needed = ['open', 'high', 'low', 'close', 'volume']
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0

    # Enforce float dtype where possible
    for c in needed:
        try:
            # Prefer vectorized conversion if available
            series = df[c]
            # Coerce non-numeric to NaN then fill and cast
            if hasattr(series, 'astype'):
                df[c] = series.astype(float)
        except Exception:
            try:
                df[c] = [float(x) if x is not None else 0.0 for x in df[c]]
            except Exception:
                pass

    # Ensure datetime index ordering if index is datetime-like
    try:
        if hasattr(df, 'index') and not df.index.is_monotonic_increasing:
            df = df.sort_index()
    except Exception:
        pass

    return df