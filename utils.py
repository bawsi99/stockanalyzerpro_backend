import numpy as np
import json

def clean_for_json(obj):
    """Clean object for JSON serialization by handling NaN and infinity values."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    else:
        return obj

def safe_json_dumps(obj, **kwargs):
    """Safely serialize object to JSON string, handling NaN and infinity values."""
    cleaned_obj = clean_for_json(obj)
    return json.dumps(cleaned_obj, **kwargs) 