#!/usr/bin/env python3
"""
Pattern Analysis Utilities

Contains utility functions for the pattern analysis system including
data formatting, rounding, and result processing.
"""

import json
from typing import Any, Dict, List, Union
from datetime import datetime
import pandas as pd
import numpy as np


def round_numeric_values(obj: Any, decimal_places: int = 2) -> Any:
    """
    Recursively round all numeric values in a data structure to specified decimal places.
    
    Args:
        obj: The object to process (dict, list, or value)
        decimal_places: Number of decimal places to round to (default: 2)
        
    Returns:
        The processed object with rounded numeric values
    """
    if isinstance(obj, dict):
        return {key: round_numeric_values(value, decimal_places) for key, value in obj.items()}
    
    elif isinstance(obj, list):
        return [round_numeric_values(item, decimal_places) for item in obj]
    
    elif isinstance(obj, float):
        # Handle special float values
        if pd.isna(obj) or np.isinf(obj):
            return obj
        return round(obj, decimal_places)
    
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        # Handle numpy float types
        if pd.isna(obj) or np.isinf(obj):
            return float(obj)  # Convert to standard float
        return round(float(obj), decimal_places)
    
    elif isinstance(obj, (int, str, bool, type(None))):
        # Leave these types unchanged
        return obj
    
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict and processing
        try:
            obj_dict = obj.__dict__ if hasattr(obj, '__dict__') else {}
            return round_numeric_values(obj_dict, decimal_places)
        except:
            return str(obj)  # Fallback to string representation
    
    else:
        # For any other type, try to convert to string
        return str(obj)


def clean_for_json_serialization(obj: Any) -> Any:
    """
    Clean an object for JSON serialization, handling numpy types and complex objects.
    
    Args:
        obj: The object to clean
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: clean_for_json_serialization(value) for key, value in obj.items()}
    
    elif isinstance(obj, list):
        return [clean_for_json_serialization(item) for item in obj]
    
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if pd.isna(obj) or np.isinf(obj):
            return None
        return float(obj)
    
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif pd.isna(obj):
        return None
    
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat() if obj is not None else None
    
    elif hasattr(obj, '__dict__'):
        try:
            return clean_for_json_serialization(obj.__dict__)
        except:
            return str(obj)
    
    else:
        return obj


def format_analysis_results(analysis_results: Dict[str, Any], decimal_places: int = 2) -> Dict[str, Any]:
    """
    Format analysis results with proper rounding and JSON serialization.
    
    Args:
        analysis_results: The analysis results to format
        decimal_places: Number of decimal places for rounding (default: 2)
        
    Returns:
        Formatted analysis results
    """
    # First clean for JSON serialization
    cleaned_results = clean_for_json_serialization(analysis_results)
    
    # Then round all numeric values
    formatted_results = round_numeric_values(cleaned_results, decimal_places)
    
    return formatted_results


def save_formatted_json(data: Dict[str, Any], file_path: str, decimal_places: int = 2) -> None:
    """
    Save data to JSON file with proper formatting and rounding.
    
    Args:
        data: Data to save
        file_path: Path to save file
        decimal_places: Number of decimal places for rounding (default: 2)
    """
    # Format the data
    formatted_data = format_analysis_results(data, decimal_places)
    
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)


def format_confidence_score(confidence: float) -> float:
    """
    Format confidence scores to 2 decimal places with proper bounds checking.
    
    Args:
        confidence: Raw confidence score
        
    Returns:
        Formatted confidence score between 0.0 and 1.0
    """
    if pd.isna(confidence) or confidence is None:
        return 0.0
    
    # Ensure confidence is between 0 and 1
    bounded_confidence = max(0.0, min(1.0, float(confidence)))
    
    return round(bounded_confidence, 2)


def format_price_value(price: Union[float, int]) -> float:
    """
    Format price values to 2 decimal places.
    
    Args:
        price: Raw price value
        
    Returns:
        Formatted price value
    """
    if pd.isna(price) or price is None:
        return 0.0
    
    return round(float(price), 2)


def format_percentage(value: float) -> float:
    """
    Format percentage values to 1 decimal place.
    
    Args:
        value: Raw percentage value
        
    Returns:
        Formatted percentage value
    """
    if pd.isna(value) or value is None:
        return 0.0
    
    return round(float(value), 1)


def format_processing_time(time_seconds: float) -> float:
    """
    Format processing time to 3 decimal places.
    
    Args:
        time_seconds: Processing time in seconds
        
    Returns:
        Formatted processing time
    """
    if pd.isna(time_seconds) or time_seconds is None:
        return 0.0
    
    return round(float(time_seconds), 3)


# Test function
def test_rounding_utilities():
    """Test the rounding utilities with sample data"""
    sample_data = {
        "confidence_score": 0.45555555555555555,
        "price": 1441.0000000001,
        "percentage": 23.456789123456789,
        "nested_dict": {
            "float_value": 3.141592653589793,
            "list_values": [1.23456789, 2.34567891, 3.45678912]
        },
        "mixed_list": [
            {"price": 1234.56789012, "conf": 0.87654321},
            {"price": 2345.67890123, "conf": 0.76543210}
        ]
    }
    
    print("Original data:")
    print(json.dumps(sample_data, indent=2))
    
    formatted_data = format_analysis_results(sample_data)
    
    print("\nFormatted data:")
    print(json.dumps(formatted_data, indent=2))
    
    return formatted_data


if __name__ == "__main__":
    test_rounding_utilities()