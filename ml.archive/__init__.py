"""
ML Module for Backend Integration

This module provides ML inference capabilities for the backend services.
"""

# Import inference functions only if dependencies are available
try:
    from ml.inference import predict_probability, get_model_version, get_pattern_prediction_breakdown
    __all__ = ['predict_probability', 'get_model_version', 'get_pattern_prediction_breakdown']
except ImportError:
    # Graceful fallback when dependencies are missing
    __all__ = []
