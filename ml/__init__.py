"""
ML Module for Backend Integration

This module provides ML inference capabilities for the backend services.
"""

# Import analysis functions only if dependencies are available
try:
    from analysis.inference import predict_probability, get_model_version
    __all__ = ['predict_probability', 'get_model_version']
except ImportError:
    # Graceful fallback when dependencies are missing
    __all__ = []
