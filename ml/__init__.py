"""
ML Module for Backend Integration

This module provides ML inference capabilities for the backend services.
"""

from .inference import predict_probability, get_model_version

__all__ = ['predict_probability', 'get_model_version']
