#!/usr/bin/env python3
"""
Pattern Detection Module - For Cross-Validation Agent

This module provides pattern detection capabilities specifically for the cross-validation pipeline.
It detects classic chart patterns that will then be validated by the cross-validation system.

Pattern Types Detected:
- Triangle patterns (ascending, descending, symmetrical)
- Flag and pennant patterns
- Channel and rectangle patterns
- Head and shoulders patterns
- Double top/bottom patterns
"""

from .detector import PatternDetector

__all__ = ['PatternDetector']
__version__ = "1.0.0"