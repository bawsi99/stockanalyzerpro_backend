#!/usr/bin/env python3
"""
Risk Synthesis Agent Package

Provides an LLM-based synthesis layer that converts structured risk digest
into exactly-5 actionable bullets using the risk_synthesis_template.
"""

from .processor import RiskSynthesisProcessor

__all__ = ["RiskSynthesisProcessor"]
