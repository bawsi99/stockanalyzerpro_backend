"""
LLM Module - Provider-agnostic LLM client system

This module provides a clean, provider-agnostic interface for interacting with various LLM providers.
It replaces the legacy backend/gemini module with a more flexible and maintainable architecture.

Key Features:
- Provider-agnostic design (Gemini, OpenAI, Claude, etc.)
- Model-specific configuration per agent
- Simple, clean API calls without over-engineering
- Easy to test and maintain

Usage:
    from llm import LLMClient
    
    client = LLMClient(provider="gemini", model="gemini-2.5-flash")
    response = await client.call("Analyze this data...")
"""

from .client import LLMClient, get_llm_client
from .config.config import LLMConfig, get_llm_config

__version__ = "1.0.0"
__all__ = ["LLMClient", "LLMConfig", "get_llm_client", "get_llm_config"]
